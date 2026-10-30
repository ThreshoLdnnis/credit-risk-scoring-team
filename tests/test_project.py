"""The full entry must never mark a partial or overwritten run complete."""

import json

import pytest

import run_project


def controlled_source(tmp_path, monkeypatch):
    source = tmp_path / 'test-only.csv'
    source.write_text('controlled test input; not an official experiment', encoding='utf8')
    monkeypatch.setattr(run_project, 'EXPECTED_SOURCE_SHA256', run_project.sha256(source))
    return source


def test_wrong_data_is_rejected_before_creating_output(tmp_path):
    source = tmp_path / 'wrong.csv'
    source.write_text('wrong source', encoding='utf8')
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='SHA256'):
        run_project.run_complete(source, output)
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    source = controlled_source(tmp_path, monkeypatch)
    output = tmp_path / 'output'
    output.mkdir()
    marker = output / 'delivery.json'
    marker.write_text('original run', encoding='utf8')
    with pytest.raises(FileExistsError):
        run_project.run_complete(source, output)
    assert marker.read_text(encoding='utf8') == 'original run'


def test_failed_model_stage_produces_a_failed_manifest(tmp_path, monkeypatch):
    source = controlled_source(tmp_path, monkeypatch)
    output = tmp_path / 'output'
    monkeypatch.setattr(run_project, 'collect_environment', lambda: {'scope': 'test fixture'})
    monkeypatch.setattr(run_project, 'run_from_config', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('intentional failure')))
    with pytest.raises(RuntimeError, match='intentional'):
        run_project.run_complete(source, output)
    manifest = json.loads((output / 'delivery.json').read_text(encoding='utf8'))
    assert manifest['status'] == 'failed'
    assert manifest['failed_step'] == 'three_model_pipeline'
    assert 'test_metrics' not in manifest
