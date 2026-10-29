"""T03 shared API/CLI checks using clearly marked synthetic data."""

import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import run_pipeline
from src.data import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(11)
    values = rng.uniform(1, 9, size=(240, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame['age'] = rng.integers(20, 80, len(frame))
    frame.loc[::7, 'MonthlyIncome'] = np.nan
    frame[TARGET_COLUMN] = (frame['DebtRatio'] > 5).astype(int)
    frame.insert(0, 'Unnamed: 0', np.arange(1, len(frame) + 1))
    path = tmp_path / 'synthetic-training.csv'
    frame.to_csv(path, index=False)
    return path


def settings():
    return json.loads((run_pipeline.ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))


def test_public_three_argument_interface_keeps_settings_unchanged(source, tmp_path):
    config = settings()
    before = copy.deepcopy(config)
    record = run_pipeline.run_experiment(source, tmp_path / 'direct', config, synthetic=True)
    assert config == before
    assert record['entry_point'] == 'run_pipeline.run_experiment'
    assert record['audit']['saved_loaded_max_probability_difference'] == 0
    assert record['source']['acquisition'] is None
    assert 'run_pipeline.py' in record['code_sha256']
    assert 'run.log' in record['output_sha256']


def test_shared_cli_runs_and_records_actual_invocation(source, tmp_path):
    output = tmp_path / 'cli'
    result = subprocess.run([
        sys.executable, str(run_pipeline.ROOT / 'run_pipeline.py'), '--input', str(source),
        '--output', str(output), '--synthetic',
    ], cwd=run_pipeline.ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'synthetic_engineering_check'
    record = json.loads((output / 'record.json').read_text(encoding='utf-8'))
    assert record['command'][1] == 'run_pipeline.py'
    assert record['command'][-1] == '--synthetic'
    assert record['config_source']['sha256'] == run_pipeline.sha256(run_pipeline.ROOT / 'configs/experiment.json')
    assert sum(part['rows'] for part in record['split'].values()) == 240


def test_shared_cli_missing_data_returns_actionable_failure(tmp_path):
    output = tmp_path / 'absent-output'
    result = subprocess.run([
        sys.executable, str(run_pipeline.ROOT / 'run_pipeline.py'), '--input', str(tmp_path / 'missing.csv'),
        '--output', str(output),
    ], cwd=run_pipeline.ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 1
    assert 'Kaggle' in result.stderr
    assert not output.exists()


def test_acquisition_metadata_must_match_input(source, tmp_path):
    source.with_suffix('.source.json').write_text('{"sha256": "wrong"}', encoding='utf-8')
    with pytest.raises(ValueError, match='metadata'):
        run_pipeline.run_experiment(source, tmp_path / 'metadata-failure', settings(), synthetic=True)


def test_config_change_cannot_leave_a_successful_manifest(source, tmp_path, monkeypatch):
    config_path, output = tmp_path / 'config.json', tmp_path / 'changed-config'
    config_path.write_text(json.dumps(settings()), encoding='utf-8')
    original = run_pipeline.run_experiment

    def change_config_after_training(*args, **kwargs):
        record = original(*args, **kwargs)
        config_path.write_text('{"changed": true}', encoding='utf-8')
        return record

    monkeypatch.setattr(run_pipeline, 'run_experiment', change_config_after_training)
    with pytest.raises(RuntimeError, match='config changed'):
        run_pipeline.run_from_config(source, output, config_path, synthetic=True)
    record = json.loads((output / 'record.json').read_text(encoding='utf-8'))
    log = json.loads((output / 'run.log').read_text(encoding='utf-8'))
    assert record['status'] == log['status'] == 'failed_config_provenance'
    assert record['output_sha256']['run.log'] == run_pipeline.sha256(output / 'run.log')


@pytest.mark.parametrize('bad', [None, [], 'config'])
def test_settings_must_be_a_dict(source, tmp_path, bad):
    with pytest.raises(ValueError, match='dict'):
        run_pipeline.run_experiment(source, tmp_path / 'settings-failure', bad, synthetic=True)
