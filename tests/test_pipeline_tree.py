"""共享入口的树模型训练边界、三模型时序与真实推理产物检查。"""

import copy
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

import run_pipeline
from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_features,
    load_dataset,
    split_dataset,
)
from src.tree_model import predict_tree


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(37)
    frame = pd.DataFrame(rng.uniform(1, 9, (320, 10)), columns=FEATURE_COLUMNS)
    frame['age'] = rng.integers(20, 80, len(frame))
    frame.loc[::8, 'MonthlyIncome'] = np.nan
    frame[TARGET_COLUMN] = (frame.DebtRatio + rng.normal(0, 1, len(frame)) > 5).astype(int)
    frame.insert(0, 'Unnamed: 0', np.arange(101, 101+len(frame)))
    path = tmp_path / 'synthetic.csv'
    frame.to_csv(path, index=False)
    return path


def settings():
    config = json.loads((run_pipeline.ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))
    config['tree'].update(n_estimators=20, early_stopping_rounds=4, min_child_samples=10)
    return config


def test_three_models_share_rows_and_freeze_tree_rounds_and_all_thresholds_before_test(source, tmp_path, monkeypatch):
    raw = load_dataset(source)
    features = clean_features(raw)
    y = raw[TARGET_COLUMN]
    positions = split_dataset(features, y)
    output = tmp_path / 'three-models'
    config = settings()
    original_config = copy.deepcopy(config)
    fits = []
    for name in ['baseline', 'scorecard', 'tree']:
        original = getattr(run_pipeline, 'fit_'+name)

        def checked_fit(x, labels, *args, _name=name, _original=original):
            pd.testing.assert_frame_equal(x, features.iloc[positions['train']])
            pd.testing.assert_series_equal(labels, y.iloc[positions['train']])
            if _name == 'tree':
                vx, vy, received = args
                pd.testing.assert_frame_equal(vx, features.iloc[positions['validation']])
                pd.testing.assert_series_equal(vy, y.iloc[positions['validation']])
                assert received == config
            fits.append(_name)
            return _original(x, labels, *args)

        monkeypatch.setattr(run_pipeline, 'fit_'+name, checked_fit)
    original_predict = run_pipeline._predict_model

    def checked_predict(name, bundle, x):
        if np.array_equal(x.index, positions['test']):
            assert fits == ['baseline', 'scorecard', 'tree']
            frozen = json.loads((output / 'selections.json').read_text(encoding='utf-8'))
            assert set(frozen) == set(fits)
            assert all(v['source'] == 'validation' and not v['test_used_for_selection'] for v in frozen.values())
        return original_predict(name, bundle, x)

    monkeypatch.setattr(run_pipeline, '_predict_model', checked_predict)
    record = run_pipeline.run_experiment(source, output, config, synthetic=True, models=['tree','scorecard','baseline'])
    assert config == original_config
    assert record['selected_models'] == ['baseline','scorecard','tree']
    assert record['metrics'] == record['models']['baseline']['metrics']
    assert len(pd.read_csv(output / 'model_comparison.csv')) == 9
    tree = joblib.load(output / 'tree.joblib')
    metadata = record['models']['tree']
    assert metadata['model'] == 'lightgbm'
    assert metadata['best_iteration'] == tree['best_iteration']
    assert metadata['evaluation_rounds'] == len(metadata['validation_auc_history'])
    assert metadata['saved_loaded_max_output_difference'] == 0
    rows = pd.read_csv(output / 'tree_predictions.csv', float_precision='round_trip')
    np.testing.assert_array_equal(rows.row_position, np.arange(len(raw)))
    np.testing.assert_array_equal(rows.y, y)
    for part, pos in positions.items():
        np.testing.assert_array_equal(predict_tree(tree, features.iloc[pos]), rows.probability.to_numpy()[pos])
        assert rows.partition.iloc[pos].eq(part).all()
    for paths in record['figures'].values():
        assert set(paths) == {'roc','pr','calibration','credit_score'}
        for path in paths.values():
            assert record['output_sha256'][path] == run_pipeline.sha256(output / path)
    assert 'src/tree_model.py' in record['code_sha256']
    assert 'src/evaluation.py' in record['code_sha256']
    assert 'lightgbm' in record['environment']['packages']


def test_tree_only_cli_records_its_model_and_six_plots(source, tmp_path):
    output = tmp_path / 'tree-cli'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(settings()), encoding='utf-8')
    result = subprocess.run([sys.executable, str(run_pipeline.ROOT / 'run_pipeline.py'),
        '--input', str(source), '--output', str(output), '--config', str(config),
        '--models', 'tree', '--synthetic'], cwd=run_pipeline.ROOT, capture_output=True,
        text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stderr
    record = json.loads((output / 'record.json').read_text(encoding='utf-8'))
    assert record['selected_models'] == ['tree']
    assert record['model']['model'] == 'lightgbm'
    assert record['metrics'] == record['models']['tree']['metrics']
    assert record['command'][-3:] == ['--models','tree','--synthetic']
    assert all(set(paths) == {'roc','pr','calibration'} for paths in record['figures'].values())
    assert not (output / 'scorecard.joblib').exists()


def test_missing_tree_configuration_fails_before_output(source, tmp_path):
    config = settings()
    del config['tree']
    output = tmp_path / 'missing-tree'
    with pytest.raises(ValueError, match='tree settings'):
        run_pipeline.run_experiment(source, output, config, synthetic=True, models=['tree'])
    assert not output.exists()


def test_failed_tree_fit_cannot_leave_a_completed_record(source, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('tree fit failed')
    monkeypatch.setattr(run_pipeline, 'fit_tree', fail)
    output = tmp_path / 'failed-tree'
    with pytest.raises(RuntimeError, match='tree fit failed'):
        run_pipeline.run_experiment(source, output, settings(), synthetic=True, models=['baseline','tree'])
    assert not (output / 'selections.json').exists()
    assert not (output / 'record.json').exists()


def test_tree_invalid_prediction_cannot_be_saved_as_success(source, tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, 'predict_tree', lambda bundle, x: np.full(len(x), np.nan))
    output = tmp_path / 'invalid-probability'
    with pytest.raises(RuntimeError, match='invalid values'):
        run_pipeline.run_experiment(source, output, settings(), synthetic=True, models=['tree'])
    assert not (output / 'record.json').exists()


def test_compatibility_helpers_delegate_to_shared_evaluation(monkeypatch):
    marker = object()
    monkeypatch.setattr(run_pipeline, 'select_threshold', lambda y,p: marker)
    monkeypatch.setattr(run_pipeline, 'evaluate_binary', lambda y,p,t: marker)
    assert run_pipeline.validation_threshold([0,1], [.1,.8]) is marker
    assert run_pipeline.metrics([0,1], [.1,.8], .5) is marker


def test_figure_failure_does_not_produce_successful_record(source, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('plot failed')
    monkeypatch.setattr(run_pipeline, 'save_evaluation_figures', fail)
    output = tmp_path / 'failed-figures'
    with pytest.raises(RuntimeError, match='plot failed'):
        run_pipeline.run_experiment(source, output, settings(), synthetic=True, models=['tree'])
    assert not (output / 'record.json').exists()
    assert not (output / 'run.log').exists()
