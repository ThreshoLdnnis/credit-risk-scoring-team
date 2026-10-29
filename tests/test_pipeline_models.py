"""两模型共享入口：同分区、先冻结后测试、产物和评价对齐。"""

import copy
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import run_pipeline
from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_features,
    load_dataset,
    split_dataset,
)
from src.scorecard import predict_scorecard


@pytest.fixture
def model_source(tmp_path):
    rng = np.random.default_rng(203)
    size = 320
    frame = pd.DataFrame(rng.uniform(1, 9, (size, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame['age'] = rng.integers(20, 80, size)
    frame.loc[::9, 'MonthlyIncome'] = np.nan
    frame[TARGET_COLUMN] = (frame['DebtRatio'] + rng.normal(0, 2, size) > 5).astype(int)
    frame.insert(0, 'Unnamed: 0', np.arange(501, 501 + size))
    path = tmp_path / 'synthetic-models.csv'
    frame.to_csv(path, index=False)
    return path


def settings():
    return json.loads((run_pipeline.ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))


def test_two_models_fit_same_training_rows_and_freeze_all_before_test(model_source, tmp_path, monkeypatch):
    raw = load_dataset(model_source)
    features = clean_features(raw)
    positions = split_dataset(features, raw[TARGET_COLUMN])
    output = tmp_path / 'combined'
    config = settings()
    original_config = copy.deepcopy(config)
    baseline_fit, card_fit = run_pipeline.fit_baseline, run_pipeline.fit_scorecard
    original_predict = run_pipeline._predict_model
    calls = []

    def checked_baseline(x, y, config):
        np.testing.assert_array_equal(x.index, positions['train'])
        pd.testing.assert_series_equal(y, raw[TARGET_COLUMN].iloc[positions['train']])
        calls.append('baseline')
        return baseline_fit(x, y, config)

    def checked_card(x, y, card_config):
        np.testing.assert_array_equal(x.index, positions['train'])
        pd.testing.assert_series_equal(y, raw[TARGET_COLUMN].iloc[positions['train']])
        assert card_config == config['scorecard']
        calls.append('scorecard')
        return card_fit(x, y, card_config)

    def checked_predict(name, bundle, x):
        if np.array_equal(x.index, positions['test']):
            frozen = json.loads((output / 'selections.json').read_text(encoding='utf-8'))
            assert set(frozen) == {'baseline', 'scorecard'}
            assert calls == ['baseline', 'scorecard']
            assert all(v['source'] == 'validation' and not v['test_used_for_selection'] for v in frozen.values())
        return original_predict(name, bundle, x)

    monkeypatch.setattr(run_pipeline, 'fit_baseline', checked_baseline)
    monkeypatch.setattr(run_pipeline, 'fit_scorecard', checked_card)
    monkeypatch.setattr(run_pipeline, '_predict_model', checked_predict)
    record = run_pipeline.run_experiment(model_source, output, config, models=['baseline', 'scorecard'], synthetic=True)
    assert config == original_config
    assert record['status'] == 'synthetic_engineering_check'
    assert record['selected_models'] == ['baseline', 'scorecard']
    assert record['audit']['all_models_frozen_before_test_prediction']
    assert record['audit']['group_overlap'] == 0
    assert 'src/scorecard.py' in record['code_sha256']
    table = pd.read_csv(output / 'model_comparison.csv')
    assert len(table) == 6
    assert set(table['model']) == {'baseline', 'scorecard'}
    saved_positions = np.load(output / 'split_positions.npz')
    members = pd.read_csv(output / 'split_members.csv')
    for part in positions:
        np.testing.assert_array_equal(saved_positions[part], positions[part])
    for name in record['selected_models']:
        predictions = pd.read_csv(output / f'{name}_predictions.csv')
        assert predictions['row_position'].tolist() == list(range(len(raw)))
        np.testing.assert_array_equal(predictions['raw_row_id'], raw['Unnamed: 0'])
        np.testing.assert_array_equal(predictions['y'], raw[TARGET_COLUMN])
        assert predictions['partition'].tolist() == members['partition'].tolist()
        assert record['models'][name]['saved_loaded_max_output_difference'] == 0
        validation_pos = positions['validation']
        expected_threshold = run_pipeline.validation_threshold(
            raw[TARGET_COLUMN].iloc[validation_pos], predictions['probability'].iloc[validation_pos].to_numpy()
        )
        assert record['models'][name]['threshold'] == pytest.approx(expected_threshold)
        for part, pos in positions.items():
            reported = record['models'][name]['metrics'][part]
            assert reported['rows'] == len(pos)
            assert reported['auc'] == pytest.approx(roc_auc_score(predictions['y'].iloc[pos], predictions['probability'].iloc[pos]))
            assert reported['tn'] + reported['fp'] + reported['fn'] + reported['tp'] == len(pos)
    card_predictions = pd.read_csv(output / 'scorecard_predictions.csv')
    np.testing.assert_allclose(card_predictions['score'], card_predictions['base_points'] + card_predictions.filter(regex='^points_').sum(axis=1), atol=1e-10)
    bundle = joblib.load(output / 'scorecard.joblib')
    restored = predict_scorecard(bundle, features.iloc[positions['validation']])
    np.testing.assert_allclose(restored['score'], card_predictions['score'].iloc[positions['validation']], atol=1e-10)
    bins = json.loads((output / 'scorecard_binning.json').read_text(encoding='utf-8'))
    assert bins['selected_features'] == bundle['selected_features']


def test_scorecard_only_cli_has_correct_artifacts_and_actual_command(model_source, tmp_path):
    output = tmp_path / 'scorecard-only'
    result = subprocess.run([
        sys.executable, str(run_pipeline.ROOT / 'run_pipeline.py'), '--input', str(model_source),
        '--output', str(output), '--models', 'scorecard', '--synthetic',
    ], cwd=run_pipeline.ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stderr
    record = json.loads((output / 'record.json').read_text(encoding='utf-8'))
    assert record['selected_models'] == ['scorecard']
    assert record['model']['model'] == 'woe_iv_logistic_scorecard'
    assert record['metrics'] == record['models']['scorecard']['metrics']
    assert record['command'][-3:] == ['--models', 'scorecard', '--synthetic']
    assert record['audit']['saved_loaded_max_probability_difference'] == 0
    assert (output / 'scorecard.joblib').is_file()
    assert not (output / 'baseline.joblib').exists()
    assert set(json.loads(result.stdout)['models']) == {'scorecard'}


@pytest.mark.parametrize('models', [[], ['baseline', 'baseline'], ['tree'], 'baseline', None])
def test_invalid_models_fail_before_creating_output(model_source, tmp_path, models):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='models'):
        run_pipeline.run_experiment(model_source, output, settings(), synthetic=True, models=models)
    assert not output.exists()


def test_missing_scorecard_settings_fail_before_creating_output(model_source, tmp_path):
    config = settings()
    del config['scorecard']
    output = tmp_path / 'missing-settings'
    with pytest.raises(ValueError, match='scorecard'):
        run_pipeline.run_experiment(model_source, output, config, synthetic=True, models=['scorecard'])
    assert not output.exists()


def test_scorecard_failure_cannot_leave_completed_record(model_source, tmp_path, monkeypatch):
    output = tmp_path / 'failed-fit'

    def failed(*args, **kwargs):
        raise RuntimeError('scorecard did not converge')

    monkeypatch.setattr(run_pipeline, 'fit_scorecard', failed)
    with pytest.raises(RuntimeError, match='did not converge'):
        run_pipeline.run_experiment(model_source, output, settings(), synthetic=True, models=['baseline', 'scorecard'])
    assert not (output / 'record.json').exists()
    assert not (output / 'selections.json').exists()
    assert not (output / 'baseline.joblib').exists()


def test_pipeline_rejects_prediction_row_reordering(model_source, tmp_path, monkeypatch):
    actual = run_pipeline.predict_scorecard

    def reordered(bundle, features):
        return actual(bundle, features).iloc[::-1]

    monkeypatch.setattr(run_pipeline, 'predict_scorecard', reordered)
    with pytest.raises(RuntimeError, match='row order'):
        run_pipeline.run_experiment(model_source, tmp_path / 'reordered', settings(), synthetic=True, models=['scorecard'])
