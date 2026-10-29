"""T03 behavioral checks with artificial data; no claimed Kaggle performance."""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_pipeline
from src.baseline import fit_baseline, predict_baseline
from src.data import (
    CLEAN_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_features,
)


@pytest.fixture
def training():
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.uniform(1, 10, (240, 10)), columns=FEATURE_COLUMNS)
    raw['age'] = rng.integers(18, 80, len(raw))
    raw.loc[::4, 'MonthlyIncome'] = np.nan
    raw.loc[::9, 'NumberOfDependents'] = np.nan
    labels = pd.Series((raw['DebtRatio'] > 5.5).astype(int), index=raw.index)
    return clean_features(raw), labels


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / 'experiments/morebadly/run_t03.py'
    spec = importlib.util.spec_from_file_location('t03_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_only_statistics_and_inference_never_fit(training, monkeypatch):
    features, y = training
    original_x, original_y = features.copy(deep=True), y.copy(deep=True)
    bundle = fit_baseline(features, y, {})
    pipeline = bundle['pipeline']
    imputer, scaler = pipeline.named_steps['imputer'], pipeline.named_steps['scaler']
    expected = features.median().fillna(0).to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    np.testing.assert_allclose(scaler.mean_, imputer.transform(features).mean(axis=0))
    before_statistics, before_means = imputer.statistics_.copy(), scaler.mean_.copy()

    def fail_fit(*args, **kwargs):
        pytest.fail('Inference attempted to fit preprocessing')

    monkeypatch.setattr(SimpleImputer, 'fit', fail_fit)
    monkeypatch.setattr(StandardScaler, 'fit', fail_fit)
    monkeypatch.setattr(LogisticRegression, 'fit', fail_fit)
    extreme = features.iloc[:8].copy()
    extreme.loc[:, 'MonthlyIncome'] = 1e8
    probability = predict_baseline(bundle, extreme)
    assert np.isfinite(probability).all()
    np.testing.assert_array_equal(imputer.statistics_, before_statistics)
    np.testing.assert_array_equal(scaler.mean_, before_means)
    assert_frame_equal(features, original_x)
    assert_series_equal(y, original_y)


def test_positive_probability_row_alignment_and_named_column_order(training):
    features, y = training
    bundle = fit_baseline(features, y, {})
    probability = predict_baseline(bundle, features)
    assert probability.shape == (len(features),)
    assert np.all((probability >= 0) & (probability <= 1))
    np.testing.assert_array_equal(probability, bundle['pipeline'].predict_proba(features)[:, 1])
    assert probability[y.eq(1)].mean() > probability[y.eq(0)].mean()
    permuted = features.iloc[[8, 2, 9, 0]].loc[:, list(reversed(features.columns))]
    np.testing.assert_allclose(predict_baseline(bundle, permuted), probability[[8, 2, 9, 0]], rtol=0, atol=1e-14)


def test_round_trip_and_reproducibility(training, tmp_path):
    features, y = training
    config = {'seed': 42, 'baseline': {'C': 0.5, 'penalty': 'l2', 'max_iter': 2000}, 'split': {}}
    bundle = fit_baseline(features, y, config)
    path = tmp_path / 'baseline.joblib'
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(predict_baseline(bundle, features), predict_baseline(restored, features))
    np.testing.assert_array_equal(predict_baseline(bundle, features), predict_baseline(fit_baseline(features, y, config), features))
    assert bundle['feature_columns'] == CLEAN_FEATURE_COLUMNS
    assert bundle['settings']['C'] == 0.5
    assert bundle['versions']['scikit-learn'] == '1.5.2'


def test_all_missing_and_constant_columns_and_new_missing(training):
    features, y = training
    features['MonthlyIncome'] = np.nan
    features['NumberOfDependents'] = 2.0
    bundle = fit_baseline(features, y, {})
    imputer = bundle['pipeline'].named_steps['imputer']
    income_index = list(features.columns).index('MonthlyIncome')
    assert imputer.statistics_[income_index] == 0.0
    assert income_index in imputer.indicator_.features_
    assert imputer.transform(features).shape[1] == len(features.columns) + len(imputer.indicator_.features_)
    changed = features.iloc[:6].copy()
    changed['MonthlyIncome'] = 100
    changed['NumberOfDependents'] = np.nan
    assert np.isfinite(predict_baseline(bundle, changed)).all()


@pytest.mark.parametrize('damage', ['missing', 'extra', 'duplicate', 'infinite', 'text', 'complex', 'object_complex', 'empty'])
def test_invalid_features_rejected_during_training_and_prediction(training, damage):
    features, y = training
    bundle = fit_baseline(features, y, {})
    bad = features.copy()
    if damage == 'missing':
        bad = bad.drop(columns='age')
    elif damage == 'extra':
        bad[TARGET_COLUMN] = y
    elif damage == 'duplicate':
        bad.columns = ['age'] * len(bad.columns)
    elif damage == 'infinite':
        bad.iloc[0, 0] = np.inf
    elif damage == 'text':
        bad[bad.columns[0]] = 'invalid'
    elif damage == 'complex':
        bad[bad.columns[0]] = 1 + 2j
    elif damage == 'object_complex':
        bad[bad.columns[0]] = bad[bad.columns[0]].astype(object)
        bad.iloc[0, 0] = 1 + 2j
    else:
        bad = bad.iloc[:0]
    with pytest.raises(ValueError):
        fit_baseline(bad, y, {})
    with pytest.raises(ValueError):
        predict_baseline(bundle, bad)


@pytest.mark.parametrize('damage', ['index', 'length', '2d', 'single_class', 'missing', 'invalid', 'text'])
def test_invalid_or_misaligned_labels(training, damage):
    features, y = training
    bad = y.copy()
    if damage == 'index':
        bad = bad.iloc[::-1]
    elif damage == 'length':
        bad = bad.to_numpy()[:-1]
    elif damage == '2d':
        bad = bad.to_numpy()[:, None]
    elif damage == 'single_class':
        bad[:] = 0
    elif damage == 'missing':
        bad = bad.astype(float)
        bad.iloc[0] = np.nan
    elif damage == 'invalid':
        bad.iloc[0] = 2
    else:
        bad = bad.astype(str)
    with pytest.raises(ValueError):
        fit_baseline(features, bad, {})


@pytest.mark.parametrize('settings', [
    None, {'C': 0}, {'C': np.nan}, {'C': np.inf}, {'C': True}, {'max_iter': 0},
    {'max_iter': 2.5}, {'penalty': 'l1'}, {'seed': -1}, {'seed': True},
    {'baseline': None}, {'baseline': {'typo': 1}},
])
def test_invalid_settings(training, settings):
    with pytest.raises(ValueError):
        fit_baseline(*training, settings)


def test_nonconvergence_is_not_reported_as_success(training):
    with pytest.raises(RuntimeError, match='did not converge'):
        fit_baseline(*training, {'max_iter': 1})


def test_threshold_ties_and_inclusive_rule(runner):
    assert runner.validation_threshold([0, 1], np.array([0.4, 0.4])) == 0.4
    # At .8 and .2 Youden gain is tied; the larger threshold must win.
    assert runner.validation_threshold([0, 1, 0, 1], np.array([0.1, 0.2, 0.7, 0.8])) == 0.8
    result = runner.metrics([0, 0, 1, 1], np.array([0.1, 0.4, 0.35, 0.8]), 0.4)
    assert result['auc'] == 0.75
    assert result['brier'] == pytest.approx(0.158125)
    assert [result[c] for c in ['tn', 'fp', 'fn', 'tp']] == [1, 1, 1, 1]
    assert 'ap' in result and 'pr_auc_trapezoidal' in result


def test_synthetic_closed_loop_is_explicit_and_keeps_evidence_aligned(training, runner, tmp_path, monkeypatch):
    features, y = training
    source = tmp_path / 'synthetic-training.csv'
    raw = features.loc[:, list(FEATURE_COLUMNS)].copy()
    raw.insert(0, 'Unnamed: 0', np.arange(1, len(raw) + 1))
    raw[TARGET_COLUMN] = y
    raw.to_csv(source, index=False)
    original = source.read_bytes()
    output = tmp_path / 'smoke'
    with pytest.raises(ValueError, match='SHA256'):
        runner.run(source, output)
    assert not output.exists()
    expected_positions = run_pipeline.split_dataset(features, y, seed=42)
    original_fit, original_predict = run_pipeline.fit_baseline, run_pipeline.predict_baseline
    fit_calls = []

    def checked_fit(x, labels, settings):
        np.testing.assert_array_equal(x.index, expected_positions['train'])
        np.testing.assert_array_equal(labels, y.iloc[expected_positions['train']])
        fit_calls.append(len(x))
        return original_fit(x, labels, settings)

    def checked_predict(bundle, x):
        if np.array_equal(x.index, expected_positions['test']):
            assert (output / 'selection.json').is_file()
            assert json.loads((output / 'selection.json').read_text())['source'] == 'validation'
        return original_predict(bundle, x)

    monkeypatch.setattr(run_pipeline, 'fit_baseline', checked_fit)
    monkeypatch.setattr(run_pipeline, 'predict_baseline', checked_predict)
    record = runner.run(source, output, synthetic=True)
    assert fit_calls == [len(expected_positions['train'])]
    assert record['status'] == 'synthetic_engineering_check'
    assert record['source']['official_url'] is None
    assert record['audit']['saved_loaded_max_probability_difference'] == 0.0
    assert record['audit']['group_overlap'] == 0
    evidence = pd.read_csv(output / 'predictions.csv')
    assert evidence['row_position'].tolist() == list(range(len(raw)))
    assert evidence['raw_row_id'].tolist() == raw['Unnamed: 0'].tolist()
    np.testing.assert_array_equal(evidence['y'], y)
    positions = np.load(output / 'split_positions.npz')
    for name in ['train', 'validation', 'test']:
        assert set(evidence.loc[positions[name], 'partition']) == {name}
        assert record['metrics'][name]['rows'] == len(positions[name])
    assert source.read_bytes() == original
    assert json.loads((output / 'selection.json').read_text())['test_used_for_selection'] is False
    with pytest.raises(FileExistsError):
        runner.run(source, output, synthetic=True)


def test_missing_data_has_actionable_error(runner, tmp_path):
    with pytest.raises(FileNotFoundError, match='Kaggle'):
        runner.run(tmp_path / 'absent.csv', tmp_path / 'output')


@pytest.mark.parametrize('setting,value', [('seed', 7), ('seed', True), ('seed', 42.0), ('split', {}), ('threshold', {})])
def test_runner_refuses_different_shared_partition_or_selection_protocol(runner, tmp_path, setting, value):
    source = tmp_path / 'synthetic.csv'
    source.write_text('engineering check only', encoding='utf-8')
    config = json.loads((runner.ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))
    config[setting] = value
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='shared|split|threshold'):
        runner.run(source, tmp_path / 'output', config_path, synthetic=True)
