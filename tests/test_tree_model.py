"""T05 native missing handling, validation stopping and reusable inference."""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, clean_features
from src.tree_model import fit_tree, predict_tree


@pytest.fixture
def samples():
    rng = np.random.default_rng(5)
    raw = pd.DataFrame(rng.uniform(1, 9, (320, 10)), columns=FEATURE_COLUMNS)
    raw['age'] = rng.integers(20, 80, len(raw))
    raw.loc[::7, 'MonthlyIncome'] = np.nan
    raw['NumberOfDependents'] = np.nan
    y = pd.Series((raw.DebtRatio > 5).astype(int), index=raw.index)
    x = clean_features(raw)
    settings = {'n_estimators': 40, 'early_stopping_rounds': 5, 'min_child_samples': 10, 'num_leaves': 7}
    return x.iloc[:240], y.iloc[:240], x.iloc[240:], y.iloc[240:], settings


def test_best_iteration_uses_validation_auc_only(samples, monkeypatch):
    train, y, validation, vy, settings = samples
    constant_validation = validation * 0
    before_train, before_validation = train.copy(deep=True), constant_validation.copy(deep=True)
    original_fit = lgb.LGBMClassifier.fit
    observed = {}

    def checked_fit(self, x, labels, **kwargs):
        pd.testing.assert_frame_equal(x, train.astype(float))
        assert len(kwargs['eval_set']) == 1
        pd.testing.assert_frame_equal(kwargs['eval_set'][0][0], constant_validation.astype(float))
        np.testing.assert_array_equal(kwargs['eval_set'][0][1], vy)
        assert kwargs['eval_names'] == ['validation'] and kwargs['eval_metric'] == 'auc'
        observed['seen'] = True
        return original_fit(self, x, labels, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, 'fit', checked_fit)
    bundle = fit_tree(train, y, constant_validation, vy, settings)
    assert observed['seen']
    assert bundle['best_iteration'] == 1
    assert bundle['evaluation_rounds'] <= 1 + settings['early_stopping_rounds']
    assert bundle['validation_auc_at_best'] == .5
    assert bundle['preprocessing']['strategy'] == 'native_nan'
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(constant_validation, before_validation)


def test_roundtrip_order_best_round_and_no_refit(samples, tmp_path, monkeypatch):
    bundle = fit_tree(*samples)
    x = samples[2]
    expected = predict_tree(bundle, x)
    assert expected.mean() > 0 and np.all((expected >= 0) & (expected <= 1))
    np.testing.assert_array_equal(expected, bundle['model'].predict_proba(x.astype(float), num_iteration=bundle['best_iteration'])[:, 1])
    assert expected[samples[3] == 1].mean() > expected[samples[3] == 0].mean()
    joblib.dump(bundle, tmp_path / 'tree.joblib')
    restored = joblib.load(tmp_path / 'tree.joblib')

    def fail_fit(*args, **kwargs):
        pytest.fail('Prediction called fit')

    monkeypatch.setattr(lgb.LGBMClassifier, 'fit', fail_fit)
    np.testing.assert_array_equal(predict_tree(restored, x), expected)
    permuted = x.iloc[[7, 3, 2]].loc[:, list(reversed(x.columns))]
    np.testing.assert_array_equal(predict_tree(restored, permuted), expected[[7, 3, 2]])
    changed = x.iloc[:4].copy()
    changed['NumberOfDependents'] = 2
    changed['age'] = np.nan
    assert np.isfinite(predict_tree(restored, changed)).all()


def test_full_config_and_reproducibility(samples):
    train, y, validation, vy, settings = samples
    first = fit_tree(train, y, validation, vy, {'seed': 42, 'tree': settings})
    second = fit_tree(train, y, validation, vy, {'seed': 42, 'tree': settings})
    np.testing.assert_array_equal(predict_tree(first, validation), predict_tree(second, validation))


@pytest.mark.parametrize('field,value', [('metric', 'binary_logloss'), ('n_estimators', 0),
    ('early_stopping_rounds', 0), ('num_leaves', 1), ('learning_rate', np.nan),
    ('min_child_samples', True), ('seed', -1), ('seed', 2**32), ('typo', 1)])
def test_invalid_settings(samples, field, value):
    train, y, validation, vy, settings = samples
    with pytest.raises(ValueError):
        fit_tree(train, y, validation, vy, {**settings, field: value})


@pytest.mark.parametrize('where', ['train', 'validation'])
@pytest.mark.parametrize('problem', ['index', 'single_class', 'missing_label', 'invalid_label', 'length', 'infinite', 'missing_column', 'extra', 'complex'])
def test_invalid_training_or_validation_inputs(samples, where, problem):
    train, y, validation, vy, settings = samples
    x, target = (train.copy(), y.copy()) if where == 'train' else (validation.copy(), vy.copy())
    if problem == 'index':
        target = target.iloc[::-1]
    elif problem == 'single_class':
        target[:] = 0
    elif problem == 'missing_label':
        target = target.astype(float)
        target.iloc[0] = np.nan
    elif problem == 'invalid_label':
        target.iloc[0] = 2
    elif problem == 'length':
        target = target.to_numpy()[:-1]
    elif problem == 'infinite':
        x.iloc[0, 0] = np.inf
    elif problem == 'missing_column':
        x = x.drop(columns='age')
    elif problem == 'extra':
        x['SeriousDlqin2yrs'] = target
    else:
        x[x.columns[0]] = 1 + 2j
    args = (x, target, validation, vy, settings) if where == 'train' else (train, y, x, target, settings)
    with pytest.raises(ValueError):
        fit_tree(*args)
