"""T06 positive-class Tree SHAP, frozen rounds and prediction-only contract."""

import copy
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.data import CLEAN_FEATURE_COLUMNS, FEATURE_COLUMNS, clean_features
from src.explanation import explain_tree
from src.tree_model import fit_tree, predict_tree


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(61)
    raw = pd.DataFrame(rng.uniform(1, 9, (320, 10)), columns=FEATURE_COLUMNS)
    raw['age'] = rng.integers(20, 80, len(raw))
    raw.loc[::7, 'MonthlyIncome'] = np.nan
    raw['NumberOfDependents'] = np.nan
    y = pd.Series((raw.DebtRatio > 5).astype(int), index=raw.index)
    features = clean_features(raw)
    bundle = fit_tree(features.iloc[:240], y.iloc[:240], features.iloc[240:], y.iloc[240:],
                      {'n_estimators': 30, 'early_stopping_rounds': 5, 'min_child_samples': 10, 'num_leaves': 7})
    return bundle, features.iloc[240:]


def test_exact_additivity_positive_class_and_native_reference(fitted, monkeypatch):
    bundle, features = fitted
    before = bundle['model'].booster_.model_to_string(num_iteration=-1)
    original = features.copy(deep=True)

    def no_fit(*args, **kwargs):
        pytest.fail('Explanation must not fit or update the model')

    monkeypatch.setattr(lgb.LGBMClassifier, 'fit', no_fit)
    monkeypatch.setattr(lgb.Booster, 'update', no_fit)
    result = explain_tree(bundle, features)
    values, base = np.asarray(result['shap_values']), np.asarray(result['base_log_odds'])
    raw = np.asarray(result['raw_log_odds'])
    np.testing.assert_allclose(base + values.sum(axis=1), raw, rtol=0, atol=1e-6)
    np.testing.assert_allclose(expit(raw), predict_tree(bundle, features), rtol=0, atol=1e-10)
    native = bundle['model'].booster_.predict(features, pred_contrib=True, num_iteration=bundle['best_iteration'])
    np.testing.assert_allclose(values, native[:, :-1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(base, native[:, -1], rtol=0, atol=1e-10)
    assert result['feature_names'] == list(CLEAN_FEATURE_COLUMNS)
    assert result['metadata']['background_partition'] == 'train'
    assert result['metadata']['background_training_rows'] == 240
    assert result['metadata']['explicit_background_rows'] == 0
    assert result['metadata']['feature_perturbation'] == 'tree_path_dependent'
    json.dumps(result, allow_nan=False)
    assert bundle['model'].booster_.model_to_string(num_iteration=-1) == before
    pd.testing.assert_frame_equal(features, original)


def test_fixed_rounds_serialization_order_and_batch_independence(fitted, tmp_path):
    bundle, features = fitted
    later_trees = lgb.LGBMClassifier(objective='binary', n_estimators=8, num_leaves=7,
                                    min_child_samples=10, n_jobs=1, verbosity=-1)
    later_trees.fit(features, (features.DebtRatio > 5).astype(int))
    frozen_bundle = {**bundle, 'model': later_trees, 'best_iteration': 1}
    assert later_trees.booster_.current_iteration() > 1
    path = tmp_path / 'tree.joblib'
    joblib.dump(frozen_bundle, path)
    restored = joblib.load(path)
    full = explain_tree(restored, features)
    requested = features.iloc[[7, 3, 2]].loc[:, list(reversed(features.columns))]
    selected = explain_tree(restored, requested)
    np.testing.assert_array_equal(selected['shap_values'], np.asarray(full['shap_values'])[[7, 3, 2]])
    np.testing.assert_array_equal(selected['probability'], np.asarray(full['probability'])[[7, 3, 2]])
    np.testing.assert_array_equal(selected['base_log_odds'], np.asarray(full['base_log_odds'])[[7, 3, 2]])
    assert selected['row_index_repr'] == [repr(v) for v in requested.index]
    assert selected['metadata']['best_iteration'] == 1
    assert not np.allclose(full['probability'], later_trees.predict_proba(features)[:, 1])
    one = explain_tree(restored, features.iloc[[7]])
    np.testing.assert_array_equal(one['shap_values'][0], full['shap_values'][7])


@pytest.mark.parametrize('problem', ['empty', 'missing_column', 'extra', 'duplicate', 'infinity', 'complex', 'text'])
def test_invalid_features(fitted, problem):
    bundle, features = fitted
    frame = features.iloc[:3].copy()
    if problem == 'empty':
        frame = frame.iloc[:0]
    elif problem == 'missing_column':
        frame = frame.drop(columns='age')
    elif problem == 'extra':
        frame['untrained'] = 1
    elif problem == 'duplicate':
        frame.columns = ['age'] * len(frame.columns)
    else:
        frame['age'] = {'infinity': np.inf, 'complex': 1j, 'text': 'invalid'}[problem]
    with pytest.raises(ValueError):
        explain_tree(bundle, frame)


@pytest.mark.parametrize('best', [0, -1, True, 1.5, 10000])
def test_invalid_best_iteration(fitted, best):
    bundle, features = fitted
    with pytest.raises(ValueError, match='best_iteration'):
        explain_tree({**bundle, 'best_iteration': best}, features)


def test_wrong_bundle_and_nonfinite_or_inconsistent_shap(fitted, monkeypatch):
    import src.explanation as module

    bundle, features = fitted
    for wrong in [None, {}, {'model': lgb.LGBMClassifier()}]:
        with pytest.raises(ValueError):
            explain_tree(wrong, features)
    wrong = copy.copy(bundle)
    wrong['feature_columns'] = ('age',)
    with pytest.raises(ValueError):
        explain_tree(wrong, features)

    class BrokenExplainer:
        expected_value = 0.

        def __init__(self, *args, **kwargs):
            pass

        def shap_values(self, x, **kwargs):
            return np.full(x.shape, np.nan)

    monkeypatch.setattr(module.shap, 'TreeExplainer', BrokenExplainer)
    with pytest.raises(RuntimeError, match='invalid contributions'):
        explain_tree(bundle, features)
    monkeypatch.setattr(BrokenExplainer, 'shap_values', lambda self, x, **kwargs: np.zeros(x.shape))
    with pytest.raises(RuntimeError, match='additivity'):
        explain_tree(bundle, features)
