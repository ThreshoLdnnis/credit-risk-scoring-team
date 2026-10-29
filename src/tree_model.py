"""T05 LightGBM: train-only native missing handling and validation AUC stopping.

Bundle keys: model, feature_columns, settings, preprocessing, best_iteration,
validation_auc_history and versions. Inference explicitly uses best_iteration.
"""

import platform
from importlib.metadata import version
from numbers import Integral, Real

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import CLEAN_FEATURE_COLUMNS


def _features(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Features must be a nonempty DataFrame')
    if frame.columns.has_duplicates or set(frame.columns) != set(CLEAN_FEATURE_COLUMNS):
        raise ValueError('Features must contain exactly the 13 cleaned input columns')
    try:
        numeric = frame.loc[:, list(CLEAN_FEATURE_COLUMNS)].apply(pd.to_numeric, errors='raise')
        if any(pd.api.types.is_complex_dtype(numeric[c]) for c in numeric):
            raise ValueError('Features must be real numeric values')
        numeric = numeric.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError('Features must be real numeric values or missing') from error
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError('Features contain infinite values')
    return numeric


def _target(y, x):
    if isinstance(y, pd.Series) and not y.index.equals(x.index):
        raise ValueError('Label Series index must align with feature rows')
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError('Labels must be one-dimensional and match feature rows')
    if np.iscomplexobj(labels) or not pd.Series(labels).isin([0, 1]).all() or len(pd.unique(labels)) != 2:
        raise ValueError('Labels must contain both binary 0/1 classes without missing')
    return labels.astype(np.int64)


def _settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('settings must be a dict')
    values = settings.get('tree', {k: v for k, v in settings.items() if k != 'seed'})
    defaults = {'n_estimators': 500, 'learning_rate': .04, 'num_leaves': 15,
                'min_child_samples': 50, 'early_stopping_rounds': 30, 'metric': 'auc'}
    if not isinstance(values, dict) or set(values) - set(defaults):
        raise ValueError('Unknown tree settings')
    params = {**defaults, **values, 'seed': settings.get('seed', 42)}
    for name, minimum in [('n_estimators', 1), ('num_leaves', 2), ('min_child_samples', 1),
                          ('early_stopping_rounds', 1), ('seed', 0)]:
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
        params[name] = int(value)
    rate = params['learning_rate']
    if isinstance(rate, bool) or not isinstance(rate, Real) or not np.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if params['seed'] > 2**32 - 1 or params['metric'] != 'auc':
        raise ValueError('Use a valid seed and the agreed validation AUC metric')
    params['learning_rate'] = float(rate)
    return params


def fit_tree(train_x, train_y, validation_x, validation_y, settings: dict):
    """Fit native numeric/NaN bins on train; only validation AUC selects rounds.

    No imputation, scaling, class weighting, test inputs or resampling. Constant
    and entirely missing training columns remain in the schema. Zero stays a
    valid observed value. Full config and a tree sub-dict are both accepted.
    """
    train, validation = _features(train_x), _features(validation_x)
    labels, validation_labels = _target(train_y, train), _target(validation_y, validation)
    params = _settings(settings)
    model = lgb.LGBMClassifier(
        objective='binary', metric='auc', n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'], num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'], random_state=params['seed'],
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
        use_missing=True, zero_as_missing=False,
    )
    history = {}
    model.fit(train, labels, eval_set=[(validation, validation_labels)], eval_names=['validation'],
              eval_metric='auc', callbacks=[
                  lgb.early_stopping(params['early_stopping_rounds'], first_metric_only=True, verbose=False),
                  lgb.record_evaluation(history),
              ])
    best = int(model.best_iteration_)
    auc_history = [float(value) for value in history['validation']['auc']]
    if not 1 <= best <= params['n_estimators'] or not np.isfinite(auc_history).all():
        raise RuntimeError('LightGBM did not produce a valid validation AUC best iteration')
    return {
        'model': model, 'feature_columns': tuple(CLEAN_FEATURE_COLUMNS), 'settings': params,
        'preprocessing': {'strategy': 'native_nan', 'bin_fit_partition': 'train',
                          'use_missing': True, 'zero_as_missing': False,
                          'all_missing_or_constant': 'retain schema; native training bins'},
        'best_iteration': best, 'validation_auc_at_best': auc_history[best - 1],
        'validation_auc_history': auc_history, 'evaluation_rounds': len(auc_history),
        'model_parameters': model.get_params(),
        'versions': {'python': platform.python_version(), **{
            p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm']
        }},
    }


def predict_tree(bundle, features):
    """返回与输入同序的正类概率，使用保存的有效迭代与变换。"""
    ordered = _features(features)
    if tuple(bundle['feature_columns']) != tuple(CLEAN_FEATURE_COLUMNS):
        raise ValueError('Bundle feature schema differs from the cleaned input contract')
    best = bundle['best_iteration']
    if isinstance(best, bool) or not isinstance(best, Integral) or best < 1:
        raise ValueError('Bundle best_iteration must be positive')
    model = bundle['model']
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError('The fitted model has no unique positive class')
    probability = model.predict_proba(ordered, num_iteration=int(best))[:, positive[0]]
    if probability.shape != (len(ordered),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise RuntimeError('Model returned invalid probabilities')
    return probability
