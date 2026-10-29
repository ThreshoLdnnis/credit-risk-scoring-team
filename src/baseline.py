"""Train-only median imputation, missing indicators, scaling and L2 logistic LR.

The bundle is a joblib-serializable dict with the fitted ``pipeline``, canonical
``feature_columns``, effective ``settings``, ``versions`` and ``positive_class``.
No validation/test data enter fit_baseline; inference only calls transform and
predict_proba. See experiments/morebadly/README.md for persistence and integration.
"""

import platform
import warnings
from importlib.metadata import version
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CLEAN_FEATURE_COLUMNS


def _features(frame, columns):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Features must be a nonempty DataFrame')
    if frame.columns.has_duplicates or set(frame.columns) != set(columns):
        raise ValueError('Features must contain exactly the 13 cleaned feature columns')
    ordered = frame.loc[:, list(columns)]
    try:
        numeric = ordered.apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as error:
        raise ValueError('Features contain non-numeric values') from error
    if any(pd.api.types.is_complex_dtype(numeric[c]) for c in columns):
        raise ValueError('Features must be real numeric values')
    numeric = numeric.astype(float)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError('Features contain infinite values')
    return numeric


def _settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('settings must be a dict')
    config = settings.get('baseline', {k: v for k, v in settings.items() if k != 'seed'})
    if not isinstance(config, dict) or set(config) - {'C', 'penalty', 'max_iter'}:
        raise ValueError('baseline settings accept only C, penalty and max_iter')
    effective = {'C': 1.0, 'penalty': 'l2', 'max_iter': 2000, **config}
    strength, iterations = effective['C'], effective['max_iter']
    seed = settings.get('seed', 42)
    if isinstance(strength, bool) or not isinstance(strength, Real) or not np.isfinite(strength) or strength <= 0:
        raise ValueError('C must be a finite positive number')
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError('max_iter must be a positive integer')
    if effective['penalty'] != 'l2':
        raise ValueError('The baseline requires L2 regularization')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer in [0, 2**32 - 1]')
    return {**effective, 'C': float(strength), 'max_iter': int(iterations), 'seed': int(seed), 'solver': 'lbfgs'}


def fit_baseline(train_x, train_y, settings: dict):
    """Fit using only training rows; settings accepts the full config or LR dict.

    Training-observed missing columns receive binary missing indicators.
    Entirely missing training columns are retained and filled with zero; their
    indicator is retained too. Constant columns use StandardScaler's unit scale.
    Named inference columns may be reordered; labels/IDs/extras are rejected.
    """
    features = _features(train_x, CLEAN_FEATURE_COLUMNS)
    if isinstance(train_y, pd.Series) and not train_y.index.equals(train_x.index):
        raise ValueError('train_y index must align with train_x in row order')
    labels = np.asarray(train_y)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError('train_y must have one label per training row')
    if not pd.Series(labels).isin([0, 1]).all() or len(pd.unique(labels)) != 2:
        raise ValueError('train_y must contain both binary 0/1 classes without missing')
    params = _settings(settings)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=params['C'], penalty=params['penalty'], max_iter=params['max_iter'],
            solver=params['solver'], random_state=params['seed'],
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            pipeline.fit(features, labels.astype(np.int64))
        except ConvergenceWarning as error:
            raise RuntimeError('LR did not converge; review training and max_iter before evaluation') from error
    return {
        'pipeline': pipeline, 'feature_columns': tuple(CLEAN_FEATURE_COLUMNS),
        'settings': params, 'positive_class': 1,
        'versions': {'python': platform.python_version(), **{
            package: version(package) for package in ['numpy', 'pandas', 'scikit-learn', 'joblib']
        }},
    }


def predict_baseline(bundle, features):
    """返回与输入行逐一对齐的 y=1 概率，不在推理时重新 fit。"""
    ordered = _features(features, bundle['feature_columns'])
    pipeline = bundle['pipeline']
    positive_index = np.flatnonzero(pipeline.classes_ == bundle['positive_class'])
    if len(positive_index) != 1:
        raise ValueError('The fitted bundle has no unique positive class')
    probability = pipeline.predict_proba(ordered)[:, positive_index[0]]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Model returned invalid probabilities')
    return probability
