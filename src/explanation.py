"""Exact Tree SHAP for the frozen binary LightGBM, in raw log-odds.

Background: training path/leaf counts stored in the trees. No fitting occurs.
Results are JSON-compatible lists in canonical feature/input row order.
"""

from numbers import Integral

import lightgbm as lgb
import numpy as np
import shap
from scipy.special import expit

from src.tree_model import _features, predict_tree


def explain_tree(bundle, features) -> dict:
    """Explain y=1 using best_iteration; contributions are not causal effects.

    A separate Booster limited to the selected trees prevents SHAP from seeing
    later rounds or changing the prediction object. Units are log-odds.
    """
    if not isinstance(bundle, dict) or not isinstance(bundle.get('model'), lgb.LGBMClassifier):
        raise ValueError('Expected a fitted T05 binary LightGBM bundle')
    model = bundle['model']
    if not getattr(model, 'fitted_', False) or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Expected a fitted binary model with classes [0, 1]')
    best = bundle.get('best_iteration')
    if isinstance(best, bool) or not isinstance(best, Integral) or not 1 <= best <= model.booster_.current_iteration():
        raise ValueError('best_iteration must identify fitted trees')
    if model.booster_.params.get('objective') != 'binary':
        raise ValueError('Only the agreed binary logistic objective is supported')
    ordered = _features(features)
    probability = predict_tree(bundle, ordered)
    frozen = lgb.Booster(params=model.booster_.params.copy(),
                         model_str=model.booster_.model_to_string(num_iteration=int(best)))
    explainer = shap.TreeExplainer(frozen, model_output='raw', feature_perturbation='tree_path_dependent')
    values = explainer.shap_values(ordered, check_additivity=True)
    base = np.asarray(explainer.expected_value, dtype=float)
    if isinstance(values, list):  # Older binary convention: [class 0, class 1].
        if len(values) != 2 or base.shape != (2,):
            raise RuntimeError('Unsupported binary SHAP output shape')
        values, base = values[1], base[1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape == (len(ordered), len(ordered.columns), 2) and base.shape == (2,):
        values, base = values[:, :, 1], base[1]
    if values.shape != ordered.shape or base.size != 1 or not np.isfinite(values).all() or not np.isfinite(base).all():
        raise RuntimeError('SHAP returned invalid contributions or expected value')
    base_values = np.full(len(ordered), float(base.reshape(-1)[0]))
    raw = np.asarray(frozen.predict(ordered, raw_score=True, num_iteration=int(best)), dtype=float)
    if raw.shape != (len(ordered),) or not np.isfinite(raw).all():
        raise RuntimeError('LightGBM returned invalid raw log-odds')
    additive_error = float(np.max(np.abs(base_values + values.sum(axis=1) - raw)))
    probability_error = float(np.max(np.abs(expit(raw) - probability)))
    if additive_error > 1e-6 or probability_error > 1e-10:
        raise RuntimeError('SHAP additivity or raw log-odds probability mapping failed')
    root = frozen.dump_model()['tree_info'][0]['tree_structure']
    training_rows = int(root.get('internal_count', root.get('leaf_count', 0)))
    return {
        'base_log_odds': base_values.tolist(), 'shap_values': values.tolist(),
        'raw_log_odds': raw.tolist(), 'probability': probability.tolist(),
        'feature_names': list(ordered.columns), 'row_index_repr': [repr(x) for x in ordered.index],
        'metadata': {
            'explainer': 'shap.TreeExplainer', 'shap_version': shap.__version__,
            'lightgbm_version': lgb.__version__, 'model_output': 'raw', 'positive_class': 1,
            'feature_perturbation': 'tree_path_dependent', 'approximate': False,
            'background_source': 'training path/leaf sample counts stored in the fitted LightGBM trees',
            'background_partition': 'train', 'background_training_rows': training_rows,
            'explicit_background_rows': 0, 'best_iteration': int(best),
            'sample_selection': 'all supplied rows in input order; caller records partition and selection',
            'max_additivity_error': additive_error, 'additivity_atol': 1e-6,
            'max_probability_error': probability_error, 'probability_atol': 1e-10,
            'units': 'log-odds contributions; not probability percentage points or causal effects',
        },
    }
