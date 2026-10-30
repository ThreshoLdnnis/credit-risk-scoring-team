"""Explanation integrity, complete local accounting and model/run association."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.data import CLEAN_FEATURE_COLUMNS
from src.ui_explanations import (
    file_sha256,
    load_global_explanation,
    local_contributions,
    validate_explanation,
)


def explanation_case():
    features = pd.DataFrame([np.ones(13)], columns=CLEAN_FEATURE_COLUMNS, index=[42])
    values = np.arange(13, dtype=float) / 100
    raw = -3 + values.sum()
    result = {'base_log_odds': [-3.], 'shap_values': [values.tolist()], 'raw_log_odds': [raw],
              'probability': [float(expit(raw))], 'feature_names': list(features.columns),
              'row_index_repr': ['42'], 'metadata': {'model_output': 'raw', 'positive_class': 1}}
    return features, result


def test_valid_explanation_matches_displayed_probability_and_keeps_all_terms():
    features, result = explanation_case()
    validate_explanation(result, features, result['probability'])
    rows = local_contributions(result, {})
    assert len(rows) == 9
    assert '其余 5 项' in rows[-1]['因素']
    assert -3 + sum(row['贡献'] for row in rows) == pytest.approx(result['raw_log_odds'][0])


@pytest.mark.parametrize('problem', ['features', 'rows', 'class', 'raw', 'probability', 'nan', 'shape', 'displayed'])
def test_inconsistent_explanations_are_rejected(problem):
    features, result = explanation_case()
    expected = result['probability'].copy()
    if problem == 'features':
        result['feature_names'].reverse()
    elif problem == 'rows':
        result['row_index_repr'] = ['43']
    elif problem == 'class':
        result['metadata']['positive_class'] = 0
    elif problem == 'raw':
        result['raw_log_odds'][0] += 1
    elif problem == 'probability':
        result['probability'][0] += .1
    elif problem == 'nan':
        result['shap_values'][0][0] = np.nan
    elif problem == 'shape':
        result['base_log_odds'] = []
    else:
        expected = [.99]
    with pytest.raises(ValueError):
        validate_explanation(result, features, expected)


def global_run(tmp_path):
    run = tmp_path / 'model-run'
    explain = run / 'explanations'
    explain.mkdir(parents=True)
    (run / 'tree.joblib').write_bytes(b'checksum fixture, never deserialized')
    record = {'models': {'tree': {'best_iteration': 3}},
              'output_sha256': {'tree.joblib': file_sha256(run / 'tree.joblib')}}
    (run / 'record.json').write_text(json.dumps(record), encoding='utf8')
    table = pd.DataFrame({'feature': CLEAN_FEATURE_COLUMNS, 'mean_abs_shap_log_odds': np.arange(13) / 10})
    table.to_csv(explain / 'global_importance.csv', index=False)
    summary = {'status': 'completed_fixed_model_explanation_and_reproduction_audit',
               'pipeline_record_sha256': file_sha256(run / 'record.json'),
               'output_sha256': {'global_importance.csv': file_sha256(explain / 'global_importance.csv')},
               'explanation': {'best_iteration': 3}}
    (explain / 'summary.json').write_text(json.dumps(summary), encoding='utf8')
    return run, summary


def test_global_importance_is_associated_with_the_actual_model_and_record(tmp_path):
    run, _ = global_run(tmp_path)
    assert len(load_global_explanation(run)['importance']) == 13
    assert load_global_explanation(tmp_path / 'not-generated') is None


@pytest.mark.parametrize('problem', ['record', 'tree', 'table', 'rounds', 'features'])
def test_mismatched_global_artifacts_are_rejected(tmp_path, problem):
    run, summary = global_run(tmp_path)
    summary = copy.deepcopy(summary)
    if problem == 'record':
        summary['pipeline_record_sha256'] = 'another experiment'
    elif problem == 'tree':
        (run / 'tree.joblib').write_bytes(b'another model')
    elif problem == 'table':
        (run / 'explanations/global_importance.csv').write_text('altered', encoding='utf8')
    elif problem == 'rounds':
        summary['explanation']['best_iteration'] = 4
    else:
        path = run / 'explanations/global_importance.csv'
        table = pd.read_csv(path)
        table.loc[0, 'feature'] = 'unexpected'
        table.to_csv(path, index=False)
        summary['output_sha256']['global_importance.csv'] = file_sha256(path)
    (run / 'explanations/summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError):
        load_global_explanation(run)
