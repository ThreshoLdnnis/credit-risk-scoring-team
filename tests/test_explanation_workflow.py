"""T06/T07 verifier regression and synthetic end-to-end provenance checks."""

import json

import numpy as np
import pandas as pd
import pytest

import run_pipeline
from experiments.morebadly import verify_and_explain as workflow
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import evaluate_binary


def test_complete_metric_schema_includes_string_decision_rule():
    metrics = evaluate_binary([0, 0, 1, 1], [.1, .4, .35, .8], .4)
    assert isinstance(metrics['decision_rule'], str)
    workflow._assert_metrics(metrics, metrics.copy())
    with pytest.raises(RuntimeError, match='definition'):
        workflow._assert_metrics(metrics, {**metrics, 'decision_rule': 'wrong'})
    with pytest.raises(RuntimeError, match='schema'):
        workflow._assert_metrics(metrics, {**metrics, 'unknown_metric': 1})
    with pytest.raises(AssertionError):
        workflow._assert_metrics(metrics, {**metrics, 'auc': .1})


def test_synthetic_pipeline_to_explanation_and_tamper_refusal(tmp_path, monkeypatch):
    rng = np.random.default_rng(62)
    raw = pd.DataFrame(rng.uniform(1, 9, (320, 10)), columns=FEATURE_COLUMNS)
    raw['age'] = rng.integers(20, 80, len(raw))
    raw.loc[::8, 'MonthlyIncome'] = np.nan
    raw[TARGET_COLUMN] = (raw.DebtRatio + rng.normal(0, 1, len(raw)) > 5).astype(int)
    raw.insert(0, 'Unnamed: 0', np.arange(1, len(raw) + 1))
    source = tmp_path / 'synthetic.csv'
    raw.to_csv(source, index=False)
    experiment = tmp_path / 'pipeline'
    config = json.loads((run_pipeline.ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))
    config['tree'].update(n_estimators=12, early_stopping_rounds=3, min_child_samples=10)
    run_pipeline.run_experiment(source, experiment, config, synthetic=True, models=['baseline', 'scorecard', 'tree'])
    # Override specified-data gates only inside this engineered test fixture.
    record_path = experiment / 'record.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record['status'] = 'completed_official_data_experiment'
    workflow.write_json(record_path, record)
    monkeypatch.setattr(workflow, 'EXPECTED_SOURCE', workflow.sha256(source))
    output = tmp_path / 'explanations'
    summary = workflow.verify(experiment, source, output, sample_size=10)
    assert summary['sample']['size'] == 10
    assert summary['local_figure']['all_contributions_accounted_for']
    assert summary['single_row_and_reordered_columns_equal_batch']
    assert all(a['all_metrics_reproduce'] for a in summary['models'].values())
    assert summary['explanation']['max_additivity_error'] < 1e-6
    for name, digest in summary['output_sha256'].items():
        assert workflow.sha256(output / name) == digest
    before = (output / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        workflow.verify(experiment, source, output, sample_size=10)
    assert (output / 'summary.json').read_bytes() == before
    with (experiment / 'tree_predictions.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='artifact hash'):
        workflow.verify(experiment, source, tmp_path / 'tampered', sample_size=10)
    assert not (tmp_path / 'tampered').exists()
