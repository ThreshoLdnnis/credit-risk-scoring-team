"""T06/T07: audit a fixed shared experiment and explain a label-blind sample.

Never fits, tunes, overwrites source artifacts or publishes row-level records.
Run after run_pipeline.py --models baseline scorecard tree in a fresh directory.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import predict_baseline  # noqa: E402
from src.data import (  # noqa: E402
    TARGET_COLUMN,
    clean_features,
    feature_group_ids,
    load_dataset,
    split_dataset,
)
from src.evaluation import evaluate_binary, select_threshold  # noqa: E402
from src.explanation import explain_tree  # noqa: E402
from src.scorecard import predict_scorecard  # noqa: E402
from src.tree_model import predict_tree  # noqa: E402

EXPECTED_SOURCE = '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac'
DISPLAY_NAMES = {
    'RevolvingUtilizationOfUnsecuredLines': 'Revolving credit utilization', 'age': 'Age',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59 day delinquencies', 'DebtRatio': 'Debt ratio',
    'MonthlyIncome': 'Monthly income', 'NumberOfOpenCreditLinesAndLoans': 'Open credit lines and loans',
    'NumberOfTimes90DaysLate': '90+ day delinquencies',
    'NumberRealEstateLoansOrLines': 'Real estate loans or lines',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89 day delinquencies', 'NumberOfDependents': 'Dependents',
}


def display_name(name):
    original, marker, _ = name.partition('__special')
    return DISPLAY_NAMES.get(original, original) + (' (special indicator)' if marker else '')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def _assert_metrics(calculated, expected):
    if set(calculated) != set(expected):
        raise RuntimeError('Metric schema differs from the saved experiment')
    for key, value in calculated.items():
        if isinstance(value, (int, float, np.number)):
            np.testing.assert_allclose(value, expected[key], rtol=0, atol=1e-12)
        elif value != expected[key]:
            raise RuntimeError('Metric definition differs: ' + key)


def save_figures(result, output):
    names = result['feature_names']
    values = np.asarray(result['shap_values'])
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance, kind='stable')
    figure = Figure(figsize=(11, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.barh(np.arange(len(names)), importance[order], color='#2874a6')
    axes.set_yticks(np.arange(len(names)), [display_name(names[i]) for i in order])
    axes.set_xlabel('Mean absolute SHAP contribution (log-odds)')
    axes.set_title(f'LightGBM Tree SHAP | test sample n={len(values):,} | rounds={result["metadata"]["best_iteration"]}')
    figure.savefig(output / 'global_shap.png', dpi=140)
    pd.DataFrame({'feature': names, 'mean_abs_shap_log_odds': importance}).sort_values(
        'mean_abs_shap_log_odds', ascending=False, kind='stable',
    ).to_csv(output / 'global_importance.csv', index=False, lineterminator='\n')
    # Local figure is a row-level prediction: keep it only under artifacts/.
    chosen = np.argsort(-np.abs(values[0]), kind='stable')[:8]
    omitted = np.setdiff1d(np.arange(len(names)), chosen)
    terms = [*values[0, chosen], float(values[0, omitted].sum())]
    labels = [*[display_name(names[i]) for i in chosen], f'Other {len(omitted)} features (sum)']
    base = result['base_log_odds'][0]
    starts = np.r_[base, base + np.cumsum(terms[:-1])]
    if not np.isclose(base + sum(terms), result['raw_log_odds'][0], rtol=0, atol=1e-6):
        raise RuntimeError('Local figure omitted-contribution accounting failed')
    figure = Figure(figsize=(11, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.barh(np.arange(len(terms)), terms, left=starts,
              color=['#c0392b' if value >= 0 else '#2874a6' for value in terms])
    axes.set_yticks(np.arange(len(terms)), labels)
    axes.invert_yaxis()
    endpoints = np.r_[base, starts, starts + terms, result['raw_log_odds'][0]]
    margin = max(float(np.ptp(endpoints)) * .08, .05)
    axes.set_xlim(float(endpoints.min()) - margin, float(endpoints.max()) + margin)
    axes.axvline(base, color='#777777', ls='--', label=f'base={base:.4f}')
    axes.axvline(result['raw_log_odds'][0], color='#222222', label=f'raw={result["raw_log_odds"][0]:.4f}')
    axes.set_xlabel('Raw log-odds; positive contribution increases y=1 model output')
    axes.set_title(f'LightGBM local explanation | test sample 1/{len(values):,}\np={result["probability"][0]:.6f}; contributions are not probability points')
    axes.legend(loc='lower right')
    figure.savefig(output / 'local_shap.png', dpi=140)
    return {'displayed_features': len(chosen), 'omitted_features': len(omitted),
            'omitted_log_odds': float(values[0, omitted].sum()), 'all_contributions_accounted_for': True}


def verify(experiment, input_path, output, sample_size=1000, reference=None):
    experiment, input_path, output = [Path(p).resolve() for p in (experiment, input_path, output)]
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Row-level SHAP and local predictions must remain under artifacts/')
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError('sample_size must be a positive integer')
    record_path = experiment / 'record.json'
    record_hash = sha256(record_path)
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record.get('status') != 'completed_official_data_experiment' or record.get('selected_models') != ['baseline', 'scorecard', 'tree']:
        raise ValueError('Require a completed, specified-data, three-model shared experiment')
    if sha256(input_path) != EXPECTED_SOURCE or record['source']['sha256'] != EXPECTED_SOURCE:
        raise ValueError('Specified dataset SHA256 failed')
    source_hashes = {name: sha256(ROOT / name) for name in [
        'src/data.py', 'src/baseline.py', 'src/scorecard.py', 'src/tree_model.py',
        'src/evaluation.py', 'src/explanation.py', 'run_pipeline.py', 'experiments/morebadly/verify_and_explain.py',
    ]}
    for name, digest in record['code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Pipeline source differs from saved experiment: ' + name)
    for name, digest in record['output_sha256'].items():
        if sha256(experiment / name) != digest:
            raise ValueError('Pipeline artifact hash failed: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    raw = load_dataset(input_path)
    x, y = clean_features(raw), raw[TARGET_COLUMN]
    positions = split_dataset(x, y, seed=42)
    stored_positions = np.load(experiment / 'split_positions.npz')
    if not all(np.array_equal(pos, stored_positions[part]) for part, pos in positions.items()):
        raise RuntimeError('Shared partition positions do not reproduce')
    groups = feature_group_ids(x)
    sets = {part: set(groups[pos]) for part, pos in positions.items()}
    if any(sets[a] & sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise RuntimeError('Content group leakage')
    partitions = np.full(len(x), '', dtype=object)
    for part, pos in positions.items():
        partitions[pos] = part
    model_audits, bundles = {}, {}
    for name, predictor in [('baseline', predict_baseline), ('scorecard', predict_scorecard), ('tree', predict_tree)]:
        bundle = joblib.load(experiment / (name + '.joblib'))
        bundles[name] = bundle
        frame = pd.read_csv(experiment / (name + '_predictions.csv'), float_precision='round_trip')
        if len(frame) != len(x) or not np.array_equal(frame.row_position, np.arange(len(x))) or not np.array_equal(
            frame.raw_row_id, raw['Unnamed: 0'],
        ) or not np.array_equal(frame.y, y) or not np.array_equal(frame.partition, partitions):
            raise RuntimeError(name + ' row, label or partition alignment failed')
        predicted = predictor(bundle, x)
        if name == 'scorecard':
            for column in predicted:
                np.testing.assert_allclose(predicted[column], frame[column], rtol=0, atol=1e-10)
            probability = predicted.probability.to_numpy()
        else:
            probability = predicted
        np.testing.assert_allclose(probability, frame.probability, rtol=0, atol=1e-14)
        selected = record['models'][name]['threshold']
        np.testing.assert_allclose(select_threshold(y.iloc[positions['validation']], probability[positions['validation']]),
                                   selected, rtol=0, atol=1e-14)
        for part, pos in positions.items():
            calculated = evaluate_binary(y.iloc[pos], probability[pos], selected)
            _assert_metrics(calculated, record['models'][name]['metrics'][part])
        model_audits[name] = {'rows_labels_partition_aligned': True, 'saved_model_predictions_reproduce': True,
                              'validation_threshold_reproduces_without_changes': True, 'all_metrics_reproduce': True}
        if reference:
            prior = pd.read_csv(Path(reference) / (name + '_predictions.csv'), float_precision='round_trip')
            if not np.array_equal(prior.row_position, frame.row_position) or not np.array_equal(prior.partition, frame.partition):
                raise RuntimeError('Reference prediction rows or partitions differ')
            model_audits[name]['reference_max_probability_difference'] = float(np.max(np.abs(frame.probability - prior.probability)))
    sample = np.sort(np.random.default_rng(42).choice(positions['test'], min(sample_size, len(positions['test'])), replace=False))
    result = explain_tree(bundles['tree'], x.iloc[sample])
    single = explain_tree(bundles['tree'], x.iloc[sample[:1]].loc[:, list(reversed(x.columns))])
    np.testing.assert_array_equal(single['shap_values'][0], result['shap_values'][0])
    np.testing.assert_array_equal(single['probability'][0], result['probability'][0])
    write_json(output / 'row_explanations.json', result)
    np.savez_compressed(output / 'sample_positions.npz', positions=sample)
    local_figure = save_figures(result, output)
    if sha256(input_path) != EXPECTED_SOURCE or sha256(record_path) != record_hash or any(
        sha256(ROOT / name) != digest for name, digest in source_hashes.items()
    ) or any(sha256(experiment / name) != digest for name, digest in record['output_sha256'].items()):
        raise RuntimeError('Source data, code or original experiment changed during verification')
    summary = {
        'status': 'completed_fixed_model_explanation_and_reproduction_audit',
        'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source_sha256': EXPECTED_SOURCE, 'pipeline_record_sha256': record_hash,
        'code_sha256': source_hashes, 'models': model_audits,
        'split': record['split'], 'partition_positions_reproduce': True, 'group_overlap': 0,
        'sample': {'partition': 'test', 'size': len(sample), 'population': len(positions['test']), 'seed': 42,
                   'method': 'uniform without replacement, sorted by original row position; no label or probability selection',
                   'positions_sha256': hashlib.sha256(sample.astype('<i8').tobytes()).hexdigest()},
        'explanation': result['metadata'], 'single_row_and_reordered_columns_equal_batch': True,
        'local_figure': local_figure,
        'global_importance': pd.read_csv(output / 'global_importance.csv').to_dict('records'),
        'environment': {'python': platform.python_version(), 'packages': {p: version(p) for p in [
            'numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'shap', 'joblib', 'matplotlib',
        ]}},
        'output_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        'limitations': ['SHAP describes this fixed model in log-odds, not causal effects.',
                        'Path-dependent training counts define the background; no explicit interventional background.',
                        'Aggregate importance describes only this 1000-row (or requested-size) internal test sample.',
                        'Public development data; no new independent or out-of-time validation.',
                        'Human presentation, teammate UI integration and final team acceptance remain separate.'],
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps({'output': str(output), 'sample': summary['sample'], 'checks': result['metadata'],
                      'models': model_audits}, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    verify(args.experiment, args.input, args.output, args.sample_size, args.reference)


if __name__ == '__main__':
    main()
