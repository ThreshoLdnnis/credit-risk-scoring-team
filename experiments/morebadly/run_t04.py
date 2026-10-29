"""Reproduce T04 on the specified Give Me Some Credit file; keep row data local."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    TARGET_COLUMN,
    clean_features,
    feature_group_ids,
    load_dataset,
    split_dataset,
)
from src.scorecard import fit_scorecard, predict_scorecard  # noqa: E402

EXPECTED_SHA256 = '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def git_value(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args],
                                   cwd=ROOT, encoding='utf-8', timeout=30).strip()


def fingerprints(features, groups):
    lookup = {}
    for pos in np.unique(groups, return_index=True)[1]:
        values = [None if pd.isna(v) else float(0 if v == 0 else v).hex() for v in features.iloc[pos]]
        payload = json.dumps([list(features.columns), values], separators=(',', ':'))
        lookup[int(groups[pos])] = hashlib.sha256(payload.encode()).hexdigest()
    if len(set(lookup.values())) != len(lookup):
        raise RuntimeError('Group fingerprint collision')
    return [lookup[int(group)] for group in groups]


def evaluate(y, p, threshold):
    fpr, tpr, _ = roc_curve(y, p)
    curve_precision, curve_recall, _ = precision_recall_curve(y, p)
    actual, predicted = np.asarray(y), p >= threshold
    tn = int(np.sum((actual == 0) & ~predicted))
    fp = int(np.sum((actual == 0) & predicted))
    fn = int(np.sum((actual == 1) & ~predicted))
    tp = int(np.sum((actual == 1) & predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'rows': len(y), 'positive': int(np.sum(y)), 'auc': float(roc_auc_score(y, p)),
            'ks': float(np.max(tpr - fpr)), 'ap': float(average_precision_score(y, p)),
            'pr_auc_trapezoidal': float(auc(curve_recall, curve_precision)), 'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p)), 'threshold': threshold, 'rule': 'p >= threshold',
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision': precision, 'recall': recall, 'f1': f1}


def run(input_path, output_dir, config_path=ROOT / 'configs/experiment.json'):
    input_path, output_dir, config_path = map(lambda p: Path(p).resolve(), (input_path, output_dir, config_path))
    source_hash = sha256(input_path)
    if source_hash != EXPECTED_SHA256:
        raise ValueError('Dataset failed: SHA256 does not match the specified T02 official-file version')
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Row-level evidence and models must stay in ignored artifacts/')
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode('utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Configuration must be an object')
    if type(config.get('seed')) is not int or config['seed'] != 42 or config.get('split') != {
        'train': 0.7, 'validation': 0.15, 'test': 0.15, 'unit': 'cleaned_feature_group',
    }:
        raise ValueError('The shared seed=42 and 70/15/15 group protocol must be preserved')
    if config.get('threshold') != {'selection_data': 'validation', 'method': 'max_tpr_minus_fpr',
                                    'tie_break': 'largest_finite_threshold_in_0_1'}:
        raise ValueError('The agreed validation-only threshold protocol must be preserved')
    code_paths = ['src/data.py', 'src/scorecard.py', 'experiments/morebadly/run_t04.py']
    code_hashes = {p: sha256(ROOT / p) for p in code_paths}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'config.json', config)
    acquisition_path = input_path.with_suffix('.source.json')
    acquisition = json.loads(acquisition_path.read_text(encoding='utf-8')) if acquisition_path.is_file() else None
    if acquisition is not None:
        if acquisition.get('sha256') != source_hash:
            raise ValueError('Dataset acquisition metadata does not match the input hash')
        write_json(output_dir / 'acquisition.json', acquisition)
    print('Dataset hash passed; loading and auditing the fixed split.', flush=True)
    raw = load_dataset(input_path)
    features, y = clean_features(raw), raw[TARGET_COLUMN]
    positions = split_dataset(features, y, seed=42)
    repeated = split_dataset(features, y, seed=42)
    groups = feature_group_ids(features)
    group_sets = {name: set(groups[pos]) for name, pos in positions.items()}
    if not np.array_equal(np.sort(np.concatenate(list(positions.values()))), np.arange(len(raw))):
        raise RuntimeError('Split does not cover all original rows once')
    if any(group_sets[a] & group_sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise RuntimeError('Content group leakage across partitions')
    if not all(np.array_equal(positions[k], repeated[k]) for k in positions):
        raise RuntimeError('Split is not reproducible')
    print('Training native OptBinning and LR on train rows only.', flush=True)
    bundle = fit_scorecard(features.iloc[positions['train']], y.iloc[positions['train']], config['scorecard'])
    validation = predict_scorecard(bundle, features.iloc[positions['validation']])
    fpr, tpr, thresholds = roc_curve(y.iloc[positions['validation']], validation.probability, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates, gains = thresholds[valid], (tpr - fpr)[valid]
    threshold = float(candidates[gains == gains.max()].max())
    selection = {'threshold': threshold, 'source': 'validation', 'test_used_for_selection': False,
                 'fixed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'model_settings': bundle['settings'], 'selected_features': bundle['selected_features']}
    write_json(output_dir / 'selection.json', selection)
    predictions = {'validation': validation}
    predictions['train'] = predict_scorecard(bundle, features.iloc[positions['train']])
    predictions['test'] = predict_scorecard(bundle, features.iloc[positions['test']])
    joblib.dump(bundle, output_dir / 'scorecard.joblib')
    restored = joblib.load(output_dir / 'scorecard.joblib')
    max_points_error, max_probability_link_error = 0.0, 0.0
    summary, metrics, frames = {}, {}, []
    for name, pos in positions.items():
        result = predictions[name]
        pd.testing.assert_frame_equal(result, predict_scorecard(restored, features.iloc[pos]), check_exact=True)
        point_columns = [c for c in result if c.startswith('points_')]
        points_error = float(np.max(np.abs(result.score - result.base_points - result[point_columns].sum(axis=1))))
        scale = bundle['score_scale']
        probability_link_error = float(np.max(np.abs(expit((scale['offset'] - result.score) / scale['factor']) - result.probability)))
        max_points_error = max(max_points_error, points_error)
        max_probability_link_error = max(max_probability_link_error, probability_link_error)
        summary[name] = {'rows': len(pos), 'groups': len(group_sets[name]), 'positive': int(y.iloc[pos].sum()),
                         'positive_rate': float(y.iloc[pos].mean()), 'probability_min': float(result.probability.min()),
                         'probability_max': float(result.probability.max()), 'score_min': float(result.score.min()),
                         'score_max': float(result.score.max())}
        metrics[name] = evaluate(y.iloc[pos], result.probability.to_numpy(), threshold)
        frames.append(result.assign(row_position=pos, partition=name, y=y.iloc[pos].to_numpy()))
    if max_points_error > 1e-9 or max_probability_link_error > 1e-12:
        raise RuntimeError('Score decomposition or probability-to-score link audit failed')
    factor, offset = bundle['score_scale']['factor'], bundle['score_scale']['offset']
    anchor = offset + factor * np.log(20)
    doubled = offset + factor * np.log(40)
    if abs(anchor - 600) > 1e-10 or abs(doubled - anchor - 20) > 1e-10:
        raise RuntimeError('Base score / good-bad odds / PDO audit failed')
    partition = np.full(len(raw), '', dtype=object)
    for name, pos in positions.items():
        partition[pos] = name
    pd.DataFrame({'row_position': np.arange(len(raw)),
                  'raw_row_id': raw['Unnamed: 0'].to_numpy() if 'Unnamed: 0' in raw else [None] * len(raw),
                  'group_id': groups, 'group_fingerprint': fingerprints(features, groups),
                  'partition': partition}).to_csv(output_dir / 'split_members.csv', index=False)
    pd.concat(frames).sort_values('row_position').to_csv(output_dir / 'predictions.csv', index=False)
    np.savez_compressed(output_dir / 'split_positions.npz', **positions)
    write_json(output_dir / 'binning_evidence.json', bundle['binning_evidence'])
    write_json(output_dir / 'iv_summary.json', bundle['iv_summary'])
    write_json(output_dir / 'metrics.json', metrics)
    if sha256(input_path) != source_hash or config_path.read_bytes() != config_bytes or any(
        sha256(ROOT / p) != h for p, h in code_hashes.items()
    ):
        raise RuntimeError('Dataset, config or source changed during the experiment')
    audit = {'all_rows_once': True, 'group_overlap': 0, 'seed_reproducible': True,
             'saved_loaded_predictions_identical': True, 'max_points_error': max_points_error,
             'max_probability_score_link_error': max_probability_link_error,
             'score_at_good_bad_odds_20': float(anchor), 'pdo_on_doubling_good_bad_odds': float(doubled - anchor)}
    record = {'status': 'completed_specified_data_experiment',
              'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'scope': 'T04 supplement to Clement-xi PR12; morebadly validation and boundary fixes',
              'source': {'file': input_path.name, 'sha256': source_hash, 'rows': len(raw), 'acquisition': acquisition},
              'target': {'column': TARGET_COLUMN, 'positive': '90+ day delinquency within two years'},
              'feature_columns': list(features.columns), 'split': summary,
              'scorecard_settings': bundle['settings'], 'selection': selection,
              'model_parameters': bundle['logistic_regression'].get_params(),
              'iterations': bundle['logistic_regression'].n_iter_.tolist(),
              'iv_summary': bundle['iv_summary'], 'score_scale': bundle['score_scale'],
              'binning_status': {k: v['status'] for k, v in bundle['binning_evidence'].items()},
              'audit': audit, 'metrics': metrics,
              'git': {'commit': git_value('rev-parse', 'HEAD'), 'branch': git_value('branch', '--show-current'),
                      'working_tree_changes': git_value('status', '--short')},
              'code_sha256': code_hashes, 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
              'environment': {'python': platform.python_version(), 'packages': {
                  k: version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'optbinning', 'ortools', 'joblib']}},
              'command': ['python', 'experiments/morebadly/run_t04.py', '--input', str(input_path),
                          '--output', str(output_dir), '--config', str(config_path)],
              'output_sha256': {p.name: sha256(p) for p in output_dir.iterdir() if p.is_file()},
              'limitations': ['Internal research split; no out-of-time or customer-external validation.',
                              'Independent T04 module verification; shared pipeline integration is coordinated separately in T03/T07.',
                              'AI assistance is not equivalent to personal mastery or team peer approval.']}
    write_json(output_dir / 'record.json', record)
    print(json.dumps({'output': str(output_dir), 'audit': audit, 'metrics': metrics,
                      'selected_features': bundle['selected_features']}, ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.json')
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/morebadly' / ('T04-' + uuid4().hex[:12])
    try:
        run(args.input, output, args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f'T04 experiment failed: {error}\n')


if __name__ == '__main__':
    main()
