"""T03 shared entry: fixed grouped data split, train-only LR and local evidence.

Implements run_experiment(data_path, output_dir, settings) without changing
the data or model interfaces. Later tasks add scorecard/tree evaluation.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.baseline import fit_baseline, predict_baseline  # noqa: E402
from src.data import (  # noqa: E402
    TARGET_COLUMN,
    clean_features,
    feature_group_ids,
    load_dataset,
    split_dataset,
)

ROOT = Path(__file__).resolve().parent

# Same bytes as the official download recorded by Sally-yi in T02 EXP-001.
EXPECTED_SOURCE_SHA256 = '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git_value(*args):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT,
        encoding='utf-8', timeout=30,
    ).strip()


def validation_threshold(y, probability):
    """T03 experiment helper; T05 will own the shared evaluation module."""
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates, gains = thresholds[valid], (tpr - fpr)[valid]
    return float(candidates[gains == gains.max()].max())


def metrics(y, probability, threshold):
    fpr, tpr, _ = roc_curve(y, probability)
    precision_curve, recall_curve, _ = precision_recall_curve(y, probability)
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average='binary', zero_division=0,
    )
    return {
        'rows': len(y), 'positive': int(np.sum(y)), 'auc': float(roc_auc_score(y, probability)),
        'ks': float(np.max(tpr - fpr)), 'ap': float(average_precision_score(y, probability)),
        'pr_auc_trapezoidal': float(auc(recall_curve, precision_curve)),
        'brier': float(brier_score_loss(y, probability)), 'log_loss': float(log_loss(y, probability)),
        'threshold': threshold, 'decision_rule': 'probability >= threshold',
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
    }


def fingerprints(features, groups):
    """Match T02's ordered-column / float.hex / null fingerprint definition."""
    lookup = {}
    for pos in np.unique(groups, return_index=True)[1]:
        values = [None if pd.isna(v) else float(0 if v == 0 else v).hex() for v in features.iloc[pos]]
        payload = json.dumps([list(features.columns), values], separators=(',', ':'))
        lookup[int(groups[pos])] = hashlib.sha256(payload.encode()).hexdigest()
    if len(set(lookup.values())) != len(lookup):
        raise RuntimeError('Group fingerprint collision')
    return [lookup[int(group)] for group in groups]


def run_experiment(data_path, output_dir, settings: dict, *, synthetic=False):
    """Run the T03 LR route with a frozen settings dict and return its manifest.

    The existing three-argument public interface is preserved. Synthetic mode
    must be explicit and never counts as official-data model performance.
    Shared downstream model interfaces and configuration remain unchanged.
    """
    input_path, output_dir = Path(data_path).resolve(), Path(output_dir).resolve()
    if not isinstance(settings, dict):
        raise ValueError('settings must be a dict')
    config = json.loads(json.dumps(settings, allow_nan=False))
    if not input_path.is_file():
        raise FileNotFoundError('Download official cs-training.csv from Kaggle into data/raw/ first: ' + str(input_path))
    source_hash = sha256(input_path)
    if not synthetic and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError('CSV SHA256 differs from the official data version recorded in T02; verify the source before training')
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Local models and row-level evidence must stay under artifacts/')
    if type(config.get('seed')) is not int or config['seed'] != 42:
        raise ValueError('T03 must reuse the shared seed=42 partition')
    if config.get('split') != {'train': 0.7, 'validation': 0.15, 'test': 0.15, 'unit': 'cleaned_feature_group'}:
        raise ValueError('T03 requires the shared 70/15/15 cleaned-feature-group split')
    if config.get('threshold') != {
        'selection_data': 'validation', 'method': 'max_tpr_minus_fpr',
        'tie_break': 'largest_finite_threshold_in_0_1',
    }:
        raise ValueError('T03 requires the agreed validation-only threshold rule')
    acquisition = None
    source_metadata = input_path.with_suffix('.source.json')
    if source_metadata.is_file():
        acquisition = json.loads(source_metadata.read_text(encoding='utf-8'))
        if not isinstance(acquisition, dict) or acquisition.get('sha256') != source_hash:
            raise ValueError('Acquisition metadata does not match the input file SHA256')
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    if acquisition is not None:
        write_json(output_dir / 'acquisition.json', acquisition)
    code_paths = ['src/data.py', 'src/baseline.py', 'run_pipeline.py', 'experiments/morebadly/run_t03.py']
    source_code_hashes = {p: sha256(ROOT / p) for p in code_paths}
    write_json(output_dir / 'config.json', config)
    config_hash = sha256(output_dir / 'config.json')
    raw = load_dataset(input_path)
    features, y = clean_features(raw), raw[TARGET_COLUMN]
    positions = split_dataset(features, y, seed=config['seed'])
    groups = feature_group_ids(features)
    group_sets = {name: set(groups[pos]) for name, pos in positions.items()}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in [
        ('train', 'validation'), ('train', 'test'), ('validation', 'test'),
    ])
    if overlap or not np.array_equal(np.sort(np.concatenate(list(positions.values()))), np.arange(len(raw))):
        raise RuntimeError('Split group isolation or row coverage failed')
    repeated = split_dataset(features, y, seed=config['seed'])
    if not all(np.array_equal(positions[name], repeated[name]) for name in positions):
        raise RuntimeError('Split is not reproducible')
    bundle = fit_baseline(features.iloc[positions['train']], y.iloc[positions['train']], config)
    train_probability = predict_baseline(bundle, features.iloc[positions['train']])
    validation_probability = predict_baseline(bundle, features.iloc[positions['validation']])
    threshold = validation_threshold(y.iloc[positions['validation']], validation_probability)
    selection = {
        'fixed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'model': 'simple_l2_lr', 'settings': bundle['settings'],
        'threshold': threshold, 'source': 'validation', 'test_used_for_selection': False,
    }
    # Write the frozen decision before any test predictions or test metrics.
    write_json(output_dir / 'selection.json', selection)
    test_probability = predict_baseline(bundle, features.iloc[positions['test']])
    probabilities = {'train': train_probability, 'validation': validation_probability, 'test': test_probability}
    joblib.dump(bundle, output_dir / 'baseline.joblib')
    restored = joblib.load(output_dir / 'baseline.joblib')
    max_difference = 0.0
    for name, pos in positions.items():
        restored_probability = predict_baseline(restored, features.iloc[pos])
        difference = float(np.max(np.abs(restored_probability - probabilities[name])))
        max_difference = max(max_difference, difference)
        if not np.array_equal(restored_probability, probabilities[name]):
            raise RuntimeError('Saved/loaded model changed probabilities')
    result_metrics = {name: metrics(y.iloc[pos], probabilities[name], threshold) for name, pos in positions.items()}
    write_json(output_dir / 'metrics.json', result_metrics)
    partition = np.full(len(raw), '', dtype=object)
    prediction = np.empty(len(raw))
    split_summary = {}
    for name, pos in positions.items():
        partition[pos], prediction[pos] = name, probabilities[name]
        split_summary[name] = {
            'rows': len(pos), 'groups': len(group_sets[name]),
            'positive': int(y.iloc[pos].sum()), 'positive_rate': float(y.iloc[pos].mean()),
        }
    evidence = pd.DataFrame({
        'row_position': np.arange(len(raw)),
        'raw_row_id': raw['Unnamed: 0'].to_numpy() if 'Unnamed: 0' in raw else [None] * len(raw),
        'group_id': groups, 'group_fingerprint': fingerprints(features, groups), 'partition': partition,
    })
    evidence.to_csv(output_dir / 'split_members.csv', index=False)
    evidence.assign(y=y.to_numpy(), probability=prediction).to_csv(output_dir / 'predictions.csv', index=False)
    np.savez_compressed(output_dir / 'split_positions.npz', **positions)
    if sha256(input_path) != source_hash or sha256(output_dir / 'config.json') != config_hash or any(
        sha256(ROOT / p) != value for p, value in source_code_hashes.items()
    ):
        raise RuntimeError('Data, config or source code changed during the experiment')
    write_json(output_dir / 'run.log', {
        'started_at': started_at, 'completed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'synthetic_engineering_check' if synthetic else 'completed_official_data_experiment',
        'model': 'simple_l2_lr', 'rows': len(raw), 'group_overlap': overlap,
        'saved_loaded_max_probability_difference': max_difference,
    })
    imputer = bundle['pipeline'].named_steps['imputer']
    record = {
        'status': 'synthetic_engineering_check' if synthetic else 'completed_official_data_experiment',
        'executor': 'morebadly / Guo Yuzhe; Codex assisted execution; personal reproduction pending',
        'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source': {'file': input_path.name, 'sha256': source_hash, 'rows': len(raw),
                   'official_url': None if synthetic else 'https://www.kaggle.com/competitions/GiveMeSomeCredit/data',
                   'acquisition': acquisition},
        'target': {'column': TARGET_COLUMN, 'positive_class': 1, 'meaning': '90+ day serious delinquency within two years'},
        'feature_columns': list(features.columns), 'split': split_summary,
        'split_settings': {**config['split'], 'seed': config['seed'], 'stratification': 'group_max_y'},
        'audit': {'all_rows_once': True, 'group_overlap': overlap, 'seed_reproducible': True,
                  'saved_loaded_max_probability_difference': max_difference},
        'preprocessing': {
            'fit_partition': 'train', 'imputation': 'median; all-missing training columns use zero',
            'missing_indicators': [features.columns[i] for i in imputer.indicator_.features_],
            'median_values': dict(zip(features.columns, imputer.statistics_.tolist())),
            'scaler_mean': bundle['pipeline'].named_steps['scaler'].mean_.tolist(),
            'transformed_columns': imputer.get_feature_names_out().tolist(),
        },
        'model': {**selection, 'iterations': bundle['pipeline'].named_steps['classifier'].n_iter_.tolist(),
                  'parameters': bundle['pipeline'].named_steps['classifier'].get_params()},
        'metrics': result_metrics,
        'environment': {'python': platform.python_version(), 'packages': {
            p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']
        }},
        'git': {'commit': git_value('rev-parse', 'HEAD'), 'branch': git_value('branch', '--show-current'),
                'working_tree_changes': git_value('status', '--short')},
        'code_sha256': source_code_hashes, 'config_sha256': config_hash,
        'entry_point': 'run_pipeline.run_experiment',
        'command': None,
        'reproduction_command': ['python', 'run_pipeline.py', '--input', str(input_path),
                                 '--output', '<new-output-directory>', '--config',
                                 str(output_dir / 'config.json')] + (['--synthetic'] if synthetic else []),
        'output_sha256': {p.name: sha256(p) for p in output_dir.iterdir() if p.is_file()},
        'limitations': ['Internal research partitions, not out-of-time or customer-external validation.',
                        'Training metrics are apparent performance; no tuning used test metrics.',
                        'Scorecard, LightGBM, SHAP and UI are later tasks; this entry runs only the T03 LR route.'],
    }
    write_json(output_dir / 'record.json', record)
    return record


def run_from_config(data_path, output_dir, config_path=ROOT / 'configs/experiment.json', *,
                    synthetic=False, entry_script='run_pipeline.py'):
    """Read a config file once and record its bytes as well as the saved snapshot."""
    config_path = Path(config_path).resolve()
    source_bytes = config_path.read_bytes()
    original_hash = hashlib.sha256(source_bytes).hexdigest()
    settings = json.loads(source_bytes.decode('utf-8'))
    record = run_experiment(data_path, output_dir, settings, synthetic=synthetic)
    record['config_source'] = {'path': str(config_path), 'sha256': original_hash}
    if not config_path.is_file() or sha256(config_path) != original_hash:
        record['status'] = 'failed_config_provenance'
        record['error'] = 'Source config changed during the experiment'
        write_json(Path(output_dir) / 'run.log', {'status': record['status'], 'error': record['error']})
        record['output_sha256']['run.log'] = sha256(Path(output_dir) / 'run.log')
        write_json(Path(output_dir) / 'record.json', record)
        raise RuntimeError(record['error'])
    record['command'] = ['python', entry_script, '--input', str(Path(data_path).resolve()),
                         '--output', str(Path(output_dir).resolve()), '--config', str(config_path)]
    if synthetic:
        record['command'].append('--synthetic')
    write_json(Path(output_dir) / 'record.json', record)
    return record


def main(entry_script='run_pipeline.py'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.json')
    parser.add_argument('--synthetic', action='store_true', help='Engineering check only; never counts as official data results')
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/morebadly' / (
        'T03-' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8]
    )
    try:
        record = run_from_config(args.input, output, args.config, synthetic=args.synthetic, entry_script=entry_script)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f'T03 failed: {error}\n')
    print(json.dumps({'output': str(output.resolve()), 'status': record['status'], 'metrics': record['metrics']}, indent=2))


if __name__ == '__main__':
    main()
