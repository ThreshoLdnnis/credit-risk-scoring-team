"""T05: fixed-split LR/scorecard/LightGBM comparison with validation-only choices."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import fit_baseline, predict_baseline  # noqa: E402
from src.data import (  # noqa: E402
    TARGET_COLUMN,
    clean_features,
    feature_group_ids,
    load_dataset,
    split_dataset,
)
from src.evaluation import (  # noqa: E402
    evaluate_binary,
    save_evaluation_figures,
    select_threshold,
)
from src.scorecard import fit_scorecard, predict_scorecard  # noqa: E402
from src.tree_model import fit_tree, predict_tree  # noqa: E402

EXPECTED_SHA256 = '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
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


def predict(name, bundle, features):
    if name == 'scorecard':
        result = predict_scorecard(bundle, features)
    else:
        function = predict_baseline if name == 'baseline' else predict_tree
        result = pd.DataFrame({'probability': function(bundle, features)}, index=features.index)
    if not result.index.equals(features.index) or not result.probability.between(0, 1).all():
        raise RuntimeError(f'{name} prediction alignment or probability validation failed')
    return result


def run(input_path, output_dir, config_path=ROOT / 'configs/experiment.json'):
    input_path, output_dir, config_path = map(lambda p: Path(p).resolve(), (input_path, output_dir, config_path))
    source_hash = sha256(input_path)
    if source_hash != EXPECTED_SHA256:
        raise ValueError('Dataset failed SHA256 verification against the specified T02 official-file version')
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Models and row-level evidence must remain under artifacts/')
    source_config = config_path.read_bytes()
    config = json.loads(source_config.decode('utf-8'))
    if not isinstance(config, dict) or type(config.get('seed')) is not int or config['seed'] != 42 or config.get('split') != {
        'train': .7, 'validation': .15, 'test': .15, 'unit': 'cleaned_feature_group',
    }:
        raise ValueError('Preserve the shared seed42 and 70/15/15 content-group protocol')
    if config.get('threshold') != {'selection_data': 'validation', 'method': 'max_tpr_minus_fpr',
                                    'tie_break': 'largest_finite_threshold_in_0_1'}:
        raise ValueError('Preserve the agreed validation-only threshold protocol')
    code_paths = ['src/data.py', 'src/baseline.py', 'src/scorecard.py', 'src/tree_model.py',
                  'src/evaluation.py', 'experiments/morebadly/run_t05.py']
    code_hashes = {name: sha256(ROOT / name) for name in code_paths}
    acquisition_path = input_path.with_suffix('.source.json')
    acquisition = json.loads(acquisition_path.read_text(encoding='utf-8')) if acquisition_path.is_file() else None
    if acquisition is not None and acquisition.get('sha256') != source_hash:
        raise ValueError('Source acquisition metadata does not match the dataset')
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    write_json(output_dir / 'config.json', config)
    if acquisition:
        write_json(output_dir / 'acquisition.json', acquisition)
    print('Specified data verified; load, clean and audit identical partitions.', flush=True)
    raw = load_dataset(input_path)
    features, y = clean_features(raw), raw[TARGET_COLUMN]
    positions = split_dataset(features, y, seed=42)
    groups = feature_group_ids(features)
    repeated = split_dataset(features, y, seed=42)
    if not all(np.array_equal(positions[k], repeated[k]) for k in positions):
        raise RuntimeError('Group split is not reproducible')
    if not np.array_equal(np.sort(np.concatenate(list(positions.values()))), np.arange(len(raw))):
        raise RuntimeError('Split row coverage failed')
    group_sets = {name: set(groups[pos]) for name, pos in positions.items()}
    if any(group_sets[a] & group_sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise RuntimeError('Content groups cross partitions')
    train_pos, validation_pos = positions['train'], positions['validation']
    train_x, train_y = features.iloc[train_pos], y.iloc[train_pos]
    validation_x, validation_y = features.iloc[validation_pos], y.iloc[validation_pos]
    bundles, frames, selections, details = {}, {}, {}, {}
    for name in ['baseline', 'scorecard', 'tree']:
        print('Train and select validation threshold: ' + name, flush=True)
        if name == 'baseline':
            bundle = fit_baseline(train_x, train_y, config)
            details[name] = {'settings': bundle['settings'],
                             'iterations': bundle['pipeline'].named_steps['classifier'].n_iter_.tolist()}
        elif name == 'scorecard':
            bundle = fit_scorecard(train_x, train_y, config['scorecard'])
            details[name] = {'settings': bundle['settings'], 'selected_features': bundle['selected_features'],
                             'iv_summary': bundle['iv_summary'], 'score_scale': bundle['score_scale'],
                             'iterations': bundle['logistic_regression'].n_iter_.tolist()}
            write_json(output_dir / 'scorecard_binning.json', bundle['binning_evidence'])
        else:
            bundle = fit_tree(train_x, train_y, validation_x, validation_y, config)
            details[name] = {k: v for k, v in bundle.items() if k != 'model'}
        bundles[name] = bundle
        frames[name] = {'train': predict(name, bundle, train_x),
                        'validation': predict(name, bundle, validation_x)}
        selections[name] = {'threshold': select_threshold(validation_y, frames[name]['validation'].probability),
                            'source': 'validation', 'test_used_for_selection': False,
                            'fixed_at': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    # Every model and threshold is frozen before the first test prediction.
    write_json(output_dir / 'selections.json', selections)
    print('All validation decisions frozen; evaluate test once with fixed models.', flush=True)
    metrics = {}
    roundtrip = {}
    for name, bundle in bundles.items():
        frames[name]['test'] = predict(name, bundle, features.iloc[positions['test']])
        metrics[name] = {part: evaluate_binary(y.iloc[pos], frames[name][part].probability, selections[name]['threshold'])
                         for part, pos in positions.items()}
        joblib.dump(bundle, output_dir / (name + '.joblib'))
        restored = joblib.load(output_dir / (name + '.joblib'))
        for part, pos in positions.items():
            pd.testing.assert_frame_equal(predict(name, restored, features.iloc[pos]), frames[name][part], check_exact=True)
        roundtrip[name] = 0.0
        parts = [frames[name][part].assign(row_position=pos, y=y.iloc[pos].to_numpy(), partition=part)
                 for part, pos in positions.items()]
        pd.concat(parts).sort_values('row_position').to_csv(output_dir / (name + '_predictions.csv'), index=False)
    write_json(output_dir / 'metrics.json', metrics)
    comparison = pd.DataFrame([{'model': name, 'partition': part, **values}
                              for name, by_partition in metrics.items() for part, values in by_partition.items()])
    comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
    figure_paths = {}
    for part in ['validation', 'test']:
        figure_paths[part] = save_evaluation_figures(
            y.iloc[positions[part]], {name: frames[name][part] for name in bundles},
            output_dir / 'figures' / part, partition=part,
        )
    partition = np.full(len(raw), '', dtype=object)
    split_summary = {}
    for part, pos in positions.items():
        partition[pos] = part
        split_summary[part] = {'rows': len(pos), 'groups': len(group_sets[part]),
                               'positive': int(y.iloc[pos].sum()), 'positive_rate': float(y.iloc[pos].mean())}
    pd.DataFrame({'row_position': np.arange(len(raw)),
                  'raw_row_id': raw['Unnamed: 0'].to_numpy() if 'Unnamed: 0' in raw else [None] * len(raw),
                  'group_id': groups, 'group_fingerprint': fingerprints(features, groups),
                  'partition': partition}).to_csv(output_dir / 'split_members.csv', index=False)
    np.savez_compressed(output_dir / 'split_positions.npz', **positions)
    if sha256(input_path) != source_hash or config_path.read_bytes() != source_config or any(
        sha256(ROOT / name) != h for name, h in code_hashes.items()
    ):
        raise RuntimeError('Data, source configuration or code changed during the experiment')
    audit = {'all_rows_once': True, 'group_overlap': 0, 'seed_reproducible': True,
             'all_models_use_identical_partitions': True, 'all_selections_frozen_before_test': True,
             'saved_loaded_max_output_difference': roundtrip}
    write_json(output_dir / 'run.log', {'started_at': started_at, 'completed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                                        'status': 'completed', 'audit': audit})
    record = {'status': 'completed_specified_data_experiment',
              'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'source': {'file': input_path.name, 'sha256': source_hash, 'rows': len(raw), 'acquisition': acquisition},
              'feature_columns': list(features.columns), 'split': split_summary, 'model_details': details,
              'selections': selections, 'metrics': metrics, 'audit': audit,
              'figures': {part: {kind: str(Path(p).relative_to(output_dir)) for kind, p in paths.items()}
                          for part, paths in figure_paths.items()},
              'git': {'commit': git_value('rev-parse', 'HEAD'), 'branch': git_value('branch', '--show-current'),
                      'working_tree_changes': git_value('status', '--short')},
              'code_sha256': code_hashes, 'config_sha256': hashlib.sha256(source_config).hexdigest(),
              'environment': {'python': platform.python_version(), 'packages': {
                  p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'optbinning', 'matplotlib', 'joblib']}},
              'command': ['python', 'experiments/morebadly/run_t05.py', '--input', str(input_path),
                          '--output', str(output_dir), '--config', str(config_path)],
              'output_sha256': {p.relative_to(output_dir).as_posix(): sha256(p) for p in sorted(output_dir.rglob('*')) if p.is_file()},
              'limitations': ['Fixed internal research partitions, not new independent test data.',
                              'Validation selects tree rounds and each threshold; no test-based tuning.',
                              'Shared run_pipeline.py remains owner-coordinated; this script verifies all T05 module contracts.',
                              'AI assistance does not replace personal mastery or team peer review.']}
    write_json(output_dir / 'record.json', record)
    print(json.dumps({'output': str(output_dir), 'tree_best_iteration': details['tree']['best_iteration'],
                      'test_metrics': {name: by_part['test'] for name, by_part in metrics.items()}}, ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.json')
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/morebadly' / ('T05-' + uuid4().hex[:12])
    try:
        run(args.input, output, args.config)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f'T05 experiment failed: {error}\n')


if __name__ == '__main__':
    main()
