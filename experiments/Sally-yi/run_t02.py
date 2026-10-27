"""Run from repo root: python experiments/Sally-yi/run_t02.py.

All row-level outputs stay in a new ignored artifacts/ directory by default.
EDA sees training rows only; validation/test labels are used only for split audits.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    SPECIAL_COLUMNS,
    TARGET_COLUMN,
    clean_features,
    feature_group_ids,
    load_dataset,
    split_dataset,
)
from src.eda import build_training_eda  # noqa: E402


def sha256(path):
    """Hash bytes without modifying the source file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprints(features, groups):
    """SHA256 of ordered columns and canonical numeric group content.

    NaNs use null, +/-0 share 0x0.0p+0 and other values use float.hex.
    Fingerprints provide evidence; exact content grouping drives the split.
    """
    first_positions = np.unique(groups, return_index=True)[1]
    lookup = {}
    for position in first_positions:
        values = [None if pd.isna(v) else float(0 if v == 0 else v).hex()
                  for v in features.iloc[position]]
        payload = json.dumps([list(features.columns), values], separators=(',', ':'))
        lookup[int(groups[position])] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    if len(set(lookup.values())) != len(lookup):
        raise RuntimeError('Group fingerprint collision; refuse ambiguous evidence.')
    return [lookup[int(group)] for group in groups]


def git_value(*args):
    result = subprocess.run(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    if result.returncode:
        raise RuntimeError('Cannot record Git provenance: ' + ' '.join(args))
    return result.stdout.strip()


def run(input_path, output_dir, seed=42):
    """Return aggregate summary; refuse an existing output directory."""
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    # A failed attempt remains inspectable; subsequent attempts use a fresh directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(input_path)
    raw = load_dataset(input_path)
    features = clean_features(raw)
    y = raw[TARGET_COLUMN]
    positions = split_dataset(features, y, seed=seed)
    repeated = split_dataset(features, y, seed=seed)
    groups = feature_group_ids(features)
    all_positions = np.concatenate(list(positions.values()))
    all_rows_once = np.array_equal(np.sort(all_positions), np.arange(len(raw)))
    reproducible = all(np.array_equal(positions[k], repeated[k]) for k in positions)
    group_sets = {name: set(groups[pos]) for name, pos in positions.items()}
    pairs = [('train', 'validation'), ('train', 'test'), ('validation', 'test')]
    overlap = sum(len(group_sets[left] & group_sets[right]) for left, right in pairs)
    if not all_rows_once or not reproducible or overlap:
        raise RuntimeError('Split coverage, seed reproducibility or isolation audit failed.')
    partition = np.full(len(raw), '', dtype=object)
    split_summary = {}
    for name, pos in positions.items():
        partition[pos] = name
        counts = y.iloc[pos].value_counts()
        if set(counts.index) != {0, 1}:
            raise RuntimeError(f'{name} does not contain both target classes.')
        split_summary[name] = {
            'rows': len(pos), 'row_fraction': len(pos) / len(raw),
            'groups': len(group_sets[name]), 'negative': int(counts[0]),
            'positive': int(counts[1]), 'positive_rate': float(y.iloc[pos].mean()),
        }
    train_pos = positions['train']
    build_training_eda(features.iloc[train_pos], y.iloc[train_pos], output_dir / 'training_eda')
    features.to_parquet(output_dir / 'cleaned_features.parquet', index=False)
    # The official export identifier is metadata only, never a model input.
    raw_ids = raw['Unnamed: 0'] if 'Unnamed: 0' in raw else pd.Series([None] * len(raw))
    pd.DataFrame({
        'row_position': np.arange(len(raw)), 'raw_row_id': raw_ids.to_numpy(),
        'group_id': groups, 'group_fingerprint': content_fingerprints(features, groups),
        'partition': partition,
    }).to_csv(output_dir / 'split_members.csv', index=False)
    np.savez_compressed(output_dir / 'split_positions.npz', **positions)
    groups_and_y = pd.DataFrame({'group': groups, 'y': y.to_numpy()})
    group_label_counts = groups_and_y.groupby('group')['y'].nunique()
    summary = {
        'rows': len(raw), 'raw_columns': len(raw.columns), 'cleaned_columns': list(features.columns),
        'groups': int(len(group_label_counts)), 'extra_rows_in_duplicate_groups': int(len(raw) - len(group_label_counts)),
        'mixed_label_groups': int((group_label_counts > 1).sum()),
        'rules': {'age_zero_to_missing': int(raw['age'].eq(0).sum()),
                  'special_96_or_98_to_missing': {c: int(raw[c].isin([96, 98]).sum()) for c in SPECIAL_COLUMNS}},
        'missing_before': {c: int(raw[c].isna().sum()) for c in features if c in raw},
        'missing_after': {c: int(features[c].isna().sum()) for c in features},
        'splits': split_summary, 'training_eda_rows': len(train_pos),
        'audit': {'all_rows_once': bool(all_rows_once), 'group_overlap': overlap,
                  'seed_reproducible': bool(reproducible), 'each_split_has_both_classes': True},
    }
    if sha256(input_path) != source_hash:
        raise RuntimeError('Source CSV changed while processing.')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    code_paths = ['src/data.py', 'src/eda.py', 'experiments/Sally-yi/run_t02.py', 'configs/experiment.json']
    output_hashes = {p.relative_to(output_dir).as_posix(): sha256(p)
                     for p in sorted(output_dir.rglob('*')) if p.is_file()}
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'path': str(input_path), 'sha256': source_hash, 'rows': len(raw)},
        'git': {'base_or_current_commit': git_value('rev-parse', 'HEAD'),
                'branch': git_value('branch', '--show-current'),
                'working_tree_changes': git_value('status', '--short', '--untracked-files=normal')},
        'code_sha256': {p: sha256(ROOT / p) for p in code_paths},
        'environment': {'python': platform.python_version(),
                        'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'pyarrow']}},
        'split': {'seed': seed, 'target_group_proportions': [0.70, 0.15, 0.15],
                  'stratification': 'maximum original target in each cleaned-feature group',
                  'row_id_source': 'Unnamed: 0' if 'Unnamed: 0' in raw else None,
                  'fingerprint': 'SHA256(JSON([ordered columns, canonical float.hex values or null]))'},
        'command': ['python', 'experiments/Sally-yi/run_t02.py', '--input', str(input_path),
                    '--output', str(output_dir), '--seed', str(seed)],
        'output_sha256': output_hashes,
        'model_metrics': None,
        'scope': 'Data cleaning and partition audit; training-only EDA; no model trained or evaluated.',
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/Sally-yi' / (
        'T02-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8]
    )
    result = run(args.input, output, args.seed)
    print(json.dumps({'output_dir': str(output.resolve()), 'summary': result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
