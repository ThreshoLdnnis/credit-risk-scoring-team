"""Audit a completed shared-entry run against original data and optional T02 evidence.

Run from repo root with --run-dir artifacts/... --output artifacts/.../audit.json.
The output contains aggregates and fingerprints only, never individual records.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import distributions
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clean_features, load_dataset, split_dataset  # noqa: E402

FEATURES = (
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
)
SPECIAL = ('NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
           'NumberOfTime60-89DaysPastDueNotWorse')
TARGET = 'SeriousDlqin2yrs'
EXPECTED_SOURCE = '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac'
EXPECTED_ROWS = {'train': 104974, 'validation': 22518, 'test': 22508}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member_fingerprints(frame):
    """Independently factorize complete rows; don't use pipeline grouping helpers."""
    groups = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)[0]
    unique, first = np.unique(groups, return_index=True)
    lookup = {}
    for group, position in zip(unique, first):
        values = [None if pd.isna(value) else float(0 if value == 0 else value).hex()
                  for value in frame.iloc[position]]
        payload = json.dumps([list(frame.columns), values], separators=(',', ':'))
        lookup[int(group)] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    require(len(set(lookup.values())) == len(lookup), 'Content fingerprint collision')
    return groups, np.array([lookup[int(group)] for group in groups])


def audit(input_path, run_dir, reference=None):
    input_path, run_dir = Path(input_path), Path(run_dir)
    require(sha256(input_path) == EXPECTED_SOURCE, 'Original training CSV SHA256 mismatch')
    record = json.loads((run_dir / 'record.json').read_text(encoding='utf-8'))
    require(record['status'] == 'completed_official_data_experiment', 'Shared entry did not finish on official data')
    require(record['source']['sha256'] == EXPECTED_SOURCE, 'Run source mismatch')
    raw = pd.read_csv(input_path)
    loaded = load_dataset(input_path)
    pd.testing.assert_frame_equal(loaded, raw)
    require(raw.shape == (150000, 12), 'Original training dimensions changed')
    require(raw['Unnamed: 0'].tolist() == list(range(1, len(raw) + 1)), 'Original row IDs changed')
    require(raw[TARGET].notna().all() and raw[TARGET].isin([0, 1]).all(), 'Invalid original labels')

    # Construct the expected table directly from the written rules, independently.
    expected = raw.loc[:, list(FEATURES)].copy()
    expected['age'] = expected['age'].mask(expected['age'].eq(0))
    for column in SPECIAL:
        indicator = expected[column].isin([96, 98])
        expected[column + '__special'] = indicator.astype('uint8')
        expected[column] = expected[column].mask(indicator)
    clean = clean_features(loaded)
    pd.testing.assert_frame_equal(clean, expected)
    require(record['feature_columns'] == list(expected.columns), 'Model feature order mismatch')
    groups, fingerprints = member_fingerprints(expected)
    labels = raw[TARGET].to_numpy()
    members = pd.read_csv(run_dir / 'split_members.csv')
    require(members['row_position'].tolist() == list(range(len(raw))), 'Member row order mismatch')
    np.testing.assert_array_equal(members['raw_row_id'], raw['Unnamed: 0'])
    np.testing.assert_array_equal(members['group_fingerprint'], fingerprints)
    require(members.groupby('group_id')['group_fingerprint'].nunique().eq(1).all(), 'Group ID merges distinct inputs')
    require(members.groupby('group_fingerprint')['group_id'].nunique().eq(1).all(), 'Same content has multiple groups')
    require(members.groupby('group_fingerprint')['partition'].nunique().eq(1).all(), 'A content group crosses partitions')

    with np.load(run_dir / 'split_positions.npz', allow_pickle=False) as archive:
        positions = {name: archive[name].copy() for name in EXPECTED_ROWS}
    require(np.array_equal(np.sort(np.concatenate(list(positions.values()))), np.arange(len(raw))),
            'Rows are missing, duplicated or outside the original range')
    reproduced = split_dataset(clean, loaded[TARGET], seed=42)
    split_summary = {}
    for name, pos in positions.items():
        require(len(pos) == EXPECTED_ROWS[name], name + ' row count mismatch')
        np.testing.assert_array_equal(pos, np.flatnonzero(members['partition'].eq(name)))
        np.testing.assert_array_equal(pos, reproduced[name])
        require(set(labels[pos]) == {0, 1}, name + ' is missing a target class')
        require(clean.iloc[pos].index.equals(loaded[TARGET].iloc[pos].index), name + ' index alignment mismatch')
        split_summary[name] = {'rows': len(pos), 'groups': int(len(np.unique(groups[pos]))),
            'negative': int((labels[pos] == 0).sum()), 'positive': int((labels[pos] == 1).sum()),
            'positive_rate': float(labels[pos].mean()),
            'position_sha256': hashlib.sha256(np.asarray(pos, dtype='<i8').tobytes()).hexdigest(),
            'ordered_content_sha256': hashlib.sha256('\n'.join(fingerprints[pos]).encode()).hexdigest()}
    overlap = {left + '/' + right: len(set(groups[positions[left]]) & set(groups[positions[right]]))
               for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    require(not any(overlap.values()), 'Independent content grouping finds cross-partition overlap')
    model_alignment = {}
    for model in record['selected_models']:
        table = pd.read_csv(run_dir / record['models'][model]['predictions'], float_precision='round_trip')
        pd.testing.assert_frame_equal(table[members.columns], members)
        np.testing.assert_array_equal(table['y'], labels)
        model_alignment[model] = {'rows': len(table), 'row_id_group_partition_and_label_match': True}
    baseline = joblib.load(run_dir / 'baseline.joblib')
    np.testing.assert_allclose(baseline['pipeline'].named_steps['imputer'].statistics_,
                               clean.iloc[positions['train']].median().to_numpy(), rtol=0, atol=0)

    reference_result = {'checked': False}
    if reference is not None:
        reference = Path(reference)
        old_members = pd.read_csv(reference / 'split_members.csv')
        pd.testing.assert_frame_equal(members, old_members)
        with np.load(reference / 'split_positions.npz', allow_pickle=False) as old:
            for name, pos in positions.items():
                np.testing.assert_array_equal(pos, old[name])
        pd.testing.assert_frame_equal(clean.reset_index(drop=True), pd.read_parquet(reference / 'cleaned_features.parquet'))
        reference_result = {'checked': True, 'member_rows_equal': len(members),
                            'all_positions_equal': True, 'all_cleaned_cells_equal': True}
    for relative, fingerprint in record['output_sha256'].items():
        require(sha256(run_dir / relative) == fingerprint, 'Output hash mismatch: ' + relative)
    for relative, fingerprint in record['code_sha256'].items():
        require(sha256(ROOT / relative) == fingerprint, 'Execution source changed: ' + relative)
    require(sha256(input_path) == EXPECTED_SOURCE, 'Original CSV changed during audit')
    group_labels = pd.DataFrame({'group': groups, 'y': labels}).groupby('group')['y'].nunique()
    installed = {dist.metadata['Name'].lower().replace('_', '-'): dist.version for dist in distributions()}
    constraints = dict(line.strip().split('==', 1) for line in (ROOT / 'constraints.txt').read_text(encoding='utf-8').splitlines()
                       if '==' in line and not line.lstrip().startswith('#'))
    differences = {name: {'expected': pin, 'installed': installed[name.lower().replace('_', '-')]}
                   for name, pin in constraints.items()
                   if name.lower().replace('_', '-') in installed and installed[name.lower().replace('_', '-')] != pin}
    return {
        'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'scope': 'Actual shared-entry data audit; aggregates only', 'status': 'passed',
        'source_sha256': EXPECTED_SOURCE, 'execution_commit': record['git']['commit'],
        'rows': len(raw), 'cleaned_columns': list(clean.columns),
        'groups': int(len(group_labels)), 'extra_rows_in_duplicate_groups': int(len(raw) - len(group_labels)),
        'mixed_label_groups': int(group_labels.gt(1).sum()),
        'labels': {str(k): int(v) for k, v in raw[TARGET].value_counts().sort_index().items()},
        'missing_before': {column: int(raw[column].isna().sum()) for column in FEATURES},
        'missing_after': {column: int(clean[column].isna().sum()) for column in clean},
        'training_missing': {column: int(clean.iloc[positions['train']][column].isna().sum()) for column in clean},
        'rule_counts': {'age_zero': int(raw['age'].eq(0).sum()),
                        'special': {column: int(raw[column].isin([96, 98]).sum()) for column in SPECIAL}},
        'splits': split_summary, 'group_overlap': overlap, 'all_rows_once': True,
        'seed_reproducible': True, 'independent_cleaning_equal': True,
        'model_row_alignment': model_alignment, 'baseline_imputer_equals_training_medians': True,
        't02_reference': reference_result, 'checked_output_hashes': len(record['output_sha256']),
        'code_sha256': record['code_sha256'], 'pipeline_record_sha256': sha256(run_dir / 'record.json'),
        'split_member_file_sha256': sha256(run_dir / 'split_members.csv'),
        'python': platform.python_version(), 'packages': installed,
        'installed_constraint_differences': differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Optional local T02 directory with row-level evidence')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Audit output already exists; choose a new file')
    result = audit(args.input, args.run_dir, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['status', 'rows', 'groups', 'splits', 'group_overlap',
        't02_reference', 'model_row_alignment', 'installed_constraint_differences']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
