"""T02 contracts: loading, cleaning, grouped splits, training EDA and local evidence."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data
from src.eda import build_training_eda

RAW_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
]
SPECIAL = [
    'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]
CLEAN_COLUMNS = RAW_COLUMNS + [name + '__special' for name in SPECIAL]
TARGET = 'SeriousDlqin2yrs'


def raw_frame():
    frame = pd.DataFrame([
        [0.1, 42, 0, 0.2, 3000, 4, 0, 1, 0, 2],
        [2.5, 0, 96, 50000, np.nan, 3, 98, 0, 96, np.nan],
        [-0.2, -2, 98, -1, 1_000_000, 9, 96, 3, 98, 0],
        [0.3, 60, np.nan, 0, 6000, 7, np.nan, 1, np.nan, 1],
    ], columns=RAW_COLUMNS, index=[90, 40, 70, 20])
    frame[TARGET] = [0, 1, 0, 1]
    frame.insert(0, 'Unnamed: 0', [81, 12, 93, 34])
    frame['note'] = ['keep', 'row', 'order', 'intact']
    return frame


def write_csv(path, frame):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(frame.columns)
        for row in frame.itertuples(index=False, name=None):
            writer.writerow(['' if pd.isna(value) else value for value in row])
    return path


def grouped_features(groups=200):
    # Each group has two identical inputs; quarter of groups have conflicting y.
    group = np.repeat(np.arange(groups), 2)
    frame = pd.DataFrame(np.zeros((2 * groups, 13)), columns=CLEAN_COLUMNS)
    frame['MonthlyIncome'] = 1000 + group
    frame['age'] = np.where(group % 2 == 0, np.nan, 40)
    y = np.zeros(2 * groups, dtype=int)
    y[(group % 4) == 1] = 1
    y[np.arange(2 * groups) % 8 == 1] = 1
    return frame, y, group


def test_load_preserves_rows_ids_extra_columns_and_numeric_missing(tmp_path):
    source = raw_frame()
    result = data.load_dataset(write_csv(tmp_path / 'training.csv', source))
    assert result['Unnamed: 0'].tolist() == [81, 12, 93, 34]
    assert result['note'].tolist() == ['keep', 'row', 'order', 'intact']
    assert result[TARGET].tolist() == [0, 1, 0, 1]
    assert result['age'].tolist() == [42, 0, -2, 60]
    assert np.isnan(result.loc[1, 'MonthlyIncome'])
    assert result.shape == (4, 13)


def test_load_official_unnamed_id_header_and_unlabeled_empty_target(tmp_path):
    frame = raw_frame().rename(columns={'Unnamed: 0': ''})
    frame[TARGET] = np.nan
    result = data.load_dataset(write_csv(tmp_path / 'test.csv', frame), labeled=False)
    assert result['Unnamed: 0'].tolist() == [81, 12, 93, 34]
    assert result[TARGET].isna().all()


def test_load_unlabeled_without_target(tmp_path):
    result = data.load_dataset(
        write_csv(tmp_path / 'test.csv', raw_frame().drop(columns=TARGET)), labeled=False
    )
    assert len(result) == 4
    assert TARGET not in result


@pytest.mark.parametrize('column', ['age', TARGET])
def test_load_rejects_missing_required_columns(tmp_path, column):
    path = write_csv(tmp_path / 'bad.csv', raw_frame().drop(columns=column))
    with pytest.raises(ValueError, match=column):
        data.load_dataset(path)


@pytest.mark.parametrize('column', ['age', TARGET, 'note'])
def test_load_rejects_duplicate_raw_header_before_pandas_renames_it(tmp_path, column):
    frame = raw_frame()
    frame = pd.concat([frame, frame[[column]]], axis=1)
    path = write_csv(tmp_path / 'duplicate.csv', frame)
    with pytest.raises(ValueError, match='[Dd]uplicate|重复'):
        data.load_dataset(path)


@pytest.mark.parametrize('bad', ['invalid', 'null', 'nan', np.inf, -np.inf])
@pytest.mark.parametrize('column', ['age', 'MonthlyIncome', TARGET])
def test_load_rejects_non_numeric_or_infinite_relevant_values(tmp_path, column, bad):
    frame = raw_frame().astype({column: object})
    frame.loc[90, column] = bad
    with pytest.raises(ValueError, match=column):
        data.load_dataset(write_csv(tmp_path / 'bad.csv', frame))


@pytest.mark.parametrize('column', ['MonthlyIncome', 'NumberOfDependents'])
def test_load_recognizes_official_na_feature_token(tmp_path, column):
    frame = raw_frame().astype({column: object})
    frame.loc[90, column] = 'NA'
    frame.loc[90, 'note'] = 'NA'
    result = data.load_dataset(write_csv(tmp_path / 'official_missing.csv', frame))
    assert np.isnan(result.loc[0, column])
    assert result.loc[0, 'note'] == 'NA'
    assert len(result) == 4


def test_load_recognizes_na_target_as_missing_in_unlabeled_mode(tmp_path):
    frame = raw_frame()
    frame[TARGET] = 'NA'
    result = data.load_dataset(write_csv(tmp_path / 'test.csv', frame), labeled=False)
    assert result[TARGET].isna().all()


def test_load_rejects_na_target_in_labeled_mode(tmp_path):
    frame = raw_frame().astype({TARGET: object})
    frame.loc[90, TARGET] = 'NA'
    with pytest.raises(ValueError, match=TARGET):
        data.load_dataset(write_csv(tmp_path / 'bad.csv', frame))


@pytest.mark.parametrize('bad', [np.nan, 2, -1, 0.5])
def test_load_requires_nonmissing_binary_labels(tmp_path, bad):
    frame = raw_frame().astype({TARGET: float})
    frame.loc[90, TARGET] = bad
    with pytest.raises(ValueError, match=TARGET):
        data.load_dataset(write_csv(tmp_path / 'bad.csv', frame))


def test_load_rejects_invalid_present_target_even_when_unlabeled(tmp_path):
    frame = raw_frame()
    frame.loc[90, TARGET] = 2
    with pytest.raises(ValueError, match=TARGET):
        data.load_dataset(write_csv(tmp_path / 'bad.csv', frame), labeled=False)


@pytest.mark.parametrize('text', ['', ',age\n1,20\n'])
def test_load_empty_or_incomplete_csv_has_explicit_error(tmp_path, text):
    path = tmp_path / 'empty.csv'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError):
        data.load_dataset(path)


@pytest.mark.parametrize('delta', [-1, 1])
def test_load_rejects_rows_with_missing_or_extra_fields(tmp_path, delta):
    path = tmp_path / 'malformed.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(RAW_COLUMNS + [TARGET])
        writer.writerow([1] * (11 + delta))
    with pytest.raises(ValueError, match='[Rr]ow|[Ff]ield|行|列'):
        data.load_dataset(path)


def test_load_rejects_header_only_csv(tmp_path):
    path = write_csv(tmp_path / 'empty.csv', raw_frame().iloc[:0])
    with pytest.raises(ValueError, match='[Ee]mpty|[Rr]ow|空|行'):
        data.load_dataset(path)


def test_clean_returns_exact_ordered_features_without_mutating_source():
    frame = raw_frame()
    original = frame.copy(deep=True)
    clean = data.clean_features(frame[frame.columns[::-1]])
    assert clean.columns.tolist() == CLEAN_COLUMNS
    assert clean.index.tolist() == [90, 40, 70, 20]
    assert clean.shape == (4, 13)
    pd.testing.assert_frame_equal(frame, original)
    clean.loc[90, 'MonthlyIncome'] = 0
    assert frame.loc[90, 'MonthlyIncome'] == 3000


def test_clean_applies_only_agreed_age_and_delinquency_rules():
    clean = data.clean_features(raw_frame())
    np.testing.assert_allclose(clean['age'], [42, np.nan, -2, 60], equal_nan=True)
    for column in SPECIAL:
        np.testing.assert_allclose(clean[column], [0, np.nan, np.nan, np.nan], equal_nan=True)
        assert clean[column + '__special'].tolist() == [0, 1, 1, 0]
    assert clean['DebtRatio'].tolist() == [0.2, 50000, -1, 0]
    assert clean['RevolvingUtilizationOfUnsecuredLines'].tolist() == [0.1, 2.5, -0.2, 0.3]
    np.testing.assert_allclose(
        clean['MonthlyIncome'], [3000, np.nan, 1_000_000, 6000], equal_nan=True
    )


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'text', 'infinity'])
def test_clean_rejects_invalid_input_features(kind):
    frame = raw_frame()
    if kind == 'missing':
        frame = frame.drop(columns='age')
    elif kind == 'duplicate':
        frame = pd.concat([frame, frame[['age']]], axis=1)
    elif kind == 'text':
        frame = frame.astype({'age': object})
        frame.loc[90, 'age'] = 'invalid'
    else:
        frame.loc[90, 'MonthlyIncome'] = np.inf
    with pytest.raises(ValueError, match='age|MonthlyIncome|[Dd]uplicate|重复'):
        data.clean_features(frame)


def test_group_identity_uses_all_inputs_with_consistent_missing_and_no_index():
    features, _, _ = grouped_features(4)
    features.index = [100, 100, 200, 201, 300, 301, 400, 401]
    features['age'] = features['age'].astype(object)
    features.loc[:, 'age'] = [np.nan, None, 40, 40, pd.NA, np.nan, 40, 40]
    features.iloc[3, 12] = 1  # A different indicator must produce a different group.
    ids = data.feature_group_ids(features)
    np.testing.assert_array_equal(ids, [0, 0, 1, 2, 3, 3, 4, 4])
    assert np.issubdtype(ids.dtype, np.integer)


def test_split_covers_positions_once_without_group_leakage_and_preserves_y():
    features, values, groups = grouped_features()
    features.index = np.arange(9000, 9400)[::-1]
    labels = pd.Series(values, index=features.index)
    labels_before = labels.copy(deep=True)
    features_before = features.copy(deep=True)
    result = data.split_dataset(features, labels)
    assert set(result) == {'train', 'validation', 'test'}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(400))
    membership = []
    for positions in result.values():
        assert np.issubdtype(positions.dtype, np.integer)
        assert set(values[positions]) == {0, 1}
        membership.append(set(groups[positions]))
    assert not membership[0] & membership[1]
    assert not membership[0] & membership[2]
    assert not membership[1] & membership[2]
    pd.testing.assert_series_equal(labels, labels_before)
    pd.testing.assert_frame_equal(features, features_before)


def test_split_stratifies_by_group_max_even_for_mixed_label_duplicates():
    features, labels, groups = grouped_features()
    result = data.split_dataset(features, labels)
    assert [len(set(groups[result[key]])) for key in ['train', 'validation', 'test']] == [140, 30, 30]
    # Hand-derived: mod-4 group IDs 0 and 1 have maximum label 1, others 0.
    assert [
        sum(group % 4 in (0, 1) for group in set(groups[result[key]]))
        for key in ['train', 'validation', 'test']
    ] == [70, 15, 15]


def test_split_seed_is_reproducible_and_controls_membership():
    features, labels, _ = grouped_features()
    first = data.split_dataset(features, labels)
    repeated = data.split_dataset(features, labels, seed=42)
    changed = data.split_dataset(features, labels, seed=43)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    assert set(first['train']) != set(changed['train'])


def test_split_rejects_misaligned_series_labels_instead_of_silent_reordering():
    features, labels, _ = grouped_features()
    mismatched = pd.Series(labels, index=features.index[::-1])
    with pytest.raises(ValueError, match='[Ii]ndex|索引|align'):
        data.split_dataset(features, mismatched)


@pytest.mark.parametrize('kind', ['length', 'matrix', 'missing', 'nonbinary', 'single_class'])
def test_split_rejects_invalid_labels(kind):
    features, labels, _ = grouped_features()
    if kind == 'length':
        labels = labels[:-1]
    elif kind == 'matrix':
        labels = labels[:, None]
    elif kind == 'missing':
        labels = labels.astype(float)
        labels[0] = np.nan
    elif kind == 'nonbinary':
        labels[0] = 2
    else:
        labels[:] = 1
    with pytest.raises(ValueError):
        data.split_dataset(features, labels)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'duplicate', 'text', 'infinity'])
def test_split_rejects_wrong_cleaned_feature_contract(kind):
    features, labels, _ = grouped_features()
    if kind == 'missing':
        features = features.drop(columns=CLEAN_COLUMNS[-1])
    elif kind == 'extra':
        features['row_id'] = np.arange(len(features))
    elif kind == 'duplicate':
        features = pd.concat([features, features[['age']]], axis=1)
    elif kind == 'text':
        features = features.astype({'age': object})
        features.loc[0, 'age'] = 'invalid'
    else:
        features.loc[0, 'age'] = np.inf
    with pytest.raises(ValueError):
        data.split_dataset(features, labels)


def test_split_fails_explicitly_when_too_few_stratifiable_groups():
    features, labels, _ = grouped_features(4)
    with pytest.raises(ValueError, match='[Gg]roup|组'):
        data.split_dataset(features, labels)


def test_split_accepts_all_mixed_groups_when_each_partition_retains_both_labels():
    features, _, _ = grouped_features(100)
    labels = np.tile([0, 1], 100)
    result = data.split_dataset(features, labels)
    assert [len(result[key]) for key in ['train', 'validation', 'test']] == [140, 30, 30]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(200))
    group_sets = []
    for positions in result.values():
        assert set(labels[positions]) == {0, 1}
        group_sets.append(set(positions // 2))
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]


@pytest.fixture
def eda_training():
    # Losing pairwise-complete correlation or dropping missing rows breaks these values.
    frame = pd.DataFrame(
        {"amount": [0.0, 2.0, 4.0, np.nan], "income": [0.0, 4.0, 8.0, 12.0]},
        index=[101, 103, 107, 109],
    )
    target = pd.Series([0, 0, 1, 0], index=frame.index, name="SeriousDlqin2yrs")
    return frame, target


def test_eda_exports_correct_training_aggregates_and_readable_images(eda_training, tmp_path):
    frame, target = eda_training
    original_frame, original_target = frame.copy(deep=True), target.copy(deep=True)
    paths = build_training_eda(frame, target, tmp_path / "eda")

    expected_keys = {
        "missing_summary", "feature_summary", "label_counts", "correlation_matrix",
        "summary", "missing_plot", "distribution_plot", "label_plot", "correlation_plot",
    }
    assert set(paths) == expected_keys
    assert {path.name for path in (tmp_path / "eda").iterdir()} == {
        "missing_summary.csv", "feature_summary.csv", "label_counts.csv",
        "correlation_matrix.csv", "summary.json", "missing_rates.png",
        "feature_distributions.png", "label_distribution.png", "feature_correlations.png",
    }
    missing = pd.read_csv(paths["missing_summary"]).set_index("feature")
    assert missing.loc["amount", "missing_count"] == 1
    assert missing.loc["amount", "missing_rate"] == 0.25
    assert missing.loc["income", "non_missing_count"] == 4
    stats = pd.read_csv(paths["feature_summary"]).set_index("feature")
    assert stats.loc["amount", "mean"] == 2.0
    assert stats.loc["amount", "std"] == 2.0
    assert stats.loc["amount", "p25"] == 1.0
    assert stats.loc["amount", "median"] == 2.0
    assert stats.loc["amount", "p75"] == 3.0
    assert stats.loc["income", "max"] == 12.0
    counts = pd.read_csv(paths["label_counts"]).set_index("label")
    assert counts["count"].to_dict() == {0: 3, 1: 1}
    assert counts["proportion"].to_dict() == {0: 0.75, 1: 0.25}
    corr = pd.read_csv(paths["correlation_matrix"], index_col="feature")
    assert list(corr.columns) == ["amount", "income"]
    assert list(corr.index) == ["amount", "income"]
    assert corr.loc["amount", "income"] == pytest.approx(1.0)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["n_samples"] == 4
    assert summary["n_features"] == 2
    assert summary["total_missing_values"] == 1
    assert summary["correlation"]["method"] == "pearson"
    assert summary["correlation"]["missing_policy"] == "pairwise_complete"
    assert summary["correlation"]["target_included"] is False
    # Decode actual images: missing output or a placeholder file is a regression.
    from matplotlib.image import imread
    for key in ("missing_plot", "distribution_plot", "label_plot", "correlation_plot"):
        pixels = imread(paths[key])
        assert pixels.shape[0] >= 400
        assert pixels.shape[1] >= 600
        assert pixels.std() > 0.01
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(target, original_target)


def test_eda_keeps_constant_all_missing_and_extreme_features(tmp_path):
    # Invented zero correlations and clipping real extremes must be observable.
    frame = pd.DataFrame({
        "constant": [5.0, 5.0, 5.0, 5.0],
        "empty": [np.nan] * 4,
        "extreme": [0.0, 1.0, 2.0, 1e12],
        "signed": [-1e12, 0.0, 1.0, 2.0],
    })
    paths = build_training_eda(frame, pd.Series([0, 1, 0, 1]), tmp_path)
    stats = pd.read_csv(paths["feature_summary"]).set_index("feature")
    assert stats.loc["empty", "count"] == 0
    assert pd.isna(stats.loc["empty", "mean"])
    assert stats.loc["constant", "std"] == 0
    assert stats.loc["extreme", "max"] == 1e12
    assert stats.loc["extreme", "mean"] == 250000000000.75
    assert stats.loc["signed", "min"] == -1e12
    assert stats.loc["extreme", "display_transform"] == "log1p"
    assert stats.loc["signed", "display_transform"] == "raw"
    corr = pd.read_csv(paths["correlation_matrix"], index_col="feature")
    assert corr["constant"].isna().all()
    assert corr["empty"].isna().all()
    assert corr.loc["extreme", "signed"] == pytest.approx(1 / 3)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["correlation"]["unavailable_features"] == ["constant", "empty"]


def test_eda_single_row_and_one_label_class_are_reported_without_fabrication(tmp_path):
    paths = build_training_eda(pd.DataFrame({"x": [np.nan]}), pd.Series([1]), tmp_path)
    counts = pd.read_csv(paths["label_counts"]).set_index("label")
    assert counts["count"].to_dict() == {0: 0, 1: 1}
    corr = pd.read_csv(paths["correlation_matrix"], index_col="feature")
    assert pd.isna(corr.loc["x", "x"])


@pytest.mark.parametrize("case", [
    "empty_rows", "empty_columns", "length", "index", "text_feature", "infinite_feature",
    "invalid_label", "missing_label", "infinite_label", "text_label", "duplicate_columns",
    "target_in_features",
])
def test_eda_rejects_invalid_or_misaligned_inputs_before_writing(case, eda_training, tmp_path):
    frame, target = eda_training
    if case == "empty_rows":
        frame, target = frame.iloc[:0], target.iloc[:0]
    elif case == "empty_columns":
        frame = frame.iloc[:, :0]
    elif case == "length":
        target = target.iloc[:-1]
    elif case == "index":
        target = target.iloc[::-1]
    elif case == "text_feature":
        frame["amount"] = ["1", "2", "3", "bad"]
    elif case == "infinite_feature":
        frame.loc[101, "amount"] = np.inf
    elif case == "invalid_label":
        target.iloc[0] = 2
    elif case == "missing_label":
        target = target.astype(float)
        target.iloc[0] = np.nan
    elif case == "infinite_label":
        target = target.astype(float)
        target.iloc[0] = np.inf
    elif case == "text_label":
        target = target.astype(str)
    elif case == "duplicate_columns":
        frame.columns = ["amount", "amount"]
    elif case == "target_in_features":
        frame["SeriousDlqin2yrs"] = target
    with pytest.raises((ValueError, TypeError)):
        build_training_eda(frame, target, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_t02_runner_preserves_rows_and_saves_training_only_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / 'experiments/Sally-yi/run_t02.py'
    spec = importlib.util.spec_from_file_location('sally_t02_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    columns = [
        'RevolvingUtilizationOfUnsecuredLines', 'age',
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents',
    ]
    frame = pd.DataFrame(np.ones((80, 10)), columns=columns)
    frame['age'] = np.arange(20, 100)
    frame['MonthlyIncome'] = np.arange(1000, 1080)
    frame['SeriousDlqin2yrs'] = np.tile([0, 1], 40)
    frame.insert(0, 'Unnamed: 0', np.arange(501, 581))
    # Identical cleaned inputs with different labels must stay together.
    frame.loc[1, columns] = frame.loc[0, columns]
    source = tmp_path / 'input.csv'
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    output = tmp_path / 'run'
    summary = runner.run(source, output)
    assert source.read_bytes() == before
    clean = pd.read_parquet(output / 'cleaned_features.parquet')
    members = pd.read_csv(output / 'split_members.csv')
    assert clean.shape == (80, 13)
    assert members['row_position'].tolist() == list(range(80))
    assert members['raw_row_id'].tolist() == list(range(501, 581))
    assert members.loc[0, 'group_fingerprint'] == members.loc[1, 'group_fingerprint']
    assert members.groupby('group_fingerprint')['partition'].nunique().max() == 1
    assert set(members['partition']) == {'train', 'validation', 'test'}
    assert summary['rows'] == 80
    assert summary['audit']['all_rows_once']
    assert summary['audit']['group_overlap'] == 0
    assert summary['audit']['seed_reproducible']
    assert summary['training_eda_rows'] == (members['partition'] == 'train').sum()
    train_mask = members['partition'].eq('train').to_numpy()
    eda_counts = pd.read_csv(output / 'training_eda/label_counts.csv').set_index('label')['count']
    expected_counts = frame.loc[train_mask, 'SeriousDlqin2yrs'].value_counts().sort_index()
    assert eda_counts.to_dict() == expected_counts.to_dict()
    eda_stats = pd.read_csv(output / 'training_eda/feature_summary.csv').set_index('feature')
    assert eda_stats.loc['MonthlyIncome', 'mean'] == pytest.approx(frame.loc[train_mask, 'MonthlyIncome'].mean())
    assert json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['source']['sha256']
    assert (output / 'training_eda').is_dir()
    with pytest.raises(FileExistsError):
        runner.run(source, output)
