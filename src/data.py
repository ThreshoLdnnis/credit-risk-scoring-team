"""Validated CSV loading, stateless feature rules and reproducible group splits."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = 'SeriousDlqin2yrs'
FEATURE_COLUMNS = (
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
)
SPECIAL_COLUMNS = (
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
)
CLEAN_FEATURE_COLUMNS = FEATURE_COLUMNS + tuple(
    name + '__special' for name in SPECIAL_COLUMNS
)


def _validate_columns(columns, required, *, exact=False):
    if len(set(columns)) != len(columns):
        raise ValueError('Duplicate columns are not allowed')
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if exact:
        extra = [name for name in columns if name not in required]
        if extra:
            raise ValueError(f'Unexpected cleaned feature columns: {extra}')


def _numeric_column(values: pd.Series, name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors='raise')
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} contains non-numeric values') from error
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError(f'{name} contains infinite values')
    return numeric


def _numeric_features(frame: pd.DataFrame, columns, *, exact=False) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Features must be a nonempty DataFrame')
    _validate_columns(frame.columns, columns, exact=exact)
    return pd.DataFrame(
        {name: _numeric_column(frame[name], name) for name in columns},
        index=frame.index,
    ).copy()


def _validate_target(values: pd.Series, *, allow_missing=False) -> pd.Series:
    numeric = _numeric_column(values, TARGET_COLUMN)
    valid = numeric.isin([0, 1])
    if allow_missing:
        valid |= numeric.isna()
    if not valid.all():
        missing_note = '' if allow_missing else ' nonmissing'
        raise ValueError(f'{TARGET_COLUMN} must contain{missing_note} binary 0/1 labels')
    return numeric


def load_dataset(path: str | Path, labeled: bool = True) -> pd.DataFrame:
    """Read CSV in source row order, preserving IDs and extra metadata columns.

    Empty fields and the official exact ``NA`` token represent missing numeric
    values. Other text, including ``null`` and ``nan``, is rejected. Missing-token
    conversion is restricted to feature/target columns, preserving metadata text.
    The official blank export ID header is accepted. Unlabeled files may omit y
    or contain missing y; any present label must still be 0/1. Such files are
    prediction inputs, not evidence of model performance.
    """
    required = FEATURE_COLUMNS + ((TARGET_COLUMN,) if labeled else ())
    # Check the original header before pandas can rename duplicate column names.
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError('CSV is empty or has no header row')
            _validate_columns(header, required)
            rows = 0
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(
                        f'CSV row ending at line {reader.line_num} has {len(row)} '
                        f'fields; expected {len(header)}'
                    )
                rows += 1
            if rows == 0:
                raise ValueError('CSV contains no data rows')
    except csv.Error as error:
        raise ValueError(f'Malformed CSV: {error}') from error

    frame = pd.read_csv(
        path, encoding='utf-8-sig', keep_default_na=False,
        na_values={name: ['', 'NA'] for name in FEATURE_COLUMNS + (TARGET_COLUMN,)},
    )
    for name in FEATURE_COLUMNS:
        frame[name] = _numeric_column(frame[name], name)
    if TARGET_COLUMN in frame:
        frame[TARGET_COLUMN] = _validate_target(frame[TARGET_COLUMN], allow_missing=not labeled)
    return frame


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a new 13-column table; preserve rows, index and unresolved missing.

    Age zero and 96/98 delinquency codes follow the team's preregistered research
    rules, not an asserted official business definition. There is no fitting,
    imputation, clipping or row removal here. Indicators distinguish special
    codes from already-missing observations.
    """
    features = _numeric_features(frame, FEATURE_COLUMNS)
    features['age'] = features['age'].mask(features['age'].eq(0))
    for name in SPECIAL_COLUMNS:
        special = features[name].isin([96, 98])
        features[name + '__special'] = special.astype(np.uint8)
        features[name] = features[name].mask(special)
    return features


def feature_group_ids(features: pd.DataFrame) -> np.ndarray:
    """Return first-occurrence integer groups from all 13 cleaned input values.

    Missing values compare consistently and the DataFrame index is ignored.
    These IDs identify groups within this ordered table; they are not persistent
    cryptographic fingerprints across different datasets or row permutations.
    """
    numeric = _numeric_features(features, CLEAN_FEATURE_COLUMNS, exact=True)
    return numeric.groupby(list(CLEAN_FEATURE_COLUMNS), sort=False, dropna=False).ngroup().to_numpy(
        dtype=np.int64
    )


def split_dataset(features: pd.DataFrame, y, seed: int = 42) -> dict[str, np.ndarray]:
    """Split content groups 70/15/15 using group-max y; return original positions.

    A Series y must have exactly the same index (including order) as features;
    other 1-D label arrays are interpreted positionally. Labels are never changed
    or broadcast back from group maxima. Row ratios may differ when groups vary
    in size. Infeasible stratification raises ValueError rather than falling back
    to a split that leaks groups or loses a class.
    """
    groups = feature_group_ids(features)
    if isinstance(y, pd.Series) and not y.index.equals(features.index):
        raise ValueError('y index must exactly align with features.index in row order')
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError('y must be one-dimensional with one label per feature row')
    labels = _validate_target(pd.Series(labels)).to_numpy(dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError('y must contain both binary classes')

    group_targets = pd.Series(labels).groupby(groups, sort=False).max()
    try:
        train_groups, holdout_groups = train_test_split(
            group_targets.index.to_numpy(), test_size=0.30,
            random_state=seed, stratify=group_targets.to_numpy(),
        )
        validation_groups, test_groups = train_test_split(
            holdout_groups, test_size=0.50, random_state=seed,
            stratify=group_targets.loc[holdout_groups].to_numpy(),
        )
    except ValueError as error:
        raise ValueError(f'Cannot stratify groups into 70/15/15 partitions: {error}') from error

    result = {
        name: np.flatnonzero(np.isin(groups, chosen))
        for name, chosen in (
            ('train', train_groups), ('validation', validation_groups), ('test', test_groups)
        )
    }
    if any(len(np.unique(labels[positions])) != 2 for positions in result.values()):
        raise ValueError('Grouped split cannot put both classes in every partition')
    return result
