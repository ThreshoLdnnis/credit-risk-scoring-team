"""morebadly's six classroom exercises, implemented with Codex assistance.

These small synthetic examples are separate from the real-data project.
Personal reproduction and explanation remain part of the learner's handover.
"""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def _numbers(values, name, *, vector=True):
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw) or (raw.dtype == object and any(np.iscomplexobj(v) for v in raw.flat)):
            raise ValueError('Complex values are not allowed')
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} must contain real numbers') from error
    if result.size == 0 or (vector and result.ndim != 1) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be nonempty, finite and have the required shape')
    return result


def _labels(y):
    result = _numbers(y, 'y')
    if set(np.unique(result)) != {0, 1}:
        raise ValueError('y must contain both binary 0/1 classes')
    return result.astype(int)


def _probability(y, probability):
    result = _numbers(probability, 'probability')
    if len(result) != len(y) or ((result < 0) | (result > 1)).any():
        raise ValueError('Probabilities must align with y and lie in [0,1]')
    return result


def _scalar(value, name, *, positive=False):
    numeric = _numbers(value, name, vector=False)
    if numeric.ndim != 0 or (positive and numeric <= 0):
        raise ValueError(f'{name} must be a finite scalar' + (' greater than zero' if positive else ''))
    return float(numeric)


def missing_rate(frame: pd.DataFrame) -> pd.Series:
    """E01: missing count / row count, retaining the input column order."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('frame must have at least one row and one column')
    return frame.isna().sum(axis=0) / len(frame)


def split_three_way(y, groups, seed=42) -> dict[str, np.ndarray]:
    """E02: stratify whole groups 70/15/15; never replace original labels."""
    labels = _labels(y)
    identifiers = np.asarray(groups)
    if identifiers.ndim != 1 or len(identifiers) != len(labels) or pd.isna(identifiers).any():
        raise ValueError('groups must be an aligned 1-D vector without missing IDs')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer in [0, 2**32 - 1]')
    try:
        codes, _ = pd.factorize(identifiers, sort=False)
        targets = pd.Series(labels).groupby(codes, sort=False).max()
        train, holdout = train_test_split(
            targets.index.to_numpy(), test_size=0.30, random_state=int(seed), stratify=targets.to_numpy(),
        )
        validation, test = train_test_split(
            holdout, test_size=0.50, random_state=int(seed), stratify=targets.loc[holdout].to_numpy(),
        )
    except (ValueError, TypeError) as error:
        raise ValueError('Cannot stratify the supplied groups into three partitions') from error
    result = {name: np.flatnonzero(np.isin(codes, chosen)) for name, chosen in [
        ('train', train), ('validation', validation), ('test', test),
    ]}
    if any(set(labels[pos]) != {0, 1} for pos in result.values()):
        raise ValueError('Every partition must contain both classes')
    if not np.array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(len(labels))):
        raise ValueError('Partitions must cover every original row once')
    return result


def roc_auc_manual(y, probability) -> float:
    """E03: explicit positive/negative pair comparisons, with half credit for ties."""
    labels = _labels(y)
    scores = _probability(labels, probability)
    positive, negative = scores[labels == 1], scores[labels == 0]
    points = sum(float(np.sum(p > negative)) + 0.5 * float(np.sum(p == negative)) for p in positive)
    return points / (len(positive) * len(negative))


def woe_iv_from_counts(good, bad, alpha=0.5) -> tuple[np.ndarray, float]:
    """E04: classroom additive smoothing, not OptBinning's native definition."""
    good, bad = _numbers(good, 'good'), _numbers(bad, 'bad')
    alpha = _scalar(alpha, 'alpha', positive=True)
    if good.shape != bad.shape or (good < 0).any() or (bad < 0).any():
        raise ValueError('good and bad must have the same shape and be nonnegative')
    if good.sum() <= 0 or bad.sum() <= 0 or not np.isfinite([good.sum(), bad.sum()]).all():
        raise ValueError('Both class totals must be finite and positive')
    pg = (good + alpha) / (good.sum() + alpha * len(good))
    pb = (bad + alpha) / (bad.sum() + alpha * len(bad))
    woe = np.log(pg) - np.log(pb)
    iv = float(np.sum((pg - pb) * woe))
    if not np.isfinite(woe).all() or not np.isfinite(iv):
        raise ValueError('Counts or smoothing exceed floating-point range')
    return woe, iv


def probability_to_score(probability, base_score=600.0, base_odds=20.0, pdo=20.0):
    """E05: score = base_score + PDO/ln(2) * (ln(good/bad) - ln(base_odds))."""
    probability = _numbers(probability, 'probability', vector=False)
    if ((probability <= 0) | (probability >= 1)).any():
        raise ValueError('Scoring probabilities must lie strictly in (0,1)')
    base_score = _scalar(base_score, 'base_score')
    base_odds = _scalar(base_odds, 'base_odds', positive=True)
    pdo = _scalar(pdo, 'pdo', positive=True)
    with np.errstate(over='ignore', invalid='ignore'):
        scores = base_score + pdo / np.log(2) * (
            np.log1p(-probability) - np.log(probability) - np.log(base_odds)
        )
    if not np.isfinite(scores).all():
        raise ValueError('Scoring parameters exceed floating-point range')
    return float(scores) if scores.ndim == 0 else scores


def compare_models(y, predictions: dict[str, np.ndarray], thresholds: dict[str, float]):
    """E06: compare aligned probabilities at externally fixed validation thresholds."""
    labels = _labels(y)
    if not isinstance(predictions, dict) or not isinstance(thresholds, dict) or not predictions or predictions.keys() != thresholds.keys():
        raise ValueError('Nonempty predictions and thresholds must have identical model keys')
    rows = []
    for name, probability in predictions.items():
        if isinstance(y, pd.Series) and isinstance(probability, pd.Series) and not y.index.equals(probability.index):
            raise ValueError('Probability Series must have the same row order as y')
        scores = _probability(labels, probability)
        threshold = _scalar(thresholds[name], 'threshold')
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0,1]')
        tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
        rows.append({'model': name, 'roc_auc': float(roc_auc_score(labels, scores)),
                     'brier': float(brier_score_loss(labels, scores)), 'tn': int(tn), 'fp': int(fp),
                     'fn': int(fn), 'tp': int(tp), 'threshold': threshold})
    return pd.DataFrame(rows).set_index('model')
