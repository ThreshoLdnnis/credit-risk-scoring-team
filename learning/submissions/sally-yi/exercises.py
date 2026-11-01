"""Sally-yi 六项课堂练习：小样本核对公式，不代替正式业务模型。"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _real(values, name):
    """拒绝复数与非法文本，返回有限实数数组。"""
    try:
        original = np.asarray(values)
        if np.iscomplexobj(original) or original.dtype.kind in 'SU':
            raise ValueError(name)
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must contain real numbers') from error
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def _vector(values, name):
    result = _real(values, name)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f'{name} must be a nonempty one-dimensional vector')
    return result


def _labels(y):
    labels = _vector(y, 'y')
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError('y must contain both 0 and 1')
    return labels.astype(int)


def _scalar(value, name, positive=False):
    result = _real(value, name)
    if result.ndim != 0 or (positive and result <= 0):
        raise ValueError(f'{name} must be a finite scalar' + (' greater than zero' if positive else ''))
    return float(result)


def _probabilities(y, probability):
    values = _vector(probability, 'probability')
    if len(values) != len(y) or ((values < 0) | (values > 1)).any():
        raise ValueError('probability must match y length and lie in [0, 1]')
    return values


def _aligned(left, right):
    if isinstance(left, pd.Series) and isinstance(right, pd.Series) and not left.index.equals(right.index):
        raise ValueError('Series must have the same indices in the same order')


def missing_rate(frame: pd.DataFrame) -> pd.Series:
    """E01：先标出缺失，再按列求平均；0 不算缺失。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('frame must have at least one row and one column')
    return frame.isna().mean(axis=0)


def split_three_way(y, groups, seed=42) -> dict[str, np.ndarray]:
    """E02：先分组，再抽组，最后映射回原始行位置。"""
    labels = _labels(y)
    _aligned(y, groups)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    try:
        identifiers = np.asarray(groups, dtype=object)
        if identifiers.ndim != 1 or len(identifiers) != len(labels) or pd.isna(identifiers).any():
            raise ValueError('groups must match y length and have no missing values')
        codes, _ = pd.factorize(identifiers, sort=False)
    except (TypeError, ValueError) as error:
        raise ValueError('groups must contain valid one-dimensional identifiers') from error
    # 组最大标签只用于分层，绝不写回逐行 y。
    group_y = pd.Series(labels).groupby(codes, sort=False).max()
    try:
        train, rest = train_test_split(
            group_y.index.to_numpy(), test_size=.3,
            random_state=int(seed), stratify=group_y.to_numpy(),
        )
        validation, test = train_test_split(
            rest, test_size=.5, random_state=int(seed),
            stratify=group_y.loc[rest].to_numpy(),
        )
    except ValueError as error:
        raise ValueError('Not enough groups for the stratified split') from error
    parts = {
        name: np.flatnonzero(np.isin(codes, chosen))
        for name, chosen in [('train', train), ('validation', validation), ('test', test)]
    }
    if any(len(np.unique(labels[pos])) != 2 for pos in parts.values()):
        raise ValueError('Each partition must contain both target classes')
    if not np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(len(labels))):
        raise ValueError('Every row must appear exactly once')
    group_sets = [set(codes[pos]) for pos in parts.values()]
    if any(group_sets[i] & group_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('A group must not cross partitions')
    return parts


def roc_auc_manual(y, probability) -> float:
    """E03：正负配对，赢1分、平0.5分，除以总配对数。"""
    _aligned(y, probability)
    labels = _labels(y)
    values = _probabilities(labels, probability)
    positive, negative = values[labels == 1], values[labels == 0]
    points = 0.0
    for value in positive:
        points += np.count_nonzero(value > negative)
        points += .5 * np.count_nonzero(value == negative)
    return float(points / (len(positive) * len(negative)))


def _smoothed_log_distribution(counts, alpha):
    """等价于 log((计数+alpha)/(总数+alpha*K))，避免大数相加溢出。"""
    logs = np.full(counts.shape, -np.inf)
    np.log(counts, out=logs, where=counts > 0)
    smoothed = np.logaddexp(logs, np.log(alpha))
    return smoothed - np.logaddexp.reduce(smoothed)


def woe_iv_from_counts(good, bad, alpha=.5) -> tuple[np.ndarray, float]:
    """E04：好坏各算占比，取对数比，再将差异加权求和。"""
    good, bad = _vector(good, 'good'), _vector(bad, 'bad')
    alpha = _scalar(alpha, 'alpha', positive=True)
    if len(good) != len(bad) or (good < 0).any() or (bad < 0).any() or not (good > 0).any() or not (bad > 0).any():
        raise ValueError('Counts must match, be nonnegative, and have positive class totals')
    log_pg = _smoothed_log_distribution(good, alpha)
    log_pb = _smoothed_log_distribution(bad, alpha)
    pg, pb = np.exp(log_pg), np.exp(log_pb)
    woe = log_pg - log_pb
    return woe, float(np.sum((pg - pb) * woe))


def probability_to_score(probability, base_score=600.0, base_odds=20.0, pdo=20.0):
    """E05：好坏比翻倍加 PDO；对数形式避免先计算巨大赔率。"""
    values = _real(probability, 'probability')
    if values.size == 0 or ((values <= 0) | (values >= 1)).any():
        raise ValueError('probability must lie strictly in (0, 1)')
    base_score = _scalar(base_score, 'base_score')
    base_odds = _scalar(base_odds, 'base_odds', positive=True)
    pdo = _scalar(pdo, 'pdo', positive=True)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    # log((1-p)/p) = log(1-p)-log(p)：不截断，不提前四舍五入。
    score = offset + factor * (np.log1p(-values) - np.log(values))
    return float(score) if np.isscalar(probability) else np.asarray(score)


def compare_models(y, predictions: dict[str, np.ndarray], thresholds: dict[str, float]):
    """E06：同标签比较排名、概率误差和既定阈值的分类结果。"""
    labels = _labels(y)
    if not isinstance(predictions, dict) or not isinstance(thresholds, dict) or not predictions or predictions.keys() != thresholds.keys():
        raise ValueError('predictions and thresholds need the same nonempty keys')
    rows = []
    for name, probability in predictions.items():
        _aligned(y, probability)
        values = _probabilities(labels, probability)
        threshold = _scalar(thresholds[name], 'threshold')
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0, 1]')
        predicted = values >= threshold
        rows.append({
            'model': name, 'roc_auc': float(roc_auc_score(labels, values)),
            'brier': float(np.mean((values - labels)**2)),
            'tn': int(np.sum((labels == 0) & ~predicted)),
            'fp': int(np.sum((labels == 0) & predicted)),
            'fn': int(np.sum((labels == 1) & ~predicted)),
            'tp': int(np.sum((labels == 1) & predicted)),
            'threshold': threshold,
        })
    return pd.DataFrame(rows).set_index('model')
