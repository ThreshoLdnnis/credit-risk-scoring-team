"""六项学员练习。实现 TODO 后运行 pytest；小样本只用于课堂演示。"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def missing_rate(frame: pd.DataFrame) -> pd.Series:
    """E01：逐列缺失数 / 总行数，保持列顺序。0 不算缺失。"""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("输入必须为 DataFrame")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("DataFrame 至少需要一行一列")

    return frame.isna().sum(axis=0) / len(frame)


def split_three_way(y, groups, seed=42) -> dict[str, np.ndarray]:
    """E02：按组分层切分，返回原始行位置。"""
    y = np.asarray(y)
    groups = np.asarray(groups)

    if y.ndim != 1 or groups.ndim != 1:
        raise ValueError("y 和 groups 必须是一维")
    if len(y) == 0 or len(y) != len(groups):
        raise ValueError("y 和 groups 必须非空且长度相同")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y 必须同时包含 0 和 1")
    if pd.isna(groups).any():
        raise ValueError("groups 不能有缺失值")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed 必须是整数")

    group_codes, unique_groups = pd.factorize(groups, sort=False)
    if (group_codes < 0).any():
        raise ValueError("组标识无效")

    # 每组取最大标签，仅用于分层
    group_y = np.zeros(len(unique_groups), dtype=int)
    np.maximum.at(group_y, group_codes, y.astype(int))

    group_indices = np.arange(len(unique_groups))

    try:
        train_groups, remaining_groups = train_test_split(
            group_indices,
            test_size=0.30,
            random_state=seed,
            stratify=group_y
        )

        val_groups, test_groups = train_test_split(
            remaining_groups,
            test_size=0.50,
            random_state=seed,
            stratify=group_y[remaining_groups]
        )
    except ValueError as exc:
        raise ValueError("组数量或类别分布不足以完成分层切分") from exc

    train = np.flatnonzero(np.isin(group_codes, train_groups))
    validation = np.flatnonzero(np.isin(group_codes, val_groups))
    test = np.flatnonzero(np.isin(group_codes, test_groups))

    # 检查覆盖、互斥
    combined = np.concatenate([train, validation, test])
    if len(combined) != len(y) or not np.array_equal(
        np.sort(combined), np.arange(len(y))
    ):
        raise ValueError("切分结果未完整覆盖原始行")

    # 检查每个分区均有两类标签
    for indices in [train, validation, test]:
        if len(np.unique(y[indices])) != 2:
            raise ValueError("每个分区都必须包含 0 和 1")

    return {
        "train": train,
        "validation": validation,
        "test": test
    }


def roc_auc_manual(y, probability) -> float:
    """E03：通过正负样本两两比较计算 AUC。"""
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)

    if y.ndim != 1 or probability.ndim != 1:
        raise ValueError("输入必须是一维数组")
    if len(y) == 0 or len(y) != len(probability):
        raise ValueError("标签和概率必须非空且等长")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("标签必须同时包含 0 和 1")
    if not np.isfinite(probability).all():
        raise ValueError("概率必须是有限数值")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("概率必须在 [0, 1] 内")

    positive = probability[y == 1]
    negative = probability[y == 0]

    comparisons = (
        (positive[:, None] > negative[None, :]).astype(float)
        + 0.5 * (positive[:, None] == negative[None, :])
    )

    return float(comparisons.mean())


def woe_iv_from_counts(good, bad, alpha=0.5) -> tuple[np.ndarray, float]:
    """E04：根据好坏计数计算平滑 WOE 和 IV。"""
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)

    if good.ndim != 1 or bad.ndim != 1:
        raise ValueError("good 和 bad 必须是一维数组")
    if len(good) == 0 or len(good) != len(bad):
        raise ValueError("good 和 bad 必须非空且等长")
    if not np.isfinite(good).all() or not np.isfinite(bad).all():
        raise ValueError("计数必须是有限数值")
    if (good < 0).any() or (bad < 0).any():
        raise ValueError("计数不能为负数")
    if good.sum() <= 0 or bad.sum() <= 0:
        raise ValueError("好坏样本总数都必须大于 0")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha 必须是有限正数")

    k = len(good)
    pg = (good + alpha) / (good.sum() + alpha * k)
    pb = (bad + alpha) / (bad.sum() + alpha * k)

    woe = np.log(pg / pb)
    iv = np.sum((pg - pb) * woe)

    return woe, float(iv)


def probability_to_score(
    probability, base_score=600.0, base_odds=20.0, pdo=20.0
):
    """E05：将违约概率转换为信用评分。"""
    if not np.isscalar(base_score) or not np.isfinite(base_score):
        raise ValueError("base_score 必须是有限数值")
    if not np.isscalar(base_odds) or not np.isfinite(base_odds) or base_odds <= 0:
        raise ValueError("base_odds 必须是有限正数")
    if not np.isscalar(pdo) or not np.isfinite(pdo) or pdo <= 0:
        raise ValueError("pdo 必须是有限正数")

    p = np.asarray(probability, dtype=float)
    if p.size == 0:
        raise ValueError("probability cannot be empty")

    if not np.isfinite(p).all():
        raise ValueError("概率必须是有限数值")
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("概率必须严格位于 0 和 1 之间")

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    odds = (1 - p) / p
    score = offset + factor * np.log(odds)

    if p.ndim == 0:
        return float(score)
    return score


def compare_models(
    y,
    predictions: dict[str, np.ndarray],
    thresholds: dict[str, float]
):
    """E06：在同一标签下比较多个模型。"""
    y = np.asarray(y)

    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y 必须是一维非空数组")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y 必须同时包含 0 和 1")
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("predictions 必须是非空字典")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds 必须是字典")
    if list(predictions.keys()) != list(thresholds.keys()):
        raise ValueError("两个字典的模型名称和顺序必须一致")

    rows = []

    for model_name, probability in predictions.items():
        p = np.asarray(probability, dtype=float)
        threshold = thresholds[model_name]

        if p.ndim != 1 or len(p) != len(y):
            raise ValueError("概率必须与标签等长且为一维")
        if not np.isfinite(p).all():
            raise ValueError("概率必须是有限数值")
        if ((p < 0) | (p > 1)).any():
            raise ValueError("概率必须在 [0, 1] 内")
        if not np.isscalar(threshold) or not np.isfinite(threshold):
            raise ValueError("阈值必须是有限数值")
        if threshold < 0 or threshold > 1:
            raise ValueError("阈值必须在 [0, 1] 内")

        predicted = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, predicted, labels=[0, 1]
        ).ravel()

        rows.append({
            "model": model_name,
            "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "threshold": float(threshold)
        })

    return pd.DataFrame(rows).set_index("model")
