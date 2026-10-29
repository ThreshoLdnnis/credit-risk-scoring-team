"""Train and apply a WOE/IV logistic-regression credit scorecard.

The positive label is the event (bad, y=1). OptBinning native WoE is used
without reorienting it. Binning, IV selection, and logistic regression are
fitted only from the rows passed to fit_scorecard.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

# 这些键和值来自 configs/experiment.json 的 scorecard 子字典。
# 目前假设调用方传入该子字典；集成前仍需由管线负责人确认。
_REQUIRED_SETTINGS = {
    "backend",
    "max_n_bins",
    "min_iv",
    "C",
    "base_score",
    "base_odds",
    "pdo",
}
_SUPPORTED_BACKEND = "optbinning"


def _validate_features(features: Any, *, argument_name: str) -> pd.DataFrame:
    """校验特征表结构和值类型，不改变或填补输入数据。"""
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"{argument_name} 必须是 pandas.DataFrame。")
    if features.empty:
        raise ValueError(f"{argument_name} 不能为空，至少需要一行和一列。")
    if features.shape[1] == 0:
        raise ValueError(f"{argument_name} 至少需要一列特征。")
    if not features.columns.is_unique:
        repeated = features.columns[features.columns.duplicated()].tolist()
        raise ValueError(f"{argument_name} 含有重复列名：{repeated!r}。")
    if not all(isinstance(column, str) and column for column in features.columns):
        raise TypeError(f"{argument_name} 的列名必须是非空字符串。")

    non_numeric = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column].dtype)
        or pd.api.types.is_bool_dtype(features[column].dtype)
        or pd.api.types.is_complex_dtype(features[column].dtype)
    ]
    if non_numeric:
        raise TypeError(f"{argument_name} 含非数值或复数特征列：{non_numeric!r}。")

    # NaN 留给后续分箱策略处理；正负无穷不能被当作缺失或静默裁剪。
    try:
        values = features.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{argument_name} 无法转换为有限精度数值。") from exc
    infinite_mask = np.isinf(values)
    if infinite_mask.any():
        row_positions, column_positions = np.where(infinite_mask)
        locations = [
            (features.index[row], features.columns[column])
            for row, column in zip(row_positions[:5], column_positions[:5])
        ]
        raise ValueError(
            f"{argument_name} 含正无穷或负无穷；不会裁剪或转换为缺失。"
            f"前几处位置：{locations!r}。"
        )
    return features


def _validate_binary_target(train_y: Any, *, expected_length: int) -> np.ndarray:
    """校验一维、无缺失且同时包含 0 和 1 的训练标签。"""
    if isinstance(train_y, pd.DataFrame):
        raise TypeError("train_y 必须是一维标签序列，不能是 DataFrame。")
    try:
        target = np.asarray(train_y)
    except (TypeError, ValueError) as exc:
        raise TypeError("train_y 必须可转换为一维标签序列。") from exc

    if target.ndim != 1:
        raise ValueError(f"train_y 必须是一维；当前维度为 {target.ndim}。")
    if target.size != expected_length:
        raise ValueError(
            f"train_y 长度 ({target.size}) 必须与 train_x 行数 "
            f"({expected_length}) 一致。"
        )
    if pd.isna(target).any():
        raise ValueError("train_y 含缺失标签；不会填补或忽略无效标签。")

    # 不接受字符串 '0'/'1' 或布尔值，以确保训练目标确为数值二元标签。
    if target.dtype.kind not in "iuf":
        raise TypeError("train_y 必须由数值 0 和 1 构成；不接受字符串或布尔标签。")
    if not np.isfinite(target).all():
        raise ValueError("train_y 含非有限值；标签只允许数值 0 和 1。")
    unique_values = np.unique(target)
    if not np.isin(unique_values, [0, 1]).all():
        raise ValueError(
            f"train_y 只能包含 0 和 1；检测到标签 {unique_values.tolist()!r}。"
        )
    if unique_values.size != 2:
        raise ValueError("train_y 必须同时包含类别 0 和类别 1，不能只有单一类别。")
    return target.astype(np.int8, copy=False)


def _validate_settings(settings: Any) -> dict[str, Any]:
    """校验 configs/experiment.json 中的 scorecard 子字典。"""
    if not isinstance(settings, Mapping):
        raise TypeError(
            "settings 必须是映射，并暂按 configs/experiment.json 的 "
            "scorecard 子字典传入；"
            "此传递方式待集成确认。"
        )
    missing_keys = sorted(_REQUIRED_SETTINGS - settings.keys())
    if missing_keys:
        raise ValueError(f"settings 缺少评分卡配置键：{missing_keys!r}。")
    extra_keys = sorted(set(settings.keys()) - _REQUIRED_SETTINGS)
    if extra_keys:
        raise ValueError(
            f"settings 含当前契约未定义的评分卡配置键：{extra_keys!r}；"
            "请先确认接口，不会静默忽略。"
        )

    if settings["backend"] != _SUPPORTED_BACKEND:
        raise ValueError(
            f"settings['backend'] 当前只支持 {_SUPPORTED_BACKEND!r}，"
            f"收到 {settings['backend']!r}。"
        )

    max_n_bins = settings["max_n_bins"]
    if (
        isinstance(max_n_bins, bool)
        or not isinstance(max_n_bins, Integral)
        or max_n_bins < 2
    ):
        raise ValueError("settings['max_n_bins'] 必须是大于等于 2 的整数。")

    numeric_ranges = {
        "min_iv": (0.0, None),
        "C": (0.0, None),
        "base_score": (None, None),
        "base_odds": (0.0, None),
        "pdo": (0.0, None),
    }
    for key, (minimum, _) in numeric_ranges.items():
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"settings[{key!r}] 必须是数值。")
        if not np.isfinite(value):
            raise ValueError(f"settings[{key!r}] 必须是有限值。")
        if minimum is not None and value < minimum:
            relation = "大于等于" if minimum == 0 else "大于"
            if minimum == 0 and key in {"C", "base_odds", "pdo"}:
                relation = "大于"
            raise ValueError(f"settings[{key!r}] 必须{relation} {minimum}。")
    if settings["C"] <= 0:
        raise ValueError("settings['C'] 必须大于 0。")
    if settings["base_odds"] <= 0:
        raise ValueError("settings['base_odds'] 必须大于 0。")
    if settings["pdo"] <= 0:
        raise ValueError("settings['pdo'] 必须大于 0。")
    return dict(settings)


def _new_bundle_template(
    *, input_features: list[str], settings: dict[str, Any]
) -> dict[str, Any]:
    """创建明确标记为未拟合的 bundle 骨架，不伪造训练产物。"""
    return {
        "schema_version": 1,
        "status": "validated_unfitted",
        "input_features": input_features,
        "selected_features": None,
        "binning_process": None,
        "iv_summary": None,
        "logistic_regression": None,
        "score_scale": {
            "base_score": float(settings["base_score"]),
            "base_odds": float(settings["base_odds"]),
            "pdo": float(settings["pdo"]),
            "factor": None,
            "offset": None,
        },
        "binning_evidence": None,
        "settings": settings.copy(),
        "versions": None,
        "unresolved": [],
    }


def _plain_value(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fit_woe_matrix(
    features: pd.DataFrame, target: np.ndarray, max_n_bins: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float]]:
    """Fit native OptBinning objects and produce train-only WoE evidence."""
    from optbinning import OptimalBinning

    binnings: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}
    iv_values: dict[str, float] = {}

    for name in features.columns:
        values = features[name].to_numpy(dtype=np.float64, na_value=np.nan)
        observed = values[~np.isnan(values)]
        missing_seen = bool(np.isnan(values).any())
        if observed.size == 0:
            evidence[name] = {
                "status": "SKIPPED_ALL_MISSING", "iv": 0.0, "splits": [],
                "missing_policy": (
                    "all-missing training feature excluded; "
                    "future values receive zero points"
                ),
                "table": [],
            }
            iv_values[name] = 0.0
            continue
        if np.unique(observed).size < 2:
            evidence[name] = {
                "status": "SKIPPED_CONSTANT", "iv": 0.0, "splits": [],
                "missing_policy": (
                    "constant training feature excluded; "
                    "future values receive zero points"
                ),
                "table": [],
            }
            iv_values[name] = 0.0
            continue

        binning = OptimalBinning(name=name, dtype="numerical", max_n_bins=max_n_bins)
        binning.fit(values, target)
        status = str(binning.status)
        if status not in {"OPTIMAL", "FEASIBLE"}:
            raise RuntimeError(
                f"特征 {name!r} 的 OptBinning 求解状态为 {status!r}；"
                "仅接受 OPTIMAL 或 FEASIBLE。"
            )
        table = binning.binning_table.build(add_totals=False)
        iv = float(binning.binning_table.iv)
        if not np.isfinite(iv) or iv < 0:
            raise RuntimeError(f"特征 {name!r} 的 OptBinning IV 非法：{iv!r}。")
        table_rows = [
            {str(key): _plain_value(value) for key, value in row.items()}
            for row in table.to_dict(orient="records")
        ]
        binnings[name] = binning
        iv_values[name] = iv
        evidence[name] = {
            "status": status,
            "iv": iv,
            "splits": [float(split) for split in binning.splits],
            "missing_seen_in_training": missing_seen,
            "missing_policy": (
                "native empirical WoE from the fitted missing bin"
                if missing_seen else "unseen missing values map to zero WoE"
            ),
            "woe_direction": (
                "native OptBinning WoE = ln(non-event y=0 distribution / "
                "event y=1 distribution)"
            ),
            "table": table_rows,
        }
    return binnings, evidence, iv_values


def _transform_woe(
    features: pd.DataFrame,
    binnings: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply fitted bins; missing absent from training maps to neutral WoE."""
    transformed = {}
    for name in feature_names:
        if name not in binnings:
            continue
        values = features[name].to_numpy(dtype=np.float64, na_value=np.nan)
        missing_seen = bool(evidence[name]["missing_seen_in_training"])
        metric_missing: str | float = "empirical" if missing_seen else 0.0
        woe = np.asarray(
            binnings[name].transform(
                values, metric="woe", metric_missing=metric_missing
            ),
            dtype=np.float64,
        )
        if woe.shape != (len(features),) or not np.isfinite(woe).all():
            raise RuntimeError(f"特征 {name!r} 的 WoE 转换产生非法结果。")
        transformed[name] = woe
    return pd.DataFrame(transformed, index=features.index, columns=list(transformed))


def fit_scorecard(train_x, train_y, settings: dict):
    """Fit train-only supervised bins, IV-selected WoE features, and L2 LR."""
    features = _validate_features(train_x, argument_name="train_x")
    if isinstance(train_y, pd.Series) and not train_y.index.equals(features.index):
        raise ValueError("train_y 索引必须与 train_x 的行顺序完全一致。")
    target = _validate_binary_target(train_y, expected_length=len(features))
    validated_settings = _validate_settings(settings)
    binnings, evidence, iv_values = _fit_woe_matrix(
        features, target, int(validated_settings["max_n_bins"])
    )
    selected = [
        name for name in features.columns
        if name in binnings and iv_values[name] >= float(validated_settings["min_iv"])
    ]
    if not selected:
        raise ValueError("没有特征达到 settings['min_iv']；不会继续拟合。")

    train_woe = _transform_woe(features, binnings, evidence, selected)
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(validated_settings["C"]), penalty="l2",
        solver="lbfgs", max_iter=2000, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(train_woe.loc[:, selected], target)
        except ConvergenceWarning as exc:
            raise RuntimeError("评分卡 LR 未收敛；请核验训练与参数后再评价。") from exc
    factor = float(validated_settings["pdo"]) / float(np.log(2.0))
    offset = float(validated_settings["base_score"]) - factor * float(
        np.log(validated_settings["base_odds"])
    )
    bundle = _new_bundle_template(
        input_features=list(features.columns), settings=validated_settings
    )
    bundle.update({
        "status": "fitted",
        "selected_features": selected,
        "binning_process": binnings,
        "iv_summary": {
            name: {"iv": iv_values[name], "selected": name in selected,
                   "threshold_inclusive": True}
            for name in features.columns
        },
        "logistic_regression": model,
        "score_scale": {
            "base_score": float(validated_settings["base_score"]),
            "base_odds": float(validated_settings["base_odds"]),
            "pdo": float(validated_settings["pdo"]),
            "factor": factor, "offset": offset,
            "formula": "score = offset - factor * logit(P(y=1))",
        },
        "binning_evidence": evidence,
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "optbinning": __import__("optbinning").__version__,
        },
        "unresolved": [],
    })
    return bundle


def predict_scorecard(bundle, features):
    """Return aligned probabilities, scores, base points, and feature points.

    No fitted transform is recalculated. Unseen numeric values use fitted
    outer bins; missing values absent from training map to zero WoE.
    Probabilities are not clipped; saturated sigmoid values still have scores
    computed from finite raw log-odds.
    """
    checked_features = _validate_features(features, argument_name="features")
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle 必须是 fit_scorecard 返回的映射结构。")
    if bundle.get("schema_version") != 1:
        raise ValueError("bundle 的 schema_version 缺失或不受支持。")
    if bundle.get("status") != "fitted":
        raise RuntimeError(
            "bundle 尚未包含已拟合的分箱器和 LR；当前阶段只创建校验后的未拟合结构。"
        )

    expected = bundle.get("input_features")
    if not isinstance(expected, list) or not all(
        isinstance(name, str) for name in expected
    ):
        raise ValueError("bundle['input_features'] 必须是训练时保存的字段名列表。")
    missing = [name for name in expected if name not in checked_features.columns]
    extra = [name for name in checked_features.columns if name not in expected]
    if missing or extra:
        raise ValueError(
            f"features 字段与训练字段不一致；缺少字段：{missing!r}；"
            f"额外字段：{extra!r}。"
        )
    selected = bundle.get("selected_features")
    binnings = bundle.get("binning_process")
    evidence = bundle.get("binning_evidence")
    model = bundle.get("logistic_regression")
    if (
        not isinstance(selected, list) or not selected
        or not isinstance(binnings, Mapping)
        or not isinstance(evidence, Mapping)
        or model is None
    ):
        raise ValueError("bundle 缺少已拟合的分箱、IV 证据或 LR 模型。")

    ordered = checked_features.loc[:, expected]
    woe = _transform_woe(ordered, binnings, evidence, selected)
    raw_logit = np.asarray(
        model.decision_function(woe.loc[:, selected]), dtype=np.float64
    )
    if raw_logit.shape != (len(ordered),) or not np.isfinite(raw_logit).all():
        raise RuntimeError("LR 原始 log-odds 形状错误或含非有限值。")
    probability = expit(raw_logit)

    scale = bundle.get("score_scale")
    if not isinstance(scale, Mapping):
        raise ValueError("bundle['score_scale'] 必须是映射。")
    factor, offset = scale.get("factor"), scale.get("offset")
    if not isinstance(factor, Real) or not isinstance(offset, Real):
        raise ValueError("bundle 缺少有效的评分 factor 或 offset。")
    factor, offset = float(factor), float(offset)
    base_points = offset - factor * float(model.intercept_[0])
    result = pd.DataFrame(index=ordered.index)
    result["probability"] = probability
    result["base_points"] = base_points
    points_columns = []
    coefficients = dict(zip(selected, model.coef_[0]))
    for name in expected:
        column = f"points_{name}"
        if name in coefficients:
            points = -factor * float(coefficients[name]) * woe[name].to_numpy()
        else:
            points = np.zeros(len(ordered), dtype=np.float64)
        result[column] = points
        points_columns.append(column)
    result["score"] = result["base_points"] + result[points_columns].sum(axis=1)
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("评分卡预测结果含非有限值。")
    expected_score = offset - factor * raw_logit
    if not np.allclose(
        result["score"].to_numpy(), expected_score, rtol=1e-10, atol=1e-10
    ):
        raise RuntimeError("评分总分与基准分及特征分项之和不一致。")
    return result.loc[:, ["probability", "score", "base_points", *points_columns]]
