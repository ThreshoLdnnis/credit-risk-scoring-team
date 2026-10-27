"""Aggregate exploration of an already selected training partition only."""

import json
import math
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from pandas.api.types import is_complex_dtype, is_numeric_dtype


def _validate_training_inputs(train_x, train_y):
    """Reject ambiguous alignment and invalid numbers before creating artifacts."""
    if not isinstance(train_x, pd.DataFrame) or not isinstance(train_y, pd.Series):
        raise TypeError("train_x must be a DataFrame and train_y must be a Series")
    if train_x.empty:
        raise ValueError("Training features must have at least one row and one column")
    if len(train_x) != len(train_y) or not train_x.index.equals(train_y.index):
        raise ValueError("Training features and labels must have matching lengths and indices")
    if not train_x.columns.is_unique or not all(
        isinstance(column, str) and column for column in train_x.columns
    ):
        raise ValueError("Feature names must be unique, nonempty strings")
    if "SeriousDlqin2yrs" in train_x or (
        train_y.name is not None and train_y.name in train_x.columns
    ):
        raise ValueError("The target must not be included in training features")
    for name, values in train_x.items():
        if not is_numeric_dtype(values.dtype) or is_complex_dtype(values.dtype):
            raise TypeError(f"Feature {name!r} must contain real numeric values")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Feature {name!r} contains infinity")
    if (
        not is_numeric_dtype(train_y.dtype)
        or is_complex_dtype(train_y.dtype)
        or train_y.isna().any()
        or not train_y.isin([0, 1]).all()
    ):
        raise ValueError("Training labels must contain only numeric 0 or 1 without missing values")


def _save_figure(figure, path):
    """Render without pyplot, GUI windows, or changing the application's backend."""
    try:
        FigureCanvasAgg(figure)
        figure.savefig(path, dpi=140, facecolor="white", bbox_inches="tight")
    finally:
        # Direct Figure objects have no pyplot manager to close; release artists.
        figure.clear()


def _plot_missing(missing, path, n_samples):
    width = max(9, 5 + max(map(len, missing["feature"])) * 0.10)
    figure = Figure(figsize=(width, max(4, len(missing) * 0.36 + 1.5)), layout="constrained")
    axis = figure.subplots()
    bars = axis.barh(missing["feature"], missing["missing_rate"], color="#3576a8")
    axis.invert_yaxis()
    axis.set_xlim(0, 1.15)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel("Missing fraction of training rows")
    axis.set_title(f"Training feature missingness | n = {n_samples:,}")
    axis.bar_label(bars, labels=[f"{rate:.2%}" for rate in missing["missing_rate"]], padding=4)
    _save_figure(figure, path)


def _plot_distributions(features, stats, path):
    columns = min(3, len(features.columns))
    rows = math.ceil(len(features.columns) / columns)
    figure = Figure(figsize=(max(8, columns * 4.7), max(4.3, rows * 3.6)), layout="constrained")
    axes = np.asarray(figure.subplots(rows, columns, squeeze=False)).ravel()
    for axis, name in zip(axes, features.columns):
        values = features[name].dropna().to_numpy(dtype=float)
        transform = stats.loc[name, "display_transform"]
        axis.set_title(fill(name, width=33), fontsize=10)
        axis.set_ylabel("Training row count")
        axis.set_xlabel("log1p(value), display only" if transform == "log1p" else "Raw value (full range)")
        if len(values):
            displayed = np.log1p(values) if transform == "log1p" else values
            axis.hist(displayed, bins=min(30, max(1, len(np.unique(displayed)))), color="#3576a8")
            axis.text(
                0.98, 0.97, f"Observed: {len(values):,}\nMissing: {features[name].isna().sum():,}",
                transform=axis.transAxes, ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        else:
            axis.text(0.5, 0.5, "All values missing\nDistribution unavailable", ha="center", va="center",
                      transform=axis.transAxes)
        axis.tick_params(labelsize=8)
    for axis in axes[len(features.columns):]:
        axis.set_visible(False)
    figure.suptitle(
        f"Training feature distributions | n = {len(features):,}\n"
        "No clipping or imputation; CSV statistics use original values", fontsize=12,
    )
    _save_figure(figure, path)


def _plot_labels(labels, path, n_samples):
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axis = figure.subplots()
    bars = axis.bar(["0: no serious delinquency", "1: serious delinquency"], labels["count"],
                    color=["#3576a8", "#d58a3a"])
    axis.set_ylim(0, max(labels["count"]) * 1.20)
    axis.set_ylabel("Training row count")
    axis.set_title(f"Training label distribution | n = {n_samples:,}")
    axis.bar_label(bars, labels=[f"{count:,} ({proportion:.2%})" for count, proportion in
                                zip(labels["count"], labels["proportion"])], padding=4)
    _save_figure(figure, path)


def _plot_correlations(correlation, path, n_samples):
    size = len(correlation)
    figure = Figure(figsize=(max(10, size * 0.55 + 6), max(5, size * 0.50 + 2)), layout="constrained")
    axis = figure.subplots()
    color_map = colormaps["RdBu_r"].copy()
    color_map.set_bad("#e4e4e4")
    values = correlation.to_numpy()
    mesh = axis.imshow(np.ma.masked_invalid(values), cmap=color_map, vmin=-1, vmax=1)
    identifiers = [f"F{i + 1:02d}" for i in range(size)]
    axis.set_xticks(range(size), identifiers)
    axis.set_yticks(range(size), [f"{code}  {name}" for code, name in zip(identifiers, correlation.index)],
                    fontsize=9)
    for row in range(size):
        for column in range(size):
            value = values[row, column]
            text = "N/A" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=8,
                      color="white" if np.isfinite(value) and abs(value) > 0.65 else "black")
    axis.set_title(
        f"Training Pearson correlation | n = {n_samples:,}\n"
        "Pairwise complete observations (at least 2); target excluded\n"
        "N/A: insufficient paired data or zero variance", fontsize=11,
    )
    figure.colorbar(mesh, ax=axis, label="Pearson r", shrink=0.75)
    _save_figure(figure, path)


def build_training_eda(train_x, train_y, output_dir):
    """Save training-only aggregates and return ``dict[str, pathlib.Path]``.

    Inputs are a nonempty real numeric DataFrame and a binary Series with the
    same index in the same order. Missing features are retained; labels cannot
    be missing. This function never splits, fits, imputes, clips, or mutates data.

    Returns nine paths, keyed by ``missing_summary`` (feature, missing_count,
    missing_rate, non_missing_count), ``feature_summary`` (count, missingness,
    mean, sample std, min, p25, median, p75, max, skew, n_unique and display_transform
    by feature), ``label_counts`` (label, count, proportion), ``correlation_matrix``
    (feature-indexed pairwise Pearson matrix), ``summary`` (JSON metadata), and
    ``missing_plot``, ``distribution_plot``, ``label_plot``, ``correlation_plot``
    (PNG). Only aggregates are exported, never original indices or row labels.

    Correlations are NaN when undefined, including constant/all-missing fields;
    CSV writes these cells empty and the chart labels them N/A. Histograms use
    log1p only for nonnegative features whose max exceeds 10 and ten times their
    median; this display-only choice is recorded and raw statistics are retained.
    All other histograms display their full raw range.
    """
    _validate_training_inputs(train_x, train_y)
    features = train_x.astype(float)
    n_samples, n_features = features.shape
    missing = pd.DataFrame({
        "feature": features.columns,
        "missing_count": features.isna().sum().to_numpy(),
        "missing_rate": features.isna().mean().to_numpy(),
        "non_missing_count": features.count().to_numpy(),
    })
    stats = features.describe().T.rename(columns={"25%": "p25", "50%": "median", "75%": "p75"})
    stats["count"] = stats["count"].astype(int)
    stats["missing_count"] = features.isna().sum()
    stats["missing_rate"] = features.isna().mean()
    stats["skew"] = features.skew()
    stats["n_unique"] = features.nunique()
    stats["display_transform"] = [
        "log1p" if row["min"] >= 0 and row["max"] > max(10, 10 * row["median"]) else "raw"
        for _, row in stats.iterrows()
    ]
    stats.index.name = "feature"
    labels = pd.DataFrame({"label": [0, 1], "count": [(train_y == 0).sum(), (train_y == 1).sum()]})
    labels["proportion"] = labels["count"] / n_samples
    correlation = features.corr(method="pearson", min_periods=2)
    correlation.index.name = "feature"
    summary = {
        "partition": "training", "n_samples": n_samples, "n_features": n_features,
        "features": list(features.columns), "total_missing_values": int(missing["missing_count"].sum()),
        "label_counts": {str(row.label): int(row.count) for row in labels.itertuples()},
        "label_proportions": {str(row.label): float(row.proportion) for row in labels.itertuples()},
        "correlation": {
            "method": "pearson", "missing_policy": "pairwise_complete", "min_observations": 2,
            "target_included": False, "undefined_cells": "empty in CSV; N/A in chart",
            "unavailable_features": list(stats.index[stats["n_unique"] < 2]),
        },
        "distribution": {
            "display_transforms": stats["display_transform"].to_dict(),
            "log1p_rule": "Nonnegative values, max > 10 and max > 10 * median; display only",
            "statistics_scale": "original values", "std_ddof": 1,
            "missing_policy": "Exclude missing values per feature; retain missing counts",
            "clipping": False, "imputation": False,
        },
        "export_scope": "Aggregates only; no source row identifiers or per-row labels",
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "missing_summary": "missing_summary.csv", "feature_summary": "feature_summary.csv",
        "label_counts": "label_counts.csv", "correlation_matrix": "correlation_matrix.csv",
        "summary": "summary.json", "missing_plot": "missing_rates.png",
        "distribution_plot": "feature_distributions.png", "label_plot": "label_distribution.png",
        "correlation_plot": "feature_correlations.png",
    }
    paths = {key: directory / filename for key, filename in filenames.items()}
    missing.to_csv(paths["missing_summary"], index=False)
    stats.to_csv(paths["feature_summary"])
    labels.to_csv(paths["label_counts"], index=False)
    correlation.to_csv(paths["correlation_matrix"])
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
    _plot_missing(missing, paths["missing_plot"], n_samples)
    _plot_distributions(features, stats, paths["distribution_plot"])
    _plot_labels(labels, paths["label_plot"], n_samples)
    _plot_correlations(correlation, paths["correlation_plot"], n_samples)
    return paths
