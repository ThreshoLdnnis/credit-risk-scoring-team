"""Focused tests for the train-only WOE/IV scorecard."""

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.scorecard import (
    _transform_woe,
    fit_scorecard,
    predict_scorecard,
)


@pytest.fixture
def scorecard_data():
    rng = np.random.default_rng(2026)
    size = 240
    signal = rng.uniform(0.0, 1.0, size)
    noisy_signal = signal + rng.normal(0.0, 0.16, size)
    y = (noisy_signal > 0.58).astype(np.int8)
    missing_feature = rng.normal(size=size)
    missing_feature[::13] = np.nan
    x = pd.DataFrame(
        {
            "signal": signal,
            "missing_feature": missing_feature,
            "constant": np.ones(size),
            "all_missing": np.full(size, np.nan),
            "no_training_missing": rng.normal(size=size),
        },
        index=pd.Index(np.arange(size) * 3 + 100, name="row"),
    )
    settings = {
        "backend": "optbinning",
        "max_n_bins": 5,
        "min_iv": 0.005,
        "C": 1.0,
        "base_score": 600.0,
        "base_odds": 20.0,
        "pdo": 20.0,
    }
    return x, y, settings


def test_fit_and_predict_returns_aligned_probability_score_and_points(scorecard_data):
    x, y, settings = scorecard_data
    original = x.copy(deep=True)
    bundle = fit_scorecard(x, y, settings)

    assert bundle["status"] == "fitted"
    assert bundle["selected_features"]
    assert bundle["versions"]["optbinning"]
    assert bundle["binning_evidence"]["signal"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert bundle["binning_evidence"]["signal"]["table"]
    signal_evidence = bundle["binning_evidence"]["signal"]
    assert signal_evidence["woe_direction"] == (
        "native OptBinning WoE = ln(non-event y=0 distribution / "
        "event y=1 distribution)"
    )
    assert {"Non-event", "Event", "WoE", "IV"} <= set(
        signal_evidence["table"][0]
    )
    assert sum(row["IV"] for row in signal_evidence["table"]) == pytest.approx(
        signal_evidence["iv"]
    )
    assert bundle["binning_evidence"]["missing_feature"][
        "missing_seen_in_training"
    ] is True
    assert bundle["binning_evidence"]["all_missing"]["status"] == "SKIPPED_ALL_MISSING"
    assert bundle["binning_evidence"]["constant"]["status"] == "SKIPPED_CONSTANT"
    assert x.equals(original)

    inference = x.iloc[[17, 4, 190, 26]].copy().loc[
        :, list(reversed(x.columns))
    ]
    splits_before = {
        name: binning.splits.copy()
        for name, binning in bundle["binning_process"].items()
    }
    coefficients_before = bundle["logistic_regression"].coef_.copy()
    result = predict_scorecard(bundle, inference)
    for name, splits in splits_before.items():
        np.testing.assert_array_equal(bundle["binning_process"][name].splits, splits)
    np.testing.assert_array_equal(
        bundle["logistic_regression"].coef_, coefficients_before
    )
    expected_columns = [
        "probability",
        "score",
        "base_points",
        *(f"points_{name}" for name in x.columns),
    ]
    assert result.columns.tolist() == expected_columns
    assert result.index.tolist() == inference.index.tolist()
    assert result["probability"].between(0.0, 1.0).all()
    point_columns = [column for column in result if column.startswith("points_")]
    np.testing.assert_allclose(
        result["score"],
        result["base_points"] + result[point_columns].sum(axis=1),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_array_equal(result["points_constant"], np.zeros(len(result)))
    np.testing.assert_array_equal(result["points_all_missing"], np.zeros(len(result)))


def test_missing_unseen_in_training_maps_to_native_neutral_woe(scorecard_data):
    x, y, settings = scorecard_data
    settings = {**settings, "min_iv": 0.0}
    bundle = fit_scorecard(x, y, settings)
    assert bundle["binning_evidence"]["no_training_missing"][
        "missing_seen_in_training"
    ] is False

    inference = x.iloc[:5].copy()
    inference.loc[inference.index[0], "no_training_missing"] = np.nan
    transformed = _transform_woe(
        inference,
        bundle["binning_process"],
        bundle["binning_evidence"],
        bundle["selected_features"],
    )
    assert "no_training_missing" in transformed
    assert transformed.loc[inference.index[0], "no_training_missing"] == 0.0


def test_iv_threshold_with_no_selected_features_fails_explicitly(scorecard_data):
    x, y, settings = scorecard_data
    settings = {**settings, "min_iv": 1e6}
    with pytest.raises(ValueError, match="没有特征达到"):
        fit_scorecard(x, y, settings)


def test_predict_rejects_unfitted_bundle(scorecard_data):
    x, y, settings = scorecard_data
    bundle = {
        "schema_version": 1,
        "status": "validated_unfitted",
        "input_features": list(x.columns),
    }
    with pytest.raises(RuntimeError, match="尚未包含已拟合"):
        predict_scorecard(bundle, x.iloc[:2])


def test_fit_rejects_one_class_and_misaligned_target(scorecard_data):
    x, y, settings = scorecard_data
    with pytest.raises(ValueError, match="同时包含类别"):
        fit_scorecard(x, np.zeros(len(x), dtype=np.int8), settings)
    with pytest.raises(ValueError, match="长度"):
        fit_scorecard(x, y[:-1], settings)


class IdentityBinning:
    def transform(self, values, *, metric, metric_missing):
        assert metric == "woe"
        assert metric_missing == 0.0
        return np.asarray(values, dtype=float)


class DeterministicLogit:
    coef_ = np.array([[1.0]])
    intercept_ = np.array([0.0])

    def decision_function(self, values):
        return np.asarray(values, dtype=float)[:, 0]


def test_score_anchor_pdo_direction_and_probability_saturation():
    factor = 20.0 / np.log(2.0)
    offset = 600.0 - factor * np.log(20.0)
    bundle = {
        "schema_version": 1,
        "status": "fitted",
        "input_features": ["woe"],
        "selected_features": ["woe"],
        "binning_process": {"woe": IdentityBinning()},
        "binning_evidence": {
            "woe": {"missing_seen_in_training": False},
        },
        "logistic_regression": DeterministicLogit(),
        "score_scale": {"factor": factor, "offset": offset},
    }
    x = pd.DataFrame(
        {"woe": [-np.log(20.0), -np.log(40.0), 1000.0, -1000.0]},
        index=["anchor", "double_odds", "probability_zero", "probability_one"],
    )
    result = predict_scorecard(bundle, x)

    assert result.loc["anchor", "probability"] == pytest.approx(1.0 / 21.0)
    assert result.loc["anchor", "score"] == pytest.approx(600.0)
    assert result.loc["double_odds", "probability"] == pytest.approx(1.0 / 41.0)
    assert result.loc["double_odds", "score"] == pytest.approx(620.0)
    assert result.loc["probability_zero", "probability"] == 1.0
    assert result.loc["probability_one", "probability"] == 0.0
    assert np.isfinite(result["score"]).all()
    np.testing.assert_allclose(
        result["score"],
        result["base_points"] + result["points_woe"],
        rtol=0,
        atol=1e-10,
    )


def test_fit_and_predict_reject_invalid_features(scorecard_data):
    x, y, settings = scorecard_data
    non_numeric = x.assign(category="unexpected")
    with pytest.raises(TypeError, match="非数值"):
        fit_scorecard(non_numeric, y, settings)
    infinite = x.copy()
    infinite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="无穷"):
        fit_scorecard(infinite, y, settings)

    bundle = fit_scorecard(x, y, settings)
    with pytest.raises(ValueError, match="字段与训练字段不一致"):
        predict_scorecard(bundle, x.drop(columns="signal").iloc[:2])


def test_label_series_must_align_with_training_rows(scorecard_data):
    x, y, settings = scorecard_data
    labels = pd.Series(y, index=x.index)
    with pytest.raises(ValueError, match="行顺序"):
        fit_scorecard(x, labels.iloc[::-1], settings)
    bundle = fit_scorecard(x, labels, settings)
    assert bundle["status"] == "fitted"


def test_complex_features_are_rejected_before_float_conversion(scorecard_data):
    x, y, settings = scorecard_data
    complex_input = x.copy()
    complex_input["signal"] = complex_input["signal"].astype(complex) + 2j
    with pytest.raises(TypeError, match="复数"):
        fit_scorecard(complex_input, y, settings)
    bundle = fit_scorecard(x, y, settings)
    with pytest.raises(TypeError, match="复数"):
        predict_scorecard(bundle, complex_input.iloc[:2])


def test_save_load_and_inference_never_refits(scorecard_data, tmp_path, monkeypatch):
    from optbinning import OptimalBinning
    from sklearn.linear_model import LogisticRegression

    x, y, settings = scorecard_data
    bundle = fit_scorecard(x, y, settings)
    path = tmp_path / "scorecard.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    expected = predict_scorecard(bundle, x)

    def fail_fit(*args, **kwargs):
        pytest.fail("Inference attempted to refit bins or logistic regression")

    monkeypatch.setattr(OptimalBinning, "fit", fail_fit)
    monkeypatch.setattr(LogisticRegression, "fit", fail_fit)
    pd.testing.assert_frame_equal(predict_scorecard(restored, x), expected, check_exact=True)
    assert restored["selected_features"] == bundle["selected_features"]


def test_lr_nonconvergence_cannot_be_reported_as_success(scorecard_data, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def nonconverged(*args, **kwargs):
        warnings.warn("Convergence test", ConvergenceWarning)

    monkeypatch.setattr(LogisticRegression, "fit", nonconverged)
    with pytest.raises(RuntimeError, match="未收敛"):
        fit_scorecard(*scorecard_data)


def test_failed_binning_is_not_silently_accepted(scorecard_data, monkeypatch):
    import optbinning

    class FailedBinning:
        status = "INFEASIBLE"

        def __init__(self, **kwargs):
            pass

        def fit(self, *args):
            return self

    monkeypatch.setattr(optbinning, "OptimalBinning", FailedBinning)
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        fit_scorecard(*scorecard_data)
