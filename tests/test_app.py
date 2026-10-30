"""Streamlit page checks with UI and artifact dependencies isolated."""

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import app
from src.baseline import fit_baseline, predict_baseline
from src.data import FEATURE_COLUMNS, clean_features


class FakeStreamlit:
    def __init__(self, *, inputs=None, submit=False, selections=None):
        self.inputs = inputs or {}
        self.submit = submit
        self.selections = selections or {}
        self.errors = []
        self.warnings = []
        self.infos = []
        self.captions = []
        self.metrics = []
        self.dataframes = []
        self.json_values = []
        self.charts = []
        self.progress_values = []
        self.markdowns = []
        self.subheaders = []
        self.selectbox_options = {}

    def set_page_config(self, **_kwargs):
        pass

    def title(self, _text):
        pass

    def caption(self, text):
        self.captions.append(text)

    def subheader(self, text):
        self.subheaders.append(text)

    def markdown(self, text, **_kwargs):
        self.markdowns.append(text)

    def info(self, text):
        self.infos.append(text)

    def warning(self, text):
        self.warnings.append(text)

    def error(self, text):
        self.errors.append(text)

    def metric(self, label, value):
        self.metrics.append((label, value))

    def dataframe(self, data, **_kwargs):
        self.dataframes.append(data)

    def bar_chart(self, data, **kwargs):
        self.charts.append((data.copy(), kwargs))

    def altair_chart(self, chart, **kwargs):
        self.charts.append((chart.to_dict(), kwargs))

    def progress(self, value, **_kwargs):
        self.progress_values.append(value)

    @contextmanager
    def expander(self, _label):
        yield self

    def json(self, value):
        self.json_values.append(value)

    def columns(self, count, **_kwargs):
        return [self] * (count if isinstance(count, int) else len(count))

    @contextmanager
    def container(self, **_kwargs):
        yield self

    def divider(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @contextmanager
    def form(self, _name):
        yield self

    def text_input(self, _label, *, value="", key, **_kwargs):
        return self.inputs.get(key, value)

    def form_submit_button(self, _label):
        return self.submit

    def selectbox(self, label, options, **_kwargs):
        self.selectbox_options[label] = list(options)
        return self.selections.get(label, options[0])


def _raw_training_data(size=64):
    rng = np.random.default_rng(123)
    values = rng.uniform(0.1, 8.0, size=(size, len(FEATURE_COLUMNS)))
    values[:, FEATURE_COLUMNS.index("age")] = rng.integers(21, 80, size=size)
    raw = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    target = (raw[FEATURE_COLUMNS[0]] > raw[FEATURE_COLUMNS[0]].median()).astype(int)
    return raw, target


def _input_values(raw_row):
    return {f"manual_{name}": str(raw_row[name]) for name in FEATURE_COLUMNS}


def _run_dir(tmp_path, models=("baseline",)):
    output = tmp_path / "run"
    output.mkdir()
    for name in models:
        (output / app.MODEL_FILES[name]).touch()
    metrics = {
        "baseline": {
            "train": {"auc": 0.70, "ks": 0.30, "ap": 0.10, "brier": 0.20},
            "validation": {"auc": 0.75, "ks": 0.35, "ap": 0.12, "brier": 0.18},
            "test": {"auc": 0.80, "ks": 0.40, "ap": 0.15, "brier": 0.16},
        },
        "scorecard": {
            "train": {"auc": 0.72, "ks": 0.32, "ap": 0.11, "brier": 0.19},
            "validation": {"auc": 0.77, "ks": 0.37, "ap": 0.13, "brier": 0.17},
            "test": {"auc": 0.82, "ks": 0.42, "ap": 0.16, "brier": 0.15},
        },
    }
    (output / "model_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    record = {
        "source": {"file": "cs-training.csv", "sha256": "fixture-sha"},
        "split": {"train": {"rows": 100}, "validation": {"rows": 20}, "test": {"rows": 20}},
    }
    (output / "record.json").write_text(json.dumps(record), encoding="utf-8")
    return output


def _baseline_bundle():
    raw, labels = _raw_training_data()
    features = clean_features(raw)
    y = pd.Series(labels.to_numpy(), index=features.index)
    return fit_baseline(features, y, {"C": 1.0, "penalty": "l2", "max_iter": 200})


def test_valid_input_uses_offline_baseline_and_displays_same_probability(tmp_path, monkeypatch):
    raw, _ = _raw_training_data()
    bundle = _baseline_bundle()
    _run_dir(tmp_path)
    monkeypatch.setattr(app, "load_model_bundle", lambda *_args: bundle)
    manual = _input_values(raw.iloc[0])
    values = {key.removeprefix("manual_"): value for key, value in manual.items()}
    offline_input = app.parse_manual_input(values)
    offline_probability = predict_baseline(bundle, offline_input)[0]
    ui = FakeStreamlit(inputs=manual, submit=True)

    app.render_page(ui, tmp_path)

    shown = next(value for label, value in ui.metrics if "y=1" in label)
    assert shown == f"{offline_probability:.2%}"
    assert not ui.errors
    assert any("Baseline 接口提供事件概率" in message for message in ui.infos)
    assert ui.dataframes  # the saved pipeline metrics are visible
    assert ui.progress_values == [pytest.approx(offline_probability)]


def test_invalid_manual_input_shows_input_error_before_loading_model(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    values = {f"manual_{name}": "1" for name in FEATURE_COLUMNS}
    values[f"manual_{FEATURE_COLUMNS[0]}"] = "not-a-number"
    monkeypatch.setattr(
        app,
        "load_model_bundle",
        lambda *_args: pytest.fail("invalid input must be rejected before model loading"),
    )
    ui = FakeStreamlit(inputs=values, submit=True)

    app.render_page(ui, tmp_path)

    assert any(message.startswith("输入错误：") for message in ui.errors)
    assert not any("y=1" in label for label, _ in ui.metrics)


def test_blank_fields_are_passed_through_shared_missing_value_cleaning():
    values = {name: "1" for name in FEATURE_COLUMNS}
    values["age"] = "0"
    values["MonthlyIncome"] = ""
    values["NumberOfTimes90DaysLate"] = "96"

    features = app.parse_manual_input(values)

    assert pd.isna(features.loc[0, "age"])
    assert pd.isna(features.loc[0, "MonthlyIncome"])
    assert features.loc[0, "NumberOfTimes90DaysLate"] != 96
    assert features.loc[0, "NumberOfTimes90DaysLate__special"] == 1


def test_no_model_bundle_shows_friendly_message(tmp_path):
    ui = FakeStreamlit()

    app.render_page(ui, tmp_path)

    assert any("没有找到可用的本地模型文件" in message for message in ui.warnings)
    assert not ui.errors


def test_scorecard_result_shows_score_and_feature_points(tmp_path, monkeypatch):
    import src.scorecard

    _run_dir(tmp_path, models=("scorecard",))
    monkeypatch.setattr(app, "load_model_bundle", lambda *_args: {"status": "fitted"})

    def fake_offline_predict(_bundle, features):
        return pd.DataFrame(
            {
                "probability": [0.1],
                "score": [590.0],
                "base_points": [580.0],
                "points_signal": [10.0],
            },
            index=features.index,
        )

    monkeypatch.setattr(src.scorecard, "predict_scorecard", fake_offline_predict)
    values = {f"manual_{name}": "1" for name in FEATURE_COLUMNS}
    ui = FakeStreamlit(inputs=values, submit=True)

    app.render_page(ui, tmp_path)

    assert ("信用评分", "590.0") in ui.metrics
    assert any("评分卡方向" in message for message in ui.captions)
    assert any(
        isinstance(table, list)
        and any(row.get("component") == "points_signal" for row in table)
        for table in ui.dataframes
    )
    contribution_chart = next(data for data, options in ui.charts if options.get("horizontal"))
    assert contribution_chart.loc["points_signal", "points"] == 10.0
    assert ui.progress_values == [0.1]


def test_metrics_use_selected_partition_and_show_saved_provenance(tmp_path):
    run_dir = _run_dir(tmp_path, models=("baseline", "scorecard"))
    ui = FakeStreamlit(selections={"评价数据分区": "validation"})

    app._render_provenance(ui, run_dir)
    app._render_metrics(ui, run_dir)

    assert any("cs-training.csv" in caption and "fixture-sha" in caption for caption in ui.captions)
    assert any("train 100 行 / validation 20 行 / test 20 行" in caption for caption in ui.captions)
    discrimination = ui.charts[0][0]
    assert discrimination.loc["Baseline（简单逻辑回归）", "AUC"] == 0.75
    assert discrimination.loc["WOE/IV 评分卡", "KS"] == 0.37
    assert all(value != "0.8000" for _, value in ui.metrics)


def test_dashboard_uses_banking_visual_hierarchy_without_risk_band_claims(tmp_path):
    _run_dir(tmp_path, models=("baseline", "scorecard"))
    ui = FakeStreamlit()

    app.render_page(ui, tmp_path)

    rendered = "\n".join(ui.markdowns)
    assert "--cr-navy: #10243a" in rendered
    assert "class=\"cr-hero\"" in rendered
    assert "class=\"cr-model-card\"" in rendered
    assert "客户风险评估工作台" in ui.subheaders
    assert not any("高风险客户" in value or "低风险客户" in value for value in ui.infos)


def test_model_load_failure_is_reported_without_traceback(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    monkeypatch.setattr(
        app,
        "load_model_bundle",
        lambda *_args: (_ for _ in ()).throw(FileNotFoundError("baseline.joblib missing")),
    )
    values = {f"manual_{name}": "1" for name in FEATURE_COLUMNS}
    ui = FakeStreamlit(inputs=values, submit=True)

    app.render_page(ui, tmp_path)

    assert any("模型加载失败：baseline.joblib missing" in message for message in ui.errors)
    assert not any("y=1" in label for label, _ in ui.metrics)


def test_prediction_failure_is_reported_without_result(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    monkeypatch.setattr(app, "load_model_bundle", lambda *_args: {"fitted": True})
    monkeypatch.setattr(
        app,
        "predict_with_bundle",
        lambda *_args: (_ for _ in ()).throw(ValueError("offline predictor rejected input")),
    )
    ui = FakeStreamlit(
        inputs={f"manual_{name}": "1" for name in FEATURE_COLUMNS},
        submit=True,
    )

    app.render_page(ui, tmp_path)

    assert any("预测失败：offline predictor rejected input" in message for message in ui.errors)
    assert not any("y=1" in label for label, _ in ui.metrics)


def test_explanation_mapping_is_displayed_when_interface_is_available():
    ui = FakeStreamlit()
    expected = {
        "base_log_odds": [-1.0],
        "shap_values": [[0.2, -0.1]],
        "raw_log_odds": [-0.9],
        "probability": [0.2890504974],
        "feature_names": ["x", "z"],
        "row_index_repr": ["0"],
        "metadata": {"model_output": "raw", "positive_class": 1},
    }

    app.render_explanation_panel(
        ui,
        "tree",
        {"model": "tree"},
        pd.DataFrame({"x": [1], "z": [2]}),
        available=True,
        explainer=lambda *_args: expected,
    )

    assert ui.json_values == [expected]
    assert not ui.errors


def test_unimplemented_or_failed_explainer_gets_clear_message():
    features = pd.DataFrame({"x": [1]})
    unavailable = FakeStreamlit()
    app.render_explanation_panel(
        unavailable, "tree", {}, features, available=True,
        explainer=lambda *_args: (_ for _ in ()).throw(NotImplementedError()),
    )
    assert any("解释模块尚未完成" in message for message in unavailable.infos)

    failed = FakeStreamlit()
    app.render_explanation_panel(
        failed, "tree", {}, features, available=True,
        explainer=lambda *_args: (_ for _ in ()).throw(ValueError("invalid SHAP result")),
    )
    assert any("解释计算失败：invalid SHAP result" in message for message in failed.errors)


def test_explanation_is_not_called_for_non_tree_models():
    ui = FakeStreamlit()
    called = False

    def explainer(*_args):
        nonlocal called
        called = True
        return {}

    app.render_explanation_panel(
        ui, "scorecard", {}, pd.DataFrame({"x": [1]}), available=True, explainer=explainer
    )

    assert not called
    assert any("SHAP 解释预留给 LightGBM" in message for message in ui.infos)


@pytest.fixture(scope="module")
def fitted_tree_case():
    """Small synthetic model used only for interface regression tests."""
    from src.tree_model import fit_tree

    raw, labels = _raw_training_data(size=128)
    raw.index = pd.Index(np.arange(len(raw)) + 500, name="source_row")
    features = clean_features(raw)
    target = pd.Series(labels.to_numpy(), index=features.index)
    bundle = fit_tree(
        features.iloc[:80], target.iloc[:80], features.iloc[80:], target.iloc[80:],
        {"seed": 42, "tree": {"n_estimators": 30, "early_stopping_rounds": 5,
                              "min_child_samples": 5, "num_leaves": 7}},
    )
    return raw, features, bundle


@pytest.mark.parametrize("models", [("tree",), ("baseline", "scorecard", "tree")])
def test_tree_only_and_three_model_runs_are_discovered_once(tmp_path, models):
    run_dir = _run_dir(tmp_path, models=models)

    assert app.discover_model_runs(tmp_path) == [run_dir]
    assert app.available_models(run_dir) == list(models)


@pytest.mark.parametrize("missing_and_special", [False, True])
def test_saved_tree_bundle_preserves_offline_probability_and_row_order(
    tmp_path, fitted_tree_case, missing_and_special,
):
    import joblib

    from src.tree_model import predict_tree

    raw, _features, bundle = fitted_tree_case
    run_dir = _run_dir(tmp_path, models=("tree",))
    joblib.dump(bundle, run_dir / "tree.joblib")
    inputs = raw.iloc[[6, 2]].copy()
    if missing_and_special:
        inputs["MonthlyIncome"] = np.nan
        inputs["age"] = 0
        inputs["NumberOfTimes90DaysLate"] = 96
    features = clean_features(inputs)
    loaded = app.load_model_bundle(run_dir, "tree")

    result = app.predict_with_bundle("tree", loaded, features)

    np.testing.assert_array_equal(result["probability"], predict_tree(bundle, features))
    assert result.index.equals(inputs.index)
    assert list(result.columns) == ["probability"]


def test_three_model_metrics_and_model_selector_include_lightgbm(tmp_path):
    run_dir = _run_dir(tmp_path, models=("baseline", "scorecard", "tree"))
    metrics_path = run_dir / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf8"))
    metrics["tree"] = {"test": {"auc": 0.86, "ks": 0.46, "ap": 0.20, "brier": 0.14}}
    metrics_path.write_text(json.dumps(metrics), encoding="utf8")
    ui = FakeStreamlit()

    app.render_page(ui, tmp_path)

    assert ui.selectbox_options["客户评估模型"] == ["baseline", "scorecard", "tree"]
    assert ui.charts[0][0].loc["LightGBM", "AUC"] == 0.86
    assert any("LightGBM" in text for text in ui.markdowns)
    assert not ui.errors


@pytest.mark.parametrize("invalid", [
    np.array([0.1, 0.2]), np.array([[0.1]]), np.array([np.nan]),
    np.array([np.inf]), np.array([-0.1]), np.array([1.1]),
])
def test_invalid_tree_probabilities_show_error_without_customer_results(tmp_path, monkeypatch, invalid):
    import src.tree_model

    _run_dir(tmp_path, models=("tree",))
    monkeypatch.setattr(app, "load_model_bundle", lambda *_args: {"model": "fixture"})
    monkeypatch.setattr(src.tree_model, "predict_tree", lambda *_args: invalid)
    ui = FakeStreamlit(inputs={f"manual_{name}": "1" for name in FEATURE_COLUMNS}, submit=True)

    app.render_page(ui, tmp_path)

    assert any(error.startswith("预测失败：") for error in ui.errors)
    assert not ui.metrics
    assert not ui.progress_values


def test_tree_prediction_and_real_shap_use_the_same_probability(tmp_path, fitted_tree_case):
    import joblib

    from src.tree_model import predict_tree

    raw, _features, bundle = fitted_tree_case
    run_dir = _run_dir(tmp_path, models=("tree",))
    joblib.dump(bundle, run_dir / "tree.joblib")
    values = raw.iloc[0].to_dict()
    values["MonthlyIncome"] = ""
    values["NumberOfTimes90DaysLate"] = "98"
    features = app.parse_manual_input(values)
    expected = float(predict_tree(bundle, features)[0])
    ui = FakeStreamlit(inputs={f"manual_{k}": str(v) for k, v in values.items()}, submit=True)

    app.render_page(ui, tmp_path)

    assert ui.metrics == [("两年内严重逾期概率（y=1）", f"{expected:.2%}")]
    assert ui.progress_values == [expected]
    assert any("基础值" in text for text in ui.captions)
    assert not any("没有 LightGBM" in text or "当前 Baseline" in text for text in ui.infos)
    assert ui.json_values[0]['probability'] == [expected]
    assert not ui.errors


def _render_actual_page(artifacts_root):
    import streamlit as st

    import app

    app.render_page(st, artifacts_root)


def test_actual_streamlit_tree_form_uses_saved_model(tmp_path, fitted_tree_case):
    import joblib
    from streamlit.testing.v1 import AppTest

    from src.tree_model import predict_tree

    raw, _features, bundle = fitted_tree_case
    run_dir = _run_dir(tmp_path, models=("tree",))
    joblib.dump(bundle, run_dir / "tree.joblib")
    at = AppTest.from_function(_render_actual_page, args=(str(tmp_path),)).run(timeout=20)
    assert not at.exception
    assert at.selectbox[1].options == ["LightGBM"]
    for name, value in raw.iloc[0].items():
        at.text_input(key=f"manual_{name}").set_value(str(value))

    at.button[0].click().run(timeout=20)

    expected = float(predict_tree(bundle, app.parse_manual_input(raw.iloc[0].to_dict()))[0])
    assert not at.exception
    assert not at.error
    assert len(at.metric) == 1
    assert at.metric[0].value == f"{expected:.2%}"
    assert any("基础值" in message.value for message in at.caption)


def test_failed_shap_does_not_remove_a_valid_tree_prediction(tmp_path, monkeypatch, fitted_tree_case):
    import joblib

    import src.explanation

    raw, _features, bundle = fitted_tree_case
    run_dir = _run_dir(tmp_path, models=("tree",))
    joblib.dump(bundle, run_dir / 'tree.joblib')
    monkeypatch.setattr(src.explanation, 'explain_tree', lambda *_args: (_ for _ in ()).throw(RuntimeError('test explanation failure')))
    ui = FakeStreamlit(inputs=_input_values(raw.iloc[0]), submit=True)

    app.render_page(ui, tmp_path)

    assert len(ui.metrics) == 1
    assert any('解释计算失败' in message for message in ui.errors)
    assert not ui.json_values
