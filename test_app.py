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
        "base_log_odds": -1.0,
        "shap_values": [[0.2, -0.1]],
        "raw_log_odds": -0.9,
        "probability": [0.2890504974],
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
    assert any("不会把 baseline 或评分卡结果伪装成 SHAP" in message for message in ui.infos)
