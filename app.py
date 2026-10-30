"""Streamlit UI for local baseline, scorecard and LightGBM experiment bundles.

The page reuses the project's cleaning and prediction functions. It does not
fit models, alter thresholds, or make credit approval decisions.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MODEL_FILES = {
    "baseline": "baseline.joblib",
    "scorecard": "scorecard.joblib",
    "tree": "tree.joblib",
}
MODEL_LABELS = {
    "baseline": "Baseline（简单逻辑回归）",
    "scorecard": "WOE/IV 评分卡",
    "tree": "LightGBM",
}
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "无担保循环信贷使用率",
    "age": "年龄",
    "NumberOfTime30-59DaysPastDueNotWorse": "过去 2 年 30–59 天逾期次数",
    "DebtRatio": "债务比率",
    "MonthlyIncome": "月收入",
    "NumberOfOpenCreditLinesAndLoans": "未结清信贷额度数量",
    "NumberOfTimes90DaysLate": "过去 2 年 90 天以上逾期次数",
    "NumberRealEstateLoansOrLines": "房地产贷款或额度数量",
    "NumberOfTime60-89DaysPastDueNotWorse": "过去 2 年 60–89 天逾期次数",
    "NumberOfDependents": "家属人数",
}
METRIC_COLUMNS = (
    "auc",
    "ks",
    "ap",
    "pr_auc_trapezoidal",
    "brier",
    "log_loss",
    "threshold",
)


def _inject_design_tokens(st) -> None:
    """Style only app-owned HTML classes; avoid Streamlit's internal DOM selectors."""
    st.markdown(
        """<style>
        :root {
          --cr-navy: #10243a; --cr-blue: #2563a6; --cr-ink: #15283d;
          --cr-muted: #64758a; --cr-border: #dce4ec; --cr-surface: #ffffff;
          --cr-soft: #f4f7fa;
        }
        .cr-hero { box-sizing: border-box; padding: 1.7rem 2rem; margin: .25rem 0 1.25rem;
          border-radius: 18px; color: #f7fbff; background: linear-gradient(118deg,#10243a 0%,#173d62 72%,#205583 100%);
          box-shadow: 0 12px 28px rgba(16,36,58,.14); }
        .cr-eyebrow { font: 700 .72rem/1.4 sans-serif; letter-spacing: .14em; text-transform: uppercase; color: #a9c9e8; }
        .cr-hero h1 { margin: .45rem 0 .35rem; color: #fff; font: 700 clamp(1.65rem,3vw,2.3rem)/1.2 sans-serif; }
        .cr-hero p { margin: 0; max-width: 56rem; color: #d2dfec; font: 400 .96rem/1.65 sans-serif; }
        .cr-status { display: inline-block; margin-top: 1.1rem; padding: .34rem .7rem; border: 1px solid #7898b8;
          border-radius: 999px; color: #e5f0fa; font: 600 .76rem/1.3 sans-serif; }
        .cr-model-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(265px,1fr)); gap: .9rem; margin: .5rem 0 1rem; }
        .cr-model-card { padding: 1rem 1.05rem 1.05rem; border: 1px solid var(--cr-border); border-radius: 14px;
          background: var(--cr-surface); box-shadow: 0 5px 16px rgba(16,36,58,.045); }
        .cr-model-name { color: var(--cr-ink); font: 700 .95rem/1.4 sans-serif; }
        .cr-model-partition { margin-top: .15rem; color: var(--cr-muted); font: 500 .72rem/1.4 sans-serif; }
        .cr-kpi-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: .5rem; margin-top: .9rem; }
        .cr-kpi { min-width: 0; padding: .65rem .7rem; border: 1px solid #e5ebf1; border-radius: 10px; background: var(--cr-soft); }
        .cr-kpi-label { color: var(--cr-muted); font: 600 .68rem/1.3 sans-serif; }
        .cr-kpi-value { margin-top: .25rem; color: var(--cr-navy); font: 700 1.12rem/1.2 sans-serif; font-variant-numeric: tabular-nums; }
        .cr-section-kicker { margin: .2rem 0 .35rem; color: var(--cr-blue); font: 700 .7rem/1.4 sans-serif; letter-spacing: .1em; text-transform: uppercase; }
        @media (max-width: 680px) { .cr-hero { padding: 1.3rem 1.25rem; } .cr-kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } }
        </style>""",
        unsafe_allow_html=True,
    )


def _render_hero(st, run_count: int) -> None:
    status = (
        f"本地模型产物可用 · {run_count} 个实验批次"
        if run_count
        else "等待本地模型产物"
    )
    st.markdown(
        '<section class="cr-hero" aria-label="信用风险分析平台标题">'
        '<div class="cr-eyebrow">Credit Risk / Analytics</div>'
        '<h1>客户信用风险评估</h1>'
        '<p>基于团队训练产物查看模型表现，并通过同一套离线推理接口评估单条客户记录。'
        '结果用于研究演示，不构成贷款审批建议。</p>'
        f'<span class="cr-status">{html.escape(status)}</span>'
        '</section>',
        unsafe_allow_html=True,
    )


def _render_model_kpis(st, rows: list[dict[str, Any]], partition: str) -> None:
    model_cards = []
    for row in rows:
        name = html.escape(MODEL_LABELS.get(row["model"], row["model"]))
        cards = []
        for key, label in (("auc", "AUC"), ("ks", "KS"), ("ap", "AP"), ("brier", "Brier")):
            value = row.get(key)
            rendered = f"{float(value):.4f}" if isinstance(value, (int, float)) else "—"
            cards.append(
                '<div class="cr-kpi"><div class="cr-kpi-label">'
                f"{label}</div><div class=\"cr-kpi-value\">{rendered}</div></div>"
            )
        model_cards.append(
            '<article class="cr-model-card"><div class="cr-model-name">'
            f"{name}</div><div class=\"cr-model-partition\">{html.escape(partition)} 分区指标</div>"
            f'<div class="cr-kpi-grid">{"".join(cards)}</div></article>'
        )
    st.markdown(
        f'<div class="cr-model-grid">{"".join(model_cards)}</div>',
        unsafe_allow_html=True,
    )


def discover_model_runs(artifacts_root: str | Path | None = None) -> list[Path]:
    """Find local artifact directories containing a supported fitted bundle."""
    root = Path(artifacts_root) if artifacts_root is not None else Path(__file__).resolve().parent / "artifacts"
    if not root.is_dir():
        return []

    runs = {
        bundle.parent
        for filename in MODEL_FILES.values()
        for bundle in root.rglob(filename)
        if bundle.is_file()
    }
    return sorted(runs, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


def available_models(run_dir: str | Path) -> list[str]:
    """Return supported model names whose bundles exist in this run."""
    directory = Path(run_dir)
    return [name for name, filename in MODEL_FILES.items() if (directory / filename).is_file()]


def load_model_bundle(run_dir: str | Path, model_name: str) -> Any:
    """Load one trusted local joblib bundle; paths are fixed by model name."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"不支持的模型：{model_name}")
    bundle_path = Path(run_dir) / MODEL_FILES[model_name]
    if not bundle_path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{bundle_path.name}")

    import joblib

    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, Mapping):
        raise ValueError("模型文件内容不是有效的 bundle 映射")
    return bundle


def load_comparison_metrics(run_dir: str | Path) -> dict[str, Any] | None:
    """Read the shared pipeline's documented multi-model metrics file."""
    metrics_path = Path(run_dir) / "model_metrics.json"
    if not metrics_path.is_file():
        return None
    with metrics_path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not isinstance(metrics, dict):
        raise ValueError("model_metrics.json 顶层必须是模型名到分区指标的映射")
    return metrics


def load_run_record(run_dir: str | Path) -> dict[str, Any] | None:
    """Read optional experiment provenance without inferring missing metadata."""
    record_path = Path(run_dir) / "record.json"
    if not record_path.is_file():
        return None
    with record_path.open(encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError("record.json 顶层必须是对象")
    return record


def comparison_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the shared pipeline metrics into display-ready rows."""
    rows = []
    for model_name, partitions in metrics.items():
        if not isinstance(partitions, Mapping):
            continue
        for partition, values in partitions.items():
            if not isinstance(values, Mapping):
                continue
            row = {"model": model_name, "partition": partition}
            row.update({key: values[key] for key in METRIC_COLUMNS if key in values})
            rows.append(row)
    return rows


def parse_manual_input(values: Mapping[str, Any]):
    """Validate ten raw fields and apply the shared stateless cleaning rules."""
    import numpy as np
    import pandas as pd

    from src.data import FEATURE_COLUMNS, clean_features

    if not isinstance(values, Mapping):
        raise TypeError("手工输入必须是字段名到数值的映射")
    missing_columns = [name for name in FEATURE_COLUMNS if name not in values]
    extra_columns = [name for name in values if name not in FEATURE_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(
            f"输入字段不匹配；缺少：{missing_columns!r}，多余：{extra_columns!r}"
        )

    row = {}
    for name in FEATURE_COLUMNS:
        value = values[name]
        if isinstance(value, bool):
            raise ValueError(f"{FEATURE_LABELS[name]} 必须是数值或留空")
        if value is None or isinstance(value, str) and not value.strip():
            row[name] = np.nan
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{FEATURE_LABELS[name]} 不是有效数值") from error
        if not math.isfinite(number):
            raise ValueError(f"{FEATURE_LABELS[name]} 必须是有限数值，不能是 NaN 或无穷")
        row[name] = number

    raw = pd.DataFrame([row], columns=list(FEATURE_COLUMNS))
    return clean_features(raw)


def predict_with_bundle(model_name: str, bundle: Mapping[str, Any], features):
    """Call the matching offline predictor and validate aligned page output."""
    import numpy as np
    import pandas as pd

    if model_name == "baseline":
        from src.baseline import predict_baseline

        probability = np.asarray(predict_baseline(bundle, features), dtype=float)
        if probability.ndim != 1 or len(probability) != len(features):
            raise ValueError("Baseline 概率数量与输入行数不一致")
        result = pd.DataFrame({"probability": probability}, index=features.index)
    elif model_name == "tree":
        from src.tree_model import predict_tree

        probability = np.asarray(predict_tree(bundle, features), dtype=float)
        if probability.ndim != 1 or len(probability) != len(features):
            raise ValueError("LightGBM 概率数量与输入行数不一致")
        result = pd.DataFrame({"probability": probability}, index=features.index)
    elif model_name == "scorecard":
        from src.scorecard import predict_scorecard

        result = predict_scorecard(bundle, features)
        required = {"probability", "score", "base_points"}
        if not isinstance(result, pd.DataFrame) or not required.issubset(result.columns):
            raise ValueError("评分卡预测必须返回 probability、score 和 base_points")
        point_columns = [column for column in result if column.startswith("points_")]
        if not point_columns:
            raise ValueError("评分卡结果缺少 points_ 分项")
        score_sum = result["base_points"] + result[point_columns].sum(axis=1)
        if not np.allclose(result["score"], score_sum, rtol=0, atol=1e-8):
            raise ValueError("评分卡总分与基础分、变量分项之和不一致")
    else:
        raise ValueError(f"当前页面不支持模型：{model_name}")

    if not result.index.equals(features.index):
        raise ValueError("预测结果行索引与输入顺序不一致")
    probability = result["probability"].to_numpy(dtype=float)
    if (
        probability.shape != (len(features),)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("模型返回了无效概率")
    return result


def render_explanation_panel(
    st,
    model_name: str,
    bundle: Mapping[str, Any] | None,
    features,
    *,
    available: bool = False,
    explainer=None,
    expected_probability=None,
) -> None:
    """Render the declared tree explainer when an actual tree bundle exists."""
    st.subheader("模型解释（SHAP）")
    if model_name != "tree":
        st.info(
            "SHAP 解释预留给 LightGBM；当前模型的概率及评分卡分项可在上方查看。"
        )
        return
    if not available:
        st.info("填写客户信息并成功运行 LightGBM 评估后，这里会显示该客户的 SHAP 解释。")
        return

    if explainer is None:
        try:
            from src.explanation import explain_tree
        except ImportError as error:
            st.error(f"解释依赖不可用：{error}")
            return
        explainer = explain_tree

    try:
        explanation = explainer(bundle, features)
        from src.ui_explanations import validate_explanation

        explanation = validate_explanation(explanation, features, expected_probability)
    except NotImplementedError:
        st.info("解释模块尚未完成，当前没有可展示的 SHAP 结果。")
        return
    except Exception as error:
        st.error(f"解释计算失败：{error}")
        return
    if not isinstance(explanation, Mapping):
        st.error("解释接口返回格式无效；预期为包含 log-odds 与 SHAP 信息的映射。")
        return
    try:
        import altair as alt
        import pandas as pd

        from src.ui_explanations import local_contributions

        rows = local_contributions(explanation, FEATURE_LABELS)
        chart = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
            x=alt.X('起点:Q', title='原始 log-odds', scale=alt.Scale(zero=False)), x2='终点:Q',
            y=alt.Y('因素:N', sort=[row['因素'] for row in rows], title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color('方向:N', scale=alt.Scale(
                domain=['增加严重逾期倾向', '降低严重逾期倾向'], range=['#c0392b', '#2874a6']),
                legend=alt.Legend(orient='bottom', columns=1)),
            tooltip=['因素:N', alt.Tooltip('贡献:Q', format='.6f'), '方向:N'],
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        base, raw = explanation['base_log_odds'][0], explanation['raw_log_odds'][0]
        st.caption(f"基础值 {base:.6f} + 全部贡献 {raw - base:+.6f} = 模型输出 {raw:.6f}。"
                   "红色贡献增加、蓝色贡献降低模型的严重逾期倾向；单位是 log-odds，不是概率百分点或因果效应。")
        st.dataframe(pd.DataFrame(rows)[['因素', '贡献']], use_container_width=True)
        with st.expander("解释设置与数值核验"):
            st.json(dict(explanation))
    except Exception as error:
        st.error(f"解释图展示失败：{error}")


def _render_global_explanation(st, run_dir: Path) -> None:
    st.subheader("全局 SHAP：模型主要参考哪些因素")
    try:
        from src.ui_explanations import feature_label, load_global_explanation

        result = load_global_explanation(run_dir)
        if result is None:
            st.info("该实验尚无批量解释产物。完成完整运行流程后，这里会自动展示全局重要性。")
            return
        summary, table = result['summary'], result['importance'].copy()
        table['因素'] = [feature_label(name, FEATURE_LABELS) for name in table['feature']]
        display = table.rename(columns={'mean_abs_shap_log_odds': '平均绝对贡献（log-odds）'})
        st.bar_chart(display.set_index('因素')[['平均绝对贡献（log-odds）']], horizontal=True,
                     height=380, x_label='平均绝对贡献（log-odds）')
        st.caption(f"当前实验的 {summary['sample']['partition']} 分区固定样本，共 {summary['sample']['size']:,} 条。"
                   "柱长表示贡献大小，不表示方向、因果关系或业务收益。")
        if summary.get('models_with_reloaded_differences'):
            st.caption("模型复验记录了重载后的数值差异；历史评价指标保持原样，详情见实验复验记录。")
    except Exception as error:
        st.warning(f"全局解释不可用：{error}。单笔模型预测仍可使用。")


def _render_metrics(st, run_dir: Path) -> None:
    try:
        metrics = load_comparison_metrics(run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"评价指标读取失败：{error}")
        return
    if not metrics:
        st.info("该实验没有可用的 model_metrics.json；仍可使用已保存的模型预测。")
        return
    rows = comparison_rows(metrics)
    if not rows:
        st.info("指标文件中没有可展示的模型/分区指标。")
        return
    partitions = sorted({row["partition"] for row in rows})
    preferred = [name for name in ("test", "validation", "train") if name in partitions]
    options = preferred + [name for name in partitions if name not in preferred]
    selected_partition = st.selectbox(
        "评价数据分区",
        options,
        format_func=lambda name: {"test": "测试集（最终评价）", "validation": "验证集", "train": "训练集"}.get(name, name),
        key="metrics_partition",
    )
    selected_rows = [row for row in rows if row["partition"] == selected_partition]
    st.markdown('<div class="cr-section-kicker">Model Performance</div>', unsafe_allow_html=True)
    st.subheader("模型表现概览")
    st.caption(
        f"指标来自本地实验 {run_dir.name} 的 model_metrics.json；"
        f"当前展示 {selected_partition} 分区。AUC、KS、AP 越高通常代表区分能力越强；Brier 越低代表概率误差越小。"
    )
    _render_model_kpis(st, selected_rows, selected_partition)

    try:
        import pandas as pd
    except ImportError as error:
        st.error(f"图表依赖不可用：{error}")
        return
    chart_rows = [
        {
            "模型": MODEL_LABELS.get(row["model"], row["model"]),
            **{metric.upper(): row[metric] for metric in ("auc", "ks", "ap") if isinstance(row.get(metric), (int, float))},
        }
        for row in selected_rows
    ]
    brier_rows = [
        {"模型": MODEL_LABELS.get(row["model"], row["model"]), "Brier": row["brier"]}
        for row in selected_rows
        if isinstance(row.get("brier"), (int, float))
    ]
    with st.expander("模型对比图表与完整评价指标"):
        chart_columns = st.columns(2)
        if chart_rows and any(len(row) > 1 for row in chart_rows):
            chart_columns[0].markdown("**区分能力指标**")
            chart_columns[0].bar_chart(
                pd.DataFrame(chart_rows).set_index("模型"), height=290, y_label="指标值"
            )
        if brier_rows:
            chart_columns[1].markdown("**概率误差（越低越好）**")
            chart_columns[1].bar_chart(
                pd.DataFrame(brier_rows).set_index("模型"), height=290, y_label="Brier"
            )
        st.dataframe(rows, use_container_width=True)


def _render_provenance(st, run_dir: Path) -> None:
    try:
        record = load_run_record(run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.warning(f"实验来源信息读取失败：{error}")
        return
    if not record:
        st.info("该实验没有 record.json，数据来源与分区说明不可用。")
        return
    source = record.get("source", {})
    split = record.get("split", {})
    st.caption(
        "数据来源：{}；SHA256：{}".format(
            source.get("file", "unknown"), source.get("sha256", "unknown")
        )
    )
    counts = [f"{name} {split.get(name, {}).get('rows', 'unknown'):,} 行" if isinstance(split.get(name, {}).get("rows"), int) else f"{name} {split.get(name, {}).get('rows', 'unknown')} 行" for name in ("train", "validation", "test")]
    st.caption("固定分区：{}。分箱、特征筛选及模型拟合仅使用训练集；验证集用于阈值选择。".format(" / ".join(counts)))


def render_page(st, artifacts_root: str | Path | None = None) -> None:
    """Render the application using a Streamlit-compatible UI object."""
    root = Path(artifacts_root) if artifacts_root is not None else Path(__file__).resolve().parent / "artifacts"
    st.set_page_config(page_title="信用风险评分卡", page_icon="📊", layout="wide")
    _inject_design_tokens(st)
    runs = discover_model_runs(root)
    _render_hero(st, len(runs))
    if not runs:
        st.warning("没有找到可用的本地模型文件。请先运行共享模型管线并保存 baseline、scorecard 或 tree bundle。")
        return

    selector_columns = st.columns([1.7, 1])
    with selector_columns[0]:
        selected_run = st.selectbox(
            "实验批次",
            runs,
            format_func=lambda path: str(path.relative_to(root)),
        )
    models = available_models(selected_run)
    if not models:
        st.warning("所选实验目录中没有支持的模型 bundle。")
        return
    with selector_columns[1]:
        selected_model = st.selectbox(
            "客户评估模型",
            models,
            format_func=lambda name: MODEL_LABELS[name],
        )

    provenance_container = st.container(border=True)
    with provenance_container:
        _render_provenance(st, selected_run)
    _render_metrics(st, selected_run)
    if selected_model == "tree":
        _render_global_explanation(st, selected_run)

    st.divider()
    st.markdown('<div class="cr-section-kicker">Customer Assessment</div>', unsafe_allow_html=True)
    st.subheader("客户风险评估工作台")
    st.caption("目标事件：两年内发生 90 天或以上严重逾期（y=1）。本页面仅用于研究与演示，不执行授信审批。")
    submitted = False
    values = {}
    workspace_columns = st.columns([1.15, 0.85], gap="large")
    with workspace_columns[0]:
        with st.container(border=True):
            st.markdown("**客户基础信息与还款行为**")
            st.caption("录入 10 个原始特征。空白按模型策略处理；年龄 0 与逾期次数 96/98 遵循项目清洗规则。")
            with st.form("manual_prediction"):
                groups = (
                    ("信贷使用与负债", ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome")),
                    ("客户与信贷概况", ("age", "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "NumberOfDependents")),
                    ("历史逾期情况", ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse")),
                )
                for group_title, names in groups:
                    st.markdown(f"**{group_title}**")
                    columns = st.columns(2)
                    for position, name in enumerate(names):
                        values[name] = columns[position % len(columns)].text_input(
                            FEATURE_LABELS[name],
                            value="",
                            placeholder="输入有限数值；留空表示缺失",
                            key=f"manual_{name}",
                        )
                submitted = st.form_submit_button("运行客户评估")

    result = None
    features = None
    bundle = None
    error_message = None
    error_kind = None
    if submitted:
        try:
            features = parse_manual_input(values)
        except (TypeError, ValueError) as error:
            error_kind, error_message = "输入错误", str(error)
        else:
            try:
                bundle = load_model_bundle(selected_run, selected_model)
            except Exception as error:
                error_kind, error_message = "模型加载失败", str(error)
            else:
                try:
                    result = predict_with_bundle(selected_model, bundle, features)
                except Exception as error:
                    error_kind, error_message = "预测失败", str(error)

    with workspace_columns[1]:
        with st.container(border=True):
            st.markdown('<div class="cr-section-kicker">Assessment Output</div>', unsafe_allow_html=True)
            st.subheader("本次评估结果")
            if error_message:
                st.error(f"{error_kind}：{error_message}")
            elif result is None:
                st.info("填写左侧客户信息并运行评估后，结果将在此显示。")
            else:
                probability = float(result["probability"].iloc[0])
                st.metric("两年内严重逾期概率（y=1）", f"{probability:.2%}")
                st.progress(probability, text="预测概率（连续值；未设置风险等级阈值）")
                if selected_model == "scorecard":
                    import pandas as pd

                    score = float(result["score"].iloc[0])
                    st.metric("信用评分", f"{score:.1f}")
                    st.caption("评分卡方向：坏事件概率越高，信用评分越低。展示未取整分数。")
                    point_columns = [name for name in result if name.startswith("points_")]
                    details = [{"component": "base_points", "points": float(result["base_points"].iloc[0])}]
                    details.extend(
                        {"component": name, "points": float(result[name].iloc[0])}
                        for name in point_columns
                    )
                    st.markdown("**评分卡分项贡献**")
                    st.bar_chart(
                        pd.DataFrame(details).set_index("component"),
                        horizontal=True,
                        height=max(220, min(440, 34 * len(details))),
                        x_label="分数贡献",
                    )
                    st.dataframe(details, use_container_width=True)
                elif selected_model == "tree":
                    st.info("LightGBM 提供事件概率；信用评分仅由 WOE/IV 评分卡生成。")
                else:
                    st.info("当前 Baseline 接口提供事件概率，不生成评分卡信用分。")
                st.caption("页面使用离线预测接口，不使用测试集阈值对客户进行审批分类。")

        render_explanation_panel(
            st,
            selected_model,
            bundle=bundle,
            features=features,
            available=selected_model == "tree" and result is not None and error_message is None,
            expected_probability=None if result is None else result['probability'].to_numpy(),
        )


def main() -> None:
    """Start Streamlit with a friendly message when its UI dependency is absent."""
    try:
        import streamlit as st
    except ImportError as error:
        print(f"无法启动 Streamlit 页面：{error}。请使用项目环境安装 requirements.txt。")
        return
    try:
        parser = argparse.ArgumentParser(description='本地信贷模型研究页面')
        parser.add_argument('--artifacts-root', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
        args = parser.parse_args()
        render_page(st, args.artifacts_root)
    except ImportError as error:
        st.error(f"页面依赖不可用：{error}")


if __name__ == "__main__":
    main()
