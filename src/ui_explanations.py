"""Validate explanation output and associate batch explanations with one run."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def validate_explanation(result, features, expected_probability=None):
    """Refuse wrong rows, class, feature order, units or inconsistent numbers."""
    if not isinstance(result, Mapping):
        raise ValueError('解释接口必须返回映射')
    n, width = features.shape
    if result.get('feature_names') != list(features.columns):
        raise ValueError('解释特征顺序与当前输入不一致')
    if result.get('row_index_repr') != [repr(v) for v in features.index]:
        raise ValueError('解释行顺序与当前输入不一致')
    metadata = result.get('metadata', {})
    if not isinstance(metadata, Mapping) or metadata.get('model_output') != 'raw' or metadata.get('positive_class') != 1:
        raise ValueError('解释必须对应正类原始 log-odds')
    try:
        base, values, raw, probability = [np.asarray(result[key], dtype=float)
                                        for key in ('base_log_odds', 'shap_values', 'raw_log_odds', 'probability')]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('解释数值字段缺失或格式错误') from error
    if values.shape != (n, width) or any(vector.shape != (n,) for vector in (base, raw, probability)):
        raise ValueError('解释数值形状与当前输入不一致')
    if not all(np.isfinite(vector).all() for vector in (base, values, raw, probability)):
        raise ValueError('解释包含非有限数值')
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError('解释概率超出有效范围')
    if not np.allclose(base + values.sum(axis=1), raw, rtol=0, atol=1e-6):
        raise ValueError('基础值与贡献之和不等于模型输出')
    if not np.allclose(expit(raw), probability, rtol=0, atol=1e-10):
        raise ValueError('解释的模型输出与概率不一致')
    if expected_probability is not None:
        expected = np.asarray(expected_probability, dtype=float)
        if expected.shape != (n,) or not np.allclose(expected, probability, rtol=0, atol=1e-10):
            raise ValueError('解释概率与页面预测概率不一致')
    return dict(result)


def feature_label(name, labels):
    original, marker, _ = name.partition('__special')
    return labels.get(original, original) + ('（特殊值标记）' if marker else '')


def local_contributions(result, labels, top_n=8):
    """Top contributions plus a signed remainder; never drop hidden terms."""
    values = np.asarray(result['shap_values'], dtype=float)[0]
    names = result['feature_names']
    selected = np.argsort(-np.abs(values), kind='stable')[:top_n]
    omitted = np.setdiff1d(np.arange(len(names)), selected)
    terms = [(feature_label(names[i], labels), float(values[i])) for i in selected]
    if len(omitted):
        terms.append((f'其余 {len(omitted)} 项（合计）', float(values[omitted].sum())))
    current = float(result['base_log_odds'][0])
    rows = []
    for label, value in terms:
        following = current + value
        rows.append({'因素': label, '贡献': value, '起点': min(current, following),
                     '终点': max(current, following), '方向': '增加严重逾期倾向' if value >= 0 else '降低严重逾期倾向'})
        current = following
    if not np.isclose(current, result['raw_log_odds'][0], rtol=0, atol=1e-6):
        raise ValueError('展示贡献未完整还原模型输出')
    return rows


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_global_explanation(run_dir):
    """Only show batch SHAP tied to this run record and its actual tree file."""
    directory = Path(run_dir)
    summary_path = directory / 'explanations' / 'summary.json'
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    record_path = directory / 'record.json'
    record = json.loads(record_path.read_text(encoding='utf8'))
    if summary.get('status') not in {
        'completed_fixed_model_explanation_and_reproduction_audit',
        'completed_audit_with_reloaded_numeric_differences',
    }:
        raise ValueError('批量解释未完成核验')
    if summary.get('pipeline_record_sha256') != file_sha256(record_path):
        raise ValueError('全局解释属于其他实验记录')
    if record.get('output_sha256', {}).get('tree.joblib') != file_sha256(directory / 'tree.joblib'):
        raise ValueError('当前树模型与实验记录不一致')
    table_path = summary_path.parent / 'global_importance.csv'
    if summary.get('output_sha256', {}).get('global_importance.csv') != file_sha256(table_path):
        raise ValueError('全局解释汇总文件校验失败')
    from src.data import CLEAN_FEATURE_COLUMNS

    table = pd.read_csv(table_path)
    if (list(table.columns) != ['feature', 'mean_abs_shap_log_odds'] or table['feature'].duplicated().any()
            or set(table['feature']) != set(CLEAN_FEATURE_COLUMNS)):
        raise ValueError('全局解释的特征集合不符合约定')
    values = table['mean_abs_shap_log_odds'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('全局解释重要性无效')
    if summary.get('explanation', {}).get('best_iteration') != record['models']['tree']['best_iteration']:
        raise ValueError('全局解释与模型有效轮数不一致')
    return {'summary': summary, 'importance': table}
