"""统一运行简单 LR 与 WOE/IV 评分卡，共用固定分区和验证集阈值协议。

默认保持 T03 Baseline 行为；--models baseline scorecard 同时运行两套模型。
训练、模型保存加载、评价及逐行证据均留在本地 artifacts/。
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.baseline import fit_baseline, predict_baseline  # noqa: E402
from src.data import (  # noqa: E402
    TARGET_COLUMN,
    clean_features,
    feature_group_ids,
    load_dataset,
    split_dataset,
)
from src.scorecard import fit_scorecard, predict_scorecard

ROOT = Path(__file__).resolve().parent

# Same bytes as the official download recorded by Sally-yi in T02 EXP-001.
EXPECTED_SOURCE_SHA256 = '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git_value(*args):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT,
        encoding='utf-8', timeout=30,
    ).strip()


def validation_threshold(y, probability):
    """T03 experiment helper; T05 will own the shared evaluation module."""
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates, gains = thresholds[valid], (tpr - fpr)[valid]
    return float(candidates[gains == gains.max()].max())


def metrics(y, probability, threshold):
    fpr, tpr, _ = roc_curve(y, probability)
    precision_curve, recall_curve, _ = precision_recall_curve(y, probability)
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average='binary', zero_division=0,
    )
    return {
        'rows': len(y), 'positive': int(np.sum(y)), 'auc': float(roc_auc_score(y, probability)),
        'ks': float(np.max(tpr - fpr)), 'ap': float(average_precision_score(y, probability)),
        'pr_auc_trapezoidal': float(auc(recall_curve, precision_curve)),
        'brier': float(brier_score_loss(y, probability)), 'log_loss': float(log_loss(y, probability)),
        'threshold': threshold, 'decision_rule': 'probability >= threshold',
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
    }


def fingerprints(features, groups):
    """Match T02's ordered-column / float.hex / null fingerprint definition."""
    lookup = {}
    for pos in np.unique(groups, return_index=True)[1]:
        values = [None if pd.isna(v) else float(0 if v == 0 else v).hex() for v in features.iloc[pos]]
        payload = json.dumps([list(features.columns), values], separators=(',', ':'))
        lookup[int(groups[pos])] = hashlib.sha256(payload.encode()).hexdigest()
    if len(set(lookup.values())) != len(lookup):
        raise RuntimeError('Group fingerprint collision')
    return [lookup[int(group)] for group in groups]


def _selected_models(models):
    """固定模型顺序，拒绝空选择、重复及尚未实现的路线。"""
    supported = ('baseline', 'scorecard')
    if not isinstance(models, (tuple, list)) or not models:
        raise ValueError('models must be a nonempty list or tuple of baseline/scorecard')
    if any(name not in supported for name in models) or len(set(models)) != len(models):
        raise ValueError('models accepts unique baseline/scorecard names only')
    return tuple(name for name in supported if name in models)


def _predict_model(name, bundle, features):
    """统一为保留输入索引的表；评分卡保留概率、信用分及全部分项。"""
    if name == 'baseline':
        result = pd.DataFrame({'probability': predict_baseline(bundle, features)}, index=features.index)
    else:
        result = predict_scorecard(bundle, features)
    if not isinstance(result, pd.DataFrame) or not result.index.equals(features.index):
        raise RuntimeError(f'{name} predictions do not preserve input row order')
    if 'probability' not in result or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise RuntimeError(f'{name} predictions contain invalid values')
    if not result['probability'].between(0, 1).all():
        raise RuntimeError(f'{name} probabilities are outside [0, 1]')
    return result


def _model_details(name, bundle, features):
    """保存各自的预处理契约，避免将两种 bundle 结构混用。"""
    if name == 'baseline':
        imputer = bundle['pipeline'].named_steps['imputer']
        classifier = bundle['pipeline'].named_steps['classifier']
        preprocessing = {
            'fit_partition': 'train', 'imputation': 'median; all-missing training columns use zero',
            'missing_indicators': [features.columns[i] for i in imputer.indicator_.features_],
            'median_values': dict(zip(features.columns, imputer.statistics_.tolist())),
            'scaler_mean': bundle['pipeline'].named_steps['scaler'].mean_.tolist(),
            'transformed_columns': imputer.get_feature_names_out().tolist(),
        }
        model_type = 'simple_l2_lr'
    else:
        classifier = bundle['logistic_regression']
        preprocessing = {
            'fit_partition': 'train', 'binning': 'native OptBinning WOE/IV',
            'input_features': bundle['input_features'],
            'selected_features': bundle['selected_features'],
            'iv_summary': bundle['iv_summary'], 'score_scale': bundle['score_scale'],
            'missing_policy': {k: v['missing_policy'] for k, v in bundle['binning_evidence'].items()},
        }
        model_type = 'woe_iv_logistic_scorecard'
    return {
        'model': model_type, 'settings': bundle['settings'],
        'iterations': classifier.n_iter_.tolist(), 'parameters': classifier.get_params(),
        'preprocessing': preprocessing,
    }


def run_experiment(data_path, output_dir, settings: dict, *, synthetic=False, models=('baseline',)):
    """共用一次数据切分运行选定模型；原有三参数调用默认仍运行 Baseline。

    两套模型均只在训练集拟合。所有模型的验证阈值在任何测试预测前统一
    保存。metrics/model/preprocessing 保留首个模型的 T03 兼容视图；
    models 与 model_metrics.json 是完整的多模型结果。
    """
    input_path, output_dir = Path(data_path).resolve(), Path(output_dir).resolve()
    chosen_models = _selected_models(models)
    primary = chosen_models[0]
    if not isinstance(settings, dict):
        raise ValueError('settings must be a dict')
    config = json.loads(json.dumps(settings, allow_nan=False))
    if 'scorecard' in chosen_models and not isinstance(config.get('scorecard'), dict):
        raise ValueError('Scorecard route requires the scorecard settings dictionary')
    if not input_path.is_file():
        raise FileNotFoundError('Download official cs-training.csv from Kaggle into data/raw/ first: ' + str(input_path))
    source_hash = sha256(input_path)
    if not synthetic and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError('CSV SHA256 differs from the official data version recorded in T02; verify the source before training')
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Local models and row-level evidence must stay under artifacts/')
    if type(config.get('seed')) is not int or config['seed'] != 42:
        raise ValueError('T03/T04 must reuse the shared seed=42 partition')
    if config.get('split') != {'train': 0.7, 'validation': 0.15, 'test': 0.15, 'unit': 'cleaned_feature_group'}:
        raise ValueError('T03/T04 require the shared 70/15/15 cleaned-feature-group split')
    if config.get('threshold') != {
        'selection_data': 'validation', 'method': 'max_tpr_minus_fpr',
        'tie_break': 'largest_finite_threshold_in_0_1',
    }:
        raise ValueError('T03/T04 require the agreed validation-only threshold rule')
    acquisition = None
    source_metadata = input_path.with_suffix('.source.json')
    if source_metadata.is_file():
        acquisition = json.loads(source_metadata.read_text(encoding='utf-8'))
        if not isinstance(acquisition, dict) or acquisition.get('sha256') != source_hash:
            raise ValueError('Acquisition metadata does not match the input file SHA256')
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    if acquisition is not None:
        write_json(output_dir / 'acquisition.json', acquisition)
    code_paths = ['src/data.py', 'src/baseline.py', 'run_pipeline.py', 'experiments/morebadly/run_t03.py']
    if 'scorecard' in chosen_models:
        code_paths.append('src/scorecard.py')
    source_code_hashes = {p: sha256(ROOT / p) for p in code_paths}
    write_json(output_dir / 'config.json', config)
    config_hash = sha256(output_dir / 'config.json')
    raw = load_dataset(input_path)
    features, y = clean_features(raw), raw[TARGET_COLUMN]
    positions = split_dataset(features, y, seed=config['seed'])
    groups = feature_group_ids(features)
    group_sets = {name: set(groups[pos]) for name, pos in positions.items()}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in [
        ('train', 'validation'), ('train', 'test'), ('validation', 'test'),
    ])
    if overlap or not np.array_equal(np.sort(np.concatenate(list(positions.values()))), np.arange(len(raw))):
        raise RuntimeError('Split group isolation or row coverage failed')
    repeated = split_dataset(features, y, seed=config['seed'])
    if not all(np.array_equal(positions[name], repeated[name]) for name in positions):
        raise RuntimeError('Split is not reproducible')

    bundles, frames, selections, details = {}, {}, {}, {}
    for name in chosen_models:
        train_pos, validation_pos = positions['train'], positions['validation']
        train_x, train_y = features.iloc[train_pos], y.iloc[train_pos]
        bundle = (fit_baseline(train_x, train_y, config) if name == 'baseline'
                  else fit_scorecard(train_x, train_y, config['scorecard']))
        bundles[name] = bundle
        details[name] = _model_details(name, bundle, features)
        frames[name] = {
            part: _predict_model(name, bundle, features.iloc[positions[part]])
            for part in ('train', 'validation')
        }
        threshold = validation_threshold(
            y.iloc[validation_pos], frames[name]['validation']['probability'].to_numpy()
        )
        selections[name] = {
            'fixed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'model': details[name]['model'], 'settings': bundle['settings'],
            'threshold': threshold, 'source': 'validation', 'test_used_for_selection': False,
        }
    # Freeze every model's decision before the first test prediction.
    write_json(output_dir / 'selections.json', selections)
    write_json(output_dir / 'selection.json', selections[primary])
    for name in chosen_models:
        frames[name]['test'] = _predict_model(name, bundles[name], features.iloc[positions['test']])

    roundtrip_errors = {}
    for name in chosen_models:
        joblib.dump(bundles[name], output_dir / f'{name}.joblib')
        restored = joblib.load(output_dir / f'{name}.joblib')
        maximum = 0.0
        for part, pos in positions.items():
            before = frames[name][part]
            after = _predict_model(name, restored, features.iloc[pos])
            maximum = max(maximum, float(np.max(np.abs(before.to_numpy() - after.to_numpy()))))
            if not before.equals(after):
                raise RuntimeError(f'Saved/loaded {name} changed predictions or score components')
        roundtrip_errors[name] = maximum

    all_metrics = {
        name: {
            part: metrics(y.iloc[pos], frames[name][part]['probability'].to_numpy(), selections[name]['threshold'])
            for part, pos in positions.items()
        }
        for name in chosen_models
    }
    write_json(output_dir / 'metrics.json', all_metrics[primary])
    write_json(output_dir / 'model_metrics.json', all_metrics)
    pd.DataFrame([
        {'model': name, 'model_type': details[name]['model'], 'partition': part, **values}
        for name in chosen_models for part, values in all_metrics[name].items()
    ]).to_csv(output_dir / 'model_comparison.csv', index=False)
    partition = np.full(len(raw), '', dtype=object)
    split_summary = {}
    for part, pos in positions.items():
        partition[pos] = part
        split_summary[part] = {
            'rows': len(pos), 'groups': len(group_sets[part]),
            'positive': int(y.iloc[pos].sum()), 'positive_rate': float(y.iloc[pos].mean()),
        }
    evidence = pd.DataFrame({
        'row_position': np.arange(len(raw)),
        'raw_row_id': raw['Unnamed: 0'].to_numpy() if 'Unnamed: 0' in raw else [None] * len(raw),
        'group_id': groups, 'group_fingerprint': fingerprints(features, groups), 'partition': partition,
    })
    evidence.to_csv(output_dir / 'split_members.csv', index=False)
    for name in chosen_models:
        combined = pd.DataFrame(index=features.index, columns=frames[name]['train'].columns, dtype=float)
        for part, pos in positions.items():
            combined.iloc[pos] = frames[name][part].to_numpy()
        prediction_table = pd.concat([evidence.assign(y=y.to_numpy()), combined.reset_index(drop=True)], axis=1)
        prediction_table.to_csv(output_dir / f'{name}_predictions.csv', index=False)
        if name == primary:
            evidence.assign(y=y.to_numpy(), probability=combined['probability'].to_numpy()).to_csv(
                output_dir / 'predictions.csv', index=False
            )
    if 'scorecard' in chosen_models:
        card = bundles['scorecard']
        write_json(output_dir / 'scorecard_binning.json', {
            'input_features': card['input_features'], 'selected_features': card['selected_features'],
            'settings': card['settings'], 'score_scale': card['score_scale'],
            'iv_summary': card['iv_summary'], 'binning_evidence': card['binning_evidence'],
        })
    np.savez_compressed(output_dir / 'split_positions.npz', **positions)
    if sha256(input_path) != source_hash or sha256(output_dir / 'config.json') != config_hash or any(
        sha256(ROOT / p) != value for p, value in source_code_hashes.items()
    ):
        raise RuntimeError('Data, config or source code changed during the experiment')
    status = 'synthetic_engineering_check' if synthetic else 'completed_official_data_experiment'
    max_difference = max(roundtrip_errors.values())
    write_json(output_dir / 'run.log', {
        'started_at': started_at, 'completed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': status, 'models': list(chosen_models), 'model': details[primary]['model'],
        'rows': len(raw), 'group_overlap': overlap,
        'saved_loaded_max_probability_difference': max_difference,
    })
    model_records = {
        name: {
            **selections[name], 'iterations': details[name]['iterations'],
            'parameters': details[name]['parameters'], 'preprocessing': details[name]['preprocessing'],
            'metrics': all_metrics[name], 'artifact': f'{name}.joblib',
            'predictions': f'{name}_predictions.csv',
            'saved_loaded_max_output_difference': roundtrip_errors[name],
        }
        for name in chosen_models
    }
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']
    if 'scorecard' in chosen_models:
        packages.append('optbinning')
    record = {
        'status': status,
        'executor': 'Team shared entry; personal reproduction and explanation recorded separately',
        'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source': {'file': input_path.name, 'sha256': source_hash, 'rows': len(raw),
                   'official_url': None if synthetic else 'https://www.kaggle.com/competitions/GiveMeSomeCredit/data',
                   'acquisition': acquisition},
        'target': {'column': TARGET_COLUMN, 'positive_class': 1, 'meaning': '90+ day serious delinquency within two years'},
        'feature_columns': list(features.columns), 'split': split_summary,
        'split_settings': {**config['split'], 'seed': config['seed'], 'stratification': 'group_max_y'},
        'audit': {'all_rows_once': True, 'group_overlap': overlap, 'seed_reproducible': True,
                  'saved_loaded_max_probability_difference': max_difference,
                  'all_models_frozen_before_test_prediction': True},
        'selected_models': list(chosen_models), 'models': model_records,
        'preprocessing': details[primary]['preprocessing'],
        'model': {**selections[primary], 'iterations': details[primary]['iterations'],
                  'parameters': details[primary]['parameters']},
        'metrics': all_metrics[primary],
        'environment': {'python': platform.python_version(), 'packages': {p: version(p) for p in packages}},
        'git': {'commit': git_value('rev-parse', 'HEAD'), 'branch': git_value('branch', '--show-current'),
                'working_tree_changes': git_value('status', '--short')},
        'code_sha256': source_code_hashes, 'config_sha256': config_hash,
        'entry_point': 'run_pipeline.run_experiment', 'command': None,
        'reproduction_command': ['python', 'run_pipeline.py', '--input', str(input_path),
                                 '--output', '<new-output-directory>', '--config',
                                 str(output_dir / 'config.json'), '--models', *chosen_models]
                                + (['--synthetic'] if synthetic else []),
        'output_sha256': {p.name: sha256(p) for p in output_dir.iterdir() if p.is_file()},
        'limitations': [
            'Internal research partitions, not out-of-time or customer-external validation.',
            'Training metrics are apparent performance; no tuning used test metrics.',
            'LightGBM, shared evaluation plots, SHAP and UI remain later tasks.',
        ],
    }
    write_json(output_dir / 'record.json', record)
    return record


def run_from_config(data_path, output_dir, config_path=ROOT / 'configs/experiment.json', *,
                    synthetic=False, entry_script='run_pipeline.py', models=('baseline',)):
    """读取一次配置，保存原始哈希，并验证运行期间没有改变。"""
    config_path = Path(config_path).resolve()
    source_bytes = config_path.read_bytes()
    original_hash = hashlib.sha256(source_bytes).hexdigest()
    settings = json.loads(source_bytes.decode('utf-8'))
    record = run_experiment(data_path, output_dir, settings, synthetic=synthetic, models=models)
    record['config_source'] = {'path': str(config_path), 'sha256': original_hash}
    if not config_path.is_file() or sha256(config_path) != original_hash:
        record['status'] = 'failed_config_provenance'
        record['error'] = 'Source config changed during the experiment'
        write_json(Path(output_dir) / 'run.log', {'status': record['status'], 'error': record['error']})
        record['output_sha256']['run.log'] = sha256(Path(output_dir) / 'run.log')
        write_json(Path(output_dir) / 'record.json', record)
        raise RuntimeError(record['error'])
    record['command'] = ['python', entry_script, '--input', str(Path(data_path).resolve()),
                         '--output', str(Path(output_dir).resolve()), '--config', str(config_path),
                         '--models', *record['selected_models']]
    if synthetic:
        record['command'].append('--synthetic')
    write_json(Path(output_dir) / 'record.json', record)
    return record


def main(entry_script='run_pipeline.py'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.json')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'scorecard'], default=['baseline'],
                        help='选择 baseline、scorecard，或同时选择两者；默认 baseline')
    parser.add_argument('--synthetic', action='store_true', help='Engineering check only; never counts as official data results')
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/team' / (
        'T03-T04-' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8]
    )
    try:
        record = run_from_config(args.input, output, args.config, synthetic=args.synthetic,
                                 entry_script=entry_script, models=args.models)
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        parser.exit(1, f'T03/T04 failed: {error}\n')
    print(json.dumps({
        'output': str(output.resolve()), 'status': record['status'], 'metrics': record['metrics'],
        'models': {name: value['metrics'] for name, value in record['models'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
