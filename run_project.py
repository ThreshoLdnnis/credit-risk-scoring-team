"""一条命令完成三模型训练、独立数据审计及全局/局部 SHAP 产物。"""

import argparse
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from check_environment import collect_environment
from experiments.morebadly.verify_and_explain import verify
from run_pipeline import EXPECTED_SOURCE_SHA256, run_from_config, sha256, write_json
from src.ui_explanations import load_global_explanation

ROOT = Path(__file__).resolve().parent


def _data_auditor():
    path = ROOT / 'experiments/Sally-yi/check_data_reproduction.py'
    spec = importlib.util.spec_from_file_location('team_independent_data_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit


def delivery_sources():
    names = ['run_project.py', 'run_pipeline.py', 'app.py', 'check_environment.py',
             'configs/experiment.json', 'requirements.txt', 'constraints.txt',
             'experiments/Sally-yi/check_data_reproduction.py',
             'experiments/morebadly/verify_and_explain.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'src').glob('*.py'))]
    return {name: sha256(ROOT / name) for name in names}


def run_complete(input_path, output, config_path=None, sample_size=1000):
    """A completed manifest is written only after every independent check passes."""
    source, output = Path(input_path).resolve(), Path(output).resolve()
    config = Path(config_path or ROOT / 'configs/experiment.json').resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('逐行产物请保存在 artifacts/ 或仓库外的本地目录')
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError('数据 SHA256 与团队指定版本不一致')
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError('解释样本数必须为正整数')
    code_hashes = delivery_sources()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    step = 'environment'
    try:
        environment = collect_environment()
        write_json(output / 'environment.json', environment)
        step = 'three_model_pipeline'
        model_dir = output / 'models'
        record = run_from_config(source, model_dir, config, models=['baseline', 'scorecard', 'tree'])
        step = 'independent_data_audit'
        data_audit = _data_auditor()(source, model_dir)
        write_json(output / 'data_audit.json', data_audit)
        step = 'model_reproduction_and_shap'
        explanation = verify(model_dir, source, model_dir / 'explanations', sample_size=sample_size)
        step = 'page_artifact_association'
        if load_global_explanation(model_dir) is None:
            raise RuntimeError('页面无法找到本实验的全局解释')
        if delivery_sources() != code_hashes or sha256(source) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError('运行期间源码或原始数据发生变化')
        manifest = {
            'status': 'completed_team_technical_delivery', 'started_at': started,
            'completed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'source_sha256': EXPECTED_SOURCE_SHA256, 'code_sha256': code_hashes,
            'environment': environment, 'selected_models': record['selected_models'],
            'split': record['split'], 'data_audit_status': data_audit['status'],
            'test_metrics': {name: result['metrics']['test'] for name, result in record['models'].items()},
            'model_reproduction_status': explanation['status'],
            'models_with_reloaded_differences': explanation['models_with_reloaded_differences'],
            'explanation': explanation['explanation'], 'explanation_sample': explanation['sample'],
            'pipeline_record_sha256': sha256(model_dir / 'record.json'),
            'output_sha256': {p.relative_to(output).as_posix(): sha256(p)
                              for p in sorted(output.rglob('*')) if p.is_file()},
            'human_acceptance': '成员现场讲解、小改和最终认可需由本人实际完成',
            'limitations': ['公开数据的内部研究分区；不代表时间外、客户外或实际业务效果。',
                            '模型重载数值差异详见解释复验记录；不改写历史评价或冻结阈值。'],
        }
        write_json(output / 'delivery.json', manifest)
        return manifest
    except Exception as error:
        write_json(output / 'delivery.json', {'status': 'failed', 'failed_step': step,
                   'started_at': started, 'error': f'{type(error).__name__}: {error}'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cs-training.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-size', type=int, default=1000)
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/team' / ('complete-' + uuid4().hex[:8])
    try:
        result = run_complete(args.input, output, args.config, args.sample_size)
    except Exception as error:
        parser.exit(1, f'完整运行失败：{error}\n')
    print(json.dumps({'output': str(output.resolve()), 'status': result['status'],
                      'model_reproduction_status': result['model_reproduction_status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
