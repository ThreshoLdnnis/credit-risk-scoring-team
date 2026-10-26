"""检查实际交付边界与配置；不代表模型或完整系统验收。"""

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        files = [ROOT / p for p in result.stdout.decode('utf-8').split('\0') if p]
    else:
        excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache'}
        files = [p for p in ROOT.rglob('*') if p.is_file()
                 and not excluded.intersection(p.relative_to(ROOT).parts)]
    problems = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            problems.append(f'{relative}：已跟踪文件缺失')
            continue
        if path.name in {'kaggle.json', 'cs-training.csv', 'cs-test.csv', 'reference_solution.py'}:
            problems.append(f'{relative}：不应提交的凭证、原始数据或负责人答案')
        if path.name.startswith('.env') and path.name != '.env.example':
            problems.append(f'{relative}：环境凭证文件不提交')
        if path.suffix.lower() in {'.joblib', '.pkl', '.pickle', '.parquet'}:
            problems.append(f'{relative}：训练产物或行级数据应留在本地')
        if relative.startswith(('data/', 'artifacts/')) and path.name != '.gitkeep':
            problems.append(f'{relative}：本地数据或训练产物不提交')
        if path.stat().st_size > 5 * 1024 * 1024:
            problems.append(f'{relative}：超过 5 MiB，先与负责人确认共享方式')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8-sig')):
                if re.match(r'^(https?:|#|mailto:)', target):
                    continue
                target = target.split('#')[0]
                if target and not (path.parent / target).exists():
                    problems.append(f'{relative}：相对链接不存在 {target}')
    settings = json.loads((ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))
    split = settings['split']
    if abs(sum(split[key] for key in ['train', 'validation', 'test']) - 1) > 1e-9:
        problems.append('训练/验证/测试比例之和须为 1')
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'仓库边界、语法、JSON 与相对链接检查通过：{len(files)} 个文件。')
    print('检查范围为交付结构；待实现模型与业务测试另行验收。')


if __name__ == '__main__':
    main()
