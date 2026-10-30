"""只检查 Python 和依赖，不训练模型或评价效果。"""

import argparse
import importlib
import json
import platform
import site
import sys
from importlib.metadata import version
from pathlib import Path


def collect_environment():
    """Import the pinned native dependencies; this alone is not model acceptance."""
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('请使用 Python 3.11 建立团队环境。')
    packages = {
        'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',
        'scikit-learn': 'sklearn', 'optbinning': 'optbinning',
        'lightgbm': 'lightgbm', 'shap': 'shap', 'streamlit': 'streamlit',
        'matplotlib': 'matplotlib', 'seaborn': 'seaborn',
        'pyarrow': 'pyarrow', 'joblib': 'joblib', 'pytest': 'pytest',
        'ortools': 'ortools', 'cvxpy': 'cvxpy', 'osqp': 'osqp', 'ruff': None,
    }
    installed = {}
    for package, module in packages.items():
        if module is not None:
            importlib.import_module(module)
        installed[package] = version(package)
    return {
        'scope': '依赖导入检查；模型、真实流程和页面另行验收',
        'python': platform.python_version(), 'platform': platform.platform(),
        'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'enable_user_site': site.ENABLE_USER_SITE,
        'packages': installed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/environment.json'))
    args = parser.parse_args()
    record = collect_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('依赖可导入。模型与页面验收请运行完整流程和业务测试。环境记录：', args.output)


if __name__ == '__main__':
    main()
