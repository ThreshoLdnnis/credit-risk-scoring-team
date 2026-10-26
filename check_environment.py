"""只检查 Python 和依赖，不训练模型或评价效果。"""

import argparse
import importlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/environment.json'))
    args = parser.parse_args()
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
    record = {
        'scope': '环境检查；业务模块尚待实现，不代表模型或系统验收通过',
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': installed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('依赖可导入；业务模块仍需实现。环境记录：', args.output)


if __name__ == '__main__':
    main()
