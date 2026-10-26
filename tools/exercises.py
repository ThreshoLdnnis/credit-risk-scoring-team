"""创建或检查每位成员自己的六项练习目录。"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNING = ROOT / 'learning'


def personal_directory(username):
    """GitHub 用户名转小写，同一账号只有一个提交目录。"""
    username = username.lower()
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?', username):
        raise ValueError('请填写实际 GitHub 用户名，不使用空格、中文或路径。')
    return LEARNING / 'submissions' / username


def check(username, task=None):
    personal = personal_directory(username)
    if not (personal / 'exercises.py').is_file():
        raise ValueError('未找到本人答案；先执行 start 用户名。')
    env = os.environ.copy()
    env['CREDIT_RISK_EXERCISE_IMPL'] = 'exercises'
    env['PYTHONPATH'] = str(personal) + os.pathsep + str(ROOT)
    env['PYTHONIOENCODING'] = 'utf-8'
    command = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib',
               str(LEARNING / 'test_exercises.py'), str(personal)]
    if task:
        command += ['-k', task]
    return subprocess.call(command, cwd=ROOT, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ['start', 'check', 'demo']:
        child = actions.add_parser(name)
        child.add_argument('username')
        if name == 'check':
            child.add_argument('--task', choices=['missing', 'split', 'auc', 'woe', 'score', 'comparison'])
    actions.add_parser('check-all')
    args = parser.parse_args()
    try:
        if args.action == 'check-all':
            submissions = sorted(p.name for p in (LEARNING / 'submissions').iterdir()
                                 if p.is_dir() and not p.name.startswith(('.', '_')))
            if not submissions:
                print('尚无个人练习提交；没有成员被计为已通过。')
                return 0
            codes = [check(username) for username in submissions]
            return int(any(code != 0 for code in codes))
        personal = personal_directory(args.username)
        if args.action == 'start':
            if personal.exists():
                raise ValueError('本人目录已存在；保留现有实现，不重复覆盖。')
            personal.mkdir(parents=True)
            shutil.copy2(LEARNING / 'templates/exercises.py', personal / 'exercises.py')
            print('已创建 TODO：', personal.relative_to(ROOT))
            return 0
        if args.action == 'check':
            return check(args.username, args.task)
        if not (personal / 'exercises.py').is_file():
            raise ValueError('未找到本人答案；先执行 start 用户名。')
        env = os.environ.copy()
        env['CREDIT_RISK_EXERCISE_IMPL'] = 'exercises'
        env['PYTHONPATH'] = str(personal) + os.pathsep + str(ROOT)
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.call([sys.executable, str(LEARNING / 'run_demo.py')], cwd=ROOT, env=env)
        if result == 0:
            output = ROOT / 'artifacts' / personal.name / 'classroom-demo'
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LEARNING / 'output/demo_result.json', output / 'demo_result.json')
            print('人工合成演示已保存到：', output.relative_to(ROOT))
        return result
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
