"""数据成员实现；所有函数当前均为 TODO。"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: str | Path, labeled: bool = True) -> pd.DataFrame:
    """读取官方 CSV、校验字段和数值，返回原始行序的表。"""
    raise NotImplementedError('TODO：读取与校验；有标签模式必须有有效 0/1 标签')


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """只返回清洗后的模型特征，排除标签与序号；保留行序。"""
    raise NotImplementedError('TODO：无状态规则清洗，不拟合填补和截尾参数')


def split_dataset(features: pd.DataFrame, y, seed: int = 42) -> dict[str, np.ndarray]:
    """按清洗后全部输入分组分层，返回 train/validation/test 行位置。"""
    raise NotImplementedError('TODO：参考 E02；同组不跨区，所有行恰好覆盖一次')
