"""简单逻辑回归起点，Day 2 实现。"""


def fit_baseline(train_x, train_y, settings: dict):
    """训练内中位数填补、缺失指示、标准化和 L2 LR，返回整体推理对象。"""
    raise NotImplementedError('TODO：构建并拟合训练内预处理与 LR')


def predict_baseline(bundle, features):
    """返回与输入行逐一对齐的 y=1 概率，不在推理时重新 fit。"""
    raise NotImplementedError('TODO：复用保存的预处理并取得正类概率')
