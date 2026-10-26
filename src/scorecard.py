"""训练内监督分箱、WOE、LR 与信用评分，Day 3 实现。"""


def fit_scorecard(train_x, train_y, settings: dict):
    """返回分箱、变量顺序、IV 选择、LR、尺度和逐箱证据组成的 bundle。"""
    raise NotImplementedError('TODO：训练内分箱及 IV，保存完整的推理依赖')


def predict_scorecard(bundle, features):
    """返回对齐的 DataFrame：probability、score、base_points 及各 points_列。"""
    raise NotImplementedError('TODO：计算概率、总分与分项并检查加和')
