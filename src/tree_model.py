"""LightGBM 对照模型，Day 4 实现。"""


def fit_tree(train_x, train_y, validation_x, validation_y, settings: dict):
    """只用训练与验证做拟合、早停，返回模型及必要预处理。"""
    raise NotImplementedError('TODO：训练内处理，验证 AUC 早停，记录实际有效轮数')


def predict_tree(bundle, features):
    """返回与输入同序的正类概率，使用保存的有效迭代与变换。"""
    raise NotImplementedError('TODO：可复现的树模型推理')
