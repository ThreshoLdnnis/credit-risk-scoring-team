"""验证阈值、统一评价和曲线，Day 4 实现。"""


def select_threshold(validation_y, validation_probability) -> float:
    """验证 ROC 上最大化 TPR-FPR；有限合法阈值并列时取较大值。"""
    raise NotImplementedError('TODO：只接收验证数据，排除 inf 和区间外阈值')


def evaluate_binary(y, probability, threshold: float) -> dict:
    """输出 AUC、KS、AP、PR 梯形面积、Brier、Log loss 和固定阈值指标。"""
    raise NotImplementedError('TODO：检查标签概率及样本对齐，统一 p>=threshold')


def save_evaluation_figures(y, predictions: dict, output_dir):
    """保存同分区的 ROC、PR、校准图；信用分分布由评分卡输出补画。"""
    raise NotImplementedError('TODO：真实数据的标准库指标与标明范围的图表')
