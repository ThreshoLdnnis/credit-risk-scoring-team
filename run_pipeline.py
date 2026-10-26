"""团队自行实现的数据到训练评价入口。当前尚不能运行实验。"""


def run_experiment(data_path, output_dir, settings: dict):
    """读取、清洗、切分、训练、验证选择、固定后评价并保存证据。"""
    raise NotImplementedError('TODO：先联通简单 LR，再接入评分卡和树模型')


def main():
    """由团队补充命令行参数和配置读取，输出目录每次实验唯一。"""
    raise NotImplementedError('TODO：由团队定义并实现 CLI 后，在 README 记录实际命令')


if __name__ == '__main__':
    main()
