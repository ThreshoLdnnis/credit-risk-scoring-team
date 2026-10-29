# 郭宇哲（morebadly）T03：简单 LR 与基础链路

关联任务：[#3](https://github.com/ThreshoLdnnis/credit-risk-scoring-team/issues/3)。模型主责为郭宇哲。交付包括 `src/baseline.py`、共享 `run_pipeline.py`、对应测试及本人实验、日志；根 README 仅补 T03 命令和进度。共享入口根据 Issue 允许范围及用户补齐 T03 的明确授权实现，原有三参数接口、数据接口、配置、依赖和同伴模块不变；合并仍由负责人统筹。

2026-10-28 已完成模型实验 [EXP-001](EXP-001/README.md)，并通过共享入口完成 [EXP-002 集成复现](EXP-002/README.md)。用户已授权提交仅包含 T03 的 PR；个人练习保留在本地，不纳入本次提交。

## 实现与接口

`fit_baseline(train_x, train_y, settings)` 只接收训练分区，输入来自 `clean_features` 的 13 列。`settings` 可为完整 `configs/experiment.json`，也可为包含 C、penalty、max_iter 和可选 seed 的 LR 字典。默认 C=1、L2、max_iter=2000、seed=42，使用 lbfgs。无法收敛时报错，不能把优化失败当作成功实验。

`bundle` 是可由 joblib 保存的字典：

| 键 | 内容 |
| --- | --- |
| pipeline | 已拟合的 SimpleImputer → StandardScaler → LogisticRegression |
| feature_columns | 保存的固定 13 列顺序 |
| settings | 实际参数、solver 和 seed |
| versions | Python、numpy、pandas、scikit-learn、joblib 版本 |
| positive_class | 1，即两年内 90 天及以上严重逾期 |

SimpleImputer 使用训练集中位数、`add_indicator=True`、`keep_empty_features=True`。训练全缺失列保留并以 0 填补，缺失指示保留。常量列按 StandardScaler 的单位缩放处理。仅训练中出现过缺失的列生成缺失指示；预测首次出现的缺失仍用训练中位数填补，不新增列或重新拟合。这遵循 [scikit-learn 1.5 SimpleImputer 文档](https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html)。

`predict_baseline(bundle, features)` 返回与输入行顺序一致的一维 y=1 概率。具名列可换序，内部按已保存顺序恢复；缺列、额外标签/序号列、重复列、非法文本、复数、无限值、空输入均拒绝。Series 标签的索引和行序必须精确一致。

## 数据与真实运行

指定输入仅为 [Kaggle Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) 官方有标签 `cs-training.csv`。仓库不含原始数据。先自行登录官方页面，按规则下载并将文件放入 `data/raw/`。`cs-test.csv` 没有真实标签，不能用于计算本次测试指标。

本次用户授权搜索其他平台后，从公开 GitHub 镜像取得文件，字节数、150,000 行/12 列以及 SHA256 均与 T02 官方原始文件相符；固定镜像版本和实际取得方式记录在 [EXP-001/record.json](EXP-001/record.json)。不将镜像下载写成本人从 Kaggle 下载。

正式运行校验文件 SHA256 与队友 T02 EXP-001 一致：`1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac`。不同字节版本会停止；应先核对来源与版本，不能删除检查或把其他数据冒充为官方实验。

在仓库根目录使用 Python 3.11 和团队依赖：

```powershell
$env:PYTHONUTF8 = '1'
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest -q tests
.venv\Scripts\python.exe run_pipeline.py
```

可指定 `--input`、`--output`、`--config`。默认在被 Git 忽略的 `artifacts/morebadly/T03-时间-随机编号/` 创建唯一目录；已有目录拒绝覆盖。个人原命令 `python experiments/morebadly/run_t03.py` 仍可运行，内部委托同一共享入口。仓库内部的本地结果必须放在 `artifacts/` 下。`--synthetic` 仅用于显式标注的人工数据工程验证，其状态为 `synthetic_engineering_check`，不能当作真实 Kaggle 成绩。

运行直接调用数据同伴已合入的 `load_dataset`、`clean_features`、`feature_group_ids`、`split_dataset`，保持内容组、70/15/15 和 seed=42 的分区。保留全部行和原始 y。分区证据使用原始行位置、导出序号、内容组和与 T02 相同定义的 SHA256 指纹。

| 本地产物 | 作用 |
| --- | --- |
| config.json | 共享配置的运行快照 |
| selection.json | 验证集阈值、参数及固定时间，在测试预测前保存 |
| baseline.joblib | 模型与完整预处理 bundle |
| split_positions.npz、split_members.csv | 复用分区及成员证据 |
| predictions.csv | 原始行顺序的概率、标签及分区，仅本地保存 |
| metrics.json | train、validation、test 汇总指标，训练区标为表观效果 |
| record.json | 数据/代码 SHA256、环境、Git 状态、参数、分区、指标、产物哈希与验证 |
| run.log、acquisition.json（如有） | 运行状态、时间和已核验的取得来源；配置变动导致失败时，日志和记录均标为失败 |

阈值仅在 validation 上最大化 TPR-FPR，限定有限 [0,1]，并列取较大值；冻结后评价 test，统一 `p>=threshold`。输出 AUC、KS、AP、梯形 PR 面积、Brier、Log loss 和混淆矩阵、Precision/Recall/F1。AP 与梯形 PR 面积分开命名。此处评价辅助函数服务 T03 的实验闭环，T05 再统一到 `src/evaluation.py`。

## 共享入口与后续集成

`run_pipeline.py` 已实现原有 `run_experiment(data_path, output_dir, settings)` 接口和 CLI，数据清洗到 LR 评价、保存证据的链路可直接运行；也可通过下列模型接口接入后续模型比较：

```python
from src.baseline import fit_baseline, predict_baseline

bundle = fit_baseline(features.iloc[positions['train']], y.iloc[positions['train']], settings)
probability = predict_baseline(bundle, features.iloc[positions['validation']])
```

保存后只从可信本地产物加载，使用相同版本依赖。入口逐分区验证 joblib 保存加载前后概率完全一致。当前 **124 项业务测试通过**，共享入口和个人入口的真实分区、概率 CSV 及全部指标一致。此次只交付 T03；评分卡、树模型、统一曲线、SHAP 和界面属于后续任务。个人练习按用户要求不提交；同伴审查、负责人合并和本人讲解分别验收。
