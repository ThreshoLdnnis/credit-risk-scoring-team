# T05：LightGBM 与统一评价

郭宇哲（morebadly），关联 [T05 / #5](https://github.com/ThreshoLdnnis/credit-risk-scoring-team/issues/5)。2026-10-29 实际使用指定 `cs-training.csv` 运行。C 模板为 [record.json](record.json)，完整汇总、源码与产物哈希为 [run_manifest.json](run_manifest.json)，复核为 [audit.json](audit.json)，全部模型/分区指标为 [model_comparison.csv](model_comparison.csv)。原始数据、训练模型和逐行预测只在本地忽略目录。

## 结果与验收

三模型使用相同清洗结果和固定分组分区：train=104,974、validation=22,518、test=22,508 行，seed=42、组间交集为 0。全部模型与验证阈值固定后才作测试预测，没有根据测试成绩调整参数。LightGBM 验证 AUC 早停，最多 500 轮、耐心 30 轮，实际评价 287 轮，**最佳有效轮数 257**，最佳验证 AUC=0.8623491117，后续推理明确使用该轮数。

同一测试分区的结果（正类为 `SeriousDlqin2yrs=1`）：

| 模型 | AUC | KS | AP | 梯形 PR 面积 | Brier | Log loss | 验证阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 简单 LR | 0.823521 | 0.506524 | 0.352951 | 0.352391 | 0.052596 | 0.197743 | 0.0629705016 |
| 评分卡 | 0.855528 | 0.545791 | 0.372150 | 0.371930 | 0.050385 | 0.183416 | 0.0722315760 |
| LightGBM | 0.871837 | 0.577973 | 0.411703 | 0.411211 | 0.048351 | 0.174750 | 0.0667331946 |

| 模型 | TN | FP | FN | TP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 简单 LR | 17090 | 3917 | 484 | 1017 | 0.206121 | 0.677548 | 0.316084 |
| 评分卡 | 16897 | 4110 | 396 | 1105 | 0.211889 | 0.736176 | 0.329065 |
| LightGBM | 16716 | 4291 | 336 | 1165 | 0.213526 | 0.776149 | 0.334914 |

阈值只取验证集最大 TPR-FPR 的有限 [0,1] 候选，并列取较大阈值，预测规则为 `p>=threshold`；Precision/Recall/F1 零分母取 0。KS 使用有符号最大 TPR-FPR，不取绝对值。AP 与梯形 PR 面积分别报告；Log loss 使用 sklearn 的标准数值端点处理。

## 数据和处理

数据文件 SHA256：`1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac`。本机沿用 T03 时经用户授权取得的固定 GitHub 镜像，与 T02 官方下载哈希相同；取得方式和固定 URL 见 manifest，未声称本次直接从 Kaggle 下载。

原样调用主分支 `src/data.py`：排除序号与标签、age=0 置缺失、三项逾期次数 96/98 置缺失并加特殊值指示，共 13 个输入列；不删重复行、不任意裁剪。按清洗后特征内容绑定同组切分，原始逐行标签保留。LightGBM 原生处理 NaN，训练数据决定分箱；无填补、标准化或类别加权。保留全缺失/常量列的字段契约，推理时按训练字段顺序取列。

## 接口与复现

在仓库根目录、按 requirements/constraints 安装的环境中执行，输出必须是新目录：

```powershell
python experiments/morebadly/run_t05.py --input data/raw/cs-training.csv --output artifacts/morebadly/T05-REPRO-001
python -m pytest -q tests
```

脚本使用共享 `configs/experiment.json`，一次清洗/切分后依次调用现有 LR、评分卡和本次树模型；全部验证决策保存后才评价 test。预测和评分分项保存加载后逐值一致。输出包括配置、取得信息、数据/源码/产物哈希、有效轮数和验证 AUC 历史、全部指标、模型、逐行预测、分区原始行标识/内容指纹、8 图及成功日志。数据哈希或协议不符会报错，已有目录拒绝覆盖。

供负责人接入共享入口的已实现接口：

```python
from src.tree_model import fit_tree, predict_tree
from src.evaluation import select_threshold, evaluate_binary, save_evaluation_figures

bundle = fit_tree(train_x, train_y, validation_x, validation_y, settings)
validation_p = predict_tree(bundle, validation_x)
threshold = select_threshold(validation_y, validation_p)
# 所有模型和验证阈值固定后才进入 test。
test_p = predict_tree(bundle, test_x)
metrics = evaluate_binary(test_y, test_p, threshold)
paths = save_evaluation_figures(test_y, predictions, output_dir, partition='test')
```

`bundle` 保存 model、feature_columns、settings、preprocessing、best_iteration、validation_auc_at_best/history、evaluation_rounds、model_parameters 与 versions，可用 joblib 保存；预测返回输入行序对应的一维正类概率，不重新 fit。`settings` 可接收完整共享配置或 tree 子配置。`predictions` 是模型名到概率向量或含 `probability`（可选 `score`）DataFrame 的映射；与标签一一对齐，Series/DataFrame 同时检查索引。曲线函数额外提供可选关键字 `partition` 标注分区，原三参数调用仍有效。信用分图仅绘制评分卡提供的 score。

本 PR 不改共享 `run_pipeline.py`、配置、依赖、其他成员模块；共享入口当前仅支持 baseline/scorecard，后续由负责人接入 tree。本次统一比较使用上述个人目录脚本，未声称共享入口已完成树模型集成。

## 评价图

标题/图例注明分区、样本数和模型；校准图使用 10 个分位组。信用分图使用评分卡原始浮点分，按类别密度归一，图例标明类别人数。

| 分区 | ROC | PR | 校准 | 信用分分布 |
| --- | --- | --- | --- | --- |
| validation，n=22,518 | [图](figures/validation/roc.png) | [图](figures/validation/pr.png) | [图](figures/validation/calibration.png) | [图](figures/validation/credit_score.png) |
| test，n=22,508 | [图](figures/test/roc.png) | [图](figures/test/pr.png) | [图](figures/test/calibration.png) | [图](figures/test/credit_score.png) |

## 验证与限制

业务测试 **199 passed**，14 条既有 matplotlib/pyparsing 废弃提示；新增测试覆盖验证专用早停、有效轮数、预测不拟合、字段/索引/非法输入、保存加载、阈值并列与边界、已知指标数值、T03 评价兼容及四类图产物。模型、评价和实验脚本已做独立 AI 代码复核，无待修实质问题。

额外检查：全部记录产物/源码/配置哈希一致，分区位置与 T03 EXP-002 完全一致；LR 逐行概率与该次实验完全一致；三模型原始行序、标签、分区对齐，组指纹不跨区，混淆矩阵总数正确，保存加载最大输出差为 0。manifest 的输出哈希对应本地原始产物；提交的汇总 CSV 只按 Git 规则将换行规范为 LF，其提交副本哈希另记于 audit.json，数值不变。LightGBM 与 sklearn 的验证 AUC 仅差 `1.11e-16`，按 `1e-12` 数值容差复核；有效轮数由完整验证历史 argmax 确认。

这是公开训练集固定内部研究分区，之前已参与开发，无法证明新的独立、时间外或实际业务效果。验证集同时选择轮数与阈值，未额外做概率校准。阈值未经业务成本验证，不能作为实际审批政策。SHAP 属 T06，本次未运行。个人练习未提交；郭宇哲独立复现/讲解、团队同伴正式审查和负责人合并仍待完成，AI 复核不替代这些步骤。
