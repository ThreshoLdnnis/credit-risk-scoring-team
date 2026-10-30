# T06/T07：郭宇哲本人模型、评价和解释的复现

2026-10-31，郭宇哲（morebadly）的任务工作区，Codex 协助实现并实际执行。关联 [T06 #6](https://github.com/ThreshoLdnnis/credit-risk-scoring-team/issues/6) 和 [T07 #7](https://github.com/ThreshoLdnnis/credit-risk-scoring-team/issues/7)。只提交本人子项；个人练习未提交，页面和最终报告整合仍按原分工推进。

| 材料 | 内容 |
| --- | --- |
| [record.json](record.json) | C 模板，实际来源、配置、处理、结果及限制 |
| [run_manifest.json](run_manifest.json) | 共享入口本次 LR/tree 运行汇总、代码/产物哈希 |
| [reproduction.json](reproduction.json) | 空白环境、源码快照、命令/退出码、全部安装版本 |
| [explanation_audit.json](explanation_audit.json) | 复验、解释样本、加和/概率映射及发布边界 |
| [model_comparison.csv](model_comparison.csv) | 本次两模型、三分区，共6行指标 |
| [global_importance.csv](global_importance.csv) / [全局图](global_shap.png) | 固定1000条测试样本的13列平均绝对 SHAP |
| [完整环境限制](full_environment_limitation.json) | Windows 原生依赖失败尝试，保留真实非零退出码 |

## 本次成功运行范围

从空白 Python 3.11.16 venv（初始包列表为空、关闭系统和用户 site-packages）通过 `uv pip install --python <env> -r requirements.txt setuptools==75.6.0 pip==24.0` 安装78个包。受 constraints 约束的已安装版本差异为0，pip check通过。使用 `git archive` 创建干净源码副本，全部源码哈希在运行后仍相同；未包含未跟踪的个人练习/T04文件。这是同机的新环境验收，未声称第二台机器验收。

本人相关测试 **110 passed、1 deselected**：仅排除需要评分卡原生依赖的三模型E2E用例；测试选择和命令逐项记录。首版 PR 的 Linux 完整业务CI通过，最终发布状态以PR最后一次CI为准。

指定 `cs-training.csv` SHA256 为 `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac`，150,000行；沿用T03经授权取得的固定GitHub镜像，与T02官方文件字节相同。保留原始取得元数据，未冒称本次直接从Kaggle下载。

共享入口运行 `--models baseline tree`；训练/验证/测试为104974/22518/22508，按清洗后内容组70/15/15、seed42。LightGBM最佳257轮、评价287轮，模型和阈值都由验证选择后冻结。

| 模型 | 测试 AUC | KS | AP | Brier | 验证阈值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR | 0.8235213591 | 0.5065242838 | 0.3529506747 | 0.0525956283 | 0.0629705015937871 |
| LightGBM | 0.8718373023 | 0.5779727559 | 0.4117025435 | 0.0483505514 | 0.06673319462105777 |

复验对每个模型重新预测全150,000行，核对行序、原序号、标签、分区、验证阈值和全部指标；两模型与T05逐行概率最大差均为 **0**。保存加载及组隔离通过。新生成的validation/test六张评价曲线保留本地并列入manifest哈希；历史三模型对照见已合并的 [T05 实验](../T05-EXP-001/README.md)。

## SHAP 实际证据

使用 `shap.TreeExplainer`，显式 `model_output='raw'`、`feature_perturbation='tree_path_dependent'`、精确计算。背景来自训练104,974行在树中的路径/叶计数，无独立显式背景行；解释样本不作背景。独立Booster只保留最佳257轮，并保留binary目标参数；不改模型、不重新fit。

测试分区按seed42均匀无放回抽1000行，再按原始位置排序，不按标签、预测或贡献选择样本。解释加和最大绝对误差 **1.4876988529977098e-14**；sigmoid(raw)与离线正类概率最大差 **1.1102230246251565e-16**；分别小于1e-6和1e-10容差。单行、整批和列重排输出一致。

全局重要性前三为循环信贷使用率、30–59天逾期次数、年龄，均值绝对贡献约0.69694/0.29275/0.21915 log-odds。special三项在这1000行中平均绝对贡献为0，不推断它们在其他模型/样本或业务中无作用。该排序不说明方向、因果或业务收益。

局部图完整计入前8项和其余5项有符号总和，基础值最终累计到raw。局部图、样本位置、逐行SHAP/概率和模型均留在本地；公开审计已剔除首条样本的局部剩余贡献数值。发布CSV只将换行规范为LF，另记副本哈希；manifest原始产物哈希保持不变。

## 复现和接入

```powershell
python run_pipeline.py --models baseline tree --output artifacts/morebadly/T07-REPRO-NEW
python experiments/morebadly/verify_and_explain.py --experiment artifacts/morebadly/T07-REPRO-NEW --output artifacts/morebadly/T06-EXPLAIN-NEW
```

详细环境、排障和现场步骤见 [T07复现与演示](../../../reports/morebadly/T07_复现与演示.md)。页面两参数协议、字段和绘图单位见 [解释接口接入](../T06_解释接口接入.md)。模型参数和配置不随解释样本数改变；每次使用新输出目录。

## 未通过的环境项与未完成的团队项

旧venv完整测试有14项因OR-Tools原生DLL失败；通过requirements/constraints新安装后，Windows应用控制仍阻止 `ortools.sat.python.swig_helper`。因此完整 `check_environment.py` 非零退出，评分卡/全三模型新环境复现没有在本机通过，失败记录单独保留。没有修改Windows安全策略、依赖、评分卡实现或全局配置。

本次新环境成功范围是郭宇哲的LR/tree/评价/解释，不能说所有团队环境都通过。共享接口沿用GitHub约定，无app.py/共享入口变更。页面tree/SHAP接入归杨曦；本人现场讲解和小改、同伴正式审查、负责人合并与报告整合仍待相应人员完成。数据已参与开发，不能证明新独立、时间外或实际授信效果。
