# 六项个人练习

公共框架在 `templates/exercises.py`，公共验收在 `test_exercises.py`。每人用本人 GitHub 用户名建立答案目录，避免多人修改同一个答案文件。

以下命令在仓库根目录执行，把 `YOUR_USERNAME` 换成本人 GitHub 用户名：

```powershell
.venv\Scripts\python.exe tools/exercises.py start YOUR_USERNAME
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME
```

`start` 只在目录不存在时复制 TODO，不覆盖已有答案。打开 `learning/submissions/YOUR_USERNAME/exercises.py` 编写本人实现。初始代码会明确报 `NotImplementedError`，这是待完成状态。

可逐项验证，也可运行人工合成演示：

```powershell
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME --task missing
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME --task split
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME --task auc
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME --task woe
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME --task score
.venv\Scripts\python.exe tools/exercises.py check YOUR_USERNAME --task comparison
.venv\Scripts\python.exe tools/exercises.py demo YOUR_USERNAME
```

详细要求见[学员任务书](../docs/学员任务书.md)。实现六项接口、核对手算、保留测试记录、增加一个自己的边界案例。个人新增测试放在本人目录的 `test_*.py` 中，`check` 会与公共测试一起运行。

六项全部完成后提 PR；阶段性代码可开 Draft PR。CI 会检查所有已提交的答案目录，未完成 TODO 会失败。没有成员提交时明确显示“尚无个人练习提交”，不计作任何人已经通过。

`data/toy_scores.csv` 是四条人工构造数据，演示结果只用于检查实现，不能代表真实信贷模型效果。业务代码在 `src/` 另行开发和测试；六项练习通过不等于整个系统完成。
