"""运行六项练习并导出合成演示结果；不是 Kaggle 模型效果。"""

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    impl = importlib.import_module(os.environ.get("CREDIT_RISK_EXERCISE_IMPL", "exercises"))
    root = Path(__file__).resolve().parent
    rates = impl.missing_rate(pd.DataFrame({"zero": [0, 0], "income": [1, np.nan]}))
    groups = np.repeat(np.arange(200), 2)
    parts = impl.split_three_way(np.repeat(np.arange(200) % 2, 2), groups)
    toy = pd.read_csv(root / "data/toy_scores.csv")
    woe, iv = impl.woe_iv_from_counts([8, 2], [2, 8])
    comparison = impl.compare_models(toy.y, {"a": toy.p_a, "b": toy.p_b}, {"a": .4, "b": .5})
    result = {
        "scope": "合成课堂演示，不代表真实信贷模型效果",
        "missing_rate": rates.to_dict(),
        "split_rows": {k: len(v) for k, v in parts.items()},
        "auc_a": impl.roc_auc_manual(toy.y, toy.p_a),
        "woe": np.asarray(woe).tolist(), "iv": float(iv),
        "scores": np.asarray(impl.probability_to_score([1 / 21, 1 / 41, 1 / 11])).tolist(),
        "comparison": comparison.reset_index().to_dict(orient="records"),
    }
    output = root / "output"
    output.mkdir(exist_ok=True)
    (output / "demo_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
