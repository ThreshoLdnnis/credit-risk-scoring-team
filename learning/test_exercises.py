"""对同一接口进行验收。默认检查学员 exercises，负责人可切换参考模块。"""

import importlib
import os

import numpy as np
import pandas as pd
import pytest

impl = importlib.import_module(os.environ.get("CREDIT_RISK_EXERCISE_IMPL", "exercises"))


def test_missing_rates_preserve_order_and_zero():
    frame = pd.DataFrame({"zero": [0, 0], "half": [1, np.nan], "all": [np.nan, np.nan]})
    result = impl.missing_rate(frame)
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == frame.columns.tolist()
    np.testing.assert_allclose(result.to_numpy(), [0.0, 0.5, 1.0])


@pytest.mark.parametrize("bad", [pd.DataFrame(), pd.DataFrame(columns=["x"]), [], None])
def test_missing_invalid(bad):
    with pytest.raises(ValueError):
        impl.missing_rate(bad)


def test_split_groups_cover_every_row_and_are_reproducible():
    groups = np.repeat(np.arange(200), 2)
    y = np.repeat(np.arange(200) % 2, 2)
    first = impl.split_three_way(y, groups, seed=42)
    second = impl.split_three_way(y, groups, seed=42)
    assert set(first) == {"train", "validation", "test"}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(400))
    memberships = {}
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert set(y[first[name]]) == {0, 1}
        memberships[name] = set(groups[first[name]])
    assert not memberships["train"] & memberships["validation"]
    assert not memberships["train"] & memberships["test"]
    assert not memberships["validation"] & memberships["test"]
    assert [len(first[k]) for k in ["train", "validation", "test"]] == [280, 60, 60]


def test_split_unequal_groups_and_conflicting_labels_remain_intact():
    groups = np.repeat(np.arange(200), np.tile([1, 2, 3, 4], 50))
    y = (groups % 2).astype(int)
    y[np.flatnonzero(groups == 2)[0]] = 1
    original = y.copy()
    parts = impl.split_three_way(y, groups)
    np.testing.assert_array_equal(y, original)
    assert [len(set(groups[parts[k]])) for k in ["train", "validation", "test"]] == [140, 30, 30]
    for group in np.unique(groups):
        assert sum(group in set(groups[pos]) for pos in parts.values()) == 1


@pytest.mark.parametrize("y,g", [([0, 1], [0]), ([0, 1], [0, np.nan]),
    ([0, 2], [0, 1]), ([1, 1], [0, 1]), ([0, 1, 0, 1], [0, 1, 2, 3])])
def test_split_invalid(y, g):
    with pytest.raises(ValueError):
        impl.split_three_way(y, g)


@pytest.mark.parametrize("y,p,expected", [
    ([0, 0, 1, 1], [.1, .4, .35, .8], .75),
    ([0, 1], [.5, .5], .5), ([0, 1], [0, 1], 1.0), ([0, 1], [1, 0], 0.0),
])
def test_auc_examples(y, p, expected):
    assert impl.roc_auc_manual(y, p) == pytest.approx(expected)


def test_auc_is_invariant_to_row_permutation_and_monotonic_transform():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])
    assert impl.roc_auc_manual(y[::-1], p[::-1]) == pytest.approx(.75)
    assert impl.roc_auc_manual(y, p**2) == pytest.approx(.75)


@pytest.mark.parametrize("y,p", [([1, 1], [.2, .3]), ([0, 1], [.2]),
    ([0, 1], [np.nan, .2]), ([0, 1], [-.1, .4]), ([0, 1], [.1, 1.1]),
    ([[0, 1]], [[.1, .9]])])
def test_auc_invalid(y, p):
    with pytest.raises(ValueError):
        impl.roc_auc_manual(y, p)


def test_woe_hand_calculation_and_zero_cells():
    woe, iv = impl.woe_iv_from_counts([8, 2], [2, 8], alpha=.5)
    np.testing.assert_allclose(woe, [np.log(3.4), -np.log(3.4)])
    assert iv == pytest.approx(12 / 11 * np.log(3.4))
    pure, pure_iv = impl.woe_iv_from_counts([10, 0], [0, 10])
    assert np.isfinite(pure).all() and np.isfinite(pure_iv)


def test_woe_class_swap_reverses_sign_but_preserves_iv():
    w, iv = impl.woe_iv_from_counts([8, 2], [2, 8])
    swapped, swapped_iv = impl.woe_iv_from_counts([2, 8], [8, 2])
    np.testing.assert_allclose(swapped, -w)
    assert swapped_iv == pytest.approx(iv)
    neutral, neutral_iv = impl.woe_iv_from_counts([4, 6], [4, 6])
    np.testing.assert_allclose(neutral, [0, 0])
    assert neutral_iv == pytest.approx(0)


@pytest.mark.parametrize("g,b,alpha", [([1, -1], [1, 2], .5), ([1], [1, 2], .5),
    ([0, 0], [1, 2], .5), ([1, 2], [0, 0], .5), ([1, np.nan], [1, 2], .5),
    ([1, 2], [1, 2], 0), ([1, 2], [1, 2], np.inf)])
def test_woe_invalid(g, b, alpha):
    with pytest.raises(ValueError):
        impl.woe_iv_from_counts(g, b, alpha)


def test_scores_anchors_shape_direction_and_odds_doubling():
    result = impl.probability_to_score(np.array([[1 / 21, 1 / 41], [1 / 11, .5]]))
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], [600, 620])
    assert result[1, 0] == pytest.approx(580)
    assert result[1, 1] < result[1, 0]
    assert isinstance(impl.probability_to_score(1 / 21), float)
    p = np.array([.01, .05, .2])
    doubled_odds_p = p / (2 - p)
    np.testing.assert_allclose(impl.probability_to_score(doubled_odds_p) - impl.probability_to_score(p), 20)


@pytest.mark.parametrize("values", [[0], [1], [np.nan], [np.inf], [-.1], [1.1], []])
def test_score_invalid_probabilities(values):
    with pytest.raises(ValueError):
        impl.probability_to_score(values)


@pytest.mark.parametrize("kwargs", [{"pdo": 0}, {"pdo": np.inf}, {"base_odds": -1}, {"base_score": np.nan}])
def test_score_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        impl.probability_to_score([.2], **kwargs)


def test_comparison_known_metrics_and_inclusive_threshold():
    result = impl.compare_models([0, 0, 1, 1],
        {"a": [.1, .4, .35, .8], "b": [.15, .25, .55, .75]}, {"a": .4, "b": .5})
    assert result.index.tolist() == ["a", "b"]
    assert result.loc["a", "roc_auc"] == pytest.approx(.75)
    assert result.loc["a", "brier"] == pytest.approx(.158125)
    assert result.loc["a", ["tn", "fp", "fn", "tp"]].tolist() == [1, 1, 1, 1]
    assert result.loc["b", "roc_auc"] == pytest.approx(1.0)
    assert result.loc["b", "brier"] == pytest.approx(.0875)
    assert result.loc["b", ["tn", "fp", "fn", "tp"]].tolist() == [2, 0, 0, 2]


@pytest.mark.parametrize("predictions,thresholds", [({}, {}), ({"a": [.1, .9]}, {}),
    ({"a": [.2]}, {"a": .4}), ({"a": [0, np.nan]}, {"a": .5}),
    ({"a": [.2, .8]}, {"a": 1.1}), ({"a": [.2, .8]}, {"a": np.nan})])
def test_comparison_invalid(predictions, thresholds):
    with pytest.raises(ValueError):
        impl.compare_models([0, 1], predictions, thresholds)
