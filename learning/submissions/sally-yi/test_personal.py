"""自己的小样本边界案例；期望值由题目或手算得到。"""

import exercises as impl
import numpy as np
import pandas as pd
import pytest


def test_missing_thirds_keep_zero_valid():
    frame = pd.DataFrame({'one': [1, np.nan, 2], 'two': [np.nan, 0, np.nan], 'zero': [0, 0, 0]})
    np.testing.assert_allclose(impl.missing_rate(frame), [1 / 3, 2 / 3, 0])


def test_split_all_mixed_groups_keeps_original_labels_and_group_isolation():
    groups = np.repeat(np.arange(100), 2)
    y = np.tile([0, 1], 100)
    before = y.copy()
    parts = impl.split_three_way(y, groups)
    assert [len(parts[name]) for name in ['train', 'validation', 'test']] == [140, 30, 30]
    np.testing.assert_array_equal(y, before)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(200))
    for pos in parts.values():
        assert set(y[pos]) == {0, 1}
    destinations = np.empty(200, dtype=object)
    for name, pos in parts.items():
        destinations[pos] = name
    assert pd.DataFrame({'group': groups, 'part': destinations}).groupby('group')['part'].nunique().eq(1).all()


@pytest.mark.parametrize('seed', [-1, 2**32, 1.5, True])
def test_split_invalid_seed(seed):
    with pytest.raises(ValueError):
        impl.split_three_way(np.arange(200) % 2, np.arange(200), seed=seed)


def test_auc_one_tie_out_of_four_pairs():
    # 正0.5对负0.5/0.1得0.5+1；正0.9对两负得1+1，总分3.5/4。
    assert impl.roc_auc_manual([0, 0, 1, 1], [.5, .1, .5, .9]) == pytest.approx(.875)


def test_woe_one_bin_cannot_separate_classes():
    woe, iv = impl.woe_iv_from_counts([12], [3])
    np.testing.assert_allclose(woe, [0])
    assert iv == pytest.approx(0)


def test_woe_finite_large_counts_and_alpha_do_not_overflow():
    # 平滑后好样本占比为2/3、1/3，坏样本相反；不需形成溢出的2e308。
    woe, iv = impl.woe_iv_from_counts([1e308, 0], [0, 1e308], alpha=1e308)
    np.testing.assert_allclose(woe, [np.log(2), -np.log(2)])
    assert iv == pytest.approx(2 / 3 * np.log(2))


def test_score_custom_base_and_pdo():
    # p=.2对应好坏比4；p=1/9对应8。赔率翻倍，按自选PDO=30加分。
    result = impl.probability_to_score([.2, 1 / 9], base_score=640, base_odds=4, pdo=30)
    np.testing.assert_allclose(result, [640, 670])


def test_score_tiny_valid_probability_keeps_zero_dimensional_array():
    result = impl.probability_to_score(np.array(1e-320))
    assert isinstance(result, np.ndarray) and result.shape == ()
    assert np.isfinite(result) and result > 600


def test_comparison_threshold_changes_confusion_but_not_auc_or_brier():
    predictions = {'a': [.1, .4, .35, .8]}
    low = impl.compare_models([0, 0, 1, 1], predictions, {'a': .4}).loc['a']
    high = impl.compare_models([0, 0, 1, 1], predictions, {'a': .5}).loc['a']
    np.testing.assert_allclose(low[['roc_auc', 'brier']], [.75, .158125])
    np.testing.assert_allclose(high[['roc_auc', 'brier']], [.75, .158125])
    assert low[['tn', 'fp', 'fn', 'tp']].tolist() == [1, 1, 1, 1]
    assert high[['tn', 'fp', 'fn', 'tp']].tolist() == [2, 0, 1, 1]


def test_comparison_keeps_prediction_order_when_threshold_dict_order_differs():
    result = impl.compare_models([0, 1], {'z': [.1, .9], 'a': [.3, .7]}, {'a': .5, 'z': .5})
    assert result.index.tolist() == ['z', 'a']
