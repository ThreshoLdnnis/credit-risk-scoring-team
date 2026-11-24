"""Additional personal boundary cases beyond the 51 shared classroom checks."""

import numpy as np
import pandas as pd
import pytest
from exercises import (
    compare_models,
    missing_rate,
    probability_to_score,
    roc_auc_manual,
    split_three_way,
    woe_iv_from_counts,
)


def test_three_row_missing_rates_and_input_unchanged():
    frame = pd.DataFrame({'one': [0, np.nan, 1], 'two': [np.nan, 0, np.nan]})
    original = frame.copy(deep=True)
    np.testing.assert_allclose(missing_rate(frame), [1 / 3, 2 / 3])
    pd.testing.assert_frame_equal(frame, original)


def test_partial_auc_ties_receive_half_credit():
    # Four pairs: .5, 0, 1, 1, whose mean is .625.
    assert roc_auc_manual([0, 0, 1, 1], [.2, .5, .2, .8]) == pytest.approx(.625)


@pytest.mark.parametrize('seed', [True, -1, 1.5])
def test_split_rejects_invalid_seeds(seed):
    groups = np.arange(200)
    with pytest.raises(ValueError, match='seed'):
        split_three_way(groups % 2, groups, seed)


def test_score_extreme_finite_probability_without_clipping():
    result = probability_to_score([1e-300, 0.01, np.nextafter(1.0, 0.0)])
    assert np.isfinite(result).all()
    assert result[0] > result[1] > result[2]
    p = .12
    assert probability_to_score(p / (2 - p)) - probability_to_score(p) == pytest.approx(20)


def test_comparison_rejects_reversed_series_labels():
    y = pd.Series([0, 1], index=[10, 20])
    probability = pd.Series([.2, .8], index=[20, 10])
    with pytest.raises(ValueError, match='row order'):
        compare_models(y, {'model': probability}, {'model': .5})


@pytest.mark.parametrize('operation', ['auc', 'woe', 'score', 'comparison'])
def test_object_arrays_do_not_silently_discard_complex_parts(operation):
    invalid = np.array([np.complex128(.2 + .1j), .8], dtype=object)
    with pytest.raises(ValueError, match='real numbers'):
        if operation == 'auc':
            roc_auc_manual([0, 1], invalid)
        elif operation == 'woe':
            woe_iv_from_counts(invalid, [2, 1])
        elif operation == 'score':
            probability_to_score(invalid)
        else:
            compare_models([0, 1], {'model': invalid}, {'model': .5})


def test_score_rejects_nonfinite_output_for_extreme_pdo():
    with pytest.raises(ValueError, match='floating-point range'):
        probability_to_score(.01, pdo=np.finfo(float).max)


def test_comparison_threshold_changes_only_classification_metrics():
    y, probabilities = [0, 0, 1, 1], [.1, .4, .35, .8]
    low = compare_models(y, {'a': probabilities}, {'a': .4}).loc['a']
    high = compare_models(y, {'a': probabilities}, {'a': .5}).loc['a']
    assert low['roc_auc'] == high['roc_auc'] == pytest.approx(.75)
    assert low['brier'] == high['brier'] == pytest.approx(.158125)
    assert low[['tn', 'fp', 'fn', 'tp']].tolist() == [1, 1, 1, 1]
    assert high[['tn', 'fp', 'fn', 'tp']].tolist() == [2, 0, 1, 1]


def test_woe_zero_frequency_smoothing_distributions_sum_to_one():
    good, bad, alpha = np.array([10., 0., 0.]), np.array([0., 10., 0.]), .5
    pg = (good + alpha) / (good.sum() + alpha * len(good))
    pb = (bad + alpha) / (bad.sum() + alpha * len(bad))
    woe, iv = woe_iv_from_counts(good, bad, alpha)
    assert pg.sum() == pytest.approx(1) and pb.sum() == pytest.approx(1)
    np.testing.assert_allclose(woe, np.log(pg / pb))
    assert woe[2] == 0 and iv > 0
