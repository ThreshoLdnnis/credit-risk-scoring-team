"""Known metrics, threshold ties, alignment and actual rendered figure files."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation import evaluate_binary, save_evaluation_figures, select_threshold


def test_known_metrics_inclusive_threshold_and_direction():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])
    before = p.copy()
    result = evaluate_binary(y, p, .4)
    assert result['auc'] == .75
    assert result['brier'] == pytest.approx(.158125)
    assert result['ks'] == .5
    assert [result[k] for k in ['tn', 'fp', 'fn', 'tp']] == [1, 1, 1, 1]
    assert [result[k] for k in ['precision', 'recall', 'f1']] == [.5, .5, .5]
    np.testing.assert_array_equal(p, before)
    assert evaluate_binary(y, 1 - p, .4)['auc'] == .25


def test_ties_constant_scores_ap_and_pr_area():
    result = evaluate_binary([0, 1], [.5, .5], .5)
    assert result['auc'] == .5 and result['ks'] == 0
    assert result['ap'] == .5 and result['pr_auc_trapezoidal'] == .75
    assert select_threshold([0, 1], [.5, .5]) == .5
    assert select_threshold([0, 1, 0, 1], [.1, .2, .7, .8]) == .8
    assert select_threshold([0, 1], [0, 1]) == 1
    assert np.isfinite(evaluate_binary([0, 1], [0, 1], .5)['log_loss'])
    assert evaluate_binary([0, 1], [.1, .2], 1)['f1'] == 0


@pytest.mark.parametrize('y,p', [([], []), ([1, 1], [.1, .2]), ([0, 2], [.1, .2]),
    ([0, 1], [.2]), ([0, 1], [np.nan, .2]), ([0, 1], [np.inf, .2]),
    ([0, 1], [-.1, .2]), ([0, 1], [.1, 1.1]), ([[0, 1]], [[.1, .2]]),
    ([0, 1], [1j, .2])])
def test_invalid_pairs(y, p):
    with pytest.raises(ValueError):
        evaluate_binary(y, p, .5)
    with pytest.raises(ValueError):
        select_threshold(y, p)


@pytest.mark.parametrize('threshold', [np.inf, np.nan, -.1, 1.1, True, '0.5', [.5]])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        evaluate_binary([0, 1], [.2, .8], threshold)


def test_series_misalignment_rejected(tmp_path):
    y = pd.Series([0, 1], index=[5, 8])
    p = pd.Series([.2, .8], index=[8, 5])
    with pytest.raises(ValueError, match='align'):
        evaluate_binary(y, p, .5)
    with pytest.raises(ValueError, match='align'):
        save_evaluation_figures(y, {'model': p}, tmp_path)


def test_real_png_files_and_scorecard_distribution(tmp_path):
    y = pd.Series([0, 0, 1, 1], index=[4, 6, 8, 10])
    card = pd.DataFrame({'probability': [.1, .4, .35, .8], 'score': [650, 600, 610, 500]}, index=y.index)
    paths = save_evaluation_figures(y, {'scorecard': card, 'tree': [.1, .2, .7, .9]}, tmp_path, partition='test')
    assert set(paths) == {'roc', 'pr', 'calibration', 'credit_score'}
    for path in paths.values():
        with Image.open(path) as im:
            assert im.width > 600 and im.height > 400
            im.verify()
    with pytest.raises(FileExistsError):
        save_evaluation_figures(y, {'card': card}, tmp_path)


def test_invalid_model_lengths_rejected_before_plotting(tmp_path):
    with pytest.raises(ValueError):
        save_evaluation_figures([0, 1], {'a': [.1, .9], 'b': [.2]}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_compatibility_with_existing_t03_helpers():
    from run_pipeline import metrics, validation_threshold

    y, p = [0, 0, 1, 1], np.array([.1, .4, .35, .8])
    assert evaluate_binary(y, p, .4) == metrics(y, p, .4)
    assert select_threshold(y, p) == validation_threshold(y, p)
