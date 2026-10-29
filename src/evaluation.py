"""T05 aligned binary evaluation, validation-only thresholds and saved curves."""

from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def _vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) == 0 or np.iscomplexobj(raw):
        raise ValueError(f'{name} must be a nonempty real one-dimensional vector')
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} must contain real numbers') from error
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def _labels(y):
    labels = _vector(y, 'y')
    if not pd.Series(np.asarray(y)).isin([0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError('y must contain both binary 0/1 classes')
    return labels.astype(int)


def _paired(y, probability):
    if isinstance(y, pd.Series) and isinstance(probability, pd.Series) and not y.index.equals(probability.index):
        raise ValueError('Label and probability Series indices must align in row order')
    labels, p = _labels(y), _vector(probability, 'probability')
    if len(p) != len(labels) or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must align with y and lie in [0,1]')
    return labels, p


def select_threshold(validation_y, validation_probability) -> float:
    """验证 ROC 上最大化 TPR-FPR；有限合法阈值并列时取较大值。"""
    labels, p = _paired(validation_y, validation_probability)
    fpr, tpr, thresholds = roc_curve(labels, p, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates, gains = thresholds[valid], (tpr - fpr)[valid]
    return float(candidates[gains == gains.max()].max())


def evaluate_binary(y, probability, threshold: float) -> dict:
    """Use p>=threshold; AP and trapezoidal PR area are separate quantities.

    Log loss uses sklearn's numerical endpoint handling; original probabilities
    are never clipped or changed for other metrics, classification or storage.
    """
    labels, p = _paired(y, probability)
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be a finite scalar in [0,1]')
    fpr, tpr, _ = roc_curve(labels, p)
    pr_precision, pr_recall, _ = precision_recall_curve(labels, p)
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
    return {'rows': len(labels), 'positive': int(labels.sum()), 'auc': float(roc_auc_score(labels, p)),
            'ks': float(np.max(tpr - fpr)), 'ap': float(average_precision_score(labels, p)),
            'pr_auc_trapezoidal': float(auc(pr_recall, pr_precision)), 'brier': float(brier_score_loss(labels, p)),
            'log_loss': float(log_loss(labels, p)), 'threshold': float(threshold),
            'decision_rule': 'probability >= threshold', 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


def save_evaluation_figures(y, predictions: dict, output_dir, *, partition='unspecified'):
    """Save ROC/PR/calibration and credit-score distributions on identical rows.

    Values may be probability vectors/Series or DataFrames with probability and
    optional score. Series/DataFrame indices must match a Series y. Scorecard
    scores are used directly; no fabricated scores for other models. The optional
    partition label preserves existing three-argument calls. Returns file paths.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not isinstance(predictions, dict) or not predictions:
        raise ValueError('predictions must be a nonempty model dictionary')
    prepared, scores = {}, {}
    for name, value in predictions.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Model names must be nonempty strings')
        if isinstance(value, pd.DataFrame):
            if value.columns.has_duplicates or 'probability' not in value:
                raise ValueError('Prediction tables need a unique probability column')
            labels, p = _paired(y, value['probability'])
            if 'score' in value:
                score = _vector(value['score'], 'score')
                if len(score) != len(labels):
                    raise ValueError('Scores must align with y')
                scores[name] = score
        else:
            labels, p = _paired(y, value)
        prepared[name] = p
    output = Path(output_dir)
    keys = ['roc', 'pr', 'calibration'] + (['credit_score'] if scores else [])
    paths = {key: output / (key + '.png') for key in keys}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Evaluation plots already exist; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)

    def canvas(title, xlabel, ylabel):
        fig = Figure(figsize=(7.2, 5.2), dpi=160)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set(title=f'{title} | {partition} | n={len(labels):,}', xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        return fig, ax

    fig, ax = canvas('ROC', 'False positive rate', 'True positive rate')
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    for name, p in prepared.items():
        fpr, tpr, _ = roc_curve(labels, p)
        ax.plot(fpr, tpr, label=f'{name}: AUC={roc_auc_score(labels, p):.4f}')
    ax.legend()
    fig.savefig(paths['roc'], bbox_inches='tight')
    fig, ax = canvas('Precision-recall', 'Recall', 'Precision')
    ax.axhline(labels.mean(), linestyle='--', color='grey', label=f'Prevalence={labels.mean():.4f}')
    for name, p in prepared.items():
        precision, recall, _ = precision_recall_curve(labels, p)
        ax.plot(recall, precision, label=f'{name}: AP={average_precision_score(labels, p):.4f}')
    ax.legend()
    fig.savefig(paths['pr'], bbox_inches='tight')
    fig, ax = canvas('Calibration (10 quantile bins)', 'Mean predicted probability', 'Observed event rate')
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Ideal')
    for name, p in prepared.items():
        fraction, mean = calibration_curve(labels, p, n_bins=10, strategy='quantile')
        ax.plot(mean, fraction, marker='o', label=name)
    ax.legend()
    fig.savefig(paths['calibration'], bbox_inches='tight')
    if scores:
        fig, ax = canvas('Credit score distribution', 'Credit score', 'Density within class')
        for name, score in scores.items():
            edges = np.histogram_bin_edges(score, bins=30)
            for target in [0, 1]:
                mask = labels == target
                ax.hist(score[mask], bins=edges, density=True, alpha=.45,
                        label=f'{name}, y={target}, n={int(mask.sum()):,}')
        ax.legend()
        fig.savefig(paths['credit_score'], bbox_inches='tight')
    return {key: str(path) for key, path in paths.items()}
