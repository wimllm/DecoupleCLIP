import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score


def rescale(x):
    denom = x.max() - x.min()
    if denom == 0:
        return np.zeros_like(x)
    return (x - x.min()) / denom


def _as_binary_labels(gt: np.ndarray):
    gt = np.asarray(gt)
    if gt.size == 0:
        return gt.astype(np.uint8)
    return (gt > 0.5).astype(np.uint8)


def _as_finite_scores(score: np.ndarray):
    score = np.asarray(score, dtype=np.float64)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)


def is_one_class(gt: np.ndarray):
    gt_ravel = _as_binary_labels(gt).ravel()
    return gt_ravel.size == 0 or np.unique(gt_ravel).size < 2


def calculate_px_metrics(gt_px, pr_px):
    gt_px = _as_binary_labels(gt_px)
    pr_px = _as_finite_scores(pr_px)

    if is_one_class(gt_px):  # In case there are only normal pixels or no pixel-level labels
        return 0, 0, 0

    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
    precisions, recalls, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

    return auroc_px * 100, f1_px * 100, ap_px * 100


def calculate_im_metrics(gt_im, pr_im):
    gt_im = _as_binary_labels(gt_im)
    pr_im = _as_finite_scores(pr_im)

    if is_one_class(gt_im):  # In case there are only normal samples or no image-level labels
        return 0, 0, 0

    auroc_im = roc_auc_score(gt_im.ravel(), pr_im.ravel())
    precisions, recalls, _ = precision_recall_curve(gt_im.ravel(), pr_im.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_im = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_im = average_precision_score(gt_im, pr_im)

    return ap_im * 100, auroc_im * 100, f1_im * 100


def calculate_average_metric(metrics: dict):
    average = {}
    for obj, metric in metrics.items():
        for k, v in metric.items():
            if k not in average:
                average[k] = []
            average[k].append(v)

    for k, v in average.items():
        average[k] = np.mean(v)

    return average


def calculate_metric(results, obj):
    gt_px = []
    pr_px = []

    gt_im = []
    pr_im = []

    for idx in range(len(results['cls_names'])):
        if results['cls_names'][idx] == obj:
            gt_px.append(results['imgs_masks'][idx])
            pr_px.append(results['anomaly_maps'][idx])

            gt_im.append(results['imgs_gts'][idx])
            pr_im.append(results['anomaly_scores'][idx])

    gt_px = np.array(gt_px)
    pr_px = np.array(pr_px)

    gt_im = np.array(gt_im)
    pr_im = np.array(pr_im)

    auroc_px, f1_px, ap_px = calculate_px_metrics(gt_px, pr_px)
    ap_im, auroc_im, f1_im = calculate_im_metrics(gt_im, pr_im)

    metric = {
        'auroc_px': auroc_px,
        'auroc_im': auroc_im,
        'f1_px': f1_px,
        'f1_im': f1_im,
        'ap_px': ap_px,
        'ap_im': ap_im,
    }

    return metric
