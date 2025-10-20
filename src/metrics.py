# threshold picking and printable metric report

# imports
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    auc,
    roc_curve,
)
from .utils import log


# Find the smallest decision threshold τ whose recall ≥ target_recall.
# then pick the one with the best F1 score.
def choose_threshold(y_val, p_val, target_recall: float):

    precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)

    # indices where recall meets or exceeds the target
    cand = np.where(recalls >= target_recall)[0]
    if len(cand) > 0:

        # pick the threshold that gives the best F1
        best_f1, best_tau = -1.0, 0.5
        for i in cand:
            if i == 0:
                continue  # skip the very first element (no valid threshold)
            t = thresholds[i - 1]  # actual threshold for this index
            pr, rc = float(precisions[i]), float(recalls[i])
            f1 = 2 * pr * rc / (pr + rc + 1e-8)
            if f1 > best_f1:
                best_f1, best_tau = f1, float(t)
        return best_tau, precisions, recalls, thresholds

    # if no threshold achieves target recall,
    # fall back to threshold giving best overall F1
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    j = int(np.nanargmax(f1s))
    tau = float(thresholds[j - 1]) if j > 0 else 0.5
    return tau, precisions, recalls, thresholds


# Print and return a summary of model performance at threshold τ.
def comprehensive_report(split, y_true, p_prob, tau):

    # ensure proper dtypes
    y_true = np.asarray(y_true).astype(int)
    p_prob = np.asarray(p_prob).astype(float)

    # binary predictions using the chosen threshold
    y_pred = (p_prob >= tau).astype(int)

    # accuracy is the mean of correct predictions
    acc = (y_pred == y_true).mean()

    # compute full precision/recall curve to get metrics at τ
    precisions, recalls, thresholds = precision_recall_curve(y_true, p_prob)
    if len(thresholds):

        # find the index in thresholds nearest τ
        idx = np.searchsorted(thresholds, tau, side="right") - 1
        idx = np.clip(idx, 0, len(precisions) - 1)
        prec_at, rec_at = float(precisions[idx]), float(recalls[idx])

    else:
        # degenerate case: no thresholds returned then use first element
        prec_at, rec_at = float(precisions[0]), float(recalls[0])

    # F1 formula
    f1 = 2 * prec_at * rec_at / (prec_at + rec_at + 1e-8)

    # area under PR + ROC curves
    pr_auc = auc(recalls, precisions)
    fpr, tpr, _ = roc_curve(y_true, p_prob)
    roc_auc = auc(fpr, tpr)

    # formatted console summary
    log(f"\n{'='*56}\n{split} @ τ = {tau:.4f}\n{'='*56}")
    log(f"Accuracy:   {acc:.4f}")
    log(f"Precision:  {prec_at:.4f}")
    log(f"Recall:     {rec_at:.4f}")
    log(f"F1-Score:   {f1:.4f}")
    log(f"PR AUC:     {pr_auc:.4f}")
    log(f"ROC AUC:    {roc_auc:.4f}")
    log("\nClassification Report:\n" + classification_report(y_true, y_pred, digits=4))
    log("Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))

    # return metrics in a dict for saving to CSV
    return {
        "accuracy": float(acc),
        "precision": float(prec_at),
        "recall": float(rec_at),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
    }
