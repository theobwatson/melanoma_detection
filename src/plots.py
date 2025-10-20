# training curves, PR/ROC, confusion, threshold sweep

# imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc


def plot_train_curves(h1, h2, out_dir):
    # extract a list of metrics from Keras History
    def ex(hist, keys):
        return {k: hist.history.get(k, []) for k in keys}

    # keys from training
    keys = ["loss", "val_loss", "auc_pr", "val_auc_pr"]

    # get phase 1 metrics (or empty lists if h1 is None)
    p1 = ex(h1, keys) if h1 else {k: [] for k in keys}

    # get phase 2 metrics (or empty lists if h2 is None)
    p2 = ex(h2, keys) if h2 else {k: [] for k in keys}

    # merge phase 1 and 2 metrics into one dict
    m = {k: p1[k] + p2[k] for k in keys}

    # x-axis is just epoch count starting at 1
    epochs = np.arange(1, len(m["loss"]) + 1)

    # -------------------------------
    # 1) loss plot (train + val)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, m["loss"], label="train loss")
    plt.plot(epochs, m["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "train_loss.png", dpi=160)
    plt.close()

    # -------------------------------
    # 2) PR-AUC plot (train + val)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, m["auc_pr"], label="train PR-AUC")
    plt.plot(epochs, m["val_auc_pr"], label="val PR-AUC")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.title("PR-AUC over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "train_pr_auc.png", dpi=160)
    plt.close()

    # -------------------------------
    # 3) combined plot: loss and PR-AUC on two y-axes
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(epochs, m["loss"], label="train loss")
    ax1.plot(epochs, m["val_loss"], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    # secondary axis for PR-AUC
    ax2 = ax1.twinx()
    ax2.plot(epochs, m["auc_pr"], label="train PR-AUC")
    ax2.plot(epochs, m["val_auc_pr"], label="val PR-AUC")
    plt.title("Training curves (loss + PR-AUC)")

    # place legends so they don’t overlap
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "train_curves.png", dpi=160)
    plt.close()


def plot_pr_roc(y_true, p_prob, split_name, out_dir):

    # compute precision/recall at all thresholds
    precisions, recalls, _ = precision_recall_curve(y_true, p_prob)

    # compute ROC (false pos/true pos rates)
    fpr, tpr, _ = roc_curve(y_true, p_prob)

    # area under the precision-recall curve
    pr_auc = auc(recalls, precisions)

    # -------------------------------
    # 1) Precision-Recall curve
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve {split_name} (AUC={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"pr_curve_{split_name}.png", dpi=160)
    plt.close()

    # -------------------------------
    # 2) ROC curve with diagonal baseline
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")  # baseline (random)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve {split_name}")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"roc_curve_{split_name}.png", dpi=160)
    plt.close()


def plot_confusion(y_true, p_prob, tau, split_name, out_dir):
    # apply threshold tau to probabilities to get binary predictions
    y_pred = (np.asarray(p_prob) >= tau).astype(int)

    # compute 2x2 confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # visualize as a simple heatmap
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion {split_name} @ τ={tau:.3f}")
    plt.colorbar()

    # set axis ticks
    ticks = np.arange(2)
    plt.xticks(ticks, ["NEG", "POS"])
    plt.yticks(ticks, ["NEG", "POS"])

    # annotate cells with text
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thr else "black",
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"confusion_{split_name}.png", dpi=160)
    plt.close()


def plot_threshold_sweep(y_true, p_prob, out_dir):
    # compute precision/recall at all thresholds
    precisions, recalls, thr = precision_recall_curve(y_true, p_prob)

    # precision_recall_curve gives one fewer threshold than scores
    # pad with 1.0 so lengths align
    thr_used = np.r_[thr, 1.0]

    # compute F1 score for each threshold
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    # plot precision, recall, and F1 against threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thr_used, precisions, label="Precision")
    plt.plot(thr_used, recalls, label="Recall")
    plt.plot(thr_used, f1, label="F1")
    plt.xlabel("threshold τ")
    plt.ylabel("score")
    plt.title("Threshold sweep (VAL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "thr_sweep_VAL.png", dpi=160)
    plt.close()
