# Trains, fine-tunes, evaluates, and exports a classifier (EfficientNetV2B0 or MobileNetV3Small)

# imports
import argparse
import random, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import mixed_precision as mp

from src.config import Args
from src.utils import log, ensure_dir
from src.data import id_to_preproc_path, verify_data_pipeline, make_datasets
from src.model import build_model, partial_unfreeze_with_bn_freeze
from src.metrics import choose_threshold, comprehensive_report
from src.plots import (
    plot_train_curves,
    plot_pr_roc,
    plot_confusion,
    plot_threshold_sweep,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train melanoma classifier")
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["effv2b0", "mnetv3s"],
        help="Which model backbone to use",
    )
    return parser.parse_args()


def main():
    cli = parse_args()  # parse CLI backbone
    args = Args()  # load shared config

    # set output folder based on backbone
    if cli.backbone == "effv2b0":
        args.out = "../runs/effv2b0"
    elif cli.backbone == "mnetv3s":
        args.out = "../runs/mnetv3s"

    # seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # mixed precision enables fp16, which speeds up training and reduces memory use
    mp.set_global_policy("mixed_float16")
    log("Mixed precision enabled (global_policy=mixed_float16)")

    # ensures output path exists
    csv_path, pre_dir, out_dir = Path(args.csv), Path(args.preproc_dir), Path(args.out)
    ensure_dir(out_dir)

    # reads ISIC ground-truth, keep only id and melanoma flag
    log("loading CSV...")
    df = pd.read_csv(csv_path)[["image", "MEL"]].copy()

    # make an integer label
    df["label"] = (df["MEL"] == 1.0).astype("int8")

    # maps each id to actual file path
    log("mapping ids to preprocessed files...")
    df["filepath"] = df["image"].apply(lambda s: id_to_preproc_path(s, pre_dir))
    found = df["filepath"].notna().sum()
    log(f"Found preprocessed files for {found}/{len(df)} images")

    # stop early if nothing found
    if found == 0:
        log("ERROR: No preprocessed files found.")
        return

    # drops rows without files
    df = df[df["filepath"].notna()].reset_index(drop=True)

    # print ratio for imbalance statistic
    log("total usable images:", len(df))
    log("melanoma ratio:", round(df["label"].mean(), 3))

    # catches broken paths and corrupt images
    verify_data_pipeline(df, sample_size=20)

    # build selected backbone model (EffNet or MobileNet) + preprocess function + backbone handle
    model, preprocess_fn, base = build_model(
        args.img_size, args.learning_rate, backbone_name=cli.backbone
    )

    # builds batched tf.data pipelines with optional positive oversampling
    ds_train, ds_val, ds_test = make_datasets(
        df=df,
        img_size=args.img_size,
        batch=args.batch,
        seed=args.seed,
        oversample_pos_ratio=args.oversample_pos_ratio,
        preprocess_fn=preprocess_fn,
    )

    # compute inverse frequency weights so each class contributes equally
    pos_cnt = int(df["label"].sum())
    neg_cnt = len(df) - pos_cnt
    total = len(df)
    class_weight = {0: (1 / neg_cnt) * (total / 2.0), 1: (1 / pos_cnt) * (total / 2.0)}
    log(f"class_weight: {class_weight}")

    # two different checkpoint files to keep best weights from each stage
    ckpt1, ckpt2 = str(out_dir / "best_phase1.keras"), str(
        out_dir / "best_phase2.keras"
    )

    # early stopping on val PR-AUC, restore the best weights
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(  # halves LR on PR-AUC stalls to escape flat minima
            monitor="val_auc_pr",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(  # saves the best val PR-AUC model for the current phase
            filepath=ckpt1,
            monitor="val_auc_pr",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    log("Phase 1: training classifier head (base frozen)")

    # warm up the classifier head while backbone is frozen
    h1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.warmup_epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    log("Phase 2: fine-tuning with BN frozen and partial unfreeze")

    # partial unfreeze = adapt high level features without overfitting, freeze BN to keep stats stable
    unfrozen = partial_unfreeze_with_bn_freeze(base, unfreeze_non_bn_last_k=40)
    log(f"Unfroze last {unfrozen} non-BN layers in backbone")

    # recompile for fine tuning
    loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.ft_learning_rate),
        loss=loss,
        steps_per_execution=32,
    )

    # swap checkpoint path to phase 2 file
    callbacks[2] = keras.callbacks.ModelCheckpoint(
        filepath=ckpt2, monitor="val_auc_pr", save_best_only=True, mode="max", verbose=1
    )

    # run fine-tuning epochs with same early stop/LR schedule
    h2 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.main_epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # visual check
    plot_train_curves(h1, h2, out_dir)

    # pick decision threshold τ on validation
    log(f"Optimizing τ for recall ≥ {args.target_recall:.2f} on validation")
    y_val, p_val = [], []
    for x, y in ds_val:
        y_val.append(y.numpy())
        p_val.append(model.predict(x, verbose=0))
    y_val = np.vstack(y_val).ravel()
    p_val = np.vstack(p_val).ravel()

    # sweep thresholds to find the smallest τ achieving target recall
    tau, precisions, recalls, thresholds = choose_threshold(
        y_val, p_val, args.target_recall
    )
    log(f"Chosen τ: {tau:.4f}")

    # save PR/ROC plots and compute metrics at chosen τ
    plot_threshold_sweep(y_val, p_val, out_dir)
    plot_pr_roc(y_val, p_val, "VAL", out_dir)
    val_metrics = comprehensive_report("VALIDATION", y_val, p_val, tau)

    # evaluate once on test using validation selected τ, save plots and confusion matrices for both splits
    y_test, p_test = [], []
    for x, y in ds_test:
        y_test.append(y.numpy())
        p_test.append(model.predict(x, verbose=0))
    y_test = np.vstack(y_test).ravel()
    p_test = np.vstack(p_test).ravel()
    plot_pr_roc(y_test, p_test, "TEST", out_dir)
    test_metrics = comprehensive_report("TEST", y_test, p_test, tau)
    plot_confusion(y_val, p_val, tau, "VAL", out_dir)
    plot_confusion(y_test, p_test, tau, "TEST", out_dir)

    # save keras file
    log("Saving models/exports")
    final_keras = out_dir / "classifier_final.keras"
    model.save(str(final_keras))
    log(f"Saved: {final_keras}")

    # export a TF savedmodel
    try:
        save_root = out_dir / "saved_model"
        model.export(str(save_root))
        log(f"SavedModel: {save_root}")
    except Exception as e:
        log(f"Export warning: {e}")

    # put key metrics in a CSV
    pd.DataFrame({"validation": val_metrics, "test": test_metrics}).T.to_csv(
        out_dir / "results_summary.csv"
    )

    # summary in console
    log("Training Complete!")
    log(
        f"VAL recall:  {val_metrics['recall']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}"
    )
    log(
        f"TEST recall: {test_metrics['recall']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
