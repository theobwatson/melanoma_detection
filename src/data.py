# imports
import random, numpy as np, tensorflow as tf, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from .utils import log


# turns image name into file path for mapping
def id_to_preproc_path(img_id: str, pre_dir: Path):
    stem = (
        str(img_id).split("_downsampled")[0].split(".")[0]
    )  # strips suffixes and extensions
    for c in [
        pre_dir / f"{stem}.jpg",
        pre_dir / f"{stem}.jpeg",
        pre_dir / f"ISIC_{stem}.jpg",
        pre_dir / f"ISIC_{stem}.jpeg",
    ]:
        if c.exists():  # return first direct match if it exists
            return str(c)
    try:
        # crawl for partial matches
        for fp in pre_dir.rglob(f"*{stem}*"):
            if fp.suffix.lower() in (".jpg", ".jpeg"):
                return str(fp)
    except Exception as e:
        log(f"Warn searching {stem}: {e}")
    return None


# decode imae into a rank 3 RGB tensor
def decode_any_image(bytestr):
    img = tf.io.decode_image(bytestr, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def verify_data_pipeline(df, sample_size=20):

    # try loading small sample to catch broken paths
    ok, bad = 0, []
    sample = df[df["filepath"].notna()].sample(
        min(sample_size, df["filepath"].notna().sum()), random_state=0
    )

    for _, row in sample.iterrows():
        try:
            b = tf.io.read_file(row["filepath"])  # read raw bytes
            img = decode_any_image(b)  # decode any common image format

            # count as ok if it has three channels and positive spatial dims
            ok += int(img.shape[-1] == 3 and img.shape[0] > 0 and img.shape[1] > 0)
        except Exception as e:
            bad.append(f"{row['filepath']} :: {e}")  # record failures for debugging
    log(f"Data verification: {ok}/{len(sample)} OK")
    if bad:
        log("Sample failures:", bad[:3])  # print up to three example errors
    return ok == len(sample)  # True only if all sampled images loaded fine


# 80/10/10 split using label column to preserve class ratio
def split_indices(df, seed):
    idx = np.arange(len(df))
    tr, tmp = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=df["label"].values
    )
    va, te = train_test_split(
        tmp, test_size=0.5, random_state=seed, stratify=df.iloc[tmp]["label"].values
    )
    return tr, va, te


def make_datasets(df, img_size, batch, seed, oversample_pos_ratio, preprocess_fn):

    # builds three tf.data pipelines with optional positive oversampling
    AUTOTUNE = tf.data.AUTOTUNE

    # lightweight geometric augmentation for training
    geo_aug = keras.Sequential(
        [
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomZoom(0.1),
            layers.RandomFlip("horizontal_and_vertical"),
        ],
        name="geo_aug",
    )

    def _load(path, label):
        # read, decode, and resize
        b = tf.io.read_file(path)
        img = decode_any_image(b)
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        img = tf.cast(img, tf.float32)
        label = tf.cast(label, tf.float32)
        return img, tf.expand_dims(label, -1)  # ensure label shape (1,)

    # apply random geometric transforms only in training
    def _aug(img, label):
        return geo_aug(img, training=True), label

    # model/backbone specific normalisation
    def _pre(img, label):
        return preprocess_fn(img), label

    # split once, then matrialise filepaths for each split
    tr, va, te = split_indices(df, seed)

    # extract parallel lists given indices
    def pick(ix):
        return (
            df["filepath"].iloc[ix].tolist(),
            df["label"].iloc[ix].astype("float32").tolist(),
        )

    tr_p, tr_y = pick(tr)
    va_p, va_y = pick(va)
    te_p, te_y = pick(te)

    # oversample positives in the train split to reach target positive ratio
    if oversample_pos_ratio and oversample_pos_ratio > 0:
        tp = [(p, l) for p, l in zip(tr_p, tr_y) if l == 1.0]  # positive pairs
        tn = [(p, l) for p, l in zip(tr_p, tr_y) if l == 0.0]  # negative pairs
        n_pos, n_neg = len(tp), len(tn)
        if n_pos > 0:

            # compute replication factor k so pos fraction = oversample_pos_ratio
            k = int(
                np.ceil(
                    (oversample_pos_ratio * n_neg)
                    / (n_pos * (1 - oversample_pos_ratio))
                )
            )
            pairs = tn + tp * max(1, k)  # replicate positives
            random.Random(seed).shuffle(pairs)  # deterministic shuffle
            tr_p, tr_y = map(list, zip(*pairs))  # unpack back into lists
            log(
                f"Oversampling: pos x{k} -> train {len(tr_p)} (~pos {oversample_pos_ratio:.2f})"
            )
        else:
            log("Oversampling requested but no positives; skipping.")

    # pipeline: load -> augment -> preprocess -> shuffle -> batch -> prefetch
    def _pipe(paths, labels, train=False, aug=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(_load, AUTOTUNE)
        if train and aug:
            ds = ds.map(_aug, AUTOTUNE)
        ds = ds.map(_pre, AUTOTUNE)
        if train:
            ds = ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch).prefetch(AUTOTUNE)

    # train uses aug and shuffle
    return _pipe(tr_p, tr_y, True, True), _pipe(va_p, va_y), _pipe(te_p, te_y)
