# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _resolve_mobilenetv3():
    # Try common TF-Keras export
    try:
        from tensorflow.keras.applications import MobileNetV3Small as MNetV3Small
        from tensorflow.keras.applications.mobilenet_v3 import (
            preprocess_input as mnet_pre,
        )

        # return contructor and its preproces function
        return MNetV3Small, mnet_pre
    except Exception:
        pass

    # Try standalone Keras 3
    try:
        from keras.applications import MobileNetV3Small as MNetV3Small  # type: ignore
        from keras.applications.mobilenet_v3 import preprocess_input as mnet_pre  # type: ignore

        return MNetV3Small, mnet_pre
    except Exception:
        pass

    # Fallback via attributes
    try:
        MNetV3Small = tf.keras.applications.MobileNetV3Small
        from tensorflow.keras.applications import mobilenet_v3 as mnet_module

        mnet_pre = mnet_module.preprocess_input
        return MNetV3Small, mnet_pre
    except Exception as e:
        raise ImportError("Could not import MobileNetV3Small. ") from e


def get_backbone_and_preprocess(name, img_size):

    # choose backbone and its preprocess function based on the string name
    if name == "effv2b0":
        from tensorflow.keras.applications.efficientnet_v2 import (
            EfficientNetV2B0,
            preprocess_input,
        )

        base = EfficientNetV2B0(
            include_top=False,
            pooling="avg",
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
        return base, preprocess_input

    if name == "effv2b1":
        from tensorflow.keras.applications.efficientnet_v2 import (
            EfficientNetV2B1,
            preprocess_input,
        )

        base = EfficientNetV2B1(
            include_top=False,
            pooling="avg",
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
        return base, preprocess_input

    if name == "mnetv3s":
        MNetV3Small, mnet_pre = _resolve_mobilenetv3()
        base = MNetV3Small(
            include_top=False,
            pooling="avg",
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
        return base, mnet_pre

    raise ValueError("Unknown backbone")


def build_model(img_size, learning_rate, backbone_name):

    # build the chosen backbone and get its preprocess function
    base, preprocess_fn = get_backbone_and_preprocess(backbone_name, img_size)

    # phase 1, keep the backbone frozen while training the classifier head
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32)
    x = base(inputs)

    # three layer MLP head with BN and dropout for regularisation
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # single-node sigmoid for melanoma probability
    out = layers.Dense(1, activation="sigmoid", name="mel_prob")(x)

    # compile with PR/ROC AUC, and precision/recall/accuracy with light label smoothing
    model = keras.Model(inputs, out, name=f"melanoma_{backbone_name}")
    loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            keras.metrics.AUC(name="auc_pr", curve="PR"),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.BinaryAccuracy(name="acc"),
        ],
        steps_per_execution=32,
    )

    # return the compiled model, the preprocess function, and the frozen backbone
    return model, preprocess_fn, base


def freeze_batchnorm(layer):
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


def partial_unfreeze_with_bn_freeze(base, unfreeze_non_bn_last_k=40):

    # phase 3 set up, unfreeze only the last K non-BN layers in the backbone
    base.trainable = True

    # freeze everything first, but always freeze BN
    for l in base.layers:
        freeze_batchnorm(l)
        l.trainable = False

    # unfreeze last K non-BN layers
    c = 0
    for l in reversed(base.layers):
        if not isinstance(l, layers.BatchNormalization):
            l.trainable = True
            c += 1
            if c >= unfreeze_non_bn_last_k:
                break

    # return how many non-BN layers were unfroze
    return c
