# imports
from pathlib import Path
import tensorflow as tf

# paths for mnetv3s
SAVED_MODEL_DIR = Path("../runs/mnetv3s/saved_model")
OUT_TFLITE = Path("../runs/mnetv3s/export_android/classifier_fp16.tflite")

# paths for effv2b0
SAVED_MODEL_DIR = Path("../runs/effv2b0/saved_model")
OUT_TFLITE = Path("../runs/effv2b0/export_android/classifier_fp16.tflite")

if __name__ == "__main__":

    # ensure output folder exists
    OUT_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting from: {SAVED_MODEL_DIR}")

    # create a converter that reads the SavedModel directory
    conv = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))

    # enable post-training optimisations
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    # request fp16 weights
    conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # perform conversion
    tflite = conv.convert()

    # write the .tflite file
    OUT_TFLITE.write_bytes(tflite)

    print(f"Wrote: {OUT_TFLITE}")
