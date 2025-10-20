# Melanoma Classification using EfficientNetV2 / MobileNetV3

This project trains deep learning models to detect melanoma from dermoscopic images.
It supports multiple backbones (EfficientNetV2B0 and MobileNetV3Small) and uses transfer learning, fine-tuning, and threshold optimization for high recall.

## To Train a Model

### EfficientNetV2B0 (default)

python -m src.train --backbone effv2b0

### MobileNetV3Small

python -m src.train --backbone mnetv3s

## To Export a Model to TFLite

python -m src.export_tflite
