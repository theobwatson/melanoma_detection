from dataclasses import dataclass


@dataclass
class Args:
    # data
    csv: str = "./data/input/ground_truth.csv"
    preproc_dir: str = "./data/output/preprocessed/"

    # output dir is set dynamically in train.py based on backbone
    out: str = ""

    # image + batch
    img_size: int = 224
    batch: int = 32

    # schedule
    seed: int = 42
    warmup_epochs: int = 3
    main_epochs: int = 12

    # optimization
    learning_rate: float = 1e-3
    ft_learning_rate: float = 3e-5

    # imbalance
    oversample_pos_ratio: float = 0.15

    # operating point
    target_recall: float = 0.70

    # performance
    mixed_precision: bool = True
