# tiny logging + path utilities

# imports
from pathlib import Path


def log(*a):
    # print and flush immediately so logs show up even with buffering
    print(*a, flush=True)


def ensure_dir(pathlike):
    # create the directory if it doesn't exist, no error if it does
    Path(pathlike).mkdir(parents=True, exist_ok=True)
