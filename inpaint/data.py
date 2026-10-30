"""Loading Fashion-MNIST.

I download the raw files straight from the dataset's GitHub repo, so I do not need any extra library.
"""
import gzip
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {"train": "train-images-idx3-ubyte.gz", "test": "t10k-images-idx3-ubyte.gz"}
EXPECTED_N = {"train": 60000, "test": 10000}


def load_fashion_mnist(split: str, root: str = "data") -> np.ndarray:
    """Return the images as an array of shape (N, 28, 28), with pixel values between 0 and 1."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / FILES[split]
    if not path.exists():
        print(f"Downloading {FILES[split]} ...")
        urllib.request.urlretrieve(BASE_URL + FILES[split], path)
    with gzip.open(path, "rb") as f:
        # the first 16 bytes of the file are a header, not pixels, so I skip them
        raw = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    images = raw.reshape(-1, 28, 28)
    # I check the size so a broken download fails right away instead of giving odd results later
    assert images.shape[0] == EXPECTED_N[split], f"unexpected {split} size {images.shape}"
    return images.astype(np.float32) / 255.0


def train_val_split(images: np.ndarray, n_val: int = 5000):
    """I hold out the last n_val training images for validation.

    The test set stays untouched until the final evaluation.
    """
    return images[:-n_val], images[-n_val:]
