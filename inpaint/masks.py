"""Mask generators. A mask is a boolean array where True means the pixel is missing.

Each generator takes the fraction of the image that should be missing and returns a mask close to it.
Random masks are exact. Rectangles and blocks are rounded to whole pixels, so I always report the
achieved fraction (mask.mean()) and not the one I asked for.
"""
import numpy as np

KINDS = ("rect", "random", "blocks")
SIZE = 28


def rect_mask(frac: float, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """One rectangle at a random position, with a random shape (height/width between 0.5 and 2)."""
    area = frac * size * size
    # uniform in log space, so a wide rectangle is as likely as a tall one
    aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    h = int(min(size, max(1, round(np.sqrt(area * aspect)))))
    w = int(min(size, max(1, round(area / h))))
    top = int(rng.integers(0, size - h + 1))
    left = int(rng.integers(0, size - w + 1))
    m = np.zeros((size, size), dtype=bool)
    m[top:top + h, left:left + w] = True
    return m


def random_mask(frac: float, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Remove exactly round(frac * size^2) pixels, chosen uniformly at random."""
    n = int(round(frac * size * size))
    idx = rng.choice(size * size, size=n, replace=False)
    m = np.zeros(size * size, dtype=bool)
    m[idx] = True
    return m.reshape(size, size)


def blocks_mask(frac: float, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Four equal square blocks, one in each quarter of the image, each at a random spot in its quarter."""
    half = size // 2
    # each block gets a quarter of the missing area; it must fit inside its quarter, so very large fractions are capped
    side = int(min(half, max(1, round(np.sqrt(frac * size * size / 4)))))
    m = np.zeros((size, size), dtype=bool)
    for qr in (0, half):
        for qc in (0, half):
            top = qr + int(rng.integers(0, half - side + 1))
            left = qc + int(rng.integers(0, half - side + 1))
            m[top:top + side, left:left + side] = True
    return m


_GENERATORS = {"rect": rect_mask, "random": random_mask, "blocks": blocks_mask}


def make_mask(kind: str, frac: float, rng: np.random.Generator) -> np.ndarray:
    return _GENERATORS[kind](frac, rng)


def batch_masks(kind: str, fracs, rng: np.random.Generator) -> np.ndarray:
    """Make one mask for each value in fracs. If kind is 'mixed', each mask gets a random kind."""
    out = np.empty((len(fracs), SIZE, SIZE), dtype=bool)
    for i, f in enumerate(fracs):
        k = KINDS[int(rng.integers(0, len(KINDS)))] if kind == "mixed" else kind
        out[i] = make_mask(k, float(f), rng)
    return out
