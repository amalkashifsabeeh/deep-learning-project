"""Baselines that do not learn anything.

Each one takes images (N, 28, 28) with values between 0 and 1 and boolean masks (True = missing).
Each returns full images with the visible pixels left alone, so they are scored exactly like the model.
"""
import cv2
import numpy as np


def _paste(images, masks, fill):
    return np.where(masks, fill, images)


def mean_fill(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Fill the hole with the average of that image's visible pixels."""
    vis = ~masks
    m = (images * vis).sum((1, 2)) / np.maximum(vis.sum((1, 2)), 1)
    return _paste(images, masks, m[:, None, None])


def mean_image_fill(images: np.ndarray, masks: np.ndarray, mean_image: np.ndarray) -> np.ndarray:
    """Fill the hole with the average training image, pixel by pixel. This is a guess of a typical garment."""
    return _paste(images, masks, mean_image[None])


def telea_fill(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Classical inpainting from OpenCV (Telea). It fills the hole from nearby pixels and learns nothing."""
    out = np.empty_like(images)
    for i in range(len(images)):
        u8 = np.clip(images[i] * 255.0, 0, 255).astype(np.uint8)
        filled = cv2.inpaint(u8, masks[i].astype(np.uint8), 3, cv2.INPAINT_TELEA)
        out[i] = filled.astype(np.float32) / 255.0
    return _paste(images, masks, out)


# I left the mean image out of this list, because it needs the training average, so the caller handles it
BASELINES = {"mean_fill": mean_fill, "telea": telea_fill}
