"""Metrics for each image. The prediction must already have the visible pixels pasted back in.

The hole metrics only look at the missing pixels. The full-image metrics are inflated, because the
visible pixels are copied through perfectly, so I treat the hole metrics as the honest comparison.
"""
import numpy as np
from skimage.metrics import structural_similarity


def psnr_from_mse(mse):
    # the largest possible pixel value is 1, so PSNR = 10 log10(1 / MSE); the floor stops log(0) for perfect images
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))


def hole_mse(pred, target, masks):
    # squared error summed over the hole only, divided by the number of hole pixels in that image
    return (((pred - target) ** 2) * masks).sum((1, 2)) / np.maximum(masks.sum((1, 2)), 1)


def full_mse(pred, target):
    return ((pred - target) ** 2).mean((1, 2))


def ssim_full_and_hole(pred, target, masks):
    """SSIM over the whole image, and the average of the SSIM map over the hole pixels only."""
    full = np.empty(len(pred))
    hole = np.empty(len(pred))
    for i in range(len(pred)):
        s, smap = structural_similarity(target[i], pred[i], data_range=1.0, full=True)
        full[i] = s
        # SSIM looks at small windows, so I take the full map and average it over the hole pixels
        hole[i] = smap[masks[i]].mean() if masks[i].any() else np.nan
    return full, hole


def all_metrics(pred, target, masks) -> dict:
    """A dict with one array per metric, with one value per image."""
    mh = hole_mse(pred, target, masks)
    mf = full_mse(pred, target)
    sf, sh = ssim_full_and_hole(pred, target, masks)
    return {
        "mse_hole": mh, "psnr_hole": psnr_from_mse(mh), "ssim_hole": sh,
        "mse_full": mf, "psnr_full": psnr_from_mse(mf), "ssim_full": sf,
    }
