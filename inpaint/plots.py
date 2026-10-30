"""Plotting helpers.

The colours follow a fixed order (blue, orange, aqua, yellow) that I checked for colour blindness.
Every series also gets its own marker, so no line is told apart by colour alone.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

METHOD_STYLE = {  # color, marker, label
    "model": (BLUE, "o", "Autoencoder"),
    "telea": (ORANGE, "s", "Telea (classical)"),
    "mean_image": (AQUA, "^", "Mean image"),
    "mean_fill": (YELLOW, "D", "Visible-pixel mean"),
}
KIND_TITLE = {"rect": "One rectangle", "random": "Random pixels", "blocks": "Four blocks"}
METRIC_LABEL = {
    "psnr_hole": "PSNR on missing pixels (dB)",
    "mse_hole": "MSE on missing pixels",
    "ssim_hole": "SSIM on missing pixels",
    "psnr_full": "PSNR on full image (dB)",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 9,
    })


def plot_loss_curves(history_path, out_path, title=None):
    _style()
    h = json.load(open(history_path))
    ep = np.arange(1, len(h["train_hole_mse"]) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ep, h["train_hole_mse"], color=BLUE, lw=2, marker="o", ms=6, label="Train (hole MSE)")
    ax.plot(ep, h["val_hole_mse"], color=ORANGE, lw=2, marker="s", ms=6, label="Validation (hole MSE)")
    if "val_meanfill_hole_mse" in h:
        base = h["val_meanfill_hole_mse"]
        ax.axhline(base, color=MUTED, lw=1.5, ls="--")
        ax.set_ylim(0, base * 1.18)
        ax.text(ep[0] - 0.3, base * 1.02, "visible-pixel mean baseline", color=INK2,
                fontsize=8, va="bottom", ha="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on missing pixels")
    ax.set_xticks(ep)
    ax.set_title(title or "Training curves")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_degradation(df, out_path, metric="psnr_hole", kinds=("rect", "random", "blocks")):
    """One panel per test mask type, showing quality against the fraction missing.

    Each panel has the matching model and the three baselines.
    """
    _style()
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.9 * len(kinds), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        sub = df[df.test_kind == kind]
        for key in ("telea", "mean_image", "mean_fill", "model"):
            method = f"model:{kind}" if key == "model" else key
            s = sub[sub.method == method].sort_values("frac_nominal")
            if s.empty:
                continue
            color, marker, label = METHOD_STYLE[key]
            x = s.frac_achieved * 100
            y, e = s[f"{metric}_mean"], s[f"{metric}_sem"]
            ax.plot(x, y, color=color, lw=2, marker=marker, ms=6, label=label, zorder=3 if key == "model" else 2)
            ax.fill_between(x, y - 1.96 * e, y + 1.96 * e, color=color, alpha=0.18, lw=0)
        ax.set_title(KIND_TITLE[kind])
        ax.set_xlabel("Pixels missing (%)")
    axes[0].set_ylabel(METRIC_LABEL.get(metric, metric))
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _gray_rgb(img):
    return np.repeat(img[..., None], 3, axis=2)


def plot_recon_grid(originals, masks, preds_by_method, out_path, col_labels, title=None):
    """Rows are the original, the corrupted image (missing pixels in orange), then one row per method."""
    _style()
    rows = [("Original", [_gray_rgb(o) for o in originals])]
    corrupted = []
    for o, m in zip(originals, masks):
        rgb = _gray_rgb(o).copy()
        rgb[m] = (0.92, 0.41, 0.20)
        corrupted.append(rgb)
    rows.append(("Corrupted", corrupted))
    for label, preds in preds_by_method:
        rows.append((label, [_gray_rgb(p) for p in preds]))
    n = len(originals)
    fig, axes = plt.subplots(len(rows), n, figsize=(0.95 * n + 1.3, 0.95 * len(rows) + 0.7))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(np.clip(imgs[c], 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, color=INK2, labelpad=4)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=8, color=INK2, fontweight="normal")
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cross_matrix(df, out_path, frac_nominal, metric="psnr_hole", kinds=("rect", "random", "blocks")):
    """Rows are the mask type the model was trained on. Columns are the mask type at test time."""
    _style()
    M = np.full((len(kinds), len(kinds)), np.nan)
    for i, tr in enumerate(kinds):
        for j, te in enumerate(kinds):
            s = df[(df.method == f"model:{tr}") & (df.test_kind == te) & np.isclose(df.frac_nominal, frac_nominal)]
            if len(s):
                M[i, j] = s[f"{metric}_mean"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(M, cmap=SEQ_BLUE)
    ax.grid(False)
    ax.set_xticks(range(len(kinds)), [KIND_TITLE[k].replace(" ", "\n") for k in kinds], fontsize=8)
    ax.set_yticks(range(len(kinds)), [KIND_TITLE[k] for k in kinds], fontsize=8)
    ax.set_xlabel("Mask type at test time")
    ax.set_ylabel("Mask type at training time")
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(len(kinds)):
        for j in range(len(kinds)):
            if not np.isnan(M[i, j]):
                dark = (M[i, j] - lo) / max(hi - lo, 1e-9) > 0.5
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if dark else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Generalisation across mask types\n(PSNR on missing pixels, ~{int(frac_nominal * 100)}% missing)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
