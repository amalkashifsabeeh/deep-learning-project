"""Build demo/index.html, a web page that runs my trained autoencoders in the browser.

Example:  python make_demo.py

It reads the trained models in runs/rect, runs/random and runs/blocks, and puts their weights into
demo/template.html together with 20 test images and the mean training image. The result is one file
with no other dependencies, so I can open it by double clicking it, and nothing needs to be installed.
Run this again after retraining, so the page uses the new models.
Use --content-only for a page without the <html> wrapper (that is what I publish as a hosted page).
"""
import argparse
import base64
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

from evaluate import load_model
from inpaint.data import BASE_URL, load_fashion_mnist, train_val_split

# I put the weights into the page as one long float32 list, in this order, and the page reads them in the same order
KEYS = ["encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias",
        "encoder.4.weight", "encoder.4.bias", "encoder.7.weight", "encoder.7.bias",
        "decoder_fc.0.weight", "decoder_fc.0.bias",
        "decoder.0.weight", "decoder.0.bias", "decoder.2.weight", "decoder.2.bias",
        "decoder.4.weight", "decoder.4.bias"]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(a.tobytes()).decode()


def weights_b64(model) -> str:
    sd = model.state_dict()
    flat = np.concatenate([sd[k].detach().cpu().numpy().astype("<f4").ravel() for k in KEYS])
    return b64(flat)


def load_labels(root: str = "data") -> np.ndarray:
    path = Path(root) / "t10k-labels-idx1-ubyte.gz"
    if not path.exists():
        print("Downloading test labels ...")
        urllib.request.urlretrieve(BASE_URL + "t10k-labels-idx1-ubyte.gz", path)
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)  # I skip the first 8 bytes because they are a header


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/rect", "runs/random", "runs/blocks"])
    p.add_argument("--out", default="demo/index.html")
    p.add_argument("--content-only", action="store_true")
    p.add_argument("--data", default="data")
    args = p.parse_args()

    weights, latent = {}, None
    for r in args.runs:
        kind, model = load_model(r)
        weights[kind] = weights_b64(model)
        latent = model.latent_dim if latent is None else latent
        assert model.latent_dim == latent, "all models need the same bottleneck size"

    train, _ = train_val_split(load_fashion_mnist("train", args.data))
    mean_image = train.mean(0).astype("<f4")
    test = load_fashion_mnist("test", args.data)
    labels = load_labels(args.data)

    # I take two test images from each of the 10 classes (the first two I find for each)
    samples = []
    for cls in range(10):
        for i in np.where(labels == cls)[0][:2]:
            px = np.round(test[i] * 255).astype(np.uint8)
            samples.append({"cls": int(cls), "px": b64(px)})

    data = {"latent": int(latent), "weights": weights, "samples": samples, "meanImage": b64(mean_image)}
    page = Path("demo/template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    if not args.content_only:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
                + page + "\n</body>\n</html>\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) with models {list(weights)}")


if __name__ == "__main__":
    main()
