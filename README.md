# Image inpainting: how do the amount and type of missing information affect reconstruction?

**Read this first.** This is my Deep Learning course project. I train a small convolutional autoencoder
(PyTorch) to fill in missing pixels of Fashion-MNIST images. I want to know how reconstruction changes when
more of the image is missing (10 to 70%) and when the missing part has a different shape (one rectangle,
scattered random pixels, or four blocks). I compare the autoencoder against simple baselines that do not learn.

- Screen recording: **[ADD LINK]**
- Try it yourself: open `demo/index.html` in a browser (double click it, nothing to install). Pick an image, draw over it, and see what each trained model guesses.
- Status: the pipeline, the evaluation, the sanity checks, the seed comparison and the interactive demo all work end to end. I still have to try skip connections and study the size of the bottleneck (see "Next steps").

## Where to look (2 minutes)

| If you want to see | Open |
|---|---|
| An interactive version of the project | `demo/index.html` (open it in a browser) |
| The main result | `results/degradation_psnr_hole.png` |
| What the outputs look like | `results/recon_rect.png`, `results/recon_random.png`, `results/recon_blocks.png` |
| The training curves | `runs/rect/loss_curve.png` (also `random` and `blocks`) |
| The numbers behind the plots | `results/summary.md` and `results/metrics.csv` |
| Train on one mask type, test on another | `results/cross_mask_generalisation.png` |
| How much training randomness matters | `results/seeds_summary.md`, `results/degradation_psnr_hole_seeds.png` |
| My bug checks | `results/sanity.txt` |

## Main result so far

This is PSNR on the missing pixels only (in dB, higher is better). I used 2000 held-out test images, and every
method sees the same masks. The last column is the autoencoder trained on that same mask type.

| Mask type | % missing | Visible-pixel mean | Mean image | Telea | Autoencoder |
|---|---|---|---|---|---|
| rect | 10 | 10.13 | 10.57 | 15.16 | 19.68 |
| rect | 25 | 9.33 | 10.31 | 12.73 | 17.44 |
| rect | 50 | 8.76 | 10.26 | 10.97 | 16.31 |
| rect | 69 | 8.53 | 10.43 | 10.23 | 15.16 |
| random | 10 | 10.24 | 10.98 | 17.06 | 21.06 |
| random | 25 | 10.23 | 10.98 | 16.37 | 20.69 |
| random | 50 | 10.22 | 10.98 | 15.72 | 20.19 |
| random | 70 | 10.21 | 10.98 | 15.18 | 19.21 |
| blocks | 8 | 10.57 | 10.87 | 16.13 | 20.86 |
| blocks | 25 | 10.36 | 10.48 | 14.18 | 19.25 |
| blocks | 51 | 10.04 | 10.49 | 13.09 | 18.19 |
| blocks | 73 | 9.84 | 10.67 | 12.50 | 17.30 |

What I take from this so far:
- The autoencoder beats every baseline at all 12 settings.
- Quality gets worse gradually as more is missing. I do not see a sharp drop at the levels I tested.
- At about 50% missing, random holes are the easiest and one rectangle is the hardest. My guess is that scattered holes leave some nearby context everywhere, while a rectangle removes it completely. I have not tested this yet.
- Going from one mask type to another is not symmetric. The model trained on random pixels does no better than the mean image baseline on rectangles (about 10 dB), but the model trained on blocks does well on random holes (about 18 dB).

Things to be careful about: the shaded bands in the main plot (95% over test images) do not include the variation between
training runs, which I measure in the next section. The block masks are 8 to 73% missing, not exactly 10 to 70%, so the
plots use the fraction I actually got.

## How much does training randomness matter?

I trained each model with 3 different seeds (0, 1 and 2) and scored all of them on the same test images and masks.
The table shows the autoencoder as the mean and standard deviation over the 3 seeds, with the lowest and highest seed in brackets.
I ran all 3 seeds on my MacBook. Seed 0 is the same run as in the table above.

| Mask type | % missing | Mean image | Telea | Autoencoder |
|---|---|---|---|---|
| rect | 10 | 10.57 | 15.16 | 19.64 +/- 0.06 (19.56 to 19.68) |
| rect | 25 | 10.31 | 12.73 | 17.31 +/- 0.11 (17.24 to 17.44) |
| rect | 50 | 10.26 | 10.97 | 16.27 +/- 0.05 (16.22 to 16.31) |
| rect | 69 | 10.43 | 10.23 | 15.14 +/- 0.02 (15.11 to 15.16) |
| random | 10 | 10.98 | 17.06 | 20.96 +/- 0.21 (20.72 to 21.10) |
| random | 25 | 10.98 | 16.37 | 20.59 +/- 0.20 (20.36 to 20.72) |
| random | 50 | 10.98 | 15.72 | 20.10 +/- 0.17 (19.90 to 20.20) |
| random | 70 | 10.98 | 15.18 | 19.13 +/- 0.13 (18.98 to 19.21) |
| blocks | 8 | 10.87 | 16.13 | 20.85 +/- 0.03 (20.82 to 20.87) |
| blocks | 25 | 10.48 | 14.18 | 19.25 +/- 0.01 (19.25 to 19.26) |
| blocks | 51 | 10.49 | 13.09 | 18.13 +/- 0.06 (18.07 to 18.19) |
| blocks | 73 | 10.67 | 12.50 | 17.21 +/- 0.09 (17.13 to 17.30) |

- seed 0: random 20.19 > blocks 18.19 > rect 16.31: yes
- seed 1: random 20.20 > blocks 18.12 > rect 16.22: yes
- seed 2: random 19.90 > blocks 18.07 > rect 16.27: yes

What I take from this:
- The seeds barely move the results. The standard deviation is between 0.01 and 0.21 dB, and the largest spread is for random pixels.
- The order random > blocks > rectangle at about 50% missing holds in all 3 seeds. The gaps between mask types are about 2 dB, which is roughly ten times the largest spread between seeds, so this is not training noise.
- The autoencoder stays well above both baselines in every seed, so the main result does not depend on a lucky run.
- With only 3 seeds the standard deviation itself is rough, so I treat it as a size estimate and not as a precise number.

## Interactive demo

`demo/index.html` is a single web page that runs my three trained autoencoders in the browser. I pick one of 20 test images,
hide pixels by dragging over the second image or by choosing a shape and a share, and see the mean image fill and the
autoencoder side by side, with the PSNR on the hidden pixels. I can also switch which model fills the hole. For example, a
rectangle with the model trained on random pixels shows the cross-mask gap from the results above.

- To open it, double click `demo/index.html`. It needs no install and no internet connection (the fonts fall back to system fonts).
- The page uses the models in `runs/`. After I retrain, I rebuild it with `python make_demo.py`, which takes a few seconds.
- I checked the in-browser model against PyTorch on 54 image and mask combinations. The outputs agree to within 1e-6.
- The Telea baseline is not in the demo, because it comes from OpenCV. The demo compares against the mean image fill.

## How to run it

You need Python 3.9 or newer and an internet connection for the first run (it downloads Fashion-MNIST, about 30 MB).
No GPU is needed. I ran everything on a MacBook CPU.

### 1. Set up (once, about 5 minutes)

Open a terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

After the second line the terminal line starts with `(.venv)`. PyTorch is a large download, so the last line can take a few minutes.
Every time I open a new terminal I need to run the `source` line again.

### 2. Quick check using the trained models already in `runs/` (about 2 minutes)

```bash
python sanity.py
python evaluate.py
```

- `sanity.py` downloads Fashion-MNIST into `data/` on the first run, takes about 15 seconds, and ends with `all hard checks passed` after 19 PASS lines.
- `evaluate.py` takes under a minute. It prints the results table and writes the tables and figures into `results/`.
  The first rows should look like `| rect | 10 | 10.13 | 10.57 | 15.16 | 19.68 |` and `| rect | 50 | 8.76 | 10.26 | 10.97 | 16.31 |`.
  The last column is the autoencoder and can differ from mine by about 0.1 dB if I retrain.

### 3. Open the interactive demo (a few seconds)

```bash
open demo/index.html               # on Linux: xdg-open demo/index.html, on Windows: start demo/index.html
```

It opens in the browser. I pick an image, drag over the second picture to hide pixels, and switch between the three models.
To rebuild it from the models in `runs/`, I run `python make_demo.py` (it downloads one small label file the first time).

### 4. Train the models again (optional, about 30 minutes)

```bash
python train.py --kind rect   --epochs 12 --out runs/rect
python train.py --kind random --epochs 12 --out runs/random
python train.py --kind blocks --epochs 12 --out runs/blocks
python evaluate.py
python sanity.py                   # now also checks the results themselves, expect [ok] lines
python make_demo.py                # so the demo uses the new models
```

Each run prints one line per epoch and takes about 10 minutes on my MacBook. After epoch 12 the validation hole-MSE should be about
0.030 (rect), 0.012 (random) and 0.020 (blocks), and each run ends with `saved to runs/<kind>/`.
Training is not exactly the same on every machine. I trained everything twice on my MacBook and once on a cloud machine, and the hole PSNR stayed within 0.1 dB.

### 5. Extra seeds (optional, about an hour)

```bash
bash run_seeds.sh                  # trains seeds 1 and 2 for each mask type, 6 runs
python seeds.py                    # compares all 3 seeds, writes results/seeds_summary.md
```

The last line should say `The order random > blocks > rect holds in 3 of 3 seeds.`

### If something goes wrong

- `No module named torch` (or `cv2`, `skimage`, `pandas`): the environment is not active. Run the `source .venv/bin/activate` line again.
- `pip install` fails on torch: install PyTorch for my system from pytorch.org first, then run `pip install -r requirements.txt` again.
- The download of Fashion-MNIST fails: check the internet connection and run the command again. The files go into `data/`.
- `python` is not found on a Mac: use `python3` instead.

## Code map

```
inpaint/masks.py       the three mask generators (True means missing), where I choose the fraction
inpaint/models.py      InpaintAE: corrupted image plus mask as input, a 64-number bottleneck, then the image
inpaint/baselines.py   visible-pixel mean, mean training image, and Telea (classical inpainting from OpenCV)
inpaint/metrics.py     MSE, PSNR and SSIM, on the missing pixels and on the full image
inpaint/plots.py       the figures
train.py               trains one model per mask type
evaluate.py            scores the models and baselines on the same held-out masks
seeds.py               scores the matching model for several training seeds (the seeds come from run_seeds.sh)
make_demo.py           builds demo/index.html from the trained models in runs/
demo/template.html     the page itself, before the weights are put in
run_seeds.sh           trains seeds 1 and 2 for each mask type
sanity.py              my bug checks (mask fractions, no leaking of the hole contents, overfitting one batch, ...)
```

## Decisions I made, and why

- **Metrics on the hole only.** I paste the visible pixels back into every output, so scores on the full image look better than they should, because nobody had to predict those pixels. My main metrics (`*_hole`) only score the missing pixels.
- **The model sees the mask.** The input is the corrupted image, with the hole zeroed out, plus the mask as a second channel. Without the mask, a hole on the black background would look the same as a real black pixel.
- **The loss** is the MSE on the hole pixels plus 0.1 times the MSE on the visible pixels.
- **New masks at every step.** I draw the missing fraction between 0.1 and 0.7, so the model trains on the whole range.
- **Splits.** Validation is the last 5000 training images. I only touch the test set in `evaluate.py`.

## Next steps

1. Add a U-Net version with skip connections and see whether it changes where quality breaks down.
2. Change the size of the bottleneck (16, 64, 256) and check whether the garment class can be read from it at 70% missing. This is the part about what the representation keeps.
3. Test my guess about context: compare random holes and rectangles at the same fraction, while controlling for how much of the garment is hidden.
4. Repeat the main experiment on CIFAR-10 if I have time.

For each new model I will train 3 seeds, since the spread above is small but not zero.

I am leaving out LPIPS for now.
