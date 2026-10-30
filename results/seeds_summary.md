Hole PSNR (dB). Autoencoder: mean +/- standard deviation over 3 training seeds (min to max in brackets).

| Mask type | % missing | Mean image | Telea | Autoencoder |
|---|---|---|---|---|
| rect | 10 | 10.57 | 15.16 | 19.62 +/- 0.05 (19.56 to 19.66) |
| rect | 25 | 10.31 | 12.73 | 17.30 +/- 0.12 (17.17 to 17.41) |
| rect | 50 | 10.26 | 10.97 | 16.26 +/- 0.08 (16.17 to 16.31) |
| rect | 69 | 10.43 | 10.23 | 15.14 +/- 0.04 (15.10 to 15.17) |
| random | 10 | 10.98 | 17.06 | 20.96 +/- 0.15 (20.79 to 21.08) |
| random | 25 | 10.98 | 16.37 | 20.61 +/- 0.15 (20.44 to 20.74) |
| random | 50 | 10.98 | 15.72 | 20.11 +/- 0.14 (19.94 to 20.22) |
| random | 70 | 10.98 | 15.18 | 19.16 +/- 0.12 (19.03 to 19.27) |
| blocks | 8 | 10.87 | 16.13 | 20.88 +/- 0.11 (20.79 to 21.00) |
| blocks | 25 | 10.48 | 14.18 | 19.22 +/- 0.04 (19.18 to 19.26) |
| blocks | 51 | 10.49 | 13.09 | 18.13 +/- 0.04 (18.09 to 18.16) |
| blocks | 73 | 10.67 | 12.50 | 17.23 +/- 0.07 (17.15 to 17.29) |

Ranking of mask types at about 50% missing, one line per seed (hole PSNR of the matching model):

- seed 0: random 20.16 > blocks 18.16 > rect 16.31: yes
- seed 1: random 20.22 > blocks 18.14 > rect 16.17: yes
- seed 2: random 19.94 > blocks 18.09 > rect 16.29: yes

The order random > blocks > rect holds in 3 of 3 seeds.
