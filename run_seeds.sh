#!/bin/bash
# Trains seeds 1 and 2 for each mask type. Seed 0 is the run I already have in runs/<kind>/.
# On my laptop this takes about 10 minutes per run, so about an hour in total. Optional.
set -e
for s in 1 2; do
  for k in rect random blocks; do
    python train.py --kind $k --seed $s --epochs 12 --out runs/${k}_seed${s}
  done
done
