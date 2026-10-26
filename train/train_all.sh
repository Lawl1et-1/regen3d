#!/usr/bin/env bash
# Train, analyse and export every organ in turn (step 7). Run from the repo root.
set -e
for organ in "$@"; do
  mkdir -p train/runs/$organ
  echo "=== $organ: training $(date)"
  .venv/Scripts/python train/train.py --organ $organ --iters 2500 --batch 4 --ckpt --snap-every 250 --damage-start 0 > train/runs/$organ/train.log 2>&1
  echo "=== $organ: results $(date)"
  .venv/Scripts/python eval/brain_channels.py $organ | tail -1
  .venv/Scripts/python eval/liver_results.py $organ | tail -1
  .venv/Scripts/python train/export.py $organ
done
echo "=== all done $(date)"
