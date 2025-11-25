#!/bin/bash

python scripts/train_spectrum_classifier.py \
    --parquet-path data/train_spectra.parquet \
    --checkpoint-path checkpoints/specformer/specformer.ckpt \
    --batch-size 32 \
    --max-epochs 50 \
    --devices 2 \
    --num-workers 0 \
    --freeze-backbone