#!/bin/bash
python main.py \
  --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --train_percent 0.5 \
  --val_percent 0.5 \
  --num_epochs 5 \
  --batch_size_train 4 \
  --log_level DEBUG
