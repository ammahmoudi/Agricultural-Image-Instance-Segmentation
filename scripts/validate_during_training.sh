#!/bin/bash
python main.py \
  --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --num_epochs 3 \
  --batch_size_train 4 \
  --log_level DEBUG
