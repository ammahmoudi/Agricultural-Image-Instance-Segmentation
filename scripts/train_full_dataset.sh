#!/bin/bash
python main.py \
  --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --num_epochs 10 \
  --batch_size_train 8 \
  --log_level INFO
