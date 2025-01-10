#!/bin/bash
python main.py \
  --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --lr 0.01 \
  --momentum 0.95 \
  --weight_decay 0.001 \
  --num_epochs 20 \
  --batch_size_train 4 \
  --log_level INFO
