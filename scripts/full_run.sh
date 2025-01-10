#!/bin/bash

# Train the model
python main.py \
  --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --num_epochs 5 \
  --batch_size_train 8 \
  --log_level DEBUG

# Test the model
python main.py \
  --test \
  --test_path datasets/PhenoBench/test \
  --checkpoint checkpoints/model.pth \
  --batch_size_test 2 \
  --log_level DEBUG
