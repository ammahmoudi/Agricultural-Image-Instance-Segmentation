#!/bin/bash
python main.py \
  --test \
  --test_path datasets/PhenoBench/test \
  --checkpoint checkpoints/model.pth \
  --batch_size_test 2 \
  --log_level INFO
