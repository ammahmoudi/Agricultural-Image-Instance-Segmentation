#!/bin/bash
python main.py \
  --test \
  --test_path datasets/PhenoBench/test \
  --batch_size_test 4 \
  --checkpoint checkpoints/model.pth \
  --log_level INFO
