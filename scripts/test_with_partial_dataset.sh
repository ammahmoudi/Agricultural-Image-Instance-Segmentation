#!/bin/bash
python main.py \
  --test \
  --test_path datasets/PhenoBench/test \
  --test_percent 0.1 \
  --checkpoint checkpoints/model.pth \
  --log_level INFO
