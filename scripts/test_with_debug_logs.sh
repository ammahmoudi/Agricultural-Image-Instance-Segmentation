#!/bin/bash
python main.py \
  --test \
  --test_path datasets/PhenoBench/test \
  --checkpoint checkpoints/model.pth \
  --log_level DEBUG
