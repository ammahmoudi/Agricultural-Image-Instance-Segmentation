#!/bin/bash
python main.py --resume \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --num_epochs 5 \
  --checkpoint checkpoints/model.pth \
  --resume_checkpoint checkpoints/model_resumed.pth \
  --log_level INFO
