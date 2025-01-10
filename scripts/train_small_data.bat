@echo off
REM Run the Python script to train a small subset of data with detailed logs
python main.py ^
  --train ^
  --train_path datasets\PhenoBench\train ^
  --val_path datasets\PhenoBench\val ^
  --train_percent 0.1 ^
  --val_percent 0.1 ^
  --num_epochs 1 ^
  --batch_size_train 2 ^
  --log_level DEBUG ^
  --checkpoint checkpoints\test_model.pth

pause