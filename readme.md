# Mask R-CNN for Image Segmentation and Detection

This project implements Mask R-CNN using PyTorch for image segmentation and object detection. It includes custom datasets, training, evaluation, and logging. The code is flexible and supports training on partial datasets, checkpointing, and resuming training.

---

## Table of Contents

- [Mask R-CNN for Image Segmentation and Detection](#mask-r-cnn-for-image-segmentation-and-detection)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Testing](#testing)
    - [Resume training from a saved checkpoint](#resume-training-from-a-saved-checkpoint)
  - [Configuration](#configuration)
  - [Logging](#logging)

---

## Features

- Mask R-CNN implementation with PyTorch.
- Custom dataset support for PhenoBench and other datasets.
- Training, testing, and resuming training modes.
- Flexible argument-based configuration using `argparse`.
- Logging to console and file.
- Handles testing datasets with only images.

---

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

Prepare your dataset in the datasets/PhenoBench/ directory:

train/ and val/ folders should include subfolders like images and leaf_instances.
test/ folder only requires the images subfolder.

## Usage

Run the main.py script with the desired mode and arguments. Use the --help flag to see all options:

```bash
python main.py --help
```

### Training

Train a model from scratch:

```bash
python main.py --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --num_epochs 10 \
  --batch_size_train 4 \
  --checkpoint checkpoints/model.pth
```

### Testing

Evaluate a pretrained model:

```bash
python main.py --test \
  --test_path datasets/PhenoBench/test \
  --checkpoint checkpoints/model.pth
```

### Resume training from a saved checkpoint

```bash
python main.py --resume \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --num_epochs 5 \
  --checkpoint checkpoints/model.pth \
  --resume_checkpoint checkpoints/model_resumed.pth
  ```

## Configuration

Key arguments for main.py:

```bash
--train_path, --val_path, --test_path: Paths to datasets.
--train_percent, --val_percent, --test_percent: Use only a percentage of data.
--num_epochs: Number of epochs for training.
--batch_size_train, --batch_size_test: Batch sizes for training and testing.
--checkpoint: Path to save/load checkpoints.
--log_level: Logging level (DEBUG, INFO, etc.).
  ```

## Logging

Logs are saved in the results/logs/ directory by default. They include:

- Command-line arguments.
- Training and evaluation progress.
- Warnings and errors.
- Examples
- Train on a Small Dataset

```bash
python main.py --train \
  --train_path datasets/PhenoBench/train \
  --val_path datasets/PhenoBench/val \
  --train_percent 0.1 \
  --val_percent 0.1 \
  --num_epochs 1 \
  --batch_size_train 2 \
  --log_level DEBUG
  ```

Test on the Test Dataset

```bash
python main.py --test \
  --test_path datasets/PhenoBench/test \
  --checkpoint checkpoints/model.pth
```
