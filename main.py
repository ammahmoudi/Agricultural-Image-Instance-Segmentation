import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from engine import train_one_epoch, evaluate
from dataset import PennFudanDataset, PennFudanTestDataset, get_transform, collate_fn
from logger import setup_logging  # Import the logging setup from logger.py


def parse_args():
    """
    Parse command-line arguments for training, testing, and resuming configurations.

    Returns:
        argparse.Namespace: Parsed arguments with their respective default values.
    """
    parser = argparse.ArgumentParser(
        description="Train, resume, or test Mask R-CNN on a custom dataset."
    )

    # Dataset paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="datasets/PhenoBench/train/",
        help="Path to the training data.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="datasets/PhenoBench/val/",
        help="Path to the validation data.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="datasets/PhenoBench/test/",
        help="Path to the testing data.",
    )

    # Dataset subset percentages
    parser.add_argument(
        "--train_percent",
        type=float,
        default=1.0,
        help="Percentage of training data to use.",
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=1.0,
        help="Percentage of validation data to use.",
    )
    parser.add_argument(
        "--test_percent",
        type=float,
        default=1.0,
        help="Percentage of test data to use.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size_train", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--batch_size_test", type=int, default=1, help="Batch size for testing."
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for the optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=3,
        help="Step size for learning rate scheduler.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train."
    )

    # Mode selection
    parser.add_argument(
        "--train", action="store_true", help="Flag to start fresh training."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training from a checkpoint.",
    )
    parser.add_argument("--test", action="store_true", help="Flag to enable testing.")

    # Checkpoints
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pth",
        help="Path to load the model checkpoint.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to save the checkpoint after resuming training.",
    )

    # Logging level
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    return parser.parse_args()


def main():
    """
    Main function to train, resume, or test the Mask R-CNN model on the specified dataset.
    """
    args = parse_args()

    # Set up logging with the specified level
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    setup_logging(log_level=log_levels[args.log_level])

    logging.info("Starting the script with the following arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # Set up the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load datasets

    if args.test:
        dataset_test = PennFudanTestDataset(args.test_path, get_transform(train=False))
        test_indices = torch.randperm(len(dataset_test)).tolist()[
            : int(len(dataset_test) * args.test_percent)
        ]
        dataset_test = Subset(dataset_test, test_indices)
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size_test,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        dataset = PennFudanDataset(args.train_path, get_transform(train=True))
        dataset_val = PennFudanDataset(args.val_path, get_transform(train=False))

        # Apply subset percentages
        train_indices = torch.randperm(len(dataset)).tolist()[
            : int(len(dataset) * args.train_percent)
        ]
        val_indices = torch.randperm(len(dataset_val)).tolist()[
            : int(len(dataset_val) * args.val_percent)
        ]
        dataset = Subset(dataset, train_indices)
        dataset_val = Subset(dataset_val, val_indices)

        # Create data loaders
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size_train,
            shuffle=True,
            collate_fn=collate_fn,
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size_test,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # Initialize the model
    model = maskrcnn_resnet50_fpn(weights="DEFAULT").to(device)

    # Testing mode
    if args.test:
        logging.info("Testing mode enabled. Loading checkpoint...")
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            logging.error(f"Checkpoint file not found: {args.checkpoint}")
            return
        model.eval()
        evaluate(model, data_loader_test, device=device)
        logging.info("Testing complete.")
        return

    # Fresh training mode
    if args.train:
        logging.info("Training mode enabled. Starting training from scratch.")

    # Resume training mode
    if args.resume:
        logging.info(f"Resuming training from checkpoint: {args.checkpoint}")
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            logging.error(f"Checkpoint file not found: {args.checkpoint}")
            return

    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # Training and Resume Modes
    if args.train or args.resume:
        for epoch in range(args.num_epochs):
            logging.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            evaluate(model, data_loader_val, device=device)
            logging.info(f"Completed epoch {epoch + 1}/{args.num_epochs}")

        # Save checkpoint to the specified path
        save_path = (
            args.resume_checkpoint if args.resume_checkpoint else args.checkpoint
        )
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")

    logging.info("Execution complete.")


if __name__ == "__main__":
    main()
