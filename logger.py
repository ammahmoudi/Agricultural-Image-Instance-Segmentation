import os
import logging
from datetime import datetime

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Sets up logging to file and console for the entire application.

    Args:
        log_dir (str): Directory where log files are saved.
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    # Suppress unnecessary DEBUG logs from Pillow
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.info(f"Logging initialized. Logs will be saved to {log_file}.")
