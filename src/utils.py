# *****************************************************************************
# Basic Utilities
# *****************************************************************************

import logging
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import time

# --------------------------------------------------------------
# Basic Timing Decorator
# --------------------------------------------------------------


def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        start = time()
        result = func(*args, **kw)
        end = time()
        elapsed = round((end-start) / 60, 2)
        print(f'{func.__name__} took {elapsed} minutes.')
        return result
    return wrap


# --------------------------------------------------------------------
# Get Currenht Year
# -------------------------------------------------------------------

def get_current_year() -> str:
    "Return the current year as a string"

    return str(datetime.now().year)  # noqa: DTZ005


# -------------------------------------------------------------------
# Logging Initialization
# -------------------------------------------------------------------

def init_logging(
    log_dir: str = "logs",
    log_file: str = "agent.log",
    level: int = logging.INFO,
    logger_name: str | None = None,
) -> logging.Logger:
    """
    Initialize Python logging with file + stdout handlers.

    Args:
        log_dir: Directory where logs will be written
        log_file: Log filename
        level: Logging level (e.g. logging.INFO)
        logger_name: Optional named logger; defaults to root logger

    Returns:
        Configured logger instance
    """
    year = get_current_year()
    month = datetime.now().strftime("%m")  # noqa: DTZ005
    log_file = f"{year}-{month}-{log_file}"
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_path / log_file, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# --------------------------------------------------------------
# End of File
# --------------------------------------------------------------
