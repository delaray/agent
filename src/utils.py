# *****************************************************************************
# Basic Utilities
# *****************************************************************************

import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import time
from typing import Any

from pydantic import BaseModel

from src.types import Event, Message, ToolCall, ToolResult

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
# GUI and trace formatting
# -------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Convert framework values into objects accepted by JSON/Streamlit."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# -------------------------------------------------------------------

def classify_event(event: Event) -> str:
    """Return a concise trace category for an execution event."""
    if any(isinstance(item, ToolCall) for item in event.content):
        return "tool_call"
    if any(isinstance(item, ToolResult) for item in event.content):
        return "tool_result"
    if any(
        isinstance(item, Message) and item.role == "user"
        for item in event.content
    ):
        return "user"
    if any(isinstance(item, Message) for item in event.content):
        return "assistant"
    return "event"


# -------------------------------------------------------------------

def summarize_event(event: Event) -> str:
    """Create a readable one-line summary for an execution trace."""
    for item in event.content:
        if isinstance(item, ToolCall):
            return f"Requested {item.name}"
        if isinstance(item, ToolResult):
            return f"{item.name} returned {item.status}"
        if isinstance(item, Message):
            text = " ".join(item.content.split())
            return text if len(text) <= 100 else f"{text[:97]}..."
    return "Empty event"


def format_elapsed(seconds: float) -> str:
    """Format a duration for compact UI display."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.0f}s"


# -------------------------------------------------------------------
# Logging Initialization
# -------------------------------------------------------------------

def init_logging(
    log_dir: str = "logs",
    log_file: str = "agent",
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
