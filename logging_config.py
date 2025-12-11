"""
Logging configuration for Performance Lab.

Provides consistent logging across all modules with configurable levels
and output formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)s: %(message)s"
LOG_FORMAT_DEBUG = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# Log file location
LOG_DIR = Path.home() / ".performance_lab" / "logs"
LOG_FILE = LOG_DIR / "performance_lab.log"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    debug_format: bool = False
) -> logging.Logger:
    """
    Configure logging for Performance Lab.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        console: Whether to output to console
        debug_format: Use detailed debug format

    Returns:
        Root logger for the application
    """
    # Create log directory if needed
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Select format
    log_format = LOG_FORMAT_DEBUG if debug_format else LOG_FORMAT

    # Configure root logger
    root_logger = logging.getLogger("performance_lab")
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT_SIMPLE))
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    # Create child logger under performance_lab namespace
    if name.startswith("performance_lab."):
        logger_name = name
    else:
        logger_name = f"performance_lab.{name}"

    return logging.getLogger(logger_name)


# Module-specific loggers
class LoggerNames:
    """Standard logger names for the application."""
    MAIN = "performance_lab.main"
    DISTRIBUTED = "performance_lab.distributed"
    WORKFLOW = "performance_lab.workflow"
    LLM = "performance_lab.llm"
    NETWORK = "performance_lab.network"
    NODES = "performance_lab.nodes"


# Initialize default logging on import
_root_logger = setup_logging(
    level=logging.INFO,
    log_file=LOG_FILE,
    console=False  # Don't spam console by default
)
