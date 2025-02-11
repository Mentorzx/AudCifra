import logging
import os
from datetime import datetime


def get_logger(
    name: str, console_level: str = "WARNING", file_level: str = "DEBUG"
) -> logging.Logger:
    """
    Returns a logger with a console handler and a file handler.
    The console handler has the specified console_level, and the file handler uses file_level.
    Each run creates a new log file in the 'logs' folder, named with an incremental index and timestamp.

    :param name: Logger name.
    :param console_level: Minimum log level for console output (e.g., "WARNING").
    :param file_level: Minimum log level for file output (e.g., "DEBUG").
    :return: A configured logging.Logger instance.
    """
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    existing_logs = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
    log_count = len(existing_logs) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"log_{log_count}_{timestamp}.log")

    logger = logging.getLogger(name)

    if not logger.handlers:
        _extracted_from_get_logger_26(console_level, file_level, logger, log_filename)
    return logger


# TODO Rename this here and in `get_logger`
def _extracted_from_get_logger_26(console_level, file_level, logger, log_filename):
    # Convert level strings to actual logging levels
    console_level_num = getattr(logging, console_level.upper(), logging.WARNING)
    file_level_num = getattr(logging, file_level.upper(), logging.DEBUG)

    logger.setLevel(min(console_level_num, file_level_num))

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    _extracted_from_get_logger_32(
        file_handler,
        file_level_num,
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        logger,
    )
    console_handler = logging.StreamHandler()
    _extracted_from_get_logger_32(
        console_handler,
        console_level_num,
        "%(asctime)s - %(levelname)s - %(message)s",
        logger,
    )


# TODO Rename this here and in `get_logger`
def _extracted_from_get_logger_32(arg0, arg1, arg2, logger):
    arg0.setLevel(arg1)
    file_formatter = logging.Formatter(arg2)
    arg0.setFormatter(file_formatter)
    logger.addHandler(arg0)
