"""Centralized logging configuration for 360-Video-Manager."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from typing import Optional


_RUN_FILE_HANDLER: Optional[logging.FileHandler] = None
_RUN_FILE_HANDLER_LOCK = threading.Lock()


class ColorizedFormatter(logging.Formatter):
    """Console formatter with ANSI colour coding by log level / keyword."""

    RESET = "\033[0m"
    WHITE = "\033[37m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        msg = str(record.getMessage())
        if "[SUCCESS]" in msg:
            color = self.GREEN
        elif "[DISCARD]" in msg or "[ERROR]" in msg:
            color = self.RED
        elif record.levelno >= logging.WARNING:
            color = self.YELLOW
        else:
            color = self.WHITE
        return f"{color}{base}{self.RESET}"


def configure_console_logging(level: int = logging.INFO) -> None:
    """Attach a colorised console handler to the root logger.

    Safe to call multiple times; will not add duplicate handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setFormatter(
                    ColorizedFormatter(
                        fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
                        datefmt="%H:%M:%S",
                    )
                )
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        ColorizedFormatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.setLevel(level)
    root.addHandler(handler)


def configure_file_logging(logs_dir: str) -> Optional[str]:
    """Add a rotating file handler that writes to *logs_dir*.

    Returns the path of the log file created, or ``None`` on failure.
    """
    global _RUN_FILE_HANDLER
    with _RUN_FILE_HANDLER_LOCK:
        try:
            os.makedirs(logs_dir, exist_ok=True)
            root = logging.getLogger()

            if _RUN_FILE_HANDLER is not None:
                root.removeHandler(_RUN_FILE_HANDLER)
                _RUN_FILE_HANDLER.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(logs_dir, f"run_{timestamp}.log")

            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(fh)
            _RUN_FILE_HANDLER = fh
            return log_path
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not create file log handler: %s", exc
            )
            return None


def setup_logging(logs_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """One-call setup: console + optional file logging."""
    configure_console_logging(level=level)
    if logs_dir:
        configure_file_logging(logs_dir)
