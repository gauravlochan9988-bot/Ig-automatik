"""Logging utilities for IG-AUTOMATIK pipeline."""

import json
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from ..config.paths import find_project_root, system_dir

# This module lives in _SYSTEM/app/ig_automatik/utils/. Start discovery at the
# enclosing _SYSTEM folder; passing the utils path directly could otherwise
# treat _SYSTEM/app as a fresh project and create _SYSTEM/app/_SYSTEM/logs.
PROJECT_ROOT = find_project_root(Path(__file__).resolve().parents[3])


class Logger:
    """Centralized logging with console and file output."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or system_dir(PROJECT_ROOT) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / ("ig_" + time.strftime("%Y%m%d") + ".jsonl")

    def _write(self, level: str, msg: str, data: Optional[dict] = None):
        """Write log entry to both console and file."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [{level}]"

        # Console output
        print(f"{prefix} {msg}")

        # File output (JSONL)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": msg,
        }
        if data:
            entry.update(data)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARNING] Failed to write log: {e}")

    def info(self, msg: str, **data):
        """Log info level."""
        self._write("INFO", msg, data or None)

    def warn(self, msg: str, **data):
        """Log warning level."""
        self._write("WARN", msg, data or None)

    def error(self, msg: str, error: Optional[Exception] = None, **data):
        """Log error level with optional exception."""
        if error:
            data["error_type"] = type(error).__name__
            data["error_msg"] = str(error)
        self._write("ERROR", msg, data or None)

    def success(self, msg: str, **data):
        """Log success level."""
        self._write("SUCCESS", msg, data or None)

    @contextmanager
    def operation(self, operation_name: str):
        """Context manager for tracking operation timing."""
        start = time.time()
        self.info(f"Starting: {operation_name}")
        try:
            yield
            elapsed = time.time() - start
            self.success(f"Completed: {operation_name}", elapsed_seconds=round(elapsed, 2))
        except Exception as e:
            elapsed = time.time() - start
            self.error(f"Failed: {operation_name}", error=e, elapsed_seconds=round(elapsed, 2))
            raise


# Global logger instance
_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get or create global logger instance."""
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger


def set_logger(logger: Logger):
    """Set global logger instance."""
    global _logger
    _logger = logger
