"""
Structured Logger

Provides structured logging for the CryptoAlpha system:
- Console and file logging
- JSON format for log aggregation
- Log levels and filtering
- Performance timing
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import threading


class StructuredFormatter(logging.Formatter):
    """JSON-based structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Time
        time_str = datetime.now().strftime("%H:%M:%S")

        # Base message
        msg = f"{color}[{time_str}] {record.levelname:8}{reset} {record.name}: {record.getMessage()}"

        # Add extra data if present
        if hasattr(record, "extra_data") and record.extra_data:
            data_str = " | ".join(f"{k}={v}" for k, v in record.extra_data.items())
            msg += f" ({data_str})"

        return msg


class StructuredLogger:
    """
    Structured logger with context support.

    Usage:
        logger = get_logger("my_module")
        logger.info("Processing signal", symbol="BTC/USDT", confidence=0.72)
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_dir: str = "logs",
        console: bool = True,
        file: bool = True,
        json_format: bool = False,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Log level
            log_dir: Directory for log files
            console: Enable console logging
            file: Enable file logging
            json_format: Use JSON format for files
        """
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear existing handlers

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            console_handler.setLevel(level)
            self._logger.addHandler(console_handler)

        # File handler
        if file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = Path(log_dir) / f"cryptoalpha_{today}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")

            if json_format:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                    )
                )

            file_handler.setLevel(level)
            self._logger.addHandler(file_handler)

    def _log(
        self,
        level: int,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log with extra data."""
        record = self._logger.makeRecord(
            name=self.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        if kwargs:
            record.extra_data = kwargs

        self._logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, extra={"extra_data": kwargs})


class TimingContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        **extra: Any,
    ):
        """
        Initialize timing context.

        Args:
            logger: Logger to use
            operation: Operation name
            extra: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.extra = extra
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimingContext":
        """Start timing."""
        self.start_time = datetime.now().timestamp()
        self.logger.debug(f"Starting {self.operation}", **self.extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """End timing and log duration."""
        duration = datetime.now().timestamp() - self.start_time
        if exc_type:
            self.logger.error(
                f"Failed {self.operation}",
                duration_seconds=round(duration, 3),
                error=str(exc_val),
                **self.extra,
            )
        else:
            self.logger.info(
                f"Completed {self.operation}",
                duration_seconds=round(duration, 3),
                **self.extra,
            )
        return False


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_lock = threading.Lock()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str = "logs",
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name
        level: Log level
        log_dir: Log directory

    Returns:
        StructuredLogger instance
    """
    with _lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(
                name=name,
                level=level,
                log_dir=log_dir,
            )
        return _loggers[name]


def configure_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    json_format: bool = False,
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Default log level
        log_dir: Log directory
        json_format: Use JSON format
    """
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ConsoleFormatter())
    console.setLevel(level)
    root.addHandler(console)

    # File handler
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(log_dir) / f"cryptoalpha_{today}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    if json_format:
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    file_handler.setLevel(level)
    root.addHandler(file_handler)


if __name__ == "__main__":
    print("Testing Structured Logger...")

    # Configure logging
    configure_logging(level=logging.DEBUG)

    # Get logger
    logger = get_logger("test")

    # Test different levels
    print("\n1. Testing log levels...")
    logger.debug("This is a debug message", key="value")
    logger.info("This is an info message", symbol="BTC/USDT", price=42000)
    logger.warning("This is a warning", threshold=0.5, actual=0.3)
    logger.error("This is an error", code=500, reason="timeout")

    # Test timing
    print("\n2. Testing timing context...")
    import time

    with TimingContext(logger, "data fetch", symbol="ETH/USDT"):
        time.sleep(0.1)

    # Test exception
    print("\n3. Testing exception logging...")
    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("Caught an error", context="testing")

    print("\nTest complete!")
