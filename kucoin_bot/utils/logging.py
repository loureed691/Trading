"""Logging configuration and utilities."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging based on configuration."""
    level = getattr(logging, config.get("level", "INFO").upper())
    log_file = config.get("file", "logs/bot.log")
    max_size = config.get("max_size_mb", 10) * 1024 * 1024
    backup_count = config.get("backup_count", 5)

    # Create logs directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count,
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


class TradingLogger:
    """Specialized logger for trading operations."""

    def __init__(self, name: str = "trading"):
        self.logger = logging.getLogger(name)

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        strength: float,
        metadata: dict | None = None,
    ) -> None:
        """Log a trading signal."""
        self.logger.info(
            f"SIGNAL: {symbol} {signal_type} @ {price:.6f} "
            f"(strength={strength:.2f}) {metadata or ''}"
        )

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str,
    ) -> None:
        """Log an order placement."""
        self.logger.info(
            f"ORDER: {order_id} {symbol} {side} {size:.6f} @ {price:.6f} ({order_type})"
        )

    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        fees: float,
    ) -> None:
        """Log an order fill."""
        self.logger.info(
            f"FILL: {order_id} {symbol} {side} {size:.6f} @ {price:.6f} (fees={fees:.6f})"
        )

    def log_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        unrealized_pnl: float,
    ) -> None:
        """Log position status."""
        self.logger.info(
            f"POSITION: {symbol} {side} {size:.6f} @ {entry_price:.6f} "
            f"(unrealized PnL={unrealized_pnl:.2f})"
        )

    def log_risk_event(
        self,
        event_type: str,
        details: str,
    ) -> None:
        """Log a risk management event."""
        self.logger.warning(f"RISK: {event_type} - {details}")

    def log_error(
        self,
        operation: str,
        error: Exception,
    ) -> None:
        """Log an error."""
        self.logger.error(f"ERROR in {operation}: {error}", exc_info=True)
