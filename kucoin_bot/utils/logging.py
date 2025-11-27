"""Logging configuration and utilities."""

import json
import logging
import sys
from datetime import datetime, timezone
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


class AuditLogger:
    """Structured audit logging for compliance and debugging."""

    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audit file handler
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        audit_file = self.log_dir / "audit.jsonl"
        handler = RotatingFileHandler(
            audit_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a structured audit event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        self.logger.info(json.dumps(event))

    def log_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str,
        leverage: int = 1,
        market: str = "spot",
    ) -> None:
        """Log order placement for audit."""
        self._log_event("ORDER_PLACED", {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "order_type": order_type,
            "leverage": leverage,
            "market": market,
        })

    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        fill_price: float,
        fill_size: float,
        fees: float,
    ) -> None:
        """Log order fill for audit."""
        self._log_event("ORDER_FILLED", {
            "order_id": order_id,
            "symbol": symbol,
            "fill_price": fill_price,
            "fill_size": fill_size,
            "fees": fees,
        })

    def log_order_cancelled(
        self,
        order_id: str,
        symbol: str,
        reason: str = "",
    ) -> None:
        """Log order cancellation for audit."""
        self._log_event("ORDER_CANCELLED", {
            "order_id": order_id,
            "symbol": symbol,
            "reason": reason,
        })

    def log_position_opened(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        leverage: int,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> None:
        """Log position opening for audit."""
        self._log_event("POSITION_OPENED", {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "leverage": leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })

    def log_position_closed(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_time_hours: float,
    ) -> None:
        """Log position closure for audit."""
        self._log_event("POSITION_CLOSED", {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_time_hours": hold_time_hours,
        })

    def log_risk_event(
        self,
        event_type: str,
        symbol: str | None,
        details: dict[str, Any],
    ) -> None:
        """Log risk management events for audit."""
        self._log_event("RISK_EVENT", {
            "risk_event_type": event_type,
            "symbol": symbol,
            "details": details,
        })

    def log_regime_change(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str,
        confidence: float,
    ) -> None:
        """Log regime change detection for audit."""
        self._log_event("REGIME_CHANGE", {
            "symbol": symbol,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence,
        })

    def log_system_event(
        self,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log system events for audit."""
        self._log_event("SYSTEM_EVENT", {
            "system_event_type": event_type,
            "details": details,
        })


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


class MonitoringMetrics:
    """Collect and expose monitoring metrics."""

    def __init__(self):
        self._metrics: dict[str, Any] = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "current_positions": 0,
            "last_update": None,
        }
        self._position_pnls: list[float] = []

    def record_order_placed(self) -> None:
        """Record order placement."""
        self._metrics["orders_placed"] += 1
        self._metrics["last_update"] = datetime.utcnow().isoformat()

    def record_order_filled(self) -> None:
        """Record order fill."""
        self._metrics["orders_filled"] += 1
        self._metrics["last_update"] = datetime.utcnow().isoformat()

    def record_order_cancelled(self) -> None:
        """Record order cancellation."""
        self._metrics["orders_cancelled"] += 1
        self._metrics["last_update"] = datetime.utcnow().isoformat()

    def record_position_closed(self, pnl: float) -> None:
        """Record position closure."""
        self._metrics["total_pnl"] += pnl
        self._position_pnls.append(pnl)
        
        if pnl > 0:
            self._metrics["win_count"] += 1
        else:
            self._metrics["loss_count"] += 1
        
        self._metrics["last_update"] = datetime.utcnow().isoformat()

    def update_position_count(self, count: int) -> None:
        """Update current position count."""
        self._metrics["current_positions"] = count
        self._metrics["last_update"] = datetime.utcnow().isoformat()

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        total_trades = self._metrics["win_count"] + self._metrics["loss_count"]
        
        return {
            **self._metrics,
            "win_rate": (
                self._metrics["win_count"] / total_trades
                if total_trades > 0 else 0.0
            ),
            "avg_pnl": (
                self._metrics["total_pnl"] / total_trades
                if total_trades > 0 else 0.0
            ),
            "total_trades": total_trades,
        }

    def get_pnl_summary(self) -> dict[str, float]:
        """Get PnL summary statistics."""
        if not self._position_pnls:
            return {
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0,
                "std_pnl": 0.0,
            }
        
        import numpy as np
        
        pnls = np.array(self._position_pnls)
        return {
            "total_pnl": float(np.sum(pnls)),
            "avg_pnl": float(np.mean(pnls)),
            "max_pnl": float(np.max(pnls)),
            "min_pnl": float(np.min(pnls)),
            "std_pnl": float(np.std(pnls)),
        }
