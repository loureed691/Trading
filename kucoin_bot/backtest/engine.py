"""Backtesting engine with walk-forward validation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..risk.risk_manager import RiskManager
from ..strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Executed trade record."""

    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    side: str = "long"
    size: float = 0.0
    leverage: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest results summary."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_hold_time: float
    trades: list[Trade]
    equity_curve: pd.Series


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.current_position: Trade | None = None

    def reset(self) -> None:
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.current_position = None

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price."""
        if side == "long":
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _calculate_fees(self, size: float, price: float) -> float:
        """Calculate trading fees."""
        return size * price * self.fee_rate

    def _open_position(
        self,
        signal: Signal,
        timestamp: datetime,
        data: pd.DataFrame,
        risk_manager: RiskManager | None = None,
    ) -> bool:
        """Open a new position based on signal."""
        if self.current_position is not None:
            return False

        side = "long" if signal.type == SignalType.LONG else "short"
        entry_price = self._apply_slippage(signal.price, side)

        # Calculate position size
        if risk_manager:
            risk_manager.update_portfolio(self.capital)
            sizing = risk_manager.calculate_position_size(
                signal, data, self.capital
            )
            if sizing is None:
                return False
            size = sizing.size
            leverage = sizing.leverage
        else:
            # Simple position sizing
            size = (self.capital * 0.1) / entry_price
            leverage = signal.suggested_leverage

        # Calculate fees
        fees = self._calculate_fees(size, entry_price)

        self.current_position = Trade(
            symbol=signal.symbol,
            entry_time=timestamp,
            entry_price=entry_price,
            side=side,
            size=size,
            leverage=leverage,
            fees=fees,
            metadata=signal.metadata or {},
        )

        self.capital -= fees
        return True

    def _close_position(
        self,
        exit_price: float,
        timestamp: datetime,
    ) -> Trade | None:
        """Close current position."""
        if self.current_position is None:
            return None

        position = self.current_position
        exit_price = self._apply_slippage(
            exit_price,
            "short" if position.side == "long" else "long",
        )

        # Calculate P&L
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.size * position.leverage
        else:
            pnl = (position.entry_price - exit_price) * position.size * position.leverage

        # Deduct exit fees
        exit_fees = self._calculate_fees(position.size, exit_price)
        pnl -= exit_fees

        position.exit_time = timestamp
        position.exit_price = exit_price
        position.pnl = pnl
        position.fees += exit_fees
        position.pnl_pct = pnl / (position.entry_price * position.size) * 100

        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        self.trades.append(position)
        self.current_position = None

        return position

    def _check_stop_loss_take_profit(
        self,
        high: float,
        low: float,
        timestamp: datetime,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> bool:
        """Check if stop loss or take profit was hit."""
        if self.current_position is None:
            return False

        position = self.current_position

        if position.side == "long":
            if stop_loss and low <= stop_loss:
                self._close_position(stop_loss, timestamp)
                return True
            if take_profit and high >= take_profit:
                self._close_position(take_profit, timestamp)
                return True
        else:  # short
            if stop_loss and high >= stop_loss:
                self._close_position(stop_loss, timestamp)
                return True
            if take_profit and low <= take_profit:
                self._close_position(take_profit, timestamp)
                return True

        return False

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        risk_manager: RiskManager | None = None,
    ) -> BacktestResult:
        """Run backtest on historical data."""
        self.reset()

        if risk_manager:
            risk_manager.update_portfolio(self.initial_capital)

        # Ensure data has timestamp column
        if "timestamp" not in data.columns:
            data = data.reset_index()
            if "timestamp" not in data.columns:
                data["timestamp"] = pd.to_datetime(data.index)

        last_signal: Signal | None = None

        for i in range(50, len(data)):  # Start after enough data for indicators
            current_bar = data.iloc[i]
            timestamp = pd.to_datetime(current_bar["timestamp"])
            lookback_data = data.iloc[max(0, i - 100) : i + 1].copy()

            # Check stop loss / take profit
            if last_signal:
                self._check_stop_loss_take_profit(
                    current_bar["high"],
                    current_bar["low"],
                    timestamp,
                    last_signal.stop_loss,
                    last_signal.take_profit,
                )

            # Generate signal
            signal = strategy.generate_signal(lookback_data)

            if signal:
                if signal.type in (SignalType.LONG, SignalType.SHORT):
                    if self.current_position is None:
                        if self._open_position(signal, timestamp, lookback_data, risk_manager):
                            last_signal = signal
                    elif (
                        self.current_position.side == "long"
                        and signal.type == SignalType.SHORT
                    ) or (
                        self.current_position.side == "short"
                        and signal.type == SignalType.LONG
                    ):
                        # Close existing and open opposite
                        self._close_position(current_bar["close"], timestamp)
                        if self._open_position(signal, timestamp, lookback_data, risk_manager):
                            last_signal = signal

                elif signal.type == SignalType.CLOSE:
                    if self.current_position:
                        self._close_position(current_bar["close"], timestamp)
                        last_signal = None

            # Update equity curve
            if self.current_position:
                unrealized = self._calculate_unrealized_pnl(current_bar["close"])
                equity = self.capital + unrealized
            else:
                equity = self.capital

            self.equity_curve.append((timestamp, equity))

        # Close any remaining position
        if self.current_position:
            self._close_position(data.iloc[-1]["close"], pd.to_datetime(data.iloc[-1]["timestamp"]))

        return self._calculate_results()

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for current position."""
        if self.current_position is None:
            return 0.0

        position = self.current_position
        if position.side == "long":
            return (current_price - position.entry_price) * position.size * position.leverage
        else:
            return (position.entry_price - current_price) * position.size * position.leverage

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest result metrics."""
        if not self.trades:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_trade_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                avg_hold_time=0.0,
                trades=[],
                equity_curve=pd.Series(dtype=float),
            )

        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_series = equity_df["equity"]

        # Returns
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Calculate period for annualization
        if len(self.equity_curve) > 1:
            start_time = self.equity_curve[0][0]
            end_time = self.equity_curve[-1][0]
            days = (end_time - start_time).days
            years = max(days / 365, 0.01)
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        # Daily returns for Sharpe/Sortino
        daily_equity = equity_series.resample("D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1:
            sharpe_ratio = (
                daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                if daily_returns.std() > 0
                else 0.0
            )
            negative_returns = daily_returns[daily_returns < 0]
            sortino_ratio = (
                daily_returns.mean() / negative_returns.std() * np.sqrt(252)
                if len(negative_returns) > 0 and negative_returns.std() > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Max drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Trade statistics
        pnls = [t.pnl for t in self.trades]
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades)
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

        # Hold time
        hold_times = []
        for trade in self.trades:
            if trade.exit_time and trade.entry_time:
                hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                hold_times.append(hold_time)
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_pnl=np.mean(pnls),
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=max(pnls),
            worst_trade=min(pnls),
            avg_hold_time=avg_hold_time,
            trades=self.trades,
            equity_curve=equity_series,
        )


class WalkForwardValidator:
    """Walk-forward validation for strategy optimization."""

    def __init__(
        self,
        n_periods: int = 5,
        train_pct: float = 0.7,
    ):
        self.n_periods = n_periods
        self.train_pct = train_pct

    def validate(
        self,
        strategy_class: type[BaseStrategy],
        data: pd.DataFrame,
        config: dict[str, Any],
        risk_manager: RiskManager | None = None,
        initial_capital: float = 10000.0,
    ) -> list[BacktestResult]:
        """Run walk-forward validation."""
        results = []
        period_size = len(data) // self.n_periods

        for i in range(self.n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < self.n_periods - 1 else len(data)
            
            period_data = data.iloc[start_idx:end_idx].copy()
            train_size = int(len(period_data) * self.train_pct)
            
            train_data = period_data.iloc[:train_size]
            test_data = period_data.iloc[train_size:]

            if len(train_data) < 100 or len(test_data) < 50:
                logger.warning(f"Period {i+1} has insufficient data, skipping")
                continue

            # Train on training data
            strategy = strategy_class(config)
            train_engine = BacktestEngine(initial_capital=initial_capital)
            train_result = train_engine.run(strategy, train_data, risk_manager)

            logger.info(
                f"Period {i+1} Training: Return={train_result.total_return:.2%}, "
                f"Sharpe={train_result.sharpe_ratio:.2f}"
            )

            # Test on out-of-sample data
            test_engine = BacktestEngine(initial_capital=initial_capital)
            test_result = test_engine.run(strategy, test_data, risk_manager)

            logger.info(
                f"Period {i+1} Testing: Return={test_result.total_return:.2%}, "
                f"Sharpe={test_result.sharpe_ratio:.2f}"
            )

            results.append(test_result)

        return results

    def aggregate_results(self, results: list[BacktestResult]) -> dict[str, Any]:
        """Aggregate walk-forward validation results."""
        if not results:
            return {}

        return {
            "avg_return": np.mean([r.total_return for r in results]),
            "std_return": np.std([r.total_return for r in results]),
            "avg_sharpe": np.mean([r.sharpe_ratio for r in results]),
            "avg_win_rate": np.mean([r.win_rate for r in results]),
            "avg_max_drawdown": np.mean([r.max_drawdown for r in results]),
            "total_trades": sum(r.total_trades for r in results),
            "consistent_positive": all(r.total_return > 0 for r in results),
            "n_periods": len(results),
        }
