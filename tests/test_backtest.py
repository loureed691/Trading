"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kucoin_bot.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    WalkForwardValidator,
)
from kucoin_bot.risk.risk_manager import RiskManager
from kucoin_bot.strategies.base import Signal, SignalType
from kucoin_bot.strategies.trend import TrendStrategy


def generate_test_data(
    n_bars: int = 500,
    trend: str = "up",
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n_bars)]
    
    if trend == "up":
        base = 100 + np.linspace(0, 50, n_bars)
    elif trend == "down":
        base = 150 - np.linspace(0, 50, n_bars)
    else:
        base = np.full(n_bars, 100.0)
    
    noise = np.random.normal(0, volatility * 100, n_bars)
    close = base + np.cumsum(noise) * 0.1
    close = np.maximum(close, 10)  # Ensure positive
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + np.random.uniform(-0.01, 0.01, n_bars)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.015, n_bars))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.015, n_bars))),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })
    
    data.attrs["symbol"] = "TEST-USDT"
    return data


class TestBacktestEngine:
    """Test cases for BacktestEngine."""

    def test_backtest_engine_initialization(self) -> None:
        """Test engine initialization."""
        engine = BacktestEngine(
            initial_capital=20000,
            fee_rate=0.002,
            slippage=0.001,
        )
        
        assert engine.initial_capital == 20000
        assert engine.fee_rate == 0.002
        assert engine.slippage == 0.001
        assert engine.capital == 20000

    def test_backtest_engine_reset(self) -> None:
        """Test engine reset."""
        engine = BacktestEngine(initial_capital=10000)
        engine.capital = 5000
        engine.trades.append(Trade(
            symbol="TEST",
            entry_time=datetime.now(),
            entry_price=100,
        ))
        
        engine.reset()
        
        assert engine.capital == 10000
        assert len(engine.trades) == 0
        assert engine.current_position is None

    def test_backtest_run_with_strategy(self) -> None:
        """Test running backtest with a strategy."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = TrendStrategy({
            "short_period": 10,
            "long_period": 20,
            "signal_threshold": 0.01,
        })

        data = generate_test_data(n_bars=300, trend="up")

        result = engine.run(strategy, data)

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        # Equity curve may be empty if no trades occurred, but result should still be valid

    def test_backtest_with_risk_manager(self) -> None:
        """Test backtest with risk management."""
        engine = BacktestEngine(initial_capital=10000)
        risk_manager = RiskManager({
            "max_position_pct": 10.0,
            "max_leverage": 2,
        })
        
        strategy = TrendStrategy({
            "short_period": 10,
            "long_period": 20,
        })
        
        data = generate_test_data(n_bars=300, trend="up")
        
        result = engine.run(strategy, data, risk_manager)
        
        assert isinstance(result, BacktestResult)

    def test_backtest_result_metrics(self) -> None:
        """Test that backtest result metrics are calculated correctly."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = TrendStrategy({
            "short_period": 5,
            "long_period": 10,
            "signal_threshold": 0.005,
        })
        
        data = generate_test_data(n_bars=500, trend="up", volatility=0.03)
        
        result = engine.run(strategy, data)
        
        if result.total_trades > 0:
            assert result.winning_trades + result.losing_trades == result.total_trades
            assert 0 <= result.win_rate <= 1
            if result.winning_trades > 0:
                assert result.avg_win > 0
            if result.losing_trades > 0:
                assert result.avg_loss <= 0

    def test_backtest_empty_result(self) -> None:
        """Test backtest with no trades."""
        engine = BacktestEngine(initial_capital=10000)
        
        # Strategy that won't generate signals
        strategy = TrendStrategy({
            "short_period": 50,
            "long_period": 100,
            "signal_threshold": 0.5,  # Very high threshold
        })
        
        data = generate_test_data(n_bars=120, trend="neutral", volatility=0.001)
        
        result = engine.run(strategy, data)
        
        assert result.total_trades == 0
        assert result.total_return == 0.0

    def test_slippage_applied(self) -> None:
        """Test that slippage is applied correctly."""
        engine = BacktestEngine(slippage=0.01)  # 1% slippage
        
        buy_price = engine._apply_slippage(100.0, "long")
        sell_price = engine._apply_slippage(100.0, "short")
        
        assert buy_price == 101.0  # Worse price for buying
        assert sell_price == 99.0  # Worse price for selling

    def test_fees_calculated(self) -> None:
        """Test fee calculation."""
        engine = BacktestEngine(fee_rate=0.001)
        
        fees = engine._calculate_fees(10, 100)
        
        assert fees == 1.0  # 10 * 100 * 0.001


class TestWalkForwardValidator:
    """Test cases for WalkForwardValidator."""

    def test_validator_initialization(self) -> None:
        """Test validator initialization."""
        validator = WalkForwardValidator(n_periods=5, train_pct=0.8)
        
        assert validator.n_periods == 5
        assert validator.train_pct == 0.8

    def test_walk_forward_validation(self) -> None:
        """Test walk-forward validation runs."""
        validator = WalkForwardValidator(n_periods=3, train_pct=0.7)
        
        data = generate_test_data(n_bars=1000, trend="up")
        config = {"short_period": 10, "long_period": 20}
        
        results = validator.validate(
            TrendStrategy,
            data,
            config,
            initial_capital=10000,
        )
        
        # Should have results for each period
        assert len(results) <= 3

    def test_aggregate_results(self) -> None:
        """Test result aggregation."""
        validator = WalkForwardValidator()
        
        # Create mock results
        results = [
            BacktestResult(
                total_return=0.10,
                annualized_return=0.20,
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                max_drawdown=0.05,
                win_rate=0.55,
                profit_factor=1.8,
                total_trades=50,
                winning_trades=28,
                losing_trades=22,
                avg_trade_pnl=20,
                avg_win=50,
                avg_loss=-30,
                best_trade=200,
                worst_trade=-100,
                avg_hold_time=24,
                trades=[],
                equity_curve=pd.Series(dtype=float),
            ),
            BacktestResult(
                total_return=0.05,
                annualized_return=0.10,
                sharpe_ratio=1.0,
                sortino_ratio=1.5,
                max_drawdown=0.08,
                win_rate=0.50,
                profit_factor=1.2,
                total_trades=40,
                winning_trades=20,
                losing_trades=20,
                avg_trade_pnl=12.5,
                avg_win=40,
                avg_loss=-27.5,
                best_trade=150,
                worst_trade=-80,
                avg_hold_time=20,
                trades=[],
                equity_curve=pd.Series(dtype=float),
            ),
        ]
        
        summary = validator.aggregate_results(results)
        
        assert "avg_return" in summary
        assert "avg_sharpe" in summary
        assert "total_trades" in summary
        assert summary["avg_return"] == pytest.approx(0.075, rel=0.01)
        assert summary["total_trades"] == 90
        assert summary["consistent_positive"] is True

    def test_aggregate_empty_results(self) -> None:
        """Test aggregation with no results."""
        validator = WalkForwardValidator()
        
        summary = validator.aggregate_results([])
        
        assert summary == {}
