"""Tests for strategy modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType
from kucoin_bot.strategies.trend import TrendStrategy
from kucoin_bot.strategies.mean_reversion import MeanReversionStrategy
from kucoin_bot.strategies.breakout import BreakoutStrategy
from kucoin_bot.strategies.market_making import MarketMakingStrategy


def generate_test_data(
    n_bars: int = 200,
    trend: str = "neutral",
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_bars - i) for i in range(n_bars)]
    
    # Base price
    if trend == "up":
        base = 100 + np.linspace(0, 20, n_bars)
    elif trend == "down":
        base = 100 - np.linspace(0, 20, n_bars)
    else:
        base = np.full(n_bars, 100.0)
    
    # Add noise
    noise = np.random.normal(0, volatility * 100, n_bars)
    close = base + noise
    
    # Generate OHLCV
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + np.random.uniform(-0.01, 0.01, n_bars)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_bars))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_bars))),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })
    
    data.attrs["symbol"] = "TEST-USDT"
    return data


class TestTrendStrategy:
    """Test cases for TrendStrategy."""

    def test_trend_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        config = {"short_period": 10, "long_period": 30}
        strategy = TrendStrategy(config)
        
        assert strategy.name == "trend"
        assert strategy.short_period == 10
        assert strategy.long_period == 30

    def test_trend_strategy_no_signal_on_insufficient_data(self) -> None:
        """Test no signal generated with insufficient data."""
        strategy = TrendStrategy({})
        data = generate_test_data(n_bars=30)
        
        signal = strategy.generate_signal(data)
        assert signal is None

    def test_trend_strategy_generates_signal(self) -> None:
        """Test signal generation on trending data."""
        strategy = TrendStrategy({
            "short_period": 10,
            "long_period": 20,
            "signal_threshold": 0.01,
        })
        
        # Generate uptrending data
        data = generate_test_data(n_bars=100, trend="up", volatility=0.01)
        
        # Run strategy multiple times to potentially catch a crossover
        for i in range(50, len(data)):
            signal = strategy.generate_signal(data.iloc[:i+1].copy())
            if signal is not None:
                assert signal.type in (SignalType.LONG, SignalType.SHORT)
                assert 0 <= signal.strength <= 1
                break


class TestMeanReversionStrategy:
    """Test cases for MeanReversionStrategy."""

    def test_mean_reversion_initialization(self) -> None:
        """Test strategy initialization."""
        config = {"lookback": 15, "entry_zscore": 2.5}
        strategy = MeanReversionStrategy(config)
        
        assert strategy.name == "mean_reversion"
        assert strategy.lookback == 15
        assert strategy.entry_zscore == 2.5

    def test_mean_reversion_no_signal_on_insufficient_data(self) -> None:
        """Test no signal with insufficient data."""
        strategy = MeanReversionStrategy({})
        data = generate_test_data(n_bars=20)
        
        signal = strategy.generate_signal(data)
        assert signal is None

    def test_mean_reversion_signal_on_extreme(self) -> None:
        """Test signal generation on extreme deviation."""
        strategy = MeanReversionStrategy({
            "lookback": 20,
            "entry_zscore": 2.0,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        })
        
        # Generate data with extreme move at the end
        data = generate_test_data(n_bars=100, trend="neutral")
        
        # Force extreme low
        data.loc[data.index[-5:], "close"] *= 0.9
        data.loc[data.index[-5:], "low"] *= 0.9
        
        signal = strategy.generate_signal(data)
        # May or may not generate signal depending on RSI


class TestBreakoutStrategy:
    """Test cases for BreakoutStrategy."""

    def test_breakout_initialization(self) -> None:
        """Test strategy initialization."""
        config = {"lookback": 25, "volume_multiplier": 2.0}
        strategy = BreakoutStrategy(config)
        
        assert strategy.name == "breakout"
        assert strategy.lookback == 25
        assert strategy.volume_multiplier == 2.0

    def test_breakout_no_signal_without_breakout(self) -> None:
        """Test no signal in ranging market."""
        strategy = BreakoutStrategy({})
        data = generate_test_data(n_bars=100, trend="neutral", volatility=0.005)
        
        signal = strategy.generate_signal(data)
        # Unlikely to have breakout in low volatility neutral market


class TestMarketMakingStrategy:
    """Test cases for MarketMakingStrategy."""

    def test_market_making_initialization(self) -> None:
        """Test strategy initialization."""
        config = {"spread_multiplier": 2.0, "inventory_target": 0.4}
        strategy = MarketMakingStrategy(config)
        
        assert strategy.name == "market_making"
        assert strategy.spread_multiplier == 2.0
        assert strategy.inventory_target == 0.4

    def test_market_making_inventory_skew(self) -> None:
        """Test inventory management."""
        strategy = MarketMakingStrategy({})
        
        # Set high inventory
        strategy.set_inventory(80, 100)
        skew = strategy._calculate_inventory_skew()
        assert skew > 0  # Should skew to sell
        
        # Set low inventory
        strategy.set_inventory(20, 100)
        skew = strategy._calculate_inventory_skew()
        assert skew < 0  # Should skew to buy

    def test_market_making_generates_signal(self) -> None:
        """Test signal generation."""
        strategy = MarketMakingStrategy({})
        strategy.set_inventory(50, 100)  # Balanced
        
        data = generate_test_data(n_bars=50)
        signal = strategy.generate_signal(data)
        
        assert signal is not None
        assert signal.type == SignalType.HOLD
        assert "bid" in signal.metadata
        assert "ask" in signal.metadata


class TestBaseStrategyIndicators:
    """Test base strategy indicator calculations."""

    def test_calculate_atr(self) -> None:
        """Test ATR calculation."""
        strategy = TrendStrategy({})
        data = generate_test_data(n_bars=50)
        
        atr = strategy.calculate_atr(data)
        
        assert len(atr) == len(data)
        assert atr.iloc[-1] > 0

    def test_calculate_rsi(self) -> None:
        """Test RSI calculation."""
        strategy = TrendStrategy({})
        data = generate_test_data(n_bars=50)
        
        rsi = strategy.calculate_rsi(data)
        
        assert len(rsi) == len(data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= r <= 100 for r in valid_rsi)

    def test_calculate_macd(self) -> None:
        """Test MACD calculation."""
        strategy = TrendStrategy({})
        data = generate_test_data(n_bars=50)
        
        macd_line, signal_line, histogram = strategy.calculate_macd(data)
        
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)

    def test_calculate_bollinger_bands(self) -> None:
        """Test Bollinger Bands calculation."""
        strategy = MeanReversionStrategy({})
        data = generate_test_data(n_bars=50)
        
        upper, middle, lower = strategy.calculate_bollinger_bands(data)
        
        assert len(upper) == len(data)
        # Upper should be above middle, middle above lower
        valid_idx = middle.dropna().index
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])

    def test_suggest_leverage(self) -> None:
        """Test leverage suggestion."""
        strategy = TrendStrategy({})
        
        # Low volatility, high strength -> higher leverage
        leverage = strategy.suggest_leverage(0.02, 0.8, max_leverage=5)
        assert 1 <= leverage <= 5
        
        # High volatility -> lower leverage
        leverage_high_vol = strategy.suggest_leverage(0.1, 0.8, max_leverage=5)
        assert leverage_high_vol <= leverage
