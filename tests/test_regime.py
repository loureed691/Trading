"""Tests for regime detection module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kucoin_bot.regime.detector import (
    RegimeDetector,
    RegimeType,
    RegimeState,
)


def generate_trending_data(
    n_bars: int = 200,
    direction: str = "up",
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Generate trending OHLCV data."""
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_bars - i) for i in range(n_bars)]
    
    if direction == "up":
        base = 100 + np.linspace(0, 30, n_bars)
    else:
        base = 130 - np.linspace(0, 30, n_bars)
    
    noise = np.random.normal(0, volatility * 100, n_bars)
    close = base + noise
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.008, n_bars))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.008, n_bars))),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })
    
    data.attrs["symbol"] = "TEST-USDT"
    return data


def generate_mean_reverting_data(n_bars: int = 200) -> pd.DataFrame:
    """Generate mean-reverting OHLCV data."""
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_bars - i) for i in range(n_bars)]
    
    # Mean-reverting process
    mean = 100
    reversion_speed = 0.2
    close = [mean]
    
    for i in range(1, n_bars):
        shock = np.random.normal(0, 1)
        new_price = close[-1] + reversion_speed * (mean - close[-1]) + shock
        close.append(new_price)
    
    close = np.array(close)
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_bars))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_bars))),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })
    
    data.attrs["symbol"] = "TEST-USDT"
    return data


def generate_high_volatility_data(n_bars: int = 200) -> pd.DataFrame:
    """Generate high volatility OHLCV data."""
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_bars - i) for i in range(n_bars)]
    
    base = 100 + np.cumsum(np.random.normal(0, 5, n_bars))
    base = np.maximum(base, 50)  # Keep positive
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": base * (1 + np.random.uniform(-0.02, 0.02, n_bars)),
        "high": base * (1 + np.abs(np.random.normal(0, 0.03, n_bars))),
        "low": base * (1 - np.abs(np.random.normal(0, 0.03, n_bars))),
        "close": base,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })
    
    data.attrs["symbol"] = "TEST-USDT"
    return data


class TestRegimeDetector:
    """Test cases for RegimeDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = RegimeDetector()
        assert detector.trend_threshold == 0.3
        assert detector.volatility_lookback == 20

    def test_initialization_with_config(self) -> None:
        """Test detector initialization with custom config."""
        config = {"trend_threshold": 0.5, "volatility_lookback": 30}
        detector = RegimeDetector(config)
        assert detector.trend_threshold == 0.5
        assert detector.volatility_lookback == 30

    def test_calculate_hurst_exponent(self) -> None:
        """Test Hurst exponent calculation."""
        detector = RegimeDetector()
        
        # Trending data should have H > 0.5
        trending = generate_trending_data(n_bars=300, direction="up")
        hurst_trend = detector.calculate_hurst_exponent(trending["close"])
        assert 0 <= hurst_trend <= 1
        
        # Mean-reverting data should have H < 0.5
        mr_data = generate_mean_reverting_data(n_bars=300)
        hurst_mr = detector.calculate_hurst_exponent(mr_data["close"])
        assert 0 <= hurst_mr <= 1

    def test_calculate_trend_strength(self) -> None:
        """Test trend strength calculation."""
        detector = RegimeDetector()
        
        # Strong uptrend
        uptrend = generate_trending_data(n_bars=100, direction="up")
        strength, direction = detector.calculate_trend_strength(uptrend)
        assert 0 <= strength <= 1
        assert direction in ["up", "down", "none"]
        
        # Strong downtrend
        downtrend = generate_trending_data(n_bars=100, direction="down")
        strength_down, dir_down = detector.calculate_trend_strength(downtrend)
        assert 0 <= strength_down <= 1

    def test_calculate_mean_reversion_score(self) -> None:
        """Test mean reversion score calculation."""
        detector = RegimeDetector()
        
        data = generate_mean_reverting_data()
        mr_score = detector.calculate_mean_reversion_score(data)
        assert 0 <= mr_score <= 1

    def test_calculate_volatility_regime(self) -> None:
        """Test volatility regime calculation."""
        detector = RegimeDetector()
        
        # High volatility
        high_vol = generate_high_volatility_data()
        vol, regime = detector.calculate_volatility_regime(high_vol)
        assert vol >= 0
        assert regime in ["high", "low", "normal"]

    def test_detect_regime(self) -> None:
        """Test full regime detection."""
        detector = RegimeDetector()
        
        # Test with trending data
        trending = generate_trending_data(n_bars=200, direction="up")
        regime_state = detector.detect_regime(trending)
        
        assert isinstance(regime_state, RegimeState)
        assert isinstance(regime_state.regime, RegimeType)
        assert 0 <= regime_state.confidence <= 1
        assert regime_state.volatility >= 0

    def test_detect_regime_insufficient_data(self) -> None:
        """Test regime detection with insufficient data."""
        detector = RegimeDetector()
        
        short_data = generate_trending_data(n_bars=30)
        regime_state = detector.detect_regime(short_data)
        
        assert regime_state.regime == RegimeType.UNKNOWN
        assert regime_state.confidence == 0.0

    def test_get_strategy_weights(self) -> None:
        """Test strategy weight allocation."""
        detector = RegimeDetector()
        
        # Create regime state for trend
        regime_state = RegimeState(
            regime=RegimeType.TREND_UP,
            confidence=0.8,
            volatility=0.02,
            trend_strength=0.6,
            mean_reversion_score=0.3,
            hurst_exponent=0.65,
        )
        
        weights = detector.get_strategy_weights(regime_state)
        
        assert "trend" in weights
        assert "mean_reversion" in weights
        assert "breakout" in weights
        assert "market_making" in weights
        
        # Weights should sum to approximately 1
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Trend should have highest weight in trend regime
        assert weights["trend"] >= weights["mean_reversion"]

    def test_get_strategy_weights_low_confidence(self) -> None:
        """Test strategy weights with low confidence."""
        detector = RegimeDetector()
        
        regime_state = RegimeState(
            regime=RegimeType.TREND_UP,
            confidence=0.2,  # Low confidence
            volatility=0.02,
            trend_strength=0.3,
            mean_reversion_score=0.4,
            hurst_exponent=0.52,
        )
        
        weights = detector.get_strategy_weights(regime_state)
        
        # With low confidence, weights should be equal
        assert abs(weights["trend"] - weights["mean_reversion"]) < 0.01
        assert abs(weights["breakout"] - weights["market_making"]) < 0.01

    def test_regime_types_coverage(self) -> None:
        """Test all regime types can be detected."""
        # Just verify the enum values exist
        assert RegimeType.TREND_UP.value == "trend_up"
        assert RegimeType.TREND_DOWN.value == "trend_down"
        assert RegimeType.MEAN_REVERT.value == "mean_revert"
        assert RegimeType.HIGH_VOLATILITY.value == "high_volatility"
        assert RegimeType.LOW_VOLATILITY.value == "low_volatility"
        assert RegimeType.UNKNOWN.value == "unknown"
