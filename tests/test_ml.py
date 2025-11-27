"""Tests for ML forecaster and position sizer modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kucoin_bot.ml.forecaster import MLForecaster, ForecastResult
from kucoin_bot.ml.position_sizer import MLPositionSizer, MLPositionSize


def generate_test_data(
    n_bars: int = 500,
    trend: str = "neutral",
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_bars - i) for i in range(n_bars)]
    
    if trend == "up":
        base = 100 + np.linspace(0, 30, n_bars)
    elif trend == "down":
        base = 130 - np.linspace(0, 30, n_bars)
    else:
        base = np.full(n_bars, 100.0)
    
    noise = np.random.normal(0, volatility * 100, n_bars)
    close = base + noise
    close = np.maximum(close, 10)
    
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


class TestMLForecaster:
    """Test cases for MLForecaster."""

    def test_initialization(self) -> None:
        """Test forecaster initialization."""
        forecaster = MLForecaster()
        assert forecaster.lookback == 100
        assert forecaster.forecast_horizon == 5
        assert forecaster.min_cv_score == 0.51

    def test_initialization_with_config(self) -> None:
        """Test forecaster initialization with custom config."""
        config = {
            "lookback": 50,
            "forecast_horizon": 10,
            "min_cv_score": 0.55,
        }
        forecaster = MLForecaster(config)
        assert forecaster.lookback == 50
        assert forecaster.forecast_horizon == 10
        assert forecaster.min_cv_score == 0.55

    def test_create_features(self) -> None:
        """Test feature creation."""
        forecaster = MLForecaster()
        data = generate_test_data(n_bars=200)
        
        features_df = forecaster._create_features(data)
        
        # Check that features are created
        assert "return_1" in features_df.columns
        assert "volatility_10" in features_df.columns
        assert "momentum_5" in features_df.columns
        assert "zscore_20" in features_df.columns
        assert "target" in features_df.columns

    def test_cross_validate(self) -> None:
        """Test cross-validation."""
        forecaster = MLForecaster({"min_training_samples": 100})
        data = generate_test_data(n_bars=300)
        
        cv_score = forecaster.cross_validate(data)
        
        # Score should be between 0 and 1
        assert 0 <= cv_score <= 1

    def test_cross_validate_insufficient_data(self) -> None:
        """Test cross-validation with insufficient data."""
        forecaster = MLForecaster({"min_training_samples": 500})
        data = generate_test_data(n_bars=100)
        
        cv_score = forecaster.cross_validate(data)
        
        # Should return 0 for insufficient data
        assert cv_score == 0.0

    def test_forecast(self) -> None:
        """Test forecast generation."""
        forecaster = MLForecaster({"min_training_samples": 100, "min_cv_score": 0.0})
        data = generate_test_data(n_bars=300, trend="up")
        
        result = forecaster.forecast(data)
        
        assert isinstance(result, ForecastResult)
        assert result.direction in ["up", "down", "neutral"]
        assert 0 <= result.cv_score <= 1

    def test_forecast_insufficient_data(self) -> None:
        """Test forecast with insufficient data."""
        forecaster = MLForecaster({"min_training_samples": 500})
        data = generate_test_data(n_bars=100)
        
        result = forecaster.forecast(data)
        
        assert result.is_reliable is False
        assert result.direction == "neutral"

    def test_check_data_leakage(self) -> None:
        """Test data leakage checks."""
        forecaster = MLForecaster()
        data = generate_test_data(n_bars=200)
        
        warnings = forecaster.check_data_leakage(data)
        
        # Should return list of warnings (may be empty if no issues)
        assert isinstance(warnings, list)

    def test_purged_cv_split(self) -> None:
        """Test purged cross-validation splitting."""
        forecaster = MLForecaster()
        
        splits = forecaster._purged_cv_split(n_samples=500, n_folds=5, gap=5)
        
        # Should have splits
        assert len(splits) > 0
        
        # Check that train and test indices don't overlap
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestMLPositionSizer:
    """Test cases for MLPositionSizer."""

    def test_initialization(self) -> None:
        """Test position sizer initialization."""
        sizer = MLPositionSizer()
        assert sizer.target_volatility == 0.10
        assert sizer.max_position_pct == 0.05
        assert sizer.min_position_pct == 0.005

    def test_initialization_with_config(self) -> None:
        """Test position sizer initialization with custom config."""
        config = {
            "target_volatility": 0.15,
            "max_position_pct": 0.10,
            "min_position_pct": 0.01,
        }
        sizer = MLPositionSizer(config)
        assert sizer.target_volatility == 0.15
        assert sizer.max_position_pct == 0.10
        assert sizer.min_position_pct == 0.01

    def test_calculate_volatility_adjusted_size(self) -> None:
        """Test volatility-adjusted position sizing."""
        sizer = MLPositionSizer()
        data = generate_test_data(n_bars=100)
        
        size = sizer.calculate_volatility_adjusted_size(data, portfolio_value=10000)
        
        assert sizer.min_position_pct <= size <= sizer.max_position_pct

    def test_calculate_kelly_size(self) -> None:
        """Test Kelly criterion sizing."""
        sizer = MLPositionSizer()
        
        # Positive edge
        size = sizer.calculate_kelly_size(
            win_rate=0.6,
            avg_win=100,
            avg_loss=-50,
        )
        assert size >= 0
        assert size <= sizer.max_position_pct
        
        # Negative edge
        size_neg = sizer.calculate_kelly_size(
            win_rate=0.3,
            avg_win=50,
            avg_loss=-100,
        )
        assert size_neg >= 0

    def test_get_position_size(self) -> None:
        """Test full position sizing calculation."""
        sizer = MLPositionSizer()
        data = generate_test_data(n_bars=100)
        
        result = sizer.get_position_size(
            data=data,
            portfolio_value=10000,
            signal_strength=0.8,
            forecast_confidence=0.6,
        )
        
        assert isinstance(result, MLPositionSize)
        assert sizer.min_position_pct <= result.suggested_size_pct <= sizer.max_position_pct
        assert result.volatility_adjusted is True
        assert len(result.constraints_applied) > 0

    def test_get_position_size_low_confidence(self) -> None:
        """Test position sizing with low forecast confidence."""
        sizer = MLPositionSizer({"min_cv_score": 0.55})
        data = generate_test_data(n_bars=100)
        
        result = sizer.get_position_size(
            data=data,
            portfolio_value=10000,
            signal_strength=0.8,
            forecast_confidence=0.4,  # Below min_cv_score
        )
        
        # Should have low confidence reduction
        assert "low_forecast_confidence_reduction" in result.constraints_applied

    def test_get_position_size_with_backtest_data(self) -> None:
        """Test position sizing with historical backtest data."""
        sizer = MLPositionSizer()
        data = generate_test_data(n_bars=100)
        
        result = sizer.get_position_size(
            data=data,
            portfolio_value=10000,
            signal_strength=0.7,
            forecast_confidence=0.0,
            historical_win_rate=0.55,
            historical_avg_win=100,
            historical_avg_loss=-60,
        )
        
        # Should use Kelly criterion
        assert "kelly_criterion" in result.constraints_applied

    def test_cross_validate_sizing(self) -> None:
        """Test cross-validation of sizing strategy."""
        sizer = MLPositionSizer()
        data = generate_test_data(n_bars=200)
        
        # Create mock signals
        signals = [(i, 0.5 + np.random.random() * 0.5) for i in range(50, 180)]
        
        results = sizer.cross_validate_sizing(data, signals, portfolio_value=10000)
        
        assert "cv_score" in results
        assert "avg_return" in results
        assert 0 <= results["cv_score"] <= 1
