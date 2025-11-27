"""ML-based position sizing with robust validation."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MLPositionSize:
    """ML-recommended position size."""

    suggested_size_pct: float
    confidence: float
    volatility_adjusted: bool
    forecast_adjusted: bool
    constraints_applied: list[str]


class MLPositionSizer:
    """ML-based position sizing with cross-validation.
    
    Combines volatility targeting, forecast confidence, and risk constraints
    to suggest position sizes that adapt to market conditions.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.target_volatility = config.get("target_volatility", 0.10)
        self.max_position_pct = config.get("max_position_pct", 0.05)
        self.min_position_pct = config.get("min_position_pct", 0.005)
        self.vol_lookback = config.get("vol_lookback", 20)
        self.min_cv_score = config.get("min_cv_score", 0.55)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)  # Fractional Kelly

    def calculate_volatility_adjusted_size(
        self, data: pd.DataFrame, portfolio_value: float
    ) -> float:
        """Calculate position size based on volatility targeting."""
        if len(data) < self.vol_lookback:
            return self.min_position_pct
        
        # Calculate recent volatility
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        recent_vol = returns.iloc[-self.vol_lookback:].std() * np.sqrt(252)  # Annualized
        
        if recent_vol <= 0:
            return self.min_position_pct
        
        # Target volatility sizing
        vol_adjusted_size = self.target_volatility / recent_vol
        
        # Cap at max position
        return min(self.max_position_pct, max(self.min_position_pct, vol_adjusted_size))

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate Kelly criterion position size.
        
        Uses fractional Kelly for safety.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self.min_position_pct
        
        # Kelly formula: f = p - (1-p)/b where b = avg_win/avg_loss
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        kelly = win_rate - (1 - win_rate) / b
        
        # Apply fractional Kelly
        fractional_kelly = kelly * self.kelly_fraction
        
        return max(0, min(self.max_position_pct, fractional_kelly))

    def get_position_size(
        self,
        data: pd.DataFrame,
        portfolio_value: float,
        signal_strength: float,
        forecast_confidence: float = 0.0,
        historical_win_rate: float = 0.5,
        historical_avg_win: float = 0.0,
        historical_avg_loss: float = 0.0,
    ) -> MLPositionSize:
        """Calculate recommended position size.
        
        Combines multiple sizing approaches with confidence weighting.
        """
        constraints_applied = []
        
        # Base size from volatility targeting
        vol_size = self.calculate_volatility_adjusted_size(data, portfolio_value)
        constraints_applied.append("volatility_targeting")
        
        # Kelly sizing if we have reliable backtest data
        kelly_size = self.max_position_pct
        if historical_win_rate > 0 and historical_avg_win != 0:
            kelly_size = self.calculate_kelly_size(
                historical_win_rate, historical_avg_win, historical_avg_loss
            )
            constraints_applied.append("kelly_criterion")
        
        # Combine sizes (take minimum for safety)
        base_size = min(vol_size, kelly_size)
        
        # Adjust for signal strength
        signal_adjusted = base_size * signal_strength
        constraints_applied.append("signal_strength")
        
        # Adjust for forecast confidence (if ML is reliable)
        forecast_adjusted = False
        if forecast_confidence >= self.min_cv_score:
            # Increase size if forecast is confident
            confidence_multiplier = 1 + (forecast_confidence - 0.5)
            signal_adjusted *= confidence_multiplier
            forecast_adjusted = True
            constraints_applied.append("forecast_confidence")
        elif forecast_confidence > 0 and forecast_confidence < self.min_cv_score:
            # Reduce size if forecast is unreliable
            signal_adjusted *= 0.5
            constraints_applied.append("low_forecast_confidence_reduction")
        
        # Final constraints
        final_size = max(self.min_position_pct, min(self.max_position_pct, signal_adjusted))
        
        return MLPositionSize(
            suggested_size_pct=final_size,
            confidence=forecast_confidence,
            volatility_adjusted=True,
            forecast_adjusted=forecast_adjusted,
            constraints_applied=constraints_applied,
        )

    def cross_validate_sizing(
        self,
        data: pd.DataFrame,
        signals: list[tuple[int, float]],  # (index, signal_strength)
        portfolio_value: float,
    ) -> dict[str, float]:
        """Cross-validate position sizing strategy.
        
        Returns metrics about sizing performance.
        """
        if len(signals) < 10:
            return {"cv_score": 0.0, "avg_return": 0.0}
        
        # Split into train/test
        split_idx = len(signals) * 3 // 4
        train_signals = signals[:split_idx]
        test_signals = signals[split_idx:]
        
        # Simulate trades with different sizing
        results = []
        
        for idx, strength in test_signals:
            if idx >= len(data) - 5:
                continue
            
            # Get position size
            hist_data = data.iloc[:idx].copy()
            if len(hist_data) < 50:
                continue
            
            size_result = self.get_position_size(
                hist_data, portfolio_value, strength
            )
            
            # Calculate forward return
            forward_return = (
                data["close"].iloc[idx + 5] / data["close"].iloc[idx] - 1
            )
            
            # PnL with sizing
            pnl = forward_return * size_result.suggested_size_pct
            results.append(pnl)
        
        if not results:
            return {"cv_score": 0.0, "avg_return": 0.0}
        
        avg_return = np.mean(results)
        win_rate = np.mean([1 if r > 0 else 0 for r in results])
        
        return {
            "cv_score": win_rate,
            "avg_return": avg_return,
            "sharpe": avg_return / np.std(results) if np.std(results) > 0 else 0,
            "n_trades": len(results),
        }
