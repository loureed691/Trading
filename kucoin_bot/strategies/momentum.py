"""Momentum strategy using rate of change and RSI divergence."""

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using ROC, RSI, and volume momentum.
    
    Combines multiple momentum indicators:
    - Rate of Change (ROC) for price momentum
    - RSI for momentum confirmation
    - Volume momentum for confirmation
    - Price acceleration for trend strength
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__("momentum", config)
        self.roc_period = config.get("roc_period", 10)
        self.rsi_period = config.get("rsi_period", 14)
        self.volume_period = config.get("volume_period", 10)
        self.momentum_threshold = config.get("momentum_threshold", 0.02)  # 2% momentum
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)

    def _calculate_roc(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Rate of Change."""
        return (data["close"] - data["close"].shift(period)) / data["close"].shift(period)

    def _calculate_momentum_score(
        self,
        roc: float,
        rsi: float,
        volume_momentum: float,
        acceleration: float,
    ) -> float:
        """Calculate composite momentum score (-1 to 1).
        
        Positive values indicate bullish momentum, negative values indicate bearish.
        """
        # ROC score (capped at +/- 10%)
        roc_score = min(1.0, max(-1.0, roc * 10))
        
        # RSI score (normalized to -1 to 1)
        rsi_normalized = (rsi - 50) / 50
        
        # Volume momentum score
        vol_score = min(1.0, max(-1.0, volume_momentum - 1))
        
        # Acceleration score
        accel_score = min(1.0, max(-1.0, acceleration * 20))
        
        # Weighted combination
        composite = (
            roc_score * 0.4 +
            rsi_normalized * 0.25 +
            vol_score * 0.2 +
            accel_score * 0.15
        )
        
        return composite

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate momentum-based trading signal."""
        min_bars = max(self.roc_period, self.rsi_period, self.volume_period) + 10
        if len(data) < min_bars:
            return None

        # Calculate indicators
        roc = self._calculate_roc(data, self.roc_period)
        rsi = self.calculate_rsi(data, self.rsi_period)
        atr = self.calculate_atr(data)
        volatility = self.calculate_volatility(data)

        # Volume momentum
        volume_ma = data["volume"].rolling(window=self.volume_period).mean()
        volume_momentum = data["volume"] / volume_ma

        # Price acceleration (second derivative of price)
        price_velocity = data["close"].diff()
        price_acceleration = price_velocity.diff() / data["close"]

        # Current values
        current_price = data["close"].iloc[-1]
        current_roc = roc.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_vol_momentum = volume_momentum.iloc[-1]
        current_acceleration = price_acceleration.iloc[-1]
        current_atr = atr.iloc[-1]

        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(
            current_roc, current_rsi, current_vol_momentum, current_acceleration
        )

        # Generate signals based on momentum
        # Strong bullish momentum
        if (
            current_roc > self.momentum_threshold and
            current_rsi < self.rsi_overbought and
            current_rsi > 40 and  # Not too weak
            momentum_score > 0.3
        ):
            strength = min(1.0, abs(momentum_score))
            stop_loss = current_price - (current_atr * self.atr_multiplier)
            take_profit = current_price + (current_atr * self.atr_multiplier * 2.5)

            return Signal(
                type=SignalType.LONG,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_leverage=self.suggest_leverage(volatility, strength),
                metadata={
                    "roc": current_roc,
                    "rsi": current_rsi,
                    "momentum_score": momentum_score,
                    "volume_momentum": current_vol_momentum,
                    "acceleration": current_acceleration,
                },
            )

        # Strong bearish momentum
        elif (
            current_roc < -self.momentum_threshold and
            current_rsi > self.rsi_oversold and
            current_rsi < 60 and  # Not too strong
            momentum_score < -0.3
        ):
            strength = min(1.0, abs(momentum_score))
            stop_loss = current_price + (current_atr * self.atr_multiplier)
            take_profit = current_price - (current_atr * self.atr_multiplier * 2.5)

            return Signal(
                type=SignalType.SHORT,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_leverage=self.suggest_leverage(volatility, strength),
                metadata={
                    "roc": current_roc,
                    "rsi": current_rsi,
                    "momentum_score": momentum_score,
                    "volume_momentum": current_vol_momentum,
                    "acceleration": current_acceleration,
                },
            )

        return None
