"""Mean reversion strategy using Bollinger Bands and RSI."""

from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI confirmation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("mean_reversion", config)
        self.lookback = config.get("lookback", 20)
        self.entry_zscore = config.get("entry_zscore", 2.0)
        self.exit_zscore = config.get("exit_zscore", 0.5)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.atr_multiplier = config.get("atr_multiplier", 1.5)

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate mean reversion signal."""
        if len(data) < self.lookback + 14:  # Need extra for RSI
            return None

        # Calculate indicators
        upper, middle, lower = self.calculate_bollinger_bands(
            data, self.lookback, self.entry_zscore
        )
        rsi = self.calculate_rsi(data)
        atr = self.calculate_atr(data)
        volatility = self.calculate_volatility(data)

        # Current values
        current_price = data["close"].iloc[-1]
        current_upper = upper.iloc[-1]
        current_middle = middle.iloc[-1]
        current_lower = lower.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]

        # Calculate z-score
        std = data["close"].rolling(window=self.lookback).std().iloc[-1]
        z_score = (current_price - current_middle) / std if std > 0 else 0

        # Generate signals
        # Buy when price below lower band and RSI oversold
        if current_price < current_lower and current_rsi < self.rsi_oversold:
            strength = min(1.0, abs(z_score) / (self.entry_zscore * 1.5))
            stop_loss = current_price - (current_atr * self.atr_multiplier)
            take_profit = current_middle  # Target is mean
            
            return Signal(
                type=SignalType.LONG,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_leverage=self.suggest_leverage(volatility, strength),
                metadata={
                    "z_score": z_score,
                    "rsi": current_rsi,
                    "bb_lower": current_lower,
                    "bb_middle": current_middle,
                    "atr": current_atr,
                },
            )

        # Sell when price above upper band and RSI overbought
        elif current_price > current_upper and current_rsi > self.rsi_overbought:
            strength = min(1.0, abs(z_score) / (self.entry_zscore * 1.5))
            stop_loss = current_price + (current_atr * self.atr_multiplier)
            take_profit = current_middle  # Target is mean
            
            return Signal(
                type=SignalType.SHORT,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_leverage=self.suggest_leverage(volatility, strength),
                metadata={
                    "z_score": z_score,
                    "rsi": current_rsi,
                    "bb_upper": current_upper,
                    "bb_middle": current_middle,
                    "atr": current_atr,
                },
            )

        # Exit signals when returning to mean
        if abs(z_score) < self.exit_zscore:
            return Signal(
                type=SignalType.CLOSE,
                symbol=data.attrs.get("symbol", ""),
                strength=1.0 - abs(z_score) / self.exit_zscore,
                price=current_price,
                metadata={"z_score": z_score, "reason": "mean_reached"},
            )

        return None
