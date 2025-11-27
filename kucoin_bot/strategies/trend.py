"""Trend following strategy using moving averages and MACD."""

from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class TrendStrategy(BaseStrategy):
    """Trend following strategy using dual moving averages and MACD confirmation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("trend", config)
        self.short_period = config.get("short_period", 20)
        self.long_period = config.get("long_period", 50)
        self.signal_threshold = config.get("signal_threshold", 0.02)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate trend-following signal."""
        if len(data) < self.long_period + 10:
            return None

        # Calculate indicators
        short_ma = data["close"].rolling(window=self.short_period).mean()
        long_ma = data["close"].rolling(window=self.long_period).mean()
        macd_line, signal_line, _ = self.calculate_macd(data)
        atr = self.calculate_atr(data)
        volatility = self.calculate_volatility(data)

        # Current values
        current_price = data["close"].iloc[-1]
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_atr = atr.iloc[-1]

        # MA cross detection
        bullish_cross = prev_short <= prev_long and current_short > current_long
        bearish_cross = prev_short >= prev_long and current_short < current_long

        # MACD confirmation
        macd_bullish = current_macd > current_signal
        macd_bearish = current_macd < current_signal

        # Trend strength (distance between MAs)
        ma_diff_pct = abs(current_short - current_long) / current_long

        # Generate signals
        if bullish_cross and macd_bullish and ma_diff_pct > self.signal_threshold:
            strength = min(1.0, ma_diff_pct / (self.signal_threshold * 2))
            stop_loss = current_price - (current_atr * self.atr_multiplier)
            take_profit = current_price + (current_atr * self.atr_multiplier * 2)
            
            return Signal(
                type=SignalType.LONG,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_leverage=self.suggest_leverage(volatility, strength),
                metadata={
                    "short_ma": current_short,
                    "long_ma": current_long,
                    "macd": current_macd,
                    "atr": current_atr,
                },
            )

        elif bearish_cross and macd_bearish and ma_diff_pct > self.signal_threshold:
            strength = min(1.0, ma_diff_pct / (self.signal_threshold * 2))
            stop_loss = current_price + (current_atr * self.atr_multiplier)
            take_profit = current_price - (current_atr * self.atr_multiplier * 2)
            
            return Signal(
                type=SignalType.SHORT,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_leverage=self.suggest_leverage(volatility, strength),
                metadata={
                    "short_ma": current_short,
                    "long_ma": current_long,
                    "macd": current_macd,
                    "atr": current_atr,
                },
            )

        return None
