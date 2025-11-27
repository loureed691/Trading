"""Breakout strategy using price channels and volume confirmation."""

from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy using price channels with volume confirmation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("breakout", config)
        self.lookback = config.get("lookback", 20)
        self.volume_multiplier = config.get("volume_multiplier", 1.5)
        self.atr_multiplier = config.get("atr_multiplier", 1.0)
        self.confirmation_bars = config.get("confirmation_bars", 2)

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate breakout signal."""
        if len(data) < self.lookback + 10:
            return None

        # Calculate price channels
        high_channel = data["high"].rolling(window=self.lookback).max()
        low_channel = data["low"].rolling(window=self.lookback).min()
        
        # Calculate average volume
        avg_volume = data["volume"].rolling(window=self.lookback).mean()
        
        # Calculate ATR
        atr = self.calculate_atr(data)
        volatility = self.calculate_volatility(data)

        # Current values
        current_price = data["close"].iloc[-1]
        current_high = data["high"].iloc[-1]
        current_low = data["low"].iloc[-1]
        current_volume = data["volume"].iloc[-1]
        
        # Previous channel values (excluding current bar)
        prev_high_channel = high_channel.iloc[-2]
        prev_low_channel = low_channel.iloc[-2]
        prev_avg_volume = avg_volume.iloc[-2]
        current_atr = atr.iloc[-1]

        # Volume confirmation
        volume_confirmed = current_volume > prev_avg_volume * self.volume_multiplier

        # Check for breakout with confirmation
        # Bullish breakout: price closes above channel high with volume
        if current_high > prev_high_channel and volume_confirmed:
            # Check for false breakout (price should stay above)
            breakout_strength = (current_price - prev_high_channel) / current_atr
            
            if breakout_strength > 0:
                strength = min(1.0, breakout_strength / 2)
                stop_loss = prev_high_channel - (current_atr * self.atr_multiplier)
                take_profit = current_price + (current_atr * 3)
                
                return Signal(
                    type=SignalType.LONG,
                    symbol=data.attrs.get("symbol", ""),
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    suggested_leverage=self.suggest_leverage(volatility, strength),
                    metadata={
                        "breakout_level": prev_high_channel,
                        "breakout_strength": breakout_strength,
                        "volume_ratio": current_volume / prev_avg_volume,
                        "atr": current_atr,
                    },
                )

        # Bearish breakout: price closes below channel low with volume
        elif current_low < prev_low_channel and volume_confirmed:
            breakout_strength = (prev_low_channel - current_price) / current_atr
            
            if breakout_strength > 0:
                strength = min(1.0, breakout_strength / 2)
                stop_loss = prev_low_channel + (current_atr * self.atr_multiplier)
                take_profit = current_price - (current_atr * 3)
                
                return Signal(
                    type=SignalType.SHORT,
                    symbol=data.attrs.get("symbol", ""),
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    suggested_leverage=self.suggest_leverage(volatility, strength),
                    metadata={
                        "breakout_level": prev_low_channel,
                        "breakout_strength": breakout_strength,
                        "volume_ratio": current_volume / prev_avg_volume,
                        "atr": current_atr,
                    },
                )

        return None
