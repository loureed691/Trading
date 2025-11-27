"""Base strategy class and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Trading signal types."""

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal."""

    type: SignalType
    symbol: str
    strength: float  # 0 to 1
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    suggested_leverage: int = 1
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class OHLCV:
    """OHLCV candlestick data."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self._indicators: dict[str, Any] = {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate a trading signal based on market data.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            Trading signal or None if no signal
        """
        pass

    def calculate_atr(
        self, data: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_volatility(
        self, data: pd.DataFrame, period: int = 20
    ) -> float:
        """Calculate recent volatility (standard deviation of returns)."""
        returns = np.log(data["close"] / data["close"].shift(1))
        return returns.tail(period).std()

    def calculate_rsi(
        self, data: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = data["close"].rolling(window=period).mean()
        std = data["close"].rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    def suggest_leverage(
        self,
        volatility: float,
        signal_strength: float,
        max_leverage: int = 5,
    ) -> int:
        """Suggest leverage based on volatility and signal strength."""
        # Lower leverage for higher volatility
        vol_factor = max(0.2, 1.0 - (volatility * 10))
        
        # Scale by signal strength
        leverage = max(1, int(max_leverage * vol_factor * signal_strength))
        
        return min(leverage, max_leverage)
