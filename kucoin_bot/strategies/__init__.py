"""Strategy modules for trading."""

from .base import BaseStrategy, Signal, SignalType
from .breakout import BreakoutStrategy
from .ensemble import EnsembleStrategy
from .market_making import MarketMakingStrategy, ArbitrageStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .trend import TrendStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "BreakoutStrategy",
    "EnsembleStrategy",
    "MarketMakingStrategy",
    "ArbitrageStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "TrendStrategy",
]
