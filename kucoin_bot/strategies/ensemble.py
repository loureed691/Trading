"""Ensemble strategy with regime-based allocation."""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from ..regime.detector import RegimeDetector, RegimeState

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    """Ensemble signal combining multiple strategy signals."""

    signal: Signal | None
    contributing_strategies: list[str]
    weights_used: dict[str, float]
    regime_state: RegimeState | None


class EnsembleStrategy(BaseStrategy):
    """Ensemble strategy that combines multiple strategies based on regime."""

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        config: dict[str, Any] | None = None,
    ):
        super().__init__("ensemble", config or {})
        self.strategies = strategies
        self.regime_detector = RegimeDetector(config.get("regime", {}) if config else {})
        self.min_agreement = config.get("min_agreement", 0.5) if config else 0.5
        self.strength_threshold = config.get("strength_threshold", 0.3) if config else 0.3
        self.current_regime: RegimeState | None = None

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate ensemble signal based on regime-weighted strategy combination."""
        # Detect current regime
        self.current_regime = self.regime_detector.detect_regime(data)
        
        # Get strategy weights based on regime
        weights = self.regime_detector.get_strategy_weights(self.current_regime)
        
        # Collect signals from all strategies
        signals: list[tuple[str, Signal, float]] = []
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(data)
                if signal and signal.type != SignalType.HOLD:
                    weight = weights.get(name, 0.0)
                    if weight > 0:
                        signals.append((name, signal, weight))
            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}")
        
        if not signals:
            return None
        
        # Aggregate signals
        return self._aggregate_signals(signals, data)

    def _aggregate_signals(
        self,
        signals: list[tuple[str, Signal, float]],
        data: pd.DataFrame,
    ) -> Signal | None:
        """Aggregate multiple signals into a single ensemble signal."""
        if not signals:
            return None
        
        # Separate by direction
        long_signals = [
            (n, s, w) for n, s, w in signals if s.type == SignalType.LONG
        ]
        short_signals = [
            (n, s, w) for n, s, w in signals if s.type == SignalType.SHORT
        ]
        close_signals = [
            (n, s, w) for n, s, w in signals if s.type == SignalType.CLOSE
        ]
        
        # Calculate weighted votes
        long_vote = sum(s.strength * w for _, s, w in long_signals)
        short_vote = sum(s.strength * w for _, s, w in short_signals)
        close_vote = sum(s.strength * w for _, s, w in close_signals)
        
        total_vote = long_vote + short_vote + close_vote
        if total_vote == 0:
            return None
        
        # Determine direction
        if close_vote > long_vote and close_vote > short_vote:
            signal_type = SignalType.CLOSE
            contributing = close_signals
            strength = close_vote / total_vote
        elif long_vote > short_vote:
            signal_type = SignalType.LONG
            contributing = long_signals
            strength = long_vote / total_vote
        else:
            signal_type = SignalType.SHORT
            contributing = short_signals
            strength = short_vote / total_vote
        
        # Check minimum agreement
        agreement = len(contributing) / len(signals)
        if agreement < self.min_agreement:
            logger.debug(f"Insufficient agreement: {agreement:.2f} < {self.min_agreement}")
            return None
        
        if strength < self.strength_threshold:
            logger.debug(f"Insufficient strength: {strength:.2f} < {self.strength_threshold}")
            return None
        
        # Calculate weighted average price and stops
        current_price = data["close"].iloc[-1]
        
        weighted_stop_loss = 0.0
        weighted_take_profit = 0.0
        total_weight = 0.0
        
        for name, signal, weight in contributing:
            if signal.stop_loss:
                weighted_stop_loss += signal.stop_loss * weight * signal.strength
            if signal.take_profit:
                weighted_take_profit += signal.take_profit * weight * signal.strength
            total_weight += weight * signal.strength
        
        stop_loss = weighted_stop_loss / total_weight if total_weight > 0 else None
        take_profit = weighted_take_profit / total_weight if total_weight > 0 else None
        
        # Average suggested leverage
        avg_leverage = sum(
            s.suggested_leverage * w for _, s, w in contributing
        ) / sum(w for _, _, w in contributing)
        
        return Signal(
            type=signal_type,
            symbol=data.attrs.get("symbol", ""),
            strength=strength,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            suggested_leverage=int(avg_leverage),
            metadata={
                "regime": self.current_regime.regime.value if self.current_regime else "unknown",
                "regime_confidence": self.current_regime.confidence if self.current_regime else 0,
                "contributing_strategies": [n for n, _, _ in contributing],
                "agreement": agreement,
            },
        )

    def get_ensemble_signal(self, data: pd.DataFrame) -> EnsembleSignal:
        """Get detailed ensemble signal with all metadata."""
        signal = self.generate_signal(data)
        
        # Get weights used
        weights = {}
        if self.current_regime:
            weights = self.regime_detector.get_strategy_weights(self.current_regime)
        
        return EnsembleSignal(
            signal=signal,
            contributing_strategies=signal.metadata.get("contributing_strategies", []) if signal else [],
            weights_used=weights,
            regime_state=self.current_regime,
        )
