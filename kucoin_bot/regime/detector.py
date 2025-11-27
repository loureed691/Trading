"""Market regime detection using statistical methods."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with confidence."""

    regime: RegimeType
    confidence: float
    volatility: float
    trend_strength: float
    mean_reversion_score: float
    hurst_exponent: float
    metadata: dict[str, Any] | None = None


class RegimeDetector:
    """Detect market regime for strategy allocation."""

    # Default threshold for high volatility regime detection
    HIGH_VOLATILITY_THRESHOLD = 0.03
    LOW_VOLATILITY_THRESHOLD = 0.01

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.trend_threshold = config.get("trend_threshold", 0.3)
        self.volatility_lookback = config.get("volatility_lookback", 20)
        self.trend_lookback = config.get("trend_lookback", 50)
        self.hurst_lookback = config.get("hurst_lookback", 100)
        self.volatility_high_pct = config.get("volatility_high_pct", 80)
        self.volatility_low_pct = config.get("volatility_low_pct", 20)
        self.high_vol_threshold = config.get("high_vol_threshold", self.HIGH_VOLATILITY_THRESHOLD)
        self.low_vol_threshold = config.get("low_vol_threshold", self.LOW_VOLATILITY_THRESHOLD)
        # Multiplier for data window size per lag in Hurst calculation
        # Higher values provide more data points for R/S calculation but require more historical data
        self.hurst_window_multiplier = config.get("hurst_window_multiplier", 10)

    def calculate_hurst_exponent(
        self, data: pd.Series, max_lag: int = 20
    ) -> float:
        """Calculate Hurst exponent to determine trend vs mean-reversion.
        
        H > 0.5: Trending (persistent)
        H < 0.5: Mean-reverting (anti-persistent)
        H ≈ 0.5: Random walk
        """
        if len(data) < max_lag * 2:
            return 0.5  # Default to random walk

        lags = range(2, max_lag)
        rs_values = []

        for lag in lags:
            try:
                # Calculate R/S statistic using a window proportional to the lag
                returns = np.diff(np.log(data.iloc[-lag * self.hurst_window_multiplier :].values))
                if len(returns) < lag:
                    continue

                # Divide into chunks
                n_chunks = len(returns) // lag
                if n_chunks < 2:
                    continue

                chunk_rs = []
                for i in range(n_chunks):
                    chunk = returns[i * lag : (i + 1) * lag]
                    if len(chunk) < 2:
                        continue

                    # Calculate cumulative deviations
                    mean_ret = np.mean(chunk)
                    cumdev = np.cumsum(chunk - mean_ret)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = np.std(chunk, ddof=1)

                    if s > 0:
                        chunk_rs.append(r / s)

                if chunk_rs:
                    rs_values.append((lag, np.mean(chunk_rs)))
            except Exception:
                continue

        if len(rs_values) < 3:
            return 0.5

        # Linear regression on log-log scale
        lags_arr = np.array([np.log(x[0]) for x in rs_values])
        rs_arr = np.array([np.log(x[1]) for x in rs_values])

        try:
            slope, _ = np.polyfit(lags_arr, rs_arr, 1)
            return max(0.0, min(1.0, slope))
        except Exception:
            return 0.5

    def calculate_trend_strength(self, data: pd.DataFrame) -> tuple[float, str]:
        """Calculate trend strength using ADX-like measure.
        
        Returns (strength, direction) where strength is 0-1 and direction is 'up' or 'down'.
        """
        if len(data) < self.trend_lookback:
            return 0.0, "none"

        close = data["close"].iloc[-self.trend_lookback :]
        high = data["high"].iloc[-self.trend_lookback :]
        low = data["low"].iloc[-self.trend_lookback :]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        if atr <= 0:
            return 0.0, "none"

        # Directional movement
        plus_dm = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0,
        )
        minus_dm = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0,
        )

        # Smoothed DI
        plus_di = pd.Series(plus_dm).rolling(14).mean().iloc[-1] / atr * 100
        minus_di = pd.Series(minus_dm).rolling(14).mean().iloc[-1] / atr * 100

        # ADX-like measure
        if (plus_di + minus_di) > 0:
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        else:
            dx = 0

        # Normalize to 0-1
        strength = min(1.0, dx / 100)
        direction = "up" if plus_di > minus_di else "down"

        return strength, direction

    def calculate_mean_reversion_score(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion tendency score.
        
        Uses autocorrelation and deviation from mean.
        """
        if len(data) < self.volatility_lookback * 2:
            return 0.5

        returns = np.log(
            data["close"].iloc[-self.volatility_lookback * 2 :]
            / data["close"].iloc[-self.volatility_lookback * 2 :].shift(1)
        ).dropna()

        if len(returns) < 10:
            return 0.5

        # Autocorrelation at lag 1 (negative = mean reverting)
        autocorr = returns.autocorr(lag=1)
        if np.isnan(autocorr):
            autocorr = 0

        # Score: more negative autocorr = higher mean reversion
        mr_score = max(0.0, min(1.0, 0.5 - autocorr))

        return mr_score

    def calculate_volatility_regime(
        self, data: pd.DataFrame
    ) -> tuple[float, str]:
        """Calculate volatility level relative to history.
        
        Returns (volatility, regime) where regime is 'high', 'low', or 'normal'.
        """
        if len(data) < self.volatility_lookback * 5:
            return 0.0, "normal"

        # Recent volatility
        returns = np.log(
            data["close"] / data["close"].shift(1)
        ).dropna()

        recent_vol = returns.iloc[-self.volatility_lookback :].std()

        # Percentile of recent vol vs history
        rolling_vol = returns.rolling(self.volatility_lookback).std()
        pct = (rolling_vol <= recent_vol).mean() * 100

        if pct >= self.volatility_high_pct:
            regime = "high"
        elif pct <= self.volatility_low_pct:
            regime = "low"
        else:
            regime = "normal"

        return recent_vol, regime

    def detect_regime(self, data: pd.DataFrame) -> RegimeState:
        """Detect current market regime.
        
        Combines trend, mean-reversion, and volatility signals.
        """
        if len(data) < max(self.trend_lookback, self.hurst_lookback):
            return RegimeState(
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                volatility=0.0,
                trend_strength=0.0,
                mean_reversion_score=0.5,
                hurst_exponent=0.5,
            )

        # Calculate components
        hurst = self.calculate_hurst_exponent(data["close"])
        trend_strength, trend_direction = self.calculate_trend_strength(data)
        mr_score = self.calculate_mean_reversion_score(data)
        volatility, vol_regime = self.calculate_volatility_regime(data)

        # Determine primary regime
        # High volatility regime takes precedence
        if vol_regime == "high" and volatility > self.high_vol_threshold:
            regime = RegimeType.HIGH_VOLATILITY
            # Normalize confidence based on how far above threshold
            confidence = min(1.0, volatility / (self.high_vol_threshold * 1.67))
        elif vol_regime == "low" and volatility < self.low_vol_threshold:
            regime = RegimeType.LOW_VOLATILITY
            confidence = 1.0 - min(1.0, volatility / self.low_vol_threshold)
        # Trend regime
        elif hurst > 0.55 and trend_strength > self.trend_threshold:
            if trend_direction == "up":
                regime = RegimeType.TREND_UP
            else:
                regime = RegimeType.TREND_DOWN
            confidence = min(1.0, (hurst - 0.5) * 2 + trend_strength)
        # Mean reversion regime
        elif hurst < 0.45 and mr_score > 0.55:
            regime = RegimeType.MEAN_REVERT
            confidence = min(1.0, (0.5 - hurst) * 2 + (mr_score - 0.5) * 2)
        else:
            # Mixed/uncertain
            regime = RegimeType.UNKNOWN
            confidence = 0.3

        logger.debug(
            f"Regime: {regime.value} (conf={confidence:.2f}, "
            f"hurst={hurst:.3f}, trend={trend_strength:.2f}, mr={mr_score:.2f})"
        )

        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            mean_reversion_score=mr_score,
            hurst_exponent=hurst,
            metadata={
                "trend_direction": trend_direction,
                "volatility_regime": vol_regime,
            },
        )

    def get_strategy_weights(
        self, regime_state: RegimeState
    ) -> dict[str, float]:
        """Get recommended strategy weights based on regime.
        
        Returns dict mapping strategy names to allocation weights (0-1).
        Includes momentum strategy in weight calculations.
        """
        regime = regime_state.regime
        confidence = regime_state.confidence

        # Default equal weights (including momentum)
        weights = {
            "trend": 0.20,
            "mean_reversion": 0.20,
            "breakout": 0.20,
            "market_making": 0.20,
            "momentum": 0.20,
        }

        if confidence < 0.3:
            return weights  # Low confidence, stay diversified

        if regime == RegimeType.TREND_UP:
            weights = {
                "trend": 0.35 * confidence,
                "momentum": 0.30 * confidence,
                "breakout": 0.20 * confidence,
                "mean_reversion": 0.10 * (1 - confidence),
                "market_making": 0.05,
            }
        elif regime == RegimeType.TREND_DOWN:
            weights = {
                "trend": 0.35 * confidence,
                "momentum": 0.25 * confidence,
                "breakout": 0.20 * confidence,
                "mean_reversion": 0.15 * (1 - confidence),
                "market_making": 0.05,
            }
        elif regime == RegimeType.MEAN_REVERT:
            weights = {
                "mean_reversion": 0.45 * confidence,
                "market_making": 0.25 * confidence,
                "momentum": 0.10,
                "trend": 0.10 * (1 - confidence),
                "breakout": 0.10 * (1 - confidence),
            }
        elif regime == RegimeType.HIGH_VOLATILITY:
            weights = {
                "breakout": 0.30 * confidence,
                "momentum": 0.30 * confidence,
                "trend": 0.20 * confidence,
                "mean_reversion": 0.15,
                "market_making": 0.05 * (1 - confidence),
            }
        elif regime == RegimeType.LOW_VOLATILITY:
            weights = {
                "market_making": 0.40 * confidence,
                "mean_reversion": 0.25 * confidence,
                "momentum": 0.15,
                "trend": 0.10,
                "breakout": 0.10 * (1 - confidence),
            }

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights
