"""Risk management module with ATR/VaR, drawdown controls, and margin checks."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio."""

    atr: float
    var_95: float
    var_99: float
    volatility: float
    max_position_size: float
    suggested_leverage: int
    stop_loss_distance: float
    margin_required: float


@dataclass
class PositionSizing:
    """Position sizing result."""

    size: float
    leverage: int
    stop_loss: float
    take_profit: float
    margin_required: float
    risk_amount: float
    risk_pct: float


class RiskManager:
    """Comprehensive risk management for trading."""

    def __init__(self, config: dict[str, Any]):
        self.max_position_pct = config.get("max_position_pct", 5.0) / 100
        self.max_drawdown_pct = config.get("max_drawdown_pct", 10.0) / 100
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.var_confidence = config.get("var_confidence", 0.95)
        self.margin_buffer = config.get("margin_buffer", 0.2)
        self.max_leverage = config.get("max_leverage", 5)
        self.daily_loss_limit_pct = config.get("daily_loss_limit_pct", 3.0) / 100
        
        # Track portfolio state
        self.portfolio_value = 0.0
        self.peak_value = 0.0
        self.daily_pnl = 0.0
        self.positions: dict[str, float] = {}

    def update_portfolio(
        self, value: float, daily_pnl: float = 0.0
    ) -> None:
        """Update portfolio state."""
        self.portfolio_value = value
        self.peak_value = max(self.peak_value, value)
        self.daily_pnl = daily_pnl

    def calculate_atr(
        self, data: pd.DataFrame, period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1] if len(atr) > 0 else 0.0

    def calculate_var(
        self,
        data: pd.DataFrame,
        confidence: float = 0.95,
        holding_period: int = 1,
    ) -> tuple[float, float]:
        """Calculate Value at Risk (parametric method)."""
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        
        if len(returns) < 20:
            return 0.0, 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        # Z-scores for confidence levels
        z_95 = 1.645
        z_99 = 2.326

        # Scale for holding period
        sqrt_period = np.sqrt(holding_period)

        var_95 = -(mean_return - z_95 * std_return) * sqrt_period
        var_99 = -(mean_return - z_99 * std_return) * sqrt_period

        return var_95, var_99

    def calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate historical volatility."""
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        return returns.tail(period).std() if len(returns) >= period else 0.0

    def calculate_risk_metrics(
        self, data: pd.DataFrame, current_price: float
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        atr = self.calculate_atr(data)
        var_95, var_99 = self.calculate_var(data)
        volatility = self.calculate_volatility(data)

        # Max position based on volatility
        if volatility > 0:
            vol_adjusted_max = self.max_position_pct / (volatility * 10)
            max_position = min(self.max_position_pct, vol_adjusted_max)
        else:
            max_position = self.max_position_pct

        max_position_size = self.portfolio_value * max_position

        # Suggested leverage based on volatility
        if volatility > 0:
            vol_leverage = min(
                self.max_leverage, int(0.1 / volatility)
            )  # Target 10% effective volatility
        else:
            vol_leverage = 1
        suggested_leverage = max(1, vol_leverage)

        # Stop loss distance in price terms
        stop_loss_distance = atr * self.atr_multiplier

        # Margin required for max position
        margin_required = max_position_size / suggested_leverage

        return RiskMetrics(
            atr=atr,
            var_95=var_95,
            var_99=var_99,
            volatility=volatility,
            max_position_size=max_position_size,
            suggested_leverage=suggested_leverage,
            stop_loss_distance=stop_loss_distance,
            margin_required=margin_required,
        )

    def calculate_position_size(
        self,
        signal: Signal,
        data: pd.DataFrame,
        available_margin: float,
    ) -> PositionSizing | None:
        """Calculate optimal position size based on risk parameters."""
        if self.portfolio_value <= 0:
            logger.warning("Portfolio value is zero, cannot size position")
            return None

        metrics = self.calculate_risk_metrics(data, signal.price)

        # Check daily loss limit
        if abs(self.daily_pnl) >= self.portfolio_value * self.daily_loss_limit_pct:
            logger.warning("Daily loss limit reached, no new positions")
            return None

        # Check max drawdown
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if current_drawdown >= self.max_drawdown_pct:
            logger.warning(
                f"Max drawdown reached ({current_drawdown:.1%}), reducing exposure"
            )
            return None

        # Base position on signal strength and risk
        risk_amount = self.portfolio_value * self.max_position_pct * signal.strength

        # Determine leverage
        leverage = min(
            signal.suggested_leverage,
            metrics.suggested_leverage,
            self.max_leverage,
        )

        # Calculate position size
        if signal.stop_loss and signal.price > 0:
            # Size based on stop loss distance
            stop_distance_pct = abs(signal.stop_loss - signal.price) / signal.price
            if stop_distance_pct > 0:
                size_by_risk = risk_amount / (signal.price * stop_distance_pct)
            else:
                size_by_risk = risk_amount / signal.price
        else:
            # Size based on ATR
            atr_distance_pct = metrics.atr / signal.price if signal.price > 0 else 0.1
            if atr_distance_pct > 0:
                size_by_risk = risk_amount / (signal.price * atr_distance_pct * 2)
            else:
                size_by_risk = risk_amount / signal.price

        # Apply leverage
        size_with_leverage = size_by_risk * leverage

        # Check margin requirements
        margin_required = (size_with_leverage * signal.price) / leverage
        margin_with_buffer = margin_required * (1 + self.margin_buffer)

        if margin_with_buffer > available_margin:
            # Reduce size to fit available margin
            reduction_factor = available_margin / margin_with_buffer
            size_with_leverage *= reduction_factor
            margin_required = (size_with_leverage * signal.price) / leverage
            logger.info(
                f"Position reduced by {(1-reduction_factor)*100:.1f}% due to margin"
            )

        # Calculate stop loss and take profit
        if signal.stop_loss:
            stop_loss = signal.stop_loss
        else:
            if signal.type == SignalType.LONG:
                stop_loss = signal.price - (metrics.atr * self.atr_multiplier)
            else:
                stop_loss = signal.price + (metrics.atr * self.atr_multiplier)

        if signal.take_profit:
            take_profit = signal.take_profit
        else:
            if signal.type == SignalType.LONG:
                take_profit = signal.price + (metrics.atr * self.atr_multiplier * 2)
            else:
                take_profit = signal.price - (metrics.atr * self.atr_multiplier * 2)

        # Final risk calculation
        risk_per_unit = abs(signal.price - stop_loss)
        total_risk = risk_per_unit * size_with_leverage
        risk_pct = total_risk / self.portfolio_value

        return PositionSizing(
            size=size_with_leverage,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin_required=margin_required,
            risk_amount=total_risk,
            risk_pct=risk_pct,
        )

    def validate_order(
        self,
        symbol: str,
        size: float,
        price: float,
        leverage: int,
        available_margin: float,
        is_reduce_only: bool = False,
    ) -> tuple[bool, str]:
        """Validate an order against risk limits."""
        # Skip most checks for reduce-only orders
        if is_reduce_only:
            return True, "Reduce-only order approved"

        # Check position size limit
        position_value = size * price
        max_allowed = self.portfolio_value * self.max_position_pct * leverage

        if position_value > max_allowed:
            return False, f"Position size exceeds limit: {position_value:.2f} > {max_allowed:.2f}"

        # Check margin requirements
        margin_required = position_value / leverage
        margin_with_buffer = margin_required * (1 + self.margin_buffer)

        if margin_with_buffer > available_margin:
            return False, f"Insufficient margin: need {margin_with_buffer:.2f}, have {available_margin:.2f}"

        # Check leverage limit
        if leverage > self.max_leverage:
            return False, f"Leverage exceeds limit: {leverage} > {self.max_leverage}"

        # Check total exposure
        total_exposure = sum(abs(v) for v in self.positions.values()) + position_value
        max_exposure = self.portfolio_value * 3  # 300% max exposure

        if total_exposure > max_exposure:
            return False, f"Total exposure exceeds limit: {total_exposure:.2f} > {max_exposure:.2f}"

        return True, "Order validated"

    def select_leverage(
        self,
        signal: Signal,
        volatility: float,
    ) -> int:
        """Select optimal leverage based on signal and volatility."""
        # Base leverage from signal
        base_leverage = signal.suggested_leverage

        # Adjust for volatility (lower leverage for higher volatility)
        if volatility > 0:
            vol_adjusted = int(0.05 / volatility)  # Target 5% effective volatility
        else:
            vol_adjusted = self.max_leverage

        # Adjust for signal strength
        strength_adjusted = int(base_leverage * signal.strength)

        # Take minimum of all adjustments
        final_leverage = min(
            max(1, strength_adjusted),
            max(1, vol_adjusted),
            self.max_leverage,
        )

        return final_leverage

    def check_margin_call_risk(
        self,
        position_value: float,
        margin: float,
        unrealized_pnl: float,
        maintenance_margin_rate: float = 0.01,
    ) -> tuple[bool, float]:
        """Check margin call risk for a position."""
        effective_margin = margin + unrealized_pnl
        maintenance_margin = position_value * maintenance_margin_rate

        if effective_margin < maintenance_margin * 1.5:
            # At risk - calculate distance to margin call
            margin_ratio = effective_margin / maintenance_margin
            return True, margin_ratio

        return False, effective_margin / maintenance_margin
