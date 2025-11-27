"""Risk management module with ATR/VaR/ES, drawdown controls, and margin checks."""

import logging
from dataclasses import dataclass, field
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
    es_95: float  # Expected Shortfall
    es_99: float
    volatility: float
    max_position_size: float
    suggested_leverage: int
    stop_loss_distance: float
    margin_required: float
    liquidation_distance: float = 0.0
    funding_impact: float = 0.0


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


@dataclass
class DynamicRiskParams:
    """Dynamically adjusted risk parameters."""

    max_position_pct: float
    max_drawdown_pct: float
    max_leverage: int
    adjustments: list[str] = field(default_factory=list)


@dataclass
class HedgeRecommendation:
    """Hedging recommendation."""

    should_hedge: bool
    hedge_ratio: float
    hedge_instrument: str
    reason: str


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
        es_95, es_99 = self.calculate_expected_shortfall(data)
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

        # Liquidation distance
        liquidation_distance = self.calculate_liquidation_distance(
            current_price, suggested_leverage, "long"
        )

        return RiskMetrics(
            atr=atr,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            volatility=volatility,
            max_position_size=max_position_size,
            suggested_leverage=suggested_leverage,
            stop_loss_distance=stop_loss_distance,
            margin_required=margin_required,
            liquidation_distance=liquidation_distance,
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

    def calculate_expected_shortfall(
        self,
        data: pd.DataFrame,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate Expected Shortfall (Conditional VaR).
        
        ES is the expected loss given that loss exceeds VaR.
        """
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        
        if len(returns) < 20:
            return 0.0, 0.0
        
        sorted_returns = np.sort(returns)
        
        # ES at 95% and 99% confidence
        cutoff_95 = int(len(sorted_returns) * (1 - 0.95))
        cutoff_99 = int(len(sorted_returns) * (1 - 0.99))
        
        es_95 = -np.mean(sorted_returns[:max(1, cutoff_95)])
        es_99 = -np.mean(sorted_returns[:max(1, cutoff_99)])
        
        return es_95, es_99

    def calculate_liquidation_distance(
        self,
        entry_price: float,
        leverage: int,
        side: str,
        maintenance_margin_rate: float = 0.01,
    ) -> float:
        """Calculate distance to liquidation price as percentage.
        
        Returns the price move (%) that would trigger liquidation.
        """
        if leverage <= 0 or entry_price <= 0:
            return 0.0
        
        # Simplified liquidation calculation
        # Liquidation occurs when loss > (initial margin - maintenance margin)
        initial_margin_rate = 1 / leverage
        available_for_loss = initial_margin_rate - maintenance_margin_rate
        
        # Distance is the percentage move that exhausts margin
        if side.lower() == "long":
            liquidation_distance = available_for_loss
        else:
            liquidation_distance = available_for_loss
        
        return liquidation_distance * 100  # Return as percentage

    def adjust_leverage_for_funding(
        self,
        base_leverage: int,
        funding_rate: float,
        holding_period_hours: int = 8,
    ) -> int:
        """Adjust leverage based on funding rate impact.
        
        High funding rates erode profits, so reduce leverage.
        """
        # Annualized funding impact (funding paid 3x per day)
        annual_funding_impact = abs(funding_rate) * 3 * 365
        
        # Reduce leverage if funding is high
        if annual_funding_impact > 0.5:  # 50% annual impact
            reduction_factor = 0.5
        elif annual_funding_impact > 0.3:  # 30% annual impact
            reduction_factor = 0.7
        elif annual_funding_impact > 0.1:  # 10% annual impact
            reduction_factor = 0.9
        else:
            reduction_factor = 1.0
        
        adjusted_leverage = max(1, int(base_leverage * reduction_factor))
        
        if adjusted_leverage < base_leverage:
            logger.info(
                f"Leverage reduced from {base_leverage}x to {adjusted_leverage}x "
                f"due to funding rate ({funding_rate:.4%})"
            )
        
        return adjusted_leverage

    def calculate_dynamic_risk_params(
        self,
        data: pd.DataFrame,
        regime_volatility: str = "normal",
        current_drawdown: float = 0.0,
    ) -> DynamicRiskParams:
        """Auto-adjust risk parameters based on market conditions.
        
        Returns dynamically adjusted max_position_pct, max_drawdown_pct, max_leverage.
        """
        adjustments = []
        
        # Start with configured values
        position_pct = self.max_position_pct * 100
        drawdown_pct = self.max_drawdown_pct * 100
        leverage = self.max_leverage
        
        # Calculate recent volatility
        volatility = self.calculate_volatility(data)
        
        # Adjust for volatility regime
        if regime_volatility == "high" or volatility > 0.05:
            position_pct *= 0.5
            leverage = max(1, leverage // 2)
            adjustments.append("high_volatility_reduction")
        elif regime_volatility == "low" or volatility < 0.01:
            position_pct *= 1.2
            adjustments.append("low_volatility_increase")
        
        # Adjust for current drawdown
        if current_drawdown > 0.05:
            # Reduce risk as drawdown increases
            risk_reduction = 1 - min(0.5, current_drawdown)
            position_pct *= risk_reduction
            leverage = max(1, int(leverage * risk_reduction))
            adjustments.append(f"drawdown_reduction_{current_drawdown:.1%}")
        
        # Calculate VaR-based adjustment
        var_95, var_99 = self.calculate_var(data)
        if var_99 > 0.05:  # 5% daily VaR at 99%
            position_pct *= 0.7
            leverage = max(1, leverage - 1)
            adjustments.append("high_var_reduction")
        
        # ES-based adjustment
        es_95, es_99 = self.calculate_expected_shortfall(data)
        if es_99 > 0.08:  # 8% expected shortfall
            position_pct *= 0.8
            adjustments.append("high_es_reduction")
        
        # ATR-based adjustment
        atr = self.calculate_atr(data)
        current_price = data["close"].iloc[-1]
        atr_pct = atr / current_price if current_price > 0 else 0
        
        if atr_pct > 0.03:  # 3% ATR
            position_pct *= 0.8
            adjustments.append("high_atr_reduction")
        
        # Cap at original limits
        final_position = min(position_pct / 100, self.max_position_pct)
        final_drawdown = min(drawdown_pct / 100, self.max_drawdown_pct)
        final_leverage = min(leverage, self.max_leverage)
        
        logger.debug(
            f"Dynamic risk params: pos={final_position:.1%}, "
            f"dd={final_drawdown:.1%}, lev={final_leverage}x, "
            f"adjustments={adjustments}"
        )
        
        return DynamicRiskParams(
            max_position_pct=final_position,
            max_drawdown_pct=final_drawdown,
            max_leverage=final_leverage,
            adjustments=adjustments,
        )

    def calculate_hedge_recommendation(
        self,
        position_value: float,
        position_side: str,
        unrealized_pnl: float,
        volatility: float,
        correlation_with_btc: float = 0.8,
    ) -> HedgeRecommendation:
        """Calculate dynamic hedging recommendation.
        
        Recommends hedging based on exposure and market conditions.
        """
        should_hedge = False
        hedge_ratio = 0.0
        hedge_instrument = "BTC-USDT-PERP"  # Default hedge instrument
        reason = ""
        
        # Check if position is significant
        if self.portfolio_value <= 0:
            return HedgeRecommendation(False, 0.0, "", "No portfolio value")
        
        exposure_pct = position_value / self.portfolio_value
        
        # Large exposure check
        if exposure_pct > 0.3:  # 30% of portfolio
            should_hedge = True
            hedge_ratio = 0.3  # Hedge 30% of position
            reason = f"Large exposure ({exposure_pct:.1%} of portfolio)"
        
        # High volatility check
        if volatility > 0.05:  # 5% daily volatility
            should_hedge = True
            hedge_ratio = max(hedge_ratio, volatility / 0.05 * 0.2)  # Scale hedge ratio
            reason = f"High volatility ({volatility:.1%})"
        
        # Significant unrealized profit protection
        profit_pct = unrealized_pnl / position_value if position_value > 0 else 0
        if profit_pct > 0.1:  # 10% unrealized profit
            should_hedge = True
            hedge_ratio = max(hedge_ratio, profit_pct * 0.5)  # Hedge 50% of profit
            reason = f"Profit protection ({profit_pct:.1%} unrealized)"
        
        # Adjust hedge instrument based on correlation
        if correlation_with_btc < 0.5:
            hedge_instrument = "ETH-USDT-PERP"  # Use ETH if BTC correlation is low
        
        # Cap hedge ratio
        hedge_ratio = min(0.5, hedge_ratio)
        
        return HedgeRecommendation(
            should_hedge=should_hedge,
            hedge_ratio=hedge_ratio,
            hedge_instrument=hedge_instrument,
            reason=reason,
        )

    def check_portfolio_constraints(
        self,
        proposed_positions: dict[str, float],
        market_exposures: dict[str, float] | None = None,
    ) -> tuple[bool, list[str]]:
        """Check portfolio-level constraints.
        
        Returns (is_valid, list of constraint violations).
        """
        violations = []
        
        if self.portfolio_value <= 0:
            return False, ["No portfolio value"]
        
        # Total exposure check
        total_exposure = sum(abs(v) for v in proposed_positions.values())
        max_total_exposure = self.portfolio_value * 3  # 300%
        
        if total_exposure > max_total_exposure:
            violations.append(
                f"Total exposure {total_exposure:.0f} > max {max_total_exposure:.0f}"
            )
        
        # Single position concentration check
        for symbol, value in proposed_positions.items():
            concentration = abs(value) / self.portfolio_value
            if concentration > 0.2:  # 20% max per position
                violations.append(
                    f"Position {symbol} concentration {concentration:.1%} > 20%"
                )
        
        # Market exposure check (if provided)
        if market_exposures:
            for market, exposure in market_exposures.items():
                market_pct = abs(exposure) / self.portfolio_value
                if market_pct > 0.5:  # 50% max per market
                    violations.append(
                        f"Market {market} exposure {market_pct:.1%} > 50%"
                    )
        
        # Long/short balance check
        long_exposure = sum(v for v in proposed_positions.values() if v > 0)
        short_exposure = abs(sum(v for v in proposed_positions.values() if v < 0))
        
        net_exposure = (long_exposure - short_exposure) / self.portfolio_value
        if abs(net_exposure) > 1.5:  # 150% net exposure
            violations.append(f"Net exposure {net_exposure:.1%} > 150%")
        
        return len(violations) == 0, violations
