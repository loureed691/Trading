"""Market making and arbitrage strategy."""

from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class MarketMakingStrategy(BaseStrategy):
    """Market making strategy with inventory management."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("market_making", config)
        self.spread_multiplier = config.get("spread_multiplier", 1.5)
        self.inventory_target = config.get("inventory_target", 0.5)
        self.max_inventory_pct = config.get("max_inventory_pct", 30) / 100
        self.min_spread = config.get("min_spread", 0.001)
        self.skew_factor = config.get("skew_factor", 0.5)
        
        # Track inventory
        self.current_inventory = 0.0
        self.max_inventory = 0.0

    def set_inventory(self, current: float, max_size: float) -> None:
        """Update current inventory levels."""
        self.current_inventory = current
        self.max_inventory = max_size

    def _calculate_inventory_skew(self) -> float:
        """Calculate price skew based on inventory."""
        if self.max_inventory <= 0:
            return 0.0
        
        inventory_ratio = self.current_inventory / self.max_inventory
        deviation = inventory_ratio - self.inventory_target
        
        # Positive deviation = too much inventory, skew to sell
        # Negative deviation = too little inventory, skew to buy
        return deviation * self.skew_factor

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate market making signals."""
        if len(data) < 20:
            return None

        # Calculate indicators
        atr = self.calculate_atr(data)
        volatility = self.calculate_volatility(data)

        current_price = data["close"].iloc[-1]
        current_atr = atr.iloc[-1]
        
        # Calculate optimal spread based on volatility
        vol_spread = volatility * 2  # Base spread on volatility
        min_spread = max(self.min_spread, vol_spread)
        optimal_spread = min_spread * self.spread_multiplier

        # Calculate inventory skew
        skew = self._calculate_inventory_skew()

        # Calculate bid and ask prices
        half_spread = optimal_spread / 2
        bid_price = current_price * (1 - half_spread - skew)
        ask_price = current_price * (1 + half_spread - skew)

        # Determine action based on inventory
        inventory_ratio = (
            self.current_inventory / self.max_inventory
            if self.max_inventory > 0
            else 0.5
        )

        # Generate signals based on inventory state
        if inventory_ratio < self.inventory_target - 0.1:
            # Need more inventory - bias towards buying
            strength = min(1.0, (self.inventory_target - inventory_ratio) * 2)
            return Signal(
                type=SignalType.LONG,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=bid_price,
                stop_loss=bid_price * 0.99,  # Tight stops for MM
                take_profit=ask_price,
                suggested_leverage=1,  # MM typically uses low leverage
                metadata={
                    "bid": bid_price,
                    "ask": ask_price,
                    "spread": optimal_spread,
                    "inventory_ratio": inventory_ratio,
                    "skew": skew,
                },
            )

        elif inventory_ratio > self.inventory_target + 0.1:
            # Too much inventory - bias towards selling
            strength = min(1.0, (inventory_ratio - self.inventory_target) * 2)
            return Signal(
                type=SignalType.SHORT,
                symbol=data.attrs.get("symbol", ""),
                strength=strength,
                price=ask_price,
                stop_loss=ask_price * 1.01,
                take_profit=bid_price,
                suggested_leverage=1,
                metadata={
                    "bid": bid_price,
                    "ask": ask_price,
                    "spread": optimal_spread,
                    "inventory_ratio": inventory_ratio,
                    "skew": skew,
                },
            )

        # Balanced inventory - quote both sides
        return Signal(
            type=SignalType.HOLD,
            symbol=data.attrs.get("symbol", ""),
            strength=0.5,
            price=current_price,
            metadata={
                "bid": bid_price,
                "ask": ask_price,
                "spread": optimal_spread,
                "inventory_ratio": inventory_ratio,
                "skew": skew,
                "action": "quote_both",
            },
        )


class ArbitrageStrategy(BaseStrategy):
    """Simple arbitrage detection strategy."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("arbitrage", config)
        self.min_spread_pct = config.get("min_spread_pct", 0.1) / 100
        self.fee_buffer = config.get("fee_buffer", 0.001)  # Account for fees

    def detect_opportunity(
        self,
        spot_price: float,
        futures_price: float,
        funding_rate: float = 0.0,
    ) -> dict[str, Any] | None:
        """Detect spot-futures arbitrage opportunity."""
        if spot_price <= 0 or futures_price <= 0:
            return None

        # Calculate basis
        basis = (futures_price - spot_price) / spot_price
        
        # Account for costs
        effective_basis = abs(basis) - self.fee_buffer * 2  # Round trip fees
        
        if effective_basis > self.min_spread_pct:
            if basis > 0:
                # Futures premium - buy spot, sell futures
                return {
                    "type": "cash_and_carry",
                    "spot_action": "buy",
                    "futures_action": "sell",
                    "basis": basis,
                    "expected_profit": effective_basis,
                    "funding_rate": funding_rate,
                }
            else:
                # Futures discount - sell spot, buy futures
                return {
                    "type": "reverse_cash_and_carry",
                    "spot_action": "sell",
                    "futures_action": "buy",
                    "basis": basis,
                    "expected_profit": effective_basis,
                    "funding_rate": funding_rate,
                }

        return None

    def generate_signal(self, data: pd.DataFrame) -> Signal | None:
        """Generate signal (placeholder for interface compatibility)."""
        # Arbitrage requires multi-market data, handled separately
        return None
