"""Tests for risk management module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kucoin_bot.risk.risk_manager import RiskManager, RiskMetrics, PositionSizing
from kucoin_bot.strategies.base import Signal, SignalType


def generate_test_data(n_bars: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_bars - i) for i in range(n_bars)]
    base = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": base * (1 + np.random.uniform(-0.01, 0.01, n_bars)),
        "high": base * (1 + np.abs(np.random.normal(0, 0.01, n_bars))),
        "low": base * (1 - np.abs(np.random.normal(0, 0.01, n_bars))),
        "close": base,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })
    
    return data


class TestRiskManager:
    """Test cases for RiskManager."""

    def test_risk_manager_initialization(self) -> None:
        """Test RiskManager initialization."""
        config = {
            "max_position_pct": 10.0,
            "max_leverage": 3,
            "max_drawdown_pct": 15.0,
        }
        rm = RiskManager(config)
        
        assert rm.max_position_pct == 0.10
        assert rm.max_leverage == 3
        assert rm.max_drawdown_pct == 0.15

    def test_update_portfolio(self) -> None:
        """Test portfolio state update."""
        rm = RiskManager({})
        
        rm.update_portfolio(10000, 0)
        assert rm.portfolio_value == 10000
        assert rm.peak_value == 10000
        
        rm.update_portfolio(12000, 200)
        assert rm.portfolio_value == 12000
        assert rm.peak_value == 12000
        assert rm.daily_pnl == 200
        
        rm.update_portfolio(11000, -100)
        assert rm.portfolio_value == 11000
        assert rm.peak_value == 12000  # Peak should stay at max

    def test_calculate_atr(self) -> None:
        """Test ATR calculation."""
        rm = RiskManager({})
        data = generate_test_data()
        
        atr = rm.calculate_atr(data)
        
        assert atr > 0

    def test_calculate_var(self) -> None:
        """Test VaR calculation."""
        rm = RiskManager({})
        data = generate_test_data()
        
        var_95, var_99 = rm.calculate_var(data)
        
        assert var_99 >= var_95  # 99% VaR should be more extreme
        assert var_95 >= 0

    def test_calculate_volatility(self) -> None:
        """Test volatility calculation."""
        rm = RiskManager({})
        data = generate_test_data()
        
        volatility = rm.calculate_volatility(data)
        
        assert volatility >= 0

    def test_calculate_risk_metrics(self) -> None:
        """Test comprehensive risk metrics calculation."""
        rm = RiskManager({"max_position_pct": 5.0})
        rm.update_portfolio(10000)
        
        data = generate_test_data()
        current_price = data["close"].iloc[-1]
        
        metrics = rm.calculate_risk_metrics(data, current_price)
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.atr > 0
        assert metrics.max_position_size > 0
        assert metrics.suggested_leverage >= 1

    def test_calculate_position_size(self) -> None:
        """Test position sizing."""
        rm = RiskManager({
            "max_position_pct": 5.0,
            "max_leverage": 3,
        })
        rm.update_portfolio(10000)
        
        data = generate_test_data()
        current_price = data["close"].iloc[-1]
        
        signal = Signal(
            type=SignalType.LONG,
            symbol="TEST-USDT",
            strength=0.7,
            price=current_price,
            stop_loss=current_price * 0.98,
            suggested_leverage=2,
        )
        
        sizing = rm.calculate_position_size(signal, data, 5000)
        
        assert sizing is not None
        assert isinstance(sizing, PositionSizing)
        assert sizing.size > 0
        assert sizing.leverage >= 1
        assert sizing.leverage <= 3
        assert sizing.risk_pct <= 0.05  # Max position is 5%

    def test_position_size_respects_margin(self) -> None:
        """Test position sizing respects available margin."""
        rm = RiskManager({"max_position_pct": 5.0})
        rm.update_portfolio(10000)
        
        data = generate_test_data()
        current_price = data["close"].iloc[-1]
        
        signal = Signal(
            type=SignalType.LONG,
            symbol="TEST-USDT",
            strength=1.0,
            price=current_price,
            suggested_leverage=1,
        )
        
        # Very limited margin
        sizing = rm.calculate_position_size(signal, data, 100)
        
        if sizing:
            assert sizing.margin_required <= 100 * (1 + rm.margin_buffer)

    def test_position_size_blocked_by_drawdown(self) -> None:
        """Test position sizing blocked when max drawdown reached."""
        rm = RiskManager({"max_drawdown_pct": 10.0})
        rm.update_portfolio(10000)
        rm.peak_value = 12000  # Simulate previous peak
        
        # Current drawdown is (12000-10000)/12000 = 16.7% > 10%
        data = generate_test_data()
        signal = Signal(
            type=SignalType.LONG,
            symbol="TEST-USDT",
            strength=0.8,
            price=100,
        )
        
        sizing = rm.calculate_position_size(signal, data, 5000)
        
        assert sizing is None

    def test_validate_order_success(self) -> None:
        """Test order validation success."""
        rm = RiskManager({"max_position_pct": 10.0, "max_leverage": 5})
        rm.update_portfolio(10000)
        
        is_valid, reason = rm.validate_order(
            symbol="TEST-USDT",
            size=10,
            price=100,
            leverage=2,
            available_margin=2000,
        )
        
        assert is_valid is True

    def test_validate_order_exceeds_position_limit(self) -> None:
        """Test order validation fails on position limit."""
        rm = RiskManager({"max_position_pct": 5.0, "max_leverage": 1})
        rm.update_portfolio(10000)
        
        # 100 * 100 = 10000 > 10000 * 5% * 1 = 500
        is_valid, reason = rm.validate_order(
            symbol="TEST-USDT",
            size=100,
            price=100,
            leverage=1,
            available_margin=2000,
        )
        
        assert is_valid is False
        assert "exceeds limit" in reason.lower()

    def test_validate_order_insufficient_margin(self) -> None:
        """Test order validation fails on insufficient margin."""
        rm = RiskManager({"max_position_pct": 100.0})  # Allow large positions
        rm.update_portfolio(10000)

        # 10 * 100 = 1000 position value, needs 1000 * 1.2 = 1200 margin but only have 50
        is_valid, reason = rm.validate_order(
            symbol="TEST-USDT",
            size=10,
            price=100,
            leverage=1,
            available_margin=50,  # Not enough
        )

        assert is_valid is False
        assert "margin" in reason.lower()

    def test_validate_order_leverage_limit(self) -> None:
        """Test order validation fails on leverage limit."""
        rm = RiskManager({"max_leverage": 3})
        rm.update_portfolio(10000)
        
        is_valid, reason = rm.validate_order(
            symbol="TEST-USDT",
            size=10,
            price=100,
            leverage=5,  # Exceeds max
            available_margin=5000,
        )
        
        assert is_valid is False
        assert "leverage" in reason.lower()

    def test_select_leverage(self) -> None:
        """Test leverage selection."""
        rm = RiskManager({"max_leverage": 5})
        
        signal = Signal(
            type=SignalType.LONG,
            symbol="TEST-USDT",
            strength=0.8,
            price=100,
            suggested_leverage=3,
        )
        
        # Low volatility
        leverage = rm.select_leverage(signal, 0.01)
        assert 1 <= leverage <= 5
        
        # High volatility should reduce leverage
        leverage_high_vol = rm.select_leverage(signal, 0.1)
        assert leverage_high_vol <= leverage

    def test_check_margin_call_risk(self) -> None:
        """Test margin call risk detection."""
        rm = RiskManager({})
        
        # Safe position
        at_risk, ratio = rm.check_margin_call_risk(
            position_value=10000,
            margin=1000,
            unrealized_pnl=0,
            maintenance_margin_rate=0.01,
        )
        assert at_risk is False
        assert ratio > 1.5
        
        # At-risk position
        at_risk, ratio = rm.check_margin_call_risk(
            position_value=10000,
            margin=150,
            unrealized_pnl=-50,
            maintenance_margin_rate=0.01,
        )
        assert at_risk is True


class TestPortfolioRiskManagement:
    """Test cases for portfolio-level risk management."""

    def test_check_portfolio_limits_no_initial_value(self) -> None:
        """Test portfolio limits check with no initial value."""
        rm = RiskManager({})
        rm.update_portfolio(0)
        
        should_continue, action, reasons = rm.check_portfolio_limits()
        assert should_continue is True
        assert action == "continue"

    def test_check_portfolio_limits_normal(self) -> None:
        """Test portfolio limits under normal conditions."""
        rm = RiskManager({"portfolio_stop_loss_pct": 15.0})
        rm.update_portfolio(10000)
        rm.update_portfolio(10500)  # 5% gain
        
        should_continue, action, reasons = rm.check_portfolio_limits()
        assert should_continue is True
        assert action == "continue"

    def test_check_portfolio_stop_loss(self) -> None:
        """Test portfolio stop-loss trigger."""
        rm = RiskManager({"portfolio_stop_loss_pct": 15.0})
        rm.update_portfolio(10000)
        rm.initial_value = 10000  # Ensure initial is set
        rm.update_portfolio(8000)  # 20% loss
        
        should_continue, action, reasons = rm.check_portfolio_limits()
        assert should_continue is False
        assert action == "close_all"
        assert any("stop-loss" in r.lower() for r in reasons)

    def test_check_portfolio_take_profit(self) -> None:
        """Test portfolio take-profit trigger."""
        rm = RiskManager({"portfolio_take_profit_pct": 50.0})
        rm.update_portfolio(10000)
        rm.initial_value = 10000
        rm.update_portfolio(16000)  # 60% gain
        
        should_continue, action, reasons = rm.check_portfolio_limits()
        assert should_continue is False
        assert action == "take_profit"
        assert any("take-profit" in r.lower() for r in reasons)

    def test_safe_mode_trigger(self) -> None:
        """Test safe mode activation on high drawdown."""
        rm = RiskManager({"max_drawdown_pct": 20.0})  # Set higher threshold
        rm.update_portfolio(10000)
        rm.initial_value = 10000
        rm.peak_value = 12000  # 16.7% drawdown from peak (above 15% safe mode threshold)
        
        rm.check_portfolio_limits()
        assert rm.safe_mode_enabled is True

    def test_safe_mode_restrictions(self) -> None:
        """Test safe mode applies restrictions."""
        rm = RiskManager({"max_position_pct": 10.0, "max_leverage": 5})
        rm.safe_mode_enabled = True
        
        restrictions = rm.apply_safe_mode()
        assert restrictions["max_position_pct"] < 0.10
        assert restrictions["max_leverage"] < 5
        assert restrictions["new_positions_allowed"] is False

    def test_get_position_adjustments(self) -> None:
        """Test position adjustment calculations."""
        rm = RiskManager({})
        rm.update_portfolio(10000)
        
        positions = {
            "BTC-USDT": {
                "entry_price": 100,
                "size": 1,
                "side": "long",
                "stop_loss": 90,
                "take_profit": 150,
            }
        }
        
        # Test stop-loss trigger
        adjustments = rm.get_position_adjustments(positions, {"BTC-USDT": 85})
        assert len(adjustments) > 0
        assert adjustments[0]["action"] == "close"
        assert "stop-loss" in adjustments[0]["reason"].lower()

    def test_get_position_adjustments_take_profit(self) -> None:
        """Test position adjustment for take-profit."""
        rm = RiskManager({})
        rm.update_portfolio(10000)
        
        positions = {
            "BTC-USDT": {
                "entry_price": 100,
                "size": 1,
                "side": "long",
                "stop_loss": 90,
                "take_profit": 120,
            }
        }
        
        # Test take-profit trigger
        adjustments = rm.get_position_adjustments(positions, {"BTC-USDT": 125})
        assert len(adjustments) > 0
        assert adjustments[0]["action"] == "close"
        assert "take-profit" in adjustments[0]["reason"].lower()
