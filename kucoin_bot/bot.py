"""Main trading bot orchestrator."""

import asyncio
import logging
from datetime import datetime
from typing import Any

import pandas as pd

from .backtest.engine import BacktestEngine, WalkForwardValidator
from .clients.rest_client import KuCoinRestClient, Market, OrderSide, OrderType
from .clients.ws_client import KuCoinWebSocketClient, WsMessage
from .config import Config
from .ml.forecaster import MLForecaster
from .ml.position_sizer import MLPositionSizer
from .pair_selector import PairScore, PairSelector
from .regime.detector import RegimeDetector, RegimeType
from .risk.risk_manager import RiskManager
from .strategies.base import BaseStrategy, Signal, SignalType
from .strategies.breakout import BreakoutStrategy
from .strategies.ensemble import EnsembleStrategy
from .strategies.market_making import ArbitrageStrategy, MarketMakingStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.trend import TrendStrategy
from .utils.logging import AuditLogger, MonitoringMetrics, TradingLogger, setup_logging

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    STRATEGY_MAP = {
        "trend": TrendStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
        "market_making": MarketMakingStrategy,
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        
        # Setup logging
        setup_logging(self.config.get("logging", {}))
        
        self.trading_logger = TradingLogger()
        self.audit_logger = AuditLogger()
        self.metrics = MonitoringMetrics()
        
        # Initialize clients
        self.rest_client = KuCoinRestClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            api_passphrase=self.config.api_passphrase,
            sandbox=self.config.is_sandbox,
        )
        
        self.ws_client = KuCoinWebSocketClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            api_passphrase=self.config.api_passphrase,
            sandbox=self.config.is_sandbox,
        )
        
        # Initialize components
        self.pair_selector = PairSelector(
            self.rest_client,
            self.config.get("pair_selection", {}),
        )
        
        self.risk_manager = RiskManager(self.config.risk_config)
        self.regime_detector = RegimeDetector(self.config.get("regime", {}))
        self.ml_forecaster = MLForecaster(self.config.get("ml", {}))
        self.ml_position_sizer = MLPositionSizer(self.config.get("ml", {}))
        
        # Initialize strategies
        self.strategies: dict[str, BaseStrategy] = {}
        self._init_strategies()
        
        # Initialize ensemble strategy
        self.ensemble_strategy: EnsembleStrategy | None = None
        if self.config.get("strategies.use_ensemble", True):
            self._init_ensemble_strategy()
        
        # State
        self.is_running = False
        self.is_paper = self.config.trading_mode == "paper"
        self.selected_pairs: list[PairScore] = []
        self.market_data: dict[str, pd.DataFrame] = {}
        self.paper_positions: dict[str, dict] = {}
        self.paper_balance = 10000.0
        self.current_regimes: dict[str, RegimeType] = {}

    def _init_strategies(self) -> None:
        """Initialize enabled strategies."""
        strategy_config = self.config.strategy_config
        enabled = strategy_config.get("enabled", ["trend"])
        
        for name in enabled:
            if name in self.STRATEGY_MAP:
                config = strategy_config.get(name, {})
                self.strategies[name] = self.STRATEGY_MAP[name](config)
                logger.info(f"Initialized strategy: {name}")

    def _init_ensemble_strategy(self) -> None:
        """Initialize ensemble strategy with regime detection."""
        if not self.strategies:
            logger.warning("No strategies available for ensemble")
            return
        
        ensemble_config = {
            "regime": self.config.get("regime", {}),
            "min_agreement": self.config.get("strategies.ensemble.min_agreement", 0.5),
            "strength_threshold": self.config.get("strategies.ensemble.strength_threshold", 0.3),
        }
        
        self.ensemble_strategy = EnsembleStrategy(self.strategies, ensemble_config)
        logger.info(f"Initialized ensemble strategy with {len(self.strategies)} sub-strategies")

    async def initialize(self) -> None:
        """Initialize the bot and select trading pairs."""
        logger.info("Initializing trading bot...")
        
        # Get account balance
        try:
            if not self.is_paper:
                balances = await self.rest_client.get_account_balance()
                total_balance = sum(balances.values())
                self.risk_manager.update_portfolio(total_balance)
            else:
                self.risk_manager.update_portfolio(self.paper_balance)
        except Exception as e:
            logger.warning(f"Could not fetch balance: {e}, using paper balance")
            self.risk_manager.update_portfolio(self.paper_balance)
        
        # Select trading pairs
        for market in self.config.markets:
            try:
                market_enum = Market(market)
                pairs = await self.pair_selector.select_pairs(
                    market=market_enum,
                    top_n=5,
                    min_score=0.4,
                )
                self.selected_pairs.extend(pairs)
                logger.info(f"Selected {len(pairs)} pairs for {market}")
            except Exception as e:
                logger.error(f"Failed to select pairs for {market}: {e}")
        
        logger.info(f"Total selected pairs: {len(self.selected_pairs)}")

    async def _fetch_historical_data(
        self,
        symbol: str,
        market: Market = Market.SPOT,
        interval: str = "1hour",
        limit: int = 200,
    ) -> pd.DataFrame | None:
        """Fetch historical kline data."""
        try:
            klines = await self.rest_client.get_klines(
                symbol, interval, market=market
            )
            
            if not klines:
                return None
            
            df = pd.DataFrame(
                klines,
                columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"],
            )
            
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.attrs["symbol"] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    async def _on_ticker_update(self, message: WsMessage) -> None:
        """Handle ticker update from WebSocket."""
        try:
            symbol = message.topic.split(":")[-1]
            data = message.data
            
            logger.debug(f"Ticker update for {symbol}: {data}")
            
            # Update market data and check for signals
            await self._process_market_update(symbol, data)
            
        except Exception as e:
            logger.error(f"Error processing ticker: {e}")

    async def _process_market_update(
        self,
        symbol: str,
        ticker_data: dict,
    ) -> None:
        """Process market update and generate signals."""
        # Get or fetch historical data
        if symbol not in self.market_data:
            data = await self._fetch_historical_data(symbol)
            if data is None:
                return
            self.market_data[symbol] = data
        
        data = self.market_data[symbol]
        
        # Detect current regime
        regime_state = self.regime_detector.detect_regime(data)
        old_regime = self.current_regimes.get(symbol)
        self.current_regimes[symbol] = regime_state.regime
        
        # Log regime change
        if old_regime and old_regime != regime_state.regime:
            self.audit_logger.log_regime_change(
                symbol,
                old_regime.value if old_regime else "unknown",
                regime_state.regime.value,
                regime_state.confidence,
            )
            logger.info(
                f"Regime change for {symbol}: {old_regime.value if old_regime else 'unknown'} -> {regime_state.regime.value}"
            )
        
        # Calculate dynamic risk parameters
        current_drawdown = (
            (self.risk_manager.peak_value - self.risk_manager.portfolio_value)
            / self.risk_manager.peak_value
            if self.risk_manager.peak_value > 0 else 0
        )
        dynamic_params = self.risk_manager.calculate_dynamic_risk_params(
            data,
            regime_volatility=regime_state.metadata.get("volatility_regime", "normal") if regime_state.metadata else "normal",
            current_drawdown=current_drawdown,
        )
        
        # Use ensemble strategy if available, otherwise individual strategies
        if self.ensemble_strategy:
            try:
                signal = self.ensemble_strategy.generate_signal(data)
                if signal and signal.type != SignalType.HOLD:
                    await self._process_signal(symbol, "ensemble", signal, dynamic_params)
            except Exception as e:
                logger.error(f"Ensemble strategy error: {e}")
        else:
            # Run all strategies and collect signals
            signals: list[tuple[str, Signal]] = []
            
            for name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(data)
                    if signal and signal.type != SignalType.HOLD:
                        signals.append((name, signal))
                except Exception as e:
                    logger.error(f"Strategy {name} error: {e}")
            
            # Process signals
            for strategy_name, signal in signals:
                await self._process_signal(symbol, strategy_name, signal, dynamic_params)

    async def _process_signal(
        self,
        symbol: str,
        strategy_name: str,
        signal: Signal,
        dynamic_params: Any = None,
    ) -> None:
        """Process a trading signal."""
        self.trading_logger.log_signal(
            symbol,
            signal.type.value,
            signal.price,
            signal.strength,
            {"strategy": strategy_name},
        )
        
        # Calculate position size
        data = self.market_data.get(symbol)
        if data is None:
            return
        
        available_margin = (
            self.paper_balance if self.is_paper
            else await self._get_available_margin()
        )
        
        # Use ML-based position sizing if forecast is reliable
        forecast_result = self.ml_forecaster.forecast(data)
        ml_sizing = self.ml_position_sizer.get_position_size(
            data=data,
            portfolio_value=self.risk_manager.portfolio_value,
            signal_strength=signal.strength,
            forecast_confidence=forecast_result.cv_score if forecast_result.is_reliable else 0.0,
        )
        
        # Adjust signal strength based on ML forecast direction agreement
        adjusted_strength = signal.strength
        if forecast_result.is_reliable:
            if (
                (forecast_result.direction == "up" and signal.type == SignalType.LONG) or
                (forecast_result.direction == "down" and signal.type == SignalType.SHORT)
            ):
                # ML agrees with signal, increase confidence
                adjusted_strength = min(1.0, signal.strength * 1.2)
                logger.debug(f"ML forecast agrees with signal for {symbol}")
            elif forecast_result.direction != "neutral":
                # ML disagrees, reduce confidence
                adjusted_strength = signal.strength * 0.7
                logger.debug(f"ML forecast disagrees with signal for {symbol}")
        
        # Apply dynamic risk parameters
        original_max_position = self.risk_manager.max_position_pct
        original_max_leverage = self.risk_manager.max_leverage
        
        if dynamic_params:
            self.risk_manager.max_position_pct = dynamic_params.max_position_pct
            self.risk_manager.max_leverage = dynamic_params.max_leverage
        
        try:
            sizing = self.risk_manager.calculate_position_size(
                signal, data, available_margin
            )
        finally:
            # Restore original values
            self.risk_manager.max_position_pct = original_max_position
            self.risk_manager.max_leverage = original_max_leverage
        
        if sizing is None:
            logger.warning(f"Position sizing rejected for {symbol}")
            self.audit_logger.log_risk_event(
                "POSITION_SIZING_REJECTED",
                symbol,
                {"reason": "Risk limits exceeded", "signal_strength": adjusted_strength},
            )
            return
        
        # Adjust size based on ML recommendation
        if ml_sizing.suggested_size_pct < sizing.risk_pct:
            # ML suggests smaller position
            size_reduction = ml_sizing.suggested_size_pct / sizing.risk_pct
            sizing.size *= size_reduction
            logger.debug(f"Position size reduced by ML: {size_reduction:.2f}")
        
        # Validate order
        is_valid, reason = self.risk_manager.validate_order(
            symbol,
            sizing.size,
            signal.price,
            sizing.leverage,
            available_margin,
        )
        
        if not is_valid:
            self.trading_logger.log_risk_event("ORDER_REJECTED", reason)
            self.audit_logger.log_risk_event("ORDER_REJECTED", symbol, {"reason": reason})
            return
        
        # Check portfolio constraints
        proposed_positions = {**self.risk_manager.positions}
        proposed_positions[symbol] = sizing.size * signal.price
        constraints_valid, violations = self.risk_manager.check_portfolio_constraints(
            proposed_positions
        )
        
        if not constraints_valid:
            self.trading_logger.log_risk_event("PORTFOLIO_CONSTRAINT", ", ".join(violations))
            return
        
        # Execute trade
        if self.is_paper:
            await self._execute_paper_trade(symbol, signal, sizing)
        else:
            await self._execute_live_trade(symbol, signal, sizing)

    async def _get_available_margin(self) -> float:
        """Get available margin for trading."""
        try:
            balances = await self.rest_client.get_account_balance()
            return balances.get("USDT", 0.0)
        except Exception as e:
            logger.error(f"Failed to get margin: {e}")
            return 0.0

    async def _execute_paper_trade(
        self,
        symbol: str,
        signal: Signal,
        sizing: Any,
    ) -> None:
        """Execute a paper trade."""
        order_id = f"paper_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        side = "buy" if signal.type == SignalType.LONG else "sell"
        
        self.trading_logger.log_order(
            order_id,
            symbol,
            side,
            sizing.size,
            signal.price,
            "market",
        )
        
        # Audit log
        self.audit_logger.log_order_placed(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=sizing.size,
            price=signal.price,
            order_type="market",
            leverage=sizing.leverage,
            market="paper",
        )
        
        # Record metrics
        self.metrics.record_order_placed()
        
        # Simulate fill
        fees = sizing.size * signal.price * 0.001
        self.paper_balance -= fees
        
        self.paper_positions[symbol] = {
            "side": signal.type.value,
            "size": sizing.size,
            "entry_price": signal.price,
            "leverage": sizing.leverage,
            "stop_loss": sizing.stop_loss,
            "take_profit": sizing.take_profit,
            "entry_time": datetime.now(),
        }
        
        self.trading_logger.log_fill(
            order_id,
            symbol,
            side,
            sizing.size,
            signal.price,
            fees,
        )
        
        # Audit log fill
        self.audit_logger.log_order_filled(
            order_id=order_id,
            symbol=symbol,
            fill_price=signal.price,
            fill_size=sizing.size,
            fees=fees,
        )
        
        # Audit log position
        self.audit_logger.log_position_opened(
            symbol=symbol,
            side=side,
            size=sizing.size,
            entry_price=signal.price,
            leverage=sizing.leverage,
            stop_loss=sizing.stop_loss,
            take_profit=sizing.take_profit,
        )
        
        # Update metrics
        self.metrics.record_order_filled()
        self.metrics.update_position_count(len(self.paper_positions))

    async def _execute_live_trade(
        self,
        symbol: str,
        signal: Signal,
        sizing: Any,
    ) -> None:
        """Execute a live trade."""
        try:
            side = OrderSide.BUY if signal.type == SignalType.LONG else OrderSide.SELL
            
            order = await self.rest_client.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=sizing.size,
                leverage=sizing.leverage,
            )
            
            self.trading_logger.log_order(
                order.order_id,
                symbol,
                side.value,
                sizing.size,
                signal.price,
                "market",
            )
            
        except Exception as e:
            self.trading_logger.log_error("LIVE_TRADE", e)

    async def start(self) -> None:
        """Start the trading bot."""
        await self.initialize()
        
        self.is_running = True
        logger.info(f"Starting bot in {'PAPER' if self.is_paper else 'LIVE'} mode")
        
        # Subscribe to market data for selected pairs
        for pair in self.selected_pairs:
            try:
                is_futures = pair.market == Market.FUTURES
                await self.ws_client.subscribe_ticker(
                    pair.symbol,
                    self._on_ticker_update,
                    is_futures=is_futures,
                )
            except Exception as e:
                logger.error(f"Failed to subscribe to {pair.symbol}: {e}")
        
        # Main loop
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Bot shutdown requested")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot."""
        self.is_running = False
        
        await self.ws_client.close()
        await self.rest_client.close()
        
        logger.info("Trading bot stopped")

    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Run a backtest for a specific strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Fetch historical data
        data = await self._fetch_historical_data(symbol, limit=1000)
        if data is None:
            raise ValueError(f"Could not fetch data for {symbol}")
        
        # Run backtest
        backtest_config = self.config.backtest_config
        engine = BacktestEngine(
            initial_capital=backtest_config.get("initial_capital", 10000),
        )
        
        strategy = self.strategies[strategy_name]
        result = engine.run(strategy, data, self.risk_manager)
        
        logger.info(
            f"Backtest results for {strategy_name} on {symbol}:\n"
            f"  Total Return: {result.total_return:.2%}\n"
            f"  Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
            f"  Max Drawdown: {result.max_drawdown:.2%}\n"
            f"  Win Rate: {result.win_rate:.2%}\n"
            f"  Total Trades: {result.total_trades}"
        )
        
        return {
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "profit_factor": result.profit_factor,
        }

    async def run_walk_forward(
        self,
        strategy_name: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Run walk-forward validation for a strategy."""
        if strategy_name not in self.STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Fetch historical data
        data = await self._fetch_historical_data(symbol, limit=2000)
        if data is None:
            raise ValueError(f"Could not fetch data for {symbol}")
        
        backtest_config = self.config.backtest_config
        validator = WalkForwardValidator(
            n_periods=backtest_config.get("walk_forward_periods", 5),
            train_pct=backtest_config.get("train_pct", 0.7),
        )
        
        strategy_config = self.config.strategy_config.get(strategy_name, {})
        results = validator.validate(
            self.STRATEGY_MAP[strategy_name],
            data,
            strategy_config,
            self.risk_manager,
            initial_capital=backtest_config.get("initial_capital", 10000),
        )
        
        summary = validator.aggregate_results(results)
        
        logger.info(
            f"Walk-forward results for {strategy_name} on {symbol}:\n"
            f"  Avg Return: {summary.get('avg_return', 0):.2%}\n"
            f"  Avg Sharpe: {summary.get('avg_sharpe', 0):.2f}\n"
            f"  Consistent Positive: {summary.get('consistent_positive', False)}"
        )
        
        return summary


async def main() -> None:
    """Main entry point."""
    bot = TradingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
