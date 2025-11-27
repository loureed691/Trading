# KuCoin Trading Bot

A modular Python trading bot for KuCoin exchange supporting spot, margin, and futures markets with advanced regime detection, ML-based forecasting, and comprehensive risk management.

## Features

### Multi-Market Support
- **Spot, Margin, and Futures Trading**: Full support for all KuCoin market types
- **Auto Market Selection**: Automatically selects the best market (spot/margin/futures) based on liquidity, volatility, funding rates, and risk tolerance
- **REST + WebSocket Clients**: Full REST API integration with real-time WebSocket data

### Regime Detection & Strategy Allocation
- **Market Regime Detection**: Identifies trend (up/down), mean-reverting, high/low volatility regimes
- **Hurst Exponent Analysis**: Distinguishes between trending and mean-reverting markets
- **Ensemble Strategy**: Combines multiple strategies with regime-based weight allocation
- **Dynamic Strategy Selection**: Automatically adjusts strategy weights based on detected regime

### Modular Strategies
- **Trend Following**: MA crossover with MACD confirmation
- **Mean Reversion**: Bollinger Bands with RSI
- **Breakout**: Price channels with volume confirmation
- **Market Making / Arbitrage**: Dynamic spread based on volatility and inventory
- **Ensemble**: Combines all strategies with regime-aware weighting

### ML-Based Features
- **Forecasting**: Statistical forecasting with cross-validation and leakage checks
- **Position Sizing**: ML-adjusted position sizes based on forecast confidence
- **Kelly Criterion**: Fractional Kelly sizing for optimal risk-adjusted returns
- **Volatility Targeting**: Automatic position scaling based on recent volatility

### Advanced Risk Management
- **VaR/ES Calculations**: Value at Risk and Expected Shortfall metrics
- **ATR-Based Stop Losses**: Dynamic stop losses based on Average True Range
- **Liquidation Distance**: Calculates distance to liquidation for leveraged positions
- **Funding Rate Adjustments**: Reduces leverage when funding rates are high
- **Dynamic Risk Parameters**: Auto-adjusts max_position_pct, max_drawdown_pct, max_leverage based on:
  - Current volatility regime
  - Drawdown levels
  - VaR/ES metrics
- **Portfolio Constraints**: Concentration limits, exposure limits, and net exposure checks
- **Dynamic Hedging**: Recommendations for hedging based on exposure and volatility

### Backtesting & Validation
- **Backtesting Engine**: Full simulation with fees and slippage
- **Walk-Forward Validation**: Out-of-sample testing with configurable periods
- **Purged Cross-Validation**: Prevents data leakage in ML training

### Monitoring & Audit
- **Structured Audit Logs**: JSON-formatted audit trail for all trades
- **Monitoring Metrics**: Real-time metrics collection (win rate, PnL, position counts)
- **Comprehensive Logging**: Rotating file logs with structured formatting

### Safe Defaults
- **Paper Trading Mode**: Test strategies risk-free
- **Retry Logic**: Exponential backoff for API failures
- **Rate Limiting**: Built-in rate limiting for API calls
- **Margin Buffer**: Safety margin to prevent margin calls

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Trading.git
cd Trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration:
```bash
cp config.example.yaml config.yaml
```

2. Edit `config.yaml` with your settings:
```yaml
api:
  key: "your-api-key"
  secret: "your-api-secret"
  passphrase: "your-passphrase"
  sandbox: true  # Use sandbox for testing

trading:
  mode: "paper"  # paper or live
  markets:
    - spot
    - margin
    - futures

# Regime detection
regime:
  trend_threshold: 0.3
  volatility_lookback: 20

# ML settings
ml:
  min_cv_score: 0.51
  kelly_fraction: 0.25

# Risk (auto-adjusted based on conditions)
risk:
  max_position_pct: 5.0
  max_drawdown_pct: 10.0
  max_leverage: 5

# Ensemble strategy
strategies:
  use_ensemble: true
  ensemble:
    min_agreement: 0.5
```

You can also use environment variables:
```yaml
api:
  key: "${KUCOIN_API_KEY}"
  secret: "${KUCOIN_API_SECRET}"
  passphrase: "${KUCOIN_API_PASSPHRASE}"
```

## Usage

### Run the Trading Bot

```bash
# Paper trading (default)
python main.py run

# With custom config
python main.py -c my_config.yaml run
```

### Select Trading Pairs

```bash
python main.py select-pairs
```

### Run Backtest

```bash
python main.py backtest -s trend -p BTC-USDT
python main.py backtest -s mean_reversion -p ETH-USDT
```

### Walk-Forward Validation

```bash
python main.py walk-forward -s trend -p BTC-USDT
```

## Project Structure

```
Trading/
├── kucoin_bot/
│   ├── __init__.py
│   ├── bot.py              # Main bot orchestrator
│   ├── config.py           # Configuration management
│   ├── pair_selector.py    # Pair selection with auto market selection
│   ├── clients/
│   │   ├── base.py         # Base client with retry logic
│   │   ├── rest_client.py  # KuCoin REST API client
│   │   └── ws_client.py    # KuCoin WebSocket client
│   ├── regime/
│   │   ├── __init__.py
│   │   └── detector.py     # Regime detection (trend/MR/volatility)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── forecaster.py   # ML forecasting with CV
│   │   └── position_sizer.py # ML position sizing
│   ├── strategies/
│   │   ├── base.py         # Base strategy class
│   │   ├── ensemble.py     # Ensemble strategy
│   │   ├── trend.py        # Trend following
│   │   ├── mean_reversion.py
│   │   ├── breakout.py
│   │   └── market_making.py
│   ├── risk/
│   │   └── risk_manager.py # Risk management with VaR/ES
│   ├── backtest/
│   │   └── engine.py       # Backtesting engine
│   └── utils/
│       └── logging.py      # Logging and audit utilities
├── tests/
│   ├── test_config.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   ├── test_backtest.py
│   ├── test_regime.py      # Regime detection tests
│   └── test_ml.py          # ML module tests
├── main.py                 # CLI entry point
├── config.example.yaml     # Example configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Strategies

### Trend Following
Uses dual moving average crossover with MACD confirmation. Best for trending markets (high Hurst exponent).

### Mean Reversion
Uses Bollinger Bands with RSI confirmation. Best for ranging markets with mean-reverting behavior (low Hurst exponent).

### Breakout
Uses price channel breakouts with volume confirmation. Best for capturing momentum moves in high volatility regimes.

### Market Making
Provides liquidity with dynamic spread based on volatility and inventory management. Best for low volatility regimes.

### Ensemble (Default)
Combines all strategies with regime-based weight allocation:
- **Trend regime**: Higher weight to trend/breakout strategies
- **Mean-revert regime**: Higher weight to mean-reversion/market-making
- **High volatility**: Higher weight to breakout strategies
- **Low volatility**: Higher weight to market-making strategies

## Risk Management

The bot includes comprehensive risk controls that automatically adjust:

### Position Sizing
- Based on signal strength, volatility, and available margin
- ML-adjusted based on forecast confidence
- Kelly criterion with fractional sizing

### Dynamic Risk Parameters
- **max_position_pct**: Reduced in high volatility or during drawdowns
- **max_leverage**: Reduced when funding rates are high or volatility increases
- **max_drawdown_pct**: Triggers trading halt when exceeded

### Risk Metrics
- **VaR (Value at Risk)**: 95% and 99% confidence levels
- **ES (Expected Shortfall)**: Average loss beyond VaR
- **Liquidation Distance**: Percentage move to trigger liquidation

### Portfolio Constraints
- Maximum 20% concentration per position
- Maximum 50% exposure per market
- Maximum 150% net directional exposure

### Dynamic Hedging
- Automatic hedging recommendations when:
  - Position exceeds 30% of portfolio
  - Volatility exceeds 5% daily
  - Significant unrealized profits (>10%)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=kucoin_bot

# Run specific test file
pytest tests/test_strategies.py -v
pytest tests/test_regime.py -v
pytest tests/test_ml.py -v
```

## Development

### Adding a New Strategy

1. Create a new file in `kucoin_bot/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signal()` method
4. Register in `TradingBot.STRATEGY_MAP`

```python
from .base import BaseStrategy, Signal, SignalType

class MyStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__("my_strategy", config)
    
    def generate_signal(self, data):
        # Your logic here
        return Signal(
            type=SignalType.LONG,
            symbol=data.attrs.get("symbol", ""),
            strength=0.8,
            price=data["close"].iloc[-1],
        )
```

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves significant risk. Always test thoroughly in paper mode before using real funds. The authors are not responsible for any financial losses.

## License

MIT License