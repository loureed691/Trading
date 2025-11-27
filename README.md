# KuCoin Trading Bot

A modular Python trading bot for KuCoin exchange supporting spot, margin, and futures markets.

## Features

- **Multi-Market Support**: Trade on spot, margin, and futures markets
- **REST + WebSocket Clients**: Full REST API integration with real-time WebSocket data
- **Intelligent Pair Selection**: Auto-select trading pairs based on volume, spread, volatility, funding rates, and fees
- **Modular Strategies**:
  - Trend Following (MA crossover with MACD confirmation)
  - Mean Reversion (Bollinger Bands with RSI)
  - Breakout (Price channels with volume confirmation)
  - Market Making / Arbitrage
- **Backtesting Engine**: Walk-forward validation with comprehensive metrics
- **Risk Management**:
  - ATR-based stop losses
  - Value at Risk (VaR) calculations
  - Maximum drawdown controls
  - Dynamic leverage selection
  - Margin requirement checks
- **Paper & Live Trading**: Test strategies risk-free before going live
- **Comprehensive Logging**: Structured logging with rotation

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
    - futures

risk:
  max_position_pct: 5.0
  max_drawdown_pct: 10.0
  max_leverage: 5
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
│   ├── pair_selector.py    # Pair selection logic
│   ├── clients/
│   │   ├── base.py         # Base client with retry logic
│   │   ├── rest_client.py  # KuCoin REST API client
│   │   └── ws_client.py    # KuCoin WebSocket client
│   ├── strategies/
│   │   ├── base.py         # Base strategy class
│   │   ├── trend.py        # Trend following strategy
│   │   ├── mean_reversion.py
│   │   ├── breakout.py
│   │   └── market_making.py
│   ├── risk/
│   │   └── risk_manager.py # Risk management module
│   ├── backtest/
│   │   └── engine.py       # Backtesting engine
│   └── utils/
│       └── logging.py      # Logging utilities
├── tests/
│   ├── test_config.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_backtest.py
├── main.py                 # CLI entry point
├── config.example.yaml     # Example configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Strategies

### Trend Following
Uses dual moving average crossover with MACD confirmation. Best for trending markets.

### Mean Reversion
Uses Bollinger Bands with RSI confirmation. Best for ranging markets with mean-reverting behavior.

### Breakout
Uses price channel breakouts with volume confirmation. Best for capturing momentum moves.

### Market Making
Provides liquidity with dynamic spread based on volatility and inventory management.

## Risk Management

The bot includes comprehensive risk controls:

- **Position Sizing**: Based on signal strength, volatility, and available margin
- **Stop Loss**: ATR-based dynamic stop losses
- **Max Drawdown**: Stops trading when drawdown exceeds threshold
- **Daily Loss Limit**: Prevents excessive daily losses
- **Leverage Control**: Dynamic leverage based on volatility and signal strength
- **Margin Checks**: Validates margin requirements before placing orders

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=kucoin_bot

# Run specific test file
pytest tests/test_strategies.py -v
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