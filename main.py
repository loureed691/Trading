#!/usr/bin/env python3
"""KuCoin Trading Bot CLI entry point."""

import argparse
import asyncio
import sys

from kucoin_bot.bot import TradingBot


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KuCoin Trading Bot - Automated trading for spot, margin, and futures"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the trading bot")
    run_parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper trading mode",
    )
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Strategy to backtest (trend, mean_reversion, breakout, market_making)",
    )
    backtest_parser.add_argument(
        "-p", "--pair",
        required=True,
        help="Trading pair to backtest (e.g., BTC-USDT)",
    )
    
    # Walk-forward command
    wf_parser = subparsers.add_parser("walk-forward", help="Run walk-forward validation")
    wf_parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Strategy to validate",
    )
    wf_parser.add_argument(
        "-p", "--pair",
        required=True,
        help="Trading pair to validate",
    )
    
    # Select pairs command
    subparsers.add_parser("select-pairs", help="Select and display best trading pairs")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        bot = TradingBot(args.config)
        
        if args.command == "run":
            asyncio.run(bot.start())
            
        elif args.command == "backtest":
            result = asyncio.run(bot.run_backtest(args.strategy, args.pair))
            print("\nBacktest Results:")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
                    
        elif args.command == "walk-forward":
            result = asyncio.run(bot.run_walk_forward(args.strategy, args.pair))
            print("\nWalk-Forward Validation Results:")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
                    
        elif args.command == "select-pairs":
            asyncio.run(bot.initialize())
            print("\nSelected Trading Pairs:")
            for pair in bot.selected_pairs:
                print(f"  {pair.symbol} ({pair.market.value}):")
                print(f"    Score: {pair.total_score:.3f}")
                print(f"    Expected Edge: {pair.expected_edge:.3f}%")
                print(f"    Volume 24h: ${pair.volume_24h:,.0f}")
                print(f"    Spread: {pair.spread_pct:.3f}%")
                print()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
