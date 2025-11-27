"""Pair selection module based on volume, spread, volatility, funding, and fees."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .clients.rest_client import KuCoinRestClient, Market, Symbol, Ticker

logger = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Scoring for a trading pair."""

    symbol: str
    market: Market
    volume_score: float
    spread_score: float
    volatility_score: float
    funding_score: float
    fee_score: float
    total_score: float
    expected_edge: float
    
    # Raw metrics
    volume_24h: float
    spread_pct: float
    volatility: float
    funding_rate: float
    fee_rate: float


class PairSelector:
    """Select trading pairs based on multiple criteria."""

    def __init__(
        self,
        rest_client: KuCoinRestClient,
        config: dict[str, Any],
    ):
        self.client = rest_client
        self.min_volume = config.get("min_volume_24h", 1000000)
        self.max_spread = config.get("max_spread_pct", 0.5)
        self.min_volatility = config.get("min_volatility", 0.02)
        self.max_volatility = config.get("max_volatility", 0.15)
        self.max_funding = config.get("max_funding_rate", 0.001)
        self.max_fee = config.get("max_fee_pct", 0.1)
        
        # Scoring weights
        self.weights = config.get("weights", {
            "volume": 0.2,
            "spread": 0.25,
            "volatility": 0.25,
            "funding": 0.15,
            "fee": 0.15,
        })

    def _calculate_spread(self, ticker: Ticker) -> float:
        """Calculate bid-ask spread percentage."""
        if ticker.bid <= 0 or ticker.ask <= 0:
            return float("inf")
        mid_price = (ticker.bid + ticker.ask) / 2
        return ((ticker.ask - ticker.bid) / mid_price) * 100

    async def _calculate_volatility(
        self, symbol: str, market: Market = Market.SPOT
    ) -> float:
        """Calculate historical volatility from klines."""
        try:
            klines = await self.client.get_klines(symbol, "1hour", market=market)
            if not klines or len(klines) < 24:
                return 0.0
            
            # Extract closing prices
            closes = [float(k[2]) for k in klines[-24:]]  # Last 24 hours
            returns = np.diff(np.log(closes))
            volatility = np.std(returns) * np.sqrt(24)  # Annualize roughly
            return volatility
        except Exception as e:
            logger.warning(f"Failed to calculate volatility for {symbol}: {e}")
            return 0.0

    def _score_volume(self, volume: float) -> float:
        """Score volume (higher is better)."""
        if volume < self.min_volume:
            return 0.0
        # Logarithmic scaling
        return min(1.0, np.log10(volume / self.min_volume) / 3)

    def _score_spread(self, spread_pct: float) -> float:
        """Score spread (lower is better)."""
        if spread_pct > self.max_spread:
            return 0.0
        return max(0.0, 1.0 - (spread_pct / self.max_spread))

    def _score_volatility(self, volatility: float) -> float:
        """Score volatility (optimal range)."""
        if volatility < self.min_volatility or volatility > self.max_volatility:
            return 0.0
        # Peak score at midpoint
        mid = (self.min_volatility + self.max_volatility) / 2
        range_size = self.max_volatility - self.min_volatility
        return 1.0 - 2 * abs(volatility - mid) / range_size

    def _score_funding(self, funding_rate: float) -> float:
        """Score funding rate (lower absolute value is better)."""
        abs_funding = abs(funding_rate)
        if abs_funding > self.max_funding:
            return 0.0
        return 1.0 - (abs_funding / self.max_funding)

    def _score_fee(self, fee_rate: float) -> float:
        """Score fee rate (lower is better)."""
        fee_pct = fee_rate * 100
        if fee_pct > self.max_fee:
            return 0.0
        return 1.0 - (fee_pct / self.max_fee)

    def _calculate_expected_edge(
        self,
        volatility: float,
        spread_pct: float,
        fee_rate: float,
        funding_rate: float = 0.0,
    ) -> float:
        """Calculate expected edge for trading this pair."""
        # Expected edge = potential profit from volatility - costs
        # Simplified model: edge = volatility * efficiency - spread - fees - abs(funding)
        efficiency = 0.1  # Assume 10% capture of volatility
        gross_edge = volatility * efficiency * 100  # Convert to percentage
        costs = spread_pct + (fee_rate * 100 * 2) + abs(funding_rate * 100)  # Round-trip
        return gross_edge - costs

    async def score_pair(
        self,
        symbol: Symbol,
        ticker: Ticker,
        market: Market = Market.SPOT,
    ) -> PairScore | None:
        """Score a trading pair."""
        try:
            spread_pct = self._calculate_spread(ticker)
            volatility = await self._calculate_volatility(symbol.symbol, market)
            
            funding_rate = 0.0
            if market == Market.FUTURES:
                try:
                    funding_rate = await self.client.get_funding_rate(symbol.symbol)
                except Exception:
                    pass

            # Calculate individual scores
            volume_score = self._score_volume(ticker.volume_24h)
            spread_score = self._score_spread(spread_pct)
            volatility_score = self._score_volatility(volatility)
            funding_score = self._score_funding(funding_rate)
            fee_score = self._score_fee(symbol.fee_rate)

            # Calculate weighted total score
            total_score = (
                volume_score * self.weights["volume"]
                + spread_score * self.weights["spread"]
                + volatility_score * self.weights["volatility"]
                + funding_score * self.weights["funding"]
                + fee_score * self.weights["fee"]
            )

            # Calculate expected edge
            expected_edge = self._calculate_expected_edge(
                volatility, spread_pct, symbol.fee_rate, funding_rate
            )

            return PairScore(
                symbol=symbol.symbol,
                market=market,
                volume_score=volume_score,
                spread_score=spread_score,
                volatility_score=volatility_score,
                funding_score=funding_score,
                fee_score=fee_score,
                total_score=total_score,
                expected_edge=expected_edge,
                volume_24h=ticker.volume_24h,
                spread_pct=spread_pct,
                volatility=volatility,
                funding_rate=funding_rate,
                fee_rate=symbol.fee_rate,
            )

        except Exception as e:
            logger.error(f"Failed to score pair {symbol.symbol}: {e}")
            return None

    async def select_pairs(
        self,
        market: Market = Market.SPOT,
        top_n: int = 10,
        min_score: float = 0.5,
    ) -> list[PairScore]:
        """Select top trading pairs based on scoring criteria."""
        symbols = await self.client.get_symbols(market)
        tickers = await self.client.get_tickers(market)
        
        # Create ticker lookup
        ticker_map = {t.symbol: t for t in tickers}
        
        scores: list[PairScore] = []
        
        for symbol in symbols:
            if symbol.symbol not in ticker_map:
                continue
                
            ticker = ticker_map[symbol.symbol]
            
            # Quick filter
            if ticker.volume_24h < self.min_volume:
                continue
                
            score = await self.score_pair(symbol, ticker, market)
            if score and score.total_score >= min_score:
                scores.append(score)

        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Return top N
        selected = scores[:top_n]
        
        logger.info(
            f"Selected {len(selected)} pairs from {len(symbols)} total "
            f"(market={market.value}, min_score={min_score})"
        )
        
        for pair in selected:
            logger.debug(
                f"  {pair.symbol}: score={pair.total_score:.3f}, "
                f"edge={pair.expected_edge:.3f}%"
            )

        return selected
