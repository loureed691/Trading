"""Pair selection module based on volume, spread, volatility, funding, and fees."""

import logging
from dataclasses import dataclass, field
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
    
    # Market selection metadata
    recommended_market: Market | None = None
    market_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketRecommendation:
    """Recommendation for which market to trade a symbol."""

    symbol: str
    recommended_market: Market
    confidence: float
    reasons: list[str]
    alternative_markets: list[tuple[Market, float]]  # (market, score)


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

    async def recommend_market(
        self,
        base_symbol: str,
        portfolio_value: float,
        volatility: float = 0.05,
        risk_tolerance: str = "medium",
    ) -> MarketRecommendation:
        """Recommend the best market (spot/margin/futures) for a symbol.
        
        Considers:
        - Liquidity across markets
        - Funding rates (futures)
        - Borrowing costs (margin)
        - Leverage requirements
        - Risk tolerance
        """
        reasons = []
        market_scores: dict[Market, float] = {}
        
        # Score each market
        for market in [Market.SPOT, Market.MARGIN, Market.FUTURES]:
            try:
                score = await self._score_market_for_symbol(
                    base_symbol, market, portfolio_value, volatility, risk_tolerance
                )
                market_scores[market] = score
            except Exception as e:
                logger.debug(f"Could not score {market.value} for {base_symbol}: {e}")
                market_scores[market] = 0.0
        
        # Determine best market
        if not market_scores or max(market_scores.values()) == 0:
            return MarketRecommendation(
                symbol=base_symbol,
                recommended_market=Market.SPOT,
                confidence=0.0,
                reasons=["No market data available, defaulting to spot"],
                alternative_markets=[],
            )
        
        best_market = max(market_scores, key=market_scores.get)
        best_score = market_scores[best_market]
        
        # Calculate confidence
        total_score = sum(market_scores.values())
        confidence = best_score / total_score if total_score > 0 else 0
        
        # Generate reasons
        if best_market == Market.SPOT:
            reasons.append("Lowest risk, no leverage required")
            if volatility > 0.08:
                reasons.append("High volatility favors lower leverage")
        elif best_market == Market.MARGIN:
            reasons.append("Good for moderate leverage with flexibility")
            if risk_tolerance == "medium":
                reasons.append("Matches medium risk tolerance")
        elif best_market == Market.FUTURES:
            reasons.append("Best for directional trades with leverage")
            if volatility < 0.03:
                reasons.append("Low volatility can be leveraged safely")
        
        # Get alternatives
        sorted_markets = sorted(
            market_scores.items(), key=lambda x: x[1], reverse=True
        )
        alternatives = [
            (m, s) for m, s in sorted_markets[1:] if s > 0
        ]
        
        return MarketRecommendation(
            symbol=base_symbol,
            recommended_market=best_market,
            confidence=confidence,
            reasons=reasons,
            alternative_markets=alternatives,
        )

    async def _score_market_for_symbol(
        self,
        symbol: str,
        market: Market,
        portfolio_value: float,
        volatility: float,
        risk_tolerance: str,
    ) -> float:
        """Score a specific market for a symbol."""
        score = 50.0  # Base score
        
        try:
            # Get ticker data
            ticker = await self.client.get_ticker(symbol, market)
            
            # Liquidity score
            if ticker.volume_24h >= self.min_volume * 2:
                score += 20
            elif ticker.volume_24h >= self.min_volume:
                score += 10
            else:
                score -= 20
            
            # Spread score - protect against division by zero
            if ticker.bid > 0 and ticker.ask > 0:
                mid_price = (ticker.ask + ticker.bid) / 2
                if mid_price > 0:
                    spread_pct = (ticker.ask - ticker.bid) / mid_price * 100
                    if spread_pct < 0.1:
                        score += 15
                    elif spread_pct < 0.3:
                        score += 10
                    elif spread_pct > 0.5:
                        score -= 10
        except Exception:
            score -= 30  # Market likely not available
        
        # Market-specific adjustments
        if market == Market.SPOT:
            # Spot is safer
            if risk_tolerance == "low":
                score += 20
            elif volatility > 0.08:
                score += 15  # Prefer spot in high vol
        
        elif market == Market.MARGIN:
            # Margin has borrowing costs
            if risk_tolerance in ["medium", "high"]:
                score += 10
            else:
                score -= 10
            
            # Volatility adjustment
            if 0.03 <= volatility <= 0.08:
                score += 10  # Good for moderate leverage
        
        elif market == Market.FUTURES:
            # Check funding rate
            try:
                funding_rate = await self.client.get_funding_rate(symbol)
                if abs(funding_rate) < 0.0001:
                    score += 15  # Low funding is good
                elif abs(funding_rate) < 0.0005:
                    score += 5
                elif abs(funding_rate) > 0.001:
                    score -= 20  # High funding is costly
            except Exception:
                score -= 10  # Can't get funding rate
            
            # Risk tolerance adjustment
            if risk_tolerance == "high":
                score += 20
            elif risk_tolerance == "low":
                score -= 30
            
            # Volatility adjustment
            if volatility < 0.03:
                score += 15  # Low vol can be leveraged
            elif volatility > 0.08:
                score -= 15  # High vol is risky with leverage
        
        return max(0, score)

    async def select_pairs_with_market_auto(
        self,
        top_n: int = 10,
        min_score: float = 0.5,
        portfolio_value: float = 10000.0,
        risk_tolerance: str = "medium",
    ) -> list[PairScore]:
        """Select pairs with automatic market selection.
        
        For each pair, determines the best market to trade it on.
        """
        all_pairs: list[PairScore] = []
        
        # Get spot pairs as base universe
        try:
            spot_pairs = await self.select_pairs(
                market=Market.SPOT, top_n=top_n * 2, min_score=min_score * 0.8
            )
        except Exception as e:
            logger.error(f"Failed to get spot pairs: {e}")
            spot_pairs = []
        
        # For each pair, find best market
        for pair in spot_pairs:
            # Get market recommendation
            recommendation = await self.recommend_market(
                pair.symbol,
                portfolio_value,
                pair.volatility,
                risk_tolerance,
            )
            
            # Update pair with recommended market
            pair.recommended_market = recommendation.recommended_market
            pair.market_scores = {
                m.value: s for m, s in recommendation.alternative_markets
            }
            pair.market_scores[recommendation.recommended_market.value] = (
                recommendation.confidence
            )
            
            # Adjust score based on market suitability
            pair.total_score *= (0.5 + recommendation.confidence * 0.5)
            
            all_pairs.append(pair)
        
        # Re-sort by adjusted score
        all_pairs.sort(key=lambda x: x.total_score, reverse=True)
        
        selected = all_pairs[:top_n]
        
        logger.info(
            f"Auto-selected {len(selected)} pairs with market recommendations"
        )
        
        for pair in selected:
            logger.info(
                f"  {pair.symbol}: {pair.recommended_market.value if pair.recommended_market else 'spot'} "
                f"(score={pair.total_score:.3f})"
            )
        
        return selected
