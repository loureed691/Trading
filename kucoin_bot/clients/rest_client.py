"""KuCoin REST API client for spot, margin, and futures trading."""

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import aiohttp

from .base import BaseClient, RateLimiter, RetryConfig, with_retry


class Market(Enum):
    """Trading market types."""

    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Symbol:
    """Trading symbol information."""

    symbol: str
    base_currency: str
    quote_currency: str
    base_min_size: float
    base_max_size: float
    quote_min_size: float
    quote_max_size: float
    price_increment: float
    size_increment: float
    is_margin_enabled: bool = False
    fee_rate: float = 0.001


@dataclass
class Ticker:
    """Ticker data."""

    symbol: str
    price: float
    size: float
    bid: float
    ask: float
    volume_24h: float
    change_rate: float
    time: int


@dataclass
class OrderBook:
    """Order book data."""

    symbol: str
    bids: list[tuple[float, float]]  # price, size
    asks: list[tuple[float, float]]  # price, size
    time: int


@dataclass
class Order:
    """Order information."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    size: float
    filled_size: float
    status: str
    created_at: int


@dataclass
class Position:
    """Position information (for futures/margin)."""

    symbol: str
    side: str
    size: float
    avg_entry_price: float
    leverage: float
    unrealized_pnl: float
    liquidation_price: float
    margin: float


class KuCoinRestClient(BaseClient):
    """KuCoin REST API client with v2 API support for 2025."""

    # Updated API URLs for 2025
    SPOT_API_URL = "https://api.kucoin.com"
    SPOT_SANDBOX_URL = "https://openapi-sandbox.kucoin.com"
    FUTURES_API_URL = "https://api-futures.kucoin.com"
    FUTURES_SANDBOX_URL = "https://api-sandbox-futures.kucoin.com"
    
    # API version for authentication
    API_VERSION = "3"  # Updated to v3 for 2025

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        sandbox: bool = True,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(api_key, api_secret, api_passphrase, sandbox, retry_config)
        self.spot_url = self.SPOT_SANDBOX_URL if sandbox else self.SPOT_API_URL
        self.futures_url = self.FUTURES_SANDBOX_URL if sandbox else self.FUTURES_API_URL

    def _generate_signature(
        self, timestamp: str, method: str, endpoint: str, body: str = ""
    ) -> tuple[str, str]:
        """Generate API signature using updated v3 authentication."""
        str_to_sign = timestamp + method + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                str_to_sign.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        passphrase = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                self.api_passphrase.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return signature, passphrase

    def _get_headers(
        self, method: str, endpoint: str, body: str = ""
    ) -> dict[str, str]:
        """Get request headers with authentication (v3 format for 2025)."""
        timestamp = str(int(time.time() * 1000))
        signature, passphrase = self._generate_signature(
            timestamp, method, endpoint, body
        )

        return {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": self.API_VERSION,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        base_url: str | None = None,
        params: dict | None = None,
        data: dict | None = None,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        """Make API request with enhanced error handling and rate limiting."""
        await self._ensure_session()
        await self.rate_limiter.acquire()

        url = (base_url or self.spot_url) + endpoint

        body = ""
        if data:
            body = json.dumps(data)

        headers = {}
        if authenticated:
            headers = self._get_headers(method.upper(), endpoint, body)

        async with self._session.request(
            method, url, params=params, data=body if body else None, headers=headers
        ) as response:
            result = await response.json()
            if result.get("code") != "200000":
                raise Exception(f"API Error: {result.get('msg', 'Unknown error')}")
            return result.get("data", {})

    @with_retry()
    async def get_symbols(self, market: Market = Market.SPOT) -> list[Symbol]:
        """Get all trading symbols."""
        if market == Market.FUTURES:
            data = await self._request("GET", "/api/v1/contracts/active", self.futures_url)
            return [
                Symbol(
                    symbol=s["symbol"],
                    base_currency=s["baseCurrency"],
                    quote_currency=s["quoteCurrency"],
                    base_min_size=float(s.get("lotSize", 1)),
                    base_max_size=float(s.get("maxOrderQty", 1000000)),
                    quote_min_size=0.0,
                    quote_max_size=0.0,
                    price_increment=float(s.get("tickSize", 0.0001)),
                    size_increment=float(s.get("lotSize", 1)),
                    fee_rate=float(s.get("takerFeeRate", 0.0006)),
                )
                for s in data
            ]
        else:
            data = await self._request("GET", "/api/v2/symbols")
            return [
                Symbol(
                    symbol=s["symbol"],
                    base_currency=s["baseCurrency"],
                    quote_currency=s["quoteCurrency"],
                    base_min_size=float(s["baseMinSize"]),
                    base_max_size=float(s["baseMaxSize"]),
                    quote_min_size=float(s["quoteMinSize"]),
                    quote_max_size=float(s["quoteMaxSize"]),
                    price_increment=float(s["priceIncrement"]),
                    size_increment=float(s["baseIncrement"]),
                    is_margin_enabled=s.get("isMarginEnabled", False),
                )
                for s in data
            ]

    @with_retry()
    async def get_ticker(self, symbol: str, market: Market = Market.SPOT) -> Ticker:
        """Get ticker for a symbol."""
        if market == Market.FUTURES:
            data = await self._request(
                "GET", f"/api/v1/ticker?symbol={symbol}", self.futures_url
            )
        else:
            data = await self._request("GET", f"/api/v1/market/orderbook/level1?symbol={symbol}")

        return Ticker(
            symbol=symbol,
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            bid=float(data.get("bestBid", 0)),
            ask=float(data.get("bestAsk", 0)),
            volume_24h=float(data.get("vol", 0)),
            change_rate=float(data.get("changeRate", 0)),
            time=int(data.get("time", 0)),
        )

    @with_retry()
    async def get_tickers(self, market: Market = Market.SPOT) -> list[Ticker]:
        """Get all tickers."""
        if market == Market.FUTURES:
            # Futures doesn't have bulk tickers, fetch individually
            symbols = await self.get_symbols(market)
            tickers = []
            for sym in symbols[:20]:  # Limit for rate limiting
                try:
                    ticker = await self.get_ticker(sym.symbol, market)
                    tickers.append(ticker)
                except Exception:
                    pass
            return tickers
        else:
            data = await self._request("GET", "/api/v1/market/allTickers")
            return [
                Ticker(
                    symbol=t["symbol"],
                    price=float(t.get("last", 0) or 0),
                    size=0,
                    bid=float(t.get("buy", 0) or 0),
                    ask=float(t.get("sell", 0) or 0),
                    volume_24h=float(t.get("volValue", 0) or 0),
                    change_rate=float(t.get("changeRate", 0) or 0),
                    time=int(time.time() * 1000),
                )
                for t in data.get("ticker", [])
            ]

    @with_retry()
    async def get_orderbook(
        self, symbol: str, depth: int = 20, market: Market = Market.SPOT
    ) -> OrderBook:
        """Get order book for a symbol."""
        if market == Market.FUTURES:
            data = await self._request(
                "GET", f"/api/v1/level2/depth{depth}?symbol={symbol}", self.futures_url
            )
        else:
            data = await self._request(
                "GET", f"/api/v1/market/orderbook/level2_{depth}?symbol={symbol}"
            )

        return OrderBook(
            symbol=symbol,
            bids=[(float(b[0]), float(b[1])) for b in data.get("bids", [])],
            asks=[(float(a[0]), float(a[1])) for a in data.get("asks", [])],
            time=int(data.get("time", 0)),
        )

    @with_retry()
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1hour",
        start_time: int | None = None,
        end_time: int | None = None,
        market: Market = Market.SPOT,
    ) -> list[list]:
        """Get candlestick data."""
        params: dict[str, Any] = {"symbol": symbol, "type": interval}
        if start_time:
            params["startAt"] = start_time
        if end_time:
            params["endAt"] = end_time

        if market == Market.FUTURES:
            data = await self._request(
                "GET", "/api/v1/kline/query", self.futures_url, params=params
            )
        else:
            data = await self._request("GET", "/api/v1/market/candles", params=params)

        return data

    @with_retry()
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for futures contract."""
        data = await self._request(
            "GET", f"/api/v1/funding-rate/{symbol}/current", self.futures_url
        )
        return float(data.get("value", 0))

    @with_retry()
    async def get_account_balance(
        self, account_type: str = "trade"
    ) -> dict[str, float]:
        """Get account balances."""
        data = await self._request(
            "GET", f"/api/v1/accounts?type={account_type}", authenticated=True
        )
        balances = {}
        for account in data:
            currency = account["currency"]
            balances[currency] = float(account["available"])
        return balances

    @with_retry()
    async def get_futures_balance(self) -> dict[str, Any]:
        """Get futures account overview."""
        data = await self._request(
            "GET", "/api/v1/account-overview", self.futures_url, authenticated=True
        )
        return data

    @with_retry()
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: float | None = None,
        leverage: int | None = None,
        market: Market = Market.SPOT,
        client_oid: str | None = None,
    ) -> Order:
        """Place an order."""
        data: dict[str, Any] = {
            "clientOid": client_oid or str(uuid.uuid4()),
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "size": str(size),
        }

        if price and order_type == OrderType.LIMIT:
            data["price"] = str(price)

        if market == Market.FUTURES:
            if leverage:
                data["leverage"] = leverage
            result = await self._request(
                "POST", "/api/v1/orders", self.futures_url, data=data, authenticated=True
            )
        elif market == Market.MARGIN:
            data["marginModel"] = "cross"
            result = await self._request(
                "POST", "/api/v1/margin/order", data=data, authenticated=True
            )
        else:
            result = await self._request(
                "POST", "/api/v1/orders", data=data, authenticated=True
            )

        return Order(
            order_id=result.get("orderId", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price or 0,
            size=size,
            filled_size=0,
            status="open",
            created_at=int(time.time() * 1000),
        )

    @with_retry()
    async def cancel_order(
        self, order_id: str, market: Market = Market.SPOT
    ) -> bool:
        """Cancel an order."""
        if market == Market.FUTURES:
            await self._request(
                "DELETE",
                f"/api/v1/orders/{order_id}",
                self.futures_url,
                authenticated=True,
            )
        else:
            await self._request(
                "DELETE", f"/api/v1/orders/{order_id}", authenticated=True
            )
        return True

    @with_retry()
    async def get_order(self, order_id: str, market: Market = Market.SPOT) -> Order:
        """Get order details."""
        if market == Market.FUTURES:
            data = await self._request(
                "GET",
                f"/api/v1/orders/{order_id}",
                self.futures_url,
                authenticated=True,
            )
        else:
            data = await self._request(
                "GET", f"/api/v1/orders/{order_id}", authenticated=True
            )

        return Order(
            order_id=data["id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            price=float(data.get("price", 0)),
            size=float(data["size"]),
            filled_size=float(data.get("dealSize", 0)),
            status=data.get("status", ""),
            created_at=int(data.get("createdAt", 0)),
        )

    @with_retry()
    async def get_positions(self) -> list[Position]:
        """Get all open positions (futures)."""
        data = await self._request(
            "GET", "/api/v1/positions", self.futures_url, authenticated=True
        )
        return [
            Position(
                symbol=p["symbol"],
                side=p.get("side", ""),
                size=float(p.get("currentQty", 0)),
                avg_entry_price=float(p.get("avgEntryPrice", 0)),
                leverage=float(p.get("leverage", 1)),
                unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                liquidation_price=float(p.get("liquidationPrice", 0)),
                margin=float(p.get("maintMargin", 0)),
            )
            for p in data
            if float(p.get("currentQty", 0)) != 0
        ]

    @with_retry()
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a futures symbol."""
        await self._request(
            "POST",
            "/api/v1/position/leverage",
            self.futures_url,
            data={"symbol": symbol, "leverage": str(leverage)},
            authenticated=True,
        )
        return True
