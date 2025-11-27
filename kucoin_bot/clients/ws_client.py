"""KuCoin WebSocket client for real-time market data."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from .base import BaseClient, RetryConfig

logger = logging.getLogger(__name__)


class WsType(Enum):
    """WebSocket types."""

    PUBLIC = "public"
    PRIVATE = "private"


@dataclass
class WsMessage:
    """WebSocket message."""

    topic: str
    type: str
    data: dict[str, Any]
    subject: str
    time: int


class KuCoinWebSocketClient(BaseClient):
    """KuCoin WebSocket client for real-time data."""

    SPOT_WS_URL = "https://api.kucoin.com"
    SPOT_SANDBOX_WS_URL = "https://openapi-sandbox.kucoin.com"
    FUTURES_WS_URL = "https://api-futures.kucoin.com"
    FUTURES_SANDBOX_WS_URL = "https://api-sandbox-futures.kucoin.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        sandbox: bool = True,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(api_key, api_secret, api_passphrase, sandbox, retry_config)
        self._ws = None
        self._ws_futures = None
        self._subscriptions: dict[str, Callable] = {}
        self._ping_task: asyncio.Task | None = None
        self._listen_task: asyncio.Task | None = None
        self._running = False
        self._connect_id = 0

    async def _get_ws_token(
        self, ws_type: WsType = WsType.PUBLIC, is_futures: bool = False
    ) -> dict[str, Any]:
        """Get WebSocket connection token."""
        await self._ensure_session()

        if is_futures:
            base_url = (
                self.FUTURES_SANDBOX_WS_URL if self.sandbox else self.FUTURES_WS_URL
            )
        else:
            base_url = self.SPOT_SANDBOX_WS_URL if self.sandbox else self.SPOT_WS_URL

        endpoint = (
            "/api/v1/bullet-public"
            if ws_type == WsType.PUBLIC
            else "/api/v1/bullet-private"
        )

        async with self._session.post(base_url + endpoint) as response:
            result = await response.json()
            if result.get("code") != "200000":
                raise Exception(f"Failed to get WS token: {result.get('msg')}")
            return result.get("data", {})

    async def connect(
        self, ws_type: WsType = WsType.PUBLIC, is_futures: bool = False
    ) -> None:
        """Connect to WebSocket."""
        token_data = await self._get_ws_token(ws_type, is_futures)
        servers = token_data.get("instanceServers", [])
        if not servers:
            raise Exception("No WebSocket servers available")

        server = servers[0]
        endpoint = server["endpoint"]
        token = token_data["token"]
        self._connect_id += 1

        ws_url = f"{endpoint}?token={token}&connectId={self._connect_id}"

        if is_futures:
            self._ws_futures = await websockets.connect(ws_url)
        else:
            self._ws = await websockets.connect(ws_url)

        self._running = True
        self._ping_task = asyncio.create_task(self._ping_loop(is_futures))
        self._listen_task = asyncio.create_task(self._listen_loop(is_futures))

        logger.info(f"Connected to KuCoin WebSocket ({'futures' if is_futures else 'spot'})")

    async def _ping_loop(self, is_futures: bool = False) -> None:
        """Send ping messages to keep connection alive."""
        ws = self._ws_futures if is_futures else self._ws
        while self._running and ws:
            try:
                ping_msg = json.dumps(
                    {"id": str(int(time.time() * 1000)), "type": "ping"}
                )
                await ws.send(ping_msg)
                await asyncio.sleep(30)
            except Exception as e:
                logger.warning(f"Ping failed: {e}")
                break

    async def _listen_loop(self, is_futures: bool = False) -> None:
        """Listen for messages."""
        ws = self._ws_futures if is_futures else self._ws
        while self._running and ws:
            try:
                message = await ws.recv()
                data = json.loads(message)

                msg_type = data.get("type")
                if msg_type == "pong":
                    continue
                elif msg_type == "welcome":
                    logger.info("WebSocket welcome received")
                    continue
                elif msg_type == "ack":
                    continue
                elif msg_type == "message":
                    topic = data.get("topic", "")
                    ws_msg = WsMessage(
                        topic=topic,
                        type=data.get("type", ""),
                        data=data.get("data", {}),
                        subject=data.get("subject", ""),
                        time=int(time.time() * 1000),
                    )

                    # Call registered callbacks
                    for pattern, callback in self._subscriptions.items():
                        if pattern in topic:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(ws_msg)
                                else:
                                    callback(ws_msg)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Listen error: {e}")
                await asyncio.sleep(1)

    async def subscribe(
        self,
        topic: str,
        callback: Callable[[WsMessage], None],
        is_futures: bool = False,
        private: bool = False,
    ) -> None:
        """Subscribe to a topic."""
        ws = self._ws_futures if is_futures else self._ws
        if not ws:
            await self.connect(
                WsType.PRIVATE if private else WsType.PUBLIC, is_futures
            )
            ws = self._ws_futures if is_futures else self._ws

        sub_msg = json.dumps(
            {
                "id": str(int(time.time() * 1000)),
                "type": "subscribe",
                "topic": topic,
                "privateChannel": private,
                "response": True,
            }
        )

        await ws.send(sub_msg)
        self._subscriptions[topic] = callback
        logger.info(f"Subscribed to {topic}")

    async def unsubscribe(self, topic: str, is_futures: bool = False) -> None:
        """Unsubscribe from a topic."""
        ws = self._ws_futures if is_futures else self._ws
        if not ws:
            return

        unsub_msg = json.dumps(
            {
                "id": str(int(time.time() * 1000)),
                "type": "unsubscribe",
                "topic": topic,
                "response": True,
            }
        )

        await ws.send(unsub_msg)
        self._subscriptions.pop(topic, None)
        logger.info(f"Unsubscribed from {topic}")

    async def subscribe_ticker(
        self, symbol: str, callback: Callable[[WsMessage], None], is_futures: bool = False
    ) -> None:
        """Subscribe to ticker updates."""
        if is_futures:
            topic = f"/contractMarket/tickerV2:{symbol}"
        else:
            topic = f"/market/ticker:{symbol}"
        await self.subscribe(topic, callback, is_futures)

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[WsMessage], None], is_futures: bool = False
    ) -> None:
        """Subscribe to order book updates."""
        if is_futures:
            topic = f"/contractMarket/level2:{symbol}"
        else:
            topic = f"/market/level2:{symbol}"
        await self.subscribe(topic, callback, is_futures)

    async def subscribe_trades(
        self, symbol: str, callback: Callable[[WsMessage], None], is_futures: bool = False
    ) -> None:
        """Subscribe to trade updates."""
        if is_futures:
            topic = f"/contractMarket/execution:{symbol}"
        else:
            topic = f"/market/match:{symbol}"
        await self.subscribe(topic, callback, is_futures)

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[WsMessage], None],
    ) -> None:
        """Subscribe to candlestick updates (spot only)."""
        topic = f"/market/candles:{symbol}_{interval}"
        await self.subscribe(topic, callback)

    async def subscribe_orders(
        self, callback: Callable[[WsMessage], None], is_futures: bool = False
    ) -> None:
        """Subscribe to order updates (private)."""
        if is_futures:
            topic = "/contractMarket/tradeOrders"
        else:
            topic = "/spotMarket/tradeOrders"
        await self.subscribe(topic, callback, is_futures, private=True)

    async def subscribe_balance(
        self, callback: Callable[[WsMessage], None]
    ) -> None:
        """Subscribe to balance updates (private)."""
        topic = "/account/balance"
        await self.subscribe(topic, callback, private=True)

    async def subscribe_position(
        self, symbol: str, callback: Callable[[WsMessage], None]
    ) -> None:
        """Subscribe to position updates (futures, private)."""
        topic = f"/contract/position:{symbol}"
        await self.subscribe(topic, callback, is_futures=True, private=True)

    async def close(self) -> None:
        """Close WebSocket connections."""
        self._running = False

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._ws_futures:
            await self._ws_futures.close()
            self._ws_futures = None

        await super().close()
        logger.info("WebSocket connections closed")
