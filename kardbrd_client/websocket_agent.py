"""
WebSocket client for AI agents to receive real-time events.

This module provides WebSocket-based connectivity for agents,
replacing APScheduler polling with real-time event handling.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

logger = logging.getLogger(__name__)


@dataclass
class OnboardingEvent:
    """Event data for onboarding-related messages."""

    session_id: str
    message_id: str | None
    content: str
    user_id: str | None

    @classmethod
    def from_dict(cls, data: dict) -> "OnboardingEvent":
        """Create an OnboardingEvent from a dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            message_id=data.get("message_id"),
            content=data.get("content", ""),
            user_id=data.get("user_id"),
        )


class WebSocketAgentConnection:
    """
    Manages WebSocket connection for agents.

    Provides real-time event handling for:
    - Onboarding user messages
    - Board events
    - Session subscriptions

    Uses exponential backoff for reconnection.
    """

    def __init__(self, base_url: str, token: str):
        """
        Initialize WebSocket agent connection.

        Args:
            base_url: The base HTTP URL (will be converted to WS)
            token: Bot authentication token
        """
        # Convert HTTP URL to WebSocket URL
        self.ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{self.ws_url}/ws/agent/?token={token}"
        self.token = token

        self.connection: websockets.WebSocketClientProtocol | None = None
        self.reconnect_delay = 1  # Initial delay in seconds
        self.max_reconnect_delay = 60  # Maximum delay
        self.min_reconnect_delay = 1  # Minimum delay

        # Event handlers
        self.handlers: dict[str, list[Callable]] = {}

        # Subscribed sessions
        self.subscribed_sessions: set[str] = set()

        # Connection state
        self._connected = False
        self._running = False
        self._agent_id: str | None = None
        self._subscribed_boards: list[str] = []

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self.connection is not None

    def on(self, event_type: str):
        """
        Decorator to register event handlers.

        Args:
            event_type: The type of event to handle (e.g., 'onboarding_user_message')

        Example:
            @connection.on('onboarding_user_message')
            async def handle_message(event):
                print(f"User said: {event.content}")
        """

        def decorator(func: Callable):
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(func)
            return func

        return decorator

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler programmatically.

        Args:
            event_type: The type of event to handle
            handler: The handler function (async or sync)
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def connect(self) -> None:
        """
        Establish WebSocket connection with automatic reconnection.

        This method will keep trying to connect and reconnect indefinitely
        until stop() is called.
        """
        self._running = True

        while self._running:
            try:
                logger.info(f"Connecting to {self.ws_url.split('?')[0]}...")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self.connection = ws
                    self._connected = True
                    self.reconnect_delay = self.min_reconnect_delay  # Reset on success
                    logger.info("WebSocket connected")

                    # Wait for connection confirmation
                    await self._handle_connection_response()

                    # Re-subscribe to any sessions we were subscribed to before disconnect
                    for session_id in list(self.subscribed_sessions):
                        await self._send_subscribe_onboarding(session_id)

                    # Start listening for messages
                    await self._listen()

            except ConnectionClosedError as e:
                self._connected = False
                self.connection = None
                logger.warning(f"WebSocket connection closed: {e}")
                await self._handle_reconnect()

            except ConnectionClosed:
                self._connected = False
                self.connection = None
                logger.warning("WebSocket connection closed")
                await self._handle_reconnect()

            except Exception as e:
                self._connected = False
                self.connection = None
                logger.error(f"WebSocket error: {e}")
                await self._handle_reconnect()

    async def _handle_connection_response(self) -> None:
        """Handle the initial connection response from server."""
        if not self.connection:
            return

        try:
            # Wait for initial 'connected' message
            raw_message = await asyncio.wait_for(self.connection.recv(), timeout=10)
            message = json.loads(raw_message)

            if message.get("type") == "connected":
                self._agent_id = message.get("agent_id")
                self._subscribed_boards = message.get("subscribed_boards", [])
                logger.info(f"Agent {self._agent_id} connected, subscribed to {len(self._subscribed_boards)} boards")
        except TimeoutError:
            logger.warning("Timeout waiting for connection confirmation")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse connection response: {e}")

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if not self._running:
            return

        logger.info(f"Reconnecting in {self.reconnect_delay}s...")
        await asyncio.sleep(self.reconnect_delay)

        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def _listen(self) -> None:
        """Listen for incoming messages."""
        if not self.connection:
            return

        async for raw_message in self.connection:
            try:
                message = json.loads(raw_message)
                await self._dispatch_message(message)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")

    async def _dispatch_message(self, message: dict) -> None:
        """
        Dispatch a message to registered handlers.

        Args:
            message: The parsed JSON message
        """
        msg_type = message.get("type")
        if not msg_type:
            logger.warning(f"Message without type: {message}")
            return

        # Handle built-in message types
        if msg_type == "pong":
            return  # Ping/pong handled by websockets library

        if msg_type == "error":
            logger.error(f"Server error: {message.get('message')}")
            return

        if msg_type in ("subscribed_onboarding", "unsubscribed_onboarding"):
            logger.debug(f"Subscription confirmed: {msg_type} {message.get('session_id')}")
            return

        # Dispatch to registered handlers (non-blocking for async handlers)
        handlers = self.handlers.get(msg_type, [])
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                # Run async handlers as background tasks for concurrency
                task = asyncio.create_task(handler(message))
                task.add_done_callback(
                    lambda t, mt=msg_type: t.exception() and logger.error(f"Handler error for {mt}: {t.exception()}")
                )
            else:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {msg_type}: {e}")

    async def subscribe_onboarding(self, session_id: str) -> None:
        """
        Subscribe to an onboarding session's events.

        Args:
            session_id: The onboarding session public_id
        """
        self.subscribed_sessions.add(session_id)
        if self.is_connected:
            await self._send_subscribe_onboarding(session_id)

    async def _send_subscribe_onboarding(self, session_id: str) -> None:
        """Send the subscribe message to server."""
        if self.connection:
            await self.connection.send(
                json.dumps(
                    {
                        "type": "subscribe_onboarding",
                        "session_id": session_id,
                    }
                )
            )

    async def unsubscribe_onboarding(self, session_id: str) -> None:
        """
        Unsubscribe from an onboarding session.

        Args:
            session_id: The onboarding session public_id
        """
        self.subscribed_sessions.discard(session_id)
        if self.connection:
            await self.connection.send(
                json.dumps(
                    {
                        "type": "unsubscribe_onboarding",
                        "session_id": session_id,
                    }
                )
            )

    async def send_stream_start(
        self,
        session_id: str,
        agent_role: str = "planner",
    ) -> None:
        """
        Send a stream start signal to the onboarding conversation.

        Args:
            session_id: The onboarding session public_id
            agent_role: The role of the agent (planner, refiner, etc.)
        """
        if not self.connection:
            logger.warning("Cannot send stream start: not connected")
            return

        await self.connection.send(
            json.dumps(
                {
                    "type": "stream_start",
                    "session_id": session_id,
                    "agent_role": agent_role,
                }
            )
        )

    async def send_stream_chunk(
        self,
        session_id: str,
        chunk: str,
        sequence: int,
        agent_role: str = "planner",
    ) -> None:
        """
        Send a streaming chunk to the onboarding conversation.

        Args:
            session_id: The onboarding session public_id
            chunk: The text chunk to stream
            sequence: Sequence number for ordering
            agent_role: The role of the agent (planner, refiner, etc.)
        """
        if not self.connection:
            logger.warning("Cannot send stream chunk: not connected")
            return

        await self.connection.send(
            json.dumps(
                {
                    "type": "stream_chunk",
                    "session_id": session_id,
                    "chunk": chunk,
                    "sequence": sequence,
                    "agent_role": agent_role,
                }
            )
        )

    async def send_stream_end(
        self,
        session_id: str,
        agent_role: str = "planner",
    ) -> None:
        """
        Send a stream end signal to the onboarding conversation.

        Args:
            session_id: The onboarding session public_id
            agent_role: The role of the agent (planner, refiner, etc.)
        """
        if not self.connection:
            logger.warning("Cannot send stream end: not connected")
            return

        await self.connection.send(
            json.dumps(
                {
                    "type": "stream_end",
                    "session_id": session_id,
                    "agent_role": agent_role,
                }
            )
        )

    async def send_message(self, message: dict[str, Any]) -> None:
        """
        Send a raw JSON message to the server.

        Args:
            message: The message to send
        """
        if not self.connection:
            logger.warning("Cannot send message: not connected")
            return

        await self.connection.send(json.dumps(message))

    async def send_status_ping(
        self,
        subscription_info: dict[str, Any] | None = None,
        active_cards: list[str] | None = None,
    ) -> None:
        """
        Send a status ping with subscription and active card information.

        This provides visibility into what the proxy is working on.

        Args:
            subscription_info: Dict with board_id, agent_name, etc.
            active_cards: List of card IDs currently being processed
        """
        if not self.connection:
            logger.debug("Cannot send status ping: not connected")
            return

        await self.connection.send(
            json.dumps(
                {
                    "type": "status_ping",
                    "subscription": subscription_info or {},
                    "active_cards": active_cards or [],
                    "active_card_count": len(active_cards) if active_cards else 0,
                }
            )
        )

    def stop(self) -> None:
        """Stop the connection and prevent reconnection."""
        self._running = False
        if self.connection:
            asyncio.create_task(self.connection.close())

    async def close(self) -> None:
        """Close the connection gracefully."""
        self._running = False
        if self.connection:
            await self.connection.close()
        self._connected = False
        self.connection = None
