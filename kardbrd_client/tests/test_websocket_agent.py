"""Tests for WebSocket agent connection."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kardbrd_client.websocket_agent import OnboardingEvent, WebSocketAgentConnection


class TestOnboardingEvent:
    """Tests for OnboardingEvent dataclass."""

    def test_from_dict_full(self):
        """Test creating OnboardingEvent from complete dict."""
        data = {
            "session_id": "session-123",
            "message_id": "msg-456",
            "content": "Hello world",
            "user_id": "user-789",
        }
        event = OnboardingEvent.from_dict(data)

        assert event.session_id == "session-123"
        assert event.message_id == "msg-456"
        assert event.content == "Hello world"
        assert event.user_id == "user-789"

    def test_from_dict_partial(self):
        """Test creating OnboardingEvent from partial dict."""
        data = {"session_id": "session-123"}
        event = OnboardingEvent.from_dict(data)

        assert event.session_id == "session-123"
        assert event.message_id is None
        assert event.content == ""
        assert event.user_id is None

    def test_from_dict_empty(self):
        """Test creating OnboardingEvent from empty dict."""
        event = OnboardingEvent.from_dict({})

        assert event.session_id == ""
        assert event.message_id is None
        assert event.content == ""
        assert event.user_id is None


class TestWebSocketAgentConnection:
    """Tests for WebSocketAgentConnection."""

    def test_init_http_url(self):
        """Test initialization with HTTP URL."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        assert conn.ws_url == "ws://example.com/ws/agent/?token=token123"
        assert conn.token == "token123"
        assert conn.is_connected is False
        assert len(conn.handlers) == 0

    def test_init_https_url(self):
        """Test initialization with HTTPS URL."""
        conn = WebSocketAgentConnection("https://example.com", "token123")

        assert conn.ws_url == "wss://example.com/ws/agent/?token=token123"

    def test_on_decorator(self):
        """Test event handler registration via decorator."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        @conn.on("test_event")
        def handler(msg):
            pass

        assert "test_event" in conn.handlers
        assert handler in conn.handlers["test_event"]

    def test_register_handler(self):
        """Test programmatic handler registration."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        def handler(msg):
            pass

        conn.register_handler("test_event", handler)

        assert "test_event" in conn.handlers
        assert handler in conn.handlers["test_event"]

    def test_multiple_handlers_same_event(self):
        """Test registering multiple handlers for same event."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        def handler1(msg):
            pass

        def handler2(msg):
            pass

        conn.register_handler("test_event", handler1)
        conn.register_handler("test_event", handler2)

        assert len(conn.handlers["test_event"]) == 2

    def test_subscribe_onboarding_adds_to_set(self):
        """Test that subscribe_onboarding adds session to subscribed set."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Run in event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(conn.subscribe_onboarding("session-123"))
            assert "session-123" in conn.subscribed_sessions
        finally:
            loop.close()

    def test_unsubscribe_onboarding_removes_from_set(self):
        """Test that unsubscribe_onboarding removes session from set."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        conn.subscribed_sessions.add("session-123")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(conn.unsubscribe_onboarding("session-123"))
            assert "session-123" not in conn.subscribed_sessions
        finally:
            loop.close()

    def test_stop_sets_running_false(self):
        """Test that stop() sets _running to False."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        conn._running = True

        conn.stop()

        assert conn._running is False

    @pytest.mark.asyncio
    async def test_dispatch_message_calls_handler(self):
        """Test that _dispatch_message calls registered handlers."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        handler = AsyncMock()
        conn.register_handler("test_event", handler)

        await conn._dispatch_message({"type": "test_event", "data": "test"})

        handler.assert_called_once_with({"type": "test_event", "data": "test"})

    @pytest.mark.asyncio
    async def test_dispatch_message_handles_pong(self):
        """Test that pong messages are silently handled."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Should not raise
        await conn._dispatch_message({"type": "pong"})

    @pytest.mark.asyncio
    async def test_dispatch_message_handles_error(self):
        """Test that error messages are logged."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Should not raise, just log
        await conn._dispatch_message({"type": "error", "message": "test error"})

    @pytest.mark.asyncio
    async def test_dispatch_message_no_type(self):
        """Test handling message without type."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Should not raise
        await conn._dispatch_message({"data": "no type"})

    @pytest.mark.asyncio
    async def test_dispatch_message_sync_handler(self):
        """Test that sync handlers work."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        results = []

        def sync_handler(msg):
            results.append(msg)

        conn.register_handler("test_event", sync_handler)

        await conn._dispatch_message({"type": "test_event", "data": "test"})

        assert len(results) == 1
        assert results[0]["type"] == "test_event"

    @pytest.mark.asyncio
    async def test_send_stream_chunk_not_connected(self):
        """Test send_stream_chunk when not connected."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Should not raise, just log warning
        await conn.send_stream_chunk("session-123", "chunk", 0)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self):
        """Test send_message when not connected."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Should not raise, just log warning
        await conn.send_message({"type": "test"})


class TestWebSocketAgentConnectionWithMockedWebsockets:
    """Tests that mock the websockets library."""

    @pytest.mark.asyncio
    async def test_handle_connection_response_connected(self):
        """Test handling successful connection response."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps(
            {
                "type": "connected",
                "agent_id": "agent-123",
                "subscribed_boards": ["board-1", "board-2"],
            }
        )
        conn.connection = mock_ws

        await conn._handle_connection_response()

        assert conn._agent_id == "agent-123"
        assert conn._subscribed_boards == ["board-1", "board-2"]

    @pytest.mark.asyncio
    async def test_handle_connection_response_timeout(self):
        """Test handling connection response timeout."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        mock_ws = AsyncMock()
        mock_ws.recv.side_effect = TimeoutError()
        conn.connection = mock_ws

        # Should not raise
        await conn._handle_connection_response()

    @pytest.mark.asyncio
    async def test_send_stream_chunk_connected(self):
        """Test sending stream chunk when connected."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        mock_ws = AsyncMock()
        conn.connection = mock_ws
        conn._connected = True

        await conn.send_stream_chunk("session-123", "hello", 1, "planner")

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "stream_chunk"
        assert sent_data["session_id"] == "session-123"
        assert sent_data["chunk"] == "hello"
        assert sent_data["sequence"] == 1
        assert sent_data["agent_role"] == "planner"

    @pytest.mark.asyncio
    async def test_send_status_ping_connected(self):
        """Test sending status ping when connected."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        mock_ws = AsyncMock()
        conn.connection = mock_ws
        conn._connected = True

        subscription_info = {"board_id": "board-123", "agent_name": "coder"}
        active_cards = ["card-1", "card-2"]

        await conn.send_status_ping(subscription_info, active_cards)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "status_ping"
        assert sent_data["subscription"] == subscription_info
        assert sent_data["active_cards"] == active_cards
        assert sent_data["active_card_count"] == 2

    @pytest.mark.asyncio
    async def test_send_status_ping_not_connected(self):
        """Test sending status ping when not connected."""
        conn = WebSocketAgentConnection("http://example.com", "token123")

        # Should not raise, just log debug message
        await conn.send_status_ping({"board_id": "board-123"}, ["card-1"])

    @pytest.mark.asyncio
    async def test_send_status_ping_empty_values(self):
        """Test sending status ping with empty values."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        mock_ws = AsyncMock()
        conn.connection = mock_ws
        conn._connected = True

        await conn.send_status_ping(None, None)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "status_ping"
        assert sent_data["subscription"] == {}
        assert sent_data["active_cards"] == []
        assert sent_data["active_card_count"] == 0


class TestDispatchConcurrency:
    """Tests for non-blocking message dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_does_not_block_on_slow_handler(self):
        """Test that slow async handlers don't block message dispatch."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        call_times = []

        async def slow_handler(msg):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.5)  # Simulate slow work

        conn.register_handler("test_event", slow_handler)

        # Dispatch 3 messages rapidly
        start = asyncio.get_event_loop().time()
        await conn._dispatch_message({"type": "test_event", "id": 1})
        await conn._dispatch_message({"type": "test_event", "id": 2})
        await conn._dispatch_message({"type": "test_event", "id": 3})
        dispatch_time = asyncio.get_event_loop().time() - start

        # Dispatch should return immediately (not wait for handlers)
        assert dispatch_time < 0.1, f"Dispatch took {dispatch_time}s, should be instant"

        # Wait for handlers to complete
        await asyncio.sleep(0.6)

        # All 3 handlers should have been called
        assert len(call_times) == 3

        # All handlers should have started nearly simultaneously
        assert call_times[2] - call_times[0] < 0.1, "Handlers should start concurrently"

    @pytest.mark.asyncio
    async def test_dispatch_runs_handlers_concurrently(self):
        """Test that multiple async handlers run concurrently."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        running_count = []
        max_concurrent = 0

        async def counting_handler(msg):
            nonlocal max_concurrent
            running_count.append(1)
            current = sum(running_count)
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.1)
            running_count.pop()

        conn.register_handler("test_event", counting_handler)

        # Dispatch multiple messages
        for i in range(5):
            await conn._dispatch_message({"type": "test_event", "id": i})

        # Wait for all handlers to complete
        await asyncio.sleep(0.2)

        # Multiple handlers should have been running concurrently
        assert max_concurrent > 1, f"Expected concurrent handlers, but max was {max_concurrent}"

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_logged_not_raised(self):
        """Test that handler exceptions are logged but don't crash dispatch."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        call_count = 0

        async def failing_handler(msg):
            nonlocal call_count
            call_count += 1
            raise ValueError("Handler failed!")

        conn.register_handler("test_event", failing_handler)

        # Should not raise
        await conn._dispatch_message({"type": "test_event"})

        # Wait for task to complete and log error
        await asyncio.sleep(0.1)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_sync_handler_still_works(self):
        """Test that sync handlers still work correctly."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        results = []

        def sync_handler(msg):
            results.append(msg["id"])

        conn.register_handler("test_event", sync_handler)

        await conn._dispatch_message({"type": "test_event", "id": 1})
        await conn._dispatch_message({"type": "test_event", "id": 2})

        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_dispatch_mixed_sync_async_handlers(self):
        """Test dispatch with both sync and async handlers."""
        conn = WebSocketAgentConnection("http://example.com", "token123")
        sync_results = []
        async_results = []

        def sync_handler(msg):
            sync_results.append(msg["id"])

        async def async_handler(msg):
            await asyncio.sleep(0.05)
            async_results.append(msg["id"])

        conn.register_handler("test_event", sync_handler)
        conn.register_handler("test_event", async_handler)

        await conn._dispatch_message({"type": "test_event", "id": 1})

        # Sync handler runs immediately
        assert sync_results == [1]

        # Async handler runs in background
        assert async_results == []

        # Wait for async handler
        await asyncio.sleep(0.1)
        assert async_results == [1]
