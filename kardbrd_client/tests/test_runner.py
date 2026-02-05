"""Unit tests for runner.py - standalone AI conversation functions."""

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from kardbrd_client.runner import (
    _post_error_comment,
    _process_anthropic_tool_calls,
    _process_gemini_tool_calls,
    build_anthropic_messages,
    build_anthropic_tools,
    run_anthropic_conversation,
    run_gemini_cli_conversation,
    run_gemini_conversation,
    run_mistral_vibe_conversation,
    run_trigger,
)


@dataclass
class MockSubscription:
    """Mock BoardSubscription for testing."""

    board_id: str = "test-board-123"
    api_url: str = "http://localhost:8000"
    bot_token: str = "test-token"
    ai_provider: str = "anthropic_api"
    ai_model: str = "claude-3-5-sonnet"
    api_key: str = "sk-test-key"


@dataclass
class MockTriggerContext:
    """Mock TriggerContext for testing."""

    activity_id: str = "act-123"
    board_id: str = "board-123"
    card_id: str = "card-456"
    entity_id: str = "entity-789"
    entity_type: str = "comment"
    action: str = "commented"
    user: dict = None
    created_at: str = "2025-01-01T00:00:00"
    description: str = "Test activity"
    comment_content: str = "Test comment"
    subscription: MockSubscription = None

    def __post_init__(self):
        if self.user is None:
            self.user = {"display_name": "TestUser", "email": "test@example.com"}

    @property
    def user_name(self) -> str:
        return self.user.get("display_name", "Unknown")


class TestBuildAnthropicTools:
    """Tests for build_anthropic_tools()."""

    def test_builds_tools_without_web_search(self):
        """Tools are built in Anthropic format."""
        tools = build_anthropic_tools(web_search_enabled=False)
        assert isinstance(tools, list)
        assert len(tools) > 0
        # Each tool should have name, description, input_schema
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_builds_tools_with_web_search(self):
        """Web search tool is added when enabled."""
        tools = build_anthropic_tools(web_search_enabled=True)
        web_search_tools = [t for t in tools if t.get("name") == "web_search"]
        assert len(web_search_tools) == 1
        assert web_search_tools[0]["type"] == "web_search_20250305"

    def test_web_search_not_included_by_default(self):
        """Web search is not included when disabled."""
        tools = build_anthropic_tools(web_search_enabled=False)
        web_search_tools = [t for t in tools if t.get("name") == "web_search"]
        assert len(web_search_tools) == 0


class TestBuildAnthropicMessages:
    """Tests for build_anthropic_messages()."""

    def test_converts_user_messages(self):
        """User messages are converted correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        result = build_anthropic_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_converts_assistant_messages(self):
        """Assistant messages are converted correctly."""
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = build_anthropic_messages(messages)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_converts_mixed_messages(self):
        """Mixed user/assistant messages are converted."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = build_anthropic_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_empty_messages(self):
        """Empty message list returns empty list."""
        result = build_anthropic_messages([])
        assert result == []


class TestPostErrorComment:
    """Tests for _post_error_comment()."""

    def test_posts_basic_error(self):
        """Basic error is posted to card."""
        mock_client = Mock()
        _post_error_comment(
            client=mock_client,
            card_id="card-123",
            error_message="Something went wrong",
        )
        mock_client.add_comment.assert_called_once()
        call_args = mock_client.add_comment.call_args
        assert call_args[0][0] == "card-123"
        assert "Something went wrong" in call_args[0][1]
        assert "⚠️ **Agent Error**" in call_args[0][1]

    def test_posts_error_with_user_tag(self):
        """Error with user name includes @mention."""
        mock_client = Mock()
        _post_error_comment(
            client=mock_client,
            card_id="card-123",
            error_message="Error occurred",
            user_name="JohnDoe",
        )
        call_args = mock_client.add_comment.call_args
        assert "@JohnDoe" in call_args[0][1]

    def test_posts_error_with_tool_details(self):
        """Error with tool info includes technical details."""
        mock_client = Mock()
        _post_error_comment(
            client=mock_client,
            card_id="card-123",
            error_message="Tool failed",
            tool_name="get_card",
            tool_input={"card_id": "abc"},
        )
        call_args = mock_client.add_comment.call_args
        comment = call_args[0][1]
        assert "<details>" in comment
        assert "get_card" in comment
        assert "abc" in comment

    def test_handles_posting_failure(self):
        """Failure to post doesn't raise exception."""
        mock_client = Mock()
        mock_client.add_comment.side_effect = Exception("Network error")
        # Should not raise
        _post_error_comment(
            client=mock_client,
            card_id="card-123",
            error_message="Test error",
        )


class TestProcessAnthropicToolCalls:
    """Tests for _process_anthropic_tool_calls()."""

    def test_successful_tool_execution(self):
        """Successful tool execution updates messages."""
        mock_tool_executor = Mock()
        mock_tool_executor.execute.return_value = "Tool result"
        mock_client = Mock()

        tool_block = Mock()
        tool_block.name = "get_card"
        tool_block.input = {"card_id": "123"}
        tool_block.id = "tool-use-1"

        anthropic_messages = []
        _process_anthropic_tool_calls(
            tool_use_blocks=[tool_block],
            anthropic_messages=anthropic_messages,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        assert len(anthropic_messages) == 1
        assert anthropic_messages[0]["role"] == "user"
        content = anthropic_messages[0]["content"][0]
        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "tool-use-1"
        assert content["content"] == "Tool result"

    def test_tool_execution_failure(self):
        """Tool failure adds error result and posts comment."""
        mock_tool_executor = Mock()
        mock_tool_executor.execute.side_effect = Exception("Tool error")
        mock_client = Mock()

        tool_block = Mock()
        tool_block.name = "get_card"
        tool_block.input = {"card_id": "123"}
        tool_block.id = "tool-use-1"

        anthropic_messages = []
        _process_anthropic_tool_calls(
            tool_use_blocks=[tool_block],
            anthropic_messages=anthropic_messages,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
            user_name="TestUser",
        )

        # Error result added to messages
        assert len(anthropic_messages) == 1
        content = anthropic_messages[0]["content"][0]
        assert content["is_error"] is True
        assert "Tool error" in content["content"]

        # Error comment posted
        mock_client.add_comment.assert_called_once()

    def test_multiple_tool_calls(self):
        """Multiple tool calls are all processed."""
        mock_tool_executor = Mock()
        mock_tool_executor.execute.return_value = "Result"
        mock_client = Mock()

        tool_blocks = []
        for i in range(3):
            block = Mock()
            block.name = f"tool_{i}"
            block.input = {}
            block.id = f"tool-use-{i}"
            tool_blocks.append(block)

        anthropic_messages = []
        _process_anthropic_tool_calls(
            tool_use_blocks=tool_blocks,
            anthropic_messages=anthropic_messages,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        assert len(anthropic_messages) == 3
        assert mock_tool_executor.execute.call_count == 3


class TestProcessGeminiToolCalls:
    """Tests for _process_gemini_tool_calls()."""

    def test_successful_tool_execution(self):
        """Successful tool execution returns proper response."""
        mock_tool_executor = Mock()
        mock_tool_executor.execute.return_value = "Result"
        mock_client = Mock()

        tool_calls = [{"name": "get_card", "args": {"card_id": "123"}}]
        responses = _process_gemini_tool_calls(
            tool_calls=tool_calls,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        assert len(responses) == 1
        assert responses[0]["functionResponse"]["name"] == "get_card"
        assert responses[0]["functionResponse"]["response"]["result"] == "Result"

    def test_tool_execution_failure(self):
        """Tool failure returns error response."""
        mock_tool_executor = Mock()
        mock_tool_executor.execute.side_effect = Exception("Failed")
        mock_client = Mock()

        tool_calls = [{"name": "get_card", "args": {}}]
        responses = _process_gemini_tool_calls(
            tool_calls=tool_calls,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        assert len(responses) == 1
        assert "error" in responses[0]["functionResponse"]["response"]
        mock_client.add_comment.assert_called_once()

    def test_empty_args_defaults_to_empty_dict(self):
        """Tool call without args uses empty dict."""
        mock_tool_executor = Mock()
        mock_tool_executor.execute.return_value = "OK"
        mock_client = Mock()

        tool_calls = [{"name": "list_cards"}]  # No args key
        _process_gemini_tool_calls(
            tool_calls=tool_calls,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        mock_tool_executor.execute.assert_called_with("list_cards", {})


class TestRunTrigger:
    """Tests for run_trigger() main entry point."""

    def test_no_subscription_logs_error(self):
        """No subscription in context logs error and returns."""
        context = MockTriggerContext(subscription=None)
        # Should not raise, just log error
        run_trigger(context, "System prompt", [{"role": "user", "content": "Hi"}])

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_anthropic_provider_calls_anthropic_conversation(
        self, mock_executor_cls, mock_client_cls, mock_run_anthropic
    ):
        """Anthropic provider routes to run_anthropic_conversation."""
        subscription = MockSubscription(ai_provider="anthropic_api")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_run_anthropic.assert_called_once()

    @patch("kardbrd_client.runner.run_gemini_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_gemini_api_provider_calls_gemini_conversation(self, mock_executor_cls, mock_client_cls, mock_run_gemini):
        """Gemini API provider routes to run_gemini_conversation."""
        subscription = MockSubscription(ai_provider="gemini_api")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_run_gemini.assert_called_once()

    @patch("kardbrd_client.runner.run_gemini_cli_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_gemini_cli_provider_calls_gemini_cli(self, mock_executor_cls, mock_client_cls, mock_run_gemini_cli):
        """Gemini CLI provider routes to run_gemini_cli_conversation."""
        subscription = MockSubscription(ai_provider="gemini_cli")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_run_gemini_cli.assert_called_once()

    @patch("kardbrd_client.runner.run_mistral_vibe_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_mistral_vibe_provider_calls_mistral_vibe(self, mock_executor_cls, mock_client_cls, mock_run_mistral):
        """Mistral vibe provider routes to run_mistral_vibe_conversation."""
        subscription = MockSubscription(ai_provider="mistral_vibe")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_run_mistral.assert_called_once()

    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_unknown_provider_logs_error(self, mock_executor_cls, mock_client_cls):
        """Unknown AI provider logs error."""
        subscription = MockSubscription(ai_provider="unknown_provider")
        context = MockTriggerContext(subscription=subscription)

        # Should not raise
        run_trigger(context, "System", [])

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_conversation_error_posts_comment(self, mock_executor_cls, mock_client_cls, mock_run_anthropic):
        """Error during conversation posts error comment."""
        mock_run_anthropic.side_effect = Exception("API error")
        mock_client = Mock()
        mock_client_cls.return_value = mock_client

        subscription = MockSubscription(ai_provider="anthropic_api")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_client.add_comment.assert_called_once()
        call_args = mock_client.add_comment.call_args
        assert "API error" in call_args[0][1]

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_client_is_closed_after_run(self, mock_executor_cls, mock_client_cls, mock_run_anthropic):
        """KardbrdClient is closed in finally block."""
        mock_client = Mock()
        mock_client_cls.return_value = mock_client

        subscription = MockSubscription(ai_provider="anthropic_api")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_client.close.assert_called_once()

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_client_closed_even_on_error(self, mock_executor_cls, mock_client_cls, mock_run_anthropic):
        """Client is closed even when conversation raises."""
        mock_run_anthropic.side_effect = Exception("Error")
        mock_client = Mock()
        mock_client_cls.return_value = mock_client

        subscription = MockSubscription(ai_provider="anthropic_api")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_client.close.assert_called_once()


class TestRunAnthropicConversation:
    """Tests for run_anthropic_conversation()."""

    def test_no_subscription_returns_early(self):
        """No subscription logs error and returns."""
        context = MockTriggerContext(subscription=None)
        mock_client = Mock()
        mock_executor = Mock()

        # Should not raise
        run_anthropic_conversation(context, "System", [], mock_client, mock_executor)

    def test_no_api_key_returns_early(self):
        """No API key logs error and returns."""
        subscription = MockSubscription(api_key="")
        context = MockTriggerContext(subscription=subscription)
        mock_client = Mock()
        mock_executor = Mock()

        run_anthropic_conversation(context, "System", [], mock_client, mock_executor)

    def test_empty_response_breaks_loop(self):
        """Empty response from API breaks the loop."""
        # Patch anthropic at import time
        import sys

        mock_anthropic_module = MagicMock()
        mock_anthropic_client = Mock()
        mock_response = Mock()
        mock_response.content = []  # Empty content
        mock_anthropic_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_anthropic_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            subscription = MockSubscription()
            context = MockTriggerContext(subscription=subscription)

            run_anthropic_conversation(context, "System", [], Mock(), Mock())

            # Should only call once then break
            assert mock_anthropic_client.messages.create.call_count == 1

    def test_text_response_appends_to_messages(self):
        """Text-only response is appended to messages."""
        import sys

        mock_anthropic_module = MagicMock()
        mock_anthropic_client = Mock()
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Hello!"

        mock_response = Mock()
        mock_response.content = [text_block]
        mock_anthropic_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_anthropic_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            subscription = MockSubscription()
            context = MockTriggerContext(subscription=subscription)
            messages = []

            run_anthropic_conversation(context, "System", messages, Mock(), Mock())

            assert len(messages) == 1
            assert messages[0]["content"] == "Hello!"


class TestRunGeminiConversation:
    """Tests for run_gemini_conversation()."""

    def test_no_subscription_returns_early(self):
        """No subscription logs error and returns."""
        context = MockTriggerContext(subscription=None)

        run_gemini_conversation(context, "System", [], Mock(), Mock())

    def test_no_api_key_returns_early(self):
        """No API key logs error and returns."""
        subscription = MockSubscription(api_key="")
        context = MockTriggerContext(subscription=subscription)

        run_gemini_conversation(context, "System", [], Mock(), Mock())

    @patch("kardbrd_client.runner.httpx.Client")
    def test_no_candidates_breaks_loop(self, mock_httpx_client_cls):
        """No candidates in response breaks loop."""
        mock_response = Mock()
        mock_response.json.return_value = {"candidates": []}
        mock_response.raise_for_status = Mock()

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_http_client.__enter__ = Mock(return_value=mock_http_client)
        mock_http_client.__exit__ = Mock(return_value=False)
        mock_httpx_client_cls.return_value = mock_http_client

        subscription = MockSubscription(ai_provider="gemini_api")
        context = MockTriggerContext(subscription=subscription)

        run_gemini_conversation(context, "System", [], Mock(), Mock())

        assert mock_http_client.post.call_count == 1

    @patch("kardbrd_client.runner.httpx.Client")
    def test_text_response_appends_to_messages(self, mock_httpx_client_cls):
        """Text response is appended to messages."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from Gemini!"}],
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_http_client.__enter__ = Mock(return_value=mock_http_client)
        mock_http_client.__exit__ = Mock(return_value=False)
        mock_httpx_client_cls.return_value = mock_http_client

        subscription = MockSubscription(ai_provider="gemini_api")
        context = MockTriggerContext(subscription=subscription)
        messages = []

        run_gemini_conversation(context, "System", messages, Mock(), Mock())

        assert len(messages) == 1
        assert messages[0]["content"] == "Hello from Gemini!"


class TestRunGeminiCliConversation:
    """Tests for run_gemini_cli_conversation()."""

    def test_no_subscription_returns_early(self):
        """No subscription logs error and returns."""
        context = MockTriggerContext(subscription=None)

        run_gemini_cli_conversation(context, "System", [], None)

    @patch("kardbrd_client.runner.subprocess.run")
    def test_mcp_registration_failure_continues(self, mock_subprocess):
        """MCP registration failure doesn't stop execution."""
        from subprocess import CalledProcessError

        # First call (mcp add) fails, second call (gemini) succeeds
        mock_subprocess.side_effect = [
            CalledProcessError(1, "gemini", stderr=b"MCP error"),
            Mock(returncode=0, stdout="Response", stderr=""),
        ]

        subscription = MockSubscription(ai_provider="gemini_cli")
        context = MockTriggerContext(subscription=subscription)
        messages = []

        run_gemini_cli_conversation(context, "System", messages, subscription)

        # Both subprocess calls were made
        assert mock_subprocess.call_count == 2

    @patch("kardbrd_client.runner.subprocess.run")
    def test_successful_response_appends_to_messages(self, mock_subprocess):
        """Successful CLI response is appended to messages."""
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # MCP registration
            Mock(returncode=0, stdout="CLI response\n", stderr=""),  # gemini run
        ]

        subscription = MockSubscription(ai_provider="gemini_cli")
        context = MockTriggerContext(subscription=subscription)
        messages = []

        run_gemini_cli_conversation(context, "System", messages, subscription)

        assert len(messages) == 1
        assert messages[0]["content"] == "CLI response"

    @patch("kardbrd_client.runner.subprocess.run")
    def test_cli_failure_appends_error(self, mock_subprocess):
        """CLI failure appends error to messages."""
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # MCP registration
            Mock(returncode=1, stdout="", stderr="CLI error"),  # gemini run fails
        ]

        subscription = MockSubscription(ai_provider="gemini_cli")
        context = MockTriggerContext(subscription=subscription)
        messages = []

        run_gemini_cli_conversation(context, "System", messages, subscription)

        assert len(messages) == 1
        assert "Error" in messages[0]["content"]

    @patch("kardbrd_client.runner.subprocess.run")
    def test_timeout_appends_error(self, mock_subprocess):
        """Timeout appends error to messages."""
        import subprocess

        mock_subprocess.side_effect = [
            Mock(returncode=0),  # MCP registration
            subprocess.TimeoutExpired("gemini", 600),  # Timeout
        ]

        subscription = MockSubscription(ai_provider="gemini_cli")
        context = MockTriggerContext(subscription=subscription)
        messages = []

        run_gemini_cli_conversation(context, "System", messages, subscription)

        assert len(messages) == 1
        assert "timed out" in messages[0]["content"]


class TestRunMistralVibeConversation:
    """Tests for run_mistral_vibe_conversation()."""

    def test_no_subscription_returns_early(self):
        """No subscription logs error and returns."""
        context = MockTriggerContext(subscription=None)

        run_mistral_vibe_conversation(context, "System", [], None)

    def test_no_api_key_returns_early(self):
        """No API key logs error and returns."""
        subscription = MockSubscription(api_key="")
        context = MockTriggerContext(subscription=subscription)

        run_mistral_vibe_conversation(context, "System", [], subscription)

    @patch("kardbrd_client.runner.subprocess.run")
    def test_successful_list_response(self, mock_subprocess):
        """Successful list response extracts assistant content."""
        import json

        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps([{"role": "assistant", "content": "Vibe response"}]),
            stderr="",
        )

        subscription = MockSubscription(ai_provider="mistral_vibe")
        context = MockTriggerContext(subscription=subscription)
        messages = [{"role": "user", "content": "Hello"}]

        run_mistral_vibe_conversation(context, "System", messages, subscription)

        assert any(m.get("content") == "Vibe response" for m in messages)

    @patch("kardbrd_client.runner.subprocess.run")
    def test_successful_dict_response(self, mock_subprocess):
        """Successful dict response extracts content."""
        import json

        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps({"content": "Dict response"}),
            stderr="",
        )

        subscription = MockSubscription(ai_provider="mistral_vibe")
        context = MockTriggerContext(subscription=subscription)
        messages = [{"role": "user", "content": "Hello"}]

        run_mistral_vibe_conversation(context, "System", messages, subscription)

        assert any(m.get("content") == "Dict response" for m in messages)

    @patch("kardbrd_client.runner.subprocess.run")
    def test_cli_failure_raises(self, mock_subprocess):
        """CLI failure raises RuntimeError."""
        mock_subprocess.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Vibe error",
        )

        subscription = MockSubscription(ai_provider="mistral_vibe")
        context = MockTriggerContext(subscription=subscription)

        with pytest.raises(RuntimeError, match="Vibe CLI failed"):
            run_mistral_vibe_conversation(context, "System", [{"role": "user", "content": "Hi"}], subscription)

    @patch("kardbrd_client.runner.subprocess.run")
    def test_json_parse_error_logs_raw_output(self, mock_subprocess):
        """JSON parse error logs raw output."""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Not valid JSON",
            stderr="",
        )

        subscription = MockSubscription(ai_provider="mistral_vibe")
        context = MockTriggerContext(subscription=subscription)
        messages = [{"role": "user", "content": "Hello"}]

        # Should not raise, just log error
        run_mistral_vibe_conversation(context, "System", messages, subscription)

    @patch("kardbrd_client.runner.subprocess.run")
    def test_temp_file_cleaned_up(self, mock_subprocess):
        """Temporary tools file is cleaned up after execution."""
        import json

        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps({"content": "OK"}),
            stderr="",
        )

        subscription = MockSubscription(ai_provider="mistral_vibe")
        context = MockTriggerContext(subscription=subscription)

        run_mistral_vibe_conversation(context, "System", [{"role": "user", "content": "Hi"}], subscription)

        # Verify subprocess was called (temp file was used)
        assert mock_subprocess.called
        # The temp file should be cleaned up (can't easily verify without access to the path)


class TestRunTriggerWebSearchParam:
    """Tests for web_search_enabled parameter in run_trigger."""

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_web_search_passed_to_anthropic(self, mock_executor_cls, mock_client_cls, mock_run_anthropic):
        """web_search_enabled is passed to Anthropic conversation."""
        subscription = MockSubscription(ai_provider="anthropic_api")
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [], web_search_enabled=True)

        call_args = mock_run_anthropic.call_args
        assert call_args[0][5] is True  # web_search_enabled is 6th positional arg


class TestRunTriggerCreatesResources:
    """Tests that run_trigger creates its own KardbrdClient and ToolExecutor."""

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_creates_client_from_subscription(self, mock_executor_cls, mock_client_cls, mock_run_anthropic):
        """KardbrdClient is created from subscription data."""
        subscription = MockSubscription(
            api_url="http://test-api.com",
            bot_token="my-token",
        )
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_client_cls.assert_called_once_with(
            base_url="http://test-api.com",
            token="my-token",
        )

    @patch("kardbrd_client.runner.run_anthropic_conversation")
    @patch("kardbrd_client.runner.KardbrdClient")
    @patch("kardbrd_client.runner.ToolExecutor")
    def test_creates_executor_from_client(self, mock_executor_cls, mock_client_cls, mock_run_anthropic):
        """ToolExecutor is created with the new client."""
        mock_client = Mock()
        mock_client_cls.return_value = mock_client

        subscription = MockSubscription()
        context = MockTriggerContext(subscription=subscription)

        run_trigger(context, "System", [])

        mock_executor_cls.assert_called_once_with(mock_client)


class TestRunnerIndependence:
    """Test that runner.py has no dependencies on agent.py."""

    def test_runner_does_not_import_agent(self):
        """runner.py should not import anything from agent.py."""
        runner_path = Path(__file__).parent.parent / "runner.py"

        assert runner_path.exists(), "runner.py should exist"

        source = runner_path.read_text()
        tree = ast.parse(source)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # runner.py should NOT import from agent
        agent_imports = [i for i in imports if "agent" in i.lower()]
        assert not agent_imports, f"runner.py should not import from agent.py, found: {agent_imports}"

    def test_runner_only_imports_allowed_local_modules(self):
        """runner.py should only import from client, tools, triggers (not agent)."""
        runner_path = Path(__file__).parent.parent / "runner.py"

        assert runner_path.exists(), "runner.py should exist"

        source = runner_path.read_text()
        tree = ast.parse(source)

        local_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("."):
                    local_imports.append(node.module)

        # Allowed local imports (no agent)
        allowed = {".client", ".tools", ".triggers"}
        for imp in local_imports:
            assert imp in allowed, f"Unexpected local import in runner.py: {imp}"

    def test_runner_has_no_base_agent_code_references(self):
        """runner.py should not use BaseAgent class in code (comments/docstrings OK)."""
        runner_path = Path(__file__).parent.parent / "runner.py"

        assert runner_path.exists(), "runner.py should exist"

        source = runner_path.read_text()
        tree = ast.parse(source)

        # Check for actual code usage of BaseAgent/MultiBaseAgent
        # This includes: class inheritance, function calls, attribute access, names
        base_agent_refs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ("BaseAgent", "MultiBaseAgent"):
                base_agent_refs.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in ("BaseAgent", "MultiBaseAgent"):
                base_agent_refs.append(node.attr)

        assert not base_agent_refs, f"runner.py should not use BaseAgent/MultiBaseAgent in code: {base_agent_refs}"

    def test_triggers_type_checks_agent_only(self):
        """triggers.py should only TYPE_CHECK import from agent.py, not runtime import."""
        triggers_path = Path(__file__).parent.parent / "triggers.py"

        assert triggers_path.exists(), "triggers.py should exist"

        source = triggers_path.read_text()
        tree = ast.parse(source)

        # Find all ImportFrom nodes
        runtime_agent_imports = []
        type_checking_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "agent" in node.module:
                    # Check if this import is inside TYPE_CHECKING block
                    # We do this by looking at the source lines
                    import_line = source.split("\n")[node.lineno - 1]
                    # Check context - if we're after TYPE_CHECKING:
                    lines_before = source.split("\n")[: node.lineno]
                    in_type_checking = False
                    indent_level = len(import_line) - len(import_line.lstrip())

                    for line in reversed(lines_before):
                        if "TYPE_CHECKING" in line and "if" in line:
                            in_type_checking = True
                            break
                        # If we hit a non-indented line that's not empty, we're not in TYPE_CHECKING
                        stripped = line.strip()
                        if stripped and not line.startswith(" " * indent_level) and "import" not in line:
                            break

                    if in_type_checking:
                        type_checking_imports.append(node.module)
                    else:
                        runtime_agent_imports.append(node.module)

        # Runtime imports from agent should be empty
        assert not runtime_agent_imports, f"triggers.py has runtime imports from agent: {runtime_agent_imports}"
