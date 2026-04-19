"""Tests for VALIDATION_ERROR envelope handling across client, MCP, and runner."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from kardbrd_client.client import KardbrdAPIError


class TestKardbrdAPIErrorWithErrors:
    """Tests for KardbrdAPIError with structured field errors."""

    def test_errors_field_defaults_to_none(self):
        """errors is None by default for backward compatibility."""
        e = KardbrdAPIError("Not found", code="NOT_FOUND", status_code=404)
        assert e.errors is None

    def test_errors_field_stored(self):
        """errors dict is stored on the exception."""
        errors = {"title": [{"message": "This field is required.", "code": "required"}]}
        e = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)
        assert e.errors == errors
        assert e.code == "VALIDATION_ERROR"

    def test_str_without_errors_unchanged(self):
        """__str__ for non-validation errors is unchanged."""
        e = KardbrdAPIError("Not found", code="NOT_FOUND", status_code=404)
        s = str(e)
        assert "(code: NOT_FOUND)" in s
        assert "[HTTP 404]" in s

    def test_str_validation_error_omits_code(self):
        """__str__ omits (code: VALIDATION_ERROR) for cleaner output."""
        errors = {"title": [{"message": "This field is required.", "code": "required"}]}
        e = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)
        s = str(e)
        assert "(code: VALIDATION_ERROR)" not in s
        assert "Validation failed" in s
        assert "[HTTP 400]" in s

    def test_str_shows_field_errors(self):
        """__str__ includes human-readable field error lines."""
        errors = {
            "title": [{"message": "Ensure this value has at most 255 characters (it has 300).", "code": "max_length"}],
            "position": [{"message": "This field is required.", "code": "required"}],
        }
        e = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)
        s = str(e)
        assert "title: Ensure this value has at most 255 characters (it has 300)." in s
        assert "position: This field is required." in s

    def test_str_all_key_rendered_as_general(self):
        """__str__ renders __all__ field key as 'general'."""
        errors = {"__all__": [{"message": "Invalid combination.", "code": "invalid"}]}
        e = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)
        s = str(e)
        assert "general: Invalid combination." in s
        assert "__all__" not in s

    def test_str_multiple_errors_per_field(self):
        """__str__ shows all errors for a field."""
        errors = {
            "url": [
                {"message": "This field is required.", "code": "required"},
                {"message": "Enter a valid URL.", "code": "invalid"},
            ]
        }
        e = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)
        s = str(e)
        assert "url: This field is required." in s
        assert "url: Enter a valid URL." in s


class TestRequestParsesValidationError:
    """Tests for _request() parsing the VALIDATION_ERROR envelope."""

    def test_validation_error_populates_errors(self):
        """400 with VALIDATION_ERROR envelope populates errors field."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Validation failed",
            "code": "VALIDATION_ERROR",
            "errors": {
                "title": [{"message": "This field is required.", "code": "required"}]
            },
        }

        with patch("kardbrd_client.client.httpx.Client") as mock_httpx:
            from kardbrd_client.client import KardbrdClient

            mock_httpx_instance = Mock()
            mock_httpx.return_value = mock_httpx_instance
            mock_httpx_instance.request.return_value = mock_response

            client = KardbrdClient("http://test.local", "token")
            with pytest.raises(KardbrdAPIError) as exc_info:
                client._request("POST", "/api/cards/123/", json={"title": ""})

            assert exc_info.value.code == "VALIDATION_ERROR"
            assert exc_info.value.errors is not None
            assert "title" in exc_info.value.errors
            assert exc_info.value.errors["title"][0]["code"] == "required"

    def test_non_validation_error_has_none_errors(self):
        """404 with old-style envelope has errors=None."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": "Not found",
            "code": "NOT_FOUND",
        }

        with patch("kardbrd_client.client.httpx.Client") as mock_httpx:
            from kardbrd_client.client import KardbrdClient

            mock_httpx_instance = Mock()
            mock_httpx.return_value = mock_httpx_instance
            mock_httpx_instance.request.return_value = mock_response

            client = KardbrdClient("http://test.local", "token")
            with pytest.raises(KardbrdAPIError) as exc_info:
                client._request("GET", "/api/cards/999/")

            assert exc_info.value.code == "NOT_FOUND"
            assert exc_info.value.errors is None


class TestMCPServerValidationError:
    """Tests for MCP server structured validation error responses."""

    def _call_tool_handler(self, server, name, arguments):
        """Call the actual registered call_tool handler on the MCP server."""
        import asyncio

        from mcp.types import CallToolRequest, CallToolRequestParams

        handler = server.server.request_handlers[CallToolRequest]
        request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=name, arguments=arguments),
        )
        result = asyncio.get_event_loop().run_until_complete(handler(request))
        return result.root.content[0].text

    def test_validation_error_returns_structured_json(self):
        """VALIDATION_ERROR returns structured JSON for AI model."""
        from kardbrd_client.mcp_server import KardbrdMCPServer

        errors = {"title": [{"message": "This field is required.", "code": "required"}]}
        api_error = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)

        with patch("kardbrd_client.mcp_server.KardbrdClient"):
            server = KardbrdMCPServer("http://test.local", "token")
            server.executor = Mock()
            server.executor.execute.side_effect = api_error

            text = self._call_tool_handler(server, "get_card", {"card_id": "123"})

            assert "Error:" in text
            parsed = json.loads(text.replace("Error: ", "", 1))
            assert parsed["code"] == "VALIDATION_ERROR"
            assert "title" in parsed["errors"]

    def test_non_validation_error_returns_flat_string(self):
        """Non-validation errors return flat string as before."""
        from kardbrd_client.mcp_server import KardbrdMCPServer

        api_error = KardbrdAPIError("Not found", code="NOT_FOUND", status_code=404)

        with patch("kardbrd_client.mcp_server.KardbrdClient"):
            server = KardbrdMCPServer("http://test.local", "token")
            server.executor = Mock()
            server.executor.execute.side_effect = api_error

            text = self._call_tool_handler(server, "get_card", {"card_id": "123"})

            assert "Error:" in text
            assert "NOT_FOUND" in text


class TestRunnerValidationError:
    """Tests for runner structured validation error surfacing."""

    def test_anthropic_tool_error_surfaces_validation_detail(self):
        """Anthropic tool result contains structured error for VALIDATION_ERROR."""
        from kardbrd_client.runner import _process_anthropic_tool_calls

        errors = {"title": [{"message": "This field is required.", "code": "required"}]}
        api_error = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)

        mock_tool_executor = Mock()
        mock_tool_executor.execute.side_effect = api_error
        mock_client = Mock()

        tool_block = Mock()
        tool_block.name = "update_card"
        tool_block.input = {"card_id": "123", "title": ""}
        tool_block.id = "tool-use-1"

        anthropic_messages = []
        _process_anthropic_tool_calls(
            tool_use_blocks=[tool_block],
            anthropic_messages=anthropic_messages,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        content = anthropic_messages[0]["content"][0]["content"]
        assert "Validation error:" in content
        assert "VALIDATION_ERROR" in content
        assert '"title"' in content

    def test_anthropic_tool_error_non_validation_unchanged(self):
        """Non-validation errors in Anthropic tool results are unchanged."""
        from kardbrd_client.runner import _process_anthropic_tool_calls

        api_error = KardbrdAPIError("Not found", code="NOT_FOUND", status_code=404)

        mock_tool_executor = Mock()
        mock_tool_executor.execute.side_effect = api_error
        mock_client = Mock()

        tool_block = Mock()
        tool_block.name = "get_card"
        tool_block.input = {"card_id": "999"}
        tool_block.id = "tool-use-1"

        anthropic_messages = []
        _process_anthropic_tool_calls(
            tool_use_blocks=[tool_block],
            anthropic_messages=anthropic_messages,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        content = anthropic_messages[0]["content"][0]["content"]
        assert "Validation error:" not in content
        assert "NOT_FOUND" in content

    def test_gemini_tool_error_surfaces_validation_detail(self):
        """Gemini function response contains structured error for VALIDATION_ERROR."""
        from kardbrd_client.runner import _process_gemini_tool_calls

        errors = {"name": [{"message": "This field is required.", "code": "required"}]}
        api_error = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)

        mock_tool_executor = Mock()
        mock_tool_executor.execute.side_effect = api_error
        mock_client = Mock()

        tool_calls = [{"name": "create_list", "args": {"board_id": "b1", "name": ""}}]
        responses = _process_gemini_tool_calls(
            tool_calls=tool_calls,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        error_content = responses[0]["functionResponse"]["response"]["error"]
        assert "Validation error:" in error_content
        assert "VALIDATION_ERROR" in error_content
        assert '"name"' in error_content

    def test_gemini_tool_error_non_validation_unchanged(self):
        """Non-validation errors in Gemini responses are unchanged."""
        from kardbrd_client.runner import _process_gemini_tool_calls

        mock_tool_executor = Mock()
        mock_tool_executor.execute.side_effect = Exception("Network error")
        mock_client = Mock()

        tool_calls = [{"name": "get_card", "args": {}}]
        responses = _process_gemini_tool_calls(
            tool_calls=tool_calls,
            tool_executor=mock_tool_executor,
            client=mock_client,
            card_id="card-123",
        )

        error_content = responses[0]["functionResponse"]["response"]["error"]
        assert "Validation error:" not in error_content
        assert "Network error" in error_content


class TestCLIValidationError:
    """Tests for CLI handling of validation errors."""

    def test_cli_displays_field_errors(self):
        """CLI error output includes field-level error messages."""
        from click.testing import CliRunner

        from kardbrd_client.cli import cli

        errors = {"title": [{"message": "Ensure this value has at most 255 characters (it has 300).", "code": "max_length"}]}
        api_error = KardbrdAPIError("Validation failed", code="VALIDATION_ERROR", status_code=400, errors=errors)

        runner = CliRunner()
        with patch("kardbrd_client.cli.KardbrdClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.update_card.side_effect = api_error

            result = runner.invoke(cli, [
                "--api-url", "http://test.com", "--token", "tok",
                "card", "update", "c1", "--title", "x" * 300,
            ])

            assert result.exit_code == 1
            assert "title: Ensure this value has at most 255 characters" in result.output
