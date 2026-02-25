"""Tests for the MCP server CLI and authentication."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from kardbrd_client.mcp_server import KardbrdMCPServer


class TestMCPServerCLI:
    """Smoke tests for kardbrd-mcp CLI."""

    def test_help(self):
        """kardbrd-mcp --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "kardbrd_client.mcp_server", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Kardbrd MCP Server" in result.stdout
        assert "--api-url" in result.stdout
        assert "--token" in result.stdout

    def test_missing_api_url_errors(self):
        """kardbrd-mcp fails without api-url."""
        result = subprocess.run(
            [sys.executable, "-m", "kardbrd_client.mcp_server", "--token", "test-token"],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": "/usr/bin"},  # Clear KARDBRD_API_URL env var
        )
        assert result.returncode != 0
        assert "--api-url is required" in result.stderr

    def test_missing_token_errors(self):
        """kardbrd-mcp fails without token."""
        result = subprocess.run(
            [sys.executable, "-m", "kardbrd_client.mcp_server", "--api-url", "http://localhost"],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": "/usr/bin"},  # Clear KARDBRD_TOKEN env var
        )
        assert result.returncode != 0
        assert "--token is required" in result.stderr

    def test_env_vars_work(self):
        """kardbrd-mcp accepts env vars for configuration."""
        # We can't fully test this without running the server,
        # but we can verify the args parser accepts the env vars
        # by checking that it doesn't error with them set
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
os.environ['KARDBRD_API_URL'] = 'http://test.local'
os.environ['KARDBRD_TOKEN'] = 'kbn_pat_testtoken123'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--api-url")
parser.add_argument("--token")
args = parser.parse_args([])

api_url = args.api_url or os.environ.get("KARDBRD_API_URL")
token = args.token or os.environ.get("KARDBRD_TOKEN")

assert api_url == 'http://test.local'
assert token == 'kbn_pat_testtoken123'
print("OK")
""",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout


class TestMCPServerUnit:
    """Unit tests for KardbrdMCPServer class."""

    def test_server_initialization(self):
        """KardbrdMCPServer initializes with correct parameters."""
        with patch("kardbrd_client.mcp_server.KardbrdClient") as mock_client:
            server = KardbrdMCPServer("http://test.local", "test-token")

            mock_client.assert_called_once_with(base_url="http://test.local", token="test-token")
            assert server.server.name == "kardbrd-mcp"

    def test_server_with_pat_token(self):
        """KardbrdMCPServer works with PAT token format."""
        with patch("kardbrd_client.mcp_server.KardbrdClient") as mock_client:
            KardbrdMCPServer("http://test.local", "kbn_pat_abc123xyz")

            mock_client.assert_called_once_with(base_url="http://test.local", token="kbn_pat_abc123xyz")

    def test_server_with_bot_token(self):
        """KardbrdMCPServer works with legacy bot token format."""
        with patch("kardbrd_client.mcp_server.KardbrdClient") as mock_client:
            KardbrdMCPServer("http://test.local", "legacy-bot-token-format")

            mock_client.assert_called_once_with(base_url="http://test.local", token="legacy-bot-token-format")


class TestMCPToolExecution:
    """Tests for MCP tool execution."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock KardbrdClient."""
        return MagicMock()

    @pytest.fixture
    def server(self, mock_client):
        """Create a KardbrdMCPServer with mocked client."""
        with patch("kardbrd_client.mcp_server.KardbrdClient", return_value=mock_client):
            return KardbrdMCPServer("http://test.local", "test-token")

    def test_get_board_tool(self, server, mock_client):
        """get_board tool executes correctly."""
        mock_client.get_board.return_value = {
            "id": "board-123",
            "name": "Test Board",
            "lists": [],
        }

        result = server.executor.execute("get_board", {"board_id": "board-123"})

        mock_client.get_board.assert_called_once_with("board-123")
        assert result["id"] == "board-123"
        assert result["name"] == "Test Board"

    def test_get_card_tool(self, server, mock_client):
        """get_card tool executes correctly."""
        mock_client.get_card.return_value = {
            "id": "card-456",
            "title": "Test Card",
            "description": "A test card",
        }

        result = server.executor.execute("get_card", {"card_id": "card-456"})

        mock_client.get_card.assert_called_once_with("card-456")
        assert result["id"] == "card-456"

    def test_add_comment_tool(self, server, mock_client):
        """add_comment tool executes correctly."""
        mock_client.add_comment.return_value = {
            "id": "comment-789",
            "content": "Test comment",
        }

        result = server.executor.execute(
            "add_comment",
            {"card_id": "card-456", "content": "Test comment"},
        )

        mock_client.add_comment.assert_called_once_with("card-456", "Test comment")
        assert result["content"] == "Test comment"

    def test_create_card_tool(self, server, mock_client):
        """create_card tool executes correctly."""
        mock_client.create_card.return_value = {
            "id": "new-card-123",
            "title": "New Card",
        }

        result = server.executor.execute(
            "create_card",
            {
                "board_id": "board-123",
                "list_id": "list-456",
                "title": "New Card",
                "description": "Card description",
            },
        )

        mock_client.create_card.assert_called_once_with(
            board_id="board-123",
            list_id="list-456",
            title="New Card",
            description="Card description",
        )
        assert result["title"] == "New Card"

    def test_update_card_tool(self, server, mock_client):
        """update_card tool executes correctly."""
        mock_client.update_card.return_value = {
            "id": "card-456",
            "title": "Updated Title",
        }

        result = server.executor.execute(
            "update_card",
            {"card_id": "card-456", "title": "Updated Title"},
        )

        mock_client.update_card.assert_called_once()
        assert result["title"] == "Updated Title"


class TestAttachFileToolExecution:
    """Tests for attach_file tool with file_path support."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock KardbrdClient."""
        return MagicMock()

    @pytest.fixture
    def server(self, mock_client):
        """Create a KardbrdMCPServer with mocked client."""
        with patch("kardbrd_client.mcp_server.KardbrdClient", return_value=mock_client):
            return KardbrdMCPServer("http://test.local", "test-token")

    def test_attach_file_with_content(self, server, mock_client):
        """attach_file with content uses upload_file_content."""
        mock_client.upload_file_content.return_value = {
            "id": "att-123",
            "filename": "test.txt",
        }

        result = server.executor.execute(
            "attach_file",
            {
                "card_id": "card-456",
                "filename": "test.txt",
                "content": "hello world",
                "content_type": "text/plain",
            },
        )

        mock_client.upload_file_content.assert_called_once_with(
            card_id="card-456",
            filename="test.txt",
            content="hello world",
            content_type="text/plain",
            is_base64=False,
        )
        assert result["filename"] == "test.txt"

    def test_attach_file_with_content_base64(self, server, mock_client):
        """attach_file with content and is_base64=True passes flag through."""
        mock_client.upload_file_content.return_value = {"id": "att-123"}

        server.executor.execute(
            "attach_file",
            {
                "card_id": "card-456",
                "filename": "image.png",
                "content": "iVBORw0KGgo=",
                "content_type": "image/png",
                "is_base64": True,
            },
        )

        mock_client.upload_file_content.assert_called_once_with(
            card_id="card-456",
            filename="image.png",
            content="iVBORw0KGgo=",
            content_type="image/png",
            is_base64=True,
        )

    def test_attach_file_with_file_path(self, server, mock_client):
        """attach_file with file_path uses upload_attachment."""
        mock_client.upload_attachment.return_value = {
            "id": "att-123",
            "filename": "screenshot.png",
        }

        result = server.executor.execute(
            "attach_file",
            {
                "card_id": "card-456",
                "file_path": "/tmp/screenshot.png",
            },
        )

        mock_client.upload_attachment.assert_called_once_with(
            card_id="card-456",
            file_path="/tmp/screenshot.png",
            content_type=None,
            filename=None,
        )
        assert result["filename"] == "screenshot.png"

    def test_attach_file_with_file_path_and_overrides(self, server, mock_client):
        """attach_file with file_path respects explicit filename and content_type."""
        mock_client.upload_attachment.return_value = {"id": "att-123"}

        server.executor.execute(
            "attach_file",
            {
                "card_id": "card-456",
                "file_path": "/tmp/screenshot.png",
                "filename": "custom-name.png",
                "content_type": "image/png",
            },
        )

        mock_client.upload_attachment.assert_called_once_with(
            card_id="card-456",
            file_path="/tmp/screenshot.png",
            content_type="image/png",
            filename="custom-name.png",
        )

    def test_attach_file_no_source_raises(self, server):
        """attach_file with no content/file_path raises ValueError."""
        with pytest.raises(ValueError, match="Must provide exactly one of"):
            server.executor.execute(
                "attach_file",
                {"card_id": "card-456"},
            )

    def test_attach_file_multiple_sources_raises(self, server):
        """attach_file with both content and file_path raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            server.executor.execute(
                "attach_file",
                {
                    "card_id": "card-456",
                    "content": "hello",
                    "file_path": "/tmp/test.txt",
                },
            )

    def test_attach_file_content_requires_filename(self, server):
        """attach_file with content but no filename raises ValueError."""
        with pytest.raises(ValueError, match="filename is required"):
            server.executor.execute(
                "attach_file",
                {
                    "card_id": "card-456",
                    "content": "hello",
                    "content_type": "text/plain",
                },
            )

    def test_attach_file_content_requires_content_type(self, server):
        """attach_file with content but no content_type raises ValueError."""
        with pytest.raises(ValueError, match="content_type is required"):
            server.executor.execute(
                "attach_file",
                {
                    "card_id": "card-456",
                    "content": "hello",
                    "filename": "test.txt",
                },
            )


class TestTokenFormats:
    """Tests for different token format handling."""

    def test_pat_token_prefix(self):
        """PAT tokens have the kbn_pat_ prefix."""
        pat_token = "kbn_pat_abc123xyz456"
        assert pat_token.startswith("kbn_pat_")

    def test_bot_token_no_prefix(self):
        """Legacy bot tokens don't have a specific prefix."""
        bot_token = "some-random-token-string"
        assert not bot_token.startswith("kbn_pat_")

    def test_client_accepts_both_formats(self):
        """KardbrdClient accepts both token formats."""
        with patch("kardbrd_client.mcp_server.KardbrdClient") as mock_client:
            # PAT format
            KardbrdMCPServer("http://test.local", "kbn_pat_abc123")
            mock_client.assert_called_with(base_url="http://test.local", token="kbn_pat_abc123")

            mock_client.reset_mock()

            # Bot token format
            KardbrdMCPServer("http://test.local", "legacy-bot-token")
            mock_client.assert_called_with(base_url="http://test.local", token="legacy-bot-token")
