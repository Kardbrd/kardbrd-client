"""Tests for the kardbrd CLI."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from kardbrd_client.cli import cli
from kardbrd_client.client import KardbrdAPIError


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_client():
    with patch("kardbrd_client.cli.KardbrdClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


CLI_OPTS = ["--api-url", "http://test.com", "--token", "test-token"]


class TestTopLevel:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Kardbrd CLI" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_missing_api_url(self, runner):
        result = runner.invoke(cli, ["--token", "tok", "board", "list"])
        assert result.exit_code != 0
        assert "api-url" in result.output.lower() or "KARDBRD_API_URL" in result.output

    def test_missing_token(self, runner):
        result = runner.invoke(cli, ["--api-url", "http://x.com", "board", "list"])
        assert result.exit_code != 0
        assert "token" in result.output.lower() or "KARDBRD_TOKEN" in result.output


class TestMdShortcut:
    def test_md_board(self, runner, mock_client):
        mock_client.get_board_markdown.return_value = "# Board"
        result = runner.invoke(cli, [*CLI_OPTS, "md", "board", "abc123"])
        assert result.exit_code == 0
        assert "# Board" in result.output
        mock_client.get_board_markdown.assert_called_once_with("abc123")

    def test_md_card(self, runner, mock_client):
        mock_client.get_card_markdown.return_value = "# Card"
        result = runner.invoke(cli, [*CLI_OPTS, "md", "card", "def456"])
        assert result.exit_code == 0
        mock_client.get_card_markdown.assert_called_once_with("def456")

    def test_md_boards(self, runner, mock_client):
        mock_client.list_boards_markdown.return_value = "# Boards"
        result = runner.invoke(cli, [*CLI_OPTS, "md", "boards"])
        assert result.exit_code == 0
        mock_client.list_boards_markdown.assert_called_once()

    def test_md_activity(self, runner, mock_client):
        mock_client.get_board_activity_markdown.return_value = "# Activity"
        result = runner.invoke(cli, [*CLI_OPTS, "md", "activity", "abc123"])
        assert result.exit_code == 0
        mock_client.get_board_activity_markdown.assert_called_once_with("abc123")

    def test_md_board_missing_id(self, runner, mock_client):
        result = runner.invoke(cli, [*CLI_OPTS, "md", "board"])
        assert result.exit_code != 0


class TestBoardCommands:
    def test_board_get_json(self, runner, mock_client):
        mock_client.get_board.return_value = {"id": "abc", "name": "Test"}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "get", "abc"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["id"] == "abc"

    def test_board_get_md(self, runner, mock_client):
        mock_client.get_board_markdown.return_value = "# Board"
        result = runner.invoke(cli, [*CLI_OPTS, "-f", "md", "board", "get", "abc"])
        assert result.exit_code == 0
        assert "# Board" in result.output

    def test_board_get_include_archived(self, runner, mock_client):
        mock_client.get_board.return_value = {"id": "abc"}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "get", "abc", "--include-archived"])
        assert result.exit_code == 0
        mock_client.get_board.assert_called_once_with("abc", include_archived=True)

    def test_board_list(self, runner, mock_client):
        mock_client.list_boards.return_value = [{"id": "abc"}]
        result = runner.invoke(cli, [*CLI_OPTS, "board", "list"])
        assert result.exit_code == 0

    def test_board_labels(self, runner, mock_client):
        mock_client.get_board_labels.return_value = {"labels": []}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "labels", "abc"])
        assert result.exit_code == 0

    def test_board_activity(self, runner, mock_client):
        mock_client.get_board_activity.return_value = {"activities": []}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "activity", "abc", "--limit", "10"])
        assert result.exit_code == 0
        mock_client.get_board_activity.assert_called_once_with("abc", since=None, limit=10)

    def test_board_members(self, runner, mock_client):
        mock_client.get_board.return_value = {"members": [{"id": "u1", "name": "Alice"}]}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "members", "abc"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["name"] == "Alice"

    def test_board_update(self, runner, mock_client):
        mock_client.update_board.return_value = {"id": "abc", "name": "New"}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "update", "abc", "--name", "New"])
        assert result.exit_code == 0
        mock_client.update_board.assert_called_once_with("abc", name="New")

    def test_board_archive(self, runner, mock_client):
        mock_client.archive_board.return_value = {"id": "abc", "is_archived": True}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "archive", "abc"])
        assert result.exit_code == 0

    def test_board_unarchive(self, runner, mock_client):
        mock_client.unarchive_board.return_value = {"id": "abc", "is_archived": False}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "unarchive", "abc"])
        assert result.exit_code == 0

    def test_board_favorite(self, runner, mock_client):
        mock_client.toggle_board_favorite.return_value = {"is_favorite": True}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "favorite", "abc"])
        assert result.exit_code == 0

    def test_board_search(self, runner, mock_client):
        mock_client.board_card_search.return_value = {"cards": []}
        result = runner.invoke(cli, [*CLI_OPTS, "board", "search", "abc", "test query"])
        assert result.exit_code == 0
        mock_client.board_card_search.assert_called_once_with("abc", "test query", limit=10)


class TestCardCommands:
    def test_card_get(self, runner, mock_client):
        mock_client.get_card.return_value = {"id": "c1", "title": "Test"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "get", "c1"])
        assert result.exit_code == 0

    def test_card_create(self, runner, mock_client):
        mock_client.create_card.return_value = {"id": "c1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "card", "create", "--board", "b1", "--list", "l1", "--title", "New Card"]
        )
        assert result.exit_code == 0
        mock_client.create_card.assert_called_once_with(board_id="b1", list_id="l1", title="New Card")

    def test_card_update(self, runner, mock_client):
        mock_client.update_card.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "update", "c1", "--title", "Updated"])
        assert result.exit_code == 0
        mock_client.update_card.assert_called_once_with("c1", title="Updated")

    def test_card_update_labels(self, runner, mock_client):
        mock_client.update_card.return_value = {"id": "c1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "card", "update", "c1", "--label", "l1", "--label", "l2"]
        )
        assert result.exit_code == 0
        mock_client.update_card.assert_called_once_with("c1", label_ids=["l1", "l2"])

    def test_card_move(self, runner, mock_client):
        mock_client.move_card.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "move", "c1", "--list", "l1"])
        assert result.exit_code == 0

    def test_card_archive(self, runner, mock_client):
        mock_client.archive_card.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "archive", "c1"])
        assert result.exit_code == 0

    def test_card_unarchive(self, runner, mock_client):
        mock_client.unarchive_card.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "unarchive", "c1"])
        assert result.exit_code == 0

    def test_card_assign(self, runner, mock_client):
        mock_client.update_card.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "assign", "c1", "u1"])
        assert result.exit_code == 0
        mock_client.update_card.assert_called_once_with("c1", assignee_id="u1")

    def test_card_unassign(self, runner, mock_client):
        mock_client.update_card.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "unassign", "c1"])
        assert result.exit_code == 0
        mock_client.update_card.assert_called_once_with("c1", assignee_id=None)

    def test_card_activity(self, runner, mock_client):
        mock_client.get_card_activity.return_value = {"activities": []}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "activity", "c1", "--limit", "5"])
        assert result.exit_code == 0
        mock_client.get_card_activity.assert_called_once_with("c1", since=None, limit=5)

    def test_card_move_to_board(self, runner, mock_client):
        mock_client.move_card_to_board.return_value = {"id": "c1"}
        result = runner.invoke(cli, [*CLI_OPTS, "card", "move-to-board", "c1", "--board", "b2"])
        assert result.exit_code == 0
        mock_client.move_card_to_board.assert_called_once_with("c1", "b2")


class TestCommentCommands:
    def test_comment_add(self, runner, mock_client):
        mock_client.add_comment.return_value = {"id": "cm1"}
        result = runner.invoke(cli, [*CLI_OPTS, "comment", "add", "c1", "Hello!"])
        assert result.exit_code == 0
        mock_client.add_comment.assert_called_once_with("c1", "Hello!")

    def test_comment_delete(self, runner, mock_client):
        result = runner.invoke(cli, [*CLI_OPTS, "comment", "delete", "c1", "cm1"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_comment_react(self, runner, mock_client):
        mock_client.toggle_reaction.return_value = {}
        result = runner.invoke(cli, [*CLI_OPTS, "comment", "react", "c1", "cm1", "\U0001f44d"])
        assert result.exit_code == 0


class TestChecklistCommands:
    def test_checklist_create(self, runner, mock_client):
        mock_client.create_checklist.return_value = {"id": "cl1"}
        result = runner.invoke(cli, [*CLI_OPTS, "checklist", "create", "c1", "--title", "Tasks"])
        assert result.exit_code == 0

    def test_checklist_add_todo(self, runner, mock_client):
        mock_client.add_todo.return_value = {"id": "t1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "checklist", "add-todo", "c1", "--checklist", "cl1", "--title", "Do it"]
        )
        assert result.exit_code == 0

    def test_checklist_add_todos(self, runner, mock_client):
        mock_client.add_todos.return_value = {"id": "cl1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "checklist", "add-todos", "c1", "--title", "Steps", "Step 1", "Step 2"]
        )
        assert result.exit_code == 0
        mock_client.add_todos.assert_called_once_with("c1", "Steps", ["Step 1", "Step 2"])

    def test_checklist_update(self, runner, mock_client):
        mock_client.update_todo.return_value = {"id": "t1"}
        result = runner.invoke(
            cli,
            [*CLI_OPTS, "checklist", "update", "c1", "--checklist", "cl1", "--item", "t1", "--completed"],
        )
        assert result.exit_code == 0
        mock_client.update_todo.assert_called_once_with("c1", "cl1", "t1", is_completed=True)

    def test_checklist_complete(self, runner, mock_client):
        mock_client.complete_todo.return_value = {"id": "t1"}
        result = runner.invoke(cli, [*CLI_OPTS, "checklist", "complete", "c1", "t1"])
        assert result.exit_code == 0

    def test_checklist_reopen(self, runner, mock_client):
        mock_client.reopen_todo.return_value = {"id": "t1"}
        result = runner.invoke(cli, [*CLI_OPTS, "checklist", "reopen", "c1", "t1"])
        assert result.exit_code == 0

    def test_checklist_extract(self, runner, mock_client):
        mock_client.extract_todos_to_cards.return_value = {"created": 3}
        result = runner.invoke(cli, [*CLI_OPTS, "checklist", "extract", "c1", "--target-list", "l1"])
        assert result.exit_code == 0
        mock_client.extract_todos_to_cards.assert_called_once_with("c1", "l1", prefix="")

    def test_checklist_extract_specific(self, runner, mock_client):
        mock_client.extract_checklist_to_cards.return_value = {"created": 2}
        result = runner.invoke(
            cli, [*CLI_OPTS, "checklist", "extract", "c1", "--target-list", "l1", "--checklist", "cl1"]
        )
        assert result.exit_code == 0
        mock_client.extract_checklist_to_cards.assert_called_once_with("c1", "cl1", "l1", prefix="")


class TestAttachmentCommands:
    def test_attachment_list(self, runner, mock_client):
        mock_client.list_attachments.return_value = {"attachments": []}
        result = runner.invoke(cli, [*CLI_OPTS, "attachment", "list", "c1"])
        assert result.exit_code == 0

    def test_attachment_get(self, runner, mock_client):
        mock_client.get_attachment.return_value = {"content": "hello"}
        result = runner.invoke(cli, [*CLI_OPTS, "attachment", "get", "c1", "a1"])
        assert result.exit_code == 0

    def test_attachment_markdown(self, runner, mock_client):
        mock_client.upload_markdown_content.return_value = {"id": "a1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "attachment", "markdown", "c1", "--filename", "test.md", "--content", "# Hi"]
        )
        assert result.exit_code == 0


class TestLinkCommands:
    def test_link_add(self, runner, mock_client):
        mock_client.add_card_link.return_value = {"id": "lk1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "link", "add", "c1", "https://example.com", "--text", "Example"]
        )
        assert result.exit_code == 0
        mock_client.add_card_link.assert_called_once_with("c1", "https://example.com", display_text="Example")

    def test_link_list(self, runner, mock_client):
        mock_client.list_card_links.return_value = []
        result = runner.invoke(cli, [*CLI_OPTS, "link", "list", "c1"])
        assert result.exit_code == 0

    def test_link_update(self, runner, mock_client):
        mock_client.update_card_link.return_value = {"id": "lk1"}
        result = runner.invoke(
            cli, [*CLI_OPTS, "link", "update", "c1", "lk1", "--url", "https://new.com"]
        )
        assert result.exit_code == 0

    def test_link_delete(self, runner, mock_client):
        result = runner.invoke(cli, [*CLI_OPTS, "link", "delete", "c1", "lk1"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()


class TestSearchCommand:
    def test_search(self, runner, mock_client):
        mock_client.search.return_value = {"cards": [], "total_count": 0}
        result = runner.invoke(cli, [*CLI_OPTS, "search", "my query"])
        assert result.exit_code == 0
        mock_client.search.assert_called_once_with(
            "my query", workspace=None, include_archived=False, limit=30, offset=0
        )

    def test_search_with_options(self, runner, mock_client):
        mock_client.search.return_value = {"cards": [], "total_count": 0}
        result = runner.invoke(
            cli, [*CLI_OPTS, "search", "query", "--workspace", "ws1", "--no-archived", "--limit", "5"]
        )
        assert result.exit_code == 0
        mock_client.search.assert_called_once_with(
            "query", workspace="ws1", include_archived=False, limit=5, offset=0
        )


class TestActivityCommand:
    def test_activity(self, runner, mock_client):
        mock_client.get_activity.return_value = {"activities": []}
        result = runner.invoke(cli, [*CLI_OPTS, "activity"])
        assert result.exit_code == 0
        mock_client.get_activity.assert_called_once_with(
            limit=30, since=None, actor=None, source=None, period=None, board=None
        )

    def test_activity_with_filters(self, runner, mock_client):
        mock_client.get_activity.return_value = {"activities": []}
        result = runner.invoke(
            cli,
            [*CLI_OPTS, "activity", "--actor", "human", "--period", "today", "--board", "b1"],
        )
        assert result.exit_code == 0
        mock_client.get_activity.assert_called_once_with(
            limit=30, since=None, actor="human", source=None, period="today", board="b1"
        )


class TestListCommands:
    def test_list_create(self, runner, mock_client):
        mock_client.create_list.return_value = {"list_id": "l1"}
        result = runner.invoke(cli, [*CLI_OPTS, "list", "create", "b1", "--name", "New List"])
        assert result.exit_code == 0
        mock_client.create_list.assert_called_once_with("b1", "New List")

    def test_list_move(self, runner, mock_client):
        mock_client.move_list.return_value = {"id": "l1", "position": 2}
        result = runner.invoke(cli, [*CLI_OPTS, "list", "move", "l1", "--position", "2"])
        assert result.exit_code == 0
        mock_client.move_list.assert_called_once_with("l1", 2)


class TestErrorHandling:
    def test_api_error_on_board_get(self, runner, mock_client):
        mock_client.get_board.side_effect = KardbrdAPIError("Not found", code="NOT_FOUND", status_code=404)
        result = runner.invoke(cli, [*CLI_OPTS, "board", "get", "bad_id"])
        assert result.exit_code != 0
        assert "Not found" in result.output

    def test_api_error_on_card_create(self, runner, mock_client):
        mock_client.create_card.side_effect = KardbrdAPIError("Validation error", status_code=400)
        result = runner.invoke(
            cli, [*CLI_OPTS, "card", "create", "--board", "b1", "--list", "l1", "--title", "Test"]
        )
        assert result.exit_code != 0
        assert "Validation error" in result.output

    def test_api_error_on_comment_add(self, runner, mock_client):
        mock_client.add_comment.side_effect = KardbrdAPIError("Server error", status_code=500)
        result = runner.invoke(cli, [*CLI_OPTS, "comment", "add", "c1", "Hello"])
        assert result.exit_code != 0
        assert "Server error" in result.output


class TestNoOpUpdates:
    def test_card_update_no_flags(self, runner, mock_client):
        result = runner.invoke(cli, [*CLI_OPTS, "card", "update", "c1"])
        assert result.exit_code != 0
        assert "at least one update flag" in result.output

    def test_link_update_no_flags(self, runner, mock_client):
        result = runner.invoke(cli, [*CLI_OPTS, "link", "update", "c1", "lk1"])
        assert result.exit_code != 0
        assert "at least one update flag" in result.output

    def test_checklist_update_no_flags(self, runner, mock_client):
        result = runner.invoke(
            cli, [*CLI_OPTS, "checklist", "update", "c1", "--checklist", "cl1", "--item", "t1"]
        )
        assert result.exit_code != 0
        assert "at least one update flag" in result.output


class TestAttachmentUpload:
    def test_attachment_upload(self, runner, mock_client):
        mock_client.upload_attachment.return_value = {"id": "a1", "filename": "test.txt"}
        with runner.isolated_filesystem():
            with open("test.txt", "w") as f:
                f.write("hello world")
            result = runner.invoke(cli, [*CLI_OPTS, "attachment", "upload", "c1", "test.txt"])
            assert result.exit_code == 0
            mock_client.upload_attachment.assert_called_once_with("c1", "test.txt")


class TestAttachmentMarkdownContentFile:
    def test_content_file(self, runner, mock_client):
        mock_client.upload_markdown_content.return_value = {"id": "a1"}
        with runner.isolated_filesystem():
            with open("notes.md", "w") as f:
                f.write("# Notes\nSome content.")
            result = runner.invoke(
                cli,
                [*CLI_OPTS, "attachment", "markdown", "c1", "--filename", "notes.md", "--content-file", "notes.md"],
            )
            assert result.exit_code == 0
            mock_client.upload_markdown_content.assert_called_once_with("c1", "notes.md", "# Notes\nSome content.")

    def test_content_and_content_file_mutually_exclusive(self, runner, mock_client):
        with runner.isolated_filesystem():
            with open("notes.md", "w") as f:
                f.write("content")
            result = runner.invoke(
                cli,
                [*CLI_OPTS, "attachment", "markdown", "c1", "--filename", "f.md",
                 "--content", "inline", "--content-file", "notes.md"],
            )
            assert result.exit_code != 0
            assert "mutually exclusive" in result.output

    def test_no_content_provided(self, runner, mock_client):
        result = runner.invoke(
            cli, [*CLI_OPTS, "attachment", "markdown", "c1", "--filename", "f.md"]
        )
        assert result.exit_code != 0
        assert "required" in result.output


class TestBoardMembersMarkdown:
    def test_board_members_md_extracts_section(self, runner, mock_client):
        mock_client.get_board_markdown.return_value = (
            "# My Board\n\n## Members\n\n- Alice (admin)\n- Bob (member)\n\n## Lists\n\n### Todo\n"
        )
        result = runner.invoke(cli, [*CLI_OPTS, "-f", "md", "board", "members", "abc"])
        assert result.exit_code == 0
        assert "## Members" in result.output
        assert "Alice" in result.output
        assert "## Lists" not in result.output
