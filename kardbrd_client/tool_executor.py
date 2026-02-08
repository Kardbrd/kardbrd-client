"""Tool executor for MCP tools using the Kardbrd API client."""

from typing import Any

from .client import KardbrdClient
from .tool_schemas import TOOLS


def _get_required_params(tool_name: str) -> list[str]:
    """Get required parameters for a tool from its schema."""
    for tool in TOOLS:
        if tool["name"] == tool_name:
            return tool["input_schema"].get("required", [])
    return []


def _validate_tool_input(tool_name: str, tool_input: dict[str, Any]) -> None:
    """Validate that all required parameters are present.

    Raises:
        ValueError: If a required parameter is missing
    """
    required = _get_required_params(tool_name)
    missing = [param for param in required if param not in tool_input]
    if missing:
        raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")


class ToolExecutor:
    """Executes tool calls using the Kardbrd API client."""

    def __init__(self, client: KardbrdClient):
        """Initialize with a configured API client."""
        self.client = client

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        """
        Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Result from the API call

        Raises:
            ValueError: If tool_name is not recognized or required params missing
        """
        # Validate required parameters before executing
        _validate_tool_input(tool_name, tool_input)

        match tool_name:
            case "get_board":
                return self.client.get_board(tool_input["board_id"])

            case "get_card":
                return self.client.get_card(tool_input["card_id"])

            case "update_card":
                return self.client.update_card(
                    tool_input["card_id"],
                    title=tool_input.get("title"),
                    description=tool_input.get("description"),
                    due_date=tool_input.get("due_date"),
                    assignee_id=tool_input.get("assignee_id"),
                )

            case "create_card":
                return self.client.create_card(
                    board_id=tool_input["board_id"],
                    list_id=tool_input["list_id"],
                    title=tool_input["title"],
                    description=tool_input.get("description", ""),
                )

            case "move_card":
                return self.client.move_card(
                    card_id=tool_input["card_id"],
                    list_id=tool_input["list_id"],
                    position=tool_input.get("position"),
                )

            case "archive_card":
                return self.client.archive_card(tool_input["card_id"])

            case "unarchive_card":
                return self.client.unarchive_card(tool_input["card_id"])

            case "add_comment":
                return self.client.add_comment(
                    tool_input["card_id"],
                    tool_input["content"],
                )

            case "toggle_reaction":
                return self.client.toggle_reaction(
                    card_id=tool_input["card_id"],
                    comment_id=tool_input["comment_id"],
                    emoji=tool_input["emoji"],
                )

            case "create_checklist":
                return self.client.create_checklist(
                    tool_input["card_id"],
                    tool_input["title"],
                )

            case "add_todo":
                return self.client.add_todo(
                    card_id=tool_input["card_id"],
                    checklist_id=tool_input["checklist_id"],
                    title=tool_input["title"],
                )

            case "add_todos":
                return self.client.add_todos(
                    card_id=tool_input["card_id"],
                    checklist_title=tool_input["checklist_title"],
                    items=tool_input["items"],
                )

            case "update_todo":
                return self.client.update_todo(
                    card_id=tool_input["card_id"],
                    checklist_id=tool_input["checklist_id"],
                    item_id=tool_input["item_id"],
                    title=tool_input.get("title"),
                    is_completed=tool_input.get("is_completed"),
                    due_date=tool_input.get("due_date"),
                    assignee_ids=tool_input.get("assignee_ids"),
                )

            case "complete_todo":
                return self.client.complete_todo(
                    card_id=tool_input["card_id"],
                    todo_id=tool_input["todo_id"],
                )

            case "reopen_todo":
                return self.client.reopen_todo(
                    card_id=tool_input["card_id"],
                    todo_id=tool_input["todo_id"],
                )

            case "get_board_activity":
                return self.client.get_board_activity(
                    board_id=tool_input["board_id"],
                    since=tool_input.get("since"),
                    limit=tool_input.get("limit", 50),
                )

            # Markdown format tools
            case "get_board_markdown":
                return self.client.get_board_markdown(tool_input["board_id"])

            case "get_card_markdown":
                return self.client.get_card_markdown(tool_input["card_id"])

            case "list_boards_markdown":
                return self.client.list_boards_markdown()

            case "get_board_activity_markdown":
                return self.client.get_board_activity_markdown(
                    board_id=tool_input["board_id"],
                    since=tool_input.get("since"),
                    limit=tool_input.get("limit", 50),
                )

            case "extract_todos_to_cards":
                return self.client.extract_todos_to_cards(
                    source_card_id=tool_input["source_card_id"],
                    target_list_id=tool_input["target_list_id"],
                    prefix=tool_input.get("prefix", ""),
                )

            case "extract_checklist_to_cards":
                return self.client.extract_checklist_to_cards(
                    source_card_id=tool_input["source_card_id"],
                    checklist_id=tool_input["checklist_id"],
                    target_list_id=tool_input["target_list_id"],
                    prefix=tool_input.get("prefix", ""),
                )

            case "attach_markdown":
                return self.client.upload_markdown_content(
                    card_id=tool_input["card_id"],
                    filename=tool_input["filename"],
                    content=tool_input["content"],
                )

            case "attach_file":
                return self.client.upload_file_content(
                    card_id=tool_input["card_id"],
                    filename=tool_input["filename"],
                    content=tool_input["content"],
                    content_type=tool_input["content_type"],
                    is_base64=tool_input.get("is_base64", False),
                )

            case "get_attachment":
                return self.client.get_attachment(
                    card_id=tool_input["card_id"],
                    attachment_id=tool_input["attachment_id"],
                )

            case "list_attachments":
                return self.client.list_attachments(card_id=tool_input["card_id"])

            case "add_link":
                return self.client.add_card_link(
                    card_id=tool_input["card_id"],
                    url=tool_input["url"],
                    display_text=tool_input.get("display_text", ""),
                )

            case _:
                raise ValueError(f"Unknown tool: {tool_name}")
