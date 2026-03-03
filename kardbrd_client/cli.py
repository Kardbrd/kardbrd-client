"""
Kardbrd CLI — interact with Kardbrd boards from the command line.
"""

import json
import os
import sys

import click

from .client import KardbrdClient, KardbrdAPIError

__version__ = "1.0.0"


class Context:
    """Holds CLI context: the API client and output format."""

    def __init__(self, api_url: str | None, token: str | None, fmt: str):
        self._api_url = api_url
        self._token = token
        self.fmt = fmt
        self._client: KardbrdClient | None = None

    @property
    def client(self) -> KardbrdClient:
        if self._client is None:
            if not self._api_url:
                click.echo("Error: --api-url or KARDBRD_API_URL is required.", err=True)
                sys.exit(1)
            if not self._token:
                click.echo("Error: --token or KARDBRD_TOKEN is required.", err=True)
                sys.exit(1)
            self._client = KardbrdClient(self._api_url, self._token)
        return self._client


pass_ctx = click.make_pass_decorator(Context)


def _output(data, fmt: str) -> None:
    """Output data in the requested format."""
    if isinstance(data, str):
        click.echo(data)
    else:
        click.echo(json.dumps(data, indent=2))


def _handle_error(e: KardbrdAPIError) -> None:
    """Print API error and exit."""
    click.echo(f"Error: {e}", err=True)
    sys.exit(1)


# =============================================================================
# Root CLI group
# =============================================================================


@click.group()
@click.option("--api-url", envvar="KARDBRD_API_URL", help="Kardbrd API base URL.")
@click.option("--token", envvar="KARDBRD_TOKEN", help="Authentication token.")
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["json", "md"]),
    default="json",
    help="Output format.",
)
@click.version_option(version=__version__, prog_name="kardbrd")
@click.pass_context
def cli(ctx, api_url, token, fmt):
    """Kardbrd CLI — interact with Kardbrd boards from the command line."""
    ctx.ensure_object(dict)
    ctx.obj = Context(api_url=api_url, token=token, fmt=fmt)


# =============================================================================
# `kardbrd md` shortcut
# =============================================================================


@cli.command()
@click.argument("resource", type=click.Choice(["board", "card", "boards", "activity"]))
@click.argument("id", required=False)
@pass_ctx
def md(ctx, resource, id):
    """Shortcut for getting a resource in markdown format.

    \b
    Examples:
      kardbrd md board BOARD_ID
      kardbrd md card CARD_ID
      kardbrd md boards
      kardbrd md activity BOARD_ID
    """
    try:
        if resource == "boards":
            click.echo(ctx.client.list_boards_markdown())
        elif resource == "board":
            if not id:
                click.echo("Error: BOARD_ID is required.", err=True)
                sys.exit(1)
            click.echo(ctx.client.get_board_markdown(id))
        elif resource == "card":
            if not id:
                click.echo("Error: CARD_ID is required.", err=True)
                sys.exit(1)
            click.echo(ctx.client.get_card_markdown(id))
        elif resource == "activity":
            if not id:
                click.echo("Error: BOARD_ID is required.", err=True)
                sys.exit(1)
            click.echo(ctx.client.get_board_activity_markdown(id))
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Board commands
# =============================================================================


@cli.group()
def board():
    """Board operations."""
    pass


@board.command("get")
@click.argument("board_id")
@click.option("--include-archived", is_flag=True, help="Include archived cards.")
@pass_ctx
def board_get(ctx, board_id, include_archived):
    """Get board details including lists, cards, and members."""
    try:
        if ctx.fmt == "md":
            _output(ctx.client.get_board_markdown(board_id), ctx.fmt)
        else:
            _output(ctx.client.get_board(board_id, include_archived=include_archived), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("list")
@pass_ctx
def board_list(ctx):
    """List all accessible boards."""
    try:
        if ctx.fmt == "md":
            _output(ctx.client.list_boards_markdown(), ctx.fmt)
        else:
            _output(ctx.client.list_boards(), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("labels")
@click.argument("board_id")
@pass_ctx
def board_labels(ctx, board_id):
    """Get all labels defined on a board."""
    try:
        _output(ctx.client.get_board_labels(board_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("activity")
@click.argument("board_id")
@click.option("--limit", type=int, default=50, help="Max activities to return.")
@click.option("--since", default=None, help="ISO 8601 timestamp to filter after.")
@pass_ctx
def board_activity(ctx, board_id, limit, since):
    """Get recent activity on a board."""
    try:
        if ctx.fmt == "md":
            _output(ctx.client.get_board_activity_markdown(board_id, since=since, limit=limit), ctx.fmt)
        else:
            _output(ctx.client.get_board_activity(board_id, since=since, limit=limit), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("members")
@click.argument("board_id")
@pass_ctx
def board_members(ctx, board_id):
    """List all members of a board."""
    try:
        if ctx.fmt == "md":
            _output(ctx.client.get_board_markdown(board_id), ctx.fmt)
        else:
            data = ctx.client.get_board(board_id)
            _output(data.get("members", []), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("update")
@click.argument("board_id")
@click.option("--name", required=True, help="New board name.")
@pass_ctx
def board_update(ctx, board_id, name):
    """Update a board's name."""
    try:
        _output(ctx.client.update_board(board_id, name=name), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("archive")
@click.argument("board_id")
@pass_ctx
def board_archive(ctx, board_id):
    """Archive a board."""
    try:
        _output(ctx.client.archive_board(board_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("unarchive")
@click.argument("board_id")
@pass_ctx
def board_unarchive(ctx, board_id):
    """Unarchive a board."""
    try:
        _output(ctx.client.unarchive_board(board_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("favorite")
@click.argument("board_id")
@pass_ctx
def board_favorite(ctx, board_id):
    """Toggle favorite status for a board."""
    try:
        _output(ctx.client.toggle_board_favorite(board_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@board.command("search")
@click.argument("board_id")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Max results.")
@pass_ctx
def board_search(ctx, board_id, query, limit):
    """Search cards on a board by title."""
    try:
        _output(ctx.client.board_card_search(board_id, query, limit=limit), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Card commands
# =============================================================================


@cli.group()
def card():
    """Card operations."""
    pass


@card.command("get")
@click.argument("card_id")
@pass_ctx
def card_get(ctx, card_id):
    """Get card details including checklists, comments, and metadata."""
    try:
        if ctx.fmt == "md":
            _output(ctx.client.get_card_markdown(card_id), ctx.fmt)
        else:
            _output(ctx.client.get_card(card_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("create")
@click.option("--board", "board_id", required=True, help="Board ID.")
@click.option("--list", "list_id", required=True, help="List ID.")
@click.option("--title", required=True, help="Card title.")
@click.option("--description", default="", help="Card description.")
@pass_ctx
def card_create(ctx, board_id, list_id, title, description):
    """Create a new card in a list."""
    try:
        _output(ctx.client.create_card(board_id, list_id, title, description), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("update")
@click.argument("card_id")
@click.option("--title", default=None, help="New title.")
@click.option("--description", default=None, help="New description.")
@click.option("--due", "due_date", default=None, help="Due date (ISO 8601).")
@click.option("--assignee", "assignee_id", default=None, help="Assignee user ID.")
@click.option("--label", "label_ids", multiple=True, help="Label IDs (repeatable, replaces existing).")
@pass_ctx
def card_update(ctx, card_id, title, description, due_date, assignee_id, label_ids):
    """Update a card's fields. Only provided fields are changed."""
    try:
        kwargs = {}
        if title is not None:
            kwargs["title"] = title
        if description is not None:
            kwargs["description"] = description
        if due_date is not None:
            kwargs["due_date"] = due_date
        if assignee_id is not None:
            kwargs["assignee_id"] = assignee_id
        if label_ids:
            kwargs["label_ids"] = list(label_ids)
        _output(ctx.client.update_card(card_id, **kwargs), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("move")
@click.argument("card_id")
@click.option("--list", "list_id", required=True, help="Target list ID.")
@click.option("--position", type=int, default=None, help="Position in list.")
@pass_ctx
def card_move(ctx, card_id, list_id, position):
    """Move a card to a different list."""
    try:
        _output(ctx.client.move_card(card_id, list_id, position=position), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("archive")
@click.argument("card_id")
@pass_ctx
def card_archive(ctx, card_id):
    """Archive a card."""
    try:
        _output(ctx.client.archive_card(card_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("unarchive")
@click.argument("card_id")
@pass_ctx
def card_unarchive(ctx, card_id):
    """Restore an archived card."""
    try:
        _output(ctx.client.unarchive_card(card_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("assign")
@click.argument("card_id")
@click.argument("user_id")
@pass_ctx
def card_assign(ctx, card_id, user_id):
    """Assign a board member to a card."""
    try:
        _output(ctx.client.update_card(card_id, assignee_id=user_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("unassign")
@click.argument("card_id")
@pass_ctx
def card_unassign(ctx, card_id):
    """Remove the assignee from a card."""
    try:
        _output(ctx.client.update_card(card_id, assignee_id=""), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("activity")
@click.argument("card_id")
@click.option("--limit", type=int, default=20, help="Max activities to return.")
@click.option("--since", default=None, help="ISO 8601 timestamp to filter after.")
@pass_ctx
def card_activity(ctx, card_id, limit, since):
    """Get recent activity on a card."""
    try:
        _output(ctx.client.get_card_activity(card_id, since=since, limit=limit), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@card.command("move-to-board")
@click.argument("card_id")
@click.option("--board", "board_id", required=True, help="Target board ID.")
@pass_ctx
def card_move_to_board(ctx, card_id, board_id):
    """Move a card to a different board."""
    try:
        _output(ctx.client.move_card_to_board(card_id, board_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Comment commands
# =============================================================================


@cli.group()
def comment():
    """Comment operations on cards."""
    pass


@comment.command("add")
@click.argument("card_id")
@click.argument("message")
@pass_ctx
def comment_add(ctx, card_id, message):
    """Add a comment to a card."""
    try:
        _output(ctx.client.add_comment(card_id, message), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@comment.command("delete")
@click.argument("card_id")
@click.argument("comment_id")
@pass_ctx
def comment_delete(ctx, card_id, comment_id):
    """Delete a comment."""
    try:
        ctx.client.delete_comment(card_id, comment_id)
        click.echo("Comment deleted.")
    except KardbrdAPIError as e:
        _handle_error(e)


@comment.command("react")
@click.argument("card_id")
@click.argument("comment_id")
@click.argument("emoji")
@pass_ctx
def comment_react(ctx, card_id, comment_id, emoji):
    """Toggle a reaction emoji on a comment."""
    try:
        _output(ctx.client.toggle_reaction(card_id, comment_id, emoji), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Checklist commands
# =============================================================================


@cli.group()
def checklist():
    """Checklist and todo item operations."""
    pass


@checklist.command("create")
@click.argument("card_id")
@click.option("--title", required=True, help="Checklist title.")
@pass_ctx
def checklist_create(ctx, card_id, title):
    """Create a new checklist on a card."""
    try:
        _output(ctx.client.create_checklist(card_id, title), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@checklist.command("add-todo")
@click.argument("card_id")
@click.option("--checklist", "checklist_id", required=True, help="Checklist ID.")
@click.option("--title", required=True, help="Todo item title.")
@pass_ctx
def checklist_add_todo(ctx, card_id, checklist_id, title):
    """Add a todo item to a checklist."""
    try:
        _output(ctx.client.add_todo(card_id, checklist_id, title), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@checklist.command("add-todos")
@click.argument("card_id")
@click.option("--title", required=True, help="Checklist title.")
@click.argument("items", nargs=-1, required=True)
@pass_ctx
def checklist_add_todos(ctx, card_id, title, items):
    """Create a checklist with multiple items at once.

    \b
    Example:
      kardbrd checklist add-todos CARD_ID --title "Steps" "Step 1" "Step 2" "Step 3"
    """
    try:
        _output(ctx.client.add_todos(card_id, title, list(items)), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@checklist.command("update")
@click.argument("card_id")
@click.option("--checklist", "checklist_id", required=True, help="Checklist ID.")
@click.option("--item", "item_id", required=True, help="Todo item ID.")
@click.option("--title", default=None, help="New title.")
@click.option("--completed/--no-completed", default=None, help="Completion status.")
@click.option("--due", "due_date", default=None, help="Due date (ISO 8601).")
@click.option("--assignee", "assignee_ids", multiple=True, help="Assignee user IDs (repeatable).")
@pass_ctx
def checklist_update(ctx, card_id, checklist_id, item_id, title, completed, due_date, assignee_ids):
    """Update a todo item's title, completion, due date, or assignees."""
    try:
        kwargs = {}
        if title is not None:
            kwargs["title"] = title
        if completed is not None:
            kwargs["is_completed"] = completed
        if due_date is not None:
            kwargs["due_date"] = due_date
        if assignee_ids:
            kwargs["assignee_ids"] = list(assignee_ids)
        _output(ctx.client.update_todo(card_id, checklist_id, item_id, **kwargs), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@checklist.command("complete")
@click.argument("card_id")
@click.argument("todo_id")
@pass_ctx
def checklist_complete(ctx, card_id, todo_id):
    """Mark a todo item as completed."""
    try:
        _output(ctx.client.complete_todo(card_id, todo_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@checklist.command("reopen")
@click.argument("card_id")
@click.argument("todo_id")
@pass_ctx
def checklist_reopen(ctx, card_id, todo_id):
    """Reopen a completed todo item."""
    try:
        _output(ctx.client.reopen_todo(card_id, todo_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@checklist.command("extract")
@click.argument("card_id")
@click.option("--target-list", "target_list_id", required=True, help="Target list ID for new cards.")
@click.option("--checklist", "checklist_id", default=None, help="Specific checklist ID (extracts all if omitted).")
@click.option("--prefix", default="", help="Prefix for new card titles.")
@pass_ctx
def checklist_extract(ctx, card_id, target_list_id, checklist_id, prefix):
    """Extract todos into separate cards."""
    try:
        if checklist_id:
            _output(
                ctx.client.extract_checklist_to_cards(card_id, checklist_id, target_list_id, prefix=prefix),
                ctx.fmt,
            )
        else:
            _output(
                ctx.client.extract_todos_to_cards(card_id, target_list_id, prefix=prefix),
                ctx.fmt,
            )
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Attachment commands
# =============================================================================


@cli.group()
def attachment():
    """Attachment operations on cards."""
    pass


@attachment.command("upload")
@click.argument("card_id")
@click.argument("file_path", type=click.Path(exists=True))
@pass_ctx
def attachment_upload(ctx, card_id, file_path):
    """Upload a file to a card."""
    try:
        _output(ctx.client.upload_attachment(card_id, file_path), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@attachment.command("markdown")
@click.argument("card_id")
@click.option("--filename", required=True, help="Filename for the markdown attachment.")
@click.option("--content", required=True, help="Markdown content.")
@pass_ctx
def attachment_markdown(ctx, card_id, filename, content):
    """Upload markdown content as an attachment."""
    try:
        _output(ctx.client.upload_markdown_content(card_id, filename, content), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@attachment.command("list")
@click.argument("card_id")
@pass_ctx
def attachment_list(ctx, card_id):
    """List all attachments on a card."""
    try:
        _output(ctx.client.list_attachments(card_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@attachment.command("get")
@click.argument("card_id")
@click.argument("attachment_id")
@pass_ctx
def attachment_get(ctx, card_id, attachment_id):
    """Download an attachment."""
    try:
        _output(ctx.client.get_attachment(card_id, attachment_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Link commands
# =============================================================================


@cli.group()
def link():
    """Link operations on cards."""
    pass


@link.command("add")
@click.argument("card_id")
@click.argument("url")
@click.option("--text", "display_text", default="", help="Display text for the link.")
@pass_ctx
def link_add(ctx, card_id, url, display_text):
    """Add a URL link to a card."""
    try:
        _output(ctx.client.add_card_link(card_id, url, display_text=display_text), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@link.command("list")
@click.argument("card_id")
@pass_ctx
def link_list(ctx, card_id):
    """List all links on a card."""
    try:
        _output(ctx.client.list_card_links(card_id), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@link.command("update")
@click.argument("card_id")
@click.argument("link_id")
@click.option("--url", default=None, help="New URL.")
@click.option("--text", "display_text", default=None, help="New display text.")
@pass_ctx
def link_update(ctx, card_id, link_id, url, display_text):
    """Update a link."""
    try:
        _output(ctx.client.update_card_link(card_id, link_id, url=url, display_text=display_text), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@link.command("delete")
@click.argument("card_id")
@click.argument("link_id")
@pass_ctx
def link_delete(ctx, card_id, link_id):
    """Delete a link."""
    try:
        ctx.client.delete_card_link(card_id, link_id)
        click.echo("Link deleted.")
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Search commands
# =============================================================================


@cli.command("search")
@click.argument("query")
@click.option("--workspace", default=None, help="Filter to workspace by ID.")
@click.option("--include-archived/--no-archived", default=True, help="Include archived cards.")
@click.option("--limit", type=int, default=30, help="Max results.")
@click.option("--offset", type=int, default=0, help="Pagination offset.")
@pass_ctx
def search(ctx, query, workspace, include_archived, limit, offset):
    """Search cards across all accessible boards."""
    try:
        _output(
            ctx.client.search(
                query, workspace=workspace, include_archived=include_archived, limit=limit, offset=offset
            ),
            ctx.fmt,
        )
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Activity commands
# =============================================================================


@cli.command("activity")
@click.option("--limit", type=int, default=30, help="Max activities to return.")
@click.option("--since", default=None, help="ISO 8601 timestamp to filter after.")
@click.option("--actor", type=click.Choice(["all", "human", "bot"]), default=None, help="Filter by actor type.")
@click.option("--source", type=click.Choice(["all", "web", "api"]), default=None, help="Filter by source.")
@click.option(
    "--period",
    type=click.Choice(["all", "today", "yesterday", "week", "month"]),
    default=None,
    help="Time period filter.",
)
@click.option("--board", "board_id", default=None, help="Filter by board ID.")
@pass_ctx
def activity(ctx, limit, since, actor, source, period, board_id):
    """Get cross-board activity feed."""
    try:
        _output(
            ctx.client.get_activity(
                limit=limit, since=since, actor=actor, source=source, period=period, board=board_id
            ),
            ctx.fmt,
        )
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# List commands
# =============================================================================


@cli.group("list")
def list_group():
    """List operations on boards."""
    pass


@list_group.command("create")
@click.argument("board_id")
@click.option("--name", required=True, help="List name.")
@pass_ctx
def list_create(ctx, board_id, name):
    """Create a new list on a board."""
    try:
        _output(ctx.client.create_list(board_id, name), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


@list_group.command("move")
@click.argument("list_id")
@click.option("--position", type=int, required=True, help="New position (0-indexed).")
@pass_ctx
def list_move(ctx, list_id, position):
    """Move/reorder a list to a new position."""
    try:
        _output(ctx.client.move_list(list_id, position), ctx.fmt)
    except KardbrdAPIError as e:
        _handle_error(e)


# =============================================================================
# Entry point
# =============================================================================


def main():
    cli()


if __name__ == "__main__":
    main()
