"""
Kardbrd API Client implementation.
"""

import logging
import mimetypes
import os
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class KardbrdAPIError(Exception):
    """Exception raised for Kardbrd API errors."""

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"(code: {self.code})")
        if self.status_code:
            parts.append(f"[HTTP {self.status_code}]")
        return " ".join(parts)


def _is_retryable(exception: BaseException) -> bool:
    """Check if exception is retryable (network errors, 5xx server errors)."""
    if isinstance(exception, httpx.RequestError):
        return True
    if isinstance(exception, KardbrdAPIError) and exception.status_code and exception.status_code >= 500:
        return True
    return False


def _log_retry(retry_state) -> None:
    """Log retry attempts."""
    logger.warning(
        f"Retrying request (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_retryable),
    before_sleep=_log_retry,
    reraise=True,
)
def _upload_to_s3(upload_url: str, file_path: str, content_type: str) -> None:
    """Upload file to S3/MinIO with retry logic.

    Args:
        upload_url: Presigned S3 upload URL
        file_path: Path to the local file to upload
        content_type: MIME type of the file

    Raises:
        KardbrdAPIError: If upload fails after retries
    """
    with open(file_path, "rb") as f:
        try:
            response = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": content_type},
                timeout=300.0,  # Longer timeout for uploads
            )
        except httpx.RequestError as e:
            raise KardbrdAPIError(f"Upload failed: {e}", code="REQUEST_ERROR") from e

        if response.status_code >= 500:
            # Retry on 5xx errors
            raise KardbrdAPIError(
                message=f"Upload failed: {response.text}",
                status_code=response.status_code,
            )
        elif response.status_code >= 400:
            # Don't retry on 4xx errors
            raise KardbrdAPIError(
                message=f"Upload failed: {response.text}",
                status_code=response.status_code,
            )


class KardbrdClient:
    """
    Python client for the Kardbrd Board API.

    Example:
        >>> client = KardbrdClient(
        ...     base_url="http://localhost:8000",
        ...     token="XaXhGkDjoSHOMBKiZGn7iug9JZ6q0LzZWiaeiatULSU"
        ... )
        >>> boards = client.list_boards()
        >>> print(boards)
    """

    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        """
        Initialize the Kardbrd API client.

        Args:
            base_url: The base URL of the Kardbrd API (e.g., "http://localhost:8000")
            token: The bot authentication token
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def __enter__(self) -> "KardbrdClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make an API request with automatic retry for transient failures.

        Retries on network errors and 5xx server errors with exponential backoff.
        Does NOT retry on 4xx client errors (auth failures, validation errors).

        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            path: API path (e.g., "/api/boards/")
            json: JSON body for POST/PATCH requests

        Returns:
            The response data

        Raises:
            KardbrdAPIError: If the API returns an error (after retries exhausted)
        """
        try:
            response = self._client.request(method, path, json=json)
        except httpx.RequestError as e:
            raise KardbrdAPIError(f"Request failed: {e}", code="REQUEST_ERROR") from e

        # Handle error responses
        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise KardbrdAPIError(
                    message=error_data.get("error", "Unknown error"),
                    code=error_data.get("code"),
                    status_code=response.status_code,
                )
            except ValueError as e:
                raise KardbrdAPIError(
                    message=response.text or "Unknown error",
                    status_code=response.status_code,
                ) from e

        # Parse successful response
        try:
            data = response.json()
            return data.get("data", data)
        except ValueError as e:
            raise KardbrdAPIError(
                message="Invalid JSON response",
                code="INVALID_RESPONSE",
                status_code=response.status_code,
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _request_markdown(self, path: str) -> str:
        """
        Make an API request expecting markdown response with automatic retry.

        Retries on network errors and 5xx server errors with exponential backoff.

        Args:
            path: API path (e.g., "/api/boards/")

        Returns:
            The markdown content as a string

        Raises:
            KardbrdAPIError: If the API returns an error (after retries exhausted)
        """
        try:
            response = self._client.request("GET", path, headers={"Accept": "text/markdown"})
        except httpx.RequestError as e:
            raise KardbrdAPIError(f"Request failed: {e}", code="REQUEST_ERROR") from e

        # Handle error responses
        if response.status_code >= 400:
            raise KardbrdAPIError(
                message=response.text or "Unknown error",
                status_code=response.status_code,
            )

        return response.text

    # =========================================================================
    # Board Methods
    # =========================================================================

    def list_boards(self) -> list[dict[str, Any]]:
        """
        List all boards the authenticated user has access to.

        Returns:
            A list of board objects with id, name, description, workspace, etc.

        Example:
            >>> boards = client.list_boards()
            >>> for board in boards:
            ...     print(f"{board['name']} ({board['id']})")
        """
        return self._request("GET", "/api/boards/")

    def get_board(self, board_id: str) -> dict[str, Any]:
        """
        Get board details including lists, cards, and members.

        Args:
            board_id: The public_id  of the board

        Returns:
            Board object with lists, cards, members, etc.

        Example:
            >>> board = client.get_board("abc123...")
            >>> print(f"Board: {board['name']}")
            >>> for lst in board['lists']:
            ...     print(f"  - {lst['name']}: {len(lst['cards'])} cards")
        """
        return self._request("GET", f"/api/boards/{board_id}/")

    def list_boards_markdown(self) -> str:
        """
        List all boards in markdown format.

        Returns:
            Markdown-formatted string with board list

        Example:
            >>> boards_md = client.list_boards_markdown()
            >>> print(boards_md)
            # My Boards
            ...
        """
        return self._request_markdown("/api/boards/")

    def get_board_markdown(self, board_id: str) -> str:
        """
        Get board details in markdown format.

        Args:
            board_id: The public_id  of the board

        Returns:
            Markdown-formatted string with full board hierarchy

        Example:
            >>> board_md = client.get_board_markdown("abc123...")
            >>> print(board_md)
            # Board Name
            ## Members
            ...
        """
        return self._request_markdown(f"/api/boards/{board_id}/")

    # =========================================================================
    # Label Methods
    # =========================================================================

    def get_board_labels(self, board_id: str) -> dict[str, Any]:
        """
        Get all labels defined on a board.

        Args:
            board_id: The public_id of the board

        Returns:
            Dict with 'labels' list containing label objects with id, name, color, position

        Example:
            >>> labels = client.get_board_labels("abc123...")
            >>> for label in labels['labels']:
            ...     print(f"{label['name']} ({label['color']})")
        """
        return self._request("GET", f"/api/boards/{board_id}/labels/")

    def rename_board_label(
        self,
        board_id: str,
        label_id: str,
        name: str,
    ) -> dict[str, Any]:
        """
        Rename a board label.

        Args:
            board_id: The public_id of the board
            label_id: The public_id of the label
            name: New name for the label

        Returns:
            Updated label object with id, name, color, position

        Example:
            >>> label = client.rename_board_label("abc123...", "label456...", "Critical")
        """
        return self._request(
            "PATCH",
            f"/api/boards/{board_id}/labels/{label_id}/",
            json={"name": name},
        )

    def add_card_label(self, card_id: str, label_id: str) -> dict[str, Any]:
        """
        Add a label to a card.

        Args:
            card_id: The public_id of the card
            label_id: The public_id of the label to add

        Returns:
            Label object with id, name, color (201 if new, 200 if already assigned)

        Example:
            >>> label = client.add_card_label("card123...", "label456...")
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/labels/",
            json={"label_id": label_id},
        )

    def remove_card_label(self, card_id: str, label_id: str) -> dict[str, Any]:
        """
        Remove a label from a card.

        Args:
            card_id: The public_id of the card
            label_id: The public_id of the label to remove

        Returns:
            Dict with 'removed': True (404 if label not assigned)

        Example:
            >>> result = client.remove_card_label("card123...", "label456...")
        """
        return self._request("DELETE", f"/api/cards/{card_id}/labels/{label_id}/")

    # =========================================================================
    # Card Methods
    # =========================================================================

    def get_card(self, card_id: str) -> dict[str, Any]:
        """
        Get card details with checklists and comments.

        Args:
            card_id: The public_id  of the card

        Returns:
            Card object with title, description, checklists, comments, etc.

        Example:
            >>> card = client.get_card("def456...")
            >>> print(f"Card: {card['title']}")
            >>> print(f"Checklists: {len(card['checklists'])}")
        """
        return self._request("GET", f"/api/cards/{card_id}/")

    def get_card_markdown(self, card_id: str) -> str:
        """
        Get card details in markdown format.

        Args:
            card_id: The public_id  of the card

        Returns:
            Markdown-formatted string with card details, checklists, and comments

        Example:
            >>> card_md = client.get_card_markdown("def456...")
            >>> print(card_md)
            # Card Title
            ## Description
            ...
        """
        return self._request_markdown(f"/api/cards/{card_id}/")

    def update_card(
        self,
        card_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        due_date: str | None = None,
        assignee_id: str | None = None,
        label_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Update card fields.

        Args:
            card_id: The public_id  of the card
            title: New card title (optional)
            description: New card description (optional)
            due_date: New due date in ISO 8601 format (optional, e.g., "2024-12-31T23:59:59Z")
            assignee_id: public_id of the user to assign (optional, must be a board member)
            label_ids: List of label IDs to assign (optional, replaces existing labels)

        Returns:
            Updated card object

        Example:
            >>> card = client.update_card(
            ...     "def456...",
            ...     title="Updated Title",
            ...     description="New description",
            ...     due_date="2024-12-31T23:59:59Z",
            ...     label_ids=["label1", "label2"]
            ... )
        """
        data = {}
        if title is not None:
            data["title"] = title
        if description is not None:
            data["description"] = description
        if due_date is not None:
            data["due_date"] = due_date
        if assignee_id is not None:
            data["assignee_id"] = assignee_id
        if label_ids is not None:
            data["label_ids"] = label_ids

        return self._request("POST", f"/api/cards/{card_id}/", json=data)

    def create_card(
        self,
        board_id: str,
        list_id: str,
        title: str,
        description: str = "",
    ) -> dict[str, Any]:
        """
        Create a new card in a list.

        Args:
            board_id: The public_id  of the board
            list_id: The public_id  of the list
            title: Card title
            description: Card description (optional)

        Returns:
            Created card object

        Example:
            >>> card = client.create_card(
            ...     board_id="abc123...",
            ...     list_id="xyz789...",
            ...     title="New Card",
            ...     description="Card description"
            ... )
        """
        return self._request(
            "POST",
            f"/api/boards/{board_id}/lists/{list_id}/cards/",
            json={"title": title, "description": description},
        )

    def move_card(
        self,
        card_id: str,
        list_id: str,
        position: int | None = None,
    ) -> dict[str, Any]:
        """
        Move a card to a different list or reorder within the same list.

        Args:
            card_id: The public_id of the card to move
            list_id: The public_id of the target list
            position: Position in the target list (optional, defaults to end of list)

        Returns:
            Updated card object with new list and position

        Example:
            >>> card = client.move_card(
            ...     card_id="def456...",
            ...     list_id="xyz789...",
            ...     position=0
            ... )
        """
        data = {"list_id": list_id}
        if position is not None:
            data["position"] = position

        return self._request("POST", f"/api/cards/{card_id}/move/", json=data)

    def archive_card(self, card_id: str) -> dict[str, Any]:
        """
        Archive a card.

        Archived cards are hidden from the board by default but can be
        restored later with unarchive_card().

        Args:
            card_id: The public_id of the card to archive

        Returns:
            Updated card object with is_archived=True

        Example:
            >>> card = client.archive_card("def456...")
            >>> assert card['is_archived'] is True
        """
        return self._request("POST", f"/api/cards/{card_id}/archive/")

    def unarchive_card(self, card_id: str) -> dict[str, Any]:
        """
        Unarchive a card.

        Restores an archived card back to the board.

        Args:
            card_id: The public_id of the card to unarchive

        Returns:
            Updated card object with is_archived=False

        Example:
            >>> card = client.unarchive_card("def456...")
            >>> assert card['is_archived'] is False
        """
        return self._request("POST", f"/api/cards/{card_id}/unarchive/")

    # =========================================================================
    # Comment Methods
    # =========================================================================

    def add_comment(self, card_id: str, content: str) -> dict[str, Any]:
        """
        Add a comment to a card.

        Args:
            card_id: The public_id  of the card
            content: Comment text

        Returns:
            Created comment object

        Example:
            >>> comment = client.add_comment("def456...", "This is a comment")
            >>> print(f"Comment added at {comment['created_at']}")
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/comments/",
            json={"content": content},
        )

    def get_comment(self, card_id: str, comment_id: str) -> dict[str, Any]:
        """
        Get comment details.

        Args:
            card_id: The public_id of the card
            comment_id: The public_id of the comment

        Returns:
            Comment object with content, author, created_at, etc.

        Example:
            >>> comment = client.get_comment("card123...", "comm123...")
            >>> print(f"Comment by {comment['author']['display_name']}: {comment['content']}")
        """
        return self._request("GET", f"/api/cards/{card_id}/comments/{comment_id}/")

    def delete_comment(self, card_id: str, comment_id: str) -> None:
        """
        Delete a comment from a card.

        Args:
            card_id: The public_id of the card
            comment_id: The public_id of the comment

        Example:
            >>> client.delete_comment("card123...", "comm123...")
        """
        self._request("DELETE", f"/api/cards/{card_id}/comments/{comment_id}/")

    def toggle_reaction(self, card_id: str, comment_id: str, emoji: str) -> dict[str, Any]:
        """
        Toggle a reaction on a comment.

        If the user already reacted with this emoji, removes the reaction.
        If the user hasn't reacted with this emoji, adds the reaction.

        Args:
            card_id: The public_id of the card
            comment_id: The public_id of the comment
            emoji: The emoji to toggle (e.g., "👍", "🎉", "❤️")

        Returns:
            Updated reactions dict with user details

        Example:
            >>> reactions = client.toggle_reaction("card123...", "comm123...", "👍")
            >>> print(reactions)
            {'👍': [{'id': '...', 'display_name': 'User', 'avatar': '...'}]}
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/comments/{comment_id}/react/",
            json={"emoji": emoji},
        )

    # =========================================================================
    # Card Link Methods
    # =========================================================================

    def list_card_links(self, card_id: str) -> list[dict[str, Any]]:
        """
        List all links on a card.

        Args:
            card_id: The public_id  of the card

        Returns:
            List of link dictionaries with keys:
            - id: Link UUID
            - url: External URL
            - display_text: Optional display name
            - position: Display order
            - created_by: User who created the link
            - created_at, updated_at: Timestamps

        Example:
            >>> links = client.list_card_links("def456...")
            >>> for link in links:
            ...     print(f"{link['display_text']}: {link['url']}")
        """
        return self._request("GET", f"/api/cards/{card_id}/links/")

    def add_card_link(
        self,
        card_id: str,
        url: str,
        display_text: str = "",
    ) -> dict[str, Any]:
        """
        Add a link to a card.

        Args:
            card_id: The public_id  of the card
            url: External URL (e.g., GitHub PR, docs)
            display_text: Optional display name (defaults to URL)

        Returns:
            Created link object

        Example:
            >>> link = client.add_card_link(
            ...     "def456...",
            ...     "https://github.com/user/repo/pull/123",
            ...     "PR: Fix authentication"
            ... )
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/links/",
            json={"url": url, "display_text": display_text},
        )

    def update_card_link(
        self,
        card_id: str,
        link_id: str,
        *,
        url: str | None = None,
        display_text: str | None = None,
    ) -> dict[str, Any]:
        """
        Update a card link.

        Args:
            card_id: The public_id  of the card
            link_id: The public_id  of the link
            url: New URL (optional)
            display_text: New display text (optional)

        Returns:
            Updated link object

        Example:
            >>> link = client.update_card_link(
            ...     "def456...",
            ...     "link123...",
            ...     display_text="PR: Fix authentication (merged)"
            ... )
        """
        data = {}
        if url is not None:
            data["url"] = url
        if display_text is not None:
            data["display_text"] = display_text

        return self._request(
            "PATCH",
            f"/api/cards/{card_id}/links/{link_id}/",
            json=data,
        )

    def delete_card_link(self, card_id: str, link_id: str) -> None:
        """
        Delete a card link.

        Args:
            card_id: The public_id  of the card
            link_id: The public_id  of the link

        Example:
            >>> client.delete_card_link("def456...", "link123...")
        """
        self._request("DELETE", f"/api/cards/{card_id}/links/{link_id}/")

    # =========================================================================
    # Attachment Methods
    # =========================================================================

    def upload_attachment(
        self,
        card_id: str,
        file_path: str,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload an attachment to a card.

        Args:
            card_id: The public_id of the card
            file_path: Path to the local file to upload
            content_type: MIME type of the file (optional, guessed from filename if omitted)
            filename: Name of the file (optional, uses basename of file_path if omitted)

        Returns:
            Created attachment object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not filename:
            filename = os.path.basename(file_path)

        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "application/octet-stream"

        file_size = os.path.getsize(file_path)

        # 1. Get presigned URL
        presign_data = self._request(
            "POST",
            f"/api/cards/{card_id}/attachments/presign/",
            json={
                "filename": filename,
                "content_type": content_type,
                "file_size": file_size,
            },
        )

        upload_url = presign_data["upload_url"]
        s3_key = presign_data["s3_key"]

        # 2. Upload file content to S3 with retry logic
        _upload_to_s3(upload_url, file_path, content_type)

        # 3. Confirm upload
        return self._request(
            "POST",
            f"/api/cards/{card_id}/attachments/confirm/",
            json={
                "s3_key": s3_key,
                "filename": filename,
                "file_size": file_size,
                "content_type": content_type,
            },
        )

    def upload_markdown_content(
        self,
        card_id: str,
        filename: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Upload markdown content as an attachment to a card.

        Args:
            card_id: The public_id of the card
            filename: Name for the markdown file (will append .md if missing)
            content: The markdown content to upload

        Returns:
            Created attachment object
        """
        import tempfile

        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            return self.upload_attachment(
                card_id=card_id,
                file_path=temp_path,
                content_type="text/markdown",
                filename=filename,
            )
        finally:
            os.unlink(temp_path)

    def upload_file_content(
        self,
        card_id: str,
        filename: str,
        content: str,
        content_type: str,
        is_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Upload file content as an attachment to a card.

        Args:
            card_id: The public_id of the card
            filename: Name for the file
            content: File content (text or base64-encoded binary)
            content_type: MIME type of the file
            is_base64: True if content is base64-encoded binary

        Returns:
            Created attachment object
        """
        import base64
        import tempfile

        if is_base64:
            content_bytes = base64.b64decode(content)
        else:
            content_bytes = content.encode("utf-8")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content_bytes)
            temp_path = f.name

        try:
            return self.upload_attachment(
                card_id=card_id,
                file_path=temp_path,
                content_type=content_type,
                filename=filename,
            )
        finally:
            os.unlink(temp_path)

    def list_attachments(self, card_id: str) -> dict[str, Any]:
        """
        List all attachments on a card.

        Args:
            card_id: The public_id of the card

        Returns:
            Dict with 'card_id' and 'attachments' list

        Example:
            >>> attachments = client.list_attachments("def456...")
            >>> for att in attachments['attachments']:
            ...     print(f"{att['filename']} ({att['file_size_display']})")
        """
        attachments = self._request("GET", f"/api/cards/{card_id}/attachments/")
        return {"card_id": card_id, "attachments": attachments}

    def get_attachment(self, card_id: str, attachment_id: str) -> dict[str, Any]:
        """
        Get attachment content.

        Args:
            card_id: The public_id of the card
            attachment_id: The public_id of the attachment

        Returns:
            Dict with attachment info and content (text or base64-encoded binary)

        Example:
            >>> att = client.get_attachment("def456...", "att123...")
            >>> if att['is_base64']:
            ...     content = base64.b64decode(att['content'])
            ... else:
            ...     content = att['content']
        """
        return self._request("GET", f"/api/cards/{card_id}/attachments/{attachment_id}/")

    # =========================================================================
    # Checklist Methods
    # =========================================================================

    def create_checklist(self, card_id: str, title: str) -> dict[str, Any]:
        """
        Create a checklist on a card.

        Args:
            card_id: The public_id  of the card
            title: Checklist title

        Returns:
            Created checklist object

        Example:
            >>> checklist = client.create_checklist("def456...", "My Tasks")
            >>> print(f"Checklist ID: {checklist['id']}")
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/checklists/",
            json={"title": title},
        )

    def add_todo(
        self,
        card_id: str,
        checklist_id: str,
        title: str,
    ) -> dict[str, Any]:
        """
        Add a todo item to a checklist.

        Args:
            card_id: The public_id  of the card
            checklist_id: The public_id  of the checklist
            title: Todo item title

        Returns:
            Created todo item object

        Example:
            >>> todo = client.add_todo(
            ...     card_id="def456...",
            ...     checklist_id="check123...",
            ...     title="Complete task"
            ... )
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/checklists/{checklist_id}/items/",
            json={"title": title},
        )

    def add_todos(
        self,
        card_id: str,
        checklist_title: str,
        items: list[str],
    ) -> dict[str, Any]:
        """
        Create a checklist with multiple todo items in one atomic operation.

        This is more efficient than calling create_checklist + add_todo repeatedly.

        Args:
            card_id: The public_id of the card
            checklist_title: Title for the new checklist
            items: List of todo item titles to add

        Returns:
            Created checklist object with items

        Example:
            >>> checklist = client.add_todos(
            ...     card_id="def456...",
            ...     checklist_title="Tasks",
            ...     items=["Task 1", "Task 2", "Task 3"]
            ... )
        """
        return self._request(
            "POST",
            f"/api/cards/{card_id}/checklists/bulk/",
            json={"title": checklist_title, "items": items},
        )

    def update_todo(
        self,
        card_id: str,
        checklist_id: str,
        item_id: str,
        *,
        title: str | None = None,
        is_completed: bool | None = None,
        due_date: str | None = None,
        assignee_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Update a todo item.

        Args:
            card_id: The public_id  of the card
            checklist_id: The public_id  of the checklist
            item_id: The public_id  of the todo item
            title: New title (optional)
            is_completed: Completion status (optional)
            due_date: Due date in ISO 8601 format (optional)
            assignee_ids: List of user public_ids to assign (optional)

        Returns:
            Updated todo item object

        Example:
            >>> todo = client.update_todo(
            ...     card_id="def456...",
            ...     checklist_id="check123...",
            ...     item_id="item789...",
            ...     is_completed=True
            ... )
        """
        data = {}
        if title is not None:
            data["title"] = title
        if is_completed is not None:
            data["is_completed"] = is_completed
        if due_date is not None:
            data["due_date"] = due_date
        if assignee_ids is not None:
            data["assignee_ids"] = assignee_ids

        return self._request(
            "PATCH",
            f"/api/cards/{card_id}/checklists/{checklist_id}/items/{item_id}/",
            json=data,
        )

    def update_todo_completion(
        self,
        card_id: str,
        todo_id: str,
        completed: bool,
    ) -> dict[str, Any]:
        """
        Update a todo item's completion status (simplified endpoint).

        This is a simpler alternative to update_todo() when you only need to
        change completion status and don't know the checklist_id.

        Args:
            card_id: The public_id  of the card containing the todo
            todo_id: The public_id  of the todo item to update
            completed: True to mark as completed, False to reopen

        Returns:
            Updated todo object with fields: id, title, is_completed, position,
            due_date, assignees

        Example:
            >>> todo = client.update_todo_completion(
            ...     card_id="def456...",
            ...     todo_id="item789...",
            ...     completed=True
            ... )
        """
        return self._request(
            "PATCH",
            f"/api/cards/{card_id}/todos/{todo_id}/",
            json={"completed": completed},
        )

    def complete_todo(self, card_id: str, todo_id: str) -> dict[str, Any]:
        """
        Mark a todo item as completed (convenience method).

        Args:
            card_id: The public_id  of the card containing the todo
            todo_id: The public_id  of the todo item to complete

        Returns:
            Updated todo object

        Example:
            >>> todo = client.complete_todo("def456...", "item789...")
            >>> assert todo['is_completed'] is True
        """
        return self.update_todo_completion(card_id, todo_id, completed=True)

    def reopen_todo(self, card_id: str, todo_id: str) -> dict[str, Any]:
        """
        Reopen a completed todo item (convenience method).

        Args:
            card_id: The public_id  of the card containing the todo
            todo_id: The public_id  of the todo item to reopen

        Returns:
            Updated todo object

        Example:
            >>> todo = client.reopen_todo("def456...", "item789...")
            >>> assert todo['is_completed'] is False
        """
        return self.update_todo_completion(card_id, todo_id, completed=False)

    # =========================================================================
    # Activity Methods
    # =========================================================================

    def get_board_activity(
        self,
        board_id: str,
        since: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get recent activity for a board.

        Args:
            board_id: The public_id  of the board
            since: ISO timestamp to filter activities after (optional)
            limit: Maximum number of activities to return (default 50, max 200)

        Returns:
            Dict with 'activities' list and 'board_id'

        Example:
            >>> activity = client.get_board_activity("abc123...")
            >>> for item in activity['activities']:
            ...     print(f"{item['user']['display_name']} {item['action']} {item['entity_type']}")
        """
        params = {"limit": str(limit)}
        if since:
            params["since"] = since

        query_string = urlencode(params)
        return self._request("GET", f"/api/boards/{board_id}/activity/?{query_string}")

    def get_board_activity_markdown(
        self,
        board_id: str,
        since: str | None = None,
        limit: int = 50,
    ) -> str:
        """
        Get recent activity for a board in markdown format.

        Args:
            board_id: The public_id  of the board
            since: ISO timestamp to filter activities after (optional)
            limit: Maximum number of activities to return (default 50, max 200)

        Returns:
            Markdown-formatted activity log

        Example:
            >>> activity_md = client.get_board_activity_markdown("abc123...")
            >>> print(activity_md)
            # Activity Log - Board Name
            ## Recent Activity
            ...
        """
        params = {"limit": str(limit)}
        if since:
            params["since"] = since

        query_string = urlencode(params)
        return self._request_markdown(f"/api/boards/{board_id}/activity/?{query_string}")

    def report_status(self, status: str) -> dict[str, Any]:
        """
        Report the bot's current status.

        Args:
            status: Status string (e.g., "idle" or "working on: Task title")

        Returns:
            Response with status and status_updated_at

        Example:
            >>> client.report_status("idle")
            >>> client.report_status("working on: Implement login form")
        """
        return self._request("POST", "/api/bot/status", json={"status": status})

    def extract_todos_to_cards(self, source_card_id: str, target_list_id: str, prefix: str = "") -> dict[str, Any]:
        """
        Extract all todos from a card and create separate cards for each todo.

        Args:
            source_card_id: UUID of the card containing todos to extract
            target_list_id: UUID of the list where new cards should be created
            prefix: Optional prefix to add to each new card title

        Returns:
            Response with created cards and updated todos
        """
        return self._request(
            "POST",
            f"/api/cards/{source_card_id}/extract-todos-to-cards/",
            json={
                "target_list_id": target_list_id,
                "prefix": prefix,
            },
        )

    def extract_checklist_to_cards(
        self,
        source_card_id: str,
        checklist_id: str,
        target_list_id: str,
        prefix: str = "",
    ) -> dict[str, Any]:
        """
        Extract all todos from a specific checklist and create separate cards.

        Args:
            source_card_id: UUID of the card containing the checklist
            checklist_id: UUID of the specific checklist to extract
            target_list_id: UUID of the list where new cards should be created
            prefix: Optional prefix to add to each new card title

        Returns:
            Response with created cards and updated todos
        """
        return self._request(
            "POST",
            f"/api/cards/{source_card_id}/checklists/{checklist_id}/extract-to-cards/",
            json={
                "target_list_id": target_list_id,
                "prefix": prefix,
            },
        )
