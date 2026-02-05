"""MCP tool schema definitions for the Kardbrd API.

This module contains only the tool definitions (pure data, no imports)
so it can be shared between the client package and Django server
without circular dependencies.
"""

# Tool definitions for MCP/Anthropic API
TOOLS = [
    {
        "name": "get_board",
        "description": "Get board details including all lists, cards, and members. Use this to understand the current state of the board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "string",
                    "description": "The id of the board to retrieve",
                }
            },
            "required": ["board_id"],
        },
    },
    {
        "name": "get_card",
        "description": "Get detailed information about a specific card, including its checklists, comments, and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to retrieve",
                }
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "update_card",
        "description": "Update a card's title, description, due date, or assignee. IMPORTANT: When saving plans or implementation details to the description, you MUST include ALL details verbatim. Do NOT summarize, filter, or omit any information - the next agent executing this plan depends on having complete context including specific file paths, code snippets, implementation steps, and technical details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to update",
                },
                "title": {
                    "type": "string",
                    "description": "New title for the card (optional)",
                },
                "description": {
                    "type": "string",
                    "description": "New description for the card (optional). When saving plans: include ALL implementation details, file paths, code examples, and technical specifications verbatim. Never summarize.",
                },
                "due_date": {
                    "type": "string",
                    "description": "New due date in ISO 8601 format (optional)",
                },
                "assignee_id": {
                    "type": "string",
                    "description": "id of user to assign (optional)",
                },
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "create_card",
        "description": "Create a new card in a specific list on the board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "string",
                    "description": "The id of the board",
                },
                "list_id": {
                    "type": "string",
                    "description": "The id of the list to add the card to",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the new card",
                },
                "description": {
                    "type": "string",
                    "description": "Description for the new card (optional). When including plans or implementation details, preserve ALL information verbatim.",
                },
            },
            "required": ["board_id", "list_id", "title"],
        },
    },
    {
        "name": "move_card",
        "description": "Move a card to a different list on the SAME board. Use this to update card status (e.g., move from 'Ideas' to 'In progress'). IMPORTANT: This moves between LISTS, not boards. You cannot move cards between different boards with this tool. NEVER move cards to the last list on the board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to move",
                },
                "list_id": {
                    "type": "string",
                    "description": "The UUID of the target list (e.g., 'kqEvREXe'). Find list IDs in the 'Board Lists' section of card/board markdown. Extract ONLY the UUID from comments like `<!-- list-id: kqEvREXe -->`. Do NOT use board_id, card_id, or checklist_id here.",
                },
                "position": {
                    "type": "integer",
                    "description": "Position in the target list (optional, defaults to top of list). Use 0 for top, or omit to place at top.",
                },
            },
            "required": ["card_id", "list_id"],
        },
    },
    {
        "name": "archive_card",
        "description": "Archive a card. Archived cards are hidden from the board by default but can be restored later. Use this for completed tasks you want to keep for reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to archive",
                },
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "unarchive_card",
        "description": "Restore an archived card back to the board. The card will appear in its original list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to unarchive",
                },
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "add_comment",
        "description": "Add a comment to a card. Use this to report progress, ask questions, or provide updates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to comment on",
                },
                "content": {
                    "type": "string",
                    "description": "The comment text (supports markdown)",
                },
            },
            "required": ["card_id", "content"],
        },
    },
    {
        "name": "toggle_reaction",
        "description": "Toggle a reaction emoji on a comment. If the user already reacted with this emoji, removes it. If not, adds it. Useful for acknowledging comments or providing quick feedback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card containing the comment",
                },
                "comment_id": {
                    "type": "string",
                    "description": "The id of the comment to react to",
                },
                "emoji": {
                    "type": "string",
                    "description": "The emoji to toggle (e.g., '👍', '👎', '❤️', '🎉', '🚀', '👀', '🤔', '💯', '✅', '🛑', '🔄')",
                },
            },
            "required": ["card_id", "comment_id", "emoji"],
        },
    },
    {
        "name": "create_checklist",
        "description": "Create a new checklist on a card for tracking subtasks. Returns {id, title, position, items}. Use the returned 'id' with add_todo to add items to this checklist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the checklist",
                },
            },
            "required": ["card_id", "title"],
        },
    },
    {
        "name": "add_todo",
        "description": "Add a todo item to an existing checklist. WORKFLOW: First call create_checklist to get a checklist_id, then use that id here. The checklist_id is in the response from create_checklist as 'id'. For existing checklists, find the id in card markdown: `### Title <!-- checklist-id: abc123 -->`",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "checklist_id": {
                    "type": "string",
                    "description": "The id of the checklist (from create_checklist response's 'id' field, or from card markdown <!-- checklist-id: ... -->)",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the todo item",
                },
            },
            "required": ["card_id", "checklist_id", "title"],
        },
    },
    {
        "name": "add_todos",
        "description": """Create a checklist and add multiple todo items in one operation.

⚠️ IMPORTANT: This tool only CREATES new checklists with items. To UPDATE existing todo items, you must use `update_todo` one item at a time - there is no batch update.

Use this instead of calling create_checklist + add_todo repeatedly when creating new checklists.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "checklist_title": {
                    "type": "string",
                    "description": "Title for the new checklist",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of todo item titles to add",
                },
            },
            "required": ["card_id", "checklist_title", "items"],
        },
    },
    {
        "name": "update_todo",
        "description": "Update a todo item's title, completion status, due date, or assignees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "checklist_id": {
                    "type": "string",
                    "description": "The id of the checklist",
                },
                "item_id": {
                    "type": "string",
                    "description": "The id of the todo item",
                },
                "title": {
                    "type": "string",
                    "description": "New title (optional)",
                },
                "is_completed": {
                    "type": "boolean",
                    "description": "Mark as completed or not (optional)",
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date in ISO 8601 format (optional)",
                },
                "assignee_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of user ids to assign (optional)",
                },
            },
            "required": ["card_id", "checklist_id", "item_id"],
        },
    },
    {
        "name": "complete_todo",
        "description": "Mark a todo item as completed. Simpler alternative to update_todo when you only need to mark a todo as done and don't know the checklist_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card containing the todo",
                },
                "todo_id": {
                    "type": "string",
                    "description": "The id of the todo item to mark as completed",
                },
            },
            "required": ["card_id", "todo_id"],
        },
    },
    {
        "name": "reopen_todo",
        "description": "Reopen a completed todo item (mark as incomplete). Use this to undo completion or when work needs to be redone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card containing the todo",
                },
                "todo_id": {
                    "type": "string",
                    "description": "The id of the todo item to reopen",
                },
            },
            "required": ["card_id", "todo_id"],
        },
    },
    {
        "name": "get_board_activity",
        "description": "Get recent activity on the board to see what has changed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "string",
                    "description": "The id of the board",
                },
                "since": {
                    "type": "string",
                    "description": "ISO timestamp to get activity after (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of activities to return (default 50)",
                },
            },
            "required": ["board_id"],
        },
    },
    # Markdown format tools (read-only)
    {
        "name": "get_board_markdown",
        "description": "Get board details as formatted markdown. Easier to read and parse hierarchically. Includes all lists, cards, and members in a structured document. Use this instead of get_board when you want a more human-readable overview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "string",
                    "description": "The id of the board to retrieve",
                }
            },
            "required": ["board_id"],
        },
    },
    {
        "name": "get_card_markdown",
        "description": "Get card details as formatted markdown. Shows description, checklists, todos, and comments in an easy-to-read format. Use this instead of get_card when you want to quickly scan card content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card to retrieve",
                }
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "list_boards_markdown",
        "description": "List all accessible boards as formatted markdown. Shows board names, workspaces, and IDs in an easy-to-scan format.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_board_activity_markdown",
        "description": "Get board activity log as formatted markdown. Shows recent changes in chronological format with timestamps, users, and actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "string",
                    "description": "The id of the board",
                },
                "since": {
                    "type": "string",
                    "description": "ISO timestamp to get activity after (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of activities to return (default 50)",
                },
            },
            "required": ["board_id"],
        },
    },
    {
        "name": "extract_todos_to_cards",
        "description": "Extract all todo items from a card and create separate cards for each todo. The original todos will be marked as completed. This is useful for breaking down a planning card into individual actionable tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_card_id": {
                    "type": "string",
                    "description": "The id of the source card containing the todos to extract",
                },
                "target_list_id": {
                    "type": "string",
                    "description": "The id of the target list where new cards should be created",
                },
                "prefix": {
                    "type": "string",
                    "description": "Optional prefix to add to each new card title (e.g., '[Task]')",
                },
            },
            "required": ["source_card_id", "target_list_id"],
        },
    },
    {
        "name": "extract_checklist_to_cards",
        "description": "Extract all todo items from a specific checklist on a card and create separate cards for each todo. The original todos will be marked as completed. Use this when you want to extract only a specific checklist rather than all checklists on a card.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_card_id": {
                    "type": "string",
                    "description": "The id of the source card containing the checklist",
                },
                "checklist_id": {
                    "type": "string",
                    "description": "The id of the specific checklist to extract",
                },
                "target_list_id": {
                    "type": "string",
                    "description": "The id of the target list where new cards should be created",
                },
                "prefix": {
                    "type": "string",
                    "description": "Optional prefix to add to each new card title",
                },
            },
            "required": ["source_card_id", "checklist_id", "target_list_id"],
        },
    },
    {
        "name": "attach_markdown",
        "description": "Attach a markdown file to a card. Use this to save detailed research findings, implementation plans, or documentation. IMPORTANT: Include ALL content verbatim - do NOT summarize or filter. Other agents depend on complete information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for the attachment (e.g., 'research-topic.md')",
                },
                "content": {
                    "type": "string",
                    "description": "The markdown content to attach. Must include complete details - never truncate or summarize.",
                },
            },
            "required": ["card_id", "filename", "content"],
        },
    },
    {
        "name": "attach_file",
        "description": "Upload any file as an attachment to a card. For text files, provide content directly. For binary files, provide base64-encoded content with is_base64=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for the attachment",
                },
                "content": {
                    "type": "string",
                    "description": "File content (text or base64-encoded binary)",
                },
                "content_type": {
                    "type": "string",
                    "description": "MIME type (e.g., 'text/plain', 'image/png')",
                },
                "is_base64": {
                    "type": "boolean",
                    "description": "Set to true if content is base64-encoded binary",
                    "default": False,
                },
            },
            "required": ["card_id", "filename", "content", "content_type"],
        },
    },
    {
        "name": "get_attachment",
        "description": "Download attachment content from a card. Returns text content directly or base64-encoded binary content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "attachment_id": {
                    "type": "string",
                    "description": "The id of the attachment to download",
                },
            },
            "required": ["card_id", "attachment_id"],
        },
    },
    {
        "name": "list_attachments",
        "description": "List all attachments on a card with their metadata (filename, size, type).",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "add_link",
        "description": "Add a link to a card. Use this to attach external URLs like GitHub PRs, documentation, or related resources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "string",
                    "description": "The id of the card",
                },
                "url": {
                    "type": "string",
                    "description": "The URL to add (e.g., 'https://github.com/user/repo/pull/123')",
                },
                "display_text": {
                    "type": "string",
                    "description": "Optional display text for the link (defaults to URL if not provided)",
                },
            },
            "required": ["card_id", "url"],
        },
    },
]
