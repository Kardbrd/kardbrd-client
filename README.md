# kardbrd-client

Python client library for the [Kardbrd](https://kardbrd.com) Board API.

## Features

- **REST API Client** - Full-featured client with automatic retry logic for transient failures
- **MCP Server** - Expose Kardbrd tools to AI assistants via Model Context Protocol
- **Agent Framework** - Build AI-powered bots that monitor boards and respond to events

## Installation

```bash
pip install kardbrd-client
# or with uv
uv add kardbrd-client
```

## Quick Start

### API Client

```python
from kardbrd_client import KardbrdClient

client = KardbrdClient(
    base_url="https://app.kardbrd.com",
    token="your-bot-token"
)

# List boards
boards = client.list_boards()

# Get card details as markdown
card_md = client.get_card_markdown("card-id")

# Add a comment
client.add_comment("card-id", "Hello from the API!")

# Create a checklist with todos
client.add_todos(
    card_id="card-id",
    checklist_title="Tasks",
    items=["First task", "Second task", "Third task"]
)
```

### MCP Server

Run the MCP server for AI assistant integration:

```bash
kardbrd-mcp --api-url https://app.kardbrd.com --token your-bot-token
```

Or use environment variables:

```bash
export KARDBRD_API_URL=https://app.kardbrd.com
export KARDBRD_TOKEN=your-bot-token
kardbrd-mcp
```

Configure in Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kardbrd": {
      "command": "kardbrd-mcp",
      "args": ["--api-url", "https://app.kardbrd.com", "--token", "your-token"]
    }
  }
}
```

### Agent Framework

Build bots that respond to board events:

```python
from kardbrd_client import (
    BaseAgent,
    DirectoryStateManager,
    CommentMentionTrigger,
)

class MyBot(BaseAgent, CommentMentionTrigger):
    mention_keyword = "@mybot"

    def get_base_system_prompt(self):
        return "You are a helpful assistant."

    def get_logger_name(self):
        return "mybot"

    def process_activity(self, activities):
        self.process_activities(activities)  # From CommentMentionTrigger

    def handle_trigger(self, context):
        # Respond to mentions
        print(f"Mentioned by {context.user_name}: {context.comment_content}")

# Run the bot
state = DirectoryStateManager("./state")
bot = MyBot(state, poll_interval=30)
bot.run_loop()
```

## Available Triggers

- `CommentMentionTrigger` - Respond to @mentions in comments
- `CardAssignmentTrigger` - Respond when cards are assigned to the bot
- `TodoItemAssignmentTrigger` - Respond when todo items are assigned

## API Methods

| Category | Methods |
|----------|---------|
| Boards | `list_boards`, `get_board`, `get_board_markdown`, `get_board_activity` |
| Cards | `get_card`, `get_card_markdown`, `create_card`, `update_card`, `move_card`, `archive_card` |
| Comments | `add_comment`, `get_comment`, `delete_comment`, `toggle_reaction` |
| Checklists | `create_checklist`, `add_todo`, `add_todos`, `update_todo`, `complete_todo` |
| Attachments | `upload_attachment`, `upload_markdown_content`, `list_attachments`, `get_attachment` |
| Links | `add_card_link`, `list_card_links`, `update_card_link`, `delete_card_link` |

## License

MIT
