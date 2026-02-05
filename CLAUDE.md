# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Run all tests
uv run pytest kardbrd_client/tests/

# Run a single test file
uv run pytest kardbrd_client/tests/test_runner.py

# Run a specific test
uv run pytest kardbrd_client/tests/test_runner.py::test_function_name

# Lint and format (via pre-commit)
pre-commit run --all-files

# Run MCP server manually
uv run kardbrd-mcp --api-url <url> --token <token>
```

## Architecture Overview

This is a Python client library for the Kardbrd Board API, providing three main capabilities:

### 1. API Client (`client.py`)
`KardbrdClient` is an httpx-based REST client with automatic retry logic (tenacity) for transient failures. Provides methods for boards, cards, comments, checklists, attachments, and activity feeds. Supports both JSON and markdown response formats.

### 2. MCP Server (`mcp_server.py`)
Exposes Kardbrd API as an MCP (Model Context Protocol) server for AI assistants. Entry point: `kardbrd-mcp` CLI command. Tool schemas defined in `tool_schemas.py`, execution logic in `tool_executor.py`.

### 3. Agent Framework (`agent.py`, `triggers.py`, `runner.py`)
Infrastructure for building AI-powered bots that monitor Kardbrd boards:

- **BaseAgent** (`agent.py`): Multi-board monitoring agent using APScheduler. Polls boards for activity, manages per-board state via `DirectoryStateManager` (stores JSON files per board).

- **Triggers** (`triggers.py`): Event-driven mixins (`CommentMentionTrigger`, `CardAssignmentTrigger`, `TodoItemAssignmentTrigger`). Extract `TriggerContext` from activities and dispatch to handlers.

- **Runner** (`runner.py`): Standalone conversation runners for different AI providers (Anthropic, Gemini, Mistral Vibe). Completely independent of BaseAgent—receives `TriggerContext` (data only) and creates its own resources.

- **WebSocket** (`websocket_agent.py`): Real-time event handling as alternative to polling. Supports streaming responses for onboarding conversations.

### Key Design Patterns

- **Thread Safety**: TriggerContext contains only data (no live objects). Each worker thread creates its own KardbrdClient from subscription data.
- **State Isolation**: Per-board JSON files with per-board locks allow concurrent updates without blocking.
- **Provider Abstraction**: AI providers (Anthropic, Gemini, Mistral) implement conversation loops with tool execution in `runner.py`.

## Environment Variables

- `KARDBRD_API_URL`: Base URL for Kardbrd API
- `KARDBRD_TOKEN`: Bot authentication token
- `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`: AI provider keys
