---
name: ke
description: "Deep codebase exploration and analysis for a card's topic. Use when the user asks to explore, investigate, or understand a card's topic in the kardbrd-client codebase."
allowed-tools: Read, Grep, Glob, Bash, Agent
---

# Explore

Deep-dive exploration of the card's topic in the context of the kardbrd-client codebase.

## Instructions

1. Read the card description and all comments to understand the topic
2. Explore the codebase systematically through these passes:
   - **Client pass**: `kardbrd_client/client.py` — REST client, retry logic, request/response handling
   - **MCP Server pass**: `kardbrd_client/mcp_server.py` — MCP server entry point, tool registration
   - **Tool Schemas pass**: `kardbrd_client/tool_schemas.py` — tool definitions and parameter schemas
   - **Tool Executor pass**: `kardbrd_client/tool_executor.py` — tool execution logic
   - **Agent pass**: `kardbrd_client/agent.py` — BaseAgent, board monitoring, state management
   - **Triggers pass**: `kardbrd_client/triggers.py` — event-driven mixins, TriggerContext
   - **Runner pass**: `kardbrd_client/runner.py` — AI provider conversation runners
   - **WebSocket pass**: `kardbrd_client/websocket_agent.py` — real-time event handling
   - **Tests pass**: `kardbrd_client/tests/` — test patterns, fixtures, coverage
3. Look at related cards mentioned in the description (use `!cardId` references)
4. Check the board for related cards and context
5. Post findings as a structured comment on the card with:
   - Current state (what exists)
   - Relevant code paths and files
   - Proposed approach or options
   - Open questions for the operator

## Guidelines

- Be thorough but focused on what's relevant to the card
- Reference specific files and line numbers
- Note any architectural constraints or patterns to follow
- If the card mentions other cards, fetch and read them for context
