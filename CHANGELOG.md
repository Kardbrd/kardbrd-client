# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-05

### Added

- **API Client** (`KardbrdClient`)
  - Full REST API coverage for boards, cards, comments, checklists, attachments, and links
  - Automatic retry with exponential backoff for transient failures
  - Support for both JSON and markdown response formats

- **MCP Server** (`kardbrd-mcp`)
  - Model Context Protocol server for AI assistant integration
  - All Kardbrd tools exposed for Claude Desktop and other MCP clients

- **Agent Framework**
  - `BaseAgent` for building multi-board monitoring bots with APScheduler
  - `DirectoryStateManager` for per-board state persistence
  - Event triggers: `CommentMentionTrigger`, `CardAssignmentTrigger`, `TodoItemAssignmentTrigger`
  - `WebSocketAgentConnection` for real-time event handling
  - AI conversation runners for Anthropic, Gemini, and Mistral providers
