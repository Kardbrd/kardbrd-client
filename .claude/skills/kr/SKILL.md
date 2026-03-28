---
name: kr
description: "Full code review of changes on the card's branch. Use when the user asks to review code, check a PR, or audit changes for a kardbrd-client card."
allowed-tools: Read, Grep, Glob, Bash, Agent
---

# Review

Full code review of changes on the card's branch.

## Instructions

1. Read the card description and comments for context
   - Use `kardbrd md card <cardId>` to fetch the card
2. Check `git diff main...HEAD` to see all changes on this branch
3. Review each changed file for:
   - **Correctness**: Does the code do what the card describes?
   - **Tests**: Are there adequate tests? Run `uv run pytest` to verify they pass
   - **Style**: Does it follow existing patterns? Run `pre-commit run --all-files`
   - **Security**: Check for credential exposure, injection vulnerabilities, unsafe deserialization
   - **Error handling**: Are failures handled gracefully? Are async patterns correct?
4. Post a structured review comment using `kardbrd comment add <cardId> "review"`:
   - Summary of changes
   - Issues found (if any), categorized by severity
   - Suggestions for improvement
   - Verdict: approve, request changes, or needs discussion

## Security Checklist (kardbrd-client specific)

- [ ] No API tokens or credentials logged or exposed in error messages
- [ ] httpx client configured securely (timeouts, TLS)
- [ ] MCP tool inputs validated before processing
- [ ] CLI inputs validated and sanitized
- [ ] No unbounded data in prompts or API calls
- [ ] State files (JSON) validated on read
- [ ] WebSocket messages validated before processing

## Guidelines

- Be constructive — suggest fixes, not just problems
- Reference specific lines when commenting on code
- Check that new code has corresponding tests
- Verify the branch is clean: no debug prints, no commented-out code
- Use `kardbrd` CLI for all board/card interactions (not MCP server tools)
