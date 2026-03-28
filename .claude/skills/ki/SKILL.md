---
name: ki
description: "Execute an implementation plan from a card description. Use when the user asks to implement, build, or execute a plan for a kardbrd-client card."
---

# Implement

Execute the implementation plan from the card description.

## Instructions

1. Read the card description to find the implementation plan
   - Use `kardbrd md card <cardId>` to fetch the card with its description and comments
2. Read all attachments on the card for additional context
   - Use `kardbrd attachment list <cardId>` to see attachments
3. Implement changes file by file following the plan exactly
4. Run tests after each significant change: `uv run pytest kardbrd_client/tests/`
5. Run lint: `pre-commit run --all-files`
6. Fix any test failures or lint errors
7. Commit changes with a descriptive message (reference the card ID)
8. Post a comment summarizing what was implemented and any deviations from the plan
   - Use `kardbrd comment add <cardId> "message"`

## Guidelines

- Follow the plan — don't add scope or refactor unrelated code
- Use existing patterns from the codebase
- Test commands: `uv run pytest` (all tests), `uv run pytest kardbrd_client/tests/test_file.py` (single file)
- Lint: `pre-commit run --all-files`
- Commit message format: Brief description of the change
- If the plan is missing or unclear, post a comment asking for clarification instead of guessing
  - Use `kardbrd comment add <cardId> "question"` to ask
- If tests fail and you can't fix them, report the failure in a comment
- Use `kardbrd` CLI for all board/card interactions (not MCP server tools)
