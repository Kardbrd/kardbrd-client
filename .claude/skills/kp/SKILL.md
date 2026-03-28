---
name: kp
description: "Create a detailed implementation plan from exploration findings. Use when the user asks to plan, design, or architect changes for a kardbrd-client card."
---

# Plan

Create a detailed implementation plan based on exploration findings.

## Instructions

1. Read the card description, all comments, and any exploration findings
   - Use `kardbrd md card <cardId>` to fetch the card
2. Read all files that will be modified to understand current state
3. Create a plan that covers:
   - **Context**: What problem this solves and why
   - **Changes**: File-by-file breakdown of modifications
   - **New files**: Any new files to create (with rationale)
   - **Tests**: What tests to add or modify
   - **Verification**: How to confirm the changes work (`uv run pytest`, `pre-commit run --all-files`)
4. Update the card description with the plan
   - Use `kardbrd card update <cardId> --description "plan content"`
5. Post a summary comment on the card
   - Use `kardbrd comment add <cardId> "summary"`

## Guidelines

- Plans should be detailed enough to implement without ambiguity
- Include specific function signatures, class names, and file paths
- Reference existing patterns in the codebase to follow
- Note any dependencies or ordering constraints between changes
- Include the exact test commands: `uv run pytest kardbrd_client/tests/` for tests, `pre-commit run --all-files` for lint
- Keep the plan focused — don't over-engineer or add unnecessary scope
- Use `kardbrd` CLI for all board/card interactions (not MCP server tools)
