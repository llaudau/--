---
name: check-plan
description: Use when the user asks to review the current project plan, follow edits they made to a plan file, or continue work from the latest plan in a local plan directory.
---

# Check Plan

Use this skill when a task depends on a plan file that may have been edited after it was first created.

## Workflow

1. Look for a `plan/` directory in the current working directory.
2. If the user named a specific plan, read `plan/<name>.md`.
3. Otherwise, list files in `plan/` and inspect the most recently modified one.
4. Read the selected plan carefully and note:
   - User edits, annotations, or crossed-out items
   - Reordered steps
   - New constraints or requirements
   - Steps marked complete or skipped
5. Summarize the current state of the plan and continue from the correct step.

## Fallback

If `plan/` does not exist or contains no plan files, say that no plan file was found and suggest creating one before proceeding.
