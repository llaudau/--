---
name: update-report
description: Use when the user wants physics learning notes in a local report.md updated from the current session, keeping only organized subject matter rather than task logs.
---

# Update Report

Use this skill to maintain project learning notes in `report.md`.

## Workflow

1. Look for `report.md` in the current working directory.
2. If it does not exist, create it with:

```markdown
# <Project Name> - Learning Notes

## 目录

## 学习内容
```

3. Read the existing file to understand its structure and covered topics.
4. Extract only durable physics content from the session:
   - Key concepts and definitions
   - Important formulas in LaTeX
   - Physical intuition and conceptual connections
   - Comparisons between methods or frameworks
5. Do not record dates, task logs, or diary-style progress notes.
6. Insert content in the most logical section:
   - Expand existing topic sections in place
   - Add new `## <Topic>` sections when needed
   - Use `###` subsections only when they improve clarity
7. Update `## 目录` so it reflects the current top-level sections.
8. Re-read the full file and summarize what the notes now cover.

## Style

- Match the user's language.
- Prefer textbook-style notes over chronology.
- Use clear prose, bullets, and equations.
