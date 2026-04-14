Claude sync summary on 2026-04-09

Source: /home/khw/.claude

Portable items synced into Codex:
- Claude command `check_plan.md` converted into Codex user skill `check-plan`
- Claude command `update_report.md` converted into Codex user skill `update-report`

Claude permissions discovered:
- Global allow list in `settings.json`: `Bash`, `WebSearch`, `WebFetch`
- Local explicit bash grants in `.claude/settings.local.json` for several machine-specific `/Users/wangkehe/...` commands

Codex mapping:
- Web access and shell access are governed by Codex runtime tools and sandbox approvals, not a direct copy of Claude's `permissions.allow` list
- The machine-specific Claude bash grants were not copied into Codex config because their paths target a different filesystem layout and would be unsafe or invalid here
- Existing Codex config already points at the same proxy family via `https://yunwu.ai/v1` and already trusts `/home/khw` and `/home/khw/Documents/Git_repository`
