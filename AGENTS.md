
<!-- forgejo-assimilation:start -->
## Forgejo Swarm Integration

This repository is integrated with the Forgejo/orchd control plane.

- Use `forgejoctl` for issue workflow mutations (`claim`, `release`, `transition`, `comment`, `assign`).
- Keep issue comments terse, natural-language, and decision-focused.
- Use role directives in issue text when routing work (for example `@codex-lead design`, `@codex-dev impl`).
- Keep orchestration metadata out of issue comments; treat labels as the machine-visible control plane.
- If workflow tooling blocks progress, open a concise bug report in `main/forgejo-agent`.

Reference docs:
- `/home/main/forgejo-agent/docs/AGENT_WORKFLOW.md`
- `/home/main/forgejo-agent/docs/ORG_CHART.md`
- `/home/main/forgejo-agent/docs/REPO_ASSIMILATION.md`
<!-- forgejo-assimilation:end -->
