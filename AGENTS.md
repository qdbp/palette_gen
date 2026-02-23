
<!-- forgejo-assimilation:start -->
## Forgejo Swarm Integration

This repository is integrated with the Forgejo/orchd control plane.

- Use `forgejoctl` for issue workflow mutations (`claim`, `release`, `transition`, `comment`, `assign`).
- Keep issue comments terse, natural-language, and decision-focused.
- Put role directives on their own line when routing work (for example `@codex-lead design`, `@codex-dev impl`).
- Keep orchestration metadata out of issue comments; treat labels as the machine-visible control plane.
- If workflow tooling blocks progress, open a concise bug report in `main/forgejo-agent`.

Canonical swarm docs are injected into fresh agent contexts via `orchd` **Reading material** (DocPlan).
Do not maintain local “go read X” lists in issue comments or repo docs.
<!-- forgejo-assimilation:end -->
