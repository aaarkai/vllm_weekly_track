# Repository Guidelines

## Project Structure & Module Organization
- Root docs: `README.md` explains goals and outputs.
- Workflows: `.github/workflows/`
  - `gemini_sum.yml` generates structured notes weekly.
  - `fetch_prs.yml` writes a chronological PR list.
- Scripts: `.github/scripts/generate-summary.sh` (Bash; uses `gh` and `jq`).
- Outputs: `summaries/`
  - `weekly-YYYY-MM-DD.md`
  - `weekly-PR-list-YYYY-MM-DD.md`

## Build, Test, and Development Commands
- Prereqs: GitHub CLI (`gh`), `jq`, Bash.
- Auth: `gh auth login` or `export GH_TOKEN=YOUR_TOKEN` (repo read).
- Generate structured notes locally:
  - `UPSTREAM_REPO="vllm-project/vllm" TARGET_FILE="summaries/weekly-$(date +%F).md" .github/scripts/generate-summary.sh`
- Generate PR list locally (mirrors workflow):
  - `gh pr list --repo vllm-project/vllm --limit 500 --search "is:pr is:merged merged:>=$(date -v-7d +%Y-%m-%d)" --json number,title,author,url --template '{{range .}}* {{.title}} (#{{.number}}) by @{{.author.login}}\n{{end}}' > summaries/weekly-PR-list-$(date +%F).md`
- Lint script: `shellcheck .github/scripts/generate-summary.sh`

## Coding Style & Naming Conventions
- Bash: 2-space indentation; `set -euo pipefail`; prefer clear variable names; guard env inputs.
- File naming: keep lowercase-kebab; summaries follow `weekly-YYYY-MM-DD.md` and `weekly-PR-list-YYYY-MM-DD.md`.
- YAML: compact, comments for non-obvious steps; pin major actions (e.g., `actions/checkout@v4`).

## Testing Guidelines
- Dry run locally using commands above; verify sections exist and links render.
- Ensure contributors list is deduplicated and present.
- Avoid committing generated files from local tests unless intentional; scheduled workflows commit on Fridays.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `ci:`, `chore:`.
  - Example: `docs: add weekly summary for 2025-08-09`.
- PRs should include: purpose, linked issues, sample output snippet or screenshot, and note of affected files/paths.
- Keep changes scoped (scripts, workflows, or docs). Add rationale for categorization rules.

## Security & Configuration Tips
- Never commit tokens. Use `secrets.GITHUB_TOKEN` in workflows; local `GH_TOKEN` must be least-privileged (read).
- To track a different repo, edit `UPSTREAM_REPO` in both workflows and the local command; adjust cron if needed.
