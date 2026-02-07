# CLAUDE.md

## Project Overview

**vLLM Weekly Tracker** is an automated system that monitors weekly changes in the [vLLM project](https://github.com/vllm-project/vllm). Every Friday, GitHub Actions fetch merged PRs from the past 7 days, categorize them, and generate structured markdown reports committed to the `summaries/` directory.

The project has no build step, no package manager, and no runtime dependencies beyond the GitHub CLI (`gh`), `jq`, Python 3 (stdlib only), and Bash.

## Repository Structure

```
.
├── CLAUDE.md                   # This file
├── AGENTS.md                   # Developer guidelines (legacy)
├── README.md                   # Project overview and usage
├── .gitignore                  # Minimal: *.tmp, *.temp, *.log, .DS_Store
├── .github/
│   ├── workflows/
│   │   ├── fetch_prs.yml           # Workflow: chronological PR list
│   │   ├── gemini_sum.yml          # Workflow: categorized release notes
│   │   └── release_report.yml      # Workflow: formal report + optional LLM summary
│   └── scripts/
│       ├── generate-summary.sh         # Bash: fetch & categorize PRs via gh CLI + jq
│       ├── generate_release_report.py  # Python: GraphQL-based formal report generator
│       └── summarize_llm.py            # Python: optional OpenAI-powered executive summary
└── summaries/                  # Generated output (74+ markdown files)
    ├── weekly-YYYY-MM-DD.md          # Categorized release notes
    ├── weekly-PR-list-YYYY-MM-DD.md  # Chronological PR list
    └── release-YYYY-MM-DD.md         # Formal release report
```

## Key Components

### Workflows (`.github/workflows/`)

Three GitHub Actions run sequentially every Friday:

| Workflow | Cron (UTC) | Script | Output |
|---|---|---|---|
| `fetch_prs.yml` | `0 1 * * 5` | Inline bash (`gh pr list`) | `weekly-PR-list-YYYY-MM-DD.md` |
| `gemini_sum.yml` | `0 2 * * 5` | `generate-summary.sh` | `weekly-YYYY-MM-DD.md` |
| `release_report.yml` | `0 3 * * 5` | `generate_release_report.py` + `summarize_llm.py` | `release-YYYY-MM-DD.md` |

All workflows use `stefanzweifel/git-auto-commit-action@v5` to commit results, require `contents: write` permission, and support `workflow_dispatch` for manual triggers.

### Scripts (`.github/scripts/`)

**`generate-summary.sh`** (Bash)
- Uses `gh pr list` with `--json` to fetch merged PRs from the last 7 days
- Categorizes PRs via case-insensitive pattern matching on title prefixes (e.g., `[Bugfix]`, `[FEAT]`, `[PERF]`)
- Falls back to keyword matching (`fix:`, `feat:`, `optimize`, etc.)
- Outputs 9 emoji-prefixed sections plus a deduplicated contributors list
- Uses `set -euo pipefail`, temp dir with cleanup trap

**`generate_release_report.py`** (Python 3, stdlib only)
- Uses GitHub GraphQL API via `gh api graphql` for richer PR data (labels, files, additions/deletions)
- Categorizes using both labels and title keywords with priority ordering (Breaking > Bugs > Perf > Features > ...)
- Generates: executive summary, highlights (top 5 by scoring heuristic), all sections, breaking changes, upgrade notes, contributors
- Key functions: `fetch_prs()`, `pick_section()`, `build_report()`, `extract_upgrade_notes()`

**`summarize_llm.py`** (Python 3, stdlib only)
- Optional: runs only if `OPENAI_API_KEY` is set; exits gracefully otherwise
- Calls OpenAI API (default `gpt-4o-mini`, temp 0.2) to generate a Chinese-language executive summary
- Inserts/replaces `## Executive Summary` section in the release report
- Key functions: `call_openai()`, `extract_signal()`, `upsert_exec_summary()`

## Environment Variables

| Variable | Required | Used By | Description |
|---|---|---|---|
| `UPSTREAM_REPO` | Yes | All scripts | Target repo (default: `vllm-project/vllm`) |
| `TARGET_FILE` | Yes | All scripts | Output file path |
| `GH_TOKEN` | Yes | All scripts | GitHub token (read access; `secrets.GITHUB_TOKEN` in CI) |
| `OPENAI_API_KEY` | No | `summarize_llm.py` | Enables LLM executive summary |
| `OPENAI_MODEL` | No | `summarize_llm.py` | Model name (default: `gpt-4o-mini`) |
| `OPENAI_API_BASE` | No | `summarize_llm.py` | API base URL (default: `https://api.openai.com/v1`) |

## Development Commands

### Prerequisites

- GitHub CLI (`gh`) authenticated: `gh auth login` or `export GH_TOKEN=...`
- `jq` installed (for `generate-summary.sh`)
- Python 3 (no pip packages needed)
- Bash

### Run Scripts Locally

```bash
# Categorized release notes
UPSTREAM_REPO="vllm-project/vllm" TARGET_FILE="summaries/weekly-$(date +%F).md" \
  .github/scripts/generate-summary.sh

# Formal release report
UPSTREAM_REPO="vllm-project/vllm" TARGET_FILE="summaries/release-$(date +%F).md" \
  .github/scripts/generate_release_report.py

# LLM summary (requires OPENAI_API_KEY)
OPENAI_API_KEY="sk-..." TARGET_FILE="summaries/release-$(date +%F).md" \
  .github/scripts/summarize_llm.py
```

### Linting

```bash
shellcheck .github/scripts/generate-summary.sh
```

There is no Python linter configured. Python scripts use only the standard library.

### Testing

No formal test framework exists. Verify scripts manually:

1. Run the script locally and inspect the generated markdown
2. Check that all 9 category sections are present
3. Confirm PR links render correctly
4. Ensure the contributors list is deduplicated
5. Do not commit local test output unless intentional

## PR Categorization Logic

Both scripts classify PRs into 9 sections. The priority order (highest first):

1. **Breaking Changes** - title/body keywords: `breaking`, `deprecat`, `drop support`, `incompat`
2. **Bug Fixes** - tags: `[Bugfix]`, `[Bug]`, `[fix]`; labels: `bug`, `bugfix`, `fix`
3. **Performance** - tags: `[PERF]`, `[Performance]`; keywords: `optimiz`, `speed`, `throughput`
4. **Features & Enhancements** - tags: `[FEAT]`, `[Feature]`; keywords: `add`, `support`
5. **Model Support** - tags: `[Model]`, `[gpt-oss]`; keywords: `llama`, `qwen`, `glm`, `mamba`
6. **Hardware & Backend** - tags: `[ROCm]`, `[TPU]`, `[XPU]`, `[NVIDIA]`, `[CUDA]`
7. **Refactoring & Core** - tags: `[Core]`, `[Refactor]`, `[Frontend]`, `[API]`, `[Kernel]`
8. **Build, CI & Testing** - tags: `[CI]`, `[Build]`, `[Test]`
9. **Documentation** - tags: `[Doc]`, `[Docs]`
10. **Miscellaneous** - everything else

## Coding Conventions

### Bash
- `set -euo pipefail` at the top of every script
- 2-space indentation
- Validate required environment variables with guard clauses
- Use temp directories with `trap` cleanup
- Case-insensitive matching via `shopt -s nocasematch`

### Python
- Standard library only (no third-party packages)
- Functions use `snake_case`
- Type hints on function signatures (e.g., `def iso_date_days_ago(days: int) -> str:`)
- Subprocess calls wrapped in a `run()` helper with error checking
- Scripts are executable with `#!/usr/bin/env python3` shebangs

### YAML Workflows
- Pin action versions to major tags (`actions/checkout@v4`, `stefanzweifel/git-auto-commit-action@v5`)
- Use `env:` blocks for variable passing
- Comment non-obvious steps

### File Naming
- Output files: `weekly-YYYY-MM-DD.md`, `weekly-PR-list-YYYY-MM-DD.md`, `release-YYYY-MM-DD.md`
- Scripts: lowercase with hyphens (bash) or underscores (python)
- All dates in ISO 8601 format (`YYYY-MM-DD`)

## Commit Conventions

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `ci:`, `chore:`
- Automated commits use `docs:` prefix (e.g., `docs: Add weekly summary for summaries/weekly-2025-08-09.md`)
- Keep changes scoped: scripts, workflows, or docs in separate commits
- Never commit secrets or tokens

## Security Notes

- `GH_TOKEN` must be least-privileged (read-only repo access)
- Use `secrets.GITHUB_TOKEN` in workflows, never hardcoded values
- `OPENAI_API_KEY` is an optional secret; `summarize_llm.py` exits gracefully without it
- The `.gitignore` excludes `*.tmp`, `*.temp`, `*.log`, `.DS_Store`

## Common Modification Tasks

**Add a new categorization rule:** Edit the `case` block in `generate-summary.sh:77-101` and the `pick_section()` function in `generate_release_report.py:88-152`. Both must stay in sync.

**Track a different repository:** Change `UPSTREAM_REPO` in all three workflow files and in local commands.

**Add a new output format:** Create a new script in `.github/scripts/`, a new workflow in `.github/workflows/`, and follow the existing pattern (fetch, process, commit via auto-commit action).

**Modify the LLM prompt:** Edit `summarize_llm.py:109-114` (user prompt) or line 53 (system prompt).

**Change the schedule:** Edit the `cron:` field in the relevant workflow YAML. Current schedule is every Friday staggered across 01:00-03:00 UTC.
