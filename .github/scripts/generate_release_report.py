#!/usr/bin/env python3
import os
import sys
import json
import tempfile
import subprocess
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter


def run(cmd, env=None):
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout


def iso_date_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.date().isoformat()


def fetch_prs(upstream_repo: str, since_days: int = 7):
    since = iso_date_days_ago(since_days)
    owner, name = upstream_repo.split("/")
    query_str = (
        "query($q:String!, $cursor:String) {\n"
        "  search(query:$q, type:ISSUE, first:100, after:$cursor) {\n"
        "    issueCount\n"
        "    pageInfo { hasNextPage endCursor }\n"
        "    nodes {\n"
        "      ... on PullRequest {\n"
        "        number title body url mergedAt\n"
        "        author { login }\n"
        "        labels(first:50) { nodes { name } }\n"
        "        additions deletions changedFiles\n"
        "        files(first:100) { nodes { path } }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}"
    )
    search_query = f"repo:{owner}/{name} is:pr is:merged merged:>={since}"

    prs = []
    cursor = None
    while True:
        cmd = [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query_str}",
            "-f",
            f"q={search_query}",
        ]
        if cursor:
            cmd += ["-f", f"cursor={cursor}"]
        out = run(cmd)
        data = json.loads(out)
        s = data["data"]["search"]
        for n in s["nodes"]:
            if n is None:
                continue
            # Normalize
            labels = [l["name"].lower() for l in n.get("labels", {}).get("nodes", [])]
            files = [x["path"] for x in n.get("files", {}).get("nodes", [])]
            prs.append({
                "number": n.get("number"),
                "title": n.get("title") or "",
                "body": n.get("body") or "",
                "url": n.get("url"),
                "mergedAt": n.get("mergedAt"),
                "author": (n.get("author") or {}).get("login"),
                "labels": labels,
                "additions": n.get("additions") or 0,
                "deletions": n.get("deletions") or 0,
                "changedFiles": n.get("changedFiles") or 0,
                "files": files,
            })
        if not s["pageInfo"]["hasNextPage"]:
            break
        cursor = s["pageInfo"]["endCursor"]

    return prs


def pick_section(pr):
    title = (pr["title"] or "").lower()
    body = (pr["body"] or "").lower()
    labels = set(pr["labels"])

    def has_label(prefix):
        return any(l.startswith(prefix) or l == prefix for l in labels)

    # Breaking / Deprecations
    if any(k in title or k in body for k in ["breaking", "deprecat", "drop support", "remov", "incompat"]):
        return "Breaking Changes"
    if any(l in labels for l in ["breaking", "breaking-change", "deprecation"]):
        return "Breaking Changes"

    # Bug fixes
    if any(l in labels for l in ["bug", "bugfix", "fix"]):
        return "Bug Fixes"
    if title.startswith("fix") or title.startswith("bugfix"):
        return "Bug Fixes"

    # Performance
    if any(l in labels for l in ["performance", "perf", "speed"]):
        return "Performance"
    if any(k in title for k in ["perf", "optimiz", "speed", "throughput"]):
        return "Performance"

    # Features & Enhancements
    if any(l in labels for l in ["feature", "enhancement", "feat"]):
        return "Features & Enhancements"
    if any(title.startswith(k) for k in ["feat", "add ", "support "]):
        return "Features & Enhancements"

    # Model Support
    model_keywords = ["model", "llama", "qwen", "glm", "mamba", "gpt", "minicpm", "mixtral"]
    if any(l in labels for l in ["model", "models"]):
        return "Model Support"
    if any(k in title for k in model_keywords):
        return "Model Support"

    # Hardware & Backend
    hw_keywords = ["rocm", "cuda", "nvidia", "tpu", "xpu", "metal", "cpu", "kernel"]
    if any(l in labels for l in ["cuda", "rocm", "tpu", "xpu", "nvidia", "hardware", "backend"]):
        return "Hardware & Backend"
    if any(k in title for k in hw_keywords):
        return "Hardware & Backend"

    # Refactoring & Core
    if any(l in labels for l in ["refactor", "core", "api"]):
        return "Refactoring & Core"
    if any(k in title for k in ["refactor", "cleanup", "remove", "simplify", "api"]):
        return "Refactoring & Core"

    # Build, CI & Testing
    if any(l in labels for l in ["ci", "build", "testing", "test"]):
        return "Build, CI & Testing"
    if any(k in title for k in ["ci", "build", "test", "pytest", "workflow"]):
        return "Build, CI & Testing"

    # Documentation
    if any(l in labels for l in ["docs", "documentation"]):
        return "Documentation"
    if any(k in title for k in ["doc", "docs", "readme"]):
        return "Documentation"

    return "Miscellaneous"


def extract_upgrade_notes(body: str):
    if not body:
        return []
    notes = []
    for line in body.splitlines():
        l = line.strip()
        if not l:
            continue
        if any(k in l.lower() for k in ["breaking", "migration", "upgrade", "note:"]):
            # Keep it concise
            notes.append(l[:300])
    return notes[:3]


def format_pr_line(pr):
    return f"* {pr['title']} ([#{pr['number']}]({pr['url']})) by @{pr['author']}"


def build_report(upstream_repo: str, prs: list) -> str:
    if not prs:
        today = datetime.now().date().isoformat()
        return (
            f"# Weekly Release Report for {upstream_repo} ({today})\n\n"
            "No merged pull requests found in the last 7 days.\n"
        )

    # Contributors and counts
    contributors = sorted({pr["author"] for pr in prs if pr.get("author")})
    section_map = defaultdict(list)
    for pr in prs:
        section_map[pick_section(pr)].append(pr)

    # Highlights: pick up to 5 notable PRs
    def score(pr):
        s = 0
        sec = pick_section(pr)
        if sec in ("Breaking Changes", "Features & Enhancements", "Performance"):
            s += 5
        s += min(pr.get("additions", 0) // 200, 5)
        s += min(pr.get("changedFiles", 0) // 5, 4)
        if any(k in (pr.get("title") or "").lower() for k in ["support", "api", "optimiz", "kernel", "backend"]):
            s += 2
        return s

    highlights = sorted(prs, key=score, reverse=True)[:5]

    # Upgrade notes and breaking
    upgrade_notes = []
    breaking_list = []
    for pr in prs:
        if pick_section(pr) == "Breaking Changes":
            breaking_list.append(pr)
        upgrade_notes.extend(extract_upgrade_notes(pr.get("body") or ""))
    # Deduplicate notes
    seen = set()
    upgrade_notes = [n for n in upgrade_notes if not (n in seen or seen.add(n))][:8]

    today = datetime.now().date().isoformat()
    lines = []
    lines.append(f"# Weekly Release Report for {upstream_repo} ({today})")
    lines.append("")
    lines.append(
        f"This week merged {len(prs)} PRs from {len(contributors)} contributors. "
        f"Key areas: features {len(section_map['Features & Enhancements'])}, "
        f"fixes {len(section_map['Bug Fixes'])}, performance {len(section_map['Performance'])}."
    )
    lines.append("")

    # Deterministic Executive Summary (fallback when no LLM available)
    # Top sections by volume
    section_counts = {k: len(v) for k, v in section_map.items()}
    top_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
    top_sections = [s for s in top_sections if s[1] > 0][:3]

    # Top directories by changed files across PRs
    dir_counter = Counter()
    for pr in prs:
        for p in pr.get("files", []) or []:
            top = p.split("/", 1)[0]
            if top:
                dir_counter[top] += 1
    top_dirs = ", ".join(f"{d} ({c})" for d, c in dir_counter.most_common(3)) or "n/a"

    lines.append("## Executive Summary")
    lines.append("")
    if top_sections:
        sec_text = ", ".join(f"{name.lower()} {count}" for name, count in top_sections)
    else:
        sec_text = "balanced updates across areas"
    lines.append(
        f"Focus this week centers on {sec_text}. Notable activity spans {top_dirs}. "
        f"See highlights and sections below for details, risks, and upgrade notes."
    )
    lines.append("")

    # Highlights
    lines.append("## Highlights")
    lines.append("")
    if highlights:
        for pr in highlights:
            lines.append(format_pr_line(pr))
    else:
        lines.append("*No highlights identified this week.*")
    lines.append("")

    # Standard sections in desired order
    ordered_sections = [
        "Features & Enhancements",
        "Bug Fixes",
        "Performance",
        "Model Support",
        "Hardware & Backend",
        "Refactoring & Core",
        "Build, CI & Testing",
        "Documentation",
        "Miscellaneous",
    ]
    for sec in ordered_sections:
        lines.append(f"## {sec}")
        lines.append("")
        items = section_map.get(sec, [])
        if items:
            for pr in items:
                lines.append(format_pr_line(pr))
        else:
            lines.append(f"*No {sec.lower()} this week.*")
        lines.append("")

    # Breaking Changes
    lines.append("## Breaking Changes")
    lines.append("")
    if breaking_list:
        for pr in breaking_list:
            lines.append(format_pr_line(pr))
    else:
        lines.append("*No breaking changes detected.*")
    lines.append("")

    # Upgrade Notes
    lines.append("## Upgrade Notes")
    lines.append("")
    if upgrade_notes:
        for n in upgrade_notes:
            lines.append(f"- {n}")
    else:
        lines.append("- No special upgrade notes this week.")
    lines.append("")

    # Contributors
    lines.append("## Contributors")
    lines.append("")
    if contributors:
        lines.append(", ".join(f"@{c}" for c in contributors))
    else:
        lines.append("(none)")
    lines.append("")

    return "\n".join(lines)


def main():
    upstream_repo = os.getenv("UPSTREAM_REPO", "vllm-project/vllm")
    target_file = os.getenv("TARGET_FILE")
    if not target_file:
        # Default path
        target_file = os.path.join("summaries", f"release-{datetime.now().date().isoformat()}.md")

    try:
        prs = fetch_prs(upstream_repo)
    except Exception as e:
        print(f"Error fetching PRs: {e}", file=sys.stderr)
        sys.exit(1)

    report = build_report(upstream_repo, prs)
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote release report to {target_file}")


if __name__ == "__main__":
    main()
