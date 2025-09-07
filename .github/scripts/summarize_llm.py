#!/usr/bin/env python3
import os
import sys
import json
import re
from urllib import request, error


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def upsert_exec_summary(md: str, summary_text: str) -> str:
    lines = md.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## executive summary"):
            header_idx = i
            break

    # Section content to insert
    new_section = ["## Executive Summary", "", summary_text.strip(), ""]

    if header_idx is None:
        # Insert after top title (first line starting with # )
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_at = i + 2 if i + 1 < len(lines) and lines[i + 1].strip() == "" else i + 1
                break
        return "\n".join(lines[:insert_at] + new_section + lines[insert_at:])

    # Replace existing Executive Summary until next H2 (## ) or end
    end_idx = len(lines)
    for j in range(header_idx + 1, len(lines)):
        if lines[j].startswith("## "):
            end_idx = j
            break
    return "\n".join(lines[:header_idx] + new_section + lines[end_idx:])


def call_openai(api_key: str, model: str, prompt: str, base_url: str = "https://api.openai.com/v1") -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior release notes editor. Produce a concise, formal Executive Summary (120-180 words) for the weekly release report, covering key features, fixes, performance work, risks/breaking changes, and upgrade guidance. Avoid emojis and marketing language."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with request.urlopen(req, timeout=60) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"].strip()
    except error.HTTPError as e:
        raise RuntimeError(f"OpenAI HTTPError: {e.read().decode('utf-8', 'ignore')}")


def extract_signal(md: str, max_chars: int = 12000) -> str:
    # Keep the most informative sections to keep prompt compact
    wanted = []
    pattern = re.compile(r"^## (Highlights|Breaking Changes|Upgrade Notes|Features & Enhancements|Bug Fixes|Performance|Model Support)", re.I)
    capture = False
    for line in md.splitlines():
        if line.startswith("# "):
            continue
        if pattern.match(line):
            capture = True
            wanted.append(line)
            continue
        if capture and line.startswith("## "):
            capture = False
        if capture:
            wanted.append(line)
    text = "\n".join(wanted).strip()
    return text[:max_chars]


def main():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY not set; skipping LLM summary.")
        return 0

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()
    target_file = os.getenv("TARGET_FILE")
    if not target_file:
        print("TARGET_FILE not set; skipping LLM summary.")
        return 0

    md = read_file(target_file)
    signal = extract_signal(md)
    if not signal:
        print("No content to summarize; skipping.")
        return 0

    prompt = (
        "Summarize the weekly release in 1–2 paragraphs. "
        "Use Chinese, and leave proper nouns untranslated (in English)"
        "Use neutral, precise tone. Mention major themes across Features, Fixes, Performance, Model/Hardware, and any breaking or upgrade notes if present. "
        "Avoid emojis; avoid restating section headers.\n\n" + signal
    )

    try:
        summary = call_openai(api_key, model, prompt, base_url)
    except Exception as e:
        print(f"LLM summarization failed: {e}")
        return 0

    new_md = upsert_exec_summary(md, summary)
    write_file(target_file, new_md)
    print("Executive Summary inserted via LLM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
