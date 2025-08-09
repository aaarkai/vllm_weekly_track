# vLLM Weekly Tracker

This repository automatically tracks weekly changes in the [vLLM project](https://github.com/vllm-project/vllm) by generating structured release notes for merged pull requests.

## How It Works

Every Friday, a GitHub Action runs to:

1. Fetch all merged pull requests from the vLLM repository in the last 7 days
2. Categorize the PRs into detailed sections based on their content:
   - ✨ Features & Enhancements
   - 🐛 Bug Fixes
   - ⚡️ Performance
   - 🤖 Model Support
   - 🔌 Hardware & Backend
   - ⚙️ Refactoring & Core
   - 🔧 Build, CI & Testing
   - 📚 Documentation
   - 📦 Miscellaneous
3. Generate a comprehensive release notes document with these categorized PRs
4. Include a list of all contributors for the week
5. Commit the new release notes to this repository

## Example Output

The generated release notes provide a well-organized view of all changes:

```markdown
# Weekly Release Notes for vllm-project/vllm (2025-08-09)

## What's Changed

### ✨ Features & Enhancements

* Add H20-3e fused MoE kernel tuning configs for GLM-4.5 ([#22433](https://github.com/vllm-project/vllm/pull/22433)) by @JaceyShao
* add the codes to check AMD Instinct GPU number ([#22367](https://github.com/vllm-project/vllm/pull/22367)) by @zhangnju

### 🐛 Bug Fixes

* [Bugfix] Fix failing GPT-OSS initialization test ([#22557](https://github.com/vllm-project/vllm/pull/22557)) by @Isotr0py
* [Bugfix] Fix CI moe kernel failure ([#22556](https://github.com/vllm-project/vllm/pull/22556)) by @jeejeelee

### ⚡️ Performance

* [PERF] Use pybase64 to more quickly decode prompt embeddings ([#22469](https://github.com/vllm-project/vllm/pull/22469)) by @qthequartermasterman
* Optimize MiniCPMO mask creation with vectorized implementation ([#22464](https://github.com/vllm-project/vllm/pull/22464)) by @skyloevil

### 🤖 Model Support

* [gpt-oss] guard import when triton kernel is not installed ([#22529](https://github.com/vllm-project/vllm/pull/22529)) by @zyongye
* GLM-4.5V with new class name at transformers ([#22520](https://github.com/vllm-project/vllm/pull/22520)) by @zRzRzRzRzRzRzR

## Contributors

@0xjunhao, @22quinn, @Abatom, @Abirdcfly, @CLFutureX, @DarkLight1337, ...
```

The full output includes all merged PRs organized into logical categories, making it easy to quickly understand the key developments in vLLM each week.

## Accessing the Release Notes

You can find the weekly release notes in the [`summaries/`](summaries/) directory, with filenames following the pattern `weekly-YYYY-MM-DD.md`.

Each release note includes:
- Categorized pull requests with direct links to the PRs
- Author attribution for each contribution
- A comprehensive list of all contributors for the week

## Customization

To adapt this for tracking a different repository:

1. Fork this repository
2. Update the `UPSTREAM_REPO` variable in [`.github/workflows/gemini_sum.yml`](.github/workflows/gemini_sum.yml) to point to your target repository
3. Adjust the cron schedule in the workflow file if needed
4. Modify the categorization patterns in the script if your repository uses different naming conventions

## Local Testing

For local development and testing:

```bash
./test_release_notes.sh
```

This will generate a sample release notes file based on recent vLLM PRs without requiring GitHub CLI authentication.