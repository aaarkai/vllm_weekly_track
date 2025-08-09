# vLLM Weekly Tracker

This repository automatically tracks weekly changes in the [vLLM project](https://github.com/vllm-project/vllm) by generating release notes for merged pull requests.

## How It Works

Every Friday, a GitHub Action runs to:

1. Fetch all merged pull requests from the vLLM repository in the last 7 days
2. Categorize the PRs into:
   - 🚀 Features & Enhancements
   - 🐛 Bug Fixes
   - ⚡ Performance Improvements
   - 📚 Documentation
   - 📦 Other Changes
3. Generate a release notes document with these categorized PRs
4. Commit the new release notes to this repository

## Example Output

The generated release notes look like this:

```markdown
# Weekly Release Notes for vllm-project/vllm (2025-08-09)

## What's Changed

### 🚀 Features & Enhancements

* feat: Add Flashinfer MoE Support for Compressed Tensor NVFP4 (#21639) by @yewentao256
* [Feature] Non-contiguous Support for FP8 Quantization (#21961) by @yewentao256

### 🐛 Bug Fixes

* [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue (#22399) by @yewentao256
* [Bugfix] Make condition in triton kernel constexpr (#22370) by @gshtras

### ⚡ Performance Improvements

* [Perf] Optimize `reshape_and_cache_flash` CUDA Kernel (#22036) by @yewentao256

### 📚 Documentation

* [Doc] Update pooling model docs (#22186) by @DarkLight1337

### 📦 Other Changes

* [gpt-oss] add demo tool server (#22393) by @heheda12345
* [gpt-oss] Add loop for built-in tool call (#22374) by @WoosukKwon

## Contributors

@yewentao256, @heheda12345, @WoosukKwon, @gshtras, ...
```

## Accessing the Release Notes

You can find the weekly release notes in the [`summaries/`](summaries/) directory, with filenames following the pattern `weekly-YYYY-MM-DD.md`.

## Customization

If you want to run this for your own repository:

1. Fork this repository
2. Update the `UPSTREAM_REPO` variable in [`.github/workflows/gemini_sum.yml`](.github/workflows/gemini_sum.yml) to point to your repository
3. Adjust the cron schedule if needed

## Local Testing

To test the release notes generation locally:

```bash
./test_release_notes.sh
```

This will generate a sample release notes file based on recent vLLM PRs.