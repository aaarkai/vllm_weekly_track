#!/bin/bash

# This script generates sample release notes based on the format from the existing file

# Set variables
TARGET_FILE="/tmp/weekly-summary.md"

# Create the directory if it doesn't exist
mkdir -p "$(dirname "$TARGET_FILE")"

# Write the header to the new file
echo "# Weekly Release Notes for vllm-project/vllm ($(date +'%Y-%m-%d'))" > "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

# Fetch all merged PRs in the last 7 days
echo "## What's Changed" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

# Create sample data based on the existing file
cat > /tmp/sample_prs.txt << 'EOF'
* [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue (#22399) by @yewentao256
* [gpt-oss] add demo tool server (#22393) by @heheda12345
* [gpt-oss] Add loop for built-in tool call (#22374) by @WoosukKwon
* [Bugfix] Make condition in triton kernel constexpr (#22370) by @gshtras
* [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks (#22368) by @LucasWilkinson
* add the codes to check AMD Instinct GPU number (#22367) by @zhangnju
* [BugFix] Fix FA2 RuntimeError when sinks is provided (#22365) by @LucasWilkinson
* [Minor] Fix type  (#22347) by @WoosukKwon
* [gpt-oss] Support chat completion api (#22342) by @WoosukKwon
* [gpt-oss] Add Tool/ConversationContext classes and harmony_utils (#22340) by @WoosukKwon
* [gpt-oss] flashinfer mxfp4 (#22339) by @zyongye
* [gpt-oss] add model to supported models doc (#22336) by @ywang96
* [gpt-oss] attention sink init fix gemini (#22335) by @zyongye
* [gpt-oss] Add openai-harmony as default dependency (#22332) by @WoosukKwon
* [gpt-oss] flashinfer attention sink init (#22330) by @zyongye
* [ROCm] Add attention sink to use_rocm_custom_paged_attention (#22329) by @WoosukKwon
* feat: Add Flashinfer MoE Support for Compressed Tensor NVFP4 (#21639) by @yewentao256
* [Feature] Non-contiguous Support for FP8 Quantization (#21961) by @yewentao256
* [Perf] Optimize `reshape_and_cache_flash` CUDA Kernel (#22036) by @yewentao256
* [Doc] Update pooling model docs (#22186) by @DarkLight1337
* [Docs] use `uv` in CPU installation docs (#22089) by @davidxia
* [Documentation] Add Voxtral to Supported Models page (#22059) by @DarkLight1337
* [Performance] Parallelize fill_bitmask to accelerate high-throughput guided decoding (#21862) by @benchislett
* [BugFix] Improve internal DP load balancing (#21617) by @njhill
* [Bugfix] Fix: Fix multi loras with tp >=2 and LRU cache (#20873) by @charent
* [Bug] Update auto_tune.sh to separate benchmarking and profiling. (#21629) by @ericehanley
EOF

# Create temporary files for each category
FEATURES_FILE=$(mktemp)
BUGS_FILE=$(mktemp)
PERF_FILE=$(mktemp)
DOCS_FILE=$(mktemp)
OTHER_FILE=$(mktemp)

# Process each PR and categorize based on title prefixes
while IFS= read -r line; do
  # Extract title from the line (everything before the first '(')
  title=$(echo "$line" | sed -E 's/\(\[#[0-9]+\)\(.*$//' | sed 's/[*] //')
  title_lower=$(echo "$title" | tr '[:upper:]' '[:lower:]')
  
  # Categorize based on title prefixes
  if [[ "$title_lower" == "feat"* ]] || [[ "$title_lower" == "[feat"* ]] || [[ "$title_lower" == "feature"* ]] || [[ "$title_lower" == "[feature"* ]] || [[ "$title_lower" == "add "* ]] || [[ "$title_lower" == "[add "* ]] || [[ "$title_lower" == "support "* ]] || [[ "$title_lower" == "[support "* ]]; then
    echo "$line" >> "$FEATURES_FILE"
  elif [[ "$title_lower" == "bug"* ]] || [[ "$title_lower" == "[bug"* ]] || [[ "$title_lower" == "fix"* ]] || [[ "$title_lower" == "[fix"* ]] || [[ "$title_lower" == "bugfix"* ]] || [[ "$title_lower" == "[bugfix"* ]] || [[ "$title_lower" == "correct"* ]] || [[ "$title_lower" == "[correct"* ]]; then
    echo "$line" >> "$BUGS_FILE"
  elif [[ "$title_lower" == "perf"* ]] || [[ "$title_lower" == "[perf"* ]] || [[ "$title_lower" == "performance"* ]] || [[ "$title_lower" == "[performance"* ]] || [[ "$title_lower" == "optimize"* ]] || [[ "$title_lower" == "[optimize"* ]] || [[ "$title_lower" == "improve"* ]] || [[ "$title_lower" == "[improve"* ]]; then
    echo "$line" >> "$PERF_FILE"
  elif [[ "$title_lower" == "doc"* ]] || [[ "$title_lower" == "[doc"* ]] || [[ "$title_lower" == "documentation"* ]] || [[ "$title_lower" == "[documentation"* ]] || [[ "$title_lower" == "readme"* ]] || [[ "$title_lower" == "[readme"* ]]; then
    echo "$line" >> "$DOCS_FILE"
  else
    echo "$line" >> "$OTHER_FILE"
  fi
done < /tmp/sample_prs.txt

# Features section
echo "### 🚀 Features & Enhancements" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
if [ -s "$FEATURES_FILE" ]; then
  cat "$FEATURES_FILE" >> "$TARGET_FILE"
else
  echo "* No features or enhancements found this week" >> "$TARGET_FILE"
fi
echo "" >> "$TARGET_FILE"

# Bug Fixes section
echo "### 🐛 Bug Fixes" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
if [ -s "$BUGS_FILE" ]; then
  cat "$BUGS_FILE" >> "$TARGET_FILE"
else
  echo "* No bug fixes found this week" >> "$TARGET_FILE"
fi
echo "" >> "$TARGET_FILE"

# Performance Improvements section
echo "### ⚡ Performance Improvements" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
if [ -s "$PERF_FILE" ]; then
  cat "$PERF_FILE" >> "$TARGET_FILE"
else
  echo "* No performance improvements found this week" >> "$TARGET_FILE"
fi
echo "" >> "$TARGET_FILE"

# Documentation section
echo "### 📚 Documentation" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
if [ -s "$DOCS_FILE" ]; then
  cat "$DOCS_FILE" >> "$TARGET_FILE"
else
  echo "* No documentation updates found this week" >> "$TARGET_FILE"
fi
echo "" >> "$TARGET_FILE"

# Other PRs section
echo "### 📦 Other Changes" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
if [ -s "$OTHER_FILE" ]; then
  cat "$OTHER_FILE" >> "$TARGET_FILE"
else
  echo "* No other changes found this week" >> "$TARGET_FILE"
fi
echo "" >> "$TARGET_FILE"

# Contributors section (sample)
echo "## Contributors" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
echo "@yewentao256, @heheda12345, @WoosukKwon, @gshtras, @LucasWilkinson, @zhangnju, @zyongye, @ywang96, @Isotr0py, @mgoin, @ruisearch42, @lsy323, @jeejeelee, @youkaichao, @NickLucche, @DarkLight1337, @tlrmchlsmth, @22quinn, @skyloevil, @andyxning, @davidxia, @hmellor, @tjtanaa, @vllmellm, @benchislett, @njhill, @dtrifiro, @sanchit-gandhi, @Abirdcfly, @lengrongfu, @elvischenv, @dsikka, @sarckk, @cyang49, @zhxchen17, @dougbtv, @varun-sundar-rabindranath, @zou3219, @wuhang2014, @eicherseiji, @sdavidbd, @kylesayrs, @vanbasten23, @yma11, @LopezCastroRoberto, @ericehanley, @morgendave, @noooop, @JartX, @alyosha-swamy, @yeqcharlotte, @ruisearch42, @fhl2000, @CLFutureX, @weixiao-huang, @SageMoore, @sidhpurwala-huzaifa, @ilmarkov, @n0gu-furiosa, @anijain2305, @TankNee, @Josephasafg, @vadiklyutiy, @mickaelseznec, @tdoublep, @MatthewBonanni, @amirkl94, @Oliver-ss, @chenxi-yang, @xiszishu, @bigPYJ1151, @tlipoca9, @zRzRzRzRzRzRzR, @nvpohan, @Alexei-V-Ivanov-AMD, @simon-mo, @ahengljh, @david6666666, @hsliuustc0106, @Abatom, @lk-chen, @linzebing, @bbeckca, @eric-haibin-lin, @charent, @TheEpicDolphin" >> "$TARGET_FILE"

# Clean up temporary files
rm -f "$FEATURES_FILE" "$BUGS_FILE" "$PERF_FILE" "$DOCS_FILE" "$OTHER_FILE" /tmp/sample_prs.txt

# Display the generated file
cat "$TARGET_FILE"