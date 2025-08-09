#!/bin/bash

# =================================================================
# generate-summary.sh
#
# Fetches merged PRs from the last 7 days, categorizes them based
# on titles, and generates a Markdown summary file.
#
# Required Environment Variables:
# - UPSTREAM_REPO: The repository to fetch PRs from (e.g., "vllm-project/vllm").
# - TARGET_FILE: The output path for the summary (e.g., "summaries/weekly-2025-08-08.md").
# - GH_TOKEN: A GitHub token with read access to the repo.
# =================================================================

# --- Script Setup ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u
# Pipes will fail if any command in the pipe fails.
set -o pipefail

# --- Validate Inputs ---
if [[ -z "$UPSTREAM_REPO" || -z "$TARGET_FILE" ]]; then
    echo "Error: UPSTREAM_REPO and TARGET_FILE environment variables must be set." >&2
    exit 1
fi

# --- Temporary File Management ---
# Create a temporary directory to store intermediate files.
TEMP_DIR=$(mktemp -d)
# Set a trap to automatically clean up the temporary directory on script exit.
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "Temporary directory created at: $TEMP_DIR"

# --- Data Fetching ---
echo "Fetching merged PRs from '$UPSTREAM_REPO' for the last 7 days..."
# Fetch all required data in a single API call and store in a JSON file.
gh pr list --repo "$UPSTREAM_REPO" --limit 500 \
  --search "is:pr is:merged merged:>=$(date -d '7 days ago' +%Y-%m-%d)" \
  --json number,title,author,url > "$TEMP_DIR/prs.json"

# Check if any PRs were found.
if ! jq -e '. | length > 0' "$TEMP_DIR/prs.json" > /dev/null; then
    echo "No merged PRs found in the last 7 days. Exiting."
    # Create a minimal report and exit gracefully.
    mkdir -p "$(dirname "$TARGET_FILE")"
    echo "# Weekly Release Notes for $UPSTREAM_REPO ($(date +'%Y-%m-%d'))" > "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    echo "## What's Changed" >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    echo "*No merged PRs found this week.*" >> "$TARGET_FILE"
    exit 0
fi

# Format the PRs into a readable list for classification.
jq -r '.[] | "* \(.title) ([#\(.number)](\(.url))) by @\(.author.login)"' "$TEMP_DIR/prs.json" > "$TEMP_DIR/all_prs_formatted.txt"

# --- Categorization ---
echo "Categorizing PRs..."

# Define file paths for each category within the temporary directory.
FEATURES_FILE="$TEMP_DIR/features.txt"
BUGS_FILE="$TEMP_DIR/bugs.txt"
PERF_FILE="$TEMP_DIR/perf.txt"
DOCS_FILE="$TEMP_DIR/docs.txt"
REFACTOR_FILE="$TEMP_DIR/refactor.txt"
BUILD_FILE="$TEMP_DIR/build.txt"
HARDWARE_FILE="$TEMP_DIR/hardware.txt"
MODEL_FILE="$TEMP_DIR/model.txt"
MISC_FILE="$TEMP_DIR/misc.txt"

# Enable case-insensitive matching for the 'case' statement.
shopt -s nocasematch

while IFS= read -r line; do
    title=$(echo "$line" | sed -E 's/ \(\#.*$//' | sed 's/^[*] //')
    case "$title" in
        # Priority 1: Match explicit tags first
        "[Bugfix]"* | "[Bug]"* | "[fix]"* )      echo "$line" >> "$BUGS_FILE" ;;
        "[FEAT]"* | "[Feature]"* )               echo "$line" >> "$FEATURES_FILE" ;;
        "[PERF]"* | "[Performance]"* )           echo "$line" >> "$PERF_FILE" ;;
        "[Doc]"* | "[Docs]"* )                    echo "$line" >> "$DOCS_FILE" ;;
        "[CI]"* | "[Build]"* | "[Test]"* )        echo "$line" >> "$BUILD_FILE" ;;
        "[Core]"* | "[Refactor]"* | "[Frontend]"* | "[API]"* | "[Kernel]"* ) echo "$line" >> "$REFACTOR_FILE" ;;
        "[ROCm]"* | "[TPU]"* | "[XPU]"* | "[NVIDIA]"* | "[CUDA]"* ) echo "$line" >> "$HARDWARE_FILE" ;;
        "[Model]"* | "[gpt-oss]"* | "qwen"* | "glm"* | "llama"* | "mamba"* ) echo "$line" >> "$MODEL_FILE" ;;
        "[Misc]"* )                              echo "$line" >> "$MISC_FILE" ;;

        # Priority 2: Match common keywords if no tag is found
        "fix:"* | "fix "* )                      echo "$line" >> "$BUGS_FILE" ;;
        "feat:"* | "add "* | "support "* )        echo "$line" >> "$FEATURES_FILE" ;;
        "optimize"* | "speed up"* )              echo "$line" >> "$PERF_FILE" ;;
        "doc:"* | "docs:"* | "documentation"*)    echo "$line" >> "$DOCS_FILE" ;;
        "refactor"* | "simplify"* | "remove "* )  echo "$line" >> "$REFACTOR_FILE" ;;
        "update dependency"* | "update transformers"* ) echo "$line" >> "$BUILD_FILE" ;;

        # Priority 3: Default fallback
        * )                                      echo "$line" >> "$MISC_FILE" ;;
    esac
done < "$TEMP_DIR/all_prs_formatted.txt"

# Disable case-insensitive matching.
shopt -u nocasematch

# --- Report Generation ---
echo "Generating Markdown report at '$TARGET_FILE'..."

# Create the output directory if it doesn't exist.
mkdir -p "$(dirname "$TARGET_FILE")"

# Write the main header.
echo "# Weekly Release Notes for $UPSTREAM_REPO ($(date +'%Y-%m-%d'))" > "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
echo "## What's Changed" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

# Helper function to generate a section in the report.
generate_section() {
    local title="$1" data_file="$2" empty_message="$3"
    echo "$title" >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    if [ -s "$data_file" ]; then
        cat "$data_file" >> "$TARGET_FILE"
    else
        echo "$empty_message" >> "$TARGET_FILE"
    fi
    echo "" >> "$TARGET_FILE"
}

# Generate each section of the report.
generate_section "### ✨ Features & Enhancements" "$FEATURES_FILE" "*No new features this week.*"
generate_section "### 🐛 Bug Fixes" "$BUGS_FILE" "*No bug fixes this week.*"
generate_section "### ⚡️ Performance" "$PERF_FILE" "*No performance improvements this week.*"
generate_section "### 🤖 Model Support" "$MODEL_FILE" "*No new model support updates this week.*"
generate_section "### 🔌 Hardware & Backend" "$HARDWARE_FILE" "*No hardware or backend updates this week.*"
generate_section "### ⚙️ Refactoring & Core" "$REFACTOR_FILE" "*No core refactoring this week.*"
generate_section "### 🔧 Build, CI & Testing" "$BUILD_FILE" "*No build, CI, or testing changes this week.*"
generate_section "### 📚 Documentation" "$DOCS_FILE" "*No documentation updates this week.*"
generate_section "### 📦 Miscellaneous" "$MISC_FILE" "*No miscellaneous changes this week.*"

# --- Contributors Section ---
echo "## Contributors" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
# Extract unique authors from the original JSON data and format them.
jq -r '[.[] | .author.login] | unique | map("@\(.)") | join(", ")' "$TEMP_DIR/prs.json" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

echo "✅ Report generated successfully!"
