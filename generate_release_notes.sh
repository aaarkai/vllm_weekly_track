#!/bin/bash

# This script simulates the GitHub Action workflow locally for testing purposes

# Set variables
UPSTREAM_REPO="vllm-project/vllm"
TARGET_FILE="/tmp/weekly-summary.md"

# Create the directory if it doesn't exist
mkdir -p "$(dirname "$TARGET_FILE")"

# Write the header to the new file
echo "# Weekly Release Notes for $UPSTREAM_REPO ($(date +'%Y-%m-%d'))" > "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

# Fetch all merged PRs in the last 7 days
echo "## What's Changed" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

# Create temporary files for each category
FEATURES_FILE=$(mktemp)
BUGS_FILE=$(mktemp)
PERF_FILE=$(mktemp)
DOCS_FILE=$(mktemp)
OTHER_FILE=$(mktemp)

# Check if gh is available
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed. Please install it to run this script."
    echo "You can install it with: brew install gh"
    exit 1
fi

# Fetch all PRs (limiting to 20 for testing)
echo "Fetching PRs from $UPSTREAM_REPO..."
gh pr list --repo $UPSTREAM_REPO --limit 20 \\
  --search "is:pr is:merged" \\
  --json number,title,author,url,labels \\
  --jq '.[] | "\\(.number)|\\(.title)|\\(.author.login)|\\(.url)|\\(.labels | map(.name) | join(\\"\\",\\"))"' > /tmp/all_prs.txt

# Check if we got any PRs
if [ ! -s /tmp/all_prs.txt ]; then
    echo "No PRs found or error fetching PRs."
    exit 1
fi

# Process each PR and categorize
while IFS='|' read -r number title author url labels; do
  # Convert to lowercase for matching
  title_lower=$(echo "$title" | tr '[:upper:]' '[:lower:]')
  labels_lower=$(echo "$labels" | tr '[:upper:]' '[:lower:]')
  
  # Categorize based on labels or title keywords
  if [[ "$labels_lower" == *"feature"* ]] || [[ "$labels_lower" == *"enhancement"* ]] || [[ "$title_lower" == *"feat"* ]] || [[ "$title_lower" == *"add "* ]] || [[ "$title_lower" == *"support"* ]]; then
    echo "* $title ([#$number]($url)) by @$author" >> "$FEATURES_FILE"
  elif [[ "$labels_lower" == *"bug"* ]] || [[ "$title_lower" == *"bug"* ]] || [[ "$title_lower" == *"fix"* ]] || [[ "$title_lower" == *"correct"* ]]; then
    echo "* $title ([#$number]($url)) by @$author" >> "$BUGS_FILE"
  elif [[ "$labels_lower" == *"perf"* ]] || [[ "$labels_lower" == *"performance"* ]] || [[ "$title_lower" == *"perf"* ]] || [[ "$title_lower" == *"optimize"* ]] || [[ "$title_lower" == *"improve"* ]]; then
    echo "* $title ([#$number]($url)) by @$author" >> "$PERF_FILE"
  elif [[ "$labels_lower" == *"doc"* ]] || [[ "$labels_lower" == *"documentation"* ]] || [[ "$title_lower" == *"doc"* ]] || [[ "$title_lower" == *"readme"* ]]; then
    echo "* $title ([#$number]($url)) by @$author" >> "$DOCS_FILE"
  else
    echo "* $title ([#$number]($url)) by @$author" >> "$OTHER_FILE"
  fi
done < /tmp/all_prs.txt

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

# Contributors section
echo "## Contributors" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"
gh pr list --repo $UPSTREAM_REPO --limit 20 \
  --search "is:pr is:merged" \
  --json author \
  --jq 'map(.author.login) | unique | map("@\\(.)") | join(", ")' >> "$TARGET_FILE"

# Clean up temporary files
rm -f "$FEATURES_FILE" "$BUGS_FILE" "$PERF_FILE" "$DOCS_FILE" "$OTHER_FILE" /tmp/all_prs.txt

# Display the generated file
cat "$TARGET_FILE"