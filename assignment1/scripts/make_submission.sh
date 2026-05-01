#!/usr/bin/env bash
set -euo pipefail

# Create a standardized submission zip.
#
# Produces: code.zip

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

echo "Running public tests..."
uv run pytest

OUT="code.zip"
rm -f "$OUT"

echo "Creating $OUT..."
zip -j "$OUT" student.py >/dev/null

echo "Done: $OUT"
echo "Upload code.zip to Brightspace."
