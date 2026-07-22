#!/usr/bin/env bash
# Assemble the AudioMind API and push it to a Hugging Face Docker Space.
#
# Prereqs (once):
#   1. Create a Space at https://huggingface.co/new-space
#        - SDK: Docker  (blank template)
#        - Visibility: Public  (so the browser can call it without a token)
#   2. Authenticate git for huggingface.co:
#        pip install -U "huggingface_hub[cli]" && hf auth login
#      (paste a token with write access; this sets up the git credential helper)
#
# Usage:
#   deploy/hf-space/deploy-space.sh <hf-username>/<space-name>
# Example:
#   deploy/hf-space/deploy-space.sh Daniel-Chacha/audiomind-api
set -euo pipefail

SPACE="${1:?usage: deploy-space.sh <hf-username>/<space-name>}"
ROOT="$(git rev-parse --show-toplevel)"
HERE="$ROOT/deploy/hf-space"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ Cloning Space https://huggingface.co/spaces/$SPACE"
git clone "https://huggingface.co/spaces/$SPACE" "$TMP/space"

echo "→ Assembling Space contents"
cp "$HERE/Dockerfile" "$TMP/space/Dockerfile"
cp "$HERE/README.md"  "$TMP/space/README.md"
rm -rf "$TMP/space/serving"
cp -r "$ROOT/serving" "$TMP/space/serving"
# strip local-only artifacts that must never ship to the Space
rm -rf "$TMP/space/serving/.venv" "$TMP/space/serving/tests"
find "$TMP/space/serving" -name '__pycache__' -type d -prune -exec rm -rf {} +
cp "$ROOT/best_model.keras" "$TMP/space/best_model.keras"

cd "$TMP/space"
git add -A
if git diff --cached --quiet; then
  echo "→ No changes to push."
else
  git commit -m "Deploy AudioMind API"
  echo "→ Pushing to the Space (it will build automatically)"
  git push
fi

echo "✓ Done. Watch the build at https://huggingface.co/spaces/$SPACE"
echo "  Once live, the API is at:  https://$(echo "$SPACE" | tr 'A-Z/' 'a-z-').hf.space"
echo "  Remember: set the ALLOWED_ORIGINS variable in the Space's Settings"
echo "  to your Vercel URL, and point NEXT_PUBLIC_API_BASE at the Space URL."
