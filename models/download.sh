#!/usr/bin/env bash
# Downloads the GGUF models used by the real-model test suite into models/.
#
# Usage:
#   ./download.sh                  # all models
#   ./download.sh LFM2.5-230M      # a subset, by filename substring
#
# Re-running is safe: a file already present at the server's current
# Content-Length is kept; interrupted downloads resume via `curl -C -`.
# Requires curl or wget. ~730 MB total (LFM2.5-1.2B is the big one).
set -euo pipefail

cd "$(dirname "$0")"

declare -A MODELS=(
  ["LFM2.5-230M-Q4_K_M.gguf"]="https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_K_M.gguf"
  ["LFM2.5-1.2B-Thinking-Q4_K_M.gguf"]="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
  ["SmolLM2-135M-Instruct-Q4_K_M.gguf"]="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
  ["qwen2.5-0.5b-instruct-q4_k_m.gguf"]="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

if command -v curl >/dev/null 2>&1; then
  # HEAD to learn the server's current Content-Length (no magic numbers to
  # go stale if upstream re-uploads a different quant under the same name).
  server_size() { curl -sIL --show-error "$1" | awk -F': ' 'tolower($1) == "content-length" { len = $2 } END { gsub(/\r/, "", len); print len }'; }
  dl() { curl -L --fail --silent --show-error --progress-bar -C - -o "$1" "$2"; }
elif command -v wget >/dev/null 2>&1; then
  # wget has no cheap HEAD portability path (--spider is unreliable on HF's
  # 302 chains); fall back to plain download + post-check against the
  # previously fetched size.
  server_size() { :; }
  dl() { wget -q --show-progress -c -O "$1" "$2"; }
else
  echo "error: need curl or wget" >&2
  exit 1
fi

filter="${1:-}"

for name in "${!MODELS[@]}"; do
  if [[ -n "$filter" && "$name" != *"$filter"* ]]; then
    continue
  fi
  url="${MODELS[$name]}"
  expected="$(server_size "$url")"
  if [[ -f "$name" ]] && [[ "$(stat -c %s "$name")" -eq "$expected" ]]; then
    echo "skip: $name already downloaded"
    continue
  fi
  if [[ -n "$expected" && -f "$name" ]] && [[ "$(stat -c %s "$name")" -gt "$expected" ]]; then
    echo "replacing oversized $name (upstream changed?)"
    rm -f "$name"
  fi
  echo "downloading $name"
  dl "$name" "$url"
  actual="$(stat -c %s "$name")"
  if [[ -n "$expected" ]] && [[ "$actual" -ne "$expected" ]]; then
    echo "error: $name is $actual bytes, expected $expected" >&2
    rm -f "$name"
    exit 1
  fi
done

echo "done."
