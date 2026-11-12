#!/usr/bin/env bash
# Downloads the GGUF models used by the real-model test suite into models/.
#
# Usage:
#   ./download.sh                  # all models
#   ./download.sh LFM2.5-230M      # a subset, by filename substring
#
# Re-running is safe: a file already present at the server's current
# Content-Length is kept; interrupted downloads resume via `curl -C -`.
# Requires curl or wget. ~950 MB total (LFM2.5-1.2B is ~730 MB of it).
set -euo pipefail

cd "$(dirname "$0")"

declare -A MODELS=(
  ["LFM3-1.7B-Q4_K_M.gguf"]="https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-Q4_K_M.gguf"
)

declare -A CHECKSUMS=(
  ["LFM3-1.7B-Q4_K_M.gguf"]="3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
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
    if [[ -v CHECKSUMS[$name] ]]; then
      actual_sha="$(sha256sum "$name" | awk '{print $1}')"
      if [[ "$actual_sha" == "${CHECKSUMS[$name]}" ]]; then
        echo "skip: $name already downloaded and verified (sha256: $actual_sha)"
        continue
      else
        echo "replacing corrupted $name (sha256 mismatch: got $actual_sha, expected ${CHECKSUMS[$name]})"
        rm -f "$name"
      fi
    else
      echo "skip: $name already downloaded"
      continue
    fi
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
  if [[ -v CHECKSUMS[$name] ]]; then
    actual_sha="$(sha256sum "$name" | awk '{print $1}')"
    if [[ "$actual_sha" != "${CHECKSUMS[$name]}" ]]; then
      echo "error: sha256 mismatch for $name: got $actual_sha, expected ${CHECKSUMS[$name]}" >&2
      rm -f "$name"
      exit 1
    fi
    echo "verified $name sha256: $actual_sha"
  fi
done

echo "done."
