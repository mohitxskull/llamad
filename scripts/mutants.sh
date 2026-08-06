#!/usr/bin/env bash
# Run cargo-mutants without putting a multi-gigabyte build tree into RAM.
#
# ── Why this script exists ───────────────────────────────────────────────────
#
# cargo-mutants copies the source tree into $TMPDIR and builds there, once for
# the baseline and then incrementally per mutant. On this crate a build tree is
# ~5.6 GB, because llama.cpp is compiled from source with full debuginfo.
#
# On most Linux desktops /tmp is a tmpfs — RAM. So the default invocation
# writes several gigabytes of object files straight into memory, and on an
# 18 GB machine with a browser open it exhausts RAM and takes the box down.
# It froze this machine twice before the cause was found.
#
# Nothing about the failure points at the cause: `nice`, `--jobs`, and
# restricting the test command all leave it unchanged, because it is neither a
# CPU nor a parallelism problem.
#
# The fix is one line — put the scratch directory on disk:
TMPDIR="$(git rev-parse --show-toplevel)/target/mutants-tmp"
export TMPDIR
mkdir -p "$TMPDIR"

# Under target/, so it is already gitignored and `cargo clean` reclaims it.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

avail_gb=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
if [ "${avail_gb:-0}" -lt 15 ]; then
    echo "error: only ${avail_gb}G free on the filesystem holding $TMPDIR." >&2
    echo "       A build tree for this crate is ~6G and mutants keeps more than one." >&2
    exit 1
fi

echo "scratch dir : $TMPDIR ($(df -h --output=fstype "$TMPDIR" | tail -1 | tr -d ' '), ${avail_gb}G free)"
echo "test command: cargo test --lib   (see .cargo/mutants.toml)"
echo

# Arguments are passed through, so a narrowed run still benefits from the
# scratch-dir fix:  ./scripts/mutants.sh --file src/protocol.rs
exec cargo mutants "$@"
