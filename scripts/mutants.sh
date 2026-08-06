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

# ── Build parallelism ────────────────────────────────────────────────────────
#
# cargo-mutants parallelises builds across NCPUS by default. That is the wrong
# default for this crate: every test binary links llama.cpp's static library,
# and concurrent links exhaust a 16-18 GB machine. Measured, not assumed —
# `--jobserver-tasks 3` tripped the watchdog partway through a run here, and
# the same pressure killed a 16 GB GitHub runner mid-job.
#
# Serial by default, therefore: slower and it finishes, which beats fast and
# killed. Override if you have the headroom:
#
#     ./scripts/mutants.sh --jobserver-tasks 4
inject_tasks=1
for arg in "$@"; do
    case "$arg" in
        --jobserver-tasks|--jobserver-tasks=*) inject_tasks="" ;;  # caller chose
    esac
done
if [ -n "$inject_tasks" ]; then
    set -- --jobserver-tasks 1 "$@"
fi
echo "parallelism : $(for a in "$@"; do :; done; echo "$*" | grep -o -- '--jobserver-tasks[= ][0-9]*' || echo default)"
echo

# ── Memory watchdog ──────────────────────────────────────────────────────────
#
# The scratch-dir fix above stops the build tree going into RAM, but rustc and
# the llama.cpp C++ compile are themselves memory-hungry, and cargo-mutants
# parallelises them across NCPUS by default. Rather than guess a safe job count
# for an unknown machine, watch the actual number and stop before the kernel
# has to.
#
# This is a guard, not a tuning knob: if it ever fires, the run stops with a
# clear message instead of the machine freezing.
MIN_AVAIL_MB=2500

cargo mutants "$@" &
mutants_pid=$!

watchdog() {
    while kill -0 "$mutants_pid" 2>/dev/null; do
        avail=$(free -m | awk '/^Mem:/{print $7}')
        if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
            echo >&2
            echo "watchdog: available memory fell to ${avail}MB (floor ${MIN_AVAIL_MB}MB) — stopping." >&2
            echo "          Lower --jobserver-tasks and re-run." >&2
            pkill -TERM -P "$mutants_pid" 2>/dev/null
            kill -TERM "$mutants_pid" 2>/dev/null
            sleep 5
            kill -KILL "$mutants_pid" 2>/dev/null
            return 1
        fi
        sleep 3
    done
}
watchdog "$@" &
watchdog_pid=$!

# Propagate cargo-mutants' exit status; it uses 2 for "some mutants survived",
# which is a finding to read rather than a failure to swallow.
wait "$mutants_pid"
status=$?
kill "$watchdog_pid" 2>/dev/null
wait "$watchdog_pid" 2>/dev/null
exit "$status"
