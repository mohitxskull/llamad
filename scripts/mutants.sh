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


# ── Build in place ───────────────────────────────────────────────────────────
#
# Measured, after three wrong theories about what was eating memory:
#
#   copied tree : 567s baseline, memory pressure enough to trip the watchdog
#   --in-place  : 20 mutants in 63s, build processes peaking at 316 MB RSS
#
# The cost was never cargo-mutants itself. It was the *separate build tree*:
# copying the crate and recompiling llama.cpp from source, with debuginfo, on
# every run. In place, the already-built target/ is reused and the whole
# problem disappears.
#
# The tradeoff is that mutants are written into the real source files. That is
# safe here because the tree must be clean to start and is restored on exit —
# both enforced below — but it is why this is a script and not a bare flag in
# a config file.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: working tree is dirty." >&2
    echo "       --in-place writes mutants into your source files; refusing to" >&2
    echo "       start when uncommitted work could be confused with a mutant." >&2
    exit 1
fi

restore() {
    if ! git diff --quiet; then
        echo >&2
        echo "restoring source files mutated in place..." >&2
        git checkout -- src/ 2>/dev/null || true
    fi
}
trap restore EXIT INT TERM

case " $* " in
    *" --in-place "*) ;;                 # caller already asked
    *) set -- --in-place "$@" ;;
esac

# ── Kill a runaway before it matters ─────────────────────────────────────────
#
# The real defence against the `next_token -> Some(...)` memory bomb described
# below. cargo-mutants derives its timeout from the baseline, which lands
# around 30s — but the bomb allocates ~1.7 GB every 3 seconds, so it exhausts
# a 14 GB machine before that fires.
#
# The unit suite runs in 0.02s. Ten seconds is three orders of magnitude of
# headroom for a legitimate test and stops a runaway at roughly 5 GB, well
# inside the cgroup cap that backstops it.
case " $* " in
    *" --timeout "*|*" --timeout="*) ;;  # caller chose
    *) set -- --timeout 10 "$@" ;;
esac

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
echo "mode        : in-place (reuses target/, restored on exit)"
echo "test command: cargo test --lib   (see .cargo/mutants.toml)"
echo "scratch dir : $TMPDIR (${avail_gb}G free) — off tmpfs"
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
# Consecutive low samples required before acting. One reading is not evidence:
# a single link step can dip `available` briefly, and killing a 15-minute run
# on a blip is its own kind of failure. Three samples over ~9s distinguishes a
# transient from a machine actually heading for the OOM killer.
LOW_STREAK_LIMIT=3
MEM_TRACE="${TMPDIR}/memory-trace.log"

# ── Contain memory-bomb mutants ──────────────────────────────────────────────
#
# This is the actual reason three runs died at mutant 18 of 273, after tmpfs,
# disk and build parallelism had each been blamed and fixed in turn:
#
#     TokenStream::next_token -> Option<String> with Some(String::new())
#
# A `next_token` that always returns `Some` makes every
# `while let Some(t) = stream.next_token()` loop in the suite run forever,
# allocating each iteration. The memory trace is a straight line — ~1.7 GB
# every 3 seconds — and cargo-mutants' 30s test timeout never gets a chance,
# because RAM is gone in twenty.
#
# That mutant is not a defect in llamad; an infinite loop is a perfectly good
# thing for a mutation tester to try. It just has to be survivable. A cgroup
# with a hard cap makes it so: the runaway test binary is killed by the kernel
# inside its own scope, cargo-mutants records the mutant and moves on, and the
# machine never notices.
#
# 8G leaves ample room for a real build and link while killing a runaway in
# seconds. Falls back to running uncapped if systemd is unavailable — with a
# warning, because that is the configuration that freezes machines.
MEMORY_MAX=8G
if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope -q true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
        cargo mutants "$@" &
else
    echo "warning: systemd-run unavailable — running without a memory cap." >&2
    echo "         A mutant that loops forever while allocating can take the" >&2
    echo "         machine down. The watchdog below is the only guard." >&2
    cargo mutants "$@" &
fi
mutants_pid=$!

watchdog() {
    local streak=0 avail
    : > "$MEM_TRACE"
    while kill -0 "$mutants_pid" 2>/dev/null; do
        avail=$(free -m | awk '/^Mem:/{print $7}')
        printf '%s %s\n' "$(date +%H:%M:%S)" "$avail" >> "$MEM_TRACE"
        if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
            streak=$((streak + 1))
            if [ "$streak" -ge "$LOW_STREAK_LIMIT" ]; then
                echo >&2
                echo "watchdog: available memory stayed under ${MIN_AVAIL_MB}MB for ${streak} samples (last ${avail}MB)." >&2
                echo "          Stopping before the kernel does. Trace: $MEM_TRACE" >&2
                pkill -TERM -P "$mutants_pid" 2>/dev/null
                kill -TERM "$mutants_pid" 2>/dev/null
                sleep 5
                kill -KILL "$mutants_pid" 2>/dev/null
                return 1
            fi
        else
            streak=0
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
