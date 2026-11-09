#!/usr/bin/env bash
# Keep a terminal progress display and an on-disk transcript for one explicit task.
# Usage: bash scripts/final/run_visible.sh LABEL COMMAND [ARG ...]
set -u

if [[ $# -lt 2 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    printf 'Usage: bash scripts/final/run_visible.sh LABEL COMMAND [ARG ...]\n' >&2
    exit 2
fi
label="$1"
shift
if ! command -v script >/dev/null 2>&1; then
    printf 'The system script utility is unavailable; no command was started.\n' >&2
    exit 127
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
log_dir="$root/results/final-followup/logs"
mkdir -p -- "$log_dir" || exit 1
log="$log_dir/visible-${label}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
printf -v escaped_command '%q ' "$@"
started_seconds=$SECONDS
printf 'Host: %s\nStarted UTC: %s\nCommand: %s\nPersistent log: %s\n' \
    "$(hostname)" "$(date -u +%FT%TZ)" "$escaped_command" "$log" | tee -a "$log"

heartbeat_pid=""
cleanup() {
    if [[ -n "$heartbeat_pid" ]]; then
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
        heartbeat_pid=""
    fi
}
trap cleanup EXIT
(
    heartbeat_sleep=""
    trap 'if [[ -n "$heartbeat_sleep" ]]; then kill "$heartbeat_sleep" 2>/dev/null || true; fi; exit 0' TERM INT
    while true; do
        sleep 20 &
        heartbeat_sleep=$!
        wait "$heartbeat_sleep" || exit 0
        heartbeat_sleep=""
        elapsed=$((SECONDS - started_seconds))
        printf '\n[heartbeat] elapsed %dm %02ds: task still running; this does not mean progress increased.\n' \
            "$((elapsed / 60))" "$((elapsed % 60))" | tee -a "$log"
    done
) &
heartbeat_pid=$!

# Bash is required to interpret printf %q escaping, including unusual arguments.
# script provides a PTY and returns the actual command's exit status (-e).
SHELL="$BASH" script -q -e -f -a -c "$escaped_command" "$log"
result=$?
cleanup
elapsed=$((SECONDS - started_seconds))
if [[ $result -eq 0 ]]; then
    printf '\nCOMPLETED: %s (%dm %02ds). Log: %s\n' \
        "$label" "$((elapsed / 60))" "$((elapsed % 60))" "$log" | tee -a "$log"
else
    printf '\nFAILED: %s, exit code %s (%dm %02ds). Log: %s\n' \
        "$label" "$result" "$((elapsed / 60))" "$((elapsed % 60))" "$log" | tee -a "$log"
fi
exit "$result"
