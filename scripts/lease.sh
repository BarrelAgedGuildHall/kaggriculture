# shellcheck shell=bash
#
# Pit lease helper for long-running scripts: acquire, export, and RELEASE ON EXIT.
#
# Why this file exists. A battery script that acquires a lease and releases it on the last line
# releases nothing when it dies on line 40 -- a failed command, a Ctrl-C, an ssh drop. The lease
# then sits there until its TTL lapses (up to an hour), every other operative's `arena match` is
# refused, and on the Pit host the ladder stays paused because a live lease re-asserts the pause.
# That has now cost three separate evenings, so the convention is:
#
#     ***  a script that takes a lease installs an EXIT trap that releases it, in the same
#          breath as taking it -- never as a line at the bottom of the script  ***
#
# Usage:
#
#     #!/usr/bin/env bash
#     set -euo pipefail
#     source "$(dirname "$0")/lease.sh"
#     kaggri_lease_acquire engineer-b "expB-303 gate battery" 7200
#
#     uv run python -m arena match cand champ --mode confirm     # sees $KAGGRI_LEASE: no refusal
#     uv run python -m arena gate  cand --team B
#     # ... no release line needed: the trap has it, on every exit path
#
# Notes:
#   * the lease is EXCLUSIVE (pass `--shared` in KAGGRI_LEASE_ARGS only for a hold that measures
#     nothing) and it PAUSES the ladder; the release resumes it (`--resume`) if no lease remains;
#   * `kaggri_lease_renew [ttl]` extends it -- nothing auto-renews, by design;
#   * set KAGGRI_ARENA_CMD if the CLI is not `uv run python -m arena` here;
#   * the Python equivalent is `with arena.leases.held(owner, purpose):`, and `arena match` /
#     `arena gate` take (and release) a lease of their own when you did not declare one.

: "${KAGGRI_ARENA_CMD:=uv run python -m arena}"
: "${KAGGRI_LEASE_ARGS:=}"
: "${KAGGRI_LEASE_RESUME:=1}"

kaggri_lease_acquire() {
    local owner="${1:?owner required, e.g. engineer-b}"
    local purpose="${2:?purpose required, e.g. 'expB-303 gate battery'}"
    local ttl="${3:-3600}"

    if [ -n "${KAGGRI_LEASE:-}" ]; then
        echo "kaggri_lease_acquire: already holding ${KAGGRI_LEASE}; not taking a second" >&2
        return 0
    fi
    # --print-id puts the id on stdout and the report on stderr, so this capture is exact.
    # shellcheck disable=SC2086
    KAGGRI_LEASE="$($KAGGRI_ARENA_CMD lease acquire --owner "$owner" --purpose "$purpose" \
                     --ttl "$ttl" --print-id $KAGGRI_LEASE_ARGS)" || return $?
    [ -n "$KAGGRI_LEASE" ] || { echo "kaggri_lease_acquire: no lease id came back" >&2; return 1; }
    export KAGGRI_LEASE
    # Installed with the acquire, not after the work: nothing between the two can leak the lease.
    trap 'kaggri_lease_release' EXIT
    trap 'kaggri_lease_release; exit 130' INT
    trap 'kaggri_lease_release; exit 143' TERM
    echo "kaggri_lease_acquire: holding $KAGGRI_LEASE (${ttl}s); it will be released on exit" >&2
}

kaggri_lease_renew() {
    [ -n "${KAGGRI_LEASE:-}" ] || { echo "kaggri_lease_renew: no lease held" >&2; return 1; }
    # shellcheck disable=SC2086
    $KAGGRI_ARENA_CMD lease renew "$KAGGRI_LEASE" --ttl "${1:-3600}" >&2
}

kaggri_lease_release() {
    local id="${KAGGRI_LEASE:-}"
    [ -n "$id" ] || return 0
    # Unset FIRST: the EXIT trap fires again after the INT/TERM traps, and a double release would
    # print a confusing "no such lease" over whatever actually went wrong.
    unset KAGGRI_LEASE
    local resume=()
    [ "${KAGGRI_LEASE_RESUME}" = "1" ] && resume=(--resume)
    # shellcheck disable=SC2086
    $KAGGRI_ARENA_CMD lease release "$id" "${resume[@]}" >&2 \
        || echo "kaggri_lease_release: could not release $id -- check \`arena lease list\`" >&2
}
