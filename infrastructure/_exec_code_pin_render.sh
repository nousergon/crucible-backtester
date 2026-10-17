# shellcheck shell=bash
# infrastructure/_exec_code_pin_render.sh — sourced, never executed.
#
# Pin a spot worker's checkouts to the SF execution's code SHAs
# (alpha-engine-config-I11570).
#
# WHY. The worker bootstrap is rendered by `krepis.spot_bootstrap render
# --branch main --extra-clone ...@main`, so every worker clones whatever main
# is at the moment IT launches. A reclaim relaunch (_spot_relaunch.sh) or a
# re-issued stage launches a new worker, so one SF execution could run two
# crucible-backtester SHAs (and two executor / predictor SHAs) — the defect
# rehearsal-2026-09-24-1 hit on the data side. The launcher box already holds
# one SHA per (execution, checkout): nousergon-data's
# infrastructure/exec_code_pin.sh records it at
# $AE_EXEC_PIN_ROOT/<execution name>/<checkout basename> (mirrored to
# s3://alpha-engine-research/health/exec_code_pin/) before the stage runs, and
# the execution name reaches the launcher as $RUN_TOKEN (krepis.ssm_log_capture
# exports its --correlation-id). This helper turns those pins into
# `--pin-sha CHECKOUT=SHA` arguments, which krepis >= 0.59.75 (krepis-PR249)
# renders as "check out exactly this commit and assert HEAD", failing the
# bootstrap loudly when the SHA is unreachable.
#
# CONTRACT
#   exec_code_pin_render_args <checkout-path>...
#     Sets the array EXEC_PIN_RENDER_ARGS. For each checkout with a pin file it
#     holds `--pin-sha <checkout-path>=<sha>`; a checkout with no pin is left
#     unpinned and SAID so on stderr (an ad-hoc laptop run, or a sibling repo
#     no stage of this execution has pinned yet).
#     No RUN_TOKEN, or an explicit non-main --branch, pins nothing: the caller
#     asked for a branch tip, and that is what runs.
#     A pin file that does not hold a 40-hex SHA returns 1 — never a fallback
#     to main, because a silent fallback is how one execution runs two SHAs.
#     A pin that exists while $LIB_PYTHON's krepis predates --pin-sha returns 1
#     naming the floor, rather than letting argparse fail opaquely inside the
#     render's command substitution.
#
# All output goes to stderr.

AE_EXEC_PIN_ROOT="${AE_EXEC_PIN_ROOT:-/home/ec2-user/.ae-exec-pin}"

exec_code_pin_render_args() {
    EXEC_PIN_RENDER_ARGS=()
    if [ "${BRANCH:-main}" != "main" ]; then
        echo "  exec-code-pin: --branch ${BRANCH} is explicit; the worker runs that branch tip." >&2
        return 0
    fi
    if [ -z "${RUN_TOKEN:-}" ]; then
        echo "  exec-code-pin: no RUN_TOKEN (not an SF execution); the worker runs main as cloned." >&2
        return 0
    fi
    local checkout pin_file sha
    for checkout in "$@"; do
        pin_file="${AE_EXEC_PIN_ROOT}/${RUN_TOKEN}/$(basename "$checkout")"
        if [ ! -s "$pin_file" ]; then
            echo "  exec-code-pin: no pin for ${checkout} in execution ${RUN_TOKEN}; the worker clones main." >&2
            continue
        fi
        sha="$(tr -d '[:space:]' < "$pin_file")"
        if ! [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
            echo "ERROR: execution code pin ${pin_file} holds '${sha}', not a 40-hex SHA" >&2
            return 1
        fi
        EXEC_PIN_RENDER_ARGS+=(--pin-sha "${checkout}=${sha}")
        echo "  exec-code-pin: worker ${checkout} pinned at ${sha} (execution ${RUN_TOKEN})." >&2
    done
    local render_help=""
    if [ "${#EXEC_PIN_RENDER_ARGS[@]}" -gt 0 ]; then
        # Captured, not piped into `grep -q`: under pipefail an early grep exit
        # can SIGPIPE the producer and invert the answer.
        render_help="$("${LIB_PYTHON:-python3}" -m krepis.spot_bootstrap render --help 2>/dev/null || true)"
    fi
    if [ "${#EXEC_PIN_RENDER_ARGS[@]}" -gt 0 ] && [[ "$render_help" != *--pin-sha* ]]; then
        echo "ERROR: execution ${RUN_TOKEN} has a code pin, but the krepis in ${LIB_PYTHON:-python3} has no" \
             "'spot_bootstrap render --pin-sha' (needs krepis>=0.59.75, krepis-PR249; the launcher venv is" \
             "crucible-dashboard's requirements.txt)." >&2
        return 1
    fi
    return 0
}
