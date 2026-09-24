#!/usr/bin/env bash
# infrastructure/_spot_common.sh — shared spot-EC2 infrastructure for the
# per-stage backtester launchers (spot_backtester.sh, spot_predictor_backtest.sh,
# spot_portfolio_optimizer_backtest.sh, spot_parity.sh, spot_evaluator.sh).
#
# alpha-engine-config-I4442 (weekly-sf-policy.md §2.1 — atomicity): the
# Saturday SF's five backtest-family states are already independent SF
# states, but until this split they all called the SAME 1600+-line
# `spot_backtest.sh` monolith with different --mode/--skip-stages/
# --pit-parity-enabled flags. Bundling collapsed the blast radius and the
# diagnostic radius into one file: a syntax error or an unbound variable
# in one mode's code path could break every mode, and there was no way to
# preflight a single stage in isolation. This file extracts the parts that
# are genuinely IDENTICAL across all five stages — spot launch, bootstrap,
# dependency install, config staging, SSM transport, cleanup/relaunch,
# error-artifact publishing — so each stage script only has to carry its
# own stage-specific SSM payload and its own preflight.
#
# `spot_backtest.sh` itself is UNCHANGED and stays the currently-wired SF
# path; the SF cutover to these new scripts is a separate nousergon-data
# PR, sequenced after this one. Per policy-shared-code, this stays a
# repo-local sourced file (pure-Bash primitives may stay mirrored) rather
# than moving into nousergon-lib in this PR — nousergon-lib is the right
# home on SECOND adoption (crucible-data / crucible-predictor's spot
# launchers carry near-identical launch/bootstrap/SSM-transport blocks;
# lifting this into the lib is the natural next step once a second repo
# needs it, tracked as a follow-up rather than done speculatively here).
#
# MUST be sourced, never executed directly — it defines functions and sets
# defaults into the caller's global scope; it has no entry point of its own.
#
# shellcheck disable=SC2034
# This file sets many variables (SHOW_HELP, BACKTEST_SKIP_PHASE4_FLAG, etc.)
# that are only READ by the stage script that sources it — shellcheck
# analyzes this file in isolation and cannot see that cross-file usage.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: _spot_common.sh must be sourced, not executed directly." >&2
    echo "       e.g. source \"\$SCRIPT_DIR/_spot_common.sh\"" >&2
    exit 1
fi

# Stage-coverage window (config-I7214): the instant this launcher started.
# An artifact older than this is a leftover from a previous cycle, not this
# run's output — an existence-only probe cannot tell those apart.
_STAGE_WINDOW_START="${_STAGE_WINDOW_START:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# ── Unconditional global defaults ────────────────────────────────────────────
# Every var a downstream function reads is initialized HERE, unconditionally,
# regardless of which flags the caller later parses or which code path runs.
# Motivating defect (2026-07-25 weekly SF recovery, alpha-engine-config-I4442):
# a var initialized only inside one mode's heredoc branch was unbound under
# `set -u` in every OTHER mode — `_BACKTEST_WAS_SKIPPED` in the monolith hit
# exactly this class and was later fixed by moving its init ahead of
# `set -euo pipefail` (see spot_backtest.sh's own `export ... MUST precede
# set -euo pipefail` comment). Every stage-carried variable in this file and
# in each spot_*.sh caller is set here or at the top of the caller — never
# inside an `if`/`case`/heredoc branch on first use.
spot_common_init_defaults() {
    AWS_REGION="${AWS_REGION:-us-east-1}"
    S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
    BRANCH="${BRANCH:-main}"
    # Capacity-resilient instance-type fallback set (2026-05-22 incident:
    # the Evaluator invocation hit InsufficientInstanceCapacity for c5.large
    # in subnet-e07166ec / us-east-1f). All 2 vCPU / 4-8 GB RAM — equivalent
    # for the backtester (memory-bound).
    # ORDER IS LOAD-BEARING, current-generation-first (alpha-engine-config-I11412).
    # krepis.ec2_spot.launch walks types x subnets IN ORDER, and launch_with_fallback
    # buys the FIRST entry on the on-demand rung, so the head of this list is both
    # where spot launches concentrate and what an escalation is billed as. August
    # 2026: 456.8 on-demand BoxUsage:c5.large hours alongside 489.0 spot hours, a 48%
    # escalation rate, because c5.large led. c6a/c7a/c7i were granted by
    # nous-ergon-ops-PR1388 (2 vCPU / 4096 MiB / x86_64, verified against
    # ec2:DescribeInstanceTypes); they add an AMD gen6 and a two-vendor gen7 rung, so
    # the rotation now spans three generations and two silicon vendors instead of
    # exhausting four gen5/gen6 pools and escalating. m5.large stays LAST: it is the
    # only 8 GiB member and the only non-c family here.
    INSTANCE_TYPES="${INSTANCE_TYPES:-c6i.large,c6a.large,c7i.large,c7a.large,c5.large,c5a.large,m5.large}"
    INSTANCE_TYPE=""  # --instance-type X collapses INSTANCE_TYPES to single value
    AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
    # Spot-side watchdog budget. Kept at the monolith's combined-run value
    # (7200s) for every split script in THIS PR — right-sizing each stage's
    # own budget to its own p95 (weekly-sf-policy.md §4) is deferred to the
    # SF-cutover PR, where the SF state's own executionTimeout is set
    # alongside it; until then this stays a conservative shared ceiling.
    MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"
    # KEY_NAME kept ONLY as a launch attribute — nothing in this script SSHs
    # in. Communication is via SSM (2026-05-27 SSH→SSM migration).
    KEY_NAME="alpha-engine-key"
    # alpha-engine-config#3018: SECURITY_GROUP / SUBNETS are EC2 *launch*
    # attributes, not an IAM policy surface — ec2:RunInstances on
    # alpha-engine-executor-role is granted with Resource:"*", so neither
    # value gates access or can "drift" against an IAM policy document.
    # Duplicated verbatim across the data/predictor/backtester spot
    # launchers (same default VPC vpc-566f002e, same SG) — accepted-by-design
    # as plain launch config, intentionally out of IAM-as-code scope.
    SECURITY_GROUP="sg-03cd3c4bd91e610b0"
    SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
    # IAM_PROFILE backs alpha-engine-executor-role, tracked+applied+drift-
    # checked from crucible-executor/infrastructure/iam/alpha-engine-executor-role/.
    IAM_PROFILE="alpha-engine-executor-profile"
    # Lib CLI path. The ops-owned guard /opt/nousergon/bin/lib-python
    # (nous-ergon-ops: alpha-engine-dashboard/live/infrastructure/bin/lib-python)
    # execs the box's DECLARED krepis venv and aborts with EX_CONFIG (78), naming
    # the version it found, rather than silently falling back to a co-tenant
    # checkout — the defect alpha-engine-config-I6931/I7343 removes.
    #
    # It is NOT the default here, because THIS SCRIPT DOES NOT RUN ON THAT BOX
    # (alpha-engine-config-I7386). The guard is installed by
    # nous-ergon-ops/alpha-engine-dashboard/live/infrastructure/bin/install-box-config.sh,
    # whose whole tree provisions the DASHBOARD BOX. This file is sourced by the
    # spot_*.sh scripts the weekly SF delivers as ssm:sendCommand payloads to
    # $.ec2_instance_id — an ephemeral spot, bootstrapped by
    # nousergon-data/infrastructure/lambdas/weekly-freshness-spot-dispatcher/index.py
    # (_bootstrap_command), which builds
    # /home/ec2-user/alpha-engine-dashboard/.venv and never creates
    # /opt/nousergon. Measured on execution
    # friday-shell-2026-08-14-validate-i7382 (nousergon-data's copy of this same
    # line, MorningEnrich): "No such file or directory", exit 127.
    #
    # So the default names the interpreter that host actually has. The
    # ${LIB_PYTHON:-...} override is preserved, so a caller ON a box that does
    # have the guard still names it explicitly and gets the declared floor.
    # Do NOT add a guard block here: the contract lives ONCE, in the repo that
    # owns the box's provisioning (nine copies across five repos is I6922). The
    # SOTA close — install the guard on the spot too, then restore this default —
    # is alpha-engine-config-I7383.
    LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"

    # Common flag-parsed values — every script surfaces this same subset of
    # the monolith's flag surface (branch/instance-type/run-date/dry-run/
    # preflight-only/phase-control/freeze-evaluator). Stage-selecting flags
    # (--mode, --skip-stages, --pit-parity-enabled, --smoke-only) do NOT
    # exist on these scripts — each script IS one stage, hardcoded.
    SHOW_HELP=0
    PREFLIGHT_ONLY=0
    SKIP_PHASE4="${SKIP_PHASE4_EVALUATIONS:-false}"
    SKIP_PHASES="${SKIP_PHASES:-}"
    ONLY_PHASES="${ONLY_PHASES:-}"
    FORCE_ALL="${FORCE_ALL:-false}"
    FORCE_PHASES="${FORCE_PHASES:-}"
    DRY_RUN="${DRY_RUN:-false}"
    FREEZE_EVALUATOR="${FREEZE_EVALUATOR:-false}"
    USE_VECTORIZED_SWEEP="${USE_VECTORIZED_SWEEP:-false}"
    EVAL_HALF="${EVAL_HALF:-all}"
    RUN_DATE="${RUN_DATE:-$(date -u +%Y-%m-%d)}"
    # alpha-engine-config-I8155: initialized to empty (never to $RUN_DATE —
    # that would silently reintroduce the trading-day-normalized value this
    # carrier exists to avoid) purely so `set -u` doesn't abort the whole
    # launch when the SF hasn't set it. The SF definition
    # (nousergon-data/infrastructure/step_function.json) exports this
    # directly into the SSM command environment; an empty value here means
    # it genuinely wasn't set, and each `krepis.stage_coverage assert
    # --run-date "$EXECUTION_RUN_DATE"` call site is what raises the loud
    # signal (EXIT_NO_RUN_DATE) via its existing observe-mode `|| echo
    # WARNING` guard — never fall back to RUN_DATE here.
    EXECUTION_RUN_DATE="${EXECUTION_RUN_DATE:-}"

    # #883 bounded mid-run spot-reclaim relaunch — env-only, no CLI flag on
    # the monolith either. MAX_SPOT_ATTEMPTS=4 preserves the prior
    # RECLAIM_RELAUNCH_MAX=3 budget (3 relaunches = 4 total attempts).
    MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-4}"
    SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
    SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"

    # Derived flag strings, built by spot_common_compute_phase_flags() after
    # parsing — initialized empty here so `set -u` never trips if a caller
    # (e.g. spot_parity.sh / spot_evaluator.sh) never calls that builder.
    BACKTEST_SKIP_PHASE4_FLAG=""
    BACKTEST_PHASE_FLAGS=""

    # Populated later in the launch sequence; initialized here so any early
    # exit path (trap cleanup) can reference them without tripping `set -u`.
    INSTANCE_ID=""
    RUN_ID=""
    S3_STAGING_PREFIX=""
    S3_STAGING=""
    LAST_SSM_DESC=""
    EXECUTOR_CONFIG=""
    PREDICTOR_CONFIG=""
    STAGED_PREDICTOR_CONFIG=0
    ENV_SOURCE=""
    REMOTE_PYTHON=""
}

# ── Flag parsing ──────────────────────────────────────────────────────────────
# Common subset of the monolith's flag surface. Unknown flags hard-fail
# (no-silent-fails) — same as the monolith. Callers pass "$@" through.
spot_common_parse_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h) SHOW_HELP=1; shift ;;
            --preflight-only) PREFLIGHT_ONLY=1; shift ;;
            --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
            --instance-type=*) INSTANCE_TYPE="${1#*=}"; shift ;;
            --branch) BRANCH="$2"; shift 2 ;;
            --branch=*) BRANCH="${1#*=}"; shift ;;
            --run-date) RUN_DATE="$2"; shift 2 ;;
            --run-date=*) RUN_DATE="${1#*=}"; shift ;;
            --skip-phase4-evaluations) SKIP_PHASE4="true"; shift ;;
            --skip-phases) SKIP_PHASES="$2"; shift 2 ;;
            --skip-phases=*) SKIP_PHASES="${1#*=}"; shift ;;
            --only-phases) ONLY_PHASES="$2"; shift 2 ;;
            --only-phases=*) ONLY_PHASES="${1#*=}"; shift ;;
            --force) FORCE_ALL="true"; shift ;;
            --force-phases) FORCE_PHASES="$2"; shift 2 ;;
            --force-phases=*) FORCE_PHASES="${1#*=}"; shift ;;
            --dry-run) DRY_RUN="true"; shift ;;
            --freeze-evaluator) FREEZE_EVALUATOR="true"; shift ;;
            --use-vectorized-sweep) USE_VECTORIZED_SWEEP="true"; shift ;;
            --eval-half) EVAL_HALF="$2"; shift 2 ;;
            --eval-half=*) EVAL_HALF="${1#*=}"; shift ;;
            *)
                echo "Unknown flag: $1" >&2
                echo "(stage-selecting flags --mode / --skip-stages / --pit-parity-enabled / --smoke-only do not exist on this script — it IS one stage. Use spot_backtest.sh directly for those.)" >&2
                exit 1
                ;;
        esac
    done

    case "$EVAL_HALF" in
        all|diagnostics|optimize) ;;
        *)
            echo "ERROR: unknown --eval-half='$EVAL_HALF'" >&2
            echo "       Valid values: all diagnostics optimize" >&2
            exit 1
            ;;
    esac
}

# Builds BACKTEST_SKIP_PHASE4_FLAG / BACKTEST_PHASE_FLAGS from the parsed
# phase-control vars. Called by the three backtest-family stage scripts
# (spot_backtester.sh, spot_predictor_backtest.sh,
# spot_portfolio_optimizer_backtest.sh) only — parity/evaluator don't pass
# these through to backtest.py.
spot_common_compute_phase_flags() {
    if [ "$SKIP_PHASE4" = "true" ]; then
        BACKTEST_SKIP_PHASE4_FLAG="--skip-phase4-evaluations"
    else
        BACKTEST_SKIP_PHASE4_FLAG=""
    fi
    BACKTEST_PHASE_FLAGS=""
    if [ -n "$SKIP_PHASES" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --skip-phases=$SKIP_PHASES"
    fi
    if [ -n "$ONLY_PHASES" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --only-phases=$ONLY_PHASES"
    fi
    if [ "$FORCE_ALL" = "true" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --force"
    fi
    if [ -n "$FORCE_PHASES" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --force-phases=$FORCE_PHASES"
    fi
    if [ "$DRY_RUN" = "true" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --dry-run"
    fi
    if [ "$USE_VECTORIZED_SWEEP" = "true" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --use-vectorized-sweep"
    fi
}

# Stages that materialise the ~900-ticker universe need >=16 GB RAM (I3280:
# 8 GB instances can dip to ~6 GB available under OS overhead, leaving zero
# margin above the 6.0 GB headroom guard). Skipped when the operator passes an
# explicit --instance-type.
#
# RENAMED 2026-08-13 from `spot_common_apply_predictor_ram_floor`. The old name
# named the wrong thing, and the misnomer had already cost two production OOMs:
#
#   * spot_backtester.sh was left off the floor on the reasoning "param-sweep
#     does not run predictor_pipeline -> stays on the cheap default rotation".
#     OOM-killed 2026-08-13 (config-I7216, PR653/PR657).
#   * spot_evaluator.sh carried the SAME sentence and was OOM-killed the SAME
#     DAY on a 4 GB c5.large, immediately after
#     `Load complete: 921 price tickers, 903 feature tickers`
#     (execution watch-rerun-2026-08-13-2, instance i-077d2a5479affe1d3).
#
# The driver is the ArcticDB universe read, not the GBM tensor. Not loading the
# tensor does not make a stage cheap, and every launcher that reads the full
# universe needs this whether or not `predictor_pipeline` appears anywhere in
# it. The name now says the condition an author must actually check.
#
# ── THIS FLOOR IS A GUESS, AND HERE IS WHAT WILL REPLACE IT ──────────────────
#
# `floor_types` below is NOT derived from any measurement. It is the value at
# which the OOM kills stopped, reached by raising the previous cap — and a cap
# derived from another cap is not a budget. That was unavoidable: an OOM-killed
# process reports no peak RSS, so only a SURVIVING run can state the real
# requirement, and until 2026-08-13 no stage recorded it on success either.
#
# It is now recorded. `krepis.rss_budget`, applied at the
# `krepis.ssm_dispatcher` chokepoint every stage in this repo runs its remote
# work through (krepis-PR149, alpha-engine-config-I7260), measures the peak
# resident set of each stage's whole process subtree on every SUCCESSFUL run and
# publishes headroom to `ops/checks/ae-rss-<stage>/latest.json`, which renders
# on the console. Read one stage's row with:
#
#   aws s3 cp s3://alpha-engine-research/ops/checks/ae-rss-evaluator/latest.json -
#
# WHAT REPLACES THIS FLOOR, AND WHEN. Once each stage on the floor has at least
# 4 readings in its row's `history` (≈4 weekly runs, i.e. from ~2026-09-10),
# re-derive `floor_types` from the measured peak: pick the smallest instance
# class whose RAM leaves the declared warn margin above the MAXIMUM observed
# peak, not the median, and record the derivation and the readings it used in
# the commit that changes this line. Right-sizing DOWN is real money — this
# rotation is 16 GB where several stages may need far less — and it is the
# right-sizing that is currently unjustifiable in either direction.
#
# Until then this comment, not the value, is the honest statement of what the
# floor is: a bound that has stopped the kills, with the measurement that will
# falsify it now running. Do NOT lower it on intuition; do not raise it either
# without saying which reading forced the raise. (alpha-engine-config-I7260,
# which also tracks re-deriving the two headroom thresholds themselves.)
spot_common_apply_large_universe_ram_floor() {
    local floor_types="m5.xlarge,m6i.xlarge,m5a.xlarge,c5.2xlarge,c6i.2xlarge"
    if [ -z "$INSTANCE_TYPE" ]; then
        echo "  Stage reads the full ~900-ticker universe -> applying >=16 GB instance floor"
        INSTANCE_TYPES="$floor_types"
    fi
    if [ -n "$INSTANCE_TYPE" ]; then
        INSTANCE_TYPES="$INSTANCE_TYPE"
    fi
}

spot_common_collapse_instance_type() {
    if [ -n "$INSTANCE_TYPE" ]; then
        INSTANCE_TYPES="$INSTANCE_TYPE"
    fi
}

# ── DATE_CONVENTIONS: normalize RUN_DATE to the NYSE trading day ────────────
# Single dispatcher-side chokepoint, BEFORE RUN_DATE is threaded into the
# stage's --date and any bash s3 path. Defensive: keep the calendar value if
# the lib call fails — a normalization miss must not abort the launch; the
# python entry points re-normalize idempotently as a backstop.
spot_common_normalize_run_date() {
    local _run_date_td
    _run_date_td="$("$LIB_PYTHON" -c "import datetime as d; from nousergon_lib import trading_calendar as tc; x=d.date.fromisoformat('${RUN_DATE}'[:10]); print(x.isoformat() if tc.is_trading_day(x) else tc.previous_trading_day(x).isoformat())" 2>/dev/null || true)"
    if [ -n "$_run_date_td" ]; then
        if [ "$_run_date_td" != "$RUN_DATE" ]; then
            echo "==> Normalized RUN_DATE ${RUN_DATE} (calendar) -> ${_run_date_td} (trading day) per DATE_CONVENTIONS"
        fi
        RUN_DATE="$_run_date_td"
    else
        echo "WARNING: trading-day normalization of RUN_DATE=${RUN_DATE} failed — keeping calendar value (python entry points will re-normalize)" >&2
    fi
    # alpha-engine-config-I8155: RUN_DATE is deliberately the (normalized)
    # NYSE TRADING day — it drives artifact keys (backtest/{trading_day}/...,
    # parity/$RUN_DATE/...) and must stay exactly as normalized above.
    # EXECUTION_RUN_DATE is a SEPARATE carrier: the SF execution's own
    # run_date, exported by the Step Functions definition
    # (nousergon-data/infrastructure/step_function.json) and NEVER
    # normalized by anything downstream. It is what every
    # `krepis.stage_coverage assert --run-date` call groups verdicts on, so
    # it must reach the CLI exactly as the SF set it. Do NOT default
    # EXECUTION_RUN_DATE from RUN_DATE here or anywhere else in this file —
    # an unset EXECUTION_RUN_DATE must reach krepis.stage_coverage empty so
    # it exits loudly (EXIT_NO_RUN_DATE) under the existing `|| echo
    # WARNING` observe-mode guard at each call site, rather than silently
    # grouping under the wrong day.
}

# ── Dispatcher-side pre-launch preflight (L4485) ─────────────────────────────
# Fail fast on the DISPATCHER, before provisioning a spot. Two cheap
# (<2s, no deps) checks that catch the two cheapest-to-miss classes at
# second zero: a SyntaxError in a load-bearing entrypoint, and a
# requirements.txt lib pin below preflight.py's own floor. Callers pass the
# list of entrypoints THEIR stage actually imports.
spot_common_pre_launch_preflight() {
    local py
    py="$LIB_PYTHON"
    [ -x "$py" ] || py="$(command -v python3 || echo python3)"

    if ! "$py" -c 'import ast,sys; [ast.parse(open(f).read(), filename=f) for f in sys.argv[1:]]' "$@" 2>/tmp/prelaunch_syntax.err; then
        echo "ERROR: pre-launch syntax check FAILED — a SyntaxError would crash the spot ~15 min into boot+deps. Fix before launching:" >&2
        cat /tmp/prelaunch_syntax.err >&2
        exit 1
    fi

    local pin floor lowest
    pin=$(grep -oE '@v[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/requirements.txt" | head -1 | tr -d '@v')
    floor=$(grep -oE 'MIN_LIB_VERSION[[:space:]]*=[[:space:]]*"[0-9.]+"' "$REPO_ROOT/preflight.py" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -n "$pin" ] && [ -n "$floor" ]; then
        lowest=$(printf '%s\n%s\n' "$floor" "$pin" | sort -V | head -1)
        if [ "$lowest" != "$floor" ]; then
            echo "ERROR: requirements.txt alpha-engine-lib pin v$pin < preflight.py MIN_LIB_VERSION $floor." >&2
            echo "       The spot's pip install would pull a version the code rejects. Bump the pin or the floor." >&2
            exit 1
        fi
        echo "  pre-launch: lib pin v$pin >= MIN_LIB_VERSION $floor OK"
    else
        echo "  pre-launch: WARNING — could not parse lib pin (pin='$pin' floor='$floor'); skipping pin cross-check" >&2
    fi

    local dirty
    dirty=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | grep -E '\.(py|sh)$' || true)
    if [ -n "$dirty" ]; then
        echo "  pre-launch: WARNING — uncommitted tracked .py/.sh changes; the spot clones --branch $BRANCH and will NOT see these:" >&2
        echo "$dirty" | sed 's/^/      /' >&2
    fi
    git -C "$REPO_ROOT" fetch --quiet origin "$BRANCH" 2>/dev/null || true
    local lhead rhead
    lhead=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)
    rhead=$(git -C "$REPO_ROOT" rev-parse "origin/$BRANCH" 2>/dev/null || true)
    if [ -n "$lhead" ] && [ -n "$rhead" ] && ! git -C "$REPO_ROOT" merge-base --is-ancestor "$lhead" "$rhead" 2>/dev/null; then
        echo "  pre-launch: WARNING — local HEAD ($lhead) is not in origin/$BRANCH; the spot clones origin/$BRANCH and will run WITHOUT your local commits. Push first." >&2
    fi

    local cfg_real cfg_git_root cfg_rel cfg_dirty
    if [ -L "$REPO_ROOT/config.yaml" ]; then
        cfg_real=$(readlink -f "$REPO_ROOT/config.yaml" 2>/dev/null || true)
        if [ -n "$cfg_real" ]; then
            cfg_git_root=$(git -C "$(dirname "$cfg_real")" rev-parse --show-toplevel 2>/dev/null || true)
            if [ -z "$cfg_git_root" ]; then
                echo "  pre-launch: WARNING — config.yaml symlinks to $cfg_real, which is NOT inside a git repo; operator flags there have no audit trail and will vanish on rebuild." >&2
            else
                cfg_rel="${cfg_real#"$cfg_git_root"/}"
                if ! git -C "$cfg_git_root" ls-files --error-unmatch "$cfg_rel" >/dev/null 2>&1; then
                    echo "  pre-launch: WARNING — config.yaml symlinks to $cfg_real, which is NOT git-tracked in $cfg_git_root; operator flags there have no audit trail and will vanish on rebuild." >&2
                else
                    cfg_dirty=$(git -C "$cfg_git_root" status --porcelain -- "$cfg_rel" 2>/dev/null || true)
                    if [ -n "$cfg_dirty" ]; then
                        echo "  pre-launch: WARNING — config.yaml ($cfg_real) has uncommitted changes not captured in git ($cfg_git_root); these operator flags will NOT survive a rebuild of this symlink. Commit + PR the change:" >&2
                        echo "$cfg_dirty" | sed 's/^/      /' >&2
                    fi
                fi
            fi
        fi
    fi

    echo "  pre-launch preflight OK."
}

# ── Config resolution ────────────────────────────────────────────────────────
# Executor risk.yaml: needed by every stage (the executor sim/backtest path
# loads it at process startup regardless of --mode). Fails loud if missing —
# silently falling back to risk.yaml.example produces placeholder bucket
# names and an ArcticDB KeyNotFoundException deep in the run.
spot_common_resolve_executor_config() {
    local experiment_id="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
    EXECUTOR_CONFIG=""

    # alpha-engine-config-I7247: refresh the alpha-engine-config checkout
    # itself before resolving risk.yaml out of it. Unlike predictor.yaml (a
    # DIFFERENT repo, staged in by an explicit `cp` per
    # spot_common_resolve_predictor_config / config-I7216 / PR657), risk.yaml
    # is read straight out of the alpha-engine-config checkout on the
    # dispatch box — standing infrastructure that only freshens as a SIDE
    # EFFECT of two independently-skippable early SF states
    # (MorningEnrich/skip_morning_enrich, DataPhase1/skip_data_phase1) plus,
    # incidentally, PredictorTraining/ModelZooSelect. A mechanical rerun that
    # sets all three skip flags (recovery from a failure downstream of them,
    # e.g. a Backtester-family failure) never refreshes the checkout for that
    # execution — this function still finds a file (the checkout is standing
    # infrastructure, not created per-run), so nothing fails loud and the
    # stage runs against whatever risk.yaml content was on disk from a
    # previous week's `git pull`. This is a STALENESS exposure, not the
    # missing-file class PR657 fixed for predictor.yaml — no live failure was
    # observed, only a silent-degrade window.
    #
    # Fixed at the layer that DECLARES the requirement (sf-pipeline-policy.md
    # §2.2 — every stage proves its own preconditions): pull each checkout
    # candidate to origin/main before searching it, rather than relying on
    # three unrelated skip flags to keep it fresh. `git pull --ff-only` is
    # cheap and idempotent — safe to run unconditionally, every invocation,
    # not gated behind any skip flag. A failed refresh (offline runner,
    # detached HEAD, local edits, network blip) is reported but never blocks
    # resolution: the existing checkout, however stale, is still the best
    # available input, and the checkout going missing entirely is still
    # caught by the loud failure below.
    local config_root
    for config_root in "$HOME/alpha-engine-config" "$HOME/Development/alpha-engine-config"; do
        if [ -d "$config_root/.git" ]; then
            if ! git -C "$config_root" pull --ff-only --quiet 2>/dev/null; then
                echo "  WARNING: git -C $config_root pull --ff-only failed — risk.yaml may be stale (config-I7247)" >&2
            fi
        fi
    done

    local candidate
    for candidate in \
        "$HOME/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" \
        "$HOME/Development/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" \
        "$HOME/alpha-engine-config/executor/risk.yaml" \
        "$HOME/Development/alpha-engine-config/executor/risk.yaml" \
        "$HOME/alpha-engine/config/risk.yaml" \
        "$HOME/Development/alpha-engine/config/risk.yaml"; do
        if [ -f "$candidate" ]; then
            EXECUTOR_CONFIG="$candidate"
            break
        fi
    done
    if [ -z "$EXECUTOR_CONFIG" ]; then
        echo "ERROR: executor risk.yaml not found in any search path:" >&2
        echo "  ~/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" >&2
        echo "  ~/Development/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" >&2
        echo "  ~/alpha-engine-config/executor/risk.yaml" >&2
        echo "  ~/Development/alpha-engine-config/executor/risk.yaml" >&2
        echo "  ~/alpha-engine/config/risk.yaml (legacy)" >&2
        echo "  ~/Development/alpha-engine/config/risk.yaml (legacy)" >&2
        exit 1
    fi
}

# Predictor predictor.yaml. `required=1` (spot_predictor_backtest.sh /
# spot_portfolio_optimizer_backtest.sh — these stages run predictor_pipeline
# and cannot produce a real result without it) hard-fails when missing,
# per-stage preflight assertion rather than the monolith's uniform soft-skip
# (the monolith couldn't know per-invocation whether the mode needed it; each
# split script does). `required=0` (spot_backtester.sh / spot_parity.sh /
# spot_evaluator.sh) keeps the monolith's soft-skip-with-sentinel behavior.
spot_common_resolve_predictor_config() {
    local required="${1:-0}"
    PREDICTOR_CONFIG=""
    local candidate
    for candidate in \
        "$HOME/alpha-engine-predictor/config/predictor.yaml" \
        "$HOME/Development/alpha-engine-predictor/config/predictor.yaml"; do
        if [ -f "$candidate" ]; then
            PREDICTOR_CONFIG="$candidate"
            break
        fi
    done
    # alpha-engine-config-I7216: stage it from the private config repo before
    # declaring it missing. predictor.yaml is gitignored (the .example pattern),
    # so a fresh dispatcher only has it because SOMETHING copied it there — and
    # until now the only thing that did was ResearchPredictorParallel, three
    # `cp` invocations buried in the weekly SF's PredictorTraining branch.
    #
    # That made this stage's prerequisite a SIDE EFFECT of a different stage.
    # Measured 2026-08-13: a mechanical rerun (weekly_sf_rerun.py) correctly
    # derived skip_predictor_training, because that stage had already completed
    # — and PredictorBacktest then died here on a config the skipped stage
    # would have staged. The recovery path is exactly the path that skips
    # completed stages, so the dependency is invisible precisely when it bites.
    # PredictorBacktest writes the live entry feed
    # (predictor/research_free_backfill/), so this blocked the cohort refresh.
    #
    # Staging here rather than adding a fourth `cp` to the SF keeps the fix at
    # the layer that DECLARES the requirement: the stage that needs the file
    # obtains it, instead of every caller remembering to.
    if [ -z "$PREDICTOR_CONFIG" ]; then
        local _ref="$HOME/alpha-engine-config/experiments/reference/predictor/predictor.yaml"
        local _dest="$HOME/alpha-engine-predictor/config/predictor.yaml"
        if [ -f "$_ref" ] && [ -d "$(dirname "$_dest")" ]; then
            echo "  predictor.yaml absent — staging from alpha-engine-config reference (config-I7216)"
            # `rm -f` then plain `cp`, NOT `cp --remove-destination`: the SF's
            # own three copies use the GNU flag, which exists only to replace a
            # SYMLINKED destination and is unsupported by BSD cp. This form is
            # equivalent on Linux and also runs on a developer's machine, which
            # is what let this staging path be exercised before it shipped.
            if rm -f "$_dest" && cp "$_ref" "$_dest"; then
                PREDICTOR_CONFIG="$_dest"
            else
                echo "  WARNING: staging predictor.yaml from $_ref failed" >&2
            fi
        fi
    fi

    if [ -z "$PREDICTOR_CONFIG" ]; then
        if [ "$required" = "1" ]; then
            echo "ERROR: predictor.yaml not found in any search path — this stage runs predictor_pipeline and cannot produce a real result without it:" >&2
            echo "  ~/alpha-engine-predictor/config/predictor.yaml" >&2
            echo "  ~/Development/alpha-engine-predictor/config/predictor.yaml" >&2
            echo "  and staging from ~/alpha-engine-config/experiments/reference/predictor/predictor.yaml did not succeed" >&2
            exit 1
        fi
        echo "  WARNING: predictor.yaml not found — predictor backtest will be skipped"
        STAGED_PREDICTOR_CONFIG=0
    else
        STAGED_PREDICTOR_CONFIG=1
    fi
}

# ── Declared instance-type allow-list (alpha-engine-config-I11227) ───────────
# Mirrors the single source,
# nous-ergon-ops/infrastructure/iam/spot-launch-declared-instance-types.json,
# from which the executor and dashboard roles' ec2:RunInstances
# `ec2:InstanceType` condition values are asserted. These launchers live in
# PUBLIC repos and must not read a private file at runtime, so the list is
# duplicated here and held in lockstep by that repo's
# tests/test_spot_launch_instance_type_allowlist.py, which reads THIS constant
# out of the public repo and fails on divergence in either direction.
# Adding a type is a two-PR change: the declared file first, this constant
# second. Without this check the operator sees an opaque UnauthorizedOperation
# from RunInstances and nothing naming the list that refused it.
ALLOWED_INSTANCE_TYPES="c5.2xlarge,c5.large,c5.xlarge,c5a.large,c6a.large,c6i.2xlarge,c6i.large,c6i.xlarge,c7a.large,c7i.large,m5.large,m5.xlarge,m5a.large,m5a.xlarge,m6i.large,m6i.xlarge,r5.large,r5a.large,r6i.large"

# Refuse, before any AWS call, a type IAM will refuse. Placed at the single
# krepis.ec2_spot chokepoint rather than at argument parsing so that every
# later override — a RAM floor, `--instance-type`, a per-stage INSTANCE_TYPES
# assignment — is covered by construction rather than by remembering to add a
# second check next to it.
spot_assert_instance_types_allowed() {
    local _bad="" _t
    for _t in $(echo "${1:-}" | tr ',' ' '); do
        case ",${ALLOWED_INSTANCE_TYPES}," in
            *",${_t},"*) ;;
            *) _bad="${_bad} ${_t}" ;;
        esac
    done
    if [ -z "$_bad" ]; then return 0; fi
    echo "ERROR: instance type(s) not on the spot-launch allow-list:${_bad}" >&2
    echo "       allow-list: ${ALLOWED_INSTANCE_TYPES}" >&2
    echo "       declared in: nous-ergon-ops/infrastructure/iam/spot-launch-declared-instance-types.json" >&2
    echo "       mirrored in: this script's ALLOWED_INSTANCE_TYPES constant" >&2
    echo "       ec2:RunInstances would refuse this launch with UnauthorizedOperation" >&2
    echo "       (alpha-engine-config-I11227). Add the type to the declared file and to" >&2
    echo "       every mirrored constant, or pick one from the list above." >&2
    return 1
}

# ── Launch spot instance ──────────────────────────────────────────────────────
# Capacity-resilient launch via krepis.ec2_spot (rotates instance_type x
# subnet on InsufficientInstanceCapacity etc). Direct fix for the 2026-05-22
# incident (InsufficientInstanceCapacity for c5.large in us-east-1f).
spot_common_launch_instance() {
    spot_assert_instance_types_allowed "$INSTANCE_TYPES" || exit 2

    echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."
    local ec2_spot_rc=0
    INSTANCE_ID=$("$LIB_PYTHON" -m krepis.ec2_spot launch \
        --types "$INSTANCE_TYPES" \
        --subnets "$SUBNETS" \
        --image-id "$AMI_ID" \
        --key-name "$KEY_NAME" \
        --security-group "$SECURITY_GROUP" \
        --iam-profile "$IAM_PROFILE" \
        --name "alpha-engine-${SPOT_STAGE_NAME}-$(date +%Y%m%d)" \
        --region "$AWS_REGION") || ec2_spot_rc=$?
    if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$INSTANCE_ID" ]; then
        if [ "$ec2_spot_rc" -eq 64 ]; then
            echo "ERROR: capacity exhausted across all instance_type x subnet combinations" >&2
        fi
        if [ "$ec2_spot_rc" -eq 0 ]; then
            # rc=0 with an EMPTY instance id = the launch layer produced
            # nothing. `${ec2_spot_rc:-1}` defaults only when UNSET — a
            # captured 0 passed through would be a silent success. An empty
            # id must always fail loud (config#1646).
            echo "ERROR: ec2_spot launch exited 0 without an instance id — failing loud (config#1646)" >&2
            ec2_spot_rc=1
        fi
        exit "$ec2_spot_rc"
    fi
    echo "  Instance ID: $INSTANCE_ID"

    # (config-I7442) The launch-time prune retired into
    # `krepis.spot_evidence teardown`, which sweeps both tmp/spot_<slug>/ and
    # _spot_evidence/<slug>/ on every teardown — including for stages that
    # have stopped running, which the per-stage prune here could never reach.
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
    S3_STAGING_PREFIX="tmp/spot_${SPOT_STAGE_NAME}/${RUN_ID}"
    S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"
}

# ── Staging teardown (config-I7442, superseding config-I7396) ──────────────
# The staging prefix holds the ONLY full copy of a stage's stdout and stderr:
# `run_ssm` points SSM's OutputS3KeyPrefix at ${S3_STAGING_PREFIX}/ssm-output,
# and GetCommandInvocation returns just the FIRST 24 KB, so on any long stage
# the tail -- where the failure is -- exists nowhere else.
#
# config-I7396 (#675) stopped this path deleting that prefix on failure, and
# bounded the resulting retention with a per-stage bash prune at launch. That
# fixed THIS repo. It was still the wrong shape for the class:
#
#   * three `_spot_common.sh` twins and four surviving monoliths ran the same
#     unguarded `aws s3 rm "$S3_STAGING" --recursive` inside their own failure
#     handlers, and mirroring a 55-line retain-plus-prune into each is exactly
#     what policy-shared-code forbids at the second adoption;
#   * the prune only ever reached the prefix of a stage that RAN AGAIN, so a
#     retired stage's staging leaked forever -- named in #675's own comment as
#     the residual;
#   * it needed a `date -u -v-Nd` / `date -u -d` fallback pair to compute a
#     cutoff, and the evidence stayed inside `tmp/`, the prefix everything
#     here recursively deletes.
#
# `krepis.spot_evidence teardown` is the chokepoint that replaces both. It
# copies the prefix to `_spot_evidence/<slug>/<date>/<run-id>/` -- outside
# `tmp/` -- and deletes staging ONLY if that copy succeeded; on a successful
# run it deletes with nothing retained, so no green run accrues storage. It
# also sweeps both trees on every teardown, which is where the prune went.
# The ordering lives in that module's call graph, so it cannot be undone here
# by adding a line.

# ── Cleanup / reclaim-relaunch / error-artifact publishing ──────────────────
# Always terminate the instance + clean S3 staging, with diagnostics on
# failure, and re-exec on a CONFIRMED spot reclaim (#883). Installs a
# `trap cleanup EXIT` — call once, after INSTANCE_ID is set.
spot_common_install_cleanup_trap() {
    spot_common_teardown_staging() {
        local _exit_code="$1"
        if [ -z "${S3_STAGING:-}" ]; then
            echo "  Instance terminated; no S3 staging was provisioned."
            return 0
        fi
        if "$LIB_PYTHON" -m krepis.spot_evidence teardown \
                --staging "$S3_STAGING" \
                --slug "$SPOT_STAGE_NAME" \
                --exit-code "$_exit_code"; then
            return 0
        fi
        # The chokepoint is unreachable on this box -- a krepis older than the
        # release carrying `spot_evidence`, or a broken interpreter. RETAIN rather
        # than fall back to `aws s3 rm`: an unswept prefix is a cost bounded by the
        # next teardown's sweep, where losing the diagnosis is config-I7442 again.
        # This is also what makes the merge order safe -- this change is correct on
        # a box whose krepis pin has not yet been bumped.
        echo "  spot_evidence: chokepoint unavailable via $LIB_PYTHON — S3 staging RETAINED at $S3_STAGING/ (not deleted)" >&2
        echo "    Full stage stdout/stderr: $S3_STAGING/ssm-output/" >&2
        return 0
    }

    # shellcheck disable=SC2329
    # invoked via `trap cleanup EXIT` below — shellcheck doesn't associate
    # a same-function trap registration as a call site.
    cleanup() {
        local exit_code=$?
        local _will_relaunch=0 _alert_sev="error"
        echo ""
        echo "==> Dispatcher EXIT (code=$exit_code)"
        local state="<not yet provisioned>" reason_code="<none>" state_reason="<none>"
        if [ "$exit_code" -ne 0 ]; then
            local last_desc="${LAST_SSM_DESC:-<none — failed before any SSM call>}"
            echo "    last run_ssm: $last_desc"
            if [ -n "${INSTANCE_ID:-}" ]; then
                local _desc
                _desc=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --query 'Reservations[0].Instances[0].[State.Name,StateReason.Code,StateTransitionReason]' --output text 2>/dev/null || true)
                state=$(printf '%s' "$_desc" | cut -f1)
                reason_code=$(printf '%s' "$_desc" | cut -f2)
                state_reason=$(printf '%s' "$_desc" | cut -f3-)
                [ -z "$state" ] && state="<lookup-failed>"
                [ -z "$reason_code" ] && reason_code="<none>"
                [ -z "$state_reason" ] && state_reason="<none>"
                echo "    spot state: $state"
                echo "    spot state-reason-code: $reason_code"
                echo "    spot state-transition-reason: $state_reason"
            fi
            # See alpha-engine-config-I7009 — migrated off the exit-code contract to --json.
            if [ -n "${INSTANCE_ID:-}" ] && [ "$SPOT_ATTEMPT" -lt "$MAX_SPOT_ATTEMPTS" ]; then
                local _decide_json="" _decide_rc=0
                _decide_json="$("$LIB_PYTHON" -m krepis.ec2_spot relaunch-decision \
                    --instance-id "$INSTANCE_ID" \
                    --region "$AWS_REGION" \
                    --attempt "$SPOT_ATTEMPT" \
                    --max-attempts "$MAX_SPOT_ATTEMPTS" \
                    ${SF_EXECUTION_TIMEOUT:+--sf-execution-timeout "$SF_EXECUTION_TIMEOUT" --per-attempt-seconds "$MAX_RUNTIME_SECONDS"} \
                    --json \
                    2>/dev/null)" || _decide_rc=$?
                if [ "$_decide_rc" -ne 0 ]; then
                    echo "    spot relaunch-decision: CLI failed to answer (rc=$_decide_rc) — treating as hold" >&2
                else
                    local _relaunch=""
                    _relaunch="$(printf '%s' "$_decide_json" | "$LIB_PYTHON" -c 'import json,sys; print("1" if json.load(sys.stdin).get("relaunch") else "0")')"
                    echo "    spot relaunch-decision (attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS): $_decide_json"
                    if [ "$_relaunch" = "1" ]; then
                        _will_relaunch=1
                        _alert_sev="warning"
                    fi
                fi
            fi
            # Independent-channel surveillance (SNS + flow-doctor forum
            # topics). Best-effort: keeps cleanup running even if Python /
            # lib / SNS / flow-doctor are unreachable — stdout diagnostic
            # above is primary. This IS the "error-artifact publishing"
            # shared infra named in alpha-engine-config-I4442.
            local _alert_python _alert_msg _fo_err
            _alert_msg="exit_code=$exit_code last_run_ssm='$last_desc' spot_state=$state spot_reason_code='$reason_code' spot_transition_reason='$state_reason' instance_id=${INSTANCE_ID:-<none>} will_relaunch=$_will_relaunch"
            # alpha-engine-config-I11508: $LIB_PYTHON (set above), never this repo's
            # .venv or bare python3. The dispatcher box builds only the dashboard
            # and data venvs, so the old .venv probe fell through to system
            # python3, which has no nousergon_lib — every fan-out died at import
            # and the stderr that said so went to /dev/null.
            _alert_python="$LIB_PYTHON"
            _fo_err="$( (cd "$REPO_ROOT" && "$_alert_python" -c "
import sys
from ops_alerts import publish_ops_alert
publish_ops_alert(
    sys.argv[1],
    severity=sys.argv[2],
    source='alpha-engine-backtester/${SPOT_STAGE_NAME}',
)
" "$_alert_msg" "$_alert_sev") \
                2>&1 >/dev/null)" \
                || echo "    (ops alert fan-out failed via $_alert_python: ${_fo_err##*$'\n'}; primary stdout diagnostic above is the surface)"
        fi
        echo "==> Terminating spot instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
        spot_common_teardown_staging "$exit_code"
        if [ "$_will_relaunch" = "1" ]; then
            echo "==> Spot RECLAIMED by AWS (reason_code='$reason_code' state='$state' transition='$state_reason') — relaunching on a fresh spot (attempt $((SPOT_ATTEMPT + 1))/$MAX_SPOT_ATTEMPTS)"
            trap - EXIT
            SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${_ORIG_ARGS[@]+"${_ORIG_ARGS[@]}"}
        fi
        # CRITICAL: re-exit with the captured status — a bash EXIT trap that
        # ends on a successful command (the `|| true` cleanup steps) would
        # otherwise leave the script exiting 0, masking a Failed SSM step as
        # success to the orchestration wrapper.
        exit "$exit_code"
    }
    trap cleanup EXIT
}

# ── Stage config staging ─────────────────────────────────────────────────────
spot_common_stage_configs() {
    echo "==> Staging configs to ${S3_STAGING}/"
    aws s3 cp "$REPO_ROOT/config.yaml" "${S3_STAGING}/config.yaml" --region "$AWS_REGION" --quiet
    echo "  staged config.yaml"
    aws s3 cp "$EXECUTOR_CONFIG" "${S3_STAGING}/risk.yaml" --region "$AWS_REGION" --quiet
    echo "  staged risk.yaml from $EXECUTOR_CONFIG"
    if [ -n "$PREDICTOR_CONFIG" ]; then
        aws s3 cp "$PREDICTOR_CONFIG" "${S3_STAGING}/predictor.yaml" --region "$AWS_REGION" --quiet
        echo "  staged predictor.yaml from $PREDICTOR_CONFIG"
    fi
}

# ── Wait for the SSM agent to register ───────────────────────────────────────
spot_common_wait_for_ssm_agent() {
    echo "==> Waiting for SSM agent to come Online..."
    local i ping
    for i in $(seq 1 36); do  # 36 x 5s = 180s budget
        ping=$(aws ssm describe-instance-information \
            --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
            --query 'InstanceInformationList[0].PingStatus' \
            --output text --region "$AWS_REGION" 2>/dev/null || true)
        if [ "$ping" = "Online" ]; then
            echo "  SSM agent Online."
            return 0
        fi
        if [ "$i" -eq 36 ]; then
            echo "ERROR: SSM agent not Online after 180s (instance $INSTANCE_ID)"
            exit 1
        fi
        sleep 5
    done
}

# ── SSM dispatch primitive (lib chokepoint) ──────────────────────────────────
# run_ssm "<description>" [timeout_seconds] <<HEREDOC ... HEREDOC
# --diagnostics-bucket/--diagnostics-prefix activate the lib's failure-record
# writer: s3://${S3_BUCKET}/_spot_diagnostics/ae-${SPOT_STAGE_NAME}/{date}.json
# on terminal non-Success. Best-effort inside the lib; inner SSM exit always
# preserved.
run_ssm() {
    local description="$1" timeout_s="${2:-3600}"
    LAST_SSM_DESC="$description"
    "$LIB_PYTHON" -m krepis.ssm_dispatcher run \
        --instance-id "$INSTANCE_ID" \
        --description "${SPOT_STAGE_NAME}: $description" \
        --timeout "$timeout_s" \
        --output-bucket "$S3_BUCKET" \
        --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
        --region "$AWS_REGION" \
        --diagnostics-bucket "$S3_BUCKET" \
        --diagnostics-prefix "_spot_diagnostics/ae-${SPOT_STAGE_NAME}" \
        --script-stdin
}

# ── Bootstrap spot: watchdog + hard timeout + python + clones + configs ─────
# The 30-line heredoc this file carried is GONE — cutover to the fleet's single
# renderer `krepis.spot_bootstrap` (alpha-engine-config-I7372, I6922, I4992).
#
# Why this fork was the last one found. Every function here is prefixed
# `spot_common_*`, so the fleet's original name-anchored sweep
# (`grep -rl ec2-spot-watchdog`) structurally could not see it: it never
# adopted the unit that grep anchored on. Consequently this fork carried the
# HARD RUNTIME CAP (`systemd-run --on-active`) and NO SSM-liveness watchdog at
# all — the exact inverse of `nousergon-data` / `crucible-predictor`, which
# carried the watchdog and no cap. Each fork was uncovered against the other's
# failure mode, and neither could see it in itself.
#
# The two are SEPARATE guarantees and the renderer emits BOTH, always:
#   - hard timeout  → bounds spend when the workload runs long.
#   - SSM watchdog  → terminates an instance whose SSM agent has died, which no
#                     dispatcher can reach and no timeout the dispatcher owns
#                     will ever fire against.
# This cutover is where this repo GAINS the second one; it has never had it.
#
# Where each thing the fork guarded now lives:
#   hard runtime cap           → --max-runtime-seconds (now FATAL if unarmable;
#                                the bare `systemd-run` here failed silently
#                                into `set -e` with no diagnostic)
#   three sibling checkouts    → --extra-clone. Checkout paths are preserved
#                                EXACTLY: the dirs stay `alpha-engine-*` while
#                                the repos are `crucible-*` (2026-06-15 rename),
#                                because the predictor replay imports through
#                                /home/ec2-user/alpha-engine-predictor by path.
#   conditional predictor.yaml → --config-copy-if. The skip is still ANNOUNCED;
#                                an optional artifact vanishing silently is
#                                indistinguishable from one that failed to copy.
#   interpreter                → strict `command -v python3.12 || exit 1`. The
#                                `|| PYTHON_BIN=python3` fallback is gone:
#                                requirements.txt resolves against 3.12, and the
#                                AMI python3 resolves different wheels.
#   branch + clone URLs        → launcher-side literals baked in at render time,
#                                so no `git clone` line names a shell variable
#                                (the crucible-predictor#463 class).
# Also GAINED, absent from the fork: the clone is idempotent
# (`if [ ! -d <checkout>/.git ]`), so a re-dispatched bootstrap no longer fails
# on an existing directory.
spot_common_bootstrap() {
    echo "==> Bootstrapping spot (SSM watchdog, hard timeout, python, 3 clones, configs)..."
    local _script
    _script="$("$LIB_PYTHON" -m krepis.spot_bootstrap render \
        --repo-url "https://github.com/nousergon/crucible-backtester.git" \
        --checkout /home/ec2-user/alpha-engine-backtester \
        --branch "${BRANCH:-main}" \
        --region "$AWS_REGION" \
        --max-runtime-seconds "$MAX_RUNTIME_SECONDS" \
        --export "S3_STAGING=${S3_STAGING}" \
        --extra-clone "/home/ec2-user/alpha-engine=https://github.com/nousergon/crucible-executor.git@${BRANCH:-main}" \
        --extra-clone "/home/ec2-user/alpha-engine-predictor=https://github.com/nousergon/crucible-predictor.git@${BRANCH:-main}" \
        --config-copy "config.yaml:/home/ec2-user/alpha-engine-backtester/config.yaml" \
        --config-copy "risk.yaml:/home/ec2-user/alpha-engine/config/risk.yaml" \
        --config-copy-if "${STAGED_PREDICTOR_CONFIG}:predictor.yaml:/home/ec2-user/alpha-engine-predictor/config/predictor.yaml")"
    # Here-STRING, not a pipe: `run_ssm` records LAST_SSM_DESC for the EXIT
    # trap's "last run_ssm" diagnostic, and a pipeline would run it in a
    # subshell where that assignment dies with the subshell.
    run_ssm "bootstrap" 600 <<<"$_script"
    echo "  Bootstrap complete: 3 repos cloned, configs fetched from ${S3_STAGING}."
}

# ── Install python dependencies ──────────────────────────────────────────────
spot_common_install_deps() {
    echo "==> Installing Python dependencies..."
    run_ssm "deps" 1200 <<DEPS
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=${AWS_REGION} AWS_DEFAULT_REGION=${AWS_REGION}
cd /home/ec2-user/alpha-engine-backtester

# Strict interpreter resolution — NEVER a silent fallback to the AMI python3
# (alpha-engine-config-I7372). requirements.txt is resolved against 3.12;
# python3 resolves different wheels, and the divergence surfaces as an
# ImportError deep inside the workload rather than here, at deps time, with
# the log still in hand. The bootstrap already asserted python3.12 exists;
# if it does not exist HERE, something changed between the two SSM steps and
# that is a hard fail, not a substitution.
command -v python3.12 >/dev/null || {
    echo "FATAL: python3.12 not found on the spot — the bootstrap step asserted it. Refusing to install requirements.txt against a different interpreter (different wheel resolution)." >&2
    exit 1
}
PIP="python3.12 -m pip"

\$PIP install --upgrade pip -q
\$PIP install -q -r requirements.txt

# The predictor checkout is CODE-ONLY (config#3031): its requirements.txt is
# deliberately NOT installed here — co-installing two repos' requirements
# into one resolver namespace let predictor's numpy floor silently override
# the backtester's numpy cap (numba/vectorbt hard ceiling). Every library the
# in-process predictor replay needs at runtime is declared in the
# backtester's OWN requirements.txt.
cd /home/ec2-user/alpha-engine-predictor
if [ ! -d "/home/ec2-user/alpha-engine-predictor/model" ]; then
    echo "FATAL: predictor checkout missing (code-only sys.path dependency)" >&2
    exit 1
fi

cd /home/ec2-user/alpha-engine-backtester
PYBIN="\${PIP% -m pip}"
\$PYBIN -c "import nousergon_lib.quant.stats.multiple_testing, nousergon_lib.quant" || {
    echo "FATAL: nousergon-lib is missing quant.stats — a co-installed sibling repo's pin likely downgraded it below v0.49.0. Resolved version:" >&2
    \$PIP show nousergon-lib | grep -E '^Version:' >&2 || true
    exit 1
}
\$PIP show nousergon-lib | grep -E '^Version:'

\$PYBIN -c "from synthetic.predictor_backtest import run; from synthetic.production_signal_backtest import build_production_signal_inputs" || {
    echo "FATAL: predictor modules missing or failed to import" >&2
    exit 1
}

# Fail-loud numpy-2 consistency guard (config#2815). Asserts the exact
# import chains that have broken production runs before, so any future
# co-installed pin that downgrades numpy breaks LOUD here at deps time
# (seconds) instead of deep into the run.
\$PYBIN -c "import numpy, scipy.sparse, lightgbm, numba, vectorbt; assert int(numpy.__version__.split('.')[0]) >= 2, 'numpy '+numpy.__version__+' < 2.0 is inconsistent with the numpy-2-built scipy/cvxpy stack (config#2815)'; print('numpy-2 guard OK: numpy='+numpy.__version__+' scipy='+scipy.__version__+' lightgbm='+lightgbm.__version__+' numba='+numba.__version__+' vectorbt='+vectorbt.__version__)" || {
    echo "FATAL: import-chain consistency check failed — a co-installed pin, stale downgrade, or numba/numpy ABI mismatch broke the scipy/lightgbm or numba/vectorbt import chain (config#2815, config-I3279). See traceback above." >&2
    exit 1
}

# Fail-loud pip-check dependency-consistency gate (config#2973). Gate on
# pip check's EXIT CODE, not on output emptiness — a clean env prints
# non-empty "No broken requirements found." and exits 0.
PIP_CHECK_ALLOWLIST=""
PIP_CHECK_RC=0
PIP_CHECK_OUT=\$(\$PYBIN -m pip check 2>&1) || PIP_CHECK_RC=\$?
if [ "\$PIP_CHECK_RC" -eq 0 ]; then
    echo "pip check: clean (exit 0)."
else
    if [ -z "\$PIP_CHECK_ALLOWLIST" ]; then
        PIP_CHECK_REMAIN="\$PIP_CHECK_OUT"
    else
        PIP_CHECK_REMAIN=\$(printf '%s\n' "\$PIP_CHECK_OUT" | grep -vFf <(printf '%s\n' "\$PIP_CHECK_ALLOWLIST") || true)
    fi
    if [ -n "\$PIP_CHECK_REMAIN" ]; then
        echo "FATAL: pip check reported non-allowlisted dependency conflicts:" >&2
        printf '%s\n' "\$PIP_CHECK_REMAIN" >&2
        exit 1
    fi
    echo "pip check: all reported conflicts are allowlisted."
fi

echo "Dependencies installed."
DEPS
}

# ── Predictor sector_map cache fetch ─────────────────────────────────────────
spot_common_fetch_predictor_cache() {
    echo "==> Downloading predictor sector_map from S3..."
    run_ssm "predictor-cache" 300 <<'CACHE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp
CACHE_DIR="/home/ec2-user/alpha-engine-predictor/data/cache"
mkdir -p "$CACHE_DIR"
aws s3 cp s3://alpha-engine-research/reference/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null \
    || aws s3 cp s3://alpha-engine-research/predictor/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null \
    || true
echo "Predictor cache dir: sector_map.json $([ -f "$CACHE_DIR/sector_map.json" ] && echo present || echo MISSING)"
CACHE
}

# ── Build env export command ─────────────────────────────────────────────────
# PYTHONUNBUFFERED=1: line-buffering so SSM ships log lines as emitted.
# ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true: the sim hot loop would
# otherwise emit ~50k-200k per-decision S3 PUTs and blow the watchdog.
# PYTHON_BIN is resolved STRICTLY here — `python3.12` or a non-zero exit.
# This one line is injected into EVERY downstream stage heredoc (smoke, backtest,
# parity, evaluate, preflight), so the old
# `command -v python3.12 && PYTHON_BIN=python3.12 || PYTHON_BIN=python3`
# was not one silent fallback but the fallback for the entire run: fixing only
# the bootstrap would have left every workload step free to resolve the AMI
# python3 against wheels installed for 3.12 (alpha-engine-config-I7372).
spot_common_build_env_source() {
    ENV_SOURCE='export XDG_CACHE_HOME=/tmp; export PYTHONUNBUFFERED=1; export ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true; export AWS_REGION=us-east-1; export AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1; command -v python3.12 >/dev/null || { echo "FATAL: python3.12 absent on the spot — bootstrap asserted it and deps installed against it. Refusing to run this step on a different interpreter (alpha-engine-config-I7372)." >&2; exit 1; }; PYTHON_BIN=python3.12; export PYTHON_BIN;'
    # shellcheck disable=SC2016
    # Deliberately single-quoted: this is a literal '$PYTHON_BIN' TOKEN
    # interpolated into each stage heredoc, resolved by the REMOTE spot's
    # own shell (via the PYTHON_BIN export in ENV_SOURCE above) — not by
    # this dispatcher.
    REMOTE_PYTHON='$PYTHON_BIN'
    # Carry the SF execution name onto the spot (alpha-engine-config-I11505).
    # krepis.ssm_log_capture exports --correlation-id $$.Execution.Name to this
    # dispatcher as RUN_TOKEN; without forwarding it the spot cannot tell a
    # rehearsal from a real weekly run, and rehearsal-2026-09-23-2 cut over
    # live config/executor_params.json. optimizer/run_role.py reads it. The
    # value is reduced to [A-Za-z0-9._-] (SF execution-name charset) so it is
    # safe inside the single-quoted export.
    local _run_token="${RUN_TOKEN:-}"
    _run_token="${_run_token//[^A-Za-z0-9._-]/}"
    ENV_SOURCE="${ENV_SOURCE} export RUN_TOKEN='${_run_token}';"
}

# ── Preflight-only (Friday shell_run dry path) ──────────────────────────────
# Runs ONLY backtest.py --mode=smoke (BacktesterPreflight + _runtime_smoke —
# lib-pin / imports / predictor-weights / universe-freshness, ~30-60s,
# read-only), then exit 0 — before ANY stage-specific spend. Returns 0 (does
# NOT exit) when PREFLIGHT_ONLY=0 so the caller continues into its own stage.
spot_common_run_preflight_only_and_maybe_exit() {
    if [ "$PREFLIGHT_ONLY" != "1" ]; then
        return 0
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PREFLIGHT-ONLY (Friday shell_run dry path)"
    echo "  boot + deps done; running bootstrap-class smoke harness only,"
    echo "  then exit 0 — NO stage workload, ZERO external API calls,"
    echo "  ZERO S3/config writes."
    echo "═══════════════════════════════════════════════════════════════"
    run_ssm "preflight-only" 900 <<PREFLIGHT
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

echo "==> Preflight: backtest.py --mode=smoke"
$REMOTE_PYTHON -u backtest.py --mode=smoke --log-level INFO 2>&1
PREFLIGHT

    echo ""
    echo "==> Preflight-only PASSED — bootstrap-class smoke clean."
    echo "==> Instance will be terminated (no stage workload, no S3/config writes performed)."
    exit 0
}

# ── Per-stage CloudWatch heartbeat ───────────────────────────────────────────
spot_common_emit_heartbeat() {
    local _process="$1"
    aws cloudwatch put-metric-data \
        --namespace "AlphaEngine" \
        --metric-name "Heartbeat" \
        --dimensions "Process=${_process}" \
        --value 1 --unit "Count" \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
        && echo "Heartbeat emitted: ${_process}" \
        || echo "WARNING: Failed to emit heartbeat for ${_process} (non-fatal)"
}

# ── Cheap dispatcher-side S3 substrate checks (weekly-sf-policy.md §2.2) ────
# `aws s3 ls` on a single key/prefix — no spend, no spot launch. Each stage
# script's own preflight_<stage>() function calls these to assert its
# specific upstream artifact exists BEFORE provisioning a spot, rather than
# discovering the miss 15-40 min into boot+deps (the exact 2026-07-25
# Parity incident this issue is named for — parity failed after 37 minutes
# because its code path wasn't exercised until late in the monolith).
spot_common_s3_key_exists() {
    aws s3 ls "s3://${S3_BUCKET}/$1" --region "$AWS_REGION" >/dev/null 2>&1
}

spot_common_s3_prefix_nonempty() {
    local out
    out=$(aws s3 ls "s3://${S3_BUCKET}/$1" --region "$AWS_REGION" 2>/dev/null || true)
    [ -n "$out" ]
}
