#!/usr/bin/env bash
# infrastructure/spot_backtest.sh — Run weekly backtest on a spot EC2 instance.
#
# Launches a c5.large spot instance (~$0.03/hr), clones the backtester +
# predictor + executor repos, runs the full backtest pipeline with 10y of
# price data, uploads results to S3, and self-terminates.
#
# Usage:
#   ./infrastructure/spot_backtest.sh                   # full run (--mode all)
#   ./infrastructure/spot_backtest.sh --smoke-only      # quick validation, then terminate
#   ./infrastructure/spot_backtest.sh --preflight-only  # boot + deps + the
#                                                       #   bootstrap-class smoke
#                                                       #   harness only
#                                                       #   (backtest.py --mode=smoke:
#                                                       #   BacktesterPreflight +
#                                                       #   _runtime_smoke — lib-pin /
#                                                       #   imports / predictor-weights /
#                                                       #   universe-freshness, ~30-60s,
#                                                       #   from PRs #43-#48), then
#                                                       #   exit 0 — NO param sweep,
#                                                       #   NO portfolio sim, NO parity,
#                                                       #   NO evaluator, NO config/*.json
#                                                       #   auto-apply, ZERO external API
#                                                       #   calls, ZERO S3/config writes.
#                                                       #   Friday shell_run dry path
#                                                       #   (ROADMAP "Friday shell-run —
#                                                       #   per-module dry-path
#                                                       #   activation" owed-item #3).
#   ./infrastructure/spot_backtest.sh --mode simulate   # override backtest mode
#   ./infrastructure/spot_backtest.sh --instance-type c5.xlarge  # override instance type
#   ./infrastructure/spot_backtest.sh --dry-run         # full-universe exercise without
#                                                       #   production S3 pollution:
#                                                       #   markers + artifacts + reports
#                                                       #   go to .dry-run/{date}/, no
#                                                       #   optimizer config writes, no
#                                                       #   reporter upload. Safe to run
#                                                       #   concurrently with scheduled SF.
#   ./infrastructure/spot_backtest.sh --use-vectorized-sweep  # run predictor_param_sweep
#                                                       #   through the matrix-axis vectorized
#                                                       #   engine (Tier 4). Default off until
#                                                       #   v14 spot validation confirms parity.
#
# Prerequisites:
#   - AWS CLI with perms to RunInstances / TerminateInstances /
#     DescribeInstances / SendCommand / GetCommandInvocation
#   - /opt/nousergon/bin/lib-python present on the dispatcher host — the
#     ops-owned guard over the declared krepis venv (ec2_spot +
#     ssm_dispatcher CLIs); LIB_PYTHON names it
#   - Code committed and pushed to origin (instance clones from GitHub
#     via HTTPS — no SSH key needed)
#   - config.yaml + executor risk.yaml + predictor predictor.yaml
#     (gitignored — staged to S3 by this script for the spot to fetch
#     via its alpha-engine-executor-profile IAM role). config.yaml carries
#     the non-secret runtime config (EMAIL_SENDER, EMAIL_RECIPIENTS,
#     OUTPUT_BUCKET) the .env used to hold (#890 deprecated the .env).
#
# **2026-05-27 — SSH/SCP → SSM transport migration (ROADMAP L342 PR 3).**
# Mirrors alpha-engine-data PR 2 (#330). Communication with the spot is
# now via `aws ssm send-command` wrapped at the lib chokepoint
# `python -m krepis.ssm_dispatcher run` (invoked directly via krepis per
# config#1649 — the nousergon_lib re-export shim is guard-less under
# `python -m` on lib >=0.81.0 and silently no-ops). No port-22 inbound on
# the spot SG; no ssh / scp / ssh-keyscan. The 3 config files (no .env post
# #890) are staged to a temporary S3 prefix and pulled down by the spot. PR 3 of
# the 5-PR L342 arc.
#
# For scheduled weekly runs, call this script from the always-on EC2 cron
# or from an EventBridge → Lambda trigger:
#
#   0 8 * * 1  cd ~/alpha-engine-backtester && bash infrastructure/spot_backtest.sh >> /var/log/backtester-spot.log 2>&1

set -euo pipefail

# ── Ensure HOME is set (SSM RunCommand does not set it) ──────────────────────
export HOME="${HOME:-/home/ec2-user}"

# ── Path setup ───────────────────────────────────────────────────────────────
# .env fully deprecated (#890). Secrets load from SSM via
# alpha_engine_lib.secrets.get_secret() at Python startup (the EC2 instance
# role grants ssm:GetParameter on /alpha-engine/*). The remaining non-secret
# runtime config the .env used to carry (EMAIL_SENDER, EMAIL_RECIPIENTS,
# OUTPUT_BUCKET) now lives in config.yaml — already staged to S3 and fetched
# by the spot — so no .env is staged, fetched, or sourced anywhere below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
# Capacity-resilient instance-type fallback set (2026-05-22 incident:
# THIS LAUNCHER's Evaluator invocation hit InsufficientInstanceCapacity
# for c5.large in subnet-e07166ec / us-east-1f).
#
# CORRECTED 2026-08-13 (alpha-engine-config-I7216): this set is six 4 GB types
# followed by one 8 GB type (m5.large), and this comment used to call them all
# "equivalent for the backtester (memory-bound)". A memory-bound job's instance types are not
# equivalent across a 2x RAM spread — that pairing made the launcher's memory
# budget depend on which spot capacity pool answered, so an OOM presented as
# intermittent flakiness. Every mode that actually needs memory now sets its
# own floor below (see the two _*_RAM_FLOOR_TYPES blocks); this rotation is
# the capacity-resilient default for modes with no floor, and must not be
# read as a statement that its members are interchangeable under memory
# pressure.
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
INSTANCE_TYPE=""  # backward-compat: --instance-type X collapses INSTANCE_TYPES to single value
AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
# Spot-side watchdog budget: backtester's 10y simulate + param sweep
# historically runs 60-100 min. 120 min with headroom. Bump (don't
# silently rely on the orphan reaper) if a run legitimately needs more.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"
# KEY_NAME kept ONLY as launch attribute for alpha_engine_lib.ec2_spot's
# --key-name flag — the spot still launches with the key associated, but
# NOTHING in this script SSHs in. Communication is via SSM. KEY_FILE was
# removed in the 2026-05-27 SSH→SSM migration (PR 3 of L342); manual
# break-glass SSH is possible only by temporarily re-opening the SG's
# port-22 inbound (which it should NOT be in steady state).
KEY_NAME="alpha-engine-key"
# alpha-engine-config#3018 (IAM-as-code surface 4, fleet audit config#2340):
# SECURITY_GROUP / SUBNETS are EC2 *launch* attributes, not an IAM policy
# surface — ec2:RunInstances on alpha-engine-executor-role is granted with
# Resource:"*" (see crucible-executor/infrastructure/iam/
# alpha-engine-executor-role/alpha-engine-ec2-spot.json), i.e. it is not
# scoped by subnet or SG, so neither value gates access and neither can
# "drift" against an IAM policy document. SUBNETS is the shared, non-secret
# 6-subnet capacity-fallback rotation set duplicated verbatim across the
# data/predictor/backtester spot launchers (same default VPC vpc-566f002e,
# same SG) — accepted-by-design as plain launch config, intentionally out
# of IAM-as-code scope. SECURITY_GROUP likewise has no SG-as-code tracking
# convention anywhere in this fleet (SGs only appear as ARN scoping inside
# individual IAM policy *statements*, e.g. alpha-engine-dashboard-role's
# spot-dispatch policy — there is no standalone tracked SG doc to mirror).
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
# All 6 default-VPC subnets across us-east-1{a..f}. The lib CLI rotates
# across this list on capacity error. Lockstep with data + predictor
# launchers (same VPC vpc-566f002e, same SG).
SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
# IAM_PROFILE is NOT ungoverned: alpha-engine-executor-profile backs
# alpha-engine-executor-role, which IS tracked+applied+drift-checked —
# source of truth is crucible-executor/infrastructure/iam/
# alpha-engine-executor-role/ (apply.sh + check-drift.py, daily CI +
# per-PR via .github/workflows/iam-drift-check.yml, OIDC role
# github-actions-iam-drift-check). That role's ReadWriteBacktestResults
# S3 statement (backtest/*) exists specifically for this script's spot
# runs. This repo was already a scanned sibling in crucible-executor's
# check-no-foreign-writers.py (foreign-writers-check job); this script
# only ever *references* the profile as a launch attribute (--iam-profile)
# and never writes IAM, so it was already compliant with the single-writer
# rule — this comment just makes the cross-repo ownership link legible
# from the consumer side, which is what config#3018 flagged as missing.
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
BACKTEST_MODE="all"

# ── Parse flags ──────────────────────────────────────────────────────────────
RUN_MODE="full"  # full | smoke-only
# PREFLIGHT_ONLY is a MODIFIER, orthogonal to RUN_MODE — matching the
# data (spot_data_weekly.sh #259) and predictor (spot_train.sh #175)
# siblings' verbatim --preflight-only flag for cross-script consistency
# (the Friday shell_run SF keystone follow-on dispatches the same flag
# name to every module). When set, the script boots + installs deps for
# real, runs ONLY the bootstrap-class smoke harness (backtest.py
# --mode=smoke = BacktesterPreflight + _runtime_smoke; ~30-60s,
# read-only), then `exit 0` BEFORE the per-phase smoke modes, the
# evaluate.py S3-probe diagnostics, AND the entire full-backtest heredoc
# (param sweep / portfolio sim / parity / pit_parity / evaluator /
# config/*.json optimizer auto-apply / CloudWatch heartbeats). Catches
# bootstrap-class breakage (lib-pin drift, sys.path collision, stale
# ArcticDB universe, missing predictor weights, SSM timeout, image gap)
# ~12h before the real Saturday Backtester. backtest.py --mode=smoke
# itself `return`s before _init_pipeline / the optimizer, so it writes
# no S3 config; gating in front of the full heredoc + the
# evaluate.py/per-phase smoke block makes every sweep/sim/parity/
# evaluator and every config/{executor,scoring,predictor,research,
# scanner}_params*.json writer statically unreachable under this flag.
PREFLIGHT_ONLY=0
# All PhaseRegistry-adjacent flags are also routable from the
# Saturday SF input via env vars. When set they pass through as
# CLI args to backtest.py.
SKIP_PHASE4="${SKIP_PHASE4_EVALUATIONS:-false}"
SKIP_PHASES="${SKIP_PHASES:-}"            # comma-separated phase names
ONLY_PHASES="${ONLY_PHASES:-}"            # comma-separated phase names
FORCE_ALL="${FORCE_ALL:-false}"           # true → --force
FORCE_PHASES="${FORCE_PHASES:-}"          # comma-separated phase names
DRY_RUN="${DRY_RUN:-false}"               # true → --dry-run
# Pipeline-level stage control: comma-separated subset of {backtest, parity,
# evaluator}. All three stages run by default on the spot. Used for fast
# iteration against a single stage (e.g. parity-only when debugging a cred
# divergence).
SKIP_STAGES="${SKIP_STAGES:-}"
# config-I3112 (2026-07-20 Brian ruling): the weekly SF's single bundled
# Evaluator state is decomposed into two sequential states —
# EvaluatorDiagnostics (--eval-half=diagnostics) then EvaluatorOptimize
# (--eval-half=optimize) — each with its own executionTimeout + Catch path,
# reusing this same spot box. This flag selects which HALF of evaluate.py's
# internal pipeline this spot invocation runs: "all" (default, the bundled
# behavior) maps to evaluate.py --mode all; the two halves map to
# --mode diagnostics / --mode optimize. The optimize half reads the
# diagnostics snapshot the diagnostics half wrote (evaluate_handoff.py) —
# see evaluate.py's S3-mediated handoff. Unknown values hard-fail below
# (no-silent-fails), mirroring the --skip-stages typo guard.
EVAL_HALF="${EVAL_HALF:-all}"
# pit_parity observational stage (ROADMAP L2371 / plan §D4). DEFAULT ON
# 2026-05-17 (Brian): every Saturday SF spot run now emits
# backtest/{date}/pit_parity.json (the skilled-risk-basket contamination
# report). NON-BLOCKING + writes no configs + does NOT flip --walk-forward
# (the L2371 close is the separate, manual, post-review step). Opt out per
# run with --no-pit-parity or PIT_PARITY_ENABLED=0 (ad-hoc/dry iterations
# where the extra predictor-sim pass isn't wanted).
PIT_PARITY_ENABLED="${PIT_PARITY_ENABLED:-1}"
# RUN_DATE: the single artifact-date label for backtest/{date}/ (param
# sweep + portfolio_stats + parity + pit_parity + evaluator inputs). The
# Saturday SF stamps this ONCE at InitializeInput from
# $$.Execution.StartTime and threads it (export RUN_DATE=…) into the
# Backtester / Parity / Evaluator SSM commands — each a SEPARATE spot
# instance with its own spot_backtest.sh invocation. Resolving it here
# from the injected env (not per-stage wall-clock) is what keeps all
# three stages keyed to the SAME prefix when a multi-hour run straddles
# UTC midnight (the 2026-05-17 Evaluator failure: Backtester wrote
# backtest/2026-05-17/, Evaluator looked in backtest/2026-05-18/). Same
# dispatcher→heredoc bake-in mechanism as SKIP_STAGES / PIT_PARITY_ENABLED.
# Falls back to wall-clock UTC for ad-hoc manual runs that don't inject it.
RUN_DATE="${RUN_DATE:-$(date -u +%Y-%m-%d)}"
# DATE_CONVENTIONS: normalize RUN_DATE to the NYSE TRADING DAY at this single
# dispatcher-side chokepoint, BEFORE it is threaded into every stage's --date
# AND the bash s3 uploads below (so python + bash never split). The SF threads
# $.run_date = date(Execution.StartTime) (CALENDAR — Sat 2026-05-30 on a
# Saturday firing) but Research + signals.json + the standalone scanner key by
# trading day (Fri 2026-05-29); keying backtest/{date}/ (incl. pit_parity.json
# + parity_metrics) by the calendar date is what surfaced the research↔backtester
# pit-parity drift (L4466). $LIB_PYTHON (line ~125) carries nousergon_lib.
# Defensive: keep the calendar value if the lib call fails (a normalization
# miss must not abort the backtester) — the python entry points re-normalize
# idempotently as a backstop.
_RUN_DATE_TD="$("$LIB_PYTHON" -c "import datetime as d; from nousergon_lib import trading_calendar as tc; x=d.date.fromisoformat('${RUN_DATE}'[:10]); print(x.isoformat() if tc.is_trading_day(x) else tc.previous_trading_day(x).isoformat())" 2>/dev/null || true)"
if [ -n "$_RUN_DATE_TD" ]; then
    if [ "$_RUN_DATE_TD" != "$RUN_DATE" ]; then
        echo "==> Normalized RUN_DATE ${RUN_DATE} (calendar) → ${_RUN_DATE_TD} (trading day) per DATE_CONVENTIONS"
    fi
    RUN_DATE="$_RUN_DATE_TD"
else
    echo "WARNING: trading-day normalization of RUN_DATE=${RUN_DATE} failed — keeping calendar value (python entry points will re-normalize)" >&2
fi
# Freeze the evaluator (passes --freeze to evaluate.py → suppresses per-
# optimizer S3 config writes; report artifacts + email still upload). Use
# for off-cycle test runs so mid-week sweeps don't auto-promote weights/
# params/thresholds against Monday trading. Replaces the retired SF
# CheckEvaluatorFreeze Choice state (evaluator consolidated into spot
# 2026-04-24); the freeze_evaluator SF input param is no longer honored.
FREEZE_EVALUATOR="${FREEZE_EVALUATOR:-false}"
USE_VECTORIZED_SWEEP="${USE_VECTORIZED_SWEEP:-false}"
# Accept both --flag value and --flag=value forms for every value-taking
# flag. The equals form is GNU-getopt-style muscle memory and it's cheap to
# support — each value flag gets a companion `--foo=*` case that splits on
# `=`. Boolean flags (--smoke-only, --force, --dry-run, etc.) accept no
# value and don't need the companion case.

# #883 — bounded mid-run spot-reclaim relaunch. The Saturday SF's per-state
# Retry is on the `ssm:sendCommand` Task, which only SENDS the command and
# returns — the actual run is polled by a separate Choice loop, so a worker
# spot reclaimed mid-run (Server.SpotInstanceTermination /
# instance-terminated-no-capacity) surfaces in the poll as a generic Failed,
# NOT a sendCommand TaskFailed → the SF Retry never fires (its "handles spot
# interruption" comment is structurally wrong). This dispatcher (which OWNS
# the worker-spot lifecycle) is the only layer that can see the reclaim
# reason. Originally (L4485-b, #283/#289) this classified the reclaim INLINE
# and self-relaunched via a decrementing RECLAIM_RELAUNCH_MAX budget — a
# divergent copy of the identical logic in alpha-engine-data's #349 reference
# implementation and the predictor launcher's gap (no relaunch at all). Per
# #883, the classify→decide DECISION is now the lib chokepoint
# `python -m krepis.ec2_spot relaunch-decision` (lib v0.65.0+; already
# satisfied by this repo's nousergon-lib@v0.78.0 / krepis>=0.4.0 pins — no
# bump needed). Invoked directly via krepis, NOT `nousergon_lib.ec2_spot`
# (config#1649 / config#1646): on lib >=0.81.0 `nousergon_lib.ec2_spot` is a
# guard-less re-export shim that silently no-ops under `python -m` — this
# launcher's other lib CLI callsites already migrated off it, so the new
# relaunch-decision callsite must follow the same convention from day one.
# ONLY a confirmed reclaim relaunches; a genuine workload failure (OOM /
# crash / timeout) classifies as "other"/"unknown" and fails loud — a blind
# retry would mask a real bug (feedback_no_silent_fails). SPOT_ATTEMPT is
# threaded across re-execs via the env (first run = 1). Happy path unchanged.
#
# MAX_SPOT_ATTEMPTS ↔ per-attempt-budget coupling (#883 requirement): each
# attempt costs boot time plus up to MAX_RUNTIME_SECONDS of workload. The
# lib's --sf-execution-timeout/--per-attempt-seconds guard refuses to advise a
# relaunch the OUTER budget cannot absorb. The Saturday backtest runs from a
# weekly cron (`spot_backtest.sh`), NOT under a Step-Functions
# executionTimeout, so there is no outer SF budget to couple to —
# SF_EXECUTION_TIMEOUT defaults empty (guard inert; bound is
# MAX_SPOT_ATTEMPTS only). If this launcher is ever wired under an SF state
# with an executionTimeout, set SF_EXECUTION_TIMEOUT to that budget and the
# lib guard activates ((attempt+1)*MAX_RUNTIME_SECONDS must fit).
#
# MAX_SPOT_ATTEMPTS=4 preserves the prior RECLAIM_RELAUNCH_MAX=3 budget (3
# relaunches = 4 total attempts): 2026-06-06 saw TWO consecutive reclaims
# during a capacity-volatile window, so a lower bound would exhaust on such a
# streak. Each relaunch resumes cheaply via the S3 phase auto-skip markers
# (completed phases are skipped on the fresh spot), so the higher bound is
# low-cost and only ever burns on a CLASSIFIED reclaim.
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-4}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"
_ORIG_ARGS=("$@")  # captured pre-parse for the relaunch exec

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only) RUN_MODE="smoke-only"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --instance-type=*) INSTANCE_TYPE="${1#*=}"; shift ;;
        --mode) BACKTEST_MODE="$2"; shift 2 ;;
        --mode=*) BACKTEST_MODE="${1#*=}"; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --branch=*) BRANCH="${1#*=}"; shift ;;
        --skip-phase4-evaluations) SKIP_PHASE4="true"; shift ;;
        --skip-phases) SKIP_PHASES="$2"; shift 2 ;;
        --skip-phases=*) SKIP_PHASES="${1#*=}"; shift ;;
        --only-phases) ONLY_PHASES="$2"; shift 2 ;;
        --only-phases=*) ONLY_PHASES="${1#*=}"; shift ;;
        --force) FORCE_ALL="true"; shift ;;
        --force-phases) FORCE_PHASES="$2"; shift 2 ;;
        --force-phases=*) FORCE_PHASES="${1#*=}"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --skip-stages) SKIP_STAGES="$2"; shift 2 ;;
        --skip-stages=*) SKIP_STAGES="${1#*=}"; shift ;;
        --eval-half) EVAL_HALF="$2"; shift 2 ;;
        --eval-half=*) EVAL_HALF="${1#*=}"; shift ;;
        --no-pit-parity) PIT_PARITY_ENABLED="0"; shift ;;
        --pit-parity-enabled) PIT_PARITY_ENABLED="$2"; shift 2 ;;
        --pit-parity-enabled=*) PIT_PARITY_ENABLED="${1#*=}"; shift ;;
        --run-date) RUN_DATE="$2"; shift 2 ;;
        --run-date=*) RUN_DATE="${1#*=}"; shift ;;
        --freeze-evaluator) FREEZE_EVALUATOR="true"; shift ;;
        --use-vectorized-sweep) USE_VECTORIZED_SWEEP="true"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Validate --eval-half against the known vocabulary ───────────────────────
# config-I3112: {all, diagnostics, optimize} map 1:1 to evaluate.py's
# --mode values. Hard-fail on anything else per no-silent-fails — a typo
# like --eval-half=diagnostic would otherwise silently run the FULL
# evaluator (mode "all"-ish default path) and mislead the operator.
case "$EVAL_HALF" in
    all|diagnostics|optimize) ;;
    *)
        echo "ERROR: unknown --eval-half='$EVAL_HALF'" >&2
        echo "       Valid values: all diagnostics optimize" >&2
        exit 1
        ;;
esac

# ── Validate --skip-stages against the known stage vocabulary ────────────────
# Hard-fail on unknown names per no-silent-fails: a typo like
# --skip-stages=evaulator would silently run evaluator (no match) and mislead
# the operator into thinking the pipeline respected their request.
_KNOWN_STAGES="backtest pit_parity parity evaluator"
if [ -n "$SKIP_STAGES" ]; then
    IFS=',' read -ra _SKIP_ARR <<< "$SKIP_STAGES"
    for _s in "${_SKIP_ARR[@]}"; do
        _s_trim="$(echo "$_s" | tr -d '[:space:]')"
        case " $_KNOWN_STAGES " in
            *" $_s_trim "*) ;;
            *)
                echo "ERROR: unknown stage '$_s_trim' in --skip-stages=$SKIP_STAGES" >&2
                echo "       Valid stages: $_KNOWN_STAGES" >&2
                exit 1
                ;;
        esac
    done
fi

# Convert each flag to a backtest.py CLI arg suffix (empty string when
# disabled, so we don't pass an invalid empty arg through the heredoc).
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

# Smoke-safe subset of BACKTEST_PHASE_FLAGS. Smoke modes set their own
# only-/skip-phases via `_apply_smoke_fixture`, so propagating the
# operator's --skip-phases / --only-phases / --force-phases would
# conflict with the fixture's narrowing semantics. Only flags that
# affect compute behavior (not phase selection) flow through. Currently
# just --use-vectorized-sweep — added for Tier 4 Layer 2 smoke
# validation (ROADMAP P0 2026-04-27). Without this, the host parses
# --use-vectorized-sweep but no smoke command ever sees the flag, so
# `smoke-predictor-param-sweep` would silently exercise the scalar path.
SMOKE_PHASE_FLAGS=""
if [ "$USE_VECTORIZED_SWEEP" = "true" ]; then
    SMOKE_PHASE_FLAGS="$SMOKE_PHASE_FLAGS --use-vectorized-sweep"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Backtester Spot Run — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"

# ── Phase-aware instance-type floor (L4485) ──────────────────────────────────
# Modes that run predictor_pipeline (10y GBM inference over ~900 tickers;
# peak RSS ~2.8 GB measured 2026-06-01) need ≥16 GB RAM. The 4 GB c5.large —
# FIRST in the default rotation — OOM-killed predictor_pipeline on the
# 2026-06-01 off-cycle run. CRITICAL: the Saturday SF's PredictorBacktest +
# PortfolioOptimizerBacktest states invoke this script with NO --instance-type,
# so without this floor they inherit the c5.large-first default and OOM
# identically on the next weekly cycle (the operator's "why would Saturday
# succeed?" — it wouldn't). Setting the floor HERE fixes both the off-cycle
# --mode=all path and the SF split-states with zero edits to the Step Function.
# param-sweep / simulate / signal-quality don't load the predictor tensor and
# stay on the cheap 4 GB-first rotation. Skipped when the operator passes an
# explicit --instance-type (their choice wins, incl. deliberate small debug).
_PREDICTOR_RAM_FLOOR_TYPES="m5.xlarge,m6i.xlarge,m5a.xlarge,c5.2xlarge,c6i.2xlarge"
# I3280 (2026-07-23): the universal predictor floor was bumped from 8 GB to 16 GB
# instances — the RAM headroom guard requires ≥6.0 GB available MemAvailable, but
# 8 GB instances can dip to ~6 GB under OS overhead + ArcticDB caches, leaving
# zero margin against the requirement. 16 GB instances (~13-14 GB available)
# provide comfortable margin. pit_parity shares this same universal floor
# (subprocess isolation already bounded its per-pass footprint to ~2.8 GB).
# alpha-engine-config-I7216 (2026-08-13): param-sweep needs a floor too.
#
# The block above carved param-sweep OUT of the predictor floor on the stated
# assumption that it "doesn't load the predictor tensor and stays on the cheap
# 4 GB-first rotation". That assumption is now false, and it failed silently
# for days:
#
#   bash: line 16: 26748 Killed  python -u backtest.py --mode param-sweep ...
#
# A kernel OOM kill on a 4 GB c5.large, 2026-08-13 (execution
# rehearsal-2026-08-13-2, instance i-0eba8344f3a124b3c). The blast radius was
# not "one backtest run": Backtester precedes PredictorBacktest, which writes
# predictor/research_free_backfill/predictor_outcomes_research_free.parquet —
# the LIVE ENTRY FEED while scanner_predictor_direct is champion. With
# Backtester dying, that cohort froze at prediction_date 2026-08-07 and every
# trading day since drew its entry candidates from the same stale pool.
# Distinct names newly entered fell from ~20/month to 3.
#
# It also failed INTERMITTENTLY rather than always, which is why it read as
# flakiness: the default rotation `c5.large,m5.large,c6i.large,c5a.large` is
# 4 GB, 8 GB, 4 GB, 4 GB. Whether the job survived depended on which capacity
# pool answered — a coin flip on memory, described in this file's own comment
# as "All 2 vCPU / 4-8 GB RAM — equivalent for the backtester (memory-bound...)".
# For a memory-bound job those types are not equivalent, and that is the sentence
# the defect lived inside.
#
# SIZING — deliberately over-provisioned, and deliberately not a guess dressed
# as a measurement. The only fact in hand is that it died at 4 GB: an OOM-killed
# process reports no peak, so the true requirement is UNKNOWN.
#
# Reusing the 16 GB predictor tier rather than stepping to 8 GB is an asymmetry
# call, not a capacity estimate. The block above records that predictor_pipeline
# peaks at ~2.8 GB RSS and is STILL given 16 GB, because 8 GB instances "dip to
# ~6 GB available under OS overhead + ArcticDB caches" (config-I3280). This job
# reads the same ArcticDB feature store. Against that, the cost of guessing low
# is another failed nightly run — and because Backtester gates PredictorBacktest,
# a failed run is another day of trading on a frozen entry cohort. The cost of
# guessing high is a few cents of spot per run.
#
# This is therefore a CEILING to trade under until the number is known, not a
# budget. Right-sizing DOWN from a measured peak RSS on a surviving run is the
# follow-up on alpha-engine-config-I7216 — a cap derived from another cap is not
# a budget, and this comment exists so the next reader does not treat 16 GB as
# evidence of anything.
_PARAM_SWEEP_RAM_FLOOR_TYPES="$_PREDICTOR_RAM_FLOOR_TYPES"

case "$BACKTEST_MODE" in
    all|predictor-backtest|portfolio-optimizer-backtest)
        if [ -z "$INSTANCE_TYPE" ]; then
            echo "  Mode '$BACKTEST_MODE' runs predictor_pipeline → applying ≥16 GB instance floor"
            INSTANCE_TYPES="$_PREDICTOR_RAM_FLOOR_TYPES"
        fi
        ;;
    param-sweep|simulate|signal-quality)
        if [ -z "$INSTANCE_TYPE" ]; then
            echo "  Mode '$BACKTEST_MODE' → applying ≥8 GB instance floor (config-I7216: OOM-killed on 4 GB c5.large 2026-08-13)"
            INSTANCE_TYPES="$_PARAM_SWEEP_RAM_FLOOR_TYPES"
        fi
        ;;
esac

if [ -n "$INSTANCE_TYPE" ]; then
    INSTANCE_TYPES="$INSTANCE_TYPE"  # --instance-type X collapses to single value
fi
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Backtest mode : $BACKTEST_MODE"
echo "  Run mode      : $RUN_MODE"
echo "  Preflight-only: $PREFLIGHT_ONLY  (1 = boot + deps + smoke harness + exit 0, NO sweep/sim/parity/evaluator/auto-apply, ZERO writes)"
echo "  Skip phase 4  : $SKIP_PHASE4"
echo "  Skip phases   : ${SKIP_PHASES:-(none)}"
echo "  Only phases   : ${ONLY_PHASES:-(none)}"
echo "  Force all     : $FORCE_ALL"
echo "  Force phases  : ${FORCE_PHASES:-(none)}"
echo "  Dry-run       : $DRY_RUN"
echo "  Skip stages   : ${SKIP_STAGES:-(none)}"
echo "  Freeze eval   : $FREEZE_EVALUATOR"
echo "  Vectorized sw : $USE_VECTORIZED_SWEEP"
echo "  S3 bucket     : $S3_BUCKET"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS  (#883 — relaunch on confirmed mid-run reclaim)"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
# Preflight-only runs only backtest.py --mode=smoke (reads-only, zero
# writes, zero external API calls) and exits BEFORE the full-backtest
# path — it does not need config.yaml. The unconditional check here
# broke the Friday shell_run dry path on fresh spot instances that
# hadn't yet provisioned config.yaml (the check ran before the
# PREFLIGHT_ONLY short-circuit at ~line 1147). Guard the check so
# preflight-only is exempt.
if [ "$PREFLIGHT_ONLY" != "1" ] && [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi

# Locate the executor risk.yaml + predictor predictor.yaml on the dispatcher
# so we can stage them to S3 for the spot. Mirrors the legacy SCP-source
# resolution; only the transport changed.
#
# Experiment-package first (config#1042): risk.yaml resolves from
# alpha-engine-config/experiments/$ALPHA_ENGINE_EXPERIMENT_ID/executor/risk.yaml
# (default experiment `reference`) ahead of the legacy top-level
# alpha-engine-config/executor/risk.yaml, then the repo-local fallback —
# mirroring pipeline_common.load_config + preflight._check_executor_config.
# Behavior-preserving: config#1159 made the package copy byte-identical to legacy.
EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
EXECUTOR_CONFIG=""
for candidate in \
    "$HOME/alpha-engine-config/experiments/$EXPERIMENT_ID/executor/risk.yaml" \
    "$HOME/Development/alpha-engine-config/experiments/$EXPERIMENT_ID/executor/risk.yaml" \
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
    echo "  ~/alpha-engine-config/experiments/$EXPERIMENT_ID/executor/risk.yaml" >&2
    echo "  ~/Development/alpha-engine-config/experiments/$EXPERIMENT_ID/executor/risk.yaml" >&2
    echo "  ~/alpha-engine-config/executor/risk.yaml" >&2
    echo "  ~/Development/alpha-engine-config/executor/risk.yaml" >&2
    echo "  ~/alpha-engine/config/risk.yaml (legacy)" >&2
    echo "  ~/Development/alpha-engine/config/risk.yaml (legacy)" >&2
    echo "Backtester simulation cannot run without the executor config — silently" >&2
    echo "falling back to risk.yaml.example produces all-placeholder bucket names" >&2
    echo "and ArcticDB KeyNotFoundException deep in the executor-sim run." >&2
    exit 1
fi

PREDICTOR_CONFIG=""
for candidate in \
    "$HOME/alpha-engine-predictor/config/predictor.yaml" \
    "$HOME/Development/alpha-engine-predictor/config/predictor.yaml"; do
    if [ -f "$candidate" ]; then
        PREDICTOR_CONFIG="$candidate"
        break
    fi
done
# PREDICTOR_CONFIG may be empty — predictor backtest is skipped if so.

# ── Dispatcher-side pre-launch preflight (L4485) ─────────────────────────────
# Fail fast on the DISPATCHER, before provisioning a spot, per the standing
# rule "every preflight fails fast before expensive work"
# ([[feedback_preflight_fast_fail_before_expensive_work]]). The existing
# BacktesterPreflight + smoke harness run ON the spot — only AFTER ~10-15 min
# of boot + 3 clones + dep install — so a syntax error or a lib-pin drift
# burns that whole window before surfacing. These checks cost <2 s locally
# and catch the two cheapest-to-miss classes at second zero:
#   (1) py_compile — a SyntaxError anywhere in the load-bearing entrypoints
#       would crash the spot deep in the run. Byte-compile needs no deps.
#   (2) lib-pin drift — requirements.txt's alpha-engine-lib pin must be ≥ the
#       MIN_LIB_VERSION the in-process preflight asserts, or the spot's pip
#       install pulls a version the code rejects (the 2026-04-21 80-min burn).
# Plus a SOFT warning when local tracked .py/.sh edits aren't on origin/$BRANCH
# (the spot clones --branch $BRANCH from GitHub — local-only commits won't run).
pre_launch_preflight() {
    local py
    py="$LIB_PYTHON"
    [ -x "$py" ] || py="$(command -v python3 || echo python3)"

    # (1) Syntax-check the load-bearing entrypoints via ast.parse — a PURE
    #     parse with ZERO filesystem writes. py_compile writes .pyc into
    #     __pycache__, which fails with EACCES on this shared dispatcher
    #     where the cache dir is owned by another uid (root, from a prior
    #     SF run) — a FALSE failure that wrongly blocked a clean launch
    #     (caught live 2026-06-02). ast.parse raises SyntaxError on the same
    #     bug class without touching disk, so it can never false-fail on a
    #     read-only / mixed-ownership tree.
    if ! "$py" -c 'import ast,sys; [ast.parse(open(f).read(), filename=f) for f in sys.argv[1:]]' \
        "$REPO_ROOT/backtest.py" \
        "$REPO_ROOT/evaluate.py" \
        "$REPO_ROOT/preflight.py" \
        "$REPO_ROOT/pipeline_common.py" \
        "$REPO_ROOT/synthetic/predictor_backtest.py" 2>/tmp/prelaunch_syntax.err; then
        echo "ERROR: pre-launch syntax check FAILED — a SyntaxError would crash the spot ~15 min into boot+deps. Fix before launching:" >&2
        cat /tmp/prelaunch_syntax.err >&2
        exit 1
    fi

    # (2) Cross-check the requirements.txt lib pin against preflight.py's floor.
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
        echo "  pre-launch: lib pin v$pin ≥ MIN_LIB_VERSION $floor ✓"
    else
        echo "  pre-launch: WARNING — could not parse lib pin (pin='$pin' floor='$floor'); skipping pin cross-check" >&2
    fi

    # (3) SOFT: warn on local tracked .py/.sh edits not on origin/$BRANCH.
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

    # (4) SOFT: config#2871 — config.yaml is commonly a symlink into the
    # alpha-engine-config repo's tracked backtester/config.yaml (operator
    # flags like pit_parity_sweep live there). If the symlink resolves
    # outside a git-tracked path, or the resolved file has uncommitted
    # drift, an operator hand-edit would silently vanish on the next
    # symlink/box rebuild with no diff and no audit trail.
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
echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
pre_launch_preflight

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
# Capacity-resilient launch via krepis.ec2_spot (lib v0.26.0+ as
# alpha_engine_lib.ec2_spot / nousergon_lib.ec2_spot; invoked directly via
# krepis per config#1649 — the nousergon_lib re-export shim is guard-less
# under `python -m` on lib >=0.81.0 and silently no-ops).
# Rotates (instance_type × subnet) on InsufficientInstanceCapacity etc.
# Direct fix for the 2026-05-22 incident: THIS LAUNCHER's Evaluator
# invocation failed with InsufficientInstanceCapacity for c5.large in
# us-east-1f.
spot_assert_instance_types_allowed "$INSTANCE_TYPES" || exit 2
echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."

INSTANCE_ID=$("$LIB_PYTHON" -m krepis.ec2_spot launch \
    --types "$INSTANCE_TYPES" \
    --subnets "$SUBNETS" \
    --image-id "$AMI_ID" \
    --key-name "$KEY_NAME" \
    --security-group "$SECURITY_GROUP" \
    --iam-profile "$IAM_PROFILE" \
    --name "alpha-engine-backtest-$(date +%Y%m%d)" \
    --region "$AWS_REGION")
ec2_spot_rc=$?
if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$INSTANCE_ID" ]; then
    if [ "$ec2_spot_rc" -eq 64 ]; then
        echo "ERROR: capacity exhausted across all instance_type × subnet combinations" >&2
    fi
    if [ "$ec2_spot_rc" -eq 0 ]; then
      # rc=0 with an EMPTY instance id = the launch layer produced nothing
      # (e.g. the guard-less `-m nousergon_lib.ec2_spot` shim no-op,
      # config#1646 — closed at this launcher's transport by the krepis
      # migration, config#1649). `${ec2_spot_rc:-1}` defaults only when UNSET — a
      # captured 0 passed through and the SF recorded a silent success
      # on 2026-07-03. An empty id must always fail loud.
      echo "ERROR: ec2_spot launch exited 0 without an instance id — failing loud (config#1646)" >&2
      ec2_spot_rc=1
    fi
    exit "$ec2_spot_rc"
fi

echo "  Instance ID: $INSTANCE_ID"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
S3_STAGING_PREFIX="tmp/spot_backtest/${RUN_ID}"
S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"

# Last SSM dispatch description — captured for the EXIT-trap diagnostic
# so a non-zero exit prints which run_ssm call ran last. L2246
# (originally LAST_RUN_REMOTE_CMD pre-2026-05-27 SSH→SSM migration).
LAST_SSM_DESC=""

# Cleanup function — always terminate the instance + clean S3 staging,
# with diagnostics on failure.
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
            # Capture State.Name + StateReason.Code + StateTransitionReason
            # BEFORE terminating so the L4485 rc=-1 / empty-output failure
            # class (SSM Failed with no stdout/stderr) is classifiable
            # post-hoc and so the diagnostics below are populated (the
            # instance is gone once terminated). This describe call is for
            # STDOUT DIAGNOSTICS ONLY — the actual reclaim classify→decide
            # DECISION below is delegated to the lib chokepoint, which runs
            # its own describe-instances internally.
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
        # Independent-channel surveillance: fan out via ops_alerts
        # (SNS + flow-doctor forum topics; config#1749 T3). Best-effort:
        # ``|| echo ...`` keeps cleanup running even if Python / lib / SNS /
        # flow-doctor are unreachable — stdout diagnostic above is primary.
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
    source='alpha-engine-backtester/spot_backtest.sh',
)
" "$_alert_msg" "$_alert_sev") \
            2>&1 >/dev/null)" \
            || echo "    (ops alert fan-out failed via $_alert_python: ${_fo_err##*$'\n'}; primary stdout diagnostic above is the surface)"
    fi
    echo "==> Terminating spot instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
    # config-I7442 — NOT `aws s3 rm "$S3_STAGING" --recursive`. This prefix
    # holds SSM's own upload of the FULL remote stdout/stderr (run_ssm points
    # OutputS3KeyPrefix at ${S3_STAGING_PREFIX}/ssm-output); the recursive
    # delete that used to sit here destroyed the only un-truncated copy of a
    # failure's evidence as part of handling that failure. See the same
    # replacement in _spot_common.sh, which this retained-rollback monolith
    # mirrors. A launcher that keeps an unguarded delete is the defect
    # regardless of whether it is the live path today.
    if ! "$LIB_PYTHON" -m krepis.spot_evidence teardown \
            --staging "$S3_STAGING" \
            --slug "backtest" \
            --exit-code "$exit_code"; then
        echo "  spot_evidence: chokepoint unavailable via $LIB_PYTHON — S3 staging RETAINED at $S3_STAGING/ (not deleted)" >&2
    fi
    # #883 — on a classified reclaim, relaunch a FRESH spot with the SAME
    # argv, threading the incremented SPOT_ATTEMPT via the env. `trap - EXIT`
    # first so the exec'd process installs its own trap cleanly; exec
    # replaces this process, so the relaunch is bounded by
    # SPOT_ATTEMPT<MAX_SPOT_ATTEMPTS (re-checked above) and the pending
    # `exit` below is moot. Any non-reclaim failure falls through to the
    # status-preserving exit. The dead worker + its S3 staging are already
    # cleaned above, so re-exec is clean.
    if [ "$_will_relaunch" = "1" ]; then
        echo "==> Spot RECLAIMED by AWS (reason_code='$reason_code' state='$state' transition='$state_reason') — relaunching on a fresh spot (attempt $((SPOT_ATTEMPT + 1))/$MAX_SPOT_ATTEMPTS)"
        trap - EXIT
        SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${_ORIG_ARGS[@]+"${_ORIG_ARGS[@]}"}
    fi
    # CRITICAL (L4485): re-exit with the captured status. A bash EXIT trap
    # that ends on a successful command (the echo above, or the `|| true`
    # cleanup steps) otherwise leaves the script exiting 0 — which is
    # exactly how a Failed SSM `backtest` step (run_ssm correctly returned
    # 1) was masked as rc=0 to the orchestration wrapper on 2026-06-01,
    # letting a failed run read as success. The cleanup path must never
    # override the primary exit status (inverse of the acceptable EXIT-trap
    # carve-out in [[feedback_no_silent_fails]]).
    exit "$exit_code"
}
trap cleanup EXIT

# Wait for instance to be running
echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# ── Stage config files to S3 ─────────────────────────────────────────────────
# Replaces the pre-2026-05-27 SCP path. The spot pulls each file via its
# existing alpha-engine-executor-profile IAM role's s3:GetObject grant.
# .env is no longer staged (#890): its non-secret config moved to config.yaml.
echo "==> Staging configs to ${S3_STAGING}/"

# config.yaml: gitignored backtester runtime config.
aws s3 cp "$REPO_ROOT/config.yaml" "${S3_STAGING}/config.yaml" --region "$AWS_REGION" --quiet
echo "  staged config.yaml"

# Executor risk.yaml: prod path is alpha-engine-config/executor/risk.yaml
# (private config repo pulled daily on ae-dashboard by boot-pull). Legacy
# alpha-engine/config/ path is kept for local dev fallback but has not
# been populated on ae-dashboard since the config-repo split (2026-04-07).
# Hit 2026-04-20: spot silently fell back to risk.yaml.example, executor
# read placeholder signals_bucket="your-research-bucket-name", ArcticDB
# KeyNotFound on a nonexistent bucket. The pre-launch resolver above
# already confirms existence — fail-loud at staging if the file went
# missing in the gap.
aws s3 cp "$EXECUTOR_CONFIG" "${S3_STAGING}/risk.yaml" --region "$AWS_REGION" --quiet
echo "  staged risk.yaml from $EXECUTOR_CONFIG"

# Predictor predictor.yaml: optional — predictor backtest is skipped if
# absent. The pre-launch resolver above sets PREDICTOR_CONFIG="" when
# unfound; encode that state into S3 via a sentinel file so the spot
# bootstrap knows to skip the download.
if [ -n "$PREDICTOR_CONFIG" ]; then
    aws s3 cp "$PREDICTOR_CONFIG" "${S3_STAGING}/predictor.yaml" --region "$AWS_REGION" --quiet
    echo "  staged predictor.yaml from $PREDICTOR_CONFIG"
    STAGED_PREDICTOR_CONFIG=1
else
    echo "  WARNING: predictor.yaml not found — predictor backtest will be skipped"
    STAGED_PREDICTOR_CONFIG=0
fi

# ── Wait for the SSM agent to register ───────────────────────────────────────
# Replaces the old SSH-readiness poll. AL2023 ships the SSM agent; with the
# instance profile's AmazonSSMManagedInstanceCore (in alpha-engine-executor-
# profile) it registers within ~1 min.
echo "==> Waiting for SSM agent to come Online..."
for i in $(seq 1 36); do  # 36 × 5s = 180s budget
    ping=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text --region "$AWS_REGION" 2>/dev/null || true)
    if [ "$ping" = "Online" ]; then
        echo "  SSM agent Online."
        break
    fi
    if [ "$i" -eq 36 ]; then
        echo "ERROR: SSM agent not Online after 180s (instance $INSTANCE_ID)"
        exit 1
    fi
    sleep 5
done

# ── SSM dispatch primitive (lib chokepoint) ──────────────────────────────────
# run_ssm "<description>" [timeout_seconds] <<HEREDOC ... HEREDOC
#
# Thin wrapper around `python -m krepis.ssm_dispatcher run`
# (lib v0.35.0+ as nousergon_lib.ssm_dispatcher; invoked directly via
# krepis per config#1649 — the nousergon_lib re-export shim is guard-less
# under `python -m` on lib >=0.81.0 and silently no-ops). Body read from
# stdin via --script-stdin so the dispatcher's bash parser does not scan
# it for quote/paren balance.
# Records the description in LAST_SSM_DESC so the EXIT trap can name
# which call ran last on failure. Mirrors ae-data PR 2 (#330).
#
# L394 cascade: --diagnostics-bucket + --diagnostics-prefix activate the
# lib v0.39.0 chokepoint that writes a JSON failure record (status +
# command_id + 4KB stdout/stderr tails + instance_id) to
# s3://${S3_BUCKET}/_spot_diagnostics/ae-backtester/{YYYY-MM-DD}.json on
# terminal non-Success. Best-effort write inside the lib — S3 failure
# swallowed; inner SSM exit always preserved. No-op on Success.
run_ssm() {
    local description="$1" timeout_s="${2:-3600}"
    LAST_SSM_DESC="$description"
    "$LIB_PYTHON" -m krepis.ssm_dispatcher run \
        --instance-id "$INSTANCE_ID" \
        --description "backtester: $description" \
        --timeout "$timeout_s" \
        --output-bucket "$S3_BUCKET" \
        --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
        --region "$AWS_REGION" \
        --diagnostics-bucket "$S3_BUCKET" \
        --diagnostics-prefix "_spot_diagnostics/ae-backtester" \
        --script-stdin
}

# ── Bootstrap spot: watchdog + hard timeout + python + clones + configs ─────
# Cutover to `krepis.spot_bootstrap` (alpha-engine-config-I7372). This is the
# monolith's copy of the same bootstrap `_spot_common.sh` carried; both now
# call the SAME renderer with the same argv, which is the consolidation that
# matters. Merging the two FILES is a separate refactor and deliberately not
# attempted here: `spot_backtest.sh` is linear top-level code with its own
# variable names and its own cleanup trap, and cannot source the per-stage
# helper without restructuring the launcher inside a cutover PR.
#
# What the previous heredoc did and did NOT do:
#   DID  — arm a hard runtime cap via `systemd-run --on-active`, so the spot
#          self-terminates after MAX_RUNTIME_SECONDS even when this dispatcher
#          is cancelled, stopped or SIGKILLed and its `trap cleanup EXIT` never
#          runs (hit 3 times in April 2026). Preserved as
#          --max-runtime-seconds, and now FATAL when it cannot be armed.
#   DID NOT — install any SSM-liveness watchdog. An instance whose SSM agent
#          died was unreachable by this dispatcher AND uncapped by anything
#          except that one timer. The renderer emits the `ec2-spot-watchdog`
#          unit unconditionally; this repo GAINS it here, having never had it.
#
# The three HTTPS clones are `--extra-clone`s: the predictor replay runs the
# predictor's modules in-process and the executor supplies risk.yaml's schema.
# Checkout dirs deliberately stay `alpha-engine-*` while the repos are
# `crucible-*` (2026-06-15 org rename); every downstream path depends on that
# split, so the paths are reproduced EXACTLY. URLs are the new slugs as
# launcher-side literals rather than a rename/transfer 301 redirect.
echo "==> Bootstrapping spot (SSM watchdog, hard timeout, python, 3 clones, configs)..."
BOOTSTRAP_SCRIPT="$("$LIB_PYTHON" -m krepis.spot_bootstrap render \
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
# Here-STRING, not a pipe: `run_ssm` records LAST_SSM_DESC for the EXIT trap's
# "last run_ssm" diagnostic, and a pipeline would run it in a subshell where
# that assignment dies with the subshell.
run_ssm "bootstrap" 600 <<<"$BOOTSTRAP_SCRIPT"

# ── Install python dependencies ──────────────────────────────────────────────
echo "==> Installing Python dependencies..."
run_ssm "deps" 1200 <<DEPS
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=${AWS_REGION} AWS_DEFAULT_REGION=${AWS_REGION}
cd /home/ec2-user/alpha-engine-backtester

# No .env source (#890): the deps step only needs pip, which resolves the
# alpha-engine-lib git+https URL in requirements.txt without auth (public repo).
# Non-secret runtime config (EMAIL_*, OUTPUT_BUCKET) is read from config.yaml
# by the python pipeline and by the per-stage BUCKET resolution below.
# Strict interpreter resolution — NEVER a silent fallback to the AMI python3
# (alpha-engine-config-I7372). requirements.txt is resolved against 3.12;
# python3 resolves different wheels, and the divergence surfaces as an
# ImportError deep inside the workload rather than here, at deps time, with the
# log still in hand. The bootstrap already asserted python3.12 exists; if it is
# absent HERE, something changed between two SSM steps — a hard fail, not a
# substitution.
command -v python3.12 >/dev/null || {
    echo "FATAL: python3.12 not found on the spot — the bootstrap step asserted it. Refusing to install requirements.txt against a different interpreter (different wheel resolution)." >&2
    exit 1
}
PIP="python3.12 -m pip"

\$PIP install --upgrade pip -q
\$PIP install -q -r requirements.txt

# The predictor checkout is CODE-ONLY (config#3031, 2026-07-20): its
# requirements.txt is deliberately NOT installed here. Co-installing two
# repos' requirements files into one resolver namespace let predictor's
# numpy>=2.5.1 floor (installed second) silently override the backtester's
# numpy cap (numba/vectorbt hard ceiling) — the 2026-07-20 weekly deps
# failures — and the same class had already bitten via nousergon-lib
# (L4513) and pyarrow. Every library the in-process predictor replay
# (synthetic/predictor_backtest.py, research-free backfill) needs at
# runtime is declared in the backtester's OWN requirements.txt with
# bounds compatible with that cap. The import guards below prove the predictor
# code chain resolves against this single environment.
cd /home/ec2-user/alpha-engine-predictor
if [ ! -d "/home/ec2-user/alpha-engine-predictor/model" ]; then
    echo "FATAL: predictor checkout missing (code-only sys.path dependency)" >&2
    exit 1
fi

# Fail-loud dependency GUARD (L4513 class fix). Assert the nousergon-lib
# modules the Evaluator imports are actually present AFTER all installs — so if a
# future sibling-repo pin drift ever downgrades the lib below quant.stats, this
# breaks LOUD at deps time instead of silently at evaluate.py's import weeks
# later. Per feedback_no_silent_fails. PYBIN derives from PIP ("py -m pip" -> py).
# The lib was renamed alpha-engine-lib -> nousergon-lib (alpha_engine_lib is now a
# deprecated import alias); requirements.txt installs the nousergon-lib
# distribution, so the guard MUST verify via the real module + distribution name
# -- "pip show alpha-engine-lib" returns nothing and exits 1 under pipefail.
# (NB: no backticks in this heredoc comment -- they would command-substitute.)
cd /home/ec2-user/alpha-engine-backtester
PYBIN="\${PIP% -m pip}"
\$PYBIN -c "import nousergon_lib.quant.stats.multiple_testing, nousergon_lib.quant" || {
    echo "FATAL: nousergon-lib is missing quant.stats — a co-installed sibling repo's pin likely downgraded it below v0.49.0. Resolved version:" >&2
    \$PIP show nousergon-lib | grep -E '^Version:' >&2 || true
    exit 1
}
\$PIP show nousergon-lib | grep -E '^Version:'

# Also verify predictor modules used by backtest.py
\$PYBIN -c "from synthetic.predictor_backtest import run; from synthetic.production_signal_backtest import build_production_signal_inputs" || {
    echo "FATAL: predictor modules missing or failed to import" >&2
    exit 1
}

# Fail-loud numpy-2 consistency guard (config#2815 migration completion).
# A stale  \$PIP install 'numpy<2'  used to sit here — added 2026-03-24 (commit
# 0534004) when pyarrow wheels were still numpy-1 built. The config#2815
# numpy-2 migration (#536, 2026-07-17) lifted requirements.txt to numpy>=2
# across backtester + predictor (cvxpy/scipy/vectorbt all now require numpy>=2),
# but left this caller-side downgrade behind. On the first weekly spot run after
# the migration it force-downgraded numpy 2.5.1 -> 1.26.4 AFTER the requirements
# install, leaving the numpy-2-built scipy/cvxpy referencing np.long (removed in
# numpy 1.24-1.26) -> the backtester runtime_smoke's GBMScorer.load crashed at
# 'import scipy.sparse' with "module 'numpy' has no attribute 'long'". The
# downgrade is REMOVED (complete the migration; never re-extend a deprecated
# shim). This guard asserts the exact import chains that broke AFTER all
# installs, so any future co-installed pin that downgrades numpy breaks LOUD
# here at deps time (seconds) instead of ~40 min into the run. Two chains:
#   numpy>=2 + scipy.sparse + lightgbm — the 2026-07-18 runtime_smoke crash
#     (np.long removed) after the stale downgrade left numpy at 1.26; and
#   numba + vectorbt — the same weekend's simulate-phase crash (config-I3279):
#     numpy 2.5.1 resolved above numba 0.66's numpy<2.5 ceiling, so
#     'import vectorbt' raised "Numba needs NumPy 2.4 or less" ~18h into the
#     run and portfolio_stats.json/optimizer_gate degraded to an error stub.
#     The pip-check gate below catches that instance at the METADATA level
#     (numba declares its ceiling); this import smoke also catches ABI-level
#     numba/numpy breaks that ship with self-consistent metadata. vectorbt
#     backs vectorbt_bridge.py -> portfolio_stats + the optimizer-gate arc,
#     so it is load-bearing for the Saturday SF's promotion artifacts.
# Per feedback_no_silent_fails.
# (NB: no backticks in this heredoc body -- they would command-substitute.)
\$PYBIN -c "import numpy, scipy.sparse, lightgbm, numba, vectorbt; assert int(numpy.__version__.split('.')[0]) >= 2, 'numpy '+numpy.__version__+' < 2.0 is inconsistent with the numpy-2-built scipy/cvxpy stack (config#2815)'; print('numpy-2 guard OK: numpy='+numpy.__version__+' scipy='+scipy.__version__+' lightgbm='+lightgbm.__version__+' numba='+numba.__version__+' vectorbt='+vectorbt.__version__)" || {
    echo "FATAL: import-chain consistency check failed — a co-installed pin, stale downgrade, or numba/numpy ABI mismatch broke the scipy/lightgbm or numba/vectorbt import chain (config#2815, config-I3279). See traceback above." >&2
    exit 1
}

# Fail-loud pip-check dependency-consistency gate (config#2973). The import
# guard above only covers the TWO chains (numpy + scipy.sparse + lightgbm;
# numba + vectorbt) that have actually broken runs so far. \`pip install\` reports ANY
# OTHER co-install/transitive-dependency conflict as a post-hoc "does not
# take into account all installed packages" warning and still exits 0, so an
# internally-inconsistent env ships silently and only surfaces as an import
# crash deep into the run (this is the same silent-inconsistency CLASS the
# 2026-07-19 numpy/scipy incident belonged to, not a one-off). This gate
# turns the whole class into a deps-time (seconds) failure instead of a
# per-import-chain guard that must be hand-extended for every new breakage.
#
# Allowlist: newline-separated exact substrings of \`pip check\` conflict
# lines known to be import-safe on the backtester path -- each entry needs a
# comment above it justifying why the conflicting package is never imported
# here. Currently EMPTY: the one conflict instance seen in production so far
# (numba 0.66.0's numpy<2.5 ceiling, exposed by the config#2815 numpy-2
# migration) was fixed at the resolution layer above (the numpy<2.5 pin,
# config#2975/PR541) rather than allowlisted here -- numba backs vectorbt,
# which IS imported on this path, so silencing that conflict would be unsafe.
# (NB: no backticks in this heredoc body -- they would command-substitute.)
PIP_CHECK_ALLOWLIST=""
# Gate on pip check's EXIT CODE, not on output emptiness: a clean env prints
# "No broken requirements found." (non-empty!) and exits 0 -- the original
# output-emptiness logic here failed the deps step on the FIRST EVER clean
# environment (2026-07-20, right after config#3031's co-install removal made
# the env resolvable), because until then every run had a real conflict and
# the success path had never executed. exit 0 = clean, full stop; only a
# non-zero exit applies the allowlist filter to the conflict lines.
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

# ── Predictor sector_map cache fetch ─────────────────────────────────────────
# Only sector_map.json is consumed (predictor_backtest.load_sector_map).
# The former price_cache_slim sync was Wave-4 dead staging —
# predictor_backtest loads prices+features from ArcticDB
# (load_universe_from_arctic), never the local cache parquets; verified
# no data/cache/*.parquet reader exists. Removed in Wave-4 PR4.
echo "==> Downloading predictor sector_map from S3..."
run_ssm "predictor-cache" 300 <<'CACHE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp
CACHE_DIR="/home/ec2-user/alpha-engine-predictor/data/cache"
mkdir -p "$CACHE_DIR"
# Wave-3 reader migration (ROADMAP L1401): try new
# reference/price_cache/sector_map.json first, fall back to legacy
# predictor/price_cache/ during the write-both soak. Wave-4: former
# `aws s3 sync price_cache_slim/` removed — dead staging
# (predictor_backtest loads from ArcticDB, never reads
# data/cache/*.parquet).
aws s3 cp s3://alpha-engine-research/reference/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null \
    || aws s3 cp s3://alpha-engine-research/predictor/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null \
    || true
echo "Predictor cache dir: sector_map.json $([ -f "$CACHE_DIR/sector_map.json" ] && echo present || echo MISSING)"
CACHE

# ── Build env export command ─────────────────────────────────────────────────
# PYTHONUNBUFFERED=1: line-buffering stdout/stderr so SSM ships log lines as
# they're emitted. Without this, stdout is block-buffered when the agent
# captures it to CloudWatch — the 2026-04-22 4th Saturday SF dry-run lost
# ~16 minutes of in-flight output when the SSM agent died mid-run and
# buffered lines never reached the log. Combined with the phase markers
# in pipeline_common.phase (which explicit-flush after each START/END),
# this closes the "silent 110-minute phase" blind spot. Paired with
# `python -u` on each backtest.py invocation below as belt-and-suspenders.
#
# ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true is exported in ENV_SOURCE so it
# overrides whatever the dispatcher's env passes through (#890 removed the .env
# source that previously preceded it; the suppress export is unconditional now).
# The executor's decision_capture short-circuits at is_decision_capture_enabled()
# when this flag is truthy. Sim hot loop (param_sweep × N_dates × N_positions)
# would otherwise emit ~50k-200k per-decision S3 PUTs and blow the
# simulation_pipeline 2700s watchdog (observed 2026-05-13 spot run
# adhoc-skipto-backtester-20260513-2333). Capture artifacts exist for
# production observability — they have no semantic meaning in the sweep.
# Paired with alpha-engine #177.
# AWS_REGION/AWS_DEFAULT_REGION: #890 removed the sourced .env entirely, so
# the region env vars boto3 + lib preflight require must be exported here.
# Same #247 regression class as alpha-engine-data's spot scripts; this script
# was in a sibling repo the original arc didn't touch. System is single-region
# us-east-1 (matches this file's own ${AWS_REGION:-us-east-1} defaults).
# Origin: 2026-05-16 Saturday SF PredictorTraining failure (spot_train.sh
# sibling) — audited forward to prevent the identical Backtester/Parity/
# Evaluator failure. No .env is sourced; OUTPUT_BUCKET is now read from the
# staged config.yaml at each per-stage BUCKET resolution below.
# PYTHON_BIN is resolved STRICTLY — `python3.12` or a non-zero exit. This one
# line is injected into EVERY downstream stage heredoc (smoke, backtest, parity,
# pit-parity, evaluate, preflight), so the old
# `command -v python3.12 && PYTHON_BIN=python3.12 || PYTHON_BIN=python3`
# was not one silent fallback but the fallback for the entire run: fixing only
# the bootstrap would have left every workload step free to resolve the AMI
# python3 against wheels installed for 3.12 (alpha-engine-config-I7372).
ENV_SOURCE='export XDG_CACHE_HOME=/tmp; export PYTHONUNBUFFERED=1; export ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true; export AWS_REGION=us-east-1; export AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1; command -v python3.12 >/dev/null || { echo "FATAL: python3.12 absent on the spot — bootstrap asserted it and deps installed against it. Refusing to run this step on a different interpreter (alpha-engine-config-I7372)." >&2; exit 1; }; PYTHON_BIN=python3.12; export PYTHON_BIN;'

# Spot-side python is resolved inline per SSM step via PYTHON_BIN in the
# ENV_SOURCE above. The pre-2026-05-27 SSH transport captured this on the
# dispatcher with `REMOTE_PYTHON=$(run_remote "command -v ...")` — under
# SSM there's no native way to capture inner-script stdout into a
# dispatcher variable, so the resolution is repeated inside each heredoc
# (cheap — just a $PATH probe). REMOTE_PYTHON kept as a name alias for
# the per-heredoc reference, set to the env-var form $PYTHON_BIN that
# ENV_SOURCE binds at runtime on the spot.
REMOTE_PYTHON='$PYTHON_BIN'

# ── Preflight-only (Friday shell_run dry path) ──────────────────────────────
# ROADMAP "Friday shell-run — per-module dry-path activation" owed-item #3.
# Placed AFTER the real boot/clone/deps/config-upload (so the bootstrap
# path — lib-pin resolution, sys.path, predictor cache sync, image deps —
# is genuinely exercised) and STRICTLY BEFORE both the --smoke-only block
# (per-phase smoke modes + the evaluate.py S3-probe diagnostics) and the
# full-backtest heredoc.
#
# Runs ONLY `backtest.py --mode=smoke` — the EXISTING bootstrap-class
# smoke harness from PRs #43-#48 (BacktesterPreflight: lib-version /
# imports / predictor-weights presence / executor-config validation, then
# _runtime_smoke: universe-symbols + per-ticker ArcticDB read + recent
# signals.json load + Layer-1A GBM load/predict — all S3 *reads*, ~30-60s).
# We REUSE backtest.py's existing --mode=smoke (no new harness): per
# backtest.py:4180-4184 it runs preflight + _runtime_smoke then `return`s
# BEFORE _init_pipeline / the simulation / the optimizer, so it itself
# performs zero config writes and makes no external API (yfinance/
# Anthropic) data fetch.
#
# Hard invariant proof (what is statically unreachable under this flag):
#   * The per-phase smoke loop (smoke-simulate / smoke-param-sweep /
#     smoke-predictor-backtest / smoke-phase4 / smoke-predictor-param-sweep)
#     and the `evaluate.py --mode diagnostics` S3-probe block live INSIDE
#     the `if [ "$RUN_MODE" = "smoke-only" ]` body below — the `exit 0`
#     here never reaches it.
#   * The full-backtest heredoc (backtest stage / pit_parity / parity /
#     evaluator) and its config/{executor,scoring,predictor,research,
#     scanner}_params*.json optimizer auto-apply (evaluate.py --upload,
#     non-frozen) live further below — also unreachable.
#   * No CloudWatch heartbeat, no parity_report.json / parity_metrics.csv
#     upload, no reporter S3 upload — all of those are past this exit.
# Net: smoke harness (read-only) runs, then exit 0. Zero external API
# calls, zero S3/config writes. The `trap cleanup EXIT` still fires and
# terminates the spot instance.
if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PREFLIGHT-ONLY (Friday shell_run dry path)"
    echo "  boot + deps done; running bootstrap-class smoke harness only,"
    echo "  then exit 0 — NO sweep / sim / parity / evaluator / auto-apply,"
    echo "  ZERO external API calls, ZERO S3/config writes."
    echo "═══════════════════════════════════════════════════════════════"
    run_ssm "preflight-only" 900 <<PREFLIGHT
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

# backtest.py --mode=smoke = BacktesterPreflight + _runtime_smoke, then
# returns 0 BEFORE _init_pipeline / simulation / optimizer (see
# backtest.py:4180). No --upload, no full mode, no config write.
echo "==> Preflight: backtest.py --mode=smoke"
$REMOTE_PYTHON -u backtest.py --mode=smoke --log-level INFO 2>&1
PREFLIGHT

    echo ""
    echo "==> Preflight-only PASSED — bootstrap-class smoke clean."
    echo "==> Instance will be terminated (no sweep/sim/parity/evaluator,"
    echo "    no config/*.json auto-apply, no S3/config writes performed)."
    exit 0
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
if [ "$RUN_MODE" = "smoke-only" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SMOKE TEST"
    echo "═══════════════════════════════════════════════════════════════"

    # backtest.py --mode=smoke runs BacktesterPreflight + runtime smoke
    # (end-to-end with minimal data: universe symbols, per-ticker Arctic
    # read, recent signals.json load, Layer-1A GBM load + predict) and
    # exits 0. Keeps the smoke path in lockstep with what full modes do
    # at startup — no drift between bash-driven smoke and in-process
    # pipeline validation. Evaluate-mode smoke follows: artifact-read
    # path + BacktesterPreflight(mode="evaluate").
    run_ssm "smoke" 3600 <<SMOKE
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

# OUTPUT_BUCKET formerly came from the sourced .env (#890 removed it). Read it
# from the staged config.yaml instead, with the same default fallback so
# \`\`set -u\`\` never trips on a missing key. config.yaml was fetched into cwd
# by BOOTSTRAP; \$PYTHON_BIN is set by ENV_SOURCE.
BUCKET="\$(\$PYTHON_BIN -c 'import yaml,sys; print((yaml.safe_load(open(\"config.yaml\")) or {}).get(\"output_bucket\") or \"alpha-engine-research\")' 2>/dev/null || echo alpha-engine-research)"

# Per-mode smoke summary — collected throughout the run and printed as
# a single table at the end. Each entry: "name|status|duration|budget|usage".
# Populated regardless of pass/fail so partial runs still show which
# modes completed before the failure.
declare -a _SMOKE_SUMMARY=()

_smoke_record() {
    # args: name, status ("ok" | "FAIL"), duration_s, budget_s (may be ""), usage_pct (may be "")
    _SMOKE_SUMMARY+=("\$1|\$2|\$3|\$4|\$5")
}

_smoke_extract_budget() {
    # Pull "N.Ns <= N.Ns (N% of budget)" from a log file's last
    # budget-check line. Emits "budget_s<TAB>usage_pct" or empty.
    local log_file="\$1"
    local line
    line="\$(grep -oE 'budget check: [0-9.]+s <= [0-9.]+s \([0-9]+% of budget\)' "\$log_file" | tail -1 || true)"
    [ -z "\$line" ] && return
    local budget usage
    budget="\$(echo "\$line" | grep -oE '<= [0-9.]+s' | grep -oE '[0-9.]+s')"
    usage="\$(echo "\$line" | grep -oE '\([0-9]+%' | tr -d '(%')%"
    printf '%s\t%s' "\$budget" "\$usage"
}

_smoke_run_mode() {
    # Run one backtest.py --mode=X, tee output, record to summary.
    # Returns non-zero on Python failure so caller can decide to break.
    local mode="\$1"
    local log_file="/tmp/smoke_\${mode//\//_}.log"
    local start=\$SECONDS
    local status="ok"

    echo ""
    echo "==> Smoke: backtest.py --mode=\$mode $SMOKE_PHASE_FLAGS"
    if ! $REMOTE_PYTHON -u backtest.py --mode=\$mode --log-level INFO $SMOKE_PHASE_FLAGS 2>&1 | tee "\$log_file"; then
        status="FAIL"
    fi
    local dur=\$((SECONDS - start))

    local budget="" usage=""
    local extracted
    extracted="\$(_smoke_extract_budget "\$log_file")"
    if [ -n "\$extracted" ]; then
        budget="\${extracted%%\$'\t'*}"
        usage="\${extracted##*\$'\t'}"
    fi

    _smoke_record "\$mode" "\$status" "\${dur}s" "\$budget" "\$usage"
    [ "\$status" = "ok" ]
}

_smoke_run_evaluator() {
    # config#3121: evaluate.py --smoke — the evaluator's own cheap
    # preflight (BacktesterPreflight(mode="evaluate") + a read-only S3
    # reachability probe), analogous to _smoke_run_mode's backtest.py
    # --mode=smoke but for the separate evaluate.py entrypoint (evaluator
    # previously had only the input-check-only input_quality_gate — no
    # execution smoke of its own imports/config/S3-wiring at all).
    local log_file="/tmp/smoke_evaluator.log"
    local start=\$SECONDS
    local status="ok"

    echo ""
    echo "==> Smoke: evaluate.py --smoke"
    if ! $REMOTE_PYTHON -u evaluate.py --smoke --log-level INFO 2>&1 | tee "\$log_file"; then
        status="FAIL"
    fi
    local dur=\$((SECONDS - start))
    _smoke_record "smoke-evaluator" "\$status" "\${dur}s" "" ""
    [ "\$status" = "ok" ]
}

_smoke_print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SMOKE SUMMARY"
    echo "═══════════════════════════════════════════════════════════════"
    printf "  %-28s %-8s %-10s %-10s %-8s\n" "Mode" "Status" "Duration" "Budget" "Usage"
    printf "  %s\n" "─────────────────────────────────────────────────────────────────────"
    local any_fail=0
    for entry in "\${_SMOKE_SUMMARY[@]}"; do
        IFS='|' read -r name status dur budget usage <<< "\$entry"
        printf "  %-28s %-8s %-10s %-10s %-8s\n" "\$name" "\$status" "\$dur" "\${budget:-–}" "\${usage:-–}"
        [ "\$status" = "FAIL" ] && any_fail=1
    done
    echo "═══════════════════════════════════════════════════════════════"
    if [ "\$any_fail" = "1" ]; then
        echo "  RESULT: FAIL (one or more modes did not pass)"
    else
        echo "  RESULT: PASS (all \${#_SMOKE_SUMMARY[@]} modes ok)"
    fi
    echo "═══════════════════════════════════════════════════════════════"
}

# Always print summary, even if a mode aborts mid-run.
trap '_smoke_print_summary' EXIT

# backtest.py --mode=smoke: preflight + runtime smoke (universe symbols,
# per-ticker Arctic read, recent signals.json load, Layer-1A GBM load +
# predict). Keeps smoke in lockstep with what full modes do at startup.
if ! _smoke_run_mode smoke; then
    echo "ERROR: smoke preflight FAILED — aborting"
    exit 1
fi

# Per-phase smoke harness — exercise each pipeline phase-family with a
# tiny fixture (few dates, tiny param grid, short GBM lookback) and
# enforce per-mode wall-clock budgets from timing_budget.yaml. Ordered
# fastest → slowest so a failure in an earlier mode short-circuits
# the harder ones. ROADMAP Backtester P0 #3.
#
# Timestamp capture for the L280 SUPPRESS contract canary below — the
# smoke-param-sweep mode is the hot-loop site that would emit ~50k-200k
# decision_artifacts/ S3 PUTs if SUPPRESS were broken. Captured BEFORE
# the loop so the canary's LastModified filter covers every smoke mode
# that touches executor code, not just smoke-param-sweep.
SMOKE_SWEEP_START_ISO=\$(date -u +%Y-%m-%dT%H:%M:%S)
for SMOKE_PHASE_MODE in smoke-simulate smoke-param-sweep smoke-predictor-backtest smoke-phase4 smoke-predictor-param-sweep; do
    if ! _smoke_run_mode "\$SMOKE_PHASE_MODE"; then
        echo "ERROR: smoke phase \$SMOKE_PHASE_MODE FAILED — aborting smoke-only run"
        exit 1
    fi
done

# config#3121: smoke-pit-parity — a tiny-slice (5 tickers, 30-60 trading
# days) invocation of BOTH pit_parity passes (lookahead + walkforward,
# including the opt-in CSCV sweep path when config.yaml's pit_parity_sweep
# flag is on) that proves imports + config + subprocess wiring in well
# under a minute of real compute (numba/numpy import errors surface
# immediately, not after the ~250-450s the full tiny-slice pass takes to
# complete). Gated on PIT_PARITY_ENABLED — same condition the REAL
# pit_parity stage below uses — so a --no-pit-parity smoke-only run
# doesn't pay for smoke-testing a stage this run won't execute for real.
# Runs in the SAME pre-full-run smoke-only position as the loop above
# (this script IS what runs in the standalone Parity SF state per L4486 —
# there is no separate "Parity stage" script to wire this into).
if [ "\${PIT_PARITY_ENABLED:-0}" = "1" ]; then
    if ! _smoke_run_mode smoke-pit-parity; then
        echo "ERROR: smoke phase smoke-pit-parity FAILED — aborting smoke-only run"
        exit 1
    fi
else
    echo ""
    echo "==> [smoke-pit-parity] SKIPPED (PIT_PARITY_ENABLED!=1 — pit_parity won't run for real either)"
fi

# config#3121: smoke-parity — import/collection-only proof (see the
# stage=parity block below for why this stage doesn't get a separate
# tiny-slice DATA pass: its real pass is already a small 10-day-window
# integration test).
echo ""
echo "==> Smoke: pytest tests/test_parity_replay.py -m parity --collect-only"
_SMOKE_PARITY_START=\$SECONDS
_SMOKE_PARITY_STATUS="ok"
if ! $REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity --collect-only -q 2>&1; then
    _SMOKE_PARITY_STATUS="FAIL"
fi
_SMOKE_PARITY_DUR=\$((SECONDS - _SMOKE_PARITY_START))
_smoke_record "smoke-parity" "\$_SMOKE_PARITY_STATUS" "\${_SMOKE_PARITY_DUR}s" "" ""
if [ "\$_SMOKE_PARITY_STATUS" = "FAIL" ]; then
    echo "ERROR: smoke phase smoke-parity FAILED — aborting smoke-only run"
    exit 1
fi

# config#3121: smoke-evaluator — every _KNOWN_STAGES entry must declare an
# execution smoke (tiny-slice preflight run before its full pass); the
# evaluator previously had only input_quality_gate, which checks SIGNAL
# INPUTS, not that evaluate.py's own imports/config/S3-wiring actually
# work. Runs in the same pre-full-run smoke-only position as the other
# phase smokes above.
if ! _smoke_run_evaluator; then
    echo "ERROR: smoke phase smoke-evaluator FAILED — aborting smoke-only run"
    exit 1
fi

# ── L280 SUPPRESS contract canary ─────────────────────────────────────────
# Asserts ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true (exported in
# ENV_SOURCE above) actually prevented decision_artifacts/ S3 PUTs
# during the smoke window. Catches operational-side regressions the
# in-process CI test (tests/test_param_sweep_decision_capture_suppress.py)
# cannot: ENV_SOURCE drift on the spot AMI / .env override resetting
# the flag / IAM-role substitution that lib gating doesn't see.
# Composes with the CI test which catches code-review-time regressions
# (env-var rename, gate-semantics flip, new bypassing capture site).
echo ""
echo "==> [suppress-canary] checking decision_artifacts/ writes during smoke window..."
SUPPRESS_PREFIX="decision_artifacts/\$(date -u +%Y/%m/%d)/"
SUPPRESS_HITS=\$(aws s3api list-objects-v2 \\
    --bucket "\${BUCKET}" \\
    --prefix "\${SUPPRESS_PREFIX}" \\
    --query "Contents[?LastModified >= '\${SMOKE_SWEEP_START_ISO}'] | length(@)" \\
    --output text 2>/dev/null || echo 0)
SUPPRESS_HITS=\${SUPPRESS_HITS:-0}
[ "\${SUPPRESS_HITS}" = "None" ] && SUPPRESS_HITS=0
if [ "\${SUPPRESS_HITS}" != "0" ]; then
    echo "ERROR [suppress-canary] \${SUPPRESS_HITS} \${SUPPRESS_PREFIX} keys appeared since \${SMOKE_SWEEP_START_ISO} — ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS contract REGRESSED. Investigate ENV_SOURCE export / .env override / lib gating semantics / new bypassing capture site. ROADMAP L280." >&2
    _smoke_record "suppress-canary" "FAIL" "0s" "" ""
    exit 1
fi
echo "[suppress-canary] OK — zero \${SUPPRESS_PREFIX} writes since \${SMOKE_SWEEP_START_ISO} (SUPPRESS contract holding)"
_smoke_record "suppress-canary" "ok" "0s" "" ""

echo ""
echo "==> Resolving most recent backtest artifact date from s3://\${BUCKET}/backtest/..."
# Pick the most-recent date that ALSO has portfolio_stats.json on S3.
# The plain "sort | tail -1" approach picked stale empty prefixes
# created by prior half-complete runs (observed 2026-04-24 smoke: a
# 2026-04-24/ prefix existed but had no artifacts, causing evaluate.py
# to hard-fail with "All critical simulation artifacts missing").
# Excluding hidden prefixes (.smoke/, .dry-run/) keeps the probe
# pointing at production dates.
LATEST_DATE=""
while IFS= read -r candidate; do
    [ -z "\$candidate" ] && continue
    case "\$candidate" in .*) continue ;; esac
    if aws s3api head-object --bucket "\${BUCKET}" --key "backtest/\$candidate/portfolio_stats.json" >/dev/null 2>&1; then
        LATEST_DATE="\$candidate"
        break
    fi
done < <(aws s3 ls "s3://\${BUCKET}/backtest/" | awk '/PRE / {print \$2}' | tr -d '/' | sort -r)
if [ -z "\$LATEST_DATE" ]; then
    echo "ERROR: no backtest/{date}/ prefix with portfolio_stats.json found in s3://\${BUCKET}/backtest/"
    _smoke_record "evaluate-diagnostics" "FAIL" "0s" "" ""
    exit 1
fi
echo "Using backtest date: \$LATEST_DATE"

echo ""
echo "==> Smoke: evaluate.py --mode diagnostics --freeze --date \$LATEST_DATE"
_EVAL_START=\$SECONDS
_EVAL_STATUS="ok"
if ! $REMOTE_PYTHON -u evaluate.py --mode diagnostics --freeze --date "\$LATEST_DATE" --log-level INFO 2>&1 | tail -30; then
    _EVAL_STATUS="FAIL"
fi
_EVAL_DUR=\$((SECONDS - _EVAL_START))
_smoke_record "evaluate-diagnostics" "\$_EVAL_STATUS" "\${_EVAL_DUR}s" "" ""

echo ""
echo "Smoke test complete."
# Summary prints via trap on exit
SMOKE

    echo "==> Smoke-only mode — instance will be terminated."
    exit 0
fi

# ── Full backtest ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL BACKTEST (--mode $BACKTEST_MODE)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "backtest" "$MAX_RUNTIME_SECONDS" <<BACKTEST
# MUST precede set -euo pipefail — any code path (including sourced files
# from ${ENV_SOURCE}) that references this variable before its main init at
# line ~1423 will trigger an unbound-variable fatal exit under set -u.
export _BACKTEST_WAS_SKIPPED=false
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

# BUCKET used across all three stages. OUTPUT_BUCKET formerly came from the
# sourced .env (#890 removed it); read it from the staged config.yaml instead,
# falling back to the default so \`\`set -u\`\` doesn't blow up on a missing key.
# Matches the smoke-only heredoc's line. config.yaml is in cwd (fetched by
# BOOTSTRAP); \$PYTHON_BIN is set by ENV_SOURCE.
BUCKET="\$(\$PYTHON_BIN -c 'import yaml,sys; print((yaml.safe_load(open(\"config.yaml\")) or {}).get(\"output_bucket\") or \"alpha-engine-research\")' 2>/dev/null || echo alpha-engine-research)"
# SKIP_STAGES baked in from the dispatcher's --skip-stages flag. Stages in
# this CSV are skipped with a loud ⊘ echo; everything else runs.
SKIP_STAGES="${SKIP_STAGES}"
# PIT_PARITY_ENABLED baked in from the dispatcher (default 1 / ON since
# 2026-05-17). Same mechanism as SKIP_STAGES — the dispatcher-side value is
# interpolated at heredoc-generation time so the runtime gate below resolves
# it on the spot instance.
PIT_PARITY_ENABLED="${PIT_PARITY_ENABLED}"
# RUN_DATE baked in from the dispatcher (resolved once from the injected
# env / --run-date / wall-clock fallback above) so backtest + param-sweep
# + parity + pit_parity + evaluator uploads all land under the SAME
# backtest/{date}/ prefix — even across the 3 separate spot stages of one
# Saturday SF run that may straddle UTC midnight. Same gen-time
# interpolation as SKIP_STAGES / PIT_PARITY_ENABLED above; the prior
# per-spot \$(date -u) recompute was the 2026-05-17 Evaluator date-split.
RUN_DATE="${RUN_DATE}"
# EVAL_HALF baked in from the dispatcher's --eval-half flag (config-I3112).
# Same mechanism as RUN_DATE — the dispatcher-side value is interpolated at
# heredoc-generation time so the runtime evaluator-stage gate below resolves
# it on the spot instance.
EVAL_HALF="${EVAL_HALF}"

_stage_skipped() {
    case ",\${SKIP_STAGES}," in
        *",\$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Track whether backtest was skipped so the evaluator invocation below knows
# to pass --skip-backtester and distinguish intentional absence from unexpected
# failure (config#2887).
_BACKTEST_WAS_SKIPPED=false

# ── Stage: backtest ─────────────────────────────────────────────────────────
# If backtest.py fails we exit non-zero so parity + evaluator never run
# against stale or missing artifacts — the evaluator would otherwise
# auto-promote garbage params to S3. Fail loud so the spot run is marked
# failed, the heartbeat metric is not emitted, and the Step Function catches
# it. Replaces the previous || { echo WARNING } swallow that silently let
# evaluator run against invalid sweep results and was the root cause of
# multiple undetected param oscillations.
if _stage_skipped backtest; then
    _BACKTEST_WAS_SKIPPED=true
    echo "⊘ stage=backtest SKIPPED (--skip-stages=\${SKIP_STAGES})"
else
    echo "▶ stage=backtest START at \$(date -u +%H:%M:%S)"
    # --date "\${RUN_DATE}" pins backtest.py's artifact prefix to the
    # SF-stamped run date. Without it backtest.py defaults --date to its
    # own date.today() on this spot instance, so the backtest stage wrote
    # backtest/2026-05-17/ while the later Evaluator spot looked under
    # backtest/2026-05-18/ — the 2026-05-17 date-split. Parity / pit_parity
    # thread \${RUN_DATE}; the evaluator invocation did NOT (this comment
    # previously claimed it did — the drift stayed invisible until the
    # 2026-07-20 weekday recovery rerun, config#3133) and now threads
    # --date explicitly in its own stage block below, so ALL stages key
    # off the single SF-declared date.
    _BT_RC=0
    $REMOTE_PYTHON -u backtest.py --mode $BACKTEST_MODE --date "\${RUN_DATE}" --upload --log-level INFO $BACKTEST_SKIP_PHASE4_FLAG $BACKTEST_PHASE_FLAGS 2>&1 || _BT_RC=\$?
    if [ "\$_BT_RC" -ne 0 ]; then
        # config-I7258: preserve the REAL exit code — a bare \`exit 1\` here
        # launders rc=137/OOM into a generic 1 before krepis.ssm_dispatcher
        # (>=0.60.0) can classify it via SSM's ResponseCode.
        case "\$_BT_RC" in
            137|-9) echo "ERROR: backtest.py SIGKILLed (rc=\$_BT_RC) — likely OOM on \$(hostname). Spot run marked FAILED — check" >&2 ;;
            124|-14|143) echo "ERROR: backtest.py timed out (rc=\$_BT_RC) on \$(hostname). Spot run marked FAILED — check" >&2 ;;
            *) echo "ERROR: backtest.py failed (rc=\$_BT_RC). Spot run marked FAILED — check" >&2 ;;
        esac
        echo "       flow-doctor alerts. Parity + evaluator stages skipped" >&2
        echo "       to prevent auto-promotion of unvalidated configs." >&2
        exit "\$_BT_RC"
    fi
    echo "▶ stage=backtest END at \$(date -u +%H:%M:%S)"
fi

# ── Stage: pit_parity (observational, DEFAULT ON, NON-BLOCKING) ─────────────
# Proof-of-impact for point-in-time discipline (ROADMAP L2371 / plan §D4):
# runs the predictor backtest both ways (legacy single-pass vs
# --walk-forward) and emits the skilled-risk-basket contamination report to
# s3://{bucket}/backtest/{RUN_DATE}/pit_parity.json. This is the input to
# the manual, Brian-gated --walk-forward default flip (plan §5).
#
# DEFAULT ON 2026-05-17 (Brian: "switch pit to on"). Runs an extra
# predictor-sim pass (~+1 predictor backtest; bounded, the predictor
# pipeline is ~4 min / ~8% of the 1800s cap). NEVER fails the spot run
# (|| true) — observational only, writes no configs, and does NOT change
# --walk-forward (the optimizer-feeding default stays OFF; flipping it is
# the separate post-review L2371 step). Opt out per run: --no-pit-parity
# or PIT_PARITY_ENABLED=0. Runtime fallback stays :-0 (belt-and-suspenders
# if the dispatcher bake is ever bypassed).
# L4486 (2026-06-05): gate is on the dedicated pit_parity stage token, NOT
# the backtest token, so pit_parity runs in the standalone Parity SF state
# (which passes --skip-stages=backtest,evaluator: backtest skipped but
# pit_parity NOT skipped) in a FRESH process with full RAM headroom -- instead
# of stacked inside PredictorBacktest after the main predictor pipeline already
# held ~3.5 GB. The SF turns it OFF in PredictorBacktest via --no-pit-parity so
# it still fires EXACTLY ONCE.
# NOTE: this comment lives inside the unquoted heredoc, so it must contain NO
# backticks or dollar-paren -- they would be command-substituted at heredoc
# construction (the 2026-06-05 "pit_parity: command not found" noise).
if [ "\${PIT_PARITY_ENABLED:-0}" = "1" ] && ! _stage_skipped pit_parity; then
    # config#3121: smoke-pit-parity runs HERE — immediately before the
    # real pit_parity pass, in the SAME script invocation (the standalone
    # Parity SF state per L4486) that actually executes it — not just in
    # the separate --smoke-only preflight path. A tiny-slice (5 tickers,
    # 30-60 trading days) run of BOTH parity passes proves imports +
    # config + subprocess wiring in well under a minute; an ImportError /
    # config break here fails fast instead of ~2h into the real run below.
    # Non-fatal by the same observational posture as the real stage: a
    # smoke failure is loud (ERROR + non-zero) but does not exit the spot
    # run — pit_parity itself is non-blocking, and its smoke inherits that.
    echo "▶ stage=smoke-pit-parity START at \$(date -u +%H:%M:%S)"
    if ! $REMOTE_PYTHON -u backtest.py --mode smoke-pit-parity \\
        --date "\${RUN_DATE}" --log-level INFO 2>&1; then
        echo "WARNING: smoke-pit-parity FAILED — pit_parity's import/config/subprocess wiring is broken. Continuing (pit_parity is non-blocking) but the real pass below will likely fail the same way." >&2
    fi
    echo "▶ stage=smoke-pit-parity END at \$(date -u +%H:%M:%S)"

    echo "▶ stage=pit_parity START at \$(date -u +%H:%M:%S) (observational, non-blocking)"
    # Swallow on non-zero exit per feedback_no_silent_fails secondary-
    # observability carve-out: (a) failure mode swallowed = pit_parity
    # exception path (backtester continues either way); (b) primary
    # deliverable survives = weights archive + sweep + evaluator pipeline
    # are independent of pit_parity; (c) concrete recording surfaces =
    # (1) S3 artifact at backtest/{date}/pit_parity.json with status=failed
    # always emitted by backtest.py::main pit_parity branch (since
    # 2026-05-27); (2) Telegram + SNS alert via alpha_engine_lib.alerts
    # (sev=warning, dedup-keyed on run_date). 2026-05-17→2026-05-24
    # incident: this swallow ate 4 RecursionError silently before the
    # contract was added.
    # config#6032: the PredictorBacktest phase (earlier in this SF, same
    # RUN_DATE) already ran the SAME walk-forward (PIT) inference over the
    # same config and wrote backtest/{RUN_DATE}/predictor_stats.json — bake
    # that key in so backtest.py --pit-parity can reuse it for the walk-
    # forward pass instead of re-running the full predictor pipeline in a
    # subprocess (~25 min saved). Best-effort: a missing/unreadable artifact
    # falls back to the subprocess inside backtest.py (never fails the
    # observational stage).
    $REMOTE_PYTHON -u backtest.py --mode predictor-backtest --pit-parity \\
        --predictor-stats-key "backtest/\${RUN_DATE}/predictor_stats.json" \\
        --date "\${RUN_DATE}" --log-level INFO 2>&1 \\
        || echo "WARNING: pit_parity stage failed (observational — spot run continues; failure-artifact + Telegram alert published by the inner Python)"
    echo "▶ stage=pit_parity END at \$(date -u +%H:%M:%S)"
else
    echo "⊘ stage=pit_parity SKIPPED (PIT_PARITY_ENABLED!=1 or --skip-stages contains pit_parity — runs ONCE in the standalone Parity state per L4486)"
fi

# ── Stage: parity ───────────────────────────────────────────────────────────
# Parity is OBSERVABILITY, not a gate. Each Saturday SF run produces:
#   * parity_report.json — per-run drill-down (count + ticker-set + field
#     divergence breakdowns), uploaded to s3://{bucket}/backtest/{date}/
#   * parity_metrics.csv — append one row per run with capture_rate,
#     ticker_jaccard_avg, count_divergence_rms, field_diff_rate,
#     n_lifecycle_skipped. Time series at
#     s3://{bucket}/backtest/parity_metrics.csv. The metric trend is the
#     load-bearing signal; step-changes trigger investigation.
# The pytest assertion was removed (test always passes — its job is to
# generate the artifacts). The spot run does NOT fail the SF on parity
# divergence: 0% historical parity is structurally unreachable for a
# system with weekly auto-tuned configs and evolving executor code.
# See tests/test_parity_replay.py module docstring for the full rationale.
# Setup-level failures (missing trades.db, ArcticDB unreachable) are still
# fatal here — those are real infrastructure breakage, not "expected drift".
if _stage_skipped parity; then
    echo "⊘ stage=parity SKIPPED (--skip-stages=\${SKIP_STAGES})"
else
    # config#3121: smoke-parity — the parity stage's own "full pass"
    # (tests/test_parity_replay.py -m parity) already runs in seconds
    # against a small (default 10-day) window; there is no separate
    # multi-hour full run to precede with a SEPARATE tiny-slice data pass
    # the way smoke-pit-parity precedes pit_parity's real run (re-running
    # the same S3 download + test twice would just double the cost, not
    # add coverage). The smoke this stage actually needs — and lacked —
    # is an IMPORT/WIRING proof: pytest --collect-only proves the test
    # module imports cleanly and the parity marker resolves, in <5s,
    # before paying for the trades.db download below. Catches a broken
    # import (numba/numpy-class ImportError, a bad conftest fixture, etc.)
    # at collection time instead of surfacing as an opaque pytest error
    # after the S3 download.
    echo "▶ stage=smoke-parity START at \$(date -u +%H:%M:%S)"
    if ! $REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity --collect-only -q 2>&1; then
        echo "WARNING: smoke-parity FAILED — test_parity_replay.py failed to import/collect. Continuing (parity is non-blocking) but the real pass below will likely fail the same way." >&2
    fi
    echo "▶ stage=smoke-parity END at \$(date -u +%H:%M:%S)"

    echo "▶ stage=parity START at \$(date -u +%H:%M:%S)"
    PARITY_TRADES_DB="/tmp/trades_latest.db"
    PARITY_REPORT_DIR="/tmp/parity_report"
    mkdir -p "\$PARITY_REPORT_DIR"

    if ! aws s3 cp "s3://\${BUCKET}/trades/trades_latest.db" "\$PARITY_TRADES_DB" --quiet; then
        echo "ERROR: could not download trades_latest.db from S3 — parity cannot run" >&2
        echo "       This is infrastructure breakage (not divergence) — failing spot." >&2
        exit 1
    fi

    PARITY_EXIT=0
    # USE_REAL_ARCTICDB=1 tells tests/conftest.py to skip the default
    # MagicMock stub so the integration test hits real ArcticDB.
    # PARITY_RUN_DATE pins the time-series CSV's run_date column to
    # today's RUN_DATE so re-runs of a single Saturday cohort overwrite
    # idempotently rather than producing duplicate rows.
    TRADES_DB_PATH="\$PARITY_TRADES_DB" \\
    SIGNALS_BUCKET="\${BUCKET}" \\
    PARITY_REPORT_DIR="\$PARITY_REPORT_DIR" \\
    PARITY_RUN_DATE="\${RUN_DATE}" \\
    USE_REAL_ARCTICDB=1 \\
    $REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity -v 2>&1 || PARITY_EXIT=\$?

    # Upload the per-run report. The time-series CSV is appended by the
    # test itself (see append_parity_metrics_row) — best-effort, errors
    # WARN-not-FAIL since the per-run report is the authoritative artifact.
    if [ -f "\$PARITY_REPORT_DIR/parity_report.json" ]; then
        aws s3 cp "\$PARITY_REPORT_DIR/parity_report.json" \\
            "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
            && echo "Uploaded parity_report.json to s3://\${BUCKET}/backtest/\${RUN_DATE}/" \\
            || echo "WARNING: failed to upload parity_report.json (non-fatal)"
        else
        # alpha-engine-config#7199 — THE ABSENCE IS THE DEFECT. Until now a missing
        # parity_report.json was indistinguishable from a skipped upload: the guard
        # had no else-branch, the non-zero pytest exit degraded to a WARNING on
        # stderr, and the stage returned 0. That is how parity_report.json went
        # unwritten from 2026-07-24 to 2026-08-13 while the sibling report.md in the
        # same prefix kept updating weekly and nothing noticed. (Root cause: PR #550
        # stopped co-installing the predictor requirements, which is where pytest
        # came from; "No module named pytest", exit in 1 second, SF success path.)
        #
        # An absent artifact now becomes a PRESENT artifact that says it is absent,
        # so every downstream reader renders FAILED rather than a silently stale
        # ABSENT row. sf-pipeline-policy 2.3a rule 2: a missing verdict is never a
        # pass.
        echo "ERROR: the parity replay produced NO parity_report.json (producer exit=\$PARITY_EXIT)." >&2
        echo "       The backtester-to-executor fill-parity claim is UNPROVEN for \${RUN_DATE}." >&2
        printf '{"schema":"parity_report-0.0.0","run_date":"%s","status":"failed","verdict":"FAIL","verdict_reason":"the parity replay produced no report this run (producer exit=%s) - the backtester-to-executor fill-parity claim is unproven","producer_exit":%s}\\n' \\
            "\${RUN_DATE}" "\$PARITY_EXIT" "\$PARITY_EXIT" > /tmp/parity_report_missing.json
        aws s3 cp /tmp/parity_report_missing.json \\
            "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
            && echo "       Recorded the absence at s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" \\
            || echo "       WARNING: could not record the absence marker either." >&2
        # Deliberately NOT exiting non-zero here: this launcher still runs co-tenant
        # stages behind one script, and killing stages that do not consume the
        # parity artifact would trade one blindness for an outage
        # (sf-pipeline-policy 2.1). The recorded absence above is the surface; the
        # split ParityReplay branch (spot_parity_replay.sh) is where this fails the
        # stage, because there it costs only that branch.
fi

    # Pytest exit codes:
    #   0 = test ran (always-pass, since divergence is observability not gate)
    #   non-zero with parity_report.json present = setup-level error inside the
    #     test body (e.g. ArcticDB read failure on integration path); flag a
    #     WARNING but don't fail the spot — operator can still inspect the
    #     report. The SF alarm should fire on real infrastructure breakage
    #     (the s3 cp failure above), not on observability test signaling.
    if [ "\$PARITY_EXIT" != "0" ]; then
        echo "WARNING: parity pytest exited \$PARITY_EXIT (likely setup-side error)." >&2
        echo "         See s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json (if present)." >&2
        echo "         Continuing spot run — parity is observability, not a gate." >&2
    fi
    echo "▶ stage=parity END at \$(date -u +%H:%M:%S)"
fi

# ── Stage: evaluator ────────────────────────────────────────────────────────
# Runs evaluate.py against today's backtest artifacts in S3. Consolidated
# into the spot step 2026-04-24 — the SF's dedicated Evaluator states
# (CheckSkipEvaluator, CheckEvaluatorFreeze, Evaluator, EvaluatorFrozen,
# WaitForEvaluator, CheckEvaluatorStatus, EvaluatorWait, ExtractEvaluatorError)
# were retired. --freeze-evaluator controls config-promotion (freeze =
# diagnostic-only, no config writes). Default is live-apply for the Sat SF;
# manual iteration runs should pass --freeze-evaluator.
if _stage_skipped evaluator; then
    echo "⊘ stage=evaluator SKIPPED (--skip-stages=\${SKIP_STAGES})"
else
    # config#3121: smoke-evaluator runs HERE — immediately before the real
    # evaluator pass, in the same script invocation that executes it. Cheap
    # preflight (imports + config load + S3 reachability) via
    # evaluate.py --smoke; the evaluator previously had only
    # input_quality_gate (a signal-INPUT check, not an execution smoke of
    # evaluate.py's own imports/config/S3-wiring). Non-fatal here — same
    # rationale as smoke-pit-parity above: loud WARNING, spot run continues,
    # the real pass below will surface the same break if it's real.
    echo "▶ stage=smoke-evaluator START at \$(date -u +%H:%M:%S)"
    if ! $REMOTE_PYTHON -u evaluate.py --smoke --log-level INFO 2>&1; then
        echo "WARNING: smoke-evaluator FAILED — evaluate.py's import/config/S3 wiring is broken. Continuing but the real pass below will likely fail the same way." >&2
    fi
    echo "▶ stage=smoke-evaluator END at \$(date -u +%H:%M:%S)"

    echo "▶ stage=evaluator START at \$(date -u +%H:%M:%S) freeze=${FREEZE_EVALUATOR}"
    _EVAL_FREEZE=""
    if [ "${FREEZE_EVALUATOR}" = "true" ]; then
        _EVAL_FREEZE="--freeze"
    fi
    _EVAL_SKIP_BT=""
    if [ "\$_BACKTEST_WAS_SKIPPED" = "true" ]; then
        _EVAL_SKIP_BT="--skip-backtester"
    fi
    # --date "\${RUN_DATE}" pins evaluate.py to the SF-stamped run date. The
    # backtest stage's comment above claimed the evaluator "already threads"
    # RUN_DATE — it never did; evaluate.py silently defaulted to its own
    # date.today(). Invisible while the evaluator ran on weekend days (the
    # trading-day normalization landed on the same Friday by coincidence);
    # a WEEKDAY recovery rerun (watch-rerun-2026-07-18-12, 2026-07-20)
    # resolved today() to Monday, looked in backtest/2026-07-20/, and
    # correctly hard-failed on missing artifacts (config#3133).
    # --mode "\${EVAL_HALF}" (config-I3112): "all" keeps the bundled behavior
    # byte-identical for manual runs; the SF's split states pass
    # --eval-half=diagnostics / --eval-half=optimize, which map 1:1 to
    # evaluate.py's modes (the optimize half reads the S3 diagnostics
    # snapshot the diagnostics half wrote).
    _EVAL_RC=0
    $REMOTE_PYTHON -u evaluate.py --mode "\${EVAL_HALF}" --upload --date "\${RUN_DATE}" \$_EVAL_FREEZE \$_EVAL_SKIP_BT --log-level INFO 2>&1 || _EVAL_RC=\$?
    if [ "\$_EVAL_RC" -ne 0 ]; then
        # config-I7258: preserve the REAL exit code — see spot_evaluator.sh.
        case "\$_EVAL_RC" in
            137|-9) echo "ERROR: evaluate.py SIGKILLed (rc=\$_EVAL_RC) — likely OOM on \$(hostname). Spot run marked FAILED." >&2 ;;
            124|-14|143) echo "ERROR: evaluate.py timed out (rc=\$_EVAL_RC) on \$(hostname). Spot run marked FAILED." >&2 ;;
            *) echo "ERROR: evaluate.py failed (rc=\$_EVAL_RC). Spot run marked FAILED." >&2 ;;
        esac
        exit "\$_EVAL_RC"
    fi
    echo "▶ stage=evaluator END at \$(date -u +%H:%M:%S)"
fi

echo ""
echo "All requested stages complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Backtest complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# Per-stage CloudWatch heartbeats. Each stage gets its own heartbeat so the
# Saturday SF can split stages across separate SF states without conflating
# their alarms. Stages listed in --skip-stages are excluded from emission.
#
# ── Why the backtester gate is `backtest` alone (config-I5786) ───────────────
# This block used to require that BOTH backtest and parity ran, on the
# reasoning that "parity is observability for backtest output — they form one
# semantic unit." That was true while they were ONE SF state. On 2026-05-16
# (nousergon-data#250, the preflight task split) they became TWO, and every
# caller since passes a --skip-stages set that excludes one or the other:
#
#   Backtester state              --skip-stages=parity,evaluator
#   Parity state                  --skip-stages=backtest,evaluator
#   Evaluator state               --skip-stages=backtest,parity
#   PredictorBacktest state       --skip-stages=parity,evaluator
#   PortfolioOptimizerBacktest    --skip-stages=parity,evaluator
#
# So the old conjunction became UNSATISFIABLE BY EVERY CALLER. The last
# `Process=backtester` heartbeat in CloudWatch is 2026-05-13 — the final
# Saturday run before that split — and `alpha-engine-backtester-no-heartbeat`
# has been correctly in ALARM ever since, reporting a signal no code path
# could emit. Nothing was wrong with the backtester.
#
# The gate is now `backtest` alone: the heartbeat answers "did the backtest
# stage complete", which is what its alarm is named for. Parity is its own SF
# state now and needs its own heartbeat and alarm — tracked separately, NOT
# folded in here, because a heartbeat with no alarm is a metric with no
# subscriber (observability-policy.md §5).
#
# Enforced by tests/test_spot_backtest_heartbeat_emission.py, which executes
# this block's real gating logic against every --skip-stages combination the
# SF actually passes.
_emit_heartbeat() {
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

_stage_in_skip() {
    case ",${SKIP_STAGES}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

if ! _stage_in_skip backtest; then
    _emit_heartbeat backtester
fi
if ! _stage_in_skip evaluator; then
    _emit_heartbeat evaluator
fi
