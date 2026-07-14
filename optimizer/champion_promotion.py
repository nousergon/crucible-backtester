"""champion_promotion.py — gated weekly champion promotion/demotion engine
(config#2364 champion-promotion epic, child 3 / config#2367).

Writes the live pointer ``config/producer_champion.json`` that the
alpha-engine executor's ``executor/champion.py::load_champion_pointer``
reads at planner start to decide whether entry candidates come from the
``agentic`` research pipeline or the ``scanner_predictor_direct``
research-free-predictor arm (config#2366). Statistical correctness matters
here in a way it does not for the advisory optimizers elsewhere in this
module: a wrong pointer move silently changes which strategy is LIVE.

**Gates (all must pass to move the pointer — bidirectional, same rule
promotes and demotes):**

  (a) ``challenger_matured_cohorts >= min_matured_cohorts`` — the challenger
      arm (whichever of the two VALID_CHAMPIONS is NOT the current
      champion) must have at least this many matured FORWARD weekly
      cohorts behind its lift estimate.
  (b) Overlap-aware significance on sector-neutral 21d lift. 21d realized
      alpha measured on a WEEKLY evaluation cadence means each week's
      cohort overlaps ~3 prior weeks' holding windows (21 trading days /
      ~7 calendar days per weekly cycle ≈ 3x overlap) — naive i.i.d. pooling
      across weekly observations understates the true standard error and
      manufactures significance the data does not have (the exact
      pseudo-replication failure mode already documented and fixed for
      pooled-vs-date-clustered IC in ``analysis/end_to_end.py``, config#1405
      comment block). This gate uses the Newey-West (1994) HAC
      (heteroskedasticity- and autocorrelation-consistent) standard error
      of the mean weekly SN-lift, Bartlett kernel, lag = round(horizon_days
      / cadence_days) = round(21/7) = 3 — matching the "~3x overlapping
      windows" the issue names directly, rather than the estimator's
      generic automatic lag rule (which is tuned for long daily P&L series,
      not short weekly panels). See ``_hac_significance`` below. This
      reuses ``nousergon_lib.quant.stats.intervals.newey_west_se`` (already
      vendored + unit-tested in the shared lib; LV2-AE leverage arc,
      2026-06-03) rather than a hand-rolled implementation.
      config#1524's low-n estimator was evaluated as the issue's suggested
      alternative but is not yet shipped anywhere in this codebase (no
      ``newey``/``HAC``/``low_n_estimator`` module existed prior to this
      change) — this PR does not block on it landing; per the issue this is
      an either/or choice and HAC/Newey-West is a standard, well-documented
      technique for exactly this overlapping-window problem.
  (c) Hysteresis: the challenger must clear gate (b) on the SAME side
      (challenger beats champion) on 2 CONSECUTIVE weekly evaluations
      before the pointer moves. A single strong week is not enough — this
      is the ``consecutive_wins`` carry-forward counter, reset to 0 the
      moment the sign flips or the significance gate fails.
  (d) Bidirectional: the identical rule set (a)-(c) applies whether the
      challenger is beating the champion (→ promote) or the champion is
      losing to the challenger from the other seat (→ demote) — there is
      only one gate path, run once per week from the champion's point of
      view.
  (e) Cooldown: at least ``cooldown_weeks`` must have elapsed since the
      last pointer move before another move is permitted, even if (a)-(c)
      all clear.
  ``--freeze`` (mirrors the posture of every other evaluate.py writer,
  e.g. weight_optimizer / veto_analysis / research_optimizer): when set,
  the pointer write is unconditionally suppressed even if every gate
  cleared — but the weekly audit record is STILL written (with
  ``blocked_by=["frozen"]`` when gates had otherwise cleared), because a
  held week must never be indistinguishable from a dead engine.

**Liveness (config#2054 lesson, binding):** ``config/producer_champion.json``
mtime alone cannot prove this engine is alive — a correctly-held week does
not touch it. ``emit_champion_audit`` below is called UNCONDITIONALLY at the
end of every weekly Evaluator run (mirroring ``optimizer/apply_audit.py``'s
"the write is the liveness proxy" posture) and writes
``config/apply_audit/producer_champion/{date}.json`` (+ ``latest.json``)
regardless of outcome, including ``error``.

**Single writer function, dual caller (no parallel writer implementations):**
``write_champion_pointer`` is the ONLY code path that may write
``config/producer_champion.json``. The gate engine calls it with
``promotion_source="gate_engine"``; a future one-shot operator bootstrap
script (e.g. the 2026-07-13 bootstrap, Brian ruling config#2364) MUST call
this same function with ``promotion_source="operator_bootstrap"`` — never a
hand-edited S3 object.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date, datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
POINTER_KEY = "config/producer_champion.json"
AUDIT_PREFIX = "config/apply_audit/producer_champion"

VALID_CHAMPIONS = ("agentic", "scanner_predictor_direct")

OUTCOMES = (
    "promoted", "demoted", "held_insufficient_data", "held_cooldown",
    "held_not_significant", "error",
)

# ── Guardrail constants (config#2367 — public-repo tier rule: guardrail
# constants live in config, not code literals). These are the CODE DEFAULTS,
# overridable via the ``champion_promotion`` config.yaml.example block below;
# the live values ship from the separate alpha-engine-config repo. ─────────
_MIN_MATURED_COHORTS = 4        # gate (a)
_SIGNIFICANCE_ALPHA = 0.05      # gate (b) — two-sided test size
_HORIZON_DAYS = 21              # 21d forward-alpha horizon (config#1405 basis)
_CADENCE_DAYS = 7               # weekly evaluation cadence
_HYSTERESIS_WEEKS = 2           # gate (c) — consecutive winning weeks required
_COOLDOWN_WEEKS = 2             # gate (e) — minimum weeks between pointer moves

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("champion_promotion", {})


def _hac_lag() -> int:
    """Bartlett-kernel lag = round(horizon_days / cadence_days).

    21d horizon on a 7-day (weekly) cadence ⇒ round(21/7) = 3 — each weekly
    SN-lift observation overlaps its ~2 immediate predecessors' holding
    windows, matching the issue's "~3x overlapping windows" framing exactly.
    Configurable so a future horizon/cadence change doesn't require a code
    edit.
    """
    horizon = int(_cfg.get("horizon_days", _HORIZON_DAYS))
    cadence = int(_cfg.get("cadence_days", _CADENCE_DAYS))
    return max(0, round(horizon / cadence))


# ── Statistical gate: HAC/Newey-West-adjusted overlap-aware significance ───


def hac_significance(
    weekly_sn_lift: list[float], *, alpha: float = _SIGNIFICANCE_ALPHA,
) -> dict:
    """Overlap-aware two-sided significance test of whether the challenger's
    weekly sector-neutral 21d lift series is significantly different from
    zero, using the Newey-West (1994) HAC standard error of the mean
    (Bartlett kernel; lag = ``_hac_lag()``) in place of the naive i.i.d.
    ``s/sqrt(n)`` standard error.

    **Why HAC and not a plain t-test:** each entry in ``weekly_sn_lift`` is
    one week's mean sector-neutral 21d-forward alpha. Because the 21-trading
    -day forward window is measured on names selected on a ~weekly cadence,
    consecutive weekly observations share the majority of their realized
    trading days — the series is NOT i.i.d., it's autocorrelated with an
    induced overlap of order (horizon_days / cadence_days) ≈ 3. Pooling
    weekly observations under the naive i.i.d. assumption understates the
    true standard error of the mean lift and manufactures statistical
    significance the panel does not actually have — this is the identical
    pseudo-replication failure mode already diagnosed and fixed for pooled
    -vs-date-clustered IC elsewhere in this codebase (see
    ``analysis/end_to_end.py``'s Grinold-Kahn date-clustered IC t-stat
    comment, config#1405). Newey & West (1987, "A Simple, Positive
    Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
    Covariance Matrix", Econometrica 55(3)) is the standard, well-documented
    correction: it inflates the long-run variance estimate by the
    Bartlett-kernel-weighted autocovariances up to the chosen lag, which
    both (i) reduces smoothly to the naive i.i.d. SE as the series'
    autocorrelation → 0, and (ii) strictly inflates the SE (never shrinks
    it) for a genuinely overlapping series — so this gate is provably more
    conservative than an unadjusted t-test, never more permissive.

    Uses the vendored, independently-unit-tested
    ``nousergon_lib.quant.stats.intervals.newey_west_se`` (LV2-AE leverage
    arc, 2026-06-03) rather than a hand-rolled HAC implementation.

    Returns a dict:
      ``{"status": "ok", "n": int, "mean": float, "se": float, "lags": int,
         "t_stat": float, "p_value": float, "significant": bool}``
      or ``{"status": "insufficient_data", "n": int}`` when fewer than 2
      finite observations are available.

    The Student-t critical value (n - 1 degrees of freedom, matching
    ``scipy.stats.ttest_1samp``'s convention used elsewhere in this
    codebase) is used rather than the asymptotic normal, since weekly
    cohort counts here are small (single/low-double-digit n) — the t
    distribution's fatter tails are the conservative choice at low n.
    """
    from nousergon_lib.quant.stats.intervals import newey_west_se

    clean = [float(x) for x in weekly_sn_lift if x is not None and not math.isnan(x)]
    n = len(clean)
    if n < 2:
        return {"status": "insufficient_data", "n": n}

    nw = newey_west_se(clean, max_lags=_hac_lag())
    if nw.get("status") != "ok":
        return {"status": "insufficient_data", "n": n}

    mean = nw["estimate"]
    se = nw["se"]
    if se <= 0:
        # Degenerate (all-identical) series — cannot form a t-stat; treat as
        # significant iff the constant mean itself is nonzero (no dispersion
        # to test against is not the same as "not significant").
        return {
            "status": "ok", "n": n, "mean": mean, "se": 0.0, "lags": nw["lags"],
            "t_stat": math.inf if mean != 0 else 0.0,
            "p_value": 0.0 if mean != 0 else 1.0,
            "significant": bool(mean != 0),
        }

    from scipy.stats import t as _student_t

    t_stat = mean / se
    dof = max(n - 1, 1)
    p_value = float(2.0 * _student_t.sf(abs(t_stat), dof))
    return {
        "status": "ok",
        "n": n,
        "mean": mean,
        "se": se,
        "lags": nw["lags"],
        "t_stat": float(t_stat),
        "p_value": p_value,
        "significant": bool(p_value < alpha),
    }


# ── Gate engine ─────────────────────────────────────────────────────────────


def _other_champion(champion: str) -> str:
    others = [c for c in VALID_CHAMPIONS if c != champion]
    if len(others) != 1:
        raise ValueError(
            f"_other_champion expects exactly 2 VALID_CHAMPIONS, got {VALID_CHAMPIONS!r}"
        )
    return others[0]


def evaluate_gates(
    *,
    champion_before: str,
    challenger_matured_cohorts: int,
    challenger_weekly_sn_lift: list[float],
    prior_consecutive_wins: int,
    cooldown_until: str | None,
    as_of: str,
    freeze: bool,
) -> dict:
    """Run gates (a)-(e) and decide this week's outcome. Pure function — no
    I/O — so the gate logic is independently unit-testable against synthetic
    leaderboard fixtures without any S3/mock plumbing.

    Args:
        champion_before: current pointer champion value.
        challenger_matured_cohorts: gate (a) input.
        challenger_weekly_sn_lift: challenger's per-week sector-neutral 21d
            lift-vs-champion series (most recent last), gate (b) input.
        prior_consecutive_wins: gate (c) carry-forward counter from the
            prior audit record (0 if none).
        cooldown_until: gate (e) carry-forward — trading-day string or None.
        as_of: this run's trading day (for cooldown comparison + the new
            cooldown_until when a move happens).
        freeze: --freeze flag.

    Returns a dict with keys: outcome, champion_after, challenger,
    challenger_matured_cohorts, sn_lift_vs_champion, consecutive_wins,
    cooldown_until, blocked_by.
    """
    min_cohorts = int(_cfg.get("min_matured_cohorts", _MIN_MATURED_COHORTS))
    hysteresis_weeks = int(_cfg.get("hysteresis_weeks", _HYSTERESIS_WEEKS))
    cooldown_weeks = int(_cfg.get("cooldown_weeks", _COOLDOWN_WEEKS))
    alpha = float(_cfg.get("significance_alpha", _SIGNIFICANCE_ALPHA))

    challenger = _other_champion(champion_before)

    record: dict[str, Any] = {
        "champion_before": champion_before,
        "champion_after": champion_before,
        "challenger": challenger,
        "challenger_matured_cohorts": int(challenger_matured_cohorts),
        "sn_lift_vs_champion": None,
        "consecutive_wins": 0,
        "cooldown_until": cooldown_until,
        "blocked_by": None,
    }

    # Gate (a): matured cohort floor.
    if challenger_matured_cohorts < min_cohorts:
        record["outcome"] = "held_insufficient_data"
        record["blocked_by"] = ["insufficient_matured_cohorts"]
        record["consecutive_wins"] = 0  # thin data resets hysteresis honestly
        return record

    # Gate (b): overlap-aware significance.
    sig = hac_significance(challenger_weekly_sn_lift, alpha=alpha)
    if sig["status"] != "ok":
        record["outcome"] = "held_insufficient_data"
        record["blocked_by"] = ["insufficient_matured_cohorts"]
        record["consecutive_wins"] = 0
        return record

    record["sn_lift_vs_champion"] = round(sig["mean"], 6)
    challenger_winning = sig["significant"] and sig["mean"] > 0

    if not challenger_winning:
        record["outcome"] = "held_not_significant"
        record["blocked_by"] = ["not_significant_hac_adjusted"]
        record["consecutive_wins"] = 0  # a losing/insignificant week resets the streak
        return record

    # Gate (c): hysteresis — this is a winning week; extend the streak.
    consecutive_wins = min(prior_consecutive_wins + 1, hysteresis_weeks)
    record["consecutive_wins"] = consecutive_wins
    if consecutive_wins < hysteresis_weeks:
        record["outcome"] = "held_not_significant"
        record["blocked_by"] = ["hysteresis_not_satisfied"]
        return record

    # Gate (e): cooldown.
    if cooldown_until is not None and as_of < cooldown_until:
        record["outcome"] = "held_cooldown"
        record["blocked_by"] = ["cooldown_active"]
        return record

    # All stat/hysteresis/cooldown gates cleared — this is a promote/demote.
    # Bidirectional (d): outcome label is "promoted" when the challenger is
    # scanner_predictor_direct (moving toward the measured arm) and
    # "demoted" when the challenger is agentic (moving back) — a fixed,
    # symmetric convention, not a judgment call.
    outcome = "promoted" if challenger == "scanner_predictor_direct" else "demoted"
    new_cooldown = _add_weeks(as_of, cooldown_weeks)

    if freeze:
        record["outcome"] = outcome
        record["blocked_by"] = ["frozen"]
        # champion_after / cooldown_until / consecutive_wins are NOT advanced
        # to their post-move values — the pointer did not move, so the
        # carry-forward state must reflect reality, not the suppressed move.
        record["champion_after"] = champion_before
        return record

    record["outcome"] = outcome
    record["champion_after"] = challenger
    record["cooldown_until"] = new_cooldown
    return record


def _add_weeks(as_of: str, weeks: int) -> str:
    d = _date.fromisoformat(as_of)
    return (d + timedelta(weeks=weeks)).isoformat()


# ── config/producer_champion.json writer (single writer, dual caller) ──────


def write_champion_pointer(
    bucket: str,
    champion: str,
    *,
    promotion_source: str,
    upload: bool,
    s3_client=None,
) -> dict:
    """THE single writer for ``config/producer_champion.json``. Both the
    gate engine (``promotion_source="gate_engine"``) and any future one-shot
    operator bootstrap script (``promotion_source="operator_bootstrap"``)
    MUST call this function — never write the pointer directly. This is
    what makes "no parallel writer implementations" true even though the
    bootstrap script itself is out of scope for this module.

    Idempotent / bidirectional-safe: callers only invoke this when a gate
    decision has already determined the pointer SHOULD move (a held week
    must never call this — see ``emit_champion_audit`` below, which only
    calls this on outcome in {"promoted", "demoted"}).

    Raises on S3 write failure when ``upload=True`` — a swallowed failure
    here would silently leave the live executor trading the wrong arm.
    """
    if champion not in VALID_CHAMPIONS:
        raise ValueError(
            f"write_champion_pointer: champion={champion!r} not in {VALID_CHAMPIONS}"
        )
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "champion": champion,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promotion_source": promotion_source,
    }
    if upload:
        s3 = s3_client or boto3.client("s3")
        body = json.dumps(pointer, indent=2, allow_nan=False).encode("utf-8")
        s3.put_object(
            Bucket=bucket, Key=POINTER_KEY, Body=body, ContentType="application/json",
        )
        logger.info(
            "Champion pointer written: s3://%s/%s champion=%s source=%s",
            bucket, POINTER_KEY, champion, promotion_source,
        )
    else:
        logger.info(
            "Champion pointer write skipped (upload=False) — would have set "
            "champion=%s source=%s", champion, promotion_source,
        )
    return pointer


def read_champion_pointer(bucket: str, s3_client=None) -> dict | None:
    """Read the current pointer. Returns None on 404/NoSuchKey (pre-bootstrap
    — no promotion has ever been written; callers should treat champion as
    'agentic', mirroring executor/champion.py's own default). Any other
    error is NOT swallowed here (this is the producer side, not the fail
    -loud executor consumer) but is logged and returns None so a transient
    read hiccup degrades to "treat as agentic" rather than crashing the
    whole weekly evaluate run — the outcome is recorded as
    ``held_insufficient_data``/``error`` by the caller either way, never a
    silent pointer write."""
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=POINTER_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No champion pointer at s3://%s/%s (pre-bootstrap)", bucket, POINTER_KEY)
        else:
            logger.warning("Champion pointer read failed (%s) — treating as pre-bootstrap agentic", e)
        return None
    except Exception as e:  # noqa: BLE001 — degraded-read carve-out, see docstring.
        logger.warning("Champion pointer read failed (%s) — treating as pre-bootstrap agentic", e)
        return None


# ── config/apply_audit/producer_champion/{date}.json writer ────────────────


def load_prior_audit(bucket: str, s3_client=None) -> dict | None:
    """Read the prior weekly audit record (latest.json) for the
    consecutive_wins / cooldown_until carry-forward. Mirrors
    ``optimizer/apply_audit.py.load_prior`` exactly: absent artifact (first
    -ever run) → None; any other read failure logs WARN and returns None
    (state restarts honestly rather than crashing the run)."""
    s3 = s3_client or boto3.client("s3")
    key = f"{AUDIT_PREFIX}/latest.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No prior producer_champion audit at s3://%s/%s (first run)", bucket, key)
        else:
            logger.warning("Prior producer_champion audit read failed (%s) — state restarts", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Prior producer_champion audit read failed (%s) — state restarts", e)
        return None


def build_champion_audit(
    as_of: str,
    gate_result: dict | None,
    *,
    freeze: bool,
    error: str | None = None,
) -> dict:
    """Build the weekly audit record (schema v1,
    ``contracts/producer_champion_audit.schema.json``). Written every week
    regardless of outcome — this IS the liveness proxy (config#2054)."""
    if error is not None or gate_result is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "date": as_of,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "outcome": "error",
            "champion_before": None,
            "champion_after": None,
            "challenger_matured_cohorts": 0,
            "sn_lift_vs_champion": None,
            "consecutive_wins": 0,
            "cooldown_until": None,
            "blocked_by": ["leaderboard_unavailable" if error else "unclassified_error"],
            "freeze": freeze,
            "detail": error or "gate evaluation did not run",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "date": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outcome": gate_result["outcome"],
        "champion_before": gate_result["champion_before"],
        "champion_after": gate_result["champion_after"],
        "challenger_matured_cohorts": gate_result["challenger_matured_cohorts"],
        "sn_lift_vs_champion": gate_result["sn_lift_vs_champion"],
        "consecutive_wins": gate_result["consecutive_wins"],
        "cooldown_until": gate_result["cooldown_until"],
        "blocked_by": gate_result["blocked_by"],
        "challenger": gate_result["challenger"],
        "freeze": freeze,
    }


def write_champion_audit(bucket: str, run_date: str, audit: dict, s3_client=None) -> str:
    """Write dated + latest audit artifacts. RAISES on failure — mirrors
    ``optimizer/apply_audit.write_audit``'s fail-loud posture: this record
    is the load-bearing liveness proxy, a swallowed write failure would
    recreate the exact invisible-silence defect config#2054 exists to
    retire."""
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(audit, indent=2, allow_nan=False).encode("utf-8")
    dated_key = f"{AUDIT_PREFIX}/{run_date}.json"
    latest_key = f"{AUDIT_PREFIX}/latest.json"
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    logger.info("Champion audit written: s3://%s/%s (+ latest.json)", bucket, dated_key)
    return dated_key


def run_weekly_evaluation(
    *,
    bucket: str,
    run_date: str,
    leaderboard: dict | None,
    freeze: bool,
    upload: bool,
    s3_client=None,
) -> dict:
    """Top-level entry point wired into evaluate.py. Runs the full gate
    engine for this week and writes both artifacts:

      1. The weekly audit record (config/apply_audit/producer_champion/
         {date}.json + latest.json) — ALWAYS written, any outcome.
      2. The champion pointer (config/producer_champion.json) — written ONLY
         on outcome in {"promoted", "demoted"} AND not freeze. A held week
         (or a frozen run) never touches the pointer — idempotent,
         bidirectional-safe.

    ``leaderboard`` is the parsed ``research/producer_leaderboard_champion_gate/{date}.json``
    artifact (see ``build_leaderboard_entry`` / the leaderboard reader in
    this module) shaped as
    ``{"challenger_matured_cohorts": int, "challenger_weekly_sn_lift": [float, ...]}``
    for the CURRENT champion's challenger. ``None`` (leaderboard missing or
    unreadable) is treated as an ``error`` outcome — an engine that cannot
    read its input must never guess at a pointer move.

    Returns the audit record that was built (and, for callers that want it,
    it also carries the pointer dict under ``_pointer_write`` when a write
    happened — internal to evaluate.py wiring, not part of the frozen
    audit schema).
    """
    prior_audit = load_prior_audit(bucket, s3_client=s3_client) if upload else None
    prior_consecutive_wins = int((prior_audit or {}).get("consecutive_wins", 0) or 0)
    cooldown_until = (prior_audit or {}).get("cooldown_until")

    pointer = read_champion_pointer(bucket, s3_client=s3_client)
    champion_before = (pointer or {}).get("champion", "agentic")
    if champion_before not in VALID_CHAMPIONS:
        logger.warning(
            "Champion pointer had unrecognized champion=%r — treating as 'agentic' "
            "for gate purposes only (the pointer itself is left untouched unless "
            "gates clear a move)", champion_before,
        )
        champion_before = "agentic"

    gate_result = None
    error = None
    if leaderboard is None:
        error = (
            f"research/producer_leaderboard_champion_gate/{run_date}.json unavailable — "
            "cannot evaluate champion-promotion gates this week"
        )
    else:
        try:
            gate_result = evaluate_gates(
                champion_before=champion_before,
                challenger_matured_cohorts=int(leaderboard.get("challenger_matured_cohorts", 0)),
                challenger_weekly_sn_lift=list(leaderboard.get("challenger_weekly_sn_lift", [])),
                prior_consecutive_wins=prior_consecutive_wins,
                cooldown_until=cooldown_until,
                as_of=run_date,
                freeze=freeze,
            )
        except Exception as e:  # noqa: BLE001 — gate evaluation must never
            # crash the weekly evaluate run; record as an error outcome
            # (still written, per the liveness posture) and move on.
            logger.exception("Champion-promotion gate evaluation raised")
            error = str(e)

    audit = build_champion_audit(run_date, gate_result, freeze=freeze, error=error)

    pointer_written = None
    if gate_result is not None and gate_result["outcome"] in ("promoted", "demoted") and not freeze:
        pointer_written = write_champion_pointer(
            bucket, gate_result["champion_after"],
            promotion_source="gate_engine", upload=upload, s3_client=s3_client,
        )

    logger.info(
        "producer_champion evaluation: outcome=%s champion_before=%s champion_after=%s "
        "consecutive_wins=%s cooldown_until=%s blocked_by=%s",
        audit["outcome"], audit["champion_before"], audit["champion_after"],
        audit["consecutive_wins"], audit["cooldown_until"], audit["blocked_by"],
    )

    if upload:
        try:
            write_champion_audit(bucket, run_date, audit, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_champion audit S3 write failed — this is the "
                "liveness proxy, surfacing loudly",
            )
            raise
    else:
        logger.info("producer_champion audit S3 write skipped (upload=%s) — logged only", upload)

    result = dict(audit)
    if pointer_written is not None:
        result["_pointer_write"] = pointer_written
    return result


# ── research/producer_leaderboard_champion_gate/{date}.json ─────────────────
#
# This artifact did not exist anywhere in the repo prior to config#2367 (the
# issue's description of it was aspirational — verified by grep, config#2367
# groom notes). It is introduced here as the champion engine's own input:
# a per-run snapshot of the challenger arm's weekly sector-neutral 21d lift
# vs. the champion, derived from the existing
# ``analysis.end_to_end.compute_lift_metrics()['scanner_then_predictor_counterfactual']``
# counterfactual (methods block keyed 'scanner_then_predictor_topN' /
# 'agentic_cio_advance'), APPENDED to a running history so the HAC gate has a
# multi-week time series to test rather than a single point estimate.
#
# config#2452 (found 2026-07-13, same day as first live run post-merge): this
# key was originally `research/producer_leaderboard/{date}.json` — the SAME
# key crucible-research's `scoring/leaderboard_producers.py` already writes,
# with an incompatible schema (`{"leaderboard_id": "producer", ...}` vs this
# module's `{"weekly_points": [...]}`). Per the Saturday SF ordering
# (Research writes in Branch A, this Evaluator step runs later in Branch B),
# this module's write would silently clobber crucible-research's real
# multi-arm producer leaderboard every week, AND this module's own
# `read_prior_leaderboard_history` would never find its own prior weeks'
# `weekly_points` in what Research had written — perpetual cold start.
# Renamed to a distinct key before any collision occurred (verified via
# `aws s3 ls` — nothing had landed under the old key yet).


LEADERBOARD_KEY_TMPL = "research/producer_leaderboard_champion_gate/{date}.json"
_LEADERBOARD_HISTORY_KEEP_WEEKS = 26  # ~6 months of weekly points is plenty for HAC lag=3


def leaderboard_entry_from_e2e_lift(e2e_lift: dict | None) -> dict | None:
    """Extract this week's sector-neutral lift point (scanner_then_predictor
    vs. agentic_cio_advance — i.e. scanner_predictor_direct vs. agentic) from
    the e2e_lift diagnostic already computed earlier in the same evaluate
    run. Returns None when the counterfactual is unavailable this week
    (skipped/insufficient_data/error/missing) — an honest "no new point this
    week" rather than fabricating one.
    """
    if not isinstance(e2e_lift, dict):
        return None
    cf = e2e_lift.get("scanner_then_predictor_counterfactual")
    if not isinstance(cf, dict) or cf.get("status") != "ok":
        return None
    methods = cf.get("methods", {})
    pred = methods.get("scanner_then_predictor_topN")
    if not isinstance(pred, dict):
        return None
    sn_lift = pred.get("sn_lift_vs_agentic_cio")
    if sn_lift is None:
        return None
    return {
        "sn_lift_vs_agentic_cio": float(sn_lift),
        "n_picks": pred.get("n_picks"),
        "n_cycles": cf.get("n_cycles"),
    }


def build_leaderboard_artifact(run_date: str, history: list[dict], new_entry: dict | None) -> dict:
    """Append ``new_entry`` (if any) to ``history`` (oldest-first list of
    ``{"date": ..., "sn_lift_vs_agentic_cio": ..., "n_picks": ..., "n_cycles": ...}``),
    trim to the retention window, and return the full artifact to write to
    ``research/producer_leaderboard_champion_gate/{run_date}.json``.
    """
    points = list(history)
    if new_entry is not None:
        points = [p for p in points if p.get("date") != run_date]
        points.append({"date": run_date, **new_entry})
    points = points[-_LEADERBOARD_HISTORY_KEEP_WEEKS:]
    return {
        "schema_version": 1,
        "as_of": run_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weekly_points": points,
    }


def leaderboard_gate_inputs(artifact: dict) -> dict:
    """Reduce a leaderboard artifact to the shape ``run_weekly_evaluation``
    expects: matured cohort count (weeks with a matured point) and the
    weekly SN-lift series for the HAC gate."""
    points = artifact.get("weekly_points", []) if isinstance(artifact, dict) else []
    lifts = [p["sn_lift_vs_agentic_cio"] for p in points if p.get("sn_lift_vs_agentic_cio") is not None]
    return {
        "challenger_matured_cohorts": len(lifts),
        "challenger_weekly_sn_lift": lifts,
    }


def read_leaderboard(bucket: str, run_date: str, s3_client=None) -> dict | None:
    s3 = s3_client or boto3.client("s3")
    key = LEADERBOARD_KEY_TMPL.format(date=run_date)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    except Exception:
        return None


def read_prior_leaderboard_history(bucket: str, run_date: str, s3_client=None) -> list[dict]:
    """Read the most recent leaderboard artifact available (by scanning
    backward from ``run_date`` — NOT wall-clock today, so a ``--date``
    backfill run seeds history relative to the backfilled trading day, not
    the day the backfill happens to execute) to seed ``weekly_points``
    history. Falls back to an empty history (cold start) on any read
    failure — the engine will simply need ``min_matured_cohorts`` fresh
    weeks before it can promote/demote, which is the same honest starvation
    posture as any other new leaderboard.
    """
    s3 = s3_client or boto3.client("s3")
    anchor = _date.fromisoformat(run_date)
    for back in range(1, 15):
        probe_date = (anchor - timedelta(days=back)).isoformat()
        key = LEADERBOARD_KEY_TMPL.format(date=probe_date)
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            return list(data.get("weekly_points", []))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                continue
            logger.warning("Leaderboard history probe failed at %s: %s", key, e)
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning("Leaderboard history probe failed at %s: %s", key, e)
            return []
    return []


def write_leaderboard(bucket: str, run_date: str, artifact: dict, s3_client=None) -> str:
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(artifact, indent=2, allow_nan=False).encode("utf-8")
    key = LEADERBOARD_KEY_TMPL.format(date=run_date)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("producer_leaderboard written: s3://%s/%s (%d weekly points)", bucket, key, len(artifact.get("weekly_points", [])))
    return key


def update_leaderboard_and_get_gate_inputs(
    bucket: str, run_date: str, e2e_lift: dict | None, *, upload: bool, s3_client=None,
) -> dict:
    """Full leaderboard maintenance step: read prior history, append this
    week's point (if the counterfactual matured this run), write the
    updated artifact, and return the gate-ready reduction. Called once per
    evaluate run, BEFORE ``run_weekly_evaluation``.
    """
    history = read_prior_leaderboard_history(bucket, run_date, s3_client=s3_client) if upload else []
    new_entry = leaderboard_entry_from_e2e_lift(e2e_lift)
    artifact = build_leaderboard_artifact(run_date, history, new_entry)
    if upload:
        try:
            write_leaderboard(bucket, run_date, artifact, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_leaderboard write failed — champion gates will "
                "evaluate against in-memory history only this run",
            )
    return leaderboard_gate_inputs(artifact)
