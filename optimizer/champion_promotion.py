"""champion_promotion.py — weekly winner-take-all champion/challenger gate
(config#2364 / config#2367 origin; redesigned alpha-engine-config-I2518 /
epic I2515, 2026-07-14 ruling; scoring redesigned to direct per-arm lift,
no shared comparator, alpha-engine-config-I2998, 2026-07-20 ruling).

Writes the live pointer ``config/producer_champion.json`` that the
alpha-engine executor's ``executor/champion.py::load_champion_pointer``
reads at planner start to decide which entry-candidate producer arm is
LIVE. Statistical correctness matters here in a way it does not for the
advisory optimizers elsewhere in this module: a wrong pointer move silently
changes which strategy trades real (paper) capital.

**2026-07-14 seat swap (Brian's ruling, config-I2518, binding on this
issue):** the ``agentic`` seat retires with the multi-agent Research graph
(epic config-I2515) and is replaced by ``thinktank_coverage`` — the Think
Tank challenger arm (scanner top-~60 -> Think Tank full-coverage -> its
own top-~20 by independent TT rating). ``scanner_predictor_direct`` is the
new BASE-CASE champion (already live since 2026-07-13T22:07 UTC,
config-I2364)::

    VALID_CHAMPIONS = ("scanner_predictor_direct", "thinktank_coverage")

``agentic`` is READ-TOLERATED (a historical pointer/audit value must never
crash this engine — ``_normalize_champion_before`` below WARNs and treats
it as ``scanner_predictor_direct``) but WRITE-FORBIDDEN
(``write_champion_pointer`` raises on any value outside ``VALID_CHAMPIONS``,
which no longer includes it). No real ``config/producer_champion.json``
object was ever actually written with ``champion="agentic"`` — it was only
ever an implicit pre-bootstrap default — but the 2026-07-13 bootstrap DID
write a real audit record with ``champion_before="agentic"``
(``config/apply_audit/producer_champion/2026-07-13.json``), so the
read-tolerance is not purely defensive.

**Weekly winner-take-all policy (supersedes the entire HAC-significance /
2-week-hysteresis / 2-week-cooldown gated engine this module shipped with
under config#2367 — INCLUDING the standing ``cooldown_until:
2026-07-27`` carried in the prior ``latest.json``, which this policy no
longer reads or honors):**

    Each weekly evaluation compares the two arms' realized top-N alpha lift
    for the trailing week and flips the pointer to whichever arm scores
    higher, if that arm is not already champion. No significance test, no
    consecutive-week hysteresis, no cooldown — "whichever performs best in
    a given week is promoted at that time" (Brian's ruling, verbatim).

**Validity guards (definitional NO-CONTEST, not a statistical gate) —
``evaluate_gates`` below:** a week where either arm's realized-lift score
is unavailable (no valid ``thinktank_coverage`` selections this week, no
resolved/matured outcomes yet, the evidence artifact itself missing or
stale) is a NO-CONTEST: the pointer is left unchanged and the outcome
record says so explicitly via a machine-readable ``blocked_by`` slug. A
no-contest NEVER defaults a win to either side.

**Evidence sourcing — DIRECT per-arm realized lift, NO shared comparator
(alpha-engine-config-I2998, 2026-07-20 ruling — supersedes the
Bucher-style indirect/common-comparator design below this module shipped
with under I2518):**

  The pre-I2998 design scored both arms as "lift vs the live
  ``agentic_sector_teams``/CIO-ADVANCE baseline" on the premise that
  Research kept running its full agentic pipeline weekly regardless of
  which arm the executor traded. config-I2993 (2026-07-19/20) found that
  premise false: ``agentic_sector_teams`` retired 2026-07-12 with no
  successor ``kind=="champion"`` producer registered, so BOTH arms'
  "vs agentic" scores could go simultaneously no-contest — a materially
  worse failure than either arm alone going stale, since a no-contest week
  is a legitimate, non-alerting outcome by design (freezing
  ``config/producer_champion.json`` silently). I2998's fix removes the
  shared-comparator dependency entirely: each arm now scores its OWN
  realized lift against a FIXED, always-available neutral baseline, so
  neither arm's score can ever depend on whether Research's agentic
  pipeline (or any future comparator) happens to be live that week.

  - ``scanner_predictor_direct``'s weekly score is this run's
    ``analysis.end_to_end.compute_lift_metrics()['scanner_then_predictor_counterfactual']
    ['methods']['scanner_then_predictor_topN']['sector_neutral_mean_alpha_21d']``
    — the arm's own realized, sector-neutral 21d alpha, ALREADY benchmark
    -relative (realized log return minus the log SPY return over the same
    window, at the source, ``analysis/end_to_end.py::_scanner_then_predictor_topN``) —
    i.e. lift vs the SPY zero-line, not vs any live comparator arm. A
    backtester-internal counterfactual (research.db-derived) already
    computed earlier in the same ``evaluate.py`` run, extracted via
    ``leaderboard_entry_from_e2e_lift`` (this module's OWN
    ``research/producer_leaderboard_champion_gate/{date}.json`` history
    artifact is STILL maintained for observability and to keep
    config#2452's in-flight live-verification intact — see
    ``update_leaderboard_and_get_gate_inputs`` — but its accumulated
    ``weekly_points`` series is no longer consumed by the gate itself,
    since winner-take-all needs only THIS week's point, not a multi-week
    HAC-adjusted series). The retired ``sn_lift_vs_agentic_cio`` field is
    still carried on the leaderboard-history entry for observability but
    is no longer the gate's score source.
  - ``thinktank_coverage``'s weekly score is read from crucible-research's
    real champion/challenger producer leaderboard,
    ``research/producer_leaderboard/{date}.json``
    (``scoring/leaderboard_producers.py::build_producer_leaderboard`` +
    ``scoring/leaderboard_scoring.py::score_leaderboard``, config#1221/
    #1223, made champion-optional under I2998) — verified schema
    (2026-07-20, read from the crucible-research checkout, NOT guessed):
    ``{"champion": <research producer champion name> | None,
    "horizon_days": 21, "top_n": 50, "benchmark_ticker": "SPY", "n_dates":
    int, "specs": [{"name", "kind", "realized_rank_ic",
    "topn_alpha_vs_champion": {...} | None,
    "topn_alpha_vs_benchmark": {"mean","se","t_stat","n_dates"} | None,
    "n_dates_scored"}, ...]}``. We read the ``specs`` row named
    ``"thinktank_coverage"`` and take its ``topn_alpha_vs_benchmark.mean``
    — the SAME kind of statistic as ``scanner_predictor_direct``'s score
    (a mean top-N realized return lift vs the SPY benchmark, date
    -clustered), so the two scores remain apples-to-apples comparable
    under winner-take-all's direct "higher wins" rule. This field is
    computed champion-free (``score_leaderboard`` degrades to
    champion-free metrics for every spec when no producer is registered
    ``kind=="champion"`` — see I2998) — unlike the retired
    ``topn_alpha_vs_champion``, it is available even while config-I2993's
    "no successor champion registered" state persists. ``coverage_complete``
    validity (the full current-scan top-60 rule, Brian's ruling
    config#1580) is enforced UPSTREAM at the artifact boundary —
    crucible-research PR427 writes ``signals_shadow/thinktank_coverage/
    {trading_day}/signals.json`` (the input this leaderboard scores) ONLY
    when ``coverage_complete`` — so any date this spec contributed to
    ``n_dates_scored`` was necessarily a full-coverage day; no separate
    coverage_complete check is needed on this side of the boundary.

    **LATEST-AVAILABLE read (alpha-engine-config-I2544, 2026-07-14 ruling,
    same-session follow-up to I2518):** ``research/producer_leaderboard/
    {date}.json`` is now written by an ASYNC advisory child Step Function
    (config-I2518's persistent-dash rearchitecture) that may not have
    finished — or may have failed outright — by the time this Evaluator
    -stage gate runs in the MAIN weekly SF. An exact same-day key read is
    therefore no longer a safe assumption. This module instead lists the
    ``research/producer_leaderboard/`` prefix
    (``find_latest_research_producer_leaderboard_date``) and reads the
    LATEST artifact dated <= ``run_date`` (``read_latest_research_producer
    _leaderboard``). This is the semantically CORRECT read, not a
    compromise: the gate scores REALIZED (matured) outcomes of PRIOR
    weeks' ``thinktank_coverage`` selections — a same-day leaderboard could
    not contain resolved outcomes for same-day picks even if it existed on
    time. An honest staleness bound still applies: a selected leaderboard
    more than ``LEADERBOARD_STALENESS_DAYS`` (8) calendar days older than
    ``run_date`` is treated as unavailable (``leaderboard_stale_gt_8d``,
    a no-contest) rather than silently scored against stale evidence. The
    date actually used is threaded through to the audit record as
    ``leaderboard_date_used`` (additive, contracts/producer_champion_audit
    .schema.json) so the audit trail always shows which week's evidence
    decided (or declined to decide) a flip.

  **config-I2993 item 2 (windowing ``end_to_end.py``'s
  ``sn_lift_vs_agentic_cio`` aggregation) is NO LONGER a dependency for
  this gate's correctness** — that field is retired as this gate's score
  source under I2998 (still computed and carried for observability/other
  consumers, e.g. the evaluator tile, but this module reads
  ``sector_neutral_mean_alpha_21d`` instead). It may still be worth doing
  for the evaluator tile's own accuracy, independent of this gate.

  **KNOWN, TRACKED GAP as of 2026-07-20 (filed
  alpha-engine-config-I2519, unaffected by this redesign):**
  ``thinktank_coverage`` is NOT YET registered in crucible-research's
  ``producers/registry.py::RESEARCH_PRODUCERS`` / ``challenger_producers()``
  — PR427's own commit message explicitly deferred that wiring ("Not
  registered in producers/registry.py ... being decided separately per
  config#1683's fail-hard challenger-gap doctrine"). Until that
  registration lands, ``research/producer_leaderboard/{date}.json``'s
  ``specs`` list will NEVER contain a ``"thinktank_coverage"`` row, so
  ``_score_thinktank_coverage`` below will correctly and honestly return
  ``blocked_by=["thinktank_coverage_not_in_leaderboard"]`` (a NO-CONTEST)
  every week until it does — this is expected, not a bug in this module,
  and is now fully independent of whether a champion producer is
  registered (I2998 decoupled the two concerns). See the filed issue for
  the concrete unblock.

``hac_significance`` (Newey-West/HAC overlap-aware significance) is
RETAINED below, unchanged and still independently unit-tested — it is no
longer wired into the promotion decision (winner-take-all has no
significance gate) but remains available as a possible future diagnostic
(e.g. reporting whether a winning margin looks like signal or noise
alongside the decision) without having to re-derive it.

**Single writer function, dual caller (no parallel writer implementations):**
``write_champion_pointer`` is the ONLY code path that may write
``config/producer_champion.json``. The gate engine calls it with
``promotion_source="gate_engine"``; the one-shot 2026-07-13 operator
bootstrap (``bootstrap_champion_promotion.py``) called it with
``promotion_source="operator_bootstrap"`` — never a hand-edited S3 object.

**Liveness (config#2054 lesson, binding):** ``config/producer_champion.json``
mtime alone cannot prove this engine is alive — a correctly-held
(no-contest) week does not touch it. ``run_weekly_evaluation`` writes
``config/apply_audit/producer_champion/{date}.json`` (+ ``latest.json``)
UNCONDITIONALLY, including on ``outcome="error"``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1          # config/producer_champion.json pointer — unchanged shape
AUDIT_SCHEMA_VERSION = 2    # config/apply_audit/producer_champion/{date}.json — v2 shape (I2518)
POINTER_KEY = "config/producer_champion.json"
AUDIT_PREFIX = "config/apply_audit/producer_champion"

VALID_CHAMPIONS = ("scanner_predictor_direct", "thinktank_coverage")

# Retired seat(s) — READ-TOLERATED (a historical pointer/audit artifact using
# this value must never crash the engine) but WRITE-FORBIDDEN (excluded from
# VALID_CHAMPIONS, so write_champion_pointer raises on it).
_LEGACY_CHAMPIONS = ("agentic",)

OUTCOMES = ("promoted", "no_contest", "unchanged_winner_already_champion", "error")

# blocked_by slugs — union of the current winner-take-all vocabulary and two
# retired vocabularies kept for read-tolerance of historical audit records:
# the pre-I2518 HAC/hysteresis/cooldown engine, and the pre-I2544
# exact-date-only leaderboard read (superseded same-day by the
# latest-available read below; no code path in this module writes either
# retired group again). Slug vocabulary is unchanged by the I2998 direct
# -lift rescoring — only the underlying score SOURCE field changed per arm,
# not the failure-mode taxonomy.
_BLOCKED_BY_SLUGS = (
    # current (winner-take-all + latest-available leaderboard read + direct
    # per-arm lift scoring, I2518/I2544/I2998)
    "no_valid_scanner_predictor_direct_selections",
    "no_valid_thinktank_coverage_selections",
    "scanner_predictor_direct_counterfactual_unavailable",
    "thinktank_coverage_not_in_leaderboard",
    "thinktank_coverage_no_resolved_outcomes",
    "leaderboard_unavailable",
    "leaderboard_stale_gt_8d",
    "arm_score_unavailable",
    "frozen",
    "unclassified_error",
    # retired (pre-I2518 HAC/hysteresis/cooldown engine) — historical read-only
    "insufficient_matured_cohorts",
    "cooldown_active",
    "not_significant_hac_adjusted",
    "hysteresis_not_satisfied",
    # retired (pre-I2544 exact-date-only leaderboard read) — historical
    # read-only: this slug fired when the artifact's self-reported "date"
    # field disagreed with an exact-match run_date key read; the
    # latest-available read no longer requires an exact match, so this
    # condition can no longer occur (superseded by leaderboard_stale_gt_8d
    # for the age-bound case).
    "leaderboard_stale",
)

# Honest staleness bound (alpha-engine-config-I2544, 2026-07-14): a selected
# research/producer_leaderboard/{date}.json artifact older than this many
# calendar days relative to run_date is treated as unavailable rather than
# scored — see find_latest_research_producer_leaderboard_date /
# _score_thinktank_coverage.
LEADERBOARD_STALENESS_DAYS = 8

RESEARCH_PRODUCER_LEADERBOARD_PREFIX = "research/producer_leaderboard/"
_RESEARCH_PRODUCER_LEADERBOARD_KEY_RE = re.compile(
    r"^research/producer_leaderboard/(\d{4}-\d{2}-\d{2})\.json$"
)

# HAC lag helper constants — still consulted by hac_significance() below,
# which is retained as an independent, tested utility (see module docstring)
# even though it no longer gates the promotion decision.
_HORIZON_DAYS = 21              # 21d forward-alpha horizon (config#1405 basis)
_CADENCE_DAYS = 7               # weekly evaluation cadence

_cfg: dict = {}


def init_config(config: dict) -> None:
    """Called unconditionally by evaluate.py at optimizer-stage start. The
    winner-take-all engine currently defines no configurable thresholds
    (no significance level, no hysteresis/cooldown weeks) — this is kept as
    a stable entry point for evaluate.py's wiring and for hac_significance's
    still-configurable horizon/cadence (lag = round(horizon/cadence))."""
    global _cfg
    _cfg = config.get("champion_promotion", {})


def _hac_lag() -> int:
    """Bartlett-kernel lag = round(horizon_days / cadence_days). Feeds
    ``hac_significance`` only — not load-bearing for the winner-take-all
    decision itself."""
    horizon = int(_cfg.get("horizon_days", _HORIZON_DAYS))
    cadence = int(_cfg.get("cadence_days", _CADENCE_DAYS))
    return max(0, round(horizon / cadence))


# ── Retained utility: HAC/Newey-West-adjusted overlap-aware significance ───
# Not wired into evaluate_gates() under the winner-take-all policy (see
# module docstring) — kept as an independently unit-tested, available
# diagnostic.


def hac_significance(
    weekly_sn_lift: list[float], *, alpha: float = 0.05,
) -> dict:
    """Overlap-aware two-sided significance test of whether a weekly
    sector-neutral lift series is significantly different from zero, using
    the Newey-West (1994) HAC standard error of the mean (Bartlett kernel;
    lag = ``_hac_lag()``) in place of the naive i.i.d. ``s/sqrt(n)`` standard
    error. See the module history (git log) for the full derivation this
    docstring previously carried in-line; unchanged behavior, retained as an
    available utility (not currently gate-connected — see module docstring).

    Uses the vendored, independently-unit-tested
    ``nousergon_lib.quant.stats.intervals.newey_west_se``.

    Returns a dict:
      ``{"status": "ok", "n": int, "mean": float, "se": float, "lags": int,
         "t_stat": float, "p_value": float, "significant": bool}``
      or ``{"status": "insufficient_data", "n": int}`` when fewer than 2
      finite observations are available.
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


# ── Gate engine (weekly winner-take-all) ────────────────────────────────────


def _other_champion(champion: str) -> str:
    others = [c for c in VALID_CHAMPIONS if c != champion]
    if len(others) != 1:
        raise ValueError(
            f"_other_champion expects exactly 2 VALID_CHAMPIONS, got {VALID_CHAMPIONS!r}"
        )
    return others[0]


def _normalize_champion_before(champion: str) -> str:
    """Normalize a pointer/default champion value for GATE purposes only —
    never mutates the pointer itself (a held/no-contest week must never
    write). ``agentic`` (the retired seat, config-I2518 seat swap) WARNs and
    is treated as ``scanner_predictor_direct`` — belt-and-braces: the live
    pointer has been ``scanner_predictor_direct`` since 2026-07-13, so this
    path is not expected to fire in practice, only to guarantee a stale or
    hand-inspected historical pointer can never crash this engine. Any
    other unrecognized value is treated the same way (WARN + default to the
    base-case arm)."""
    if champion in VALID_CHAMPIONS:
        return champion
    if champion in _LEGACY_CHAMPIONS:
        logger.warning(
            "Champion pointer had legacy champion=%r (retired seat, "
            "alpha-engine-config-I2518 seat swap) — normalizing to %r for "
            "gate purposes only; the pointer itself is left untouched unless "
            "this week's gates clear a move.",
            champion, VALID_CHAMPIONS[0],
        )
        return VALID_CHAMPIONS[0]
    logger.warning(
        "Champion pointer had unrecognized champion=%r — treating as %r for "
        "gate purposes only (the pointer itself is left untouched unless "
        "gates clear a move)", champion, VALID_CHAMPIONS[0],
    )
    return VALID_CHAMPIONS[0]


def evaluate_gates(
    *,
    champion_before: str,
    arm_scores: dict,
    freeze: bool,
) -> dict:
    """Weekly winner-take-all decision (Brian's ruling, alpha-engine-config
    -I2518, 2026-07-14) — supersedes the HAC-significance / 2-week hysteresis
    / 2-week cooldown gates this module previously enforced, INCLUDING the
    standing ``cooldown_until: 2026-07-27`` carried in a prior audit record
    (no longer read or honored).

    ``arm_scores`` is the return of ``build_weekly_arm_scores`` below:
    ``{"scores": {"scanner_predictor_direct": float|None,
    "thinktank_coverage": float|None}, "unavailable_reasons": {arm: slug,
    ...}, "leaderboard_date_used": str|None}``. A ``None`` score means no
    valid evidence exists for that arm THIS week — a definitional
    NO-CONTEST (validity guard), never a statistical gate and never a
    default win for either side. ``leaderboard_date_used`` (alpha-engine
    -config-I2544) is the date of the ``research/producer_leaderboard/
    {date}.json`` artifact actually selected (latest available <=
    run_date) — carried through into every outcome record (promoted,
    no_contest, and unchanged) so the audit trail always shows which
    week's evidence decided (or declined to decide) a flip.

    Decision: whichever arm has the strictly higher score this week wins.
    A tie (or either side missing) never flips the pointer — ties favor the
    incumbent (``champion_before``) so the pointer never moves on a null or
    exactly-equal signal.

    Pure function — no I/O — independently unit-testable against synthetic
    score fixtures.

    Returns a dict with keys: outcome, champion_before, champion_after,
    challenger, champion_score, challenger_score, blocked_by,
    leaderboard_date_used.
    """
    challenger = _other_champion(champion_before)
    scores = arm_scores.get("scores", {})
    reasons = arm_scores.get("unavailable_reasons", {})
    champ_score = scores.get(champion_before)
    chall_score = scores.get(challenger)

    record: dict[str, Any] = {
        "champion_before": champion_before,
        "champion_after": champion_before,
        "challenger": challenger,
        "champion_score": champ_score,
        "challenger_score": chall_score,
        "blocked_by": None,
        "leaderboard_date_used": arm_scores.get("leaderboard_date_used"),
    }

    if champ_score is None or chall_score is None:
        blocked: list[str] = []
        if champ_score is None:
            blocked.append(reasons.get(champion_before, "arm_score_unavailable"))
        if chall_score is None:
            blocked.append(reasons.get(challenger, "arm_score_unavailable"))
        record["outcome"] = "no_contest"
        record["blocked_by"] = blocked
        return record

    winner = challenger if chall_score > champ_score else champion_before

    if winner == champion_before:
        record["outcome"] = "unchanged_winner_already_champion"
        return record

    # Challenger wins this week — a promotion, subject only to --freeze.
    if freeze:
        record["outcome"] = "promoted"
        record["blocked_by"] = ["frozen"]
        # champion_after is NOT advanced under freeze — the write is
        # suppressed, so the carry-forward state must reflect reality.
        return record

    record["outcome"] = "promoted"
    record["champion_after"] = winner
    return record


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
    gate engine (``promotion_source="gate_engine"``) and the one-shot
    2026-07-13 operator bootstrap (``promotion_source="operator_bootstrap"``)
    call this function — never write the pointer directly.

    ``champion`` MUST be in ``VALID_CHAMPIONS`` — this is the write-forbidden
    half of the read-tolerated/write-forbidden posture for retired seats
    (e.g. ``agentic``): raises ValueError for anything else, including every
    ``_LEGACY_CHAMPIONS`` value.

    Idempotent / bidirectional-safe: callers only invoke this when a gate
    decision has already determined the pointer SHOULD move (a no-contest or
    unchanged week must never call this).

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
    the base-case arm, ``VALID_CHAMPIONS[0]`` == 'scanner_predictor_direct',
    mirroring executor/champion.py's own default post-I2515). Any other
    error is NOT swallowed here (this is the producer side, not the
    fail-loud executor consumer) but is logged and returns None so a
    transient read hiccup degrades to the base-case default rather than
    crashing the whole weekly evaluate run — the outcome is recorded as
    ``error`` by the caller either way, never a silent pointer write."""
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=POINTER_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No champion pointer at s3://%s/%s (pre-bootstrap)", bucket, POINTER_KEY)
        else:
            logger.warning(
                "Champion pointer read failed (%s) — treating as pre-bootstrap "
                "base-case default", e,
            )
        return None
    except Exception as e:  # noqa: BLE001 — degraded-read carve-out, see docstring.
        logger.warning(
            "Champion pointer read failed (%s) — treating as pre-bootstrap "
            "base-case default", e,
        )
        return None


# ── config/apply_audit/producer_champion/{date}.json writer ────────────────


def load_prior_audit(bucket: str, s3_client=None) -> dict | None:
    """Read the prior weekly audit record (latest.json). Retained for API
    parity with the pre-I2518 engine (some callers/tests may still probe
    prior-run state for observability) — no longer consulted by
    evaluate_gates itself (winner-take-all carries no state forward:
    no hysteresis counter, no cooldown date). Absent artifact (first-ever
    run) -> None; any other read failure logs WARN and returns None."""
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
    """Build the weekly audit record (schema v2,
    ``contracts/producer_champion_audit.schema.json`` — bumped from v1 under
    alpha-engine-config-I2518: the HAC/hysteresis/cooldown fields
    (``challenger_matured_cohorts``, ``sn_lift_vs_champion``,
    ``consecutive_wins``, ``cooldown_until``) are retired in favor of
    ``champion_score``/``challenger_score`` — no live consumer outside this
    repo reads the audit record's fields (verified 2026-07-14), so no
    cross-repo coordination was required for the bump; v1 historical
    records remain valid documents under the frozen v1 shape and are not
    revalidated against v2). Written every week regardless of outcome —
    this IS the liveness proxy (config#2054).

    ``leaderboard_date_used`` (additive, alpha-engine-config-I2544,
    2026-07-14) is the date of the ``research/producer_leaderboard/
    {date}.json`` artifact actually consulted this run (the latest
    available <= ``as_of``, or None when no leaderboard was available at
    all / evaluation aborted before scoring) — always present (nullable),
    on every outcome including ``error``, so the audit trail is never
    silent about which week's evidence decided a flip."""
    if error is not None or gate_result is None:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "date": as_of,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "outcome": "error",
            "champion_before": None,
            "champion_after": None,
            "champion_score": None,
            "challenger_score": None,
            "blocked_by": ["leaderboard_unavailable" if error else "unclassified_error"],
            "freeze": freeze,
            "detail": error or "gate evaluation did not run",
            "leaderboard_date_used": None,
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "date": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outcome": gate_result["outcome"],
        "champion_before": gate_result["champion_before"],
        "champion_after": gate_result["champion_after"],
        "champion_score": gate_result["champion_score"],
        "challenger_score": gate_result["challenger_score"],
        "blocked_by": gate_result["blocked_by"],
        "challenger": gate_result["challenger"],
        "freeze": freeze,
        "leaderboard_date_used": gate_result.get("leaderboard_date_used"),
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


# ── Weekly arm-score sourcing ────────────────────────────────────────────────


def _score_scanner_predictor_direct(e2e_lift: dict | None) -> tuple[float | None, str | None]:
    """scanner_predictor_direct's weekly score (alpha-engine-config-I2998):
    this run's backtester-internal ``scanner_then_predictor_topN``
    counterfactual's OWN realized sector-neutral 21d alpha —
    ``sector_neutral_mean_alpha_21d`` — which is already benchmark-relative
    at the source (realized log return minus the log SPY return over the
    same window, see ``analysis/end_to_end.py::_scanner_then_predictor_topN``),
    i.e. a direct lift vs the SPY zero-line, not vs any live comparator arm. See
    ``leaderboard_entry_from_e2e_lift``."""
    entry = leaderboard_entry_from_e2e_lift(e2e_lift)
    if entry is None:
        return None, "scanner_predictor_direct_counterfactual_unavailable"
    return entry["sector_neutral_mean_alpha_21d"], None


def _score_thinktank_coverage(
    tt_leaderboard: dict | None, run_date: str, leaderboard_date_used: str | None,
) -> tuple[float | None, str | None]:
    """thinktank_coverage's weekly score: read from crucible-research's
    ``research/producer_leaderboard/{date}.json`` (see module docstring for
    the verified schema and the coverage_complete-enforced-upstream
    reasoning).

    ``leaderboard_date_used`` (alpha-engine-config-I2544) is the date of
    the ``tt_leaderboard`` artifact actually selected by
    ``find_latest_research_producer_leaderboard_date`` — the latest
    available <= ``run_date``, never the same-day exact match this
    function required before I2544. An honest staleness bound still
    applies: more than ``LEADERBOARD_STALENESS_DAYS`` calendar days older
    than ``run_date`` is treated as unavailable (this IS the semantically
    correct behavior, not a compromise — the gate scores realized outcomes
    of PRIOR weeks' selections, which a same-day leaderboard could not
    contain anyway; see module docstring)."""
    if not isinstance(tt_leaderboard, dict) or leaderboard_date_used is None:
        return None, "leaderboard_unavailable"
    age_days = (
        date.fromisoformat(run_date) - date.fromisoformat(leaderboard_date_used)
    ).days
    if age_days < 0:
        # Defensive: find_latest_research_producer_leaderboard_date never
        # selects a date > run_date, but a caller passing
        # leaderboard_date_used directly (bypassing that selection) must
        # never have a "future" artifact trusted as this week's evidence.
        return None, "leaderboard_unavailable"
    if age_days > LEADERBOARD_STALENESS_DAYS:
        return None, "leaderboard_stale_gt_8d"
    specs = tt_leaderboard.get("specs")
    if not isinstance(specs, list):
        return None, "leaderboard_unavailable"
    row = next(
        (s for s in specs if isinstance(s, dict) and s.get("name") == "thinktank_coverage"),
        None,
    )
    if row is None:
        # Expected until crucible-research registers thinktank_coverage in
        # producers/registry.py::challenger_producers() — see module
        # docstring "KNOWN, TRACKED GAP" / alpha-engine-config-I2519. This
        # condition is now fully independent of whether a champion producer
        # is registered (alpha-engine-config-I2998 decoupled the two
        # concerns — score_leaderboard writes this row champion-free).
        return None, "thinktank_coverage_not_in_leaderboard"
    if not row.get("n_dates_scored"):
        return None, "thinktank_coverage_no_resolved_outcomes"
    # alpha-engine-config-I2998: direct lift vs the SPY benchmark, computed
    # champion-free — the SAME kind of statistic as
    # scanner_predictor_direct's score (mean top-N realized return lift vs
    # SPY, date-clustered), replacing the retired topn_alpha_vs_champion
    # (which required a live comparator producer and went permanently
    # unavailable once config-I2993 retired agentic_sector_teams with no
    # successor champion registered).
    alpha = row.get("topn_alpha_vs_benchmark")
    if not isinstance(alpha, dict) or alpha.get("mean") is None:
        return None, "thinktank_coverage_no_resolved_outcomes"
    return float(alpha["mean"]), None


def build_weekly_arm_scores(
    e2e_lift: dict | None,
    tt_leaderboard: dict | None,
    *,
    run_date: str,
    leaderboard_date_used: str | None = None,
) -> dict:
    """Reduce this run's two evidence sources to the shape ``evaluate_gates``
    expects: ``{"scores": {"scanner_predictor_direct": float|None,
    "thinktank_coverage": float|None}, "unavailable_reasons": {arm: slug},
    "leaderboard_date_used": str|None}``. Both scores are lift-vs-the-shared
    -agentic-baseline (see module docstring's common-comparator reasoning)
    — comparable directly, no combined-variance step needed since
    winner-take-all performs no significance test.

    ``leaderboard_date_used`` (alpha-engine-config-I2544) MUST be supplied
    by the caller as the date actually selected via
    ``find_latest_research_producer_leaderboard_date`` /
    ``read_latest_research_producer_leaderboard`` — it is threaded straight
    through into the returned dict (and from there into every
    ``evaluate_gates`` outcome record) so the audit trail always shows
    which week's evidence decided (or declined to decide) a flip."""
    spd_score, spd_reason = _score_scanner_predictor_direct(e2e_lift)
    tt_score, tt_reason = _score_thinktank_coverage(
        tt_leaderboard, run_date, leaderboard_date_used,
    )
    reasons: dict[str, str] = {}
    if spd_reason is not None:
        reasons["scanner_predictor_direct"] = spd_reason
    if tt_reason is not None:
        reasons["thinktank_coverage"] = tt_reason
    return {
        "scores": {
            "scanner_predictor_direct": spd_score,
            "thinktank_coverage": tt_score,
        },
        "unavailable_reasons": reasons,
        "leaderboard_date_used": leaderboard_date_used,
    }


def run_weekly_evaluation(
    *,
    bucket: str,
    run_date: str,
    e2e_lift: dict | None,
    tt_leaderboard: dict | None,
    tt_leaderboard_date_used: str | None = None,
    freeze: bool,
    upload: bool,
    s3_client=None,
) -> dict:
    """Top-level entry point wired into evaluate.py. Runs the weekly
    winner-take-all decision and writes both artifacts:

      1. The weekly audit record (config/apply_audit/producer_champion/
         {date}.json + latest.json) — ALWAYS written, any outcome.
      2. The champion pointer (config/producer_champion.json) — written ONLY
         on outcome="promoted" AND not freeze. A no-contest,
         unchanged-winner-already-champion, or frozen run never touches the
         pointer — idempotent, bidirectional-safe.

    ``e2e_lift`` is the ``diagnostics["e2e_lift"]`` dict already computed
    earlier in the same evaluate.py run (scanner_predictor_direct's
    evidence). ``tt_leaderboard`` is the parsed
    ``research/producer_leaderboard/{date}.json`` artifact for
    ``tt_leaderboard_date_used`` (thinktank_coverage's evidence), read from
    crucible-research via ``read_latest_research_producer_leaderboard`` —
    the LATEST artifact available <= ``run_date`` (alpha-engine-config
    -I2544: the writer is now an async advisory child SF that may not have
    finished/may have failed by the time this gate runs; a same-day exact
    match is no longer assumed). ``tt_leaderboard_date_used`` is the date
    of that selected artifact (None if none was found <= run_date) —
    threaded through to the audit record's ``leaderboard_date_used`` field
    so the audit trail always shows which week's evidence decided a flip.
    Either evidence source being unavailable, or the selected leaderboard
    being more than ``LEADERBOARD_STALENESS_DAYS`` days stale, degrades to
    a no-contest week for that arm (never an ``error`` outcome by itself)
    via ``build_weekly_arm_scores``; only an exception raised during
    evaluation itself produces ``outcome="error"``.

    Returns the audit record that was built (and, for callers that want it,
    it also carries the pointer dict under ``_pointer_write`` when a write
    happened — internal to evaluate.py wiring, not part of the frozen
    audit schema).
    """
    pointer = read_champion_pointer(bucket, s3_client=s3_client)
    champion_before = _normalize_champion_before(
        (pointer or {}).get("champion", VALID_CHAMPIONS[0])
    )

    gate_result = None
    error = None
    try:
        arm_scores = build_weekly_arm_scores(
            e2e_lift, tt_leaderboard, run_date=run_date,
            leaderboard_date_used=tt_leaderboard_date_used,
        )
        gate_result = evaluate_gates(
            champion_before=champion_before,
            arm_scores=arm_scores,
            freeze=freeze,
        )
    except Exception as e:  # noqa: BLE001 — gate evaluation must never
        # crash the weekly evaluate run; record as an error outcome
        # (still written, per the liveness posture) and move on.
        logger.exception("Champion-promotion gate evaluation raised")
        error = str(e)

    audit = build_champion_audit(run_date, gate_result, freeze=freeze, error=error)

    pointer_written = None
    if gate_result is not None and gate_result["outcome"] == "promoted" and not freeze:
        pointer_written = write_champion_pointer(
            bucket, gate_result["champion_after"],
            promotion_source="gate_engine", upload=upload, s3_client=s3_client,
        )

    logger.info(
        "producer_champion evaluation: outcome=%s champion_before=%s champion_after=%s "
        "champion_score=%s challenger_score=%s blocked_by=%s leaderboard_date_used=%s",
        audit["outcome"], audit["champion_before"], audit["champion_after"],
        audit.get("champion_score"), audit.get("challenger_score"), audit["blocked_by"],
        audit.get("leaderboard_date_used"),
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
# This module's OWN observability artifact (config#2367) — a per-run
# snapshot of scanner_predictor_direct's realized lift vs the agentic
# baseline, appended to a running history. Under the pre-I2518 HAC engine
# this fed the significance/hysteresis gate directly; under winner-take-all
# it is NO LONGER consumed by the gate (which only needs THIS week's point,
# taken straight from ``e2e_lift`` via ``_score_scanner_predictor_direct``
# above) but is STILL MAINTAINED here for observability/history continuity
# and because config#2452 (the key-collision fix that gave this artifact its
# own distinct key, distinct from crucible-research's
# research/producer_leaderboard/{date}.json) has an open live-verification
# tail expecting this artifact to keep being written every Saturday.
#
# config#2452 (found 2026-07-13, same day as first live run post-merge): this
# key was originally `research/producer_leaderboard/{date}.json` — the SAME
# key crucible-research's `scoring/leaderboard_producers.py` already writes,
# with an incompatible schema. Renamed before any collision occurred.


LEADERBOARD_KEY_TMPL = "research/producer_leaderboard_champion_gate/{date}.json"
_LEADERBOARD_HISTORY_KEEP_WEEKS = 26  # ~6 months of weekly points

RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL = "research/producer_leaderboard/{date}.json"


def leaderboard_entry_from_e2e_lift(e2e_lift: dict | None) -> dict | None:
    """Extract this week's sector-neutral alpha point (scanner_then_predictor's
    OWN realized 21d alpha, already SPY-relative) from the e2e_lift
    diagnostic already computed earlier in the same evaluate run. Returns
    None when the counterfactual is unavailable this week
    (skipped/insufficient_data/error/missing) — an honest "no new point this
    week" rather than fabricating one.

    ``sector_neutral_mean_alpha_21d`` (alpha-engine-config-I2998) is the
    current gate's scanner_predictor_direct score source (see
    ``_score_scanner_predictor_direct``) — the arm's direct lift vs the SPY
    zero-line, gated on THIS field's presence rather than the retired
    ``sn_lift_vs_agentic_cio`` (still carried for observability, may be
    None if the agentic-CIO comparator itself is unavailable that week —
    that no longer blocks this entry from being usable).
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
    sn_alpha = pred.get("sector_neutral_mean_alpha_21d")
    if sn_alpha is None:
        return None
    sn_lift = pred.get("sn_lift_vs_agentic_cio")
    return {
        "sector_neutral_mean_alpha_21d": float(sn_alpha),
        "sn_lift_vs_agentic_cio": float(sn_lift) if sn_lift is not None else None,
        "n_picks": pred.get("n_picks"),
        "n_cycles": cf.get("n_cycles"),
    }


def build_leaderboard_artifact(run_date: str, history: list[dict], new_entry: dict | None) -> dict:
    """Append ``new_entry`` (if any) to ``history`` (oldest-first list of
    ``{"date": ..., "sector_neutral_mean_alpha_21d": ..., "sn_lift_vs_agentic_cio": ...,
    "n_picks": ..., "n_cycles": ...}``), trim to the retention window, and
    return the full artifact to write to
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
    """Reduce a leaderboard artifact to matured cohort count + weekly SN-lift
    series. Retained for API parity / observability (e.g. a future
    diagnostic reusing hac_significance) — no longer consumed by
    evaluate_gates under winner-take-all."""
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
    history."""
    s3 = s3_client or boto3.client("s3")
    from datetime import date as _date, timedelta

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
    evaluate run, BEFORE ``run_weekly_evaluation`` — maintained for
    observability / config#2452 continuity (see module docstring); its
    return value is no longer fed into the gate decision.
    """
    history = read_prior_leaderboard_history(bucket, run_date, s3_client=s3_client) if upload else []
    new_entry = leaderboard_entry_from_e2e_lift(e2e_lift)
    artifact = build_leaderboard_artifact(run_date, history, new_entry)
    if upload:
        try:
            write_leaderboard(bucket, run_date, artifact, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_leaderboard write failed — this observability "
                "artifact will be missing this week (non-fatal to the gate, "
                "which no longer depends on it)",
            )
    return leaderboard_gate_inputs(artifact)


def read_research_producer_leaderboard(bucket: str, run_date: str, s3_client=None) -> dict | None:
    """Read crucible-research's REAL champion/challenger producer leaderboard
    (``scoring/leaderboard_producers.py::build_producer_leaderboard``,
    config#1221/#1223) for ``run_date`` — the evidence source for
    thinktank_coverage's weekly score (see module docstring). Distinct key
    from this module's OWN ``research/producer_leaderboard_champion_gate/
    {date}.json`` (config#2452 collision fix) — this function only READS
    the crucible-research-owned artifact, never writes it.

    Returns None on 404/NoSuchKey (not yet written this week — e.g. before
    the Saturday eval_rolling_mean Lambda step runs, or any week
    crucible-research's build fails) or any other read/parse failure
    (logged) — a missing/malformed leaderboard degrades to a no-contest week
    for thinktank_coverage (``_score_thinktank_coverage``), never a crash
    and never a fabricated score."""
    s3 = s3_client or boto3.client("s3")
    key = RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL.format(date=run_date)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info(
                "No crucible-research producer_leaderboard at s3://%s/%s "
                "(not yet written this week)", bucket, key,
            )
        else:
            logger.warning("crucible-research producer_leaderboard read failed (%s)", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("crucible-research producer_leaderboard read failed (%s)", e)
        return None


def find_latest_research_producer_leaderboard_date(
    bucket: str, run_date: str, s3_client=None,
) -> str | None:
    """List the ``research/producer_leaderboard/`` prefix (crucible-research
    -owned, config#1221/#1223) and return the latest well-formed date <=
    ``run_date`` found among its dated keys, or None if none exist at or
    before ``run_date``.

    alpha-engine-config-I2544 (2026-07-14 ruling): the ASYNC advisory child
    SF that now writes this artifact may not have finished — or may have
    failed outright — by the time this Evaluator-stage gate runs in the
    MAIN weekly SF, so an exact same-day key read is no longer a safe
    assumption. Reading the LATEST AVAILABLE leaderboard <= ``run_date`` is
    the semantically CORRECT read, not a compromise: the gate scores
    realized (matured) outcomes of PRIOR weeks' ``thinktank_coverage``
    selections — a same-day leaderboard could not contain resolved
    outcomes for same-day picks even if it existed on time.

    A single ``list_objects_v2`` call (no pagination) is sufficient: this
    prefix is written at most weekly, so even several years of history
    stays far under the 1000-key single-page ceiling — mirrors
    ``factor_blend_optimizer._read_recent_shadow_archives``'s identical
    single-call reasoning for its own weekly-cadence prefix. Any key under
    the prefix that doesn't match the ``{date}.json`` shape (e.g. a future
    ``latest.json`` sidecar some other consumer adds) is silently skipped,
    never crashes the scan. A list failure (ClientError or otherwise) is
    logged and treated as "nothing available" — degrades to a no-contest
    week downstream, never a crash.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=RESEARCH_PRODUCER_LEADERBOARD_PREFIX)
    except ClientError as e:
        logger.warning(
            "research/producer_leaderboard/ list failed (%s) — treating as "
            "no leaderboard available", e,
        )
        return None
    except Exception as e:  # noqa: BLE001 — list must never crash the gate
        logger.warning(
            "research/producer_leaderboard/ list failed (%s) — treating as "
            "no leaderboard available", e,
        )
        return None

    anchor = date.fromisoformat(run_date)
    best: date | None = None
    for obj in resp.get("Contents") or []:
        m = _RESEARCH_PRODUCER_LEADERBOARD_KEY_RE.match(obj.get("Key", ""))
        if not m:
            continue
        candidate = date.fromisoformat(m.group(1))
        if candidate <= anchor and (best is None or candidate > best):
            best = candidate
    if best is None:
        logger.info(
            "No research/producer_leaderboard/ artifact <= %s found under "
            "s3://%s/%s", run_date, bucket, RESEARCH_PRODUCER_LEADERBOARD_PREFIX,
        )
        return None
    return best.isoformat()


def read_latest_research_producer_leaderboard(
    bucket: str, run_date: str, s3_client=None,
) -> tuple[dict | None, str | None]:
    """Combined list-then-read: find the latest
    ``research/producer_leaderboard/{date}.json`` <= ``run_date``
    (``find_latest_research_producer_leaderboard_date``) and read it
    (``read_research_producer_leaderboard``, reused unchanged — it is
    still the correct exact-date-read primitive once the date to read has
    been selected).

    THE production entry point for thinktank_coverage's evidence as of
    alpha-engine-config-I2544 — supersedes calling
    ``read_research_producer_leaderboard`` directly with ``run_date`` (an
    exact-match read that assumed same-day availability the async advisory
    child SF can no longer guarantee).

    Returns ``(leaderboard_dict, leaderboard_date_used)``: ``(None, None)``
    when no artifact <= ``run_date`` exists yet (or the list itself
    failed); ``(None, None)`` also when a date was found but the S3 read
    then failed (``read_research_producer_leaderboard`` already logs the
    specifics) — never a partial/inconsistent pairing of a leaderboard
    with the wrong date or a date with no leaderboard.
    """
    latest_date = find_latest_research_producer_leaderboard_date(
        bucket, run_date, s3_client=s3_client,
    )
    if latest_date is None:
        return None, None
    leaderboard = read_research_producer_leaderboard(bucket, latest_date, s3_client=s3_client)
    if leaderboard is None:
        return None, None
    return leaderboard, latest_date
