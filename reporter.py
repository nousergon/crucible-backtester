"""
reporter.py — build markdown report + CSV output files, upload to S3.

Consumes output from signal_quality, regime_analysis, score_analysis, attribution,
and (when available) vectorbt portfolio stats. Writes:
    - results/{date}/report.md
    - results/{date}/signal_quality.csv
    - results/{date}/metrics.json
    - s3://alpha-engine-research/backtest/{date}/report.md  (if upload=True)
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _section_data_accumulation(signal_quality: dict, config: dict) -> list[str]:
    """Data accumulation progress — shows how close each optimizer is to activation."""
    db_path = config.get("research_db")
    if not db_path or not os.path.exists(db_path):
        return []

    try:
        import sqlite3
        conn = sqlite3.connect(db_path)

        # score_performance counts
        sp_total = conn.execute("SELECT COUNT(*) FROM score_performance").fetchone()[0]
        sp_with_10d = conn.execute(
            "SELECT COUNT(*) FROM score_performance WHERE beat_spy_10d IS NOT NULL"
        ).fetchone()[0]
        sp_with_30d = conn.execute(
            "SELECT COUNT(*) FROM score_performance WHERE beat_spy_30d IS NOT NULL"
        ).fetchone()[0]
        sp_dates = conn.execute("SELECT COUNT(DISTINCT score_date) FROM score_performance").fetchone()[0]
        sp_earliest = conn.execute("SELECT MIN(score_date) FROM score_performance").fetchone()[0] or "—"
        sp_latest = conn.execute("SELECT MAX(score_date) FROM score_performance").fetchone()[0] or "—"

        # predictor_outcomes counts
        po_total = conn.execute("SELECT COUNT(*) FROM predictor_outcomes").fetchone()[0]
        po_resolved = conn.execute(
            "SELECT COUNT(*) FROM predictor_outcomes WHERE correct_5d IS NOT NULL"
        ).fetchone()[0]
        po_dates = conn.execute(
            "SELECT COUNT(DISTINCT prediction_date) FROM predictor_outcomes"
        ).fetchone()[0]

        conn.close()
    except Exception as e:
        logger.debug("Data accumulation section failed: %s", e)
        return []

    def _bar(current: int, target: int) -> str:
        pct = min(current / target, 1.0) if target > 0 else 0
        filled = int(pct * 10)
        return f"{'█' * filled}{'░' * (10 - filled)} {current}/{target}"

    lines = [
        "## Data Accumulation",
        "",
        f"Score data: **{sp_total}** signals across **{sp_dates}** dates ({sp_earliest} → {sp_latest})",
        f"Predictor data: **{po_total}** predictions across **{po_dates}** dates, **{po_resolved}** resolved",
        "",
        "| Optimizer | Metric | Progress | Status |",
        "|-----------|--------|----------|--------|",
        f"| Signal quality | 10d returns | {_bar(sp_with_10d, 5)} | {'**Active**' if sp_with_10d >= 5 else 'Accumulating'} |",
        f"| Scoring weights | 10d returns | {_bar(sp_with_10d, 43)} | {'**Active**' if sp_with_10d >= 43 else 'Accumulating'} |",
        f"| Attribution | 10d returns | {_bar(sp_with_10d, 50)} | {'**Active**' if sp_with_10d >= 50 else 'Accumulating'} |",
        f"| Predictor veto | Resolved outcomes | {_bar(po_resolved, 20)} | {'**Active**' if po_resolved >= 20 else 'Accumulating'} |",
        f"| Research params | Signals | {_bar(sp_total, 200)} | {'**Active**' if sp_total >= 200 else 'Deferred'} |",
    ]

    # Add accuracy preview if we have any data
    if sp_with_10d > 0:
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            beat = conn.execute(
                "SELECT SUM(beat_spy_10d) FROM score_performance WHERE beat_spy_10d IS NOT NULL"
            ).fetchone()[0] or 0
            acc = beat / sp_with_10d * 100
            lines.append(f"| **10d accuracy** | **Beat SPY** | **{acc:.0f}% ({int(beat)}/{sp_with_10d})** | {'✓ Above 55%' if acc >= 55 else '⚠ Below 55%'} |")
            conn.close()
        except Exception:
            pass

    return lines


def _section_scorecard(grading: dict) -> list[str]:
    """Build the System Report Card section from grading results."""
    lines = ["## System Report Card", ""]

    overall = grading.get("overall", {})
    og = overall.get("grade")
    ol = overall.get("letter", "N/A")
    if og is not None:
        lines.append(f"**OVERALL SYSTEM GRADE: {ol} ({og:.0f}/100)**")
    else:
        lines.append("**OVERALL SYSTEM GRADE: N/A** (insufficient data)")
    lines.append("")

    # Module summary table
    lines.append("| Module | Grade | Score | Key Metric |")
    lines.append("|--------|-------|-------|------------|")

    for module_key, label in [("research", "Research"), ("predictor", "Predictor"), ("executor", "Executor")]:
        mod = grading.get(module_key, {})
        mg = mod.get("grade")
        ml = mod.get("letter", "N/A")
        # Pick a single key metric for the summary row
        key_metric = _scorecard_key_metric(mod)
        if mg is not None:
            lines.append(f"| **{label}** | **{ml}** | {mg:.0f} | {key_metric} |")
        else:
            lines.append(f"| **{label}** | N/A | — | insufficient data |")

    lines.append("")

    # Research detail
    research = grading.get("research", {})
    r_comps = research.get("components", {})
    lines.append("### Research Components")
    lines.append("")
    lines.append("| Component | Grade | Score | Detail |")
    lines.append("|-----------|-------|-------|--------|")

    for comp_key, comp_label in [
        ("scanner", "Scanner"),
        ("macro_agent", "Macro Agent"),
        ("cio", "CIO"),
        ("composite_scoring", "Composite Scoring"),
        ("calibration_diagnostics", "Calibration"),
    ]:
        c = r_comps.get(comp_key)
        if c is None:
            continue  # component not wired through in this run
        _append_component_row(lines, comp_label, c)

    # Sector teams
    teams = r_comps.get("sector_teams", [])
    avg = r_comps.get("sector_teams_avg", {})
    if teams:
        avg_g = avg.get("grade")
        avg_l = avg.get("letter", "N/A")
        if avg_g is not None:
            lines.append(f"| **Sector Teams (avg)** | **{avg_l}** | {avg_g:.0f} | |")
        for t in teams:
            tid = t.get("team_id", "?").replace("_", " ").title()
            _append_component_row(lines, f"  {tid}", t)

    lines.append("")

    # Predictor detail
    predictor = grading.get("predictor", {})
    p_comps = predictor.get("components", {})
    lines.append("### Predictor Components")
    lines.append("")
    lines.append("| Component | Grade | Score | Detail |")
    lines.append("|-----------|-------|-------|--------|")
    for comp_key, comp_label in [("meta_model", "Meta Model"), ("veto_gate", "Veto Gate")]:
        c = p_comps.get(comp_key, {})
        _append_component_row(lines, comp_label, c)
    lines.append("")

    # Executor detail
    executor = grading.get("executor", {})
    e_comps = executor.get("components", {})
    lines.append("### Executor Components")
    lines.append("")
    lines.append("| Component | Grade | Score | Detail |")
    lines.append("|-----------|-------|-------|--------|")
    for comp_key, comp_label in [
        ("entry_triggers", "Entry Triggers"),
        ("risk_guard", "Risk Guard"),
        ("exit_rules", "Exit Rules"),
        ("position_sizing", "Position Sizing"),
        ("portfolio", "Portfolio"),
        ("excursion", "MFE/MAE (Excursion)"),
        ("action_entropy", "Action Entropy"),
    ]:
        c = e_comps.get(comp_key)
        if c is None:
            continue  # component not wired through in this run
        _append_component_row(lines, comp_label, c)
    lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _append_component_row(lines: list[str], label: str, comp: dict):
    """Append a single component row to the scorecard table."""
    g = comp.get("grade")
    l = comp.get("letter", "N/A")
    detail = comp.get("detail", {})
    reason = comp.get("reason")

    if g is not None:
        detail_str = ", ".join(f"{k}: {v}" for k, v in detail.items()
                               if k != "per_trigger" and not isinstance(v, list))
        lines.append(f"| {label} | {l} | {g:.0f} | {detail_str} |")
    else:
        lines.append(f"| {label} | N/A | — | {reason or 'insufficient data'} |")


def _section_skill_vs_beta(grading: dict) -> list[str]:
    """Skill vs. Beta panel — per-team skilled-risk-taking metrics + ECE.

    Renders only when the evaluator-revamp metrics are wired through.
    For each sector team, surfaces the four skill-composite signals that
    answer "given the risk you took, did you outperform the dumb version?"
    plus the system-wide calibration row.
    """
    research = grading.get("research", {})
    r_comps = research.get("components", {})
    teams = r_comps.get("sector_teams", []) or []

    # Detect skill-composite teams: their detail has IC + at least one of
    # alpha_vs_ew_high_vol / alpha_vs_beta_spy. If no team has those
    # fields, the legacy lift-based path was used and this section
    # has nothing skill-specific to render.
    skill_teams = [
        t for t in teams
        if "ic" in (t.get("detail") or {})
        and any(k in (t.get("detail") or {})
                for k in ("alpha_vs_ew_high_vol", "alpha_vs_beta_spy"))
    ]
    calibration = r_comps.get("calibration_diagnostics")

    if not skill_teams and not calibration:
        return []

    lines = ["## Skill vs. Beta", ""]
    lines.append(
        "_Risk-matched alpha + decision-quality diagnostics. Answers: "
        "given the risk taken, did the agents outperform the dumb version "
        "of taking that risk?_"
    )
    lines.append("")

    if skill_teams:
        lines.append("### Per-Team Skill Composite")
        lines.append("")
        lines.append(
            "| Team | Grade | IC | Hit% | W/L | MFE/MAE | α vs EW-high-vol | α vs β-SPY |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for t in skill_teams:
            tid = t.get("team_id", "?").replace("_", " ").title()
            d = t.get("detail") or {}
            grade = t.get("grade")
            grade_str = f"{t.get('letter', 'N/A')} ({grade:.0f})" if grade is not None else "N/A"
            lines.append(
                f"| {tid} | {grade_str} "
                f"| {d.get('ic', '—')} "
                f"| {d.get('hit_rate', '—')} "
                f"| {d.get('win_loss_ratio', '—')} "
                f"| {d.get('mfe_mae_ratio', '—')} "
                f"| {d.get('alpha_vs_ew_high_vol', '—')} "
                f"| {d.get('alpha_vs_beta_spy', '—')} |"
            )
        lines.append("")

    if calibration:
        d = calibration.get("detail") or {}
        ece = d.get("ece", "—")
        n = d.get("n", "—")
        quality = d.get("quality", "—")
        grade = calibration.get("grade")
        grade_str = (
            f"{calibration.get('letter', 'N/A')} ({grade:.0f})"
            if grade is not None else "N/A"
        )
        lines.append("### Calibration (Decision Quality)")
        lines.append("")
        lines.append(
            f"_When agents say {{conviction}}%, do those picks actually win {{conviction}}%?_"
        )
        lines.append("")
        lines.append(
            f"- ECE: **{ece}** (lower = better calibrated)"
        )
        lines.append(f"- Quality label: **{quality}**")
        lines.append(f"- Grade: {grade_str}")
        lines.append(f"- n samples: {n}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _scorecard_key_metric(mod: dict) -> str:
    """Extract a single key metric string for the module summary row."""
    comps = mod.get("components", {})

    # Research: show composite scoring accuracy
    cs = comps.get("composite_scoring", {})
    cs_detail = cs.get("detail", {})
    if "accuracy_10d" in cs_detail:
        return f"10d accuracy: {cs_detail['accuracy_10d']}"

    # Predictor: show IC
    meta = comps.get("meta_model", {})
    meta_detail = meta.get("detail", {})
    if "rank_ic" in meta_detail:
        return f"IC: {meta_detail['rank_ic']}"

    # Executor: show portfolio detail
    pf = comps.get("portfolio", {})
    pf_detail = pf.get("detail", {})
    if "sharpe" in pf_detail:
        return f"Sharpe: {pf_detail['sharpe']}"
    if "accuracy_10d" in pf_detail:
        return f"accuracy: {pf_detail['accuracy_10d']}"

    return ""


def _section_pipeline_health(health: dict) -> list[str]:
    """Build Pipeline Health section from collected metadata."""
    lines = ["## Pipeline Health", ""]

    # Data freshness
    if health.get("staleness_warning"):
        lines.append(f"> {health['staleness_warning']}")
        lines.append("")

    # Research DB status
    db_status = health.get("db_pull_status", "unknown")
    if db_status == "ok":
        lines.append("- Research DB: Loaded")
    elif db_status == "failed":
        lines.append("- Research DB: **MISSING** — signal quality analysis skipped")
    else:
        lines.append(f"- Research DB: {db_status}")

    # Simulation coverage. Render `(degraded)` when the count fields
    # aren't populated (e.g. evaluator-only runs that didn't re-execute
    # backtest, where coverage is taken from prior artifacts but the
    # sim/exp counts weren't carried through). Pre-2026-05-07 the
    # fallback emitted "?/? dates (100%)" which read as a missing-data
    # bug rather than a known-degraded mode.
    if health.get("coverage") is not None:
        cov = health["coverage"]
        sim = health.get("dates_simulated")
        exp = health.get("dates_expected")
        if sim is not None and exp is not None:
            lines.append(f"- Simulation coverage: {sim}/{exp} dates ({cov:.0%})")
        else:
            lines.append(f"- Simulation coverage: {cov:.0%} (counts not carried through this run)")

    # Skip reasons
    if health.get("skip_reasons"):
        lines.append(f"- Skipped dates: {health['skip_reasons']}")

    # Price data quality
    if health.get("price_gap_warnings"):
        n = len(health["price_gap_warnings"])
        lines.append(f"- Price gaps (>5 days): {n} tickers")
    if health.get("unfilled_gaps"):
        n = len(health["unfilled_gaps"])
        lines.append(f"- Unfilled gaps after ffill: {n} tickers")

    # Predictor feature skip reasons
    if health.get("feature_skip_reasons"):
        lines.append(f"- Predictor feature skips: {health['feature_skip_reasons']}")

    lines.append("")
    return lines


def _section_decision_capture_coverage(coverage: dict) -> list[str]:
    """Build Decision Capture Coverage section.

    Phase 2 transparency-inventory — answers the *agent decisions* row's
    coverage question: of the 8 canonical agents that run on every
    Saturday SF, how many emitted ≥1 captured artifact?
    """
    lines = ["## Decision Capture Coverage", ""]

    if coverage.get("status") == "no_recent_sf_run":
        lines.append(f"> {coverage.get('reason', 'no Saturday SF run found')}")
        lines.append("")
        return lines
    if coverage.get("status") != "ok":
        err = coverage.get("error", "unknown error")
        lines.append(f"> Coverage computation skipped: {err}")
        lines.append("")
        return lines

    sf_date = coverage.get("most_recent_sf_date", "?")
    pct = coverage.get("coverage_pct", 0.0)
    n_present = coverage.get("n_canonical_present", 0)
    n_expected = coverage.get("n_canonical_expected", 0)
    flag = "✅" if pct >= 99.0 else ("⚠️" if pct >= 75.0 else "🔴")

    lines.append(
        f"- Most-recent SF: **{sf_date}** — {flag} **{pct:.1f}%** "
        f"({n_present}/{n_expected} canonical agents)"
    )

    # Per-agent breakdown — show only missing ones inline; full set in JSON artifact.
    per_agent = coverage.get("per_agent", {})
    missing = [a for a, v in per_agent.items() if not v.get("present")]
    if missing:
        lines.append(f"- **Missing**: {', '.join(missing)}")

    thesis = coverage.get("thesis_update_count", 0)
    if thesis > 0:
        lines.append(f"- thesis_update captures: {thesis} (variable; not in coverage %)")

    uncategorized = coverage.get("uncategorized_agents", []) or []
    if uncategorized:
        lines.append(f"- Uncategorized agents: {', '.join(uncategorized)}")

    rolling = coverage.get("rolling", {})
    n_sat = rolling.get("n_saturdays_with_data", 0)
    if n_sat >= 2:
        lines.append(
            f"- Rolling ({n_sat}-Saturday avg): "
            f"{rolling.get('coverage_pct_mean', 0):.1f}% "
            f"(min {rolling.get('coverage_pct_min', 0):.1f}, "
            f"max {rolling.get('coverage_pct_max', 0):.1f})"
        )

    lines.append("")
    return lines


def _section_provenance_grounding(grounding: dict) -> list[str]:
    """Build Provenance Grounding section.

    Fourth leg of agent-justification stack — measures per-agent tool-call
    + input-trace coverage on captured artifacts. Lives next to
    decision_capture_coverage since both are pre-analytics observability
    surfaces (one counts presence, the other counts tool-equipped quality).
    """
    lines = ["## Provenance Grounding", ""]

    if grounding.get("status") == "no_recent_sf_run":
        lines.append(f"> {grounding.get('reason', 'no Saturday SF run found')}")
        lines.append("")
        return lines
    if grounding.get("status") != "ok":
        err = grounding.get("error", "unknown error")
        lines.append(f"> Provenance computation skipped: {err}")
        lines.append("")
        return lines

    sf_date = grounding.get("most_recent_sf_date", "?")
    n_artifacts = grounding.get("n_total_artifacts_read", 0)
    lines.append(f"- Most-recent SF: **{sf_date}** — {n_artifacts} artifacts read")

    alarms = grounding.get("tool_equipped_alarms", []) or []
    if alarms:
        lines.append(f"- ⚠️ Tool-equipped alarms: {len(alarms)} agent(s)")
        for a in alarms[:5]:
            lines.append(f"  - {a}")

    rolling = grounding.get("rolling", {})
    n_sat = rolling.get("n_saturdays_with_data", 0)
    if n_sat >= 2:
        lines.append(f"- Rolling window: {n_sat} Saturday(s) with provenance data")

    lines.append("")
    return lines


def _section_agent_justification(summary: dict) -> list[str]:
    """Build the Agent Justification Stack section.

    Aggregates the four eval-judge / agent-justification triple sources
    (rubric scores, rationale clustering, replay concordance, counterfactual
    rule fits). Pre-2026-05-07 SF reorder these results landed in S3 only
    AFTER Evaluator's email was generated; the reorder + this section
    surface them together for the operator's weekly review.

    Each source renders one summary line. Per-agent detail lives at the
    S3 paths called out in the section comment — kept out of the email
    to avoid noise (an 8-agent x 4-source matrix would dominate).
    """
    lines = ["## Agent Justification Stack", ""]

    judge = summary.get("judge", {})
    if judge.get("status") == "ok":
        lines.append(
            f"- **Judge** (rubric scores) — {judge['n_scored']}/{judge['n_agents']} "
            f"agents scored, mean **{judge['mean_score']:.2f}** "
            f"(min {judge['min_score']:.2f}, max {judge['max_score']:.2f}) "
            f"— SF: {judge.get('most_recent_sf_date', '?')}"
        )
    else:
        lines.append(
            f"- **Judge** — {judge.get('status', 'unknown')} "
            f"(no rubric data within 14d of run_date)"
        )

    clust = summary.get("clustering", {})
    if clust.get("status") == "ok" and clust.get("mean_top3_concentration") is not None:
        lines.append(
            f"- **Clustering** — {clust['n_agents']} agents, mean top-3 "
            f"concentration **{clust['mean_top3_concentration']:.2f}** "
            f"(week: {clust.get('most_recent_week', '?')})"
        )
    else:
        lines.append("- **Clustering** — no recent rationale-cluster data")

    cf = summary.get("counterfactual", {})
    if cf.get("status") == "ok" and cf.get("mean_match_rate") is not None:
        agents_str = ", ".join(cf.get("agents", [])) or "none"
        lines.append(
            f"- **Counterfactual** — {cf['n_agents']} agents fit, mean DT "
            f"match rate **{cf['mean_match_rate']:.1%}** "
            f"(agents: {agents_str}; week: {cf.get('most_recent_week', '?')})"
        )
    else:
        lines.append("- **Counterfactual** — no recent rule-fit data")

    conc = summary.get("concordance", {})
    if conc.get("status") == "ok":
        lines.append(
            f"- **Concordance** — {conc['n_target_models']} target model(s) "
            f"summarized (SF: {conc.get('most_recent_sf_date', '?')})"
        )
    else:
        # Concordance Lambda may not have written summaries yet — section
        # surfaces the gap rather than silently omitting it.
        lines.append(
            f"- **Concordance** — {conc.get('status', 'unknown')} "
            f"(_replay_summary/ has no entries within lookback window)"
        )

    lines.append("")
    return lines


def build_report(
    run_date: str,
    signal_quality: dict,
    regime_analysis: list[dict],
    score_analysis: list[dict],
    attribution: dict,
    portfolio_stats: dict | None = None,
    sweep_df=None,
    weight_result: dict | None = None,
    config: dict | None = None,
    predictor_stats: dict | None = None,
    predictor_sweep_df=None,
    veto_result: dict | None = None,
    executor_rec: dict | None = None,
    regression_result: dict | None = None,
    pipeline_health: dict | None = None,
    e2e_lift: dict | None = None,
    trigger_scorecard: dict | None = None,
    alpha_dist: dict | None = None,
    score_calibration: dict | None = None,
    veto_value: dict | None = None,
    shadow_book: dict | None = None,
    exit_timing: dict | None = None,
    macro_eval: dict | None = None,
    decision_capture_coverage: dict | None = None,
    provenance_grounding: dict | None = None,
    agent_justification: dict | None = None,
    trigger_opt: dict | None = None,
    predictor_sizing: dict | None = None,
    scanner_opt: dict | None = None,
    team_opt: dict | None = None,
    cio_opt: dict | None = None,
    sizing_ab: dict | None = None,
    grading: dict | None = None,
    confusion_matrix: dict | None = None,
    post_trade: dict | None = None,
    monte_carlo: dict | None = None,
) -> str:
    """
    Build a markdown report string from analysis results.

    Returns the markdown string (also written to disk by save()).
    """
    lines = [
        f"# Alpha Engine Backtest Report",
        f"_Run date: {run_date}_",
        "",
        "---",
        "",
    ]

    # Data accumulation tracker
    lines += _section_data_accumulation(signal_quality, config or {})
    lines += [""]

    # Pipeline health (data freshness, coverage, gaps)
    if pipeline_health:
        lines += _section_pipeline_health(pipeline_health)
        lines += [""]

    # Decision capture coverage (Phase 2 transparency-inventory).
    # Lives next to pipeline_health since it's the same flavor — both
    # are pre-analytics observability surfaces, not analysis findings.
    if decision_capture_coverage:
        lines += _section_decision_capture_coverage(decision_capture_coverage)

    if provenance_grounding:
        lines += _section_provenance_grounding(provenance_grounding)

    if agent_justification:
        lines += _section_agent_justification(agent_justification)

    # System Report Card (component grades)
    if grading and grading.get("status") in ("ok", "partial"):
        lines += _section_scorecard(grading)
        # Skill vs. Beta panel — surfaces the per-team risk-matched
        # alpha numbers + ECE detail when the evaluator-revamp metrics
        # are wired through. Falls back to no-op when team_metrics
        # isn't populated (PR 4 wiring).
        skill_lines = _section_skill_vs_beta(grading)
        if skill_lines:
            lines += skill_lines

    # What Changed This Week (promotion decisions, twin sim, regression)
    lines += _section_what_changed(
        weight_result=weight_result,
        veto_result=veto_result,
        executor_rec=executor_rec,
        regression_result=regression_result,
    )
    lines += [""]

    # Signal quality + dependent sections (Score threshold / Regime /
    # Sub-score attribution) all derive from research_db. When the
    # caller signals signal_quality.status == "skipped" — typically
    # because backtest.py's simulation email path doesn't load
    # research.db (that's evaluator territory) — suppress all four
    # sections rather than render misleading n=0 tables / "Deferred
    # until Week 4+" placeholders that imply data shortage when really
    # this email kind doesn't compute them. Closes Items D + L1907 of
    # the 2026-05-09 P2 ROADMAP entry.
    sq_skipped = (
        isinstance(signal_quality, dict)
        and signal_quality.get("status") == "skipped"
    )
    if not sq_skipped:
        # Signal quality summary
        lines += _section_signal_quality(signal_quality)
        lines += [""]

        # Score threshold analysis
        lines += _section_score_analysis(score_analysis)
        lines += [""]

        # Regime breakdown
        lines += _section_regime(regime_analysis)
        lines += [""]

        # Attribution
        lines += _section_attribution(attribution)
        lines += [""]

    # Alpha magnitude distribution
    if alpha_dist and alpha_dist.get("status") == "ok":
        lines += _section_alpha_distribution(alpha_dist)
        lines += [""]

    # Score calibration
    if score_calibration and score_calibration.get("status") == "ok":
        lines += _section_score_calibration(score_calibration)
        lines += [""]

    # End-to-end pipeline lift
    if e2e_lift and e2e_lift.get("status") == "ok":
        from analysis.end_to_end import format_lift_report
        lines += format_lift_report(e2e_lift)
        lines += [""]

    # Macro multiplier evaluation
    if macro_eval and macro_eval.get("status") == "ok":
        lines += _section_macro_eval(macro_eval)
        lines += [""]

    # ── Phase 4: Self-adjustment mechanisms ──────────────────────────────
    phase4_sections = []
    if trigger_opt and trigger_opt.get("status") == "ok":
        phase4_sections += _section_trigger_opt(trigger_opt)
    if predictor_sizing and predictor_sizing.get("status") == "ok":
        phase4_sections += _section_predictor_sizing(predictor_sizing)
    if scanner_opt:
        phase4_sections += _section_scanner_opt(scanner_opt)
    if team_opt:
        phase4_sections += _section_team_opt(team_opt)
    if cio_opt and cio_opt.get("status") == "ok":
        phase4_sections += _section_cio_opt(cio_opt)
    if sizing_ab and sizing_ab.get("status") == "ok":
        phase4_sections += _section_sizing_ab(sizing_ab)

    if phase4_sections:
        lines += ["", "---", "", "# Phase 4: Self-Adjustment Mechanisms", ""]
        lines += phase4_sections
        lines += [""]

    # Portfolio simulation (Mode 2)
    if portfolio_stats:
        lines += _section_portfolio(portfolio_stats)
        lines += [""]

    # Param sweep
    if sweep_df is not None and not sweep_df.empty:
        lines += _section_param_sweep(sweep_df)
        lines += [""]

    # Weight recommendation
    if weight_result:
        lines += _section_weight_recommendation(weight_result)
        lines += [""]

    # Veto threshold analysis
    if veto_result:
        lines += _section_veto_analysis(veto_result)
        lines += [""]

    # Executor parameter recommendations
    if executor_rec:
        lines += _section_executor_recommendations(executor_rec)
        lines += [""]

    # Entry trigger scorecard
    if trigger_scorecard and trigger_scorecard.get("status") == "ok":
        lines += _section_trigger_scorecard(trigger_scorecard)
        lines += [""]

    # Net veto value in dollars
    if veto_value and veto_value.get("status") == "ok":
        lines += _section_veto_value(veto_value)
        lines += [""]

    # Shadow book analysis
    if shadow_book and shadow_book.get("status") == "ok":
        lines += _section_shadow_book(shadow_book)
        lines += [""]

    # Exit timing analysis (MFE/MAE)
    if exit_timing and exit_timing.get("status") == "ok":
        lines += _section_exit_timing(exit_timing)
        lines += [""]

    # Predictor confusion matrix
    if confusion_matrix and confusion_matrix.get("status") == "ok":
        lines += _section_confusion_matrix(confusion_matrix)
        lines += [""]

    # Predictor-only backtest (2y historical)
    if predictor_stats:
        lines += _section_predictor_backtest(predictor_stats)
        lines += [""]

    # Predictor param sweep
    if predictor_sweep_df is not None and not predictor_sweep_df.empty:
        lines += _section_param_sweep_predictor(predictor_sweep_df)
        lines += [""]

    lines += [
        "---",
        f"_Generated by alpha-engine-backtester — {run_date}_",
    ]

    return "\n".join(lines)


def save(
    report_md: str,
    signal_quality: dict,
    score_analysis: list[dict],
    sweep_df=None,
    attribution: dict | None = None,
    run_date: str | None = None,
    results_dir: str = "results",
    grading: dict | None = None,
    trigger_scorecard: dict | None = None,
    shadow_book: dict | None = None,
    exit_timing: dict | None = None,
    e2e_lift: dict | None = None,
    veto_result: dict | None = None,
    confusion_matrix: dict | None = None,
    post_trade: dict | None = None,
    monte_carlo: dict | None = None,
    decision_capture_coverage: dict | None = None,
    provenance_grounding: dict | None = None,
    agent_justification: dict | None = None,
) -> Path:
    """
    Write report.md, signal_quality.csv, and metrics.json to results/{date}/.

    Returns the output directory path.
    """
    if run_date is None:
        run_date = date.today().isoformat()

    out_dir = Path(results_dir) / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    (out_dir / "report.md").write_text(report_md)
    logger.info("Wrote %s", out_dir / "report.md")

    # Signal quality CSV
    if score_analysis:
        df = pd.DataFrame(score_analysis)
        df.to_csv(out_dir / "signal_quality.csv", index=False)
        logger.info("Wrote %s", out_dir / "signal_quality.csv")

    # Metrics JSON (overall summary + report card)
    overall = signal_quality.get("overall", {})
    metrics = {
        "run_date": run_date,
        "status": signal_quality.get("status"),
        **overall,
    }
    if grading and grading.get("status") in ("ok", "partial"):
        metrics["report_card"] = grading
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Wrote %s", out_dir / "metrics.json")

    # Param sweep CSV
    if sweep_df is not None and not sweep_df.empty:
        sweep_df.to_csv(out_dir / "param_sweep.csv", index=False)
        logger.info("Wrote %s", out_dir / "param_sweep.csv")

    # Attribution JSON
    if attribution and attribution.get("status") == "ok":
        (out_dir / "attribution.json").write_text(json.dumps(attribution, indent=2, default=str))
        logger.info("Wrote %s", out_dir / "attribution.json")

    # Structured analysis files for dashboard consumption
    for filename, data in [
        ("grading.json", grading),
        ("trigger_scorecard.json", trigger_scorecard),
        ("shadow_book.json", shadow_book),
        ("exit_timing.json", exit_timing),
        ("e2e_lift.json", e2e_lift),
        ("veto_analysis.json", veto_result),
        ("confusion_matrix.json", confusion_matrix),
        ("decision_capture_coverage.json", decision_capture_coverage),
        ("provenance_grounding.json", provenance_grounding),
        ("agent_justification.json", agent_justification),
    ]:
        if data and data.get("status") in ("ok", "partial", "insufficient_lift"):
            (out_dir / filename).write_text(json.dumps(data, indent=2, default=str))
            logger.info("Wrote %s", out_dir / filename)

    return out_dir


def upload_to_s3(
    local_dir: Path,
    bucket: str,
    prefix: str = "backtest",
    run_date: str | None = None,
) -> None:
    """
    Upload all files in local_dir to s3://{bucket}/{prefix}/{run_date}/.
    """
    if run_date is None:
        run_date = date.today().isoformat()

    s3 = boto3.client("s3")
    for file_path in local_dir.iterdir():
        key = f"{prefix}/{run_date}/{file_path.name}"
        try:
            s3.upload_file(str(file_path), bucket, key)
            logger.info("Uploaded s3://%s/%s", bucket, key)
        except ClientError as e:
            logger.error("Failed to upload %s: %s", file_path, e)
            raise


# --- Section builders ---

def _section_signal_quality(sq: dict) -> list[str]:
    lines = ["## Mode 1 — Signal Quality"]
    status = sq.get("status", "unknown")

    if status == "insufficient_data":
        lines += [
            "",
            f"> **Insufficient data.** "
            f"{sq.get('rows_10d_populated', 0)} rows with 10d returns populated "
            f"(need {sq.get('rows_needed', 10)}). "
            "Results will be available after Week 4 (~200 populated rows).",
        ]
        return lines

    overall = sq.get("overall", {})
    acc_5d = overall.get("accuracy_5d")
    acc_10d = overall.get("accuracy_10d")
    acc_30d = overall.get("accuracy_30d")
    n_5d = overall.get("n_5d", 0)
    n_10d = overall.get("n_10d", 0)
    n_30d = overall.get("n_30d", 0)

    lines += [
        "",
        f"| Metric | 5d | 10d | 30d |",
        f"|--------|-----|-----|-----|",
        f"| Accuracy vs SPY | {_pct(acc_5d)} (n={n_5d}) | {_pct(acc_10d)} (n={n_10d}) | {_pct(acc_30d)} (n={n_30d}) |",
        f"| Avg alpha | {_alpha_pp(overall.get('avg_alpha_5d'))} | {_alpha_pp(overall.get('avg_alpha_10d'))} | {_alpha_pp(overall.get('avg_alpha_30d'))} |",
        "",
        "> 50% = coin flip. 55%+ over 30+ signals suggests real alpha.",
    ]

    buckets = sq.get("by_score_bucket", [])
    if buckets:
        lines += ["", "### By score bucket", ""]
        lines += ["| Bucket | Acc 5d | Acc 10d | Acc 30d | Avg α 10d | n | FDR |"]
        lines += ["|--------|--------|---------|---------|-----------|---|-----|"]
        has_exploratory = False
        has_fdr_exploratory = False
        for b in buckets:
            star = ""
            if b.get("exploratory"):
                star = "*"
                has_exploratory = True
            fdr_tag = ""
            if b.get("fdr_exploratory"):
                fdr_tag = "†"
                has_fdr_exploratory = True
            lines.append(
                f"| {b.get('bucket')} | {_pct(b.get('accuracy_5d'))}{star} | "
                f"{_pct(b.get('accuracy_10d'))}{star} | "
                f"{_pct(b.get('accuracy_30d'))}{star} | {_alpha_pp(b.get('avg_alpha_10d'))}{star} | "
                f"{b.get('n_10d', 0)} | {fdr_tag} |"
            )
        if has_exploratory:
            lines += ["", "\\* exploratory — fewer than 20 samples, treat with caution"]
        if has_fdr_exploratory:
            lines += ["", "† not FDR-significant (Benjamini-Hochberg, α=0.05) — accuracy may not differ from coin flip"]

    sectors = sq.get("by_sector", [])
    if sectors:
        lines += ["", "### By sector", ""]
        lines += ["| Sector | Acc 5d | Acc 10d | Acc 30d | Avg α 10d | n |"]
        lines += ["|--------|--------|---------|---------|-----------|---|"]
        for s in sectors:
            lines.append(
                f"| {s.get('sector', '?')} | {_pct(s.get('accuracy_5d'))} | "
                f"{_pct(s.get('accuracy_10d'))} | "
                f"{_pct(s.get('accuracy_30d'))} | {_alpha_pp(s.get('avg_alpha_10d'))} | "
                f"{s.get('n_10d', 0)} |"
            )

    return lines


def _section_score_analysis(rows: list[dict]) -> list[str]:
    lines = ["## Score threshold analysis"]
    if not rows:
        lines += ["", "> Deferred until Week 4+ (insufficient data)."]
        return lines

    lines += ["", "| Min score | Acc 5d | Acc 10d | Acc 30d | n |"]
    lines += ["|-----------|--------|---------|---------|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('threshold')} | {_pct(r.get('accuracy_5d'))} | "
            f"{_pct(r.get('accuracy_10d'))} | "
            f"{_pct(r.get('accuracy_30d'))} | {r.get('n_10d', 0)} |"
        )
    return lines


def _section_regime(rows: list[dict]) -> list[str]:
    lines = ["## Regime analysis"]
    if not rows:
        lines += ["", "> Deferred until Week 4+ (insufficient data)."]
        return lines

    lines += ["", "| Regime | Acc 5d | Acc 10d | Acc 30d | n |"]
    lines += ["|--------|--------|---------|---------|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('market_regime')} | {_pct(r.get('accuracy_5d'))} | "
            f"{_pct(r.get('accuracy_10d'))} | "
            f"{_pct(r.get('accuracy_30d'))} | {r.get('n_10d', 0)} |"
        )
    return lines


def _section_attribution(attr: dict) -> list[str]:
    lines = ["## Sub-score attribution"]
    status = attr.get("status", "unknown")

    if status != "ok":
        lines += [
            "",
            f"> **Deferred.** {attr.get('note', 'Insufficient data.')}",
        ]
        return lines

    lines += [
        "",
        f"Analyzed {attr.get('rows_analyzed', 0)} signals.",
        "",
        "### Correlation with beat_spy_10d",
        "",
        "| Sub-score | Corr (10d) | Corr (30d) | FDR sig (10d) | FDR sig (30d) |",
        "|-----------|------------|------------|---------------|---------------|",
    ]
    for label, corrs in attr.get("correlations", {}).items():
        c10 = corrs.get("beat_spy_10d")
        c30 = corrs.get("beat_spy_30d")
        fdr_10 = "Yes" if corrs.get("beat_spy_10d_fdr_significant") else "No"
        fdr_30 = "Yes" if corrs.get("beat_spy_30d_fdr_significant") else "No"
        lines.append(f"| {label} | {_fmt(c10)} | {_fmt(c30)} | {fdr_10} | {fdr_30} |")

    ranking_10d = attr.get("ranking_10d", [])
    if ranking_10d:
        lines += ["", f"**Strongest predictor (10d):** {ranking_10d[0]}"]

    fdr_ns = attr.get("fdr_non_significant")
    if fdr_ns:
        lines += ["", f"> FDR non-significant: {', '.join(fdr_ns)}"]

    lines += ["", f"> {attr.get('note', '')}"]
    return lines


def _section_portfolio(stats: dict) -> list[str]:
    status = stats.get("status")
    if status and status != "ok":
        note = stats.get("note") or stats.get("error", "No details available.")
        # Truncate long error messages (e.g. Plotly property dumps)
        if len(note) > 200:
            note = note[:200] + "..."
        return [
            "## Mode 2 — Portfolio simulation",
            "",
            f"> **Skipped.** {note}",
        ]
    lines = [
        "## Mode 2 — Portfolio simulation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total return | {_pct(stats.get('total_return'))} |",
        f"| Sharpe ratio | {_fmt(stats.get('sharpe_ratio'))} |",
        f"| Max drawdown | {_pct(stats.get('max_drawdown'))} |",
        f"| Calmar ratio | {_fmt(stats.get('calmar_ratio'))} |",
        f"| Total trades | {stats.get('total_trades', 'N/A')} |",
        f"| Win rate | {_pct(stats.get('win_rate'))} |",
    ]
    sim_assumptions = stats.get("simulation_assumptions")
    if sim_assumptions:
        lines.append(f"| Simulation | {sim_assumptions} |")
    price_gaps = stats.get("price_gap_warnings")
    if price_gaps and isinstance(price_gaps, dict):
        n_gaps = len(price_gaps)
        worst = sorted(price_gaps.items(), key=lambda x: -x[1])[:5]
        worst_str = ", ".join(f"{t} ({d}d)" for t, d in worst)
        lines += ["", f"> **Price gaps (>5 days):** {n_gaps} tickers — worst: {worst_str}"]
    elif price_gaps:
        lines += ["", f"> **Price gap warnings:** {price_gaps}"]
    staleness = stats.get("staleness_warning")
    if staleness:
        lines += [f"> **{staleness}**"]
    return lines


def _section_what_changed(
    weight_result: dict | None = None,
    veto_result: dict | None = None,
    executor_rec: dict | None = None,
    regression_result: dict | None = None,
) -> list[str]:
    """Build the 'What Changed This Week' section — promotion decisions + twin sim."""

    def _decision_for(label: str, result: dict | None) -> tuple | None:
        """Classify optimizer result into (label, decision, reason).

        Returns None when ``result is None`` so the caller can omit the row
        entirely. ``result is None`` indicates the optimizer wasn't computed
        on this code path — e.g. the simulation email (`backtest.py`) does
        not run the weight or veto optimizers (those run in `evaluate.py`'s
        evaluator email). Rendering the row as "NOT RUN" was misleading
        because it implied the optimizer was skipped within *this* run when
        really it belongs to a different email entirely. When the optimizer
        is skipped within its own run (e.g. research_db missing), the path
        below produces a non-None result dict with status="skipped" and
        renders as "SKIPPED" with the skip reason.
        """
        if result is None:
            return None
        status = result.get("status", "unknown")
        apply = result.get("apply_result", {})
        if apply.get("applied"):
            return (label, "PROMOTED", "guardrails passed — config updated in S3")
        elif apply.get("reason"):
            reason = apply["reason"]
            if "frozen" in reason:
                return (label, "FROZEN", reason)
            return (label, "REJECTED", reason)
        elif status == "ok":
            return (label, "EVALUATED", "no promotion attempted")
        elif status == "error":
            return (label, "ERROR", result.get("error", "unknown error")[:80])
        elif status in ("insufficient_data", "no_subscores"):
            note = result.get("note", status)
            return (label, "DEFERRED", note[:80] if isinstance(note, str) else str(note))
        elif status == "skipped":
            return (label, "SKIPPED", result.get("note", "skipped"))
        else:
            return (label, "DEFERRED", f"status={status}")

    decisions = [
        d for d in (
            _decision_for("Scoring weights", weight_result),
            _decision_for("Executor params", executor_rec),
            _decision_for("Veto threshold", veto_result),
        )
        if d is not None
    ]

    has_twin_sim = executor_rec and executor_rec.get("twin_sim", {}).get("status") == "ok"
    has_regression = regression_result and regression_result.get("checked")

    lines = ["## What Changed This Week", ""]

    # Promotion decisions table — only shown when at least one optimizer
    # produced a result on this code path. The simulation email
    # (`backtest.py`) only runs the executor optimizer, so weight + veto
    # rows are filtered above; the evaluator email (`evaluate.py`) runs
    # all three.
    if decisions:
        lines += [
            "### Optimizer Status",
            "",
            "| Optimizer | Decision | Detail |",
            "|-----------|----------|--------|",
        ]
        for label, decision, reason in decisions:
            lines.append(f"| {label} | {decision} | {reason} |")
        lines += [""]

    # Twin simulation results
    if has_twin_sim:
        twin = executor_rec["twin_sim"]
        current = twin.get("current_stats", {})
        proposed = twin.get("proposed_stats", {})
        delta = twin.get("delta", {})

        lines += [
            "### Twin Simulation (Current vs Proposed Executor Params)",
            "",
            "| Metric | Current Params | Proposed Params | Delta |",
            "|--------|---------------|-----------------|-------|",
        ]

        from optimizer.twin_sim import _COMPARE_METRICS
        for key, label, fmt in _COMPARE_METRICS:
            c = current.get(key)
            p = proposed.get(key)
            d = delta.get(key)
            c_str = f"{c:{fmt}}" if c is not None else "—"
            p_str = f"{p:{fmt}}" if p is not None else "—"
            if d is not None:
                sign = "+" if d >= 0 else ""
                d_str = f"{sign}{d:{fmt}}"
            else:
                d_str = "—"
            lines.append(f"| {label} | {c_str} | {p_str} | {d_str} |")

        # Promotion verdict based on twin sim
        proposed_better = twin.get("proposed_better", False)
        if proposed_better:
            lines += ["", "> Proposed params outperform current — promotion justified."]
        else:
            lines += ["", "> Proposed params did NOT outperform current."]

        # Param changes
        param_changes = twin.get("param_changes", {})
        if param_changes:
            lines += [
                "",
                "**Parameter changes (proposed vs current):**",
                "",
                "| Parameter | Current | Proposed |",
                "|-----------|---------|----------|",
            ]
            for k, v in param_changes.items():
                before = v.get("before")
                after = v.get("after")
                b_str = f"{before:.4f}" if isinstance(before, float) else str(before) if before is not None else "—"
                a_str = f"{after:.4f}" if isinstance(after, float) else str(after) if after is not None else "—"
                lines.append(f"| {k} | {b_str} | {a_str} |")

        lines += [""]
    elif executor_rec and executor_rec.get("status") == "ok":
        lines += [
            "### Twin Simulation",
            "",
            "> Twin simulation did not run (simulation setup unavailable or no current params in S3).",
            "",
        ]

    # Regression detection
    if has_regression:
        reg = regression_result
        if reg.get("regression_detected"):
            lines += [
                "### Regression Detected",
                "",
            ]
            details = reg.get("details", {})
            acc_drop = details.get("accuracy_drop")
            sharpe_drop = details.get("sharpe_drop_pct")
            if acc_drop is not None:
                lines.append(f"> Accuracy dropped {acc_drop:.1f}pp from baseline")
            if sharpe_drop is not None:
                lines.append(f"> Sharpe dropped {sharpe_drop:.1%} from baseline")
            if reg.get("rollback_triggered"):
                lines.append("> **AUTO-ROLLBACK triggered.** All configs restored to previous versions.")
            else:
                lines.append("> Thresholds not breached — no rollback.")
            lines += [""]
        else:
            lines += [
                "### Regression Monitor",
                "",
                "> No regression detected. Metrics within tolerance of promotion baseline.",
                "",
            ]

    return lines


def _section_weight_recommendation(result: dict) -> list[str]:
    lines = ["## Scoring weight recommendation"]
    status = result.get("status")

    if status in ("insufficient_data", "no_subscores", "error"):
        lines += ["", f"> **Deferred.** {result.get('note', result.get('error', 'Unavailable.'))}"]
        return lines

    n = result.get("n_samples", 0)
    confidence = result.get("confidence", "unknown")
    current = result.get("current_weights", {})
    suggested = result.get("suggested_weights", {})
    changes = result.get("changes", {})
    correlations = result.get("correlations", {})

    blend_factor = result.get("blend_factor")
    blend_str = f" · blend: {blend_factor:.2f}" if blend_factor is not None else ""
    lines += [
        "",
        f"_n={n} signals · confidence: {confidence}{blend_str}_",
        "",
        "| Sub-score | Current | Corr (10d) | Corr (30d) | Suggested | Change |",
        "|-----------|---------|------------|------------|-----------|--------|",
    ]
    for k in ("news", "research"):
        corr = correlations.get(k, {})
        c10 = corr.get("beat_spy_10d")
        c30 = corr.get("beat_spy_30d")
        chg = changes.get(k, 0)
        chg_str = f"+{chg:.1%}" if chg > 0 else f"{chg:.1%}"
        lines.append(
            f"| {k} | {_pct(current.get(k))} | {_fmt(c10)} | {_fmt(c30)} "
            f"| {_pct(suggested.get(k))} | {chg_str} |"
        )

    apply = result.get("apply_result", {})
    if apply.get("applied"):
        lines += [
            "",
            f"> ✅ **Weights updated automatically** in S3 (`config/scoring_weights.json`). "
            f"Research Lambda will use new weights on next cold-start. "
            f"n={apply.get('n_samples')}, confidence={apply.get('confidence')}.",
        ]
    else:
        reason = apply.get("reason", "guardrails not met")
        lines += [
            "",
            f"> ⏸ **Not applied** — {reason}.",
        ]

    # Recommendation stability (Gap #4)
    stability = result.get("stability", {})
    if stability.get("weeks_loaded", 0) > 0:
        if stability.get("stable"):
            lines += [f"> Recommendation stability: {stability['weeks_loaded'] + 1}/{stability['weeks_loaded'] + 1} weeks consistent"]
        else:
            reversals = stability.get("reversals", [])
            for rev in reversals:
                lines += [f"> WARNING: {rev}"]

    lines += [f"> {result.get('note', '')}"]
    return lines


def _section_confusion_matrix(result: dict) -> list[str]:
    """Build the predictor confusion matrix section."""
    lines = ["## Predictor confusion matrix (UP / FLAT / DOWN)"]
    n = result.get("n", 0)
    acc = result.get("accuracy")
    lines += [
        "",
        f"_{n} resolved predictions. Overall directional accuracy: {_pct(acc)}_",
        "",
    ]

    matrix = result.get("matrix", {})
    directions = ["UP", "FLAT", "DOWN"]

    lines.append("| Predicted \\ Actual | UP | FLAT | DOWN | Total |")
    lines.append("|--------------------|-----|------|------|-------|")
    for pred in directions:
        row = matrix.get(pred, {})
        counts = [row.get(a, 0) for a in directions]
        total = sum(counts)
        cells = " | ".join(
            f"**{c}**" if pred == a else str(c)
            for c, a in zip(counts, directions)
        )
        lines.append(f"| {pred} | {cells} | {total} |")

    # Per-class metrics
    per_class = result.get("per_class", {})
    if per_class:
        lines += [
            "",
            "### Per-direction precision / recall / F1",
            "",
            "| Direction | Precision | Recall | F1 | Predicted | Actual |",
            "|-----------|-----------|--------|----|-----------|--------|",
        ]
        for d in directions:
            c = per_class.get(d, {})
            p = _pct(c.get("precision")) if c.get("precision") is not None else "—"
            r = _pct(c.get("recall")) if c.get("recall") is not None else "—"
            f = f"{c['f1']:.3f}" if c.get("f1") is not None else "—"
            lines.append(f"| {d} | {p} | {r} | {f} | {c.get('n_predicted', 0)} | {c.get('n_actual', 0)} |")

    return lines


def _section_predictor_backtest(stats: dict) -> list[str]:
    """Build report section for predictor-only backtest results."""
    lines = ["## Layer-1A Momentum-Only Synthetic Backtest (10y component sanity check)"]
    status = stats.get("status", "unknown")

    if status in ("insufficient_data", "error"):
        note = stats.get("note", stats.get("error", "Unavailable."))
        lines += ["", f"> **Deferred.** {note}"]
        return lines

    if status == "no_orders":
        lines += [
            "",
            "> No ENTER signals passed risk rules during the simulation period.",
            f"> Dates simulated: {stats.get('dates_simulated', 'N/A')}",
        ]
        return lines

    meta = stats.get("predictor_metadata", {})
    lines += [
        "",
        "> **Component-level sanity check** — measures the Layer-1A momentum GBM "
        "in isolation; **not the production v3 ensemble** (which combines momentum "
        "+ volatility + research-score calibrator via a Layer-2 Ridge meta-learner "
        "that already downweights momentum to ~0). Production ensemble performance "
        "lives in `Mode 2 — Portfolio simulation` above and the live trades.db "
        "EOD email — interpret figures here as diagnosing the momentum component, "
        "not the system.",
        "",
        f"_GBM-only signals (no LLM research component). "
        f"{meta.get('n_tickers', 'N/A')} tickers, "
        f"{meta.get('n_dates', 'N/A')} trading days "
        f"({meta.get('date_range_start', '?')} → {meta.get('date_range_end', '?')}). "
        f"Top {meta.get('top_n_per_day', 'N/A')} ENTER signals/day, "
        f"min score {meta.get('min_score', 'N/A')}._",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total alpha** | **{_pct(stats.get('total_alpha'))}** |",
        f"| Total return | {_pct(stats.get('total_return'))} |",
        f"| SPY return | {_pct(stats.get('spy_return'))} |",
        f"| Sharpe ratio | {_fmt(stats.get('sharpe_ratio'))} |",
        f"| Max drawdown | {_pct(stats.get('max_drawdown'))} |",
        f"| Calmar ratio | {_fmt(stats.get('calmar_ratio'))} |",
        f"| Total trades | {stats.get('total_trades', 'N/A')} |",
        f"| Win rate | {_pct(stats.get('win_rate'))} |",
        f"| Dates simulated | {stats.get('dates_simulated', 'N/A')} |",
        f"| Total orders | {stats.get('total_orders', 'N/A')} |",
        "",
        "> This tests the full executor pipeline (risk guard, position sizing, ATR stops, "
        "time decay, graduated drawdown) using GBM predictions on historical price data. "
        "Macro context is neutral; sector ratings are market-weight.",
    ]
    return lines


def _section_param_sweep_predictor(df) -> list[str]:
    """Build report section for predictor-only param sweep results."""
    # All known non-parameter columns. Includes the evaluator-revamp
    # additions (sortino_ratio, cvar_95, calmar_ratio) so they don't
    # leak into param_cols and confuse the operator about which columns
    # are tunable params vs derived metrics.
    NON_PARAM_COLS = {
        "total_return", "total_alpha", "spy_return", "sharpe_ratio",
        "sortino_ratio", "cvar_95", "calmar_ratio",
        "max_drawdown", "total_trades", "win_rate",
        "status", "dates_simulated", "total_orders", "note", "error",
    }
    param_cols = [c for c in df.columns if c not in NON_PARAM_COLS]

    # Render the most-informative stat columns alongside params, in a fixed
    # order. ``total_alpha`` first (presentation), Sortino + CVaR second
    # (skilled-risk-taking metric stack from the 2026-05-06 evaluator
    # revamp), then Sharpe + return + drawdown + win rate.
    PREFERRED_STAT_ORDER = [
        "total_alpha", "sortino_ratio", "cvar_95",
        "sharpe_ratio", "total_return", "spy_return", "max_drawdown", "win_rate",
    ]
    stat_cols = [c for c in PREFERRED_STAT_ORDER if c in df.columns]

    # Header is honest about the rendered ranking. Sweep is sorted by
    # total_alpha (primary) per ``param_sweep.py::_run_combos``; surface
    # that explicitly so "by total alpha" doesn't read as misleading
    # when the alpha column is suppressed for narrowness.
    sort_label = "total_alpha" if "total_alpha" in df.columns else "sharpe_ratio"
    lines = [f"## Predictor param sweep — top combinations (sorted by {sort_label})", ""]

    show_cols = param_cols + stat_cols
    header = "| " + " | ".join(show_cols) + " |"
    sep    = "| " + " | ".join("---" for _ in show_cols) + " |"
    lines += [header, sep]
    PCT_COLS = {"total_return", "total_alpha", "spy_return", "max_drawdown", "win_rate"}
    FMT_COLS = {"sharpe_ratio", "sortino_ratio", "cvar_95", "calmar_ratio"}
    for _, row in df.head(10).iterrows():
        cells = []
        for c in show_cols:
            v = row.get(c)
            if c in PCT_COLS:
                cells.append(_pct(v))
            elif c in FMT_COLS:
                cells.append(_fmt(v))
            else:
                cells.append(str(v) if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"_Full results in param_sweep.csv (predictor-only)_"]
    return lines


def _section_param_sweep(df) -> list[str]:
    # Sweep metadata from DataFrame attrs
    sweep_mode = df.attrs.get("sweep_mode", "grid")
    sweep_trials = df.attrs.get("sweep_trials", len(df))
    sweep_total = df.attrs.get("sweep_total_grid", len(df))
    sweep_coverage = df.attrs.get("sweep_coverage", 1.0)

    title = "## Param sweep — top combinations by Sharpe ratio"
    if sweep_mode != "grid":
        title += f" ({sweep_mode}: {sweep_trials}/{sweep_total} combos, {sweep_coverage:.0%} coverage)"

    lines = [title, ""]
    # Columns that aren't sweep params — exclude from the param column list.
    # Includes stat cols (shown separately) AND metadata cols that leak into
    # the sweep_df (dates_expected, coverage, total_orders, skip_reasons,
    # price_gap_warnings, unfilled_gaps) — some of these are dict-valued
    # and would blow out the table layout. Discovered 2026-04-11 when a
    # 60-trial sweep rendered as a 60+ page email because `price_gap_warnings`
    # (a dict of ticker→gap_days) was str()-ified into every row.
    _NON_PARAM_COLS = {
        # stat cols (rendered separately as stat_cols)
        "total_return", "total_alpha", "spy_return", "sharpe_ratio",
        "max_drawdown", "calmar_ratio", "total_trades", "win_rate",
        # metadata leaked from sweep infra — not real params
        "status", "dates_simulated", "dates_expected", "coverage",
        "total_orders", "skip_reasons", "price_gap_warnings",
        "unfilled_gaps", "note", "error",
    }
    # Defensive: also drop any column whose values are non-scalar
    # (dict/list/set) — catches future leaks without needing to maintain
    # the exclude list.
    def _is_scalar_col(col: str) -> bool:
        sample = df[col].dropna()
        if sample.empty:
            return True
        return not isinstance(sample.iloc[0], (dict, list, set))

    param_cols = [
        c for c in df.columns
        if c not in _NON_PARAM_COLS and _is_scalar_col(c)
    ]
    stat_cols = [c for c in ["total_alpha", "sharpe_ratio", "total_return", "spy_return", "max_drawdown", "win_rate"] if c in df.columns]
    show_cols = param_cols + stat_cols
    header = "| " + " | ".join(show_cols) + " |"
    sep    = "| " + " | ".join("---" for _ in show_cols) + " |"
    lines += [header, sep]
    for _, row in df.head(10).iterrows():
        cells = []
        for c in show_cols:
            v = row.get(c)
            if c in ("total_return", "total_alpha", "spy_return", "max_drawdown", "win_rate"):
                cells.append(_pct(v))
            elif c in ("sharpe_ratio",):
                cells.append(_fmt(v))
            else:
                cells.append(str(v) if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"_Full results in param_sweep.csv_"]
    return lines


def _section_veto_analysis(result: dict) -> list[str]:
    lines = ["## Predictor veto threshold analysis"]
    status = result.get("status", "unknown")

    if status not in ("ok", "insufficient_lift"):
        note = result.get("note", result.get("error", "Unavailable."))
        lines += ["", f"> **Deferred.** {note}"]
        return lines

    current = result.get("current_threshold", 0.65)
    recommended = result.get("recommended_threshold")
    n_down = result.get("n_down_predictions", 0)
    base_rate = result.get("base_rate")

    base_rate_str = f" Base rate (BUY signals beating SPY): {_pct(base_rate)}." if base_rate is not None else ""
    lines += [
        "",
        f"_Analyzed {n_down} DOWN predictions with resolved outcomes.{base_rate_str}_",
        "",
        "| Confidence | Vetoes | True neg | False neg | Precision | Recall | F1 | CI 95% | Lift | Missed α |",
        "|------------|--------|----------|-----------|-----------|--------|----|--------|------|----------|",
    ]
    for t in result.get("thresholds", []):
        conf = t["confidence"]
        marker = ""
        if conf == current:
            marker = " (current)"
        if conf == recommended:
            marker = " **→**"
        prec = _pct(t["precision"]) if t["precision"] is not None else "—"
        recall_str = _pct(t.get("recall")) if t.get("recall") is not None else "—"
        f1_str = f"{t['f1']:.3f}" if t.get("f1") is not None else "—"
        ci = t.get("precision_ci_95")
        ci_str = f"[{ci[0]:.0%}–{ci[1]:.0%}]" if ci else "—"
        if t.get("low_confidence"):
            ci_str += "†"
        lift = t.get("lift")
        lift_str = f"{lift:+.1%}" if lift is not None else "—"
        lines.append(
            f"| {conf:.2f}{marker} | {t['n_vetoes']} | {t['true_negatives']} "
            f"| {t['false_negatives']} | {prec} | {recall_str} | {f1_str} | {ci_str} | {lift_str} "
            f"| {t['missed_alpha']:.4f} |"
        )

    lines += ["", "† Low confidence — fewer than 30 veto decisions"]

    if recommended is not None:
        lines += ["", f"> **Recommended:** {recommended:.2f} — {result.get('recommendation_reason', '')}"]
    else:
        lines += ["", f"> {result.get('recommendation_reason', 'No recommendation.')}"]

    # Cost sensitivity (Gap #10)
    cost_sens = result.get("cost_sensitivity")
    if cost_sens:
        details = result.get("cost_sensitivity_details", {})
        lines += [f"> Cost sensitivity: **{cost_sens}** — thresholds by cost_weight: {details}"]

    apply = result.get("apply_result", {})
    if apply.get("applied"):
        lines += [
            f"> ✅ **Veto threshold updated** in S3 (`config/predictor_params.json`). "
            f"Predictor Lambda will use {apply.get('veto_confidence'):.2f} on next cold-start.",
        ]
    else:
        reason = apply.get("reason", "guardrails not met")
        lines += [f"> ⏸ **Not applied** — {reason}."]

    by_sector = result.get("by_sector", [])
    if by_sector:
        lines += ["", "### Veto precision by sector", ""]
        lines += ["| Sector | DOWN preds | Vetoes | Precision | Recall |"]
        lines += ["|--------|-----------|--------|-----------|--------|"]
        for s in by_sector:
            p = _pct(s.get("precision")) if s.get("precision") is not None else "—"
            r = _pct(s.get("recall")) if s.get("recall") is not None else "—"
            lines.append(
                f"| {s['sector']} | {s['n_down']} | {s['n_vetoes']} | {p} | {r} |"
            )

    return lines


def _section_executor_recommendations(result: dict) -> list[str]:
    lines = ["## Executor parameter recommendations"]
    status = result.get("status", "unknown")

    if status != "ok":
        note = result.get("note", result.get("error", "Unavailable."))
        lines += ["", f"> **Deferred.** {note}"]
        return lines

    baseline = result.get("baseline_params", {})
    recommended = result.get("recommended_params", {})
    factory = result.get("factory_defaults", {})
    improvement = result.get("improvement_pct", 0)
    fit_target = result.get("fit_target", "sharpe_legacy")

    baseline_rank = result.get("baseline_combo_rank")
    n_combos = result.get("n_combos_tested", "?")
    baseline_note = ""
    if baseline_rank is not None:
        baseline_note = f" Baseline: combo #{baseline_rank} of {n_combos} (closest to current S3 params)."

    # Sweep coverage — needed by the caption below.
    swept_keys = sorted(set(list(baseline.keys()) + list(recommended.keys())))
    n_factory = len(factory)
    n_swept = len(swept_keys)

    # Caption text branches on fit_target. Skill-composite mode leads
    # with Sortino + PSR (the gating axes); raw alpha vs SPY surfaces as
    # a presentation-only stat per Brian's 2026-05-09 framing
    # ("alpha vs SPY is presentation, not the optimizer's fit target").
    # Legacy (Sharpe-with-drawdown) mode preserves the pre-cutover
    # caption shape exactly.
    if fit_target == "skill_composite":
        best_sortino = result.get("best_sortino")
        baseline_sortino = result.get("baseline_sortino")
        best_psr = result.get("best_psr")
        best_alpha = result.get("best_alpha")

        sortino_str = ""
        if best_sortino is not None and baseline_sortino is not None:
            sortino_str = (
                f"Sortino improvement: {improvement:.1%} "
                f"({baseline_sortino:.4f} → {best_sortino:.4f})"
            )
        elif best_sortino is not None:
            sortino_str = f"Best Sortino: {best_sortino:.4f}"

        psr_str = (
            f" | PSR (P(true SR>0)): {best_psr:.3f}"
            if best_psr is not None else ""
        )
        alpha_str = (
            f" | Alpha vs SPY: {best_alpha:+.1%} (presentation only)"
            if best_alpha is not None else ""
        )

        caption = (
            f"_Tested {n_combos} parameter combinations across {n_swept} of "
            f"{n_factory} safe-to-tune params (fit_target=`skill_composite`). "
            f"{sortino_str}{psr_str}{alpha_str}.{baseline_note}_"
        )
    else:
        best_alpha = result.get("best_alpha")
        alpha_str = f" | Best alpha: {best_alpha:.1%}" if best_alpha is not None else ""
        caption = (
            f"_Tested {n_combos} parameter combinations across {n_swept} of "
            f"{n_factory} safe-to-tune params. "
            f"Sharpe improvement: {improvement:.1%} "
            f"({result.get('baseline_sharpe', 0):.4f} → {result.get('best_sharpe', 0):.4f})"
            f"{alpha_str}.{baseline_note}_"
        )

    # Render only params the sweep actually exercised — params without
    # baseline AND recommended values are absent from this run's grid;
    # showing them as `—`/`—`/`—` rows pollutes the table with
    # uninformative entries (operator already knows the factory default
    # via the SAFE_PARAMS source). Matches the "drop unswept rows"
    # cleanup from the 2026-05-09 P2 ROADMAP entry.
    lines += [
        "",
        caption,
        "",
        "| Parameter | Default | Current (S3) | Recommended | Drift from default |",
        "|-----------|---------|--------------|-------------|-------------------|",
    ]
    for k in swept_keys:
        d = factory.get(k)
        b = baseline.get(k)
        r = recommended.get(k)
        d_str = f"{d:.4f}" if isinstance(d, float) else str(d) if d is not None else "—"
        b_str = f"{b:.4f}" if isinstance(b, float) else str(b) if b is not None else "—"
        r_str = f"{r:.4f}" if isinstance(r, float) else str(r) if r is not None else "—"
        if d is not None and r is not None and isinstance(d, (int, float)) and isinstance(r, (int, float)):
            drift = r - d
            drift_str = f"{'+' if drift >= 0 else ''}{drift:.4f}"
        else:
            drift_str = "—"
        lines.append(f"| {k} | {d_str} | {b_str} | {r_str} | {drift_str} |")

    # Footer: name the params that were NOT in this sweep grid so the
    # operator can see what's intentionally untouched vs accidentally
    # absent. Empty list when the grid covers everything.
    unswept = sorted(set(factory.keys()) - set(swept_keys))
    if unswept:
        lines += [
            "",
            f"> **Not in sweep grid** ({len(unswept)} of {n_factory}): "
            f"{', '.join(f'`{k}`' for k in unswept)}. "
            "Operator-tunable via `analysis/param_sweep.py` grid.",
        ]

    lines += ["", f"> {result.get('note', '')}"]

    apply = result.get("apply_result", {})
    if apply.get("applied"):
        lines += [
            f"> ✅ **Executor params updated** in S3 (`config/executor_params.json`). "
            f"Executor Lambda will use new params on next cold-start.",
        ]
    else:
        reason = apply.get("reason", "guardrails not met")
        lines += [f"> ⏸ **Not applied** — {reason}."]

    return lines


def _section_trigger_scorecard(result: dict) -> list[str]:
    """Build entry trigger scorecard section."""
    lines = ["## Entry trigger scorecard"]
    summary = result.get("summary", {})
    lines += [
        "",
        f"_{summary.get('total_entries', 0)} total entries analyzed_",
        "",
        "| Trigger | n | Avg slip vs signal | Avg slip vs open | Avg alpha | Win rate |",
        "|---------|---|--------------------|------------------|-----------|---------|",
    ]
    for t in result.get("triggers", []):
        lines.append(
            f"| {t['trigger']} | {t['n_trades']} | "
            f"{_pct_pts(t.get('avg_slippage_vs_signal'))} | "
            f"{_pct_pts(t.get('avg_slippage_vs_open'))} | "
            f"{_pct_pts(t.get('avg_realized_alpha'))} | "
            f"{_pct(t.get('win_rate_vs_spy'))} |"
        )
    lines.append(
        f"| **All** | {summary.get('total_entries', 0)} | "
        f"{_pct_pts(summary.get('avg_slippage_vs_signal'))} | "
        f"{_pct_pts(summary.get('avg_slippage_vs_open'))} | "
        f"{_pct_pts(summary.get('avg_realized_alpha'))} | "
        f"{_pct(summary.get('win_rate_vs_spy'))} |"
    )
    lines += [
        "",
        "> Negative slippage = fill below signal/open price (favorable). "
        "Triggers with negative alpha or low win rate are candidates for removal.",
    ]
    return lines


def _section_alpha_distribution(result: dict) -> list[str]:
    """Build alpha magnitude distribution section."""
    lines = ["## Alpha magnitude distribution"]

    for horizon, summary_data in result.get("summary", {}).items():
        buckets = result.get("distributions", {}).get(horizon, [])
        if not buckets:
            continue

        lines += [
            "",
            f"### {horizon} horizon (n={summary_data['n']})",
            "",
            f"Avg alpha: **{summary_data['avg_alpha']:+.2f}pp** | "
            f"Median: {summary_data['median_alpha']:+.2f}pp | "
            f"Std: {summary_data['std_alpha']:.2f}pp | "
            f"Positive: {_pct(summary_data.get('pct_positive'))}",
            "",
            "| Bucket | Count | % | Avg alpha |",
            "|--------|-------|---|-----------|",
        ]
        for b in buckets:
            lines.append(
                f"| {b['bucket']} | {b['count']} | {_pct(b['pct'])} | "
                f"{_pct_pts(b.get('avg_alpha'))} |"
            )

    return lines


def _section_score_calibration(result: dict) -> list[str]:
    """Build score calibration curve section."""
    lines = [
        "## Score calibration curve",
        "",
        f"_Horizon: {result.get('horizon', '10d')} — "
        f"{'Monotonic ✓' if result.get('monotonic') else 'Non-monotonic ✗'}_",
        "",
        "| Score range | n | Avg score | Avg alpha (pp) | Beat SPY % |",
        "|------------|---|-----------|----------------|------------|",
    ]
    for c in result.get("calibration", []):
        lines.append(
            f"| {c['score_range']} | {c['n']} | {c['avg_score']:.0f} | "
            f"{c['avg_alpha']:+.2f} | {_pct(c.get('beat_spy_pct'))} |"
        )

    if not result.get("monotonic"):
        lines += ["", "> Non-monotonic: higher scores do NOT consistently predict higher alpha. Score calibration may need adjustment."]
    else:
        lines += ["", "> Monotonic: higher scores predict higher alpha — scoring is well-calibrated."]

    # Per-bucket diagnostics: surface sector / regime / date concentration so
    # a non-monotonic curve can be distinguished from small-sample noise
    # (e.g., one bad Healthcare week dominating the mid-range bucket).
    diag_rows = [
        c for c in result.get("calibration", [])
        if c.get("top_sectors") or c.get("regime_breakdown") or c.get("n_unique_dates")
    ]
    if diag_rows:
        lines += [
            "",
            "### Bucket diagnostics",
            "",
            "_A bucket dominated by one sector or regime hints the non-monotonicity is compositional, not a scoring flaw._",
            "",
            "| Score range | n | Dates | Tickers | Top sectors | Regime mix |",
            "|-------------|---|-------|---------|-------------|------------|",
        ]
        for c in diag_rows:
            sectors = ", ".join(
                f"{s['sector']}({s['n']})" for s in c.get("top_sectors", [])
            ) or "—"
            regimes = ", ".join(
                f"{r['regime']}({r['n']})" for r in c.get("regime_breakdown", [])
            ) or "—"
            lines.append(
                f"| {c['score_range']} | {c['n']} | "
                f"{c.get('n_unique_dates', '—')} | "
                f"{c.get('n_unique_tickers', '—')} | "
                f"{sectors} | {regimes} |"
            )

    return lines


def _section_veto_value(result: dict) -> list[str]:
    """Build net veto value in dollars section."""
    lines = [
        "## Net veto value",
        "",
        f"_{result['n_vetoes']} DOWN predictions evaluated "
        f"({result['n_correct']} correct, {result['n_incorrect']} incorrect). "
        f"Precision: {_pct(result.get('precision'))}_",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Losses avoided (correct vetoes) | **${result['total_losses_avoided']:,.0f}** |",
        f"| Alpha foregone (incorrect vetoes) | ${result['total_alpha_foregone']:,.0f} |",
        f"| **Net veto value** | **${result['net_veto_value']:,.0f}** |",
        f"| Avg loss avoided per veto | ${result['avg_loss_avoided']:,.0f} |",
        f"| Avg alpha foregone per miss | ${result['avg_alpha_foregone']:,.0f} |",
        f"| Avg vetoed stock alpha (5d) | {result.get('avg_veto_alpha_pct', 0):+.2f}pp |",
    ]

    by_conf = result.get("by_confidence", [])
    if by_conf:
        lines += [
            "",
            "### By confidence level",
            "",
            "| Confidence | Vetoes | Precision | Losses avoided | Alpha foregone | Net value |",
            "|------------|--------|-----------|---------------|----------------|-----------|",
        ]
        for c in by_conf:
            lines.append(
                f"| {c['confidence_range']} | {c['n_vetoes']} | "
                f"{_pct(c.get('precision'))} | "
                f"${c['losses_avoided']:,.0f} | ${c['alpha_foregone']:,.0f} | "
                f"${c['net_value']:,.0f} |"
            )

    verdict = "positive — vetoes are net beneficial" if result['net_veto_value'] > 0 else "negative — vetoes cost more than they save"
    lines += ["", f"> Net veto value is **{verdict}**."]
    return lines


def _section_shadow_book(result: dict) -> list[str]:
    """Build shadow book analysis section."""
    lines = [
        "## Risk guard shadow book",
        "",
        f"_{result['n_blocked']} blocked entries, {result['n_traded']} traded entries_",
    ]

    assessment = result.get("assessment", "unknown")
    if result.get("blocked_avg_return_5d") is not None:
        lines += [
            "",
            "| Cohort | Avg 5d return | n |",
            "|--------|---------------|---|",
            f"| Blocked entries | {result['blocked_avg_return_5d']:.2%} | {result.get('blocked_with_returns', '?')} |",
        ]
        if result.get("traded_avg_return_5d") is not None:
            lines.append(f"| Traded entries | {result['traded_avg_return_5d']:.2%} | {result.get('traded_with_returns', '?')} |")
        if result.get("guard_lift") is not None:
            lines.append(f"| **Guard lift** | **{result['guard_lift']:.2%}** | — |")

    by_reason = result.get("by_reason", [])
    if by_reason:
        lines += [
            "",
            "### Blocks by reason",
            "",
            "| Reason | Count | % of blocks | Avg score | Avg 5d return |",
            "|--------|-------|-------------|-----------|---------------|",
        ]
        for r in by_reason:
            ret_str = f"{r['avg_return_5d']:.2%}" if r.get("avg_return_5d") is not None else "—"
            score_str = f"{r['avg_score']:.0f}" if r.get("avg_score") is not None else "—"
            lines.append(
                f"| {r['block_reason']} | {r['count']} | "
                f"{_pct(r.get('pct_of_blocks'))} | {score_str} | {ret_str} |"
            )

    # Classification metrics (if available)
    clf = result.get("classification")
    if clf:
        p = _pct(clf.get("precision")) if clf.get("precision") is not None else "—"
        r = _pct(clf.get("recall")) if clf.get("recall") is not None else "—"
        f = f"{clf['f1']:.3f}" if clf.get("f1") is not None else "—"
        lines += [
            "",
            "### Classification (blocked=predicted loser)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Precision (% of blocks that were actual losers) | {p} |",
            f"| Recall (% of all losers that were blocked) | {r} |",
            f"| F1 | {f} |",
            f"| TP={clf.get('tp', 0)}, FP={clf.get('fp', 0)}, FN={clf.get('fn', 0)}, TN={clf.get('tn', 0)} | n={clf.get('n', 0)} |",
        ]

    verdicts = {
        "appropriate": "Risk guard is appropriately calibrated — traded entries outperform blocked entries.",
        "too_tight": "Risk guard may be too conservative — blocked entries would have outperformed traded entries.",
        "neutral": "Risk guard impact is neutral — blocked and traded entries perform similarly.",
        "too_loose": "Risk guard may be too loose — blocked entries significantly underperform.",
    }
    lines += ["", f"> **Assessment:** {verdicts.get(assessment, assessment)}"]
    return lines


def _section_exit_timing(result: dict) -> list[str]:
    """Build exit timing (MFE/MAE) section."""
    summary = result.get("summary", {})
    lines = [
        "## Exit timing analysis (MFE/MAE)",
        "",
        f"_{summary.get('n_roundtrips', 0)} roundtrip trades analyzed_",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg MFE (max gain during hold) | {summary.get('avg_mfe', 0):+.2f}% |",
        f"| Avg MAE (max loss during hold) | {summary.get('avg_mae', 0):+.2f}% |",
        f"| Avg realized return | {summary.get('avg_realized_return', 0):+.2f}% |",
        f"| Avg capture ratio (realized / MFE) | {summary.get('avg_capture_ratio', 0):.0%} |",
        f"| Median MFE | {summary.get('median_mfe', 0):+.2f}% |",
        f"| Median MAE | {summary.get('median_mae', 0):+.2f}% |",
    ]

    by_exit = result.get("by_exit_type", [])
    if by_exit:
        lines += [
            "",
            "### By exit type",
            "",
            "| Exit type | n | Avg MFE | Avg MAE | Avg return | Capture |",
            "|-----------|---|---------|---------|------------|---------|",
        ]
        for e in by_exit:
            cap_str = f"{e['avg_capture']:.0%}" if e.get("avg_capture") is not None else "—"
            lines.append(
                f"| {e['exit_type']} | {e['n']} | "
                f"{e['avg_mfe']:+.2f}% | {e['avg_mae']:+.2f}% | "
                f"{e['avg_realized']:+.2f}% | {cap_str} |"
            )

    diagnosis = result.get("diagnosis", "unknown")
    diagnosis_text = {
        "exits_too_early": "Exits are leaving significant gains on the table (low capture ratio). Consider widening trailing stops.",
        "exits_well_timed": "Exits are well-timed — capturing a good portion of MFE while limiting MAE exposure.",
        "exits_could_improve": "Exit timing has room for improvement. Review stop levels and profit-take thresholds.",
        "exits_too_late": "Exits may be triggering too late — MAE is close to realized losses.",
    }
    lines += ["", f"> **Diagnosis:** {diagnosis_text.get(diagnosis, diagnosis)}"]
    return lines


def _section_macro_eval(result: dict) -> list[str]:
    """Build macro multiplier A/B evaluation section."""
    with_m = result.get("with_macro", {})
    without_m = result.get("without_macro", {})
    lines = [
        "## Macro multiplier evaluation",
        "",
        f"_{result.get('n_evaluated', 0)} CIO evaluations analyzed_",
        "",
        "| Metric | With macro | Without macro | Lift |",
        "|--------|-----------|---------------|------|",
        f"| Accuracy (beat SPY 5d) | {_pct(with_m.get('accuracy'))} | {_pct(without_m.get('accuracy'))} | {_pct(result.get('accuracy_lift'))} |",
        f"| Avg alpha (5d, pp) | {_pct_pts(with_m.get('avg_alpha'))} | {_pct_pts(without_m.get('avg_alpha'))} | {_pct_pts(result.get('alpha_lift'))} |",
        f"| n stocks selected | {with_m.get('n', 0)} | {without_m.get('n', 0)} | — |",
    ]

    impact = result.get("macro_impact", {})
    if impact.get("n_promoted", 0) > 0 or impact.get("n_demoted", 0) > 0:
        lines += [
            "",
            f"Macro shift changed BUY status for "
            f"**{impact.get('n_promoted', 0)}** stocks (promoted) and "
            f"**{impact.get('n_demoted', 0)}** stocks (demoted).",
        ]
        if impact.get("promoted_avg_alpha") is not None:
            lines.append(f"  - Promoted stocks avg alpha: {impact['promoted_avg_alpha']:+.2f}pp")
        if impact.get("demoted_avg_alpha") is not None:
            lines.append(f"  - Demoted stocks avg alpha: {impact['demoted_avg_alpha']:+.2f}pp")

    shift_stats = result.get("shift_stats", {})
    if shift_stats:
        lines += [
            "",
            f"Shift stats: avg {shift_stats.get('avg_shift', 0):+.1f}, "
            f"range [{shift_stats.get('min_shift', 0):+.1f}, {shift_stats.get('max_shift', 0):+.1f}], "
            f"positive {shift_stats.get('n_positive', 0)} / negative {shift_stats.get('n_negative', 0)}",
        ]

    verdicts = {
        "helps": "Macro shift **improves** accuracy — keep it enabled.",
        "hurts": "Macro shift **hurts** accuracy — consider disabling or reducing magnitude.",
        "neutral": "Macro shift has **no measurable effect** — may not be worth the complexity.",
    }
    lines += ["", f"> **Assessment:** {verdicts.get(result.get('assessment', ''), result.get('assessment', ''))}"]
    return lines


def _section_trigger_opt(result: dict) -> list[str]:
    """Build trigger optimizer section."""
    lines = [
        "## Trigger optimizer (4e)",
        "",
    ]
    recs = result.get("recommendations", [])
    disabled = result.get("disabled_triggers", [])

    if disabled:
        lines.append(f"**Disabled triggers:** {', '.join(disabled)}")
    else:
        lines.append("**No triggers disabled** — all performing adequately or insufficient data.")
    lines += [""]

    if recs:
        lines += [
            "| Trigger | Action | Trades | Avg Alpha | Win Rate | Reasons |",
            "|---------|--------|--------|-----------|----------|---------|",
        ]
        for r in recs:
            alpha_str = f"{r.get('avg_alpha', 0):.3%}" if r.get("avg_alpha") is not None else "—"
            wr_str = f"{r.get('win_rate', 0):.0%}" if r.get("win_rate") is not None else "—"
            reasons = ", ".join(r.get("reasons", [])) or r.get("reason", "—")
            lines.append(
                f"| {r.get('trigger', '?')} | {r.get('action', '?')} | "
                f"{r.get('n_trades', 0)} | {alpha_str} | {wr_str} | {reasons} |"
            )

    apply_r = result.get("apply_result", {})
    if apply_r.get("applied"):
        lines += ["", f"> Applied to S3: disabled {disabled}"]
    elif apply_r:
        lines += ["", f"> Not applied: {apply_r.get('reason', '—')}"]

    return lines


def _section_predictor_sizing(result: dict) -> list[str]:
    """Build predictor p_up sizing section."""
    lines = [
        "## Predictor p_up sizing (4d)",
        "",
        f"**Overall rank IC:** {result.get('overall_rank_ic', 0):.4f}",
        f"**Recent mean IC ({result.get('recent_total_weeks', 0)}w):** {result.get('recent_mean_ic', 0):.4f}",
        f"**Positive weeks:** {result.get('recent_positive_weeks', 0)}/{result.get('recent_total_weeks', 0)}",
        f"**Sizing lift:** {result.get('sizing_lift', 0):.4%}",
        "",
        f"**Recommendation:** {result.get('recommendation', '?')}",
    ]

    apply_r = result.get("apply_result", {})
    if apply_r.get("applied"):
        lines.append(f"> p_up sizing enabled in S3 (IC={apply_r.get('ic', '?')})")
    elif apply_r:
        lines.append(f"> Not applied: {apply_r.get('reason', '—')}")

    return lines


def _section_scanner_opt(result: dict) -> list[str]:
    """Build scanner optimizer section."""
    lines = ["## Scanner filter optimizer (4a)", ""]

    if result.get("status") == "insufficient_data":
        note = result.get("note", "")
        n_weeks = result.get("n_weeks", 0)
        min_req = result.get("min_required", 8)
        lines.append(f"Insufficient data ({n_weeks}/{min_req} weeks). {note}")
        return lines

    analysis = result.get("analysis", result)
    lines += [
        f"**Filter leakage:** {analysis.get('leakage_rate', 0):.1%} "
        f"(threshold: {analysis.get('leakage_threshold', 0):.1%})",
        f"**Filter lift:** {analysis.get('filter_lift', 0):.4f}" if analysis.get("filter_lift") is not None else "",
        f"**Weeks analyzed:** {analysis.get('n_weeks', 0)}",
        "",
    ]

    gates = analysis.get("gate_analysis", [])
    if gates:
        lines += [
            "| Gate | Rejected | Leakage | Avg Return |",
            "|------|----------|---------|------------|",
        ]
        for g in gates:
            lines.append(
                f"| {g.get('gate', '?')} | {g.get('n_rejected', 0)} | "
                f"{g.get('leakage_rate', 0):.1%} | "
                f"{g.get('avg_return_5d', 0):.3%} |"
                if g.get("avg_return_5d") is not None else
                f"| {g.get('gate', '?')} | {g.get('n_rejected', 0)} | "
                f"{g.get('leakage_rate', 0):.1%} | — |"
            )

    if result.get("status") == "ok" and result.get("changes"):
        lines += ["", f"**Recommended changes:** {result.get('changes')}"]
        apply_r = result.get("apply_result", {})
        if apply_r.get("applied"):
            lines.append("> Applied to S3")
        elif apply_r:
            lines.append(f"> Not applied: {apply_r.get('reason', '—')}")
    elif result.get("status") == "no_change":
        lines.append(f"> {result.get('note', 'No changes needed')}")

    return lines


def _section_team_opt(result: dict) -> list[str]:
    """Build team slot optimizer section."""
    lines = ["## Team slot allocation (4b)", ""]

    if result.get("status") == "insufficient_data":
        lines.append(f"Insufficient data ({result.get('n_weeks', 0)}/{result.get('min_required', 8)} weeks).")
        return lines

    analysis = result.get("analysis", {})
    teams = analysis.get("team_analysis", [])

    if teams:
        lines += [
            "| Team | Lift vs Sector | Lift vs Quant | Picks | Assessment | Slot Δ |",
            "|------|---------------|---------------|-------|------------|--------|",
        ]
        for t in teams:
            lift_s = f"{t.get('lift_vs_sector', 0):.3%}" if t.get("lift_vs_sector") is not None else "—"
            lift_q = f"{t.get('lift_vs_quant', 0):.3%}" if t.get("lift_vs_quant") is not None else "—"
            change = t.get("recommended_slot_change", 0)
            change_str = f"{change:+d}" if change != 0 else "—"
            lines.append(
                f"| {t.get('team_id', '?')} | {lift_s} | {lift_q} | "
                f"{t.get('n_picks', 0)} | {t.get('assessment', '?')} | {change_str} |"
            )

    if result.get("status") == "ok" and result.get("changes"):
        lines += ["", f"**Slot changes:** {result.get('changes')}"]
    elif result.get("status") == "no_change":
        lines += ["", "> No slot changes — all teams performing within bounds."]

    return lines


def _section_cio_opt(result: dict) -> list[str]:
    """Build CIO fallback optimizer section."""
    lines = [
        "## CIO mode optimizer (4c)",
        "",
        f"**CIO lift:** {result.get('cio_lift', 0):.4f}" if result.get("cio_lift") is not None else "**CIO lift:** —",
        f"**CIO vs ranking:** {result.get('cio_vs_ranking_lift', 0):.4f}" if result.get("cio_vs_ranking_lift") is not None else "**CIO vs ranking:** —",
        f"**Recommendation:** {result.get('recommendation', '?')}",
        "",
        f"> {result.get('reasoning', '')}",
    ]
    return lines


def _section_sizing_ab(result: dict) -> list[str]:
    """Build sizing A/B test section."""
    current = result.get("current_sizing", {})
    equal = result.get("equal_weight", {})
    lines = [
        "## Position sizing A/B test (4f)",
        "",
        "| Metric | Current Sizing | Equal Weight | Difference |",
        "|--------|---------------|--------------|------------|",
        f"| Sharpe | {current.get('sharpe', '—')} | {equal.get('sharpe', '—')} | {result.get('sharpe_diff', '—')} |",
        f"| Total Return | {current.get('total_return', '—')} | {equal.get('total_return', '—')} | {result.get('return_diff', '—')} |",
        f"| Total Alpha | {current.get('total_alpha', '—')} | {equal.get('total_alpha', '—')} | {result.get('alpha_diff', '—')} |",
        f"| Max Drawdown | {current.get('max_drawdown', '—')} | {equal.get('max_drawdown', '—')} | — |",
        f"| Trades | {current.get('total_trades', '—')} | {equal.get('total_trades', '—')} | — |",
        "",
        f"> **Assessment:** {result.get('detail', result.get('assessment', '?'))}",
    ]
    return lines


def _pct_pts(v) -> str:
    """Format a value as percentage points (already in pct units)."""
    if v is None:
        return "—"
    return f"{v:+.2f}pp"


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _alpha_pp(v) -> str:
    """Format alpha values that are already in percentage-point form (e.g. 5.0 = 5pp)."""
    if v is None:
        return "—"
    return f"{v:.1f}%"


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"
