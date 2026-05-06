# alpha-engine-backtester — Code Index

> Index of entry points, key files, and data contracts. Companion to [README.md](README.md). System overview lives in [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs).

## Module purpose

Weekly system evaluator + autonomous parameter optimizer. Reads historical signals + trades, runs sweeps, writes four optimized configs back to S3 — closing the system's learning loop.

## Entry points

| File | What it does |
|---|---|
| [`backtest.py`](backtest.py) | CLI for simulation modes — `--mode signal-quality / simulate / param-sweep / all / predictor-backtest` |
| [`evaluate.py`](evaluate.py) | CLI for evaluation + optimizer auto-apply — `--mode all / diagnostics / optimize`, `--module <name>` |
| [`infrastructure/spot_backtest.sh`](infrastructure/spot_backtest.sh) | Spot-instance launcher (provisions c5.large, runs, self-terminates) |

## Where things live

| Concept | File |
|---|---|
| Pipeline shared utilities (config, DB seeding, backfill) | [`pipeline_common.py`](pipeline_common.py) |
| Per-module data completeness tracker | [`completeness.py`](completeness.py) |
| Phase artifacts (per-phase output persistence) | [`phase_artifacts.py`](phase_artifacts.py) |
| S3 signals loader | [`loaders/signal_loader.py`](loaders/signal_loader.py) |
| S3 → yfinance → IBKR price loader chain | [`loaders/price_loader.py`](loaders/price_loader.py) |
| VectorBT portfolio bridge (orders → vbt.Portfolio) | [`vectorbt_bridge.py`](vectorbt_bridge.py) |
| Signal quality accuracy metrics | [`analysis/signal_quality.py`](analysis/signal_quality.py) |
| Regime-conditional analysis | [`analysis/regime_analysis.py`](analysis/regime_analysis.py) |
| Score-bucket analysis | [`analysis/score_analysis.py`](analysis/score_analysis.py) |
| Sub-score correlation + BH FDR | [`analysis/attribution.py`](analysis/attribution.py), [`analysis/stats_utils.py`](analysis/stats_utils.py) |
| Random-search param sweep (6 risk params) | [`analysis/param_sweep.py`](analysis/param_sweep.py) |
| Predictor veto threshold sweep | [`analysis/veto_analysis.py`](analysis/veto_analysis.py) |
| Veto-value diagnostic | [`analysis/veto_value.py`](analysis/veto_value.py) |
| Trigger / shadow / exit / post-trade analyses | [`analysis/{trigger_scorecard,shadow_book,exit_timing,post_trade}.py`](analysis/) |
| Component A-F grading + 52-week history | [`analysis/grading.py`](analysis/grading.py), [`analysis/grade_history.py`](analysis/grade_history.py) |
| Feature drift + retrain alerts | [`analysis/feature_drift.py`](analysis/feature_drift.py), [`analysis/retrain_alert.py`](analysis/retrain_alert.py) |
| Monte Carlo on returns | [`analysis/monte_carlo.py`](analysis/monte_carlo.py) |
| Cost report (LLM cost rolled into evaluator email) | [`analysis/cost_report.py`](analysis/cost_report.py) |
| Production health digest | [`analysis/production_health.py`](analysis/production_health.py) |
| Predictor confusion matrix | [`analysis/predictor_confusion.py`](analysis/predictor_confusion.py) |
| Sizing A/B + alpha-distribution | [`analysis/sizing_ab.py`](analysis/sizing_ab.py), [`analysis/alpha_distribution.py`](analysis/alpha_distribution.py) |
| Macro evaluation | [`analysis/macro_eval.py`](analysis/macro_eval.py) |
| End-to-end test harness | [`analysis/end_to_end.py`](analysis/end_to_end.py) |
| Scoring weight auto-apply (Research) | [`optimizer/weight_optimizer.py`](optimizer/weight_optimizer.py) |
| Executor param auto-apply | [`optimizer/executor_optimizer.py`](optimizer/executor_optimizer.py) |
| Research param auto-apply (deferred) | [`optimizer/research_optimizer.py`](optimizer/research_optimizer.py) |
| Predictor param auto-apply | [`optimizer/predictor_optimizer.py`](optimizer/predictor_optimizer.py) |
| Predictor sizing optimizer | [`optimizer/predictor_sizing_optimizer.py`](optimizer/predictor_sizing_optimizer.py) |
| Trigger / scanner optimizers | [`optimizer/trigger_optimizer.py`](optimizer/trigger_optimizer.py), [`optimizer/scanner_optimizer.py`](optimizer/scanner_optimizer.py) |
| Pipeline-level optimizer | [`optimizer/pipeline_optimizer.py`](optimizer/pipeline_optimizer.py) |
| Regression monitor | [`optimizer/regression_monitor.py`](optimizer/regression_monitor.py) |
| Twin simulation | [`optimizer/twin_sim.py`](optimizer/twin_sim.py) |
| Config rollback mechanism | [`optimizer/rollback.py`](optimizer/rollback.py) |
| Replay harness — single-artifact runner (re-runs captured `DecisionArtifact` under target model via `langchain_anthropic.with_structured_output(SchemaClass)` against the canonical contract) | [`replay/runner.py`](replay/runner.py) |
| Per-agent agreement scorers (cheap-model concordance signal) | [`replay/comparison.py`](replay/comparison.py) |
| Batch replay — date-range × target-models iteration + per-`(agent_id, target_model)` concordance aggregation + CloudWatch `agent_cheap_model_concordance` metric emission + per-target-model summary persistence | [`replay/batch.py`](replay/batch.py) |
| Counterfactual rule fit — per-agent depth-≤3 decision-tree fit on captured `(input → decision)` pairs + CloudWatch `agent_counterfactual_rule_fit` metric (third leg of the agent-justification triple alongside cross-week clustering + cheap-model concordance) | [`replay/counterfactual.py`](replay/counterfactual.py) |
| Replay CLI (`python -m replay.cli {single,batch,counterfactual} ...`) | [`replay/cli.py`](replay/cli.py) |
| Concordance Lambda — weekly SF-driven entry point that wraps `replay.batch.compute_and_emit_concordance` | [`lambda_concordance/handler.py`](lambda_concordance/handler.py), [`lambda_concordance/Dockerfile`](lambda_concordance/Dockerfile), [`infrastructure/deploy_concordance.sh`](infrastructure/deploy_concordance.sh) |
| Counterfactual Lambda — weekly SF-driven entry point that wraps `replay.counterfactual.compute_and_emit` | [`lambda_counterfactual/handler.py`](lambda_counterfactual/handler.py), [`lambda_counterfactual/Dockerfile`](lambda_counterfactual/Dockerfile), [`infrastructure/deploy_counterfactual.sh`](infrastructure/deploy_counterfactual.sh) |
| 10y synthetic predictor backtest | [`synthetic/predictor_backtest.py`](synthetic/predictor_backtest.py) |
| Synthetic signal generator | [`synthetic/signal_generator.py`](synthetic/signal_generator.py) |
| Markdown + CSV + metrics.json + S3 upload | [`reporter.py`](reporter.py) |
| SES email delivery | [`emailer.py`](emailer.py) |
| Health-status writer | [`health_status.py`](health_status.py) |
| Preflight | [`preflight.py`](preflight.py) |
| SSM secret loader | [`ssm_secrets.py`](ssm_secrets.py) |

## Inputs / outputs

### Reads
| Source | Path |
|---|---|
| Research signals (10y replay window) | `s3://alpha-engine-research/signals/{date}/signals.json` |
| Score-performance audit | `s3://alpha-engine-research/research.db` (`score_performance`) |
| Trade audit log + EOD P&L | `s3://alpha-engine-research/trades/trades_full.csv`, `eod_pnl.csv` |
| Price universe (synthetic backtest) | `s3://alpha-engine-research/arcticdb/universe/` |
| Predictor weights + metrics | `s3://alpha-engine-research/predictor/weights/`, `predictor/metrics/latest.json` |
| Per-call LLM cost JSONLs | `s3://alpha-engine-research/decision_artifacts/_cost_raw/{date}/{run_id}/` |

### Writes
| Destination | Path |
|---|---|
| Weekly backtest report (markdown + CSV + metrics) | `s3://alpha-engine-research/backtest/{date}/` |
| 52-week component grade trend | `s3://alpha-engine-research/backtest/grade_history.json` |
| Auto-applied scoring weights | `s3://alpha-engine-research/config/scoring_weights.json` |
| Auto-applied executor params | `s3://alpha-engine-research/config/executor_params.json` |
| Auto-applied predictor params (veto threshold) | `s3://alpha-engine-research/config/predictor_params.json` |
| Auto-applied research params (deferred) | `s3://alpha-engine-research/config/research_params.json` |
| Aggregated cost parquet (rolled into evaluator email) | `s3://alpha-engine-research/decision_artifacts/_cost/{date}/cost.parquet` |

## Run modes

| Mode | Where | Command |
|---|---|---|
| Production weekly | EC2 spot (c5.large) | weekly Step Function via `infrastructure/spot_backtest.sh` |
| Local signal-quality only | venv | `python backtest.py --mode signal-quality` |
| Local full run | venv | `python backtest.py --mode all --upload` |
| Local predictor-only | venv | `python backtest.py --mode predictor-backtest` |
| Local evaluation only | venv | `python evaluate.py --mode all --upload` |
| Local diagnostics (no config promotion) | venv | `python evaluate.py --mode diagnostics` |
| Single eval module | venv | `python evaluate.py --module signal-quality` |
| Freeze (compute but don't write configs) | venv | `python evaluate.py --mode all --freeze` |
| Rollback all configs to previous | venv | `python backtest.py --rollback` |
| Spot smoke test | local | `bash infrastructure/spot_backtest.sh --smoke-only` |

Deploy: `git push origin main`. The weekly SF picks up the latest commit when `spot_backtest.sh` clones `--branch main`. No persistent EC2 host.

## Tests

`pytest tests/` covers signal-quality math, regime-conditional breakdown logic, attribution + BH FDR, param-sweep determinism, optimizer auto-apply guardrails, VectorBT bridge, completeness tracking, predictor synthetic backtest, and rollback mechanism. ~189 tests passing.
