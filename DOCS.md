# alpha-engine-backtester — Full Documentation

---

## Contents

1. [Setup](#1-setup)
2. [Configuration](#2-configuration)
3. [Data sources](#3-data-sources)
4. [Mode 1 — Signal quality](#4-mode-1--signal-quality)
5. [Mode 2 — Portfolio simulation](#5-mode-2--portfolio-simulation)
6. [Param sweep](#6-param-sweep)
7. [Weight optimizer](#7-weight-optimizer)
8. [vectorbt metrics reference](#8-vectorbt-metrics-reference)
9. [Reporter output](#9-reporter-output)
10. [EC2 deployment](#10-ec2-deployment)
11. [IAM policy](#11-iam-policy)
12. [Development workflow](#12-development-workflow)

---

## 1. Setup

### Prerequisites

- Python 3.11+
- AWS credentials with access to the research S3 bucket (see [§11 IAM policy](#11-iam-policy))
- The [alpha-engine-research](https://github.com/nousergon/crucible-research) pipeline running and writing to S3
- The [alpha-engine](https://github.com/nousergon/crucible-executor) executor repo cloned locally (required for Mode 2 and param sweep)
- The [alpha-engine-research](https://github.com/nousergon/crucible-research) repo cloned locally (used to read current scoring weights)

### Install

```bash
git clone https://github.com/nousergon/crucible-backtester.git
cd alpha-engine-backtester
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your S3 bucket name, local repo paths, and email address.

---

## 2. Configuration

All settings live in `config.yaml` (gitignored). Use `config.yaml.example` as the template.

```yaml
# S3 bucket written by alpha-engine-research
signals_bucket: alpha-engine-research

# S3 bucket and prefix for backtest output
output_bucket: alpha-engine-research
output_prefix: backtest

# Local results directory
results_dir: results

# Mode 2 simulation parameters
init_cash: 1_000_000.0
min_simulation_dates: 5

# Path to alpha-engine repo — required for Mode 2 / param sweep
# First existing path wins; supports local dev and EC2 without changes.
executor_paths:
  - /path/to/local/alpha-engine
  - /home/ec2-user/alpha-engine

# Path to alpha-engine-research repo — used to read current scoring weights
# from config/universe.yaml automatically. First existing path wins.
research_paths:
  - /path/to/local/alpha-engine-research
  - /home/ec2-user/alpha-engine-research

# Minimum samples with beat_spy_10d populated before weight optimizer runs
weight_optimizer_min_samples: 30

# Score thresholds for Mode 1 accuracy-vs-threshold table
score_thresholds: [60, 65, 70, 75, 80, 85, 90]

# Minimum samples required before a bucket appears in the report
min_samples: 5

# Email (AWS SES)
email_sender: "you@example.com"
email_recipients:
  - "you@example.com"

# research.db is pulled from S3 automatically — no path needed.
# Override with --db flag for local development.

# Param sweep grid
param_sweep:
  min_score: [65, 70, 75, 80]
  max_position_pct: [0.05, 0.10, 0.15]
  drawdown_circuit_breaker: [0.10, 0.15, 0.20]
```

### research.db

`research.db` is a SQLite database maintained by the alpha-engine-research Lambda. At the start of every backtester run, `backtest.py` pulls a fresh copy from `s3://{signals_bucket}/research.db` into a temp file. The backtester never writes to it.

Override with `--db /path/to/research.db` to use a local copy during development:

```bash
aws s3 cp s3://alpha-engine-research/research.db ./research.db
python backtest.py --mode signal-quality --db ./research.db
```

---

## 3. Data sources

### Signal files

Written daily by alpha-engine-research at `s3://{bucket}/signals/{date}/signals.json`:

```json
{
  "date": "2026-03-09",
  "market_regime": "neutral",
  "signals": {
    "PLTR": {
      "ticker": "PLTR",
      "signal": "ENTER",
      "rating": "BUY",
      "score": 82,
      "conviction": "rising",
      "quant_score": 85,
      "qual_score": 79,
      "sub_scores": {"quant": 85, "qual": 79}
    }
  },
  "universe": [...],
  "buy_candidates": [...]
}
```

### Price files

Written daily by alpha-engine-research at `s3://{bucket}/prices/{date}/prices.json`:

```json
{
  "date": "2026-03-09",
  "prices": {
    "PLTR": {"open": 84.12, "close": 85.47, "high": 86.10, "low": 83.90}
  }
}
```

### Price fallback chain

`price_loader.py` resolves prices in this order for any date:

1. **S3** `prices/{date}/prices.json` — canonical source
2. **yfinance** — tickers resolved automatically from the corresponding `signals.json`
3. **IBKR `reqHistoricalData`** — optional; pass `ibkr_client=` to `build_matrix()` for gap-filling

Price data is available for all historical signal dates regardless of when `prices.json` files started being written.

### research.db schema (relevant tables)

```sql
-- One row per BUY signal. beat_spy_10d/30d populated ~10/30 trading days later.
score_performance (
    symbol, score_date, score, price_on_date,
    price_10d, price_30d,
    spy_10d_return, spy_30d_return,
    return_10d, return_30d,
    beat_spy_10d, beat_spy_30d,   -- NULL until evaluation date passes
    eval_date_10d, eval_date_30d
)

-- Daily macro snapshot written by research pipeline
macro_snapshots (
    date, market_regime,          -- "bull" | "neutral" | "bear" | "caution"
    fed_funds_rate, treasury_10yr, yield_curve_slope,
    vix, sp500_close, sp500_30d_return, ...
)

-- Full investment thesis per stock per day (includes sub-scores)
investment_thesis (
    symbol, date, rating, score,
    technical_score, quant_score, qual_score,
    conviction, signal, ...
)
```

---

## 4. Mode 1 — Signal quality

Reads `score_performance` from `research.db`, aggregates accuracy metrics, and runs the weight optimizer.

### Run

```bash
python backtest.py --mode signal-quality
python backtest.py --mode signal-quality --upload    # + S3 upload + email
```

### What it computes

| Metric | Description |
|--------|-------------|
| `accuracy_10d` | % of BUY signals where `beat_spy_10d = True` |
| `accuracy_30d` | % of BUY signals where `beat_spy_30d = True` |
| `avg_alpha_10d` | Mean of `return_10d - spy_10d_return` across all signals |
| `avg_alpha_30d` | Mean of `return_30d - spy_30d_return` |

Slices computed: overall, by score bucket (60–70, 70–80, 80–90, 90+), by conviction, by market regime.

### Interpretation

| accuracy_10d | Interpretation |
|---|---|
| < 50% | Signals are subtracting value |
| ~50% | Random — no edge |
| 55–60% | Meaningful edge |
| > 60% | Strong edge |

50 samples is the minimum for meaningful accuracy estimates. Results before ~Week 4 (~200 `score_performance` rows with 10d returns) will return `insufficient_data` and are expected.

### Score threshold analysis

`analysis/score_analysis.py` computes accuracy for every threshold in `score_thresholds`. Use this to find the optimal `min_score` cutoff — the point where raising the bar improves accuracy faster than it shrinks sample size.

### Attribution

`analysis/attribution.py` computes correlation between each sub-score (quant, qual) and `beat_spy_10d`. This feeds directly into the weight optimizer — see [§7](#7-weight-optimizer).

---

## 5. Mode 2 — Portfolio simulation

Replays all historical signal dates through the executor's logic and builds a vectorbt portfolio.

### Run

```bash
python backtest.py --mode simulate
```

### How it works

For each historical date with signals in S3:

1. Load `signals.json` from S3
2. Resolve prices from the price matrix (S3 → yfinance fallback)
3. Run `executor.main.run(simulate=True, ibkr_client=sim_client, signals_override=signals)`
4. Collect orders — the same risk guard, position sizer, and conviction logic runs unchanged

A single `SimulatedIBKRClient` is maintained across all dates so positions and NAV accumulate naturally (entering on Day 1 means the position is still open on Day 2).

Orders are then passed to `vectorbt_bridge.orders_to_portfolio()` for analytics. See [§8](#8-vectorbt-metrics-reference) for available metrics.

### Data requirement

Returns `{"status": "insufficient_data"}` if fewer than `min_simulation_dates` (default 5) signal dates are available in S3.

---

## 6. Param sweep

Runs Mode 2 across a grid of executor risk parameters to find the combination with the best Sharpe ratio.

### Run

```bash
python backtest.py --mode param-sweep
```

### How it works

The price matrix is built **once** from S3/yfinance. For each combination of `min_score`, `max_position_pct`, and `drawdown_circuit_breaker` in the grid, the simulation loop re-runs with a fresh `SimulatedIBKRClient` using those parameters. Results are sorted by Sharpe ratio.

Default grid (36 combinations):

```yaml
param_sweep:
  min_score: [65, 70, 75, 80]
  max_position_pct: [0.05, 0.10, 0.15]
  drawdown_circuit_breaker: [0.10, 0.15, 0.20]
```

Output: top 10 combinations in the report, full results in `param_sweep.csv`.

---

## 7. Weight optimizer

Automatically updates scoring weights in the research Lambda based on which sub-scores (quant / qual) best predict outperformance.

### How it works

1. **Sub-score join** — `score_performance` outcomes (from `research.db`) are joined with sub-scores from `signals.json` in S3 for each signal date. This gives a dataset of `(symbol, date, quant_score, qual_score, beat_spy_10d, beat_spy_30d)`.

2. **Correlation** — each sub-score is correlated with `beat_spy_10d` (60% weight) and `beat_spy_30d` (40% weight).

3. **Weight suggestion** — correlation scores are normalized to sum to 1.0, then blended conservatively with the current weights (30% data signal, 70% current weights) to avoid instability.

4. **Autonomous application** — if all guardrails pass, the suggested weights are written to `s3://{bucket}/config/scoring_weights.json`. The research Lambda reads this file at cold-start and uses it in place of `universe.yaml` defaults — **no Lambda redeployment needed**.

### Guardrails

All three must pass before weights are applied:

| Guardrail | Threshold |
|-----------|-----------|
| Confidence | `medium` or `high` (≥50 samples with `beat_spy_10d` populated) |
| Max single change | ≤ 15 percentage points |
| Min meaningful change | At least one weight changes by ≥ 2% |

If any guardrail fails, the run is a no-op and the report explains why.

### Feedback loop

```
Sunday backtester run
  → correlate sub-scores vs. beat_spy outcomes
  → if guardrails pass: write new weights to S3
       ↓
Monday Lambda cold-start
  → aggregator._get_weights() reads S3 override
  → new weights used for that day's scoring
       ↓
Following Sunday backtester run
  → measures impact of updated weights
  → adjusts again if warranted
```

### Current scoring weights

Read automatically from `alpha-engine-research/config/universe.yaml` via `research_paths` in `config.yaml`. No manual sync required — the "Current" column in the weekly report always reflects the live values.

### Confidence levels

| Level | Samples | Meaning |
|-------|---------|---------|
| `low` | < 50 | Too noisy — weights not updated |
| `medium` | 50–199 | Updates enabled with conservative blending |
| `high` | 200+ | Full confidence — blending still applied for stability |

---

## 8. vectorbt metrics reference

### Building the portfolio

```python
from loaders import price_loader, signal_loader
from vectorbt_bridge import orders_to_portfolio, portfolio_stats

# 1. Get all available signal dates
dates = signal_loader.list_dates(bucket="alpha-engine-research")

# 2. Build price matrix (S3 → yfinance fallback)
prices = price_loader.build_matrix(dates, bucket="alpha-engine-research")
# DataFrame: rows = datetime index, columns = ticker symbols

# 3. Run simulation to get orders
from backtest import load_config, run_simulate
config = load_config("config.yaml")
stats = run_simulate(config)

# 4. Or build the portfolio directly
# orders = [{"date": "2026-03-09", "ticker": "PLTR",
#            "action": "ENTER", "shares": 100, "price_at_order": 84.12}, ...]
# pf = orders_to_portfolio(orders, prices, init_cash=1_000_000.0)
```

### Key metrics

```python
# ── Returns ──────────────────────────────────────────────────────────────────
pf.total_return()           # float — total return over full period
pf.annualized_return()      # float — annualized
pf.daily_returns()          # pd.Series — daily P&L %

# ── Risk-adjusted ─────────────────────────────────────────────────────────────
pf.sharpe_ratio()           # float — higher is better; >1 is good, >2 is excellent
pf.sortino_ratio()          # float — like Sharpe but penalises downside only
pf.calmar_ratio()           # float — annualized return / max drawdown
pf.omega_ratio()            # float — probability-weighted return ratio

# ── Drawdown ──────────────────────────────────────────────────────────────────
pf.max_drawdown()           # float — worst peak-to-trough decline (negative)
pf.drawdown()               # pd.Series — rolling drawdown over time

# ── Trades ────────────────────────────────────────────────────────────────────
pf.trades.count()           # int — total number of closed trades
pf.trades.win_rate()        # float — % of trades that were profitable
pf.trades.avg_pnl()         # float — average P&L per trade
pf.trades.records_readable  # DataFrame — one row per trade, fully annotated

# ── Portfolio value ───────────────────────────────────────────────────────────
pf.value()                  # pd.Series — portfolio NAV over time
pf.cash()                   # pd.Series — uninvested cash over time

# ── Benchmark comparison (SPY) ────────────────────────────────────────────────
import vectorbt as vbt
import yfinance as yf

spy = yf.download("SPY", start=dates[0], end=dates[-1], auto_adjust=True)
spy_pf = vbt.Portfolio.from_holding(spy["Close"], init_cash=1_000_000.0)

print(f"Portfolio Sharpe: {pf.sharpe_ratio():.2f}")
print(f"SPY Sharpe:       {spy_pf.sharpe_ratio():.2f}")
print(f"Portfolio return: {pf.total_return()*100:.1f}%")
print(f"SPY return:       {spy_pf.total_return()*100:.1f}%")
```

### Interactive plots

```python
pf.plot().show()                                              # full dashboard
pf.value().vbt.plot(trace_kwargs=dict(name="Portfolio")).show()  # NAV over time
pf.drawdown().vbt.plot().show()                               # drawdown chart
pf.trades.plot().show()                                       # trade waterfall
```

### Summary dict

```python
from vectorbt_bridge import portfolio_stats
stats = portfolio_stats(pf)
# {
#   "total_return": 0.142,
#   "sharpe_ratio": 1.38,
#   "max_drawdown": -0.067,
#   "calmar_ratio": 2.11,
#   "total_trades": 34,
#   "win_rate": 0.59
# }
```

---

## 9. Reporter output

Every run produces files in `results/{date}/`:

| File | Contents |
|------|----------|
| `report.md` | Full markdown report — signal quality, regime, attribution, portfolio stats, weight recommendation, param sweep |
| `signal_quality.csv` | Accuracy by score threshold — one row per threshold |
| `param_sweep.csv` | All param sweep combinations sorted by Sharpe ratio |
| `metrics.json` | Overall summary — status, accuracy_10d/30d, avg_alpha |

With `--upload`, all files are also written to `s3://{output_bucket}/{output_prefix}/{date}/`.

With `email_sender` configured, an HTML-formatted email is sent via SES. Subject line format:

```
Alpha Engine Backtester | 2026-03-09 | results ready
Alpha Engine Backtester | 2026-03-09 | insufficient data (accumulating)
Alpha Engine Backtester | 2026-03-09 | ERROR
```

### Weight recommendation section

The weekly email always includes the weight optimizer output:

- **Applied** — new weights written to S3; Lambda picks them up on next cold-start
- **Not applied** — reason shown (e.g., "confidence too low", "all changes < 2%")

Current weights are read live from `universe.yaml` in the research repo — no manual sync needed.

---

## 10. EC2 deployment

The backtester runs on the same EC2 instance as the executor.

### First-time setup

```bash
# 1. SSH in and configure GitHub credentials (one-time)
ae
cat >> ~/.netrc << 'EOF'
machine github.com
login your-github-username
password your-pat
EOF
chmod 600 ~/.netrc

# 2. Clone and set up
git clone https://github.com/nousergon/crucible-backtester.git
bash ~/alpha-engine-backtester/infrastructure/setup-ec2.sh
```

`setup-ec2.sh` creates the virtualenv, installs dependencies, creates `/var/log/backtester.log`, and registers the Sunday cron job.

### Deploying updates

```bash
# From local machine (alpha-engine-backtester repo)
git push origin main && ae "cd ~/alpha-engine-backtester && git pull"
```

### Cron job

```
0 14 * * 0   cd /home/ec2-user/alpha-engine-backtester && \
             .venv/bin/python backtest.py --mode all --upload \
             >> /var/log/backtester.log 2>&1
```

Sundays, 14:00 UTC = 9:00am ET = 6:00am PT.

### Logs

```bash
ae "tail -50 /var/log/backtester.log"
```

---

## 11. IAM policy

The EC2 instance role (`alpha-engine-executor-role`) requires these S3 permissions:

```json
{
  "Statement": [
    {
      "Sid": "ReadResearchSignals",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::alpha-engine-research",
        "arn:aws:s3:::alpha-engine-research/signals/*"
      ]
    },
    {
      "Sid": "ReadResearchDb",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::alpha-engine-research/research.db"]
    },
    {
      "Sid": "WriteBacktestResults",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::alpha-engine-research/backtest/*"]
    },
    {
      "Sid": "WriteScoringWeights",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::alpha-engine-research/config/scoring_weights.json"]
    }
  ]
}
```

`ses:SendEmail` is required for email reports. The Lambda execution role also needs `s3:GetObject` on `config/scoring_weights.json` to read weight overrides at cold-start.

---

## 12. Development workflow

### Run locally against a pulled research.db

```bash
aws s3 cp s3://alpha-engine-research/research.db ./research.db
python backtest.py --mode signal-quality --db ./research.db
python backtest.py --mode signal-quality --db ./research.db --log-level DEBUG
```

### Inspect vectorbt output interactively

```python
import yaml
from backtest import load_config, run_simulate
from vectorbt_bridge import portfolio_stats

config = load_config("config.yaml")
stats = run_simulate(config)
print(stats)
# To access the portfolio object directly, call _setup_simulation + _run_simulation_loop
```

### Check score_performance directly

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("research.db")
df = pd.read_sql("SELECT * FROM score_performance ORDER BY score_date", conn)
print(f"Total rows: {len(df)}")
print(f"beat_spy_10d populated: {df['beat_spy_10d'].notna().sum()}")
print(f"beat_spy_30d populated: {df['beat_spy_30d'].notna().sum()}")

populated = df[df["beat_spy_10d"].notna()]
print(f"10d accuracy: {populated['beat_spy_10d'].mean():.1%}")
```

### Check current scoring weights in S3

```bash
# View the active weight override (if any)
aws s3 cp s3://alpha-engine-research/config/scoring_weights.json - | python -m json.tool

# Remove the override to revert to universe.yaml defaults
aws s3 rm s3://alpha-engine-research/config/scoring_weights.json
```

### Data availability timeline

| Milestone | Approx. date | What becomes available |
|-----------|--------------|------------------------|
| Pipeline start | 2026-03-05 | signals.json, investment_thesis |
| Week 4 (~2026-03-20) | +10 trading days | First `beat_spy_10d` values; Mode 1 meaningful |
| Week 8 (~2026-05-01) | +30 trading days | First `beat_spy_30d` values; attribution reliable |
| Month 3+ | +~60 trading days | Weight optimizer at medium confidence (50+ samples) |
| Month 6+ | +~120 trading days | Weight optimizer at high confidence (200+ samples) |
