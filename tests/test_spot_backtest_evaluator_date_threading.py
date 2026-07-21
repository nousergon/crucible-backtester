"""config#3133 regression: the evaluator stage must thread --date "${RUN_DATE}".

evaluate.py silently defaulted to its own date.today() because the evaluator
invocation lacked --date (while a comment in the backtest stage claimed it was
threaded). Invisible on weekend runs (trading-day normalization landed on the
same Friday by coincidence); the first WEEKDAY recovery rerun
(watch-rerun-2026-07-18-12, 2026-07-20) resolved to Monday, probed
backtest/2026-07-20/, and hard-failed on missing artifacts. Every stage that
writes or reads backtest/{date}/ must key off the single dispatcher-resolved
RUN_DATE — this test pins the evaluator's half of that contract.
"""

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"


def test_evaluator_invocation_threads_run_date():
    s = SCRIPT.read_text()
    m = re.search(r"evaluate\.py --mode all[^\n]*", s)
    assert m, "evaluator stage invocation (evaluate.py --mode all) not found"
    line = m.group(0)
    assert '--date "\\${RUN_DATE}"' in line, (
        f"the evaluator invocation must pass --date \"${{RUN_DATE}}\" — without "
        f"it evaluate.py keys off its own date.today() and misses the run's "
        f"backtest/{{date}}/ artifacts on any weekday rerun (config#3133). "
        f"Found: {line!r}"
    )


def test_all_backtest_prefix_stages_thread_run_date():
    """Every python entry point in the full-backtest heredoc that reads or
    writes backtest/{date}/ must carry --date "${RUN_DATE}"."""
    s = SCRIPT.read_text()
    # join backslash line-continuations so multi-line invocations (pit_parity)
    # are scanned as one logical command
    joined = re.sub(r"\\\\\n\s*", " ", s)
    entry_lines = [
        line for line in joined.splitlines()
        if re.search(r"(backtest|evaluate)\.py --mode ", line)
        and "smoke" not in line.lower()
        and not line.lstrip().startswith("#")
    ]
    assert entry_lines, "no full-run entry-point invocations found"
    missing = [l.strip() for l in entry_lines if '--date' not in l]
    assert not missing, (
        f"entry points missing --date threading (date-split class, "
        f"2026-05-17 + config#3133): {missing!r}"
    )
