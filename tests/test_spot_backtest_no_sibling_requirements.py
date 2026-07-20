"""config#3031 class guard: the spot deps step must never install a sibling
repo's requirements.txt into the backtester's environment.

Root cause class (three instances to date): two repos' requirements files
pip-installed SEQUENTIALLY into one resolver namespace let whichever file
installs second silently override the first's pins — nousergon-lib downgrade
(L4513), pyarrow 18->25 drift, and finally numpy: crucible-predictor's
``numpy>=2.5.1`` floor (installed second) overrode the backtester's
``numpy>=2.0,<2.5`` cap (numba/vectorbt hard ceiling), failing every weekly
deps step on 2026-07-20. The fix is structural: predictor's checkout is
CODE-ONLY (sys.path import target), and every library that code needs at
runtime is declared in the backtester's own requirements.txt (single file,
single resolve). This test makes the co-install unreintroducible.
"""

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
REQS = Path(__file__).resolve().parent.parent / "requirements.txt"


def _script() -> str:
    return SCRIPT.read_text()


def test_exactly_one_requirements_install():
    """One `pip install ... -r requirements.txt` — the backtester's own."""
    s = _script()
    installs = [
        line for line in s.splitlines()
        if "install" in line and "-r requirements.txt" in line and "pip" in line.lower()
    ]
    assert len(installs) == 1, (
        f"expected exactly ONE requirements.txt install in the deps step, found "
        f"{len(installs)}: {installs!r} — a second one reintroduces the "
        "config#3031 co-resolve class"
    )


def test_no_install_after_predictor_cd():
    """No pip install of any kind may run after cd'ing into the predictor
    checkout — that checkout is a code-only sys.path dependency."""
    s = _script()
    pred_idx = s.index("cd /home/ec2-user/alpha-engine-predictor")
    tail = s[pred_idx:]
    # the guards section legitimately follows; only installs are forbidden
    offending = [
        line for line in tail.splitlines()
        if "pip" in line.lower() and "install" in line
        and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        f"pip install found after the predictor cd (code-only checkout): "
        f"{offending!r}"
    )


def test_backtester_requirements_declare_predictor_replay_deps():
    """The libraries the in-process predictor replay imports must be declared
    HERE (consumer-declared deps) — they used to arrive via the co-install."""
    req = REQS.read_text()
    for pkg in ("lightgbm", "catboost", "hmmlearn", "scipy", "optuna", "tqdm", "psutil"):
        assert pkg in req, (
            f"requirements.txt no longer declares {pkg!r} — the predictor "
            "replay chain (synthetic/predictor_backtest.py, research-free "
            "backfill) imports it at runtime and there is no co-install to "
            "provide it anymore (config#3031)"
        )


def test_numpy_cap_still_present():
    """The numba/vectorbt ceiling must survive future edits — it is the pin
    the co-install used to silently violate."""
    req = REQS.read_text()
    assert "numpy>=2.0,<2.5" in req, (
        "requirements.txt lost the numpy>=2.0,<2.5 cap (numba 0.66 hard-caps "
        "at numpy<2.5; vectorbt imports break without it — config#2975/#2976)"
    )
