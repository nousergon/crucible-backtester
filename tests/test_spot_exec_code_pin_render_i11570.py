"""A spot worker runs the SF execution's pinned code SHA, not main at its own
launch (alpha-engine-config-I11570).

The worker bootstrap clones ``--branch main``, so a reclaim relaunch or a
re-issued stage used to clone whatever main had become by then: one execution,
two code SHAs. The launcher box already holds one SHA per (execution, checkout)
under ``$AE_EXEC_PIN_ROOT/<RUN_TOKEN>/<checkout basename>`` (nousergon-data
``infrastructure/exec_code_pin.sh``). ``infrastructure/_exec_code_pin_render.sh``
turns those pins into ``krepis.spot_bootstrap render --pin-sha`` arguments, and
both renderer call sites pass them.

The helper is exercised by EXECUTING it in bash against a temp pin root, and
the wiring is derived from the launchers' own render argv, so a checkout added
to the render without a pin lookup fails here.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HELPER = _REPO / "infrastructure" / "_exec_code_pin_render.sh"
_LAUNCHERS = (
    _REPO / "infrastructure" / "_spot_common.sh",
    _REPO / "infrastructure" / "spot_backtest.sh",
)

_SHA_A = "a" * 40
_SHA_B = "0123456789abcdef0123456789abcdef01234567"
_CHECKOUTS = (
    "/home/ec2-user/alpha-engine-backtester",
    "/home/ec2-user/alpha-engine",
    "/home/ec2-user/alpha-engine-predictor",
)


def _fake_python(tmp_path: Path, *, supports_pin_sha: bool) -> Path:
    """A LIB_PYTHON stand-in whose `render --help` does or does not list --pin-sha."""
    flag = "  --pin-sha CHECKOUT=SHA\n" if supports_pin_sha else ""
    script = tmp_path / ("py-pin" if supports_pin_sha else "py-nopin")
    script.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'usage: render [--checkout CHECKOUT] [--extra-clone SPEC]\\n%s' "
        f"'{flag}'\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _run(tmp_path: Path, *, env_extra: dict[str, str], supports_pin_sha: bool = True):
    pin_root = tmp_path / "pins"
    env = {
        "PATH": os.environ["PATH"],
        "AE_EXEC_PIN_ROOT": str(pin_root),
        "LIB_PYTHON": str(_fake_python(tmp_path, supports_pin_sha=supports_pin_sha)),
        **env_extra,
    }
    prog = (
        "set -euo pipefail\n"
        f'source "{_HELPER}"\n'
        f"exec_code_pin_render_args {' '.join(_CHECKOUTS)}\n"
        'printf "%s\\n" ${EXEC_PIN_RENDER_ARGS[@]+"${EXEC_PIN_RENDER_ARGS[@]}"}\n'
    )
    return subprocess.run(
        ["bash", "-c", prog], env=env, capture_output=True, text=True, timeout=30
    )


def _write_pin(tmp_path: Path, token: str, basename: str, value: str) -> None:
    d = tmp_path / "pins" / token
    d.mkdir(parents=True, exist_ok=True)
    (d / basename).write_text(value + "\n")


def test_pinned_checkouts_become_pin_sha_args_and_unpinned_ones_are_announced(tmp_path):
    _write_pin(tmp_path, "rehearsal-x-1", "alpha-engine-backtester", _SHA_A)
    _write_pin(tmp_path, "rehearsal-x-1", "alpha-engine-predictor", _SHA_B)
    r = _run(tmp_path, env_extra={"RUN_TOKEN": "rehearsal-x-1"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [
        "--pin-sha", f"/home/ec2-user/alpha-engine-backtester={_SHA_A}",
        "--pin-sha", f"/home/ec2-user/alpha-engine-predictor={_SHA_B}",
    ]
    # The executor checkout has no pin in this execution: it is cloned at main,
    # and the launcher SAYS so rather than leaving it to be inferred.
    assert "no pin for /home/ec2-user/alpha-engine in execution rehearsal-x-1" in r.stderr


def test_a_pin_from_another_execution_is_not_used(tmp_path):
    _write_pin(tmp_path, "rehearsal-x-1", "alpha-engine-backtester", _SHA_A)
    r = _run(tmp_path, env_extra={"RUN_TOKEN": "rehearsal-x-2"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == []


@pytest.mark.parametrize(
    "env_extra",
    [{}, {"RUN_TOKEN": "rehearsal-x-1", "BRANCH": "feature/x"}],
    ids=["no-run-token", "explicit-branch"],
)
def test_no_execution_or_an_explicit_branch_pins_nothing(tmp_path, env_extra):
    _write_pin(tmp_path, "rehearsal-x-1", "alpha-engine-backtester", _SHA_A)
    r = _run(tmp_path, env_extra=env_extra)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == []


def test_a_malformed_pin_fails_instead_of_falling_back_to_main(tmp_path):
    _write_pin(tmp_path, "rehearsal-x-1", "alpha-engine-backtester", "main")
    r = _run(tmp_path, env_extra={"RUN_TOKEN": "rehearsal-x-1"})
    assert r.returncode != 0
    assert "not a 40-hex SHA" in r.stderr


def test_a_pin_with_a_renderer_that_lacks_pin_sha_fails_naming_the_floor(tmp_path):
    _write_pin(tmp_path, "rehearsal-x-1", "alpha-engine-backtester", _SHA_A)
    r = _run(tmp_path, env_extra={"RUN_TOKEN": "rehearsal-x-1"}, supports_pin_sha=False)
    assert r.returncode != 0
    assert "krepis>=0.59.75" in r.stderr


def test_no_pin_never_probes_the_renderer(tmp_path):
    # An ad-hoc run on an older launcher venv must keep working unchanged.
    r = _run(tmp_path, env_extra={"RUN_TOKEN": "adhoc-1"}, supports_pin_sha=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == []


def _render_block(text: str) -> str:
    start = text.index("-m krepis.spot_bootstrap render")
    end = text.index(')"', start)
    return text[start:end]


def _lookup_block(text: str) -> str:
    render_at = text.index("-m krepis.spot_bootstrap render")
    call_at = text.rindex("exec_code_pin_render_args", 0, render_at)
    return text[call_at:render_at]


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_every_rendered_checkout_is_looked_up_and_the_args_reach_the_render(launcher):
    text = launcher.read_text()
    assert "_exec_code_pin_render.sh" in text, f"{launcher.name} does not source the helper"
    render = _render_block(text)
    rendered = re.findall(r"--checkout\s+(\S+)", render) + re.findall(
        r'--extra-clone\s+"([^="]+)=', render
    )
    assert sorted(rendered) == sorted(_CHECKOUTS)
    looked_up = re.findall(r"/home/ec2-user/[\w.-]+", _lookup_block(text))
    assert sorted(looked_up) == sorted(rendered), (
        f"{launcher.name}: every --checkout/--extra-clone the worker clones must be "
        "passed to exec_code_pin_render_args, or that checkout runs main's tip"
    )
    assert '${EXEC_PIN_RENDER_ARGS[@]+"${EXEC_PIN_RENDER_ARGS[@]}"}' in render
