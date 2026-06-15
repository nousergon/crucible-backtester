"""Verify flow-doctor wiring in backtester entrypoints.

Asserts the canonical alpha-engine-lib pattern (module-top setup_logging
+ exclude_patterns plumbed + yaml resolvable from the entrypoint
location) is in place for all three backtester entrypoints:

- ``backtest.py``                 — main simulation CLI (EC2 spot)
- ``evaluate.py``                 — evaluator CLI (EC2 spot)
- ``lambda_health/handler.py``    — daily predictor health-check Lambda

Distinct from the existing ``test_flow_doctor_smoke.py`` which tests
flow_doctor.report() / dedup / scrub / store directly. This file
covers the wiring shape: where setup_logging fires, what env it
loads, whether the Dockerfile ships the yaml.

Includes:
- LAMBDA_TASK_ROOT regression test (lambda_health/handler.py is at
  /var/task/handler.py post-Dockerfile-flatten; honor the env var).
- backtest.py keeps `fd = get_flow_doctor()` in main() because there
  are 7 active fd.report() call sites; evaluate.py + lambda_health
  drop the dead retrieval since they had zero call sites.

Runs without firing any LLM diagnosis: setup_logging is exercised with
FLOW_DOCTOR_ENABLED=1 + stub env vars + a redirected yaml store path,
but no ERROR records are emitted.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def stub_flow_doctor_env(monkeypatch):
    """Populate the env vars that flow-doctor.yaml's ${VAR} refs resolve."""
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    monkeypatch.setenv("EMAIL_SENDER", "test@example.com")
    monkeypatch.setenv("EMAIL_RECIPIENTS", "test@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "stub-password")
    monkeypatch.setenv("FLOW_DOCTOR_GITHUB_TOKEN", "stub-token")


@pytest.fixture
def reset_root_logger():
    """Snapshot + restore root logger handlers around each test."""
    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    root.handlers = saved


@pytest.fixture
def temp_flow_doctor_yaml(tmp_path):
    """Write a copy of the production flow-doctor.yaml with store.path
    redirected into the test's tmp_path AND the github notify channel
    stripped.

    flow_doctor.init() preflights the GitHub token against api.github.com;
    the stub `FLOW_DOCTOR_GITHUB_TOKEN=stub-token` fails the 401 check
    and would error the test. We only need the email channel to verify
    the wiring (FlowDoctorHandler attach + exclude_patterns plumbing)."""
    import yaml as yamllib
    with open(REPO_ROOT / "flow-doctor.yaml") as f:
        cfg = yamllib.safe_load(f)
    cfg["store"]["path"] = str(tmp_path / "flow_doctor_test.db")
    cfg["notify"] = [n for n in cfg.get("notify", []) if n.get("type") != "github"]
    yaml_path = tmp_path / "flow-doctor.yaml"
    with open(yaml_path, "w") as f:
        yamllib.safe_dump(cfg, f)
    return str(yaml_path)


def _flow_doctor_available() -> bool:
    try:
        import flow_doctor  # noqa: F401
        return True
    except ImportError:
        return False


flow_doctor_required = pytest.mark.skipif(
    not _flow_doctor_available(),
    reason="flow-doctor not installed (pip install alpha-engine-lib[flow_doctor])",
)


class TestFlowDoctorYamlPresence:
    """The yaml file each entrypoint resolves must exist at that path."""

    def test_yaml_at_repo_root_exists(self):
        assert (REPO_ROOT / "flow-doctor.yaml").is_file()

    def test_yaml_path_resolved_by_backtest_exists(self):
        # Mirrors backtest.py's path computation:
        #   os.path.dirname(os.path.abspath(__file__))
        # backtest.py sits at the repo root.
        bt_path = REPO_ROOT / "backtest.py"
        resolved = Path(os.path.dirname(os.path.abspath(bt_path))) / "flow-doctor.yaml"
        assert resolved.is_file(), f"backtest.py resolves to {resolved}"

    def test_yaml_path_resolved_by_evaluate_exists(self):
        ev_path = REPO_ROOT / "evaluate.py"
        resolved = Path(os.path.dirname(os.path.abspath(ev_path))) / "flow-doctor.yaml"
        assert resolved.is_file(), f"evaluate.py resolves to {resolved}"

    def test_yaml_path_resolved_by_lambda_health_handler_exists(self):
        # Local-dev path (LAMBDA_TASK_ROOT unset):
        #   dirname(dirname(abspath(__file__)))
        h_path = REPO_ROOT / "lambda_health" / "handler.py"
        resolved = Path(os.path.dirname(os.path.dirname(os.path.abspath(h_path)))) / "flow-doctor.yaml"
        assert resolved.is_file(), f"lambda_health/handler.py resolves to {resolved}"

    def test_yaml_path_resolved_by_lambda_health_handler_under_lambda_runtime(
        self, tmp_path, monkeypatch
    ):
        # Lambda flattens lambda_health/handler.py to /var/task/handler.py
        # via `COPY lambda_health/handler.py handler.py`. Two-dirs-up
        # would land at /var/. The handler must honor LAMBDA_TASK_ROOT.
        fake_task_root = tmp_path / "fake_lambda_task_root"
        fake_task_root.mkdir()
        (fake_task_root / "flow-doctor.yaml").write_text("flow_name: test\n")
        monkeypatch.setenv("LAMBDA_TASK_ROOT", str(fake_task_root))
        resolved = os.path.join(
            os.environ.get(
                "LAMBDA_TASK_ROOT",
                "/should-not-fall-back",
            ),
            "flow-doctor.yaml",
        )
        assert os.path.isfile(resolved), (
            "lambda_health/handler.py must honor LAMBDA_TASK_ROOT"
        )


class TestFlowDoctorYamlSchema:
    """flow-doctor.yaml must declare keys consistent with the lib contract."""

    def test_yaml_has_required_top_level_keys(self):
        import yaml
        with open(REPO_ROOT / "flow-doctor.yaml") as f:
            cfg = yaml.safe_load(f)
        for key in ("flow_name", "repo", "notify", "store", "rate_limits"):
            assert key in cfg, f"missing top-level key: {key}"
        assert cfg["repo"] == "nousergon/crucible-backtester"


@flow_doctor_required
class TestSetupLoggingAttach:
    """setup_logging() should attach FlowDoctorHandler when ENABLED=1.

    Does NOT fire any ERROR records, so flow-doctor's diagnose() / Anthropic
    calls are never invoked.
    """

    def test_disabled_attaches_no_flow_doctor_handler(self, monkeypatch, reset_root_logger):
        monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "0")
        from alpha_engine_lib.logging import setup_logging
        setup_logging(
            "backtester-test-disabled",
            flow_doctor_yaml=str(REPO_ROOT / "flow-doctor.yaml"),
            exclude_patterns=[],
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert attached == []

    def test_enabled_attaches_flow_doctor_handler(
        self, stub_flow_doctor_env, reset_root_logger, temp_flow_doctor_yaml
    ):
        from alpha_engine_lib.logging import setup_logging, get_flow_doctor
        setup_logging(
            "backtester-test-enabled",
            flow_doctor_yaml=temp_flow_doctor_yaml,
            exclude_patterns=[],
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert len(attached) == 1
        assert get_flow_doctor() is not None

    def test_exclude_patterns_plumbed_to_handler(
        self, stub_flow_doctor_env, reset_root_logger, temp_flow_doctor_yaml
    ):
        from alpha_engine_lib.logging import setup_logging
        patterns = [r"vectorbt fold warning", r"arcticdb retry transient"]
        setup_logging(
            "backtester-test-patterns",
            flow_doctor_yaml=temp_flow_doctor_yaml,
            exclude_patterns=patterns,
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert len(attached) == 1
        compiled = attached[0]._exclude_re
        assert [p.pattern for p in compiled] == patterns


class TestEntrypointModuleTopWiring:
    """Each entrypoint must call setup_logging at MODULE-TOP, not inside a
    function. Source-text checks; no flow_doctor.init() side effects.

    Was a real regression in the pre-PR-4 state — backtest.py + evaluate.py
    both had setup_logging inside main(), and lambda_health/handler.py had
    it inside handler(), so module-import errors (cold-start crashes,
    heavy vectorbt + boto3 imports failing) bypassed flow-doctor.
    """

    @staticmethod
    def _index_of(needle: str, text: str) -> int:
        idx = text.find(needle)
        assert idx != -1, f"missing required text: {needle!r}"
        return idx

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        import re
        stripped = re.sub(r'"""[\s\S]*?"""', "", text)
        stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
        return stripped

    def test_backtest_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "backtest.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        main_def_idx = self._index_of("def main(", text)
        assert setup_idx < main_def_idx, (
            "backtest.py setup_logging must be called at module-top, before def main()"
        )
        assert "exclude_patterns=" in text[setup_idx:main_def_idx]
        body = self._strip_comments_and_docstrings(text[main_def_idx:])
        assert "setup_logging(" not in body, (
            "duplicate setup_logging call inside main() — should only run once at module-top"
        )

    def test_evaluate_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "evaluate.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        main_def_idx = self._index_of("def main(", text)
        assert setup_idx < main_def_idx, (
            "evaluate.py setup_logging must be called at module-top, before def main()"
        )
        assert "exclude_patterns=" in text[setup_idx:main_def_idx]
        body = self._strip_comments_and_docstrings(text[main_def_idx:])
        assert "setup_logging(" not in body

    def test_lambda_health_handler_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "lambda_health" / "handler.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        handler_def_idx = self._index_of("def handler(", text)
        assert setup_idx < handler_def_idx, (
            "lambda_health/handler.py setup_logging must be at module-top"
        )
        assert "exclude_patterns=" in text[setup_idx:handler_def_idx]
        body = self._strip_comments_and_docstrings(text[handler_def_idx:])
        assert "setup_logging(" not in body


class TestFlowDoctorRetrievalIsLiveOnlyWhereUsed:
    """backtest.py keeps `fd = get_flow_doctor()` because it has 7 active
    `fd.report()` call sites for param-sweep / simulation / optimizer
    error escalation. evaluate.py + lambda_health/handler.py had zero
    call sites; the dead retrieval was dropped in this PR.

    If a future change adds an fd.report() to evaluate.py or
    lambda_health, restore the get_flow_doctor() retrieval there too.
    """

    def test_backtest_retains_get_flow_doctor_retrieval(self):
        text = (REPO_ROOT / "backtest.py").read_text()
        # Both the import + the retrieval should be present
        assert "get_flow_doctor" in text
        assert "fd = get_flow_doctor()" in text
        # And there should still be active fd.report() call sites that
        # justify keeping the retrieval (defends against accidental
        # cleanup that drops the retrieval but leaves the call sites
        # broken).
        assert text.count("fd.report(") >= 5, (
            "backtest.py should have multiple active fd.report() call sites; "
            "if these were intentionally removed, also drop fd = get_flow_doctor()"
        )

    def test_evaluate_dropped_dead_get_flow_doctor(self):
        text = (REPO_ROOT / "evaluate.py").read_text()
        # No active call sites — retrieval should be gone too.
        # Strip comments so commentary about the drop doesn't false-positive.
        import re
        body = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
        assert "fd = get_flow_doctor()" not in body, (
            "evaluate.py has no fd.report() call sites; drop the retrieval too"
        )
        assert "fd.report(" not in body, (
            "if evaluate.py needs fd.report(), restore the get_flow_doctor() retrieval"
        )

    def test_lambda_health_dropped_dead_get_flow_doctor(self):
        text = (REPO_ROOT / "lambda_health" / "handler.py").read_text()
        import re
        body = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
        assert "fd = get_flow_doctor()" not in body
        assert "fd.report(" not in body


class TestColdStartDeferral:
    """lambda_health/handler.py must NOT call load_secrets() anywhere.

    Post-L2998-PR-9c (2026-05-14): the per-repo ssm_secrets shim is
    deleted. Secrets load via alpha_engine_lib.secrets.get_secret() at
    use-site (per-process cached). The _ensure_init() function is
    retained as a deferred-init hook in case future cold-start work
    needs it, but the load_secrets() body has been stripped.

    Source-text checks; runtime behavior covered by docker smoke
    (not in CI).
    """

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        import re
        stripped = re.sub(r'"""[\s\S]*?"""', "", text)
        stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
        return stripped

    def test_lambda_health_handler_does_not_call_load_secrets(self):
        text = (REPO_ROOT / "lambda_health" / "handler.py").read_text()
        body = self._strip_comments_and_docstrings(text)
        assert "load_secrets()" not in body, (
            "lambda_health/handler.py must not call load_secrets(); "
            "the per-repo ssm_secrets shim is deleted post-PR-9c and "
            "secrets now load via alpha_engine_lib.secrets.get_secret() "
            "at use-site"
        )
        assert "from ssm_secrets" not in body, (
            "lambda_health/handler.py must not import from ssm_secrets; "
            "the shim is deleted post-PR-9c"
        )
        assert "import ssm_secrets" not in body, (
            "lambda_health/handler.py must not import ssm_secrets; "
            "the shim is deleted post-PR-9c"
        )

    def test_lambda_health_handler_calls_ensure_init_first_in_handler(self):
        text = (REPO_ROOT / "lambda_health" / "handler.py").read_text()
        handler_idx = text.find("def handler(")
        assert handler_idx != -1
        body = text[handler_idx:]
        ensure_call_idx = body.find("_ensure_init()")
        # _ensure_init() is retained as a deferred-init hook (currently
        # a no-op stub post-PR-9c). Preserve the call-shape so future
        # cold-start work has a wired entry point.
        preflight_idx = body.find("BacktesterPreflight(")
        assert ensure_call_idx != -1, (
            "handler() must call _ensure_init() (deferred-init hook)"
        )
        if preflight_idx != -1:
            assert ensure_call_idx < preflight_idx, (
                "_ensure_init() must run before BacktesterPreflight"
            )


class TestLibVersionPin:
    """alpha-engine-lib must be pinned to a stable tag, not a floating
    branch. Drift = silent breakage class.
    """

    def test_requirements_pins_lib_to_stable_tag(self):
        text = (REPO_ROOT / "requirements.txt").read_text()
        # Either tagged version, or unpinned via @main (we explicitly
        # forbid @main here — it floats and breaks reproducible builds).
        assert "@main" not in text, "alpha-engine-lib must be pinned to a tag, not @main"
        assert "@v0.59.3" in text, (
            "alpha-engine-lib should pin to v0.59.3 — flow-doctor pytest-guard "
            "fix (lib#114 keys activation on 'pytest' in sys.modules, closing "
            "the collection-time test-alert-leakage gap; config#996). Prior: "
            "v0.58.0 flow-doctor default-on "
            "roll (lib v0.58.0 makes flow-doctor activate when setup_logging is "
            "called with a flow_doctor_yaml; adds guard_entrypoint/monitor_handler "
            "crash-capture helpers). Bumped from the prior v0.53.0 fleet-alignment "
            "pin (2026-06-06, L4513). "
            "Update this test if the pin moves further forward."
        )
