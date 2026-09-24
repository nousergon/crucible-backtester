"""Add project root to sys.path so optimizer.* and analysis.* imports work,
plus centralized arcticdb stubbing for unit tests.

Also pins ``ALPHA_ENGINE_SECRETS_SOURCE=env`` for the test process so
``nousergon_lib.secrets.get_secret()`` (post 2026-05-12 .env→SSM
migration, PR 5 of the arc) reads from monkeypatched env vars only —
never real SSM. Set at module-import time so the toggle is in place
before any test module imports emailer.py / analysis/retrain_alert.py.
"""
import importlib
import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Pin secrets source BEFORE any test module imports a get_secret() caller.
os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")

# Put the crucible-executor sibling checkout (if resolvable) on sys.path
# HERE, before any test module is collected (alpha-engine-config-I7619).
# Several test_*.py files import from `executor.*` and previously did this
# insert at their own module level — whether `executor` was importable when
# test_actionable_signals_parity.py ran its `from executor.signal_reader
# import get_actionable_signals` then depended on pytest's alphabetical
# collection order (files starting after "a" inserted too late to help it).
# conftest.py always loads first, so this ordering bug can't recur.
from tests._sibling_checkout import ensure_executor_on_sys_path  # noqa: E402

ensure_executor_on_sys_path()

# Stub arcticdb by default for all unit tests — they must never hit real S3,
# and CI (GitHub Actions) has no AWS credentials, so real arcticdb calls
# would 403 (observed 2026-04-24 CI on PR #76). Integration tests that need
# the real module (parity replay on the spot) set USE_REAL_ARCTICDB=1 before
# invoking pytest; spot_backtest.sh's parity stage passes this env var.
#
# History: this lived as an unconditional sys.modules.setdefault inside
# test_parity_replay.py — which ran at module-import time and silently
# shadowed the real arcticdb for the parity integration test itself,
# producing a false-positive "ArcticDB universe library returned 0 symbols"
# failure (MagicMock.list_libraries() iterates to []). Moving it here
# centralizes the stub and lets integration tests opt in.
if not os.environ.get("USE_REAL_ARCTICDB"):
    sys.modules.setdefault("arcticdb", MagicMock())


@pytest.fixture(autouse=True)
def _isolate_secrets_from_ssm(monkeypatch):
    """Re-pin ``ALPHA_ENGINE_SECRETS_SOURCE=env`` per test + clear the
    per-process secret cache. See
    ``alpha-engine-docs/private/env-to-ssm-260512.md`` § Risks.
    """
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    try:
        from nousergon_lib.secrets import clear_cache
    except ImportError:
        yield
        return
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _block_real_alerts_publish(monkeypatch):
    """Default-deny real ``nousergon_lib.alerts.publish`` for every test.

    History: 2026-05-21 a buggy monkeypatch in test_cost_report.py let the
    real publish through on a failing test run, firing a real Telegram
    alert + (likely) SNS publish to the operator. The corrected test pins
    the publish via ``monkeypatch.setattr`` — but that's opt-in per test;
    a future test that forgets to stub can again reach production channels.

    This autouse fixture closes the recurrence class: every test starts
    with publish replaced by a no-op that returns a synthetic success.
    Tests that want to assert on publish calls override this with their
    own ``monkeypatch.setattr("nousergon_lib.alerts.publish", spy)``
    — which works because monkeypatch reverts in LIFO order at teardown.
    """
    try:
        import nousergon_lib.alerts  # noqa: F401
    except ImportError:
        # lib pin <v0.21.0 → no alerts module to block. Pre-v0.21.0
        # callers can't reach the channels anyway.
        yield
        return

    class _Chan:
        ok = True
        detail = "blocked by conftest autouse fixture"

    class _Result:
        sns = _Chan()
        telegram = _Chan()
        any_ok = True
        all_ok = True

    def _noop(*args, **kwargs):
        return _Result()

    monkeypatch.setattr("nousergon_lib.alerts.publish", _noop)
    yield


@pytest.fixture(autouse=True)
def _stage_coverage_absent_unless_stubbed(monkeypatch):
    """Default-deny the config-I7214 stage-coverage primitive in tests.

    Both Lambda handlers in this repo call
    ``krepis.stage_coverage.assert_stage_coverage`` immediately before
    they return. That function constructs its OWN boto3 S3 and CloudWatch
    clients and reads a registry object out of S3 — so once krepis
    actually shipped the module (0.59.4), every handler test that runs a
    handler end to end started making live AWS calls. Same posture as the
    arcticdb stub above: unit tests must never reach real S3.

    ``None`` in ``sys.modules`` makes the import raise ``ImportError``
    deterministically. That matters twice over: it keeps the suite
    hermetic, AND it makes the observe-mode degrade contract a SIMULATED
    condition rather than an ambient one. The four tests that broke on
    2026-08-14 asserted "krepis.stage_coverage is not importable" as a
    fact about the installed environment; publishing the module flipped
    them red without anything in this repo changing.

    Tests that need the module-present path inject their own fake, which
    overrides this entry (monkeypatch reverts LIFO at teardown).
    """
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", None)
    yield


# ── Router resolution: never reach the real edge from a test ──────────────
#
# alpha-engine-config-I7878 moved the replay/concordance target-model call
# onto `krepis.router.resolve_model_spec`. That reads the live registry and
# probes the router edge, so an unpatched test would (a) need
# LLM_MODEL_REGISTRY.yaml on disk and (b) make a network call — the two
# things `_block_real_alerts_publish` above exists to prevent for alerts.
#
# Default-deny by autouse: every test gets a deterministic fake resolution
# unless it opts out via `real_router_resolution`. The fake spec names the
# ROUTER EDGE, never a provider, so a test can assert the property the
# migration exists to guarantee.

#: Env var the stubbed spec names as its credential source. Deliberately NOT
#: a real credential variable: this is the FIRST leg of krepis' credential
#: chain (`os.environ[spec.api_key_env]`), so setting it keeps the routed path
#: hermetic — no on-disk credentials, no SSM, no dependence on what the
#: machine happens to hold.
_TEST_ROUTER_CREDENTIAL_ENV = "KREPIS_TEST_ROUTER_CREDENTIAL"


@pytest.fixture
def real_router_resolution():
    """Opt out of `_stub_router_resolution` for a test that patches
    `resolve_target_spec` itself (e.g. to assert the failure path)."""
    return True


def _fake_route(model_id: str) -> dict:
    return {
        "schema_version": 2,
        "model": model_id,
        "display_name": f"{model_id} (pinned)",
        "provider": "litellm",
        "route": "litellm_proxy",
        "api_base_url": "https://router.test:8443",
        "deployment_id": model_id,
        "auth_token_type": "litellm_master_key",
        "group": "",
        "registry_id": model_id,
        "primary_model": model_id,
        "primary_registry_id": model_id,
        "capabilities": {},
        "params": {"max_tokens": 8192, "structured_outputs": False},
        "exec_context": "lambda",
        "wire": "openai",
        "skipped_entries": [],
    }


#: A registry carrying only what a replay test addresses. It has to exist on
#: disk because a pinned call is FAITHFUL to production here: litellm's proxy
#: stamps the client-requested model back onto the response, so
#: `krepis.llm._resolve_group_served_model` resolves the echoed registry id
#: through the registry to get the billable upstream model. Without this the
#: tests would only pass by making the fake transport report something a real
#: router never reports — and it was exactly this faithfulness that caught
#: krepis' bare-id masquerade bug (krepis-PR172) before a live run did.
#:
#: Every entry declares one provider rather than mirroring the live registry's
#: mix. Nothing here reads `provider` — the tests read `model` — and a second
#: provider name would put an OpenRouter literal in this repo purely to make a
#: fixture look realistic, which is an allowlist entry earned for nothing.
_TEST_REGISTRY_YAML = """
schema_version: 1
model_groups:
  low: [deepseek-v4-flash-low, gpt-oss-120b]
models:
  - id: deepseek-v4-flash
    model: deepseek-v4-flash
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    endpoints:
      openai: http://127.0.0.1:8990
    params:
      max_tokens: 8192
      structured_outputs: false
      reasoning:
        exclude: true
    status: active
    upstream_host: api.deepseek.com
  - id: deepseek-v4-flash-low
    model: deepseek-v4-flash
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    endpoints:
      openai: http://127.0.0.1:8990
    status: active
    upstream_host: api.deepseek.com
  - id: gpt-oss-120b
    model: openai/gpt-oss-120b
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    endpoints:
      openai: http://127.0.0.1:8990
    status: active
    upstream_host: api.deepseek.com
  - id: claude-haiku-4-5
    model: anthropic/claude-haiku-4.5
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    endpoints:
      openai: http://127.0.0.1:8990
    status: active
    upstream_host: api.deepseek.com
  - id: claude-sonnet-4-6
    model: anthropic/claude-sonnet-4.6
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    endpoints:
      openai: http://127.0.0.1:8990
    status: active
    upstream_host: api.deepseek.com
"""


@pytest.fixture(autouse=True)
def _stub_router_resolution(monkeypatch, request, tmp_path_factory):
    if "real_router_resolution" in request.fixturenames:
        yield
        return

    from krepis.llm_config import ROUTER_EDGE_PROVIDER, ModelSpec

    reg = tmp_path_factory.mktemp("registry") / "LLM_MODEL_REGISTRY.yaml"
    reg.write_text(_TEST_REGISTRY_YAML, encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(reg))

    # `LLMClient._transport_client()` resolves the credential BEFORE it calls
    # `client_factory`, so a test double for the transport does not remove the
    # need for one. On the router-edge provider krepis resolves it on the full
    # chain — environment, then an on-disk credentials file, then SSM — and a developer
    # laptop has the second while CI has none of the three. That asymmetry is
    # what made these tests pass locally and fail on the runner with `no
    # router-edge credential`, pre-empting every assertion about validation
    # errors, transport errors, persistence and usage extraction with a
    # credential error (alpha-engine-config-I7878).
    #
    # So the credential is INJECTED, at the boundary the routed client reads:
    # the spec names a test-only variable and this fixture sets it. Naming a
    # real credential variable would leave the suite resolving whatever the
    # machine happens to hold — which is how the laptop/CI split arose — and
    # reaching SSM would put a network call inside a unit test. Spec and
    # environment are set from the SAME constant, so the two cannot drift.
    monkeypatch.setenv(_TEST_ROUTER_CREDENTIAL_ENV, "not-a-real-credential")

    def _fake(model_id, *, max_tokens=8192):
        # A REAL ModelSpec, shaped exactly as `resolve_model_spec` returns
        # one: the router edge as a custom OpenAI-compatible endpoint, the
        # registry id as the wire model, and a router-edge credential — never
        # a provider name, URL or key.
        spec = ModelSpec(
            provider=ROUTER_EDGE_PROVIDER,
            model=model_id,
            base_url="https://router.test:8443",
            api_key_env=_TEST_ROUTER_CREDENTIAL_ENV,
            max_tokens=max_tokens,
            structured_outputs=False,
            reasoning={"exclude": True},
            registry_id=model_id,
        )
        return spec, _fake_route(model_id)

    for module in ("replay.runner", "replay.batch"):
        try:
            mod = importlib.import_module(module)
        except Exception:  # noqa: BLE001 — module not importable in this env
            continue
        if hasattr(mod, "resolve_target_spec"):
            monkeypatch.setattr(mod, "resolve_target_spec", _fake)
    yield


@pytest.fixture
def producer_register_2026_08_28(monkeypatch):
    """Pin the producer arena to its register as it stood on 2026-08-28.

    The committed register is append-only and grows as arms reach the board,
    so a test that drives the engine over the 2026-08-28 fixture board must
    not read it: from 2026-09-23 it holds arms and retirements dated after the
    fixture's ``as_of``. This fixture points ``REGISTER_PATH`` at a frozen copy
    of the 2026-08-28 register and re-derives ``VALID_CHAMPIONS`` from it.
    Tests about the COMMITTED register read ``COMMITTED_REGISTER_PATH``
    instead (alpha-engine-config-I11393).
    """
    from pathlib import Path

    from optimizer import champion_promotion, producer_arena

    path = Path(__file__).resolve().parent / "fixtures" / "producer_register_2026-08-28.json"
    monkeypatch.setattr(producer_arena, "REGISTER_PATH", path)
    monkeypatch.setattr(
        champion_promotion,
        "VALID_CHAMPIONS",
        producer_arena.promotion_eligible_arm_names(producer_arena.load_register(path)),
    )
    return path
