"""config_guards.py — fail-loud validation of keyed config blocks (config#1842).

The bug class this retires: a config.yaml block keyed by canonical names
(sub-scores, outcome columns, rank metrics) silently drifts when the canonical
names are renamed — every downstream ``configured.get(canonical_key)`` resolves
to the fallback (usually ``0.0`` or an inert default) and the feature runs as a
zero-filled no-op for months. Three live occurrences of the class by
2026-07-06:

- ``weight_optimizer.default_weights`` carried pre-2026-03-29 ``news``/
  ``research`` keys while the canonical sub-scores are ``quant``/``qual``
  (config#1842): every weight proposal was measured against a phantom
  0.0 baseline, permanently tripping the ``max_single_change`` guardrail —
  ``config/scoring_weights.json`` was NEVER written.
- ``weight_optimizer.horizon_blend`` carried the retired 10d/30d beat-SPY
  keys after the canonical-alpha cutover (noted in the live config comment,
  fixed 2026-07-01): the tuned 60/40 blend was silently inert.
- research ``config._RP_DEFAULTS`` dropped the unconsumed ``cio_mode`` key a
  second producer wrote (config#1719) — 63-day dead write.

Per the fail-loud doctrine (config#1684) a key mismatch RAISES instead of
zero-filling. The raise surfaces at the read chokepoint inside the optimizer
module, where ``CompletenessTracker.run_module`` records the module as
``error`` (failed_modules in the digest + completeness manifest) and the
apply-audit artifact (config#1841) records ``outcome="error"`` — loud on every
surface without taking down sibling optimizers whose config blocks are fine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = ["ConfigKeyDriftError", "validate_keyed_block"]


class ConfigKeyDriftError(ValueError):
    """A configured keyed block does not match its canonical key vocabulary."""


def validate_keyed_block(
    configured: Mapping | None,
    canonical: Iterable[str],
    *,
    config_path: str,
    allow_subset: bool = False,
) -> None:
    """Validate that ``configured``'s keys match the ``canonical`` vocabulary.

    Args:
        configured: The config block as loaded (or ``None`` when the block is
            absent — absence is always valid: the in-code default applies).
        canonical: The canonical key vocabulary (e.g. ``SUB_SCORES``).
        config_path: Dotted config location for the error message
            (e.g. ``"weight_optimizer.default_weights"``).
        allow_subset: When True, ``configured`` may name any SUBSET of the
            canonical keys (optional per-key override blocks such as
            ``executor_optimizer.min_baseline_magnitude_by_rank``); unknown
            keys still raise. When False (default), the key sets must be
            EXACTLY equal — a partial authoritative block would silently
            zero-fill the missing canonical keys downstream.

    Raises:
        ConfigKeyDriftError: On any mismatch, naming the stale/missing keys
            and the canonical vocabulary so the operator can fix config.yaml
            without reading source.
    """
    if configured is None:
        return
    canonical_set = set(canonical)
    configured_set = set(configured.keys())
    unknown = sorted(configured_set - canonical_set)
    missing = sorted(canonical_set - configured_set)
    if unknown or (missing and not allow_subset):
        parts = [
            f"config key drift in `{config_path}`:",
        ]
        if unknown:
            parts.append(f"unknown key(s) {unknown} are not in the canonical vocabulary")
        if missing and not allow_subset:
            parts.append(f"canonical key(s) {missing} are missing")
        parts.append(
            f"canonical keys: {sorted(canonical_set)}. Refusing to zero-fill "
            "(fail-loud, config#1842) — fix the block in "
            "alpha-engine-config/backtester/config.yaml."
        )
        raise ConfigKeyDriftError(" ".join(parts))
