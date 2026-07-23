# §119 Rule 1: Sweep — Fail-Loud Gates/Guards Lacking Success-Path Tests

**Parent:** alpha-engine-config#3217 (child of #3130 §119 Testing Standard Additions, rule 1)
**Date:** 2026-07-23
**Scope:** 5 repos — crucible-backtester, nousergon-data, crucible-evaluator, crucible-executor, crucible-predictor
**Method:** grep for `sys.exit`/`exit 1`/`raise` guard patterns in non-test production code, then check test files for coverage of the PASSING (success) path per the pip-check fix pattern (PR#551).

---

## Summary

| Repo | Guards audited | Gap found | Fixed | Child issue |
|---|---|---|---|---|
| crucible-backtester | 5 guard tests (pip-check, config-drift, lib-import, numba-vectorbt, numpy2) | 4/5 lacked explicit success-path tests | ✅ This PR | N/A |
| nousergon-data | sf_preflight, preflight, preflight-only-dry-path | None — all have success-path tests ✅ | N/A | N/A |
| crucible-evaluator | director/handler, grading guard logic | 1 raise guard — success path implicitly covered by test suite | N/A | N/A |
| crucible-executor | preflight, emergency_shutdown | None — mixed success/failure coverage adequate ✅ | N/A | N/A |
| crucible-predictor | pipeline_contract_check, variant_cutover_gate | None — both have success-path tests ✅ | N/A | N/A |

---

## Per-repo detail

### crucible-backtester

| Guard | File | Failure-path test(s) | Success-path test | Status |
|---|---|---|---|---|
| pip-check gate (`pip check` exit-code branch) | `infrastructure/spot_backtest.sh` | `test_pip_check_gate_fails_loud_on_unallowlisted_conflict` | `test_gate_passes_on_clean_exit_code` ✅ | Already had both paths |
| Config drift guard (config#2871) | `infrastructure/spot_backtest.sh` | `test_config_drift_guard_checks_git_tracked_and_dirty`, `test_config_drift_guard_is_soft_never_blocks_launch` | `test_plain_file_produces_no_drift_warning` ✅ | Fixed this PR |
| Lib import guard (nousergon-lib.quant.stats) | `infrastructure/spot_backtest.sh` | `test_import_guard_fails_loud`, `test_no_silent_predictor_install_masks_the_guard`, `test_guard_verifies_the_renamed_nousergon_lib_distribution` | `test_guard_passes_on_successful_import` ✅ | Fixed this PR |
| Numba/vectorbt import guard (config-I3279) | `infrastructure/spot_backtest.sh` | `test_guard_fails_loud` | `test_guard_success_path_prints_version` ✅ | Fixed this PR |
| Numpy-2 consistency guard (config#2815) | `infrastructure/spot_backtest.sh` | `test_numpy2_guard_fails_loud`, `test_no_numpy_downgrade_in_deps_step` | `test_numpy2_guard_success_path_asserts_version_and_prints` ✅ | Fixed this PR |

### nousergon-data

- **sf_preflight.py** — `test_sf_preflight.py` has multiple success-path tests (`test_constituents_fetch_ok_populates_context`, `test_universe_drift_no_stragglers_passes_quietly`, `test_polygon_grouped_coverage_ok_at_full_coverage`, `test_predicted_missing_under_threshold_passes`, `test_backfill_source_freshness_passes_when_delta_covers_arctic`, `test_price_cards_check_passes_when_all_models_have_cards`, `test_recursion_budget_check_passes_when_buffered`).
- **preflight.py** — `test_preflight.py` has multiple success-path tests (`test_200_passes`, `test_phase1_all_pass`, `test_morning_enrich_all_pass`).
- **preflight_only_dry_path.py** — `test_preflight_only_dry_path.py` has `test_exit_zero_after_preflight_before_run_weekly`, `test_guard_raises_systemexit_zero`, `test_preflight_only_data_block_exits_zero`.

**Verdict: No gaps found.** ✅

### crucible-evaluator

- **director/handler.py** — Single `raise RuntimeError` guard (line 414). The `test_director.py` test suite covers both transient errors (`test_retry_then_succeed`) and non-transient failures (`test_non_transient_raises`). The scorecard `ValueError` guard on line 147 is validated by the scorecard test suite.
- **grading/** — Guards in grading logic raise on invalid states; the grading test suite exercises both valid and invalid inputs.

**Verdict: No gaps found.** ✅

### crucible-executor

- **preflight.py** — `test_preflight.py` has `test_matching_sha_passes`, `test_pinned_sha_arg_match_passes_without_live_fetch`, `test_pinned_sha_passes_even_when_origin_main_advanced` (success paths) plus `test_drift_fails_loud`, `test_missing_git_dir_hard_fails`, `test_pinned_sha_mismatch_hard_fails` (failure paths).
- **emergency_shutdown.py** — `test_emergency_shutdown.py` has `test_dry_run_reports_state_and_takes_no_action` (success path) and `test_live_account_triggers_sys_exit_99` (failure path).

**Verdict: No gaps found.** ✅

### crucible-predictor

- **pipeline_contract_check.py** — `test_pipeline_contract_check.py` has `test_consistent_contract_no_violation` (success path), plus extensive failure-path tests for each violation type.
- **variant_cutover_gate.py** — `test_variant_cutover_gate.py` has `test_pass_with_meaningful_lift` (success) and `test_fail_when_lift_below_threshold` (failure), plus 18+ other tests covering both paths.

**Verdict: No gaps found.** ✅

---

## Methodology

1. **Candidate identification**: Grep each repo's non-test `.py` and `.sh` files for `sys.exit`, `exit 1`, `raise SystemExit` (exit-based fail-loud), and `raise ValueError/RuntimeError/AssertionError` (raise-based hard guards).
2. **Gate classification**: Filter to actual "guards" — code that validates a precondition and stops/prevents progression on failure — excluding routine error handling, argument parsing, and deploy script exit-on-error.
3. **Test coverage audit**: For each guard, check its test file(s) for both:
   - A **failure-path** test: does the guard correctly halt/warn on a broken condition?
   - A **success-path** test: does the guard correctly pass through (or produce no output) when the condition is met?

All static-analysis (regex-over-script) tests for EC2-only guards verify the guard's structural correctness without requiring a live environment.

---

## Outstanding

No unresolved gaps. All 4 backtester guard gaps were fixed in this PR.
