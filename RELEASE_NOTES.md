# Release Notes

## Version: 2026-02-11 (stabilization)

### Highlights
- ML pipeline trace standardized and auditable:
  - `engine_lineup -> season_form_blend -> hybrid_blend -> confidence_guard -> consistency_guard -> series_gamma`
  - `ml_trace_last.json` now includes per-step `before/after/applied`.
- Hybrid blend layer added (data-driven):
  - momentum + early objectives + macro control.
  - configurable from `Parâmetros`.
- Patch/league calibration freshness checks:
  - auto snapshot by league/patch in `params_fitted.json`.
  - pre-delivery includes `Patch/Liga calibração em dia`.
  - calibration panel includes drift status by patch/league.
- Pre-delivery and smoke hardened:
  - combined trace/session checks fixed for new trace schema.
  - smoke now runs with active CSV (`MLCORE_CSV`) automatically.
  - UI smoke test added (Streamlit render sanity).

### Tests added/updated
- Added:
  - `mlcore/tests/test_ui_smoke.py`
  - `mlcore/tests/test_calibration_schedule.py`
  - `mlcore/tests/test_key_matchups_regression.py`
  - `mlcore/tests/test_key_matchups_extended_regression.py`
  - fixtures:
    - `mlcore/tests/fixtures/key_matchups_regression.json`
    - `mlcore/tests/fixtures/key_matchups_extended_regression.json`
- Updated:
  - `mlcore/tests/test_smoke.py` (stable finite/non-negative assertion).

### Fixed
- Pylance undefined-variable issues in `plays_app.py`:
  - removed misplaced patch/liga block from ML runtime path.
  - fixed scope usage in calibration panel (`csv_path`).
- Smoke output no longer pollutes pre-delivery detail with expected
  `missing ScriptRunContext` lines.

### Known non-blocking warnings
- Streamlit `use_container_width` deprecation warnings may still appear.
  - functional impact: none.
  - planned cleanup: migrate to `width='stretch'/'content'`.

### Validation status
- Pre-delivery: approved (9/9)
- Unit tests: passing
- Smoke: passing
