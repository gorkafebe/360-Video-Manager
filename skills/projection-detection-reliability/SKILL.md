# SKILL: projection-detection-reliability

## Purpose

Protect the detector’s reliability-first classification policy implemented across `detector/pipeline.py`, `detector/line_detection.py`, `detector/motion_analysis.py`, `detector/equirectangular_detection.py`, and `detector/projection_logic.py`.

## When to use

- Any change in `detector/` affecting:
  - line detection
  - stereo checks
  - motion scoring
  - reliability gating
  - retry policy
  - projection decision logic

## Rules

1. Preserve explicit output domain: `equirectangular`, `stereo_equi`, `eac`, `cubic`, `unknown`.
2. Keep reliability gates that can downgrade uncertain non-equirectangular outcomes to `unknown`.
3. Preserve early-classification branches (e.g., no-horizontal-line path, stereo checks) unless explicitly requested.
4. Maintain deterministic fallback-chain behavior for optical-flow selection.
5. Preserve low-motion handling and retry adaptation semantics.

## Constraints

- Treat `unknown` as a meaningful safety outcome, not an error.
- Keep debug outputs and stats fields compatible with current pipeline consumers.
- Do not infer new thresholds or policies not encoded in config or code.

## Forbidden actions

- Forcing `eac`/`cubic` labels when reliability criteria fail.
- Removing retry logic in `run_detection_with_retries` without explicit instruction.
- Deleting or silently repurposing existing stats keys used by `DetectorStats.from_dict`.

## Expected output

- Any change summary must identify which reliability safeguards remain intact.
- Tests must cover altered branch policy (especially reliability downgrade and fallback behavior).

