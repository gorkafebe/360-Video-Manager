# SKILL: unified-pipeline-contract

## Purpose

Enforce the canonical job-stage contract implemented in `workflows/unified_pipeline.py` and `core/models.py` so agent changes do not break stage ordering, result fields, or skip/optional behavior.

## When to use

- Any change touching:
  - `workflows/unified_pipeline.py`
  - `core/models.py`
  - `core/job_manifest.py`
  - GUI calls into `process_video_job`
- Any change that adds/removes/reorders processing stages.

## Rules

1. Preserve stage order semantics:
   - resolve/search → download (optional) → codec normalize → preview extraction → detect → convert (optional) → upload (optional) → manifest.
2. Keep `JobResult` as the canonical output contract.
3. Keep upload optional (`options.upload`) and conversion optional (`options.convert_if_needed`).
4. Preserve `final_video_path` priority: converted > normalized > original.
5. Keep manifest persistence behavior controlled by `save_manifest`.

## Constraints

- Do not move detection/conversion logic into GUI code.
- Do not create alternative orchestration entrypoints that bypass `process_video_job` without explicit request.
- If behavior ambiguity exists, stop and document the restriction instead of inferring new policy.

## Forbidden actions

- Breaking `JobResult` fields without coordinated updates in all consumers/tests.
- Removing fallback behavior for missing optional stages.
- Returning raw detector/upload payloads instead of structured dataclasses in pipeline outputs.

## Expected output

- Proposed or applied changes must explicitly state which stage contract is preserved.
- Any contract-affecting change must include corresponding test updates in `tests/`.

