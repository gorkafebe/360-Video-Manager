# SKILL: conversion-fallback-integrity

## Purpose

Enforce conversion safety and fallback behavior from `detector/projection_conversion.py` and workflow stage logic in `workflows/unified_pipeline.py`.

## When to use

- Any change to projection conversion mapping, ffmpeg command generation, conversion retries, or conversion-decision policy.

## Rules

1. Keep authoritative mapping:
   - `eac -> v360=eac:equirect`
   - `cubic -> v360=c3x2:equirect`
2. Preserve skip policy for:
   - `equirectangular`
   - `stereo_equi`
3. Preserve workflow fallback where stage-level conversion treats `unknown` as `eac` only for conversion attempts.
4. Preserve audio-failure retry path that retries conversion without audio (`-an`) when audio encoding fails.
5. Keep output-path safety check that prevents input overwrite.

## Constraints

- ffmpeg availability checks must remain non-destructive and explicit in result dicts.
- Conversion failures must not invalidate detector output; conversion is best-effort.
- If projection type is unsupported/unclear, skip safely with explicit reason.

## Forbidden actions

- Replacing EAC mapping with equirectangular input code (`e`) or any unstated format.
- Removing failure metadata from conversion results (`error`, `stderr`, `ffmpeg_command`).
- Silent mutation of projection labels outside defined fallback points.

## Expected output

- Conversion-related changes must include explicit reasoning about skip/convert/fallback branches.
- Affected tests in `tests/test_projection_conversion.py` and `tests/test_fallback_and_adaptive.py` must stay aligned.

