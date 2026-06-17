# Baseline Test Results — Dead-Code Cleanup (2026-06-17)

Recorded before any code change, as the comparison baseline for the phased
dead-code cleanup described in `SCAN_REPORT.md` (§4 gaps 3–4) and
`IMPROVEMENT_PLAN.md` backlog items 5–6.

Command: `python -m pytest tests/ -v`
Environment: tkinter and pytest available; no collection failures.

```
118 passed in 2.32s
```

Full pass/fail list (118 tests, all PASSED, 0 FAILED, 0 ERROR):

- tests/test_detection_retry.py — 1 test
- tests/test_fallback_and_adaptive.py — 13 tests
- tests/test_line_and_stereo_strictness.py — 4 tests
- tests/test_line_detection.py — 14 tests
- tests/test_motion_classification_policy.py — 2 tests
- tests/test_motion_flow_fallback.py — 3 tests
- tests/test_progress_and_downloader.py — 8 tests
- tests/test_projection_conversion.py — 7 tests
- tests/test_uploader.py — 14 tests
- tests/test_video_io_av1.py — 9 tests
- tests/test_video_io_sampling.py — 8 tests
- tests/test_youtube.py — 10 tests

Each subsequent cleanup phase must reproduce this exact 118-passed / 0-failed
/ 0-error result with an identical set of test IDs. Any deviation triggers an
immediate revert of that phase per the cleanup task's rules.
