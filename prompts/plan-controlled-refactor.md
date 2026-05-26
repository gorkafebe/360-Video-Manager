# /plan-controlled-refactor

## What it DOES

- Produces a constrained refactor plan for a specified module set.
- Preserves existing runtime behavior and public contracts by default.
- Defines validation checkpoints and rollback-safe sequencing.

## What it DOES NOT do

- Does not implement the refactor.
- Does not widen scope beyond explicitly named modules.
- Does not alter API contracts unless explicitly requested.

## Safety constraints

- Must identify existing invariants from code/tests first.
- Must prefer smallest viable refactor units with testable checkpoints.
- Must stop and flag restrictions where behavior is ambiguous.

