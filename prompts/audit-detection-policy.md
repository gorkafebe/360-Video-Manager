# /audit-detection-policy

## What it DOES

- Audits detector policy and reliability logic in `detector/pipeline.py` and related modules.
- Verifies that classification safeguards (`unknown`, reliability gating, low-motion handling, retry policy) remain internally consistent.
- Surfaces policy contradictions and high-risk branching points.

## What it DOES NOT do

- Does not modify detector code.
- Does not tune thresholds or propose new heuristics unless explicitly requested.
- Does not claim certainty where evidence is missing.

## Safety constraints

- Must trace policy through code paths and tests before conclusions.
- Must treat `unknown` handling as a safety mechanism unless repository code explicitly changes that rule.
- Must report any missing context as a blocking restriction.

