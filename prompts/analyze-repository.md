# /analyze-repository

## What it DOES

- Performs a read-only architectural and risk audit of this repository.
- Maps stage boundaries across `app/`, `workflows/`, `core/`, and `detector/`.
- Reports constraints, trust boundaries, and high-risk change zones.

## What it DOES NOT do

- Does not edit files.
- Does not run destructive commands.
- Does not invent architecture not visible in repository code.

## Safety constraints

- Must inspect relevant source + tests before reporting conclusions.
- Must label unknowns as restrictions, not assumptions.
- Must explicitly call out if inspection scope is incomplete.

