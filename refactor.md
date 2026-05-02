# Tulpa Refactoring Guide

This repo should be refactored toward clean internal design, not toward preserving incidental historical structure. Compatibility is important only when it protects a real external contract or keeps a migration tractable.

## Priorities

1. Clean code over preserving old internal boundaries.
2. Small, named modules over large files with comment-section boundaries.
3. Direct, readable control flow over compatibility shims that hide the real design.
4. Explicit data/state objects over long local variable lists shared across distant blocks.
5. Focused tests around the behavior being preserved.

## What To Preserve

Preserve documented external contracts:

- Public R APIs.
- Installed headers used by downstream packages.
- Stable `ModelData`, `ParamLayout`, and `LikelihoodSpec` fields unless the change is an intentional ABI/API migration.
- Numerically important behavior unless a change is explicitly planned and tested.

Do not preserve these just for familiarity:

- Old file boundaries.
- Legacy helper names.
- Large monolithic functions.
- Adapter layers whose only purpose is avoiding edits to internal callers.
- Comments that explain structure which should instead be visible in code.

## Preferred Refactor Shape

When refactoring a subsystem:

1. Identify the actual domain stages.
2. Move each stage into a named helper or module.
3. Introduce a small state/context object when many values move together.
4. Remove compatibility glue once internal callers can use the cleaner path.
5. Keep comments short and reserve them for non-obvious constraints, ordering, or contracts.
6. Verify with the smallest tests that cover the touched behavior, plus package install when templates or headers move.
7. Commit narrow, coherent steps.

## Generic Likelihood Path Example

The generic log-posterior path should continue moving in this direction:

- `compute_log_post_generic()` should read as orchestration.
- Prior/state initialization, fixed predictor precompute, effect routing, and likelihood summation should remain separate stages.
- `LikelihoodSpec` dispatch should be centralized.
- Legacy ratio code should not leak into generic model packages.

The goal is not to keep old internals intact. The goal is to make the production path obvious, modular, and hard to misuse.
