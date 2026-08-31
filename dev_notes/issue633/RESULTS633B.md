# gcol33/tulpa#633 -- where the copy_alpha saturation is, and the knob it lacked

`RESULTS633.md` measured the saturation through tulpaObs on an `occu_cover`
hurdle and left two questions open. This answers the first and acts on the
issue's own fallback ask. Probe: `probe_alpha_engine.R`, engine-side.

## Reproduced without the consumer package

The axis belongs to the engine, so the fixture is
`tulpa_nested_laplace_joint()` directly: an ICAR chain over 40 sites, a
binomial arm and a gaussian copy arm, `reps` per site standing in for visits.
That matters beyond convenience -- an engine defect needs an engine fixture, or
only a consumer can see it.

Raising the donor `sigma_grid` and reading the fitted grid back:

| request | cells | ESS (reps = 3) | ESS (reps = 30) | surviving sigma / alpha |
|---|---|---|---|---|
| 13 | 78 | 4.8 | 1.7 | 13 / **6** |
| 21 | 126 | 7.8 | 3.1 | 21 / **6** |
| 29 | 174 | 10.8 | 4.3 | 29 / **6** |

The alpha axis sits at its declared 6 nodes at every setting, and the ESS on
informative data barely moves while the cell count more than doubles. Same
shape as the reported occu_cover run, where alpha saturated at 13.

## Open question 1: it is the PLACEMENT, not the prune

`prune = TRUE` on the same fits:

| request | cells | ESS | surviving alpha |
|---|---|---|---|
| 13 | 78 | 1.7 | 6 |
| 21 | 126 | 3.1 | 6 |
| 29 | 174 | 4.3 | 6 |

Identical node counts, identical ESS to the digit. The prune is not what drops
alpha nodes -- there were never more than six to drop. (`prune` also defaults
to `FALSE` in the engine, so the reported run was not pruning unless the
consumer turned it on.)

Nothing scales the alpha axis with the other axes, and nothing can: the axis is
read from `.nl_grid_axis("copy_alpha")`, whose declared shape is `n = 5` plus
the prepended atom, and the ONLY way to change it was `alpha_grid`, which
replaces the nodes wholesale.

## The knob

The issue's fallback ask -- "a supported way to raise its node count without
displacing the declared atom and slab" -- is what shipped, because the axis
carries prior structure and stating nodes for it restates that structure. Two
fields that answer two different questions:

- `alpha_grid` STATES the nodes (existing behaviour, unchanged).
- `alpha_n` re-reads the engine's own axis at a higher RESOLUTION: same `lo` /
  `hi`, same atom at 0, more nodes between them. On the single-block path it is
  `field_coef$n`.

Supplying both is an error rather than a silent ranking. Generalized one level
down, `.nl_grid_axis(key, n =)` re-reads ANY declared axis at a different
resolution and refuses on an axis declared as explicit `nodes`, which has no
resolution to vary.

With it, the same fits at `reps = 30`:

| request | cells | ESS | surviving sigma / alpha |
|---|---|---|---|
| 13 | 182 | 2.3 | 13 / 14 |
| 21 | 462 | 6.8 | 21 / 22 |
| 29 | 870 | 12.5 | 29 / 30 |

ESS tracks the effort again (1.7 / 3.1 / 4.3 -> 2.3 / 6.8 / 12.5), and the axis
keeps its shape at every resolution: atom exactly 0, slab exactly [0.1, 3].

## A trap found on the way

`$` partial-matches on a list, so reading the new field as `fc$n` resolved to
`fc$name` on every spec that names its coefficient -- feeding a character into
an integer check and erroring every existing
`field_coef = list(name = , grid = )` fixture. The reads are `[[`. Pinned in
`test-copy-alpha-resolution.R`.

## Open question 2, and what is NOT claimed

- **Whether the target should be a declared ESS floor the engine solves for,
  rather than a node count the caller sets, is not answered here** and is not
  implied by this change. It is a design question about who owns integration
  effort; `alpha_n` gives the axis parity with every other outer axis (the
  caller can raise it), which is what the issue asked for as the concrete fix.
- The axis still does not auto-densify from its own posterior sharpness. No
  outer axis does; they are re-PLACED by the mode-Hessian recenter, not
  re-resolved.
- The consumer side is a separate change: tulpaObs closes `alpha.grid` off
  under `copy()` (correctly), so it must pass `alpha_n` through for a user of
  `occu_cover()` to reach this. Filed on that repo.
- The calibration consequence in `RESULTS633.md` (the three field
  hyperparameters leaving the SBC band as J rises) is NOT re-measured here.
  This removes the reason the axis could not be raised; whether raising it
  restores calibration on that fixture is the measurement to run next, and it
  needs the consumer pass-through first.
