# gcol33/tulpa#633 -- copy_alpha axis resolution

Measured 2026-08-31 on tulpa 0.2.4, tulpaObs 0.1.1, R 4.6.0, Windows 11.
Probe: `dev_notes/issue633/probe_alpha_axis.R`. Fixture is an 8x8 rook ICAR grid,
N = 64, lognormal cover arm, `positive = ~ pos_cov1 + copy(spatial())`.

## Grid ESS against requested node count

`sigma.grid` and `phi.grid.pos` both set to the node count in column 1. The
`copy_alpha` axis is not settable under `copy()`, so it stays on auto placement.

| requested nodes | cells | J = 10 ESS | J = 30 ESS |
|---|---|---|---|
| 13 | 1023 / 1019 | 18.3 | 13.5 |
| 17 | 1742 | 26.3 | 12.2 |
| 21 | 2655 / 2654 | 37.3 | 13.8 |
| 25 | 3758 | 41.7 | 16.0 |
| 29 | 5054 | 53.4 | 17.3 |

At J = 10 the ESS tracks the node count. At J = 30 it does not: more than doubling
the request moves ESS from 13.5 to 17.3, while the cell count rises five-fold.

## Which axis binds

Surviving distinct values per axis at J = 30:

| requested | sigma | alpha | phi_pos |
|---|---|---|---|
| 13 | 13 | 10 | 13 |
| 17 | 17 | 13 | 17 |
| 21 | 21 | 13 | 21 |
| 25 | 25 | 13 | 25 |
| 29 | 29 | 13 | 29 |

`sigma` and `phi_pos` follow the request exactly. `alpha` saturates at 13.

The declared axis is smaller than what survives, so auto placement is expanding the
axis but not densifying it:

```
tulpa:::.nl_grid_axis("copy_alpha")
# 0.0000 0.1000 0.2340 0.5477 1.2819 3.0000   (6 nodes, [0, 3])
```

against a fitted axis of 10-13 surviving nodes spanning [0, 16.4].

## Consequence in calibration

Simulation-based calibration on the same fixture, n.sim = 300, seed 20260820, 95%
simultaneous band, wide/narrow controls failing 0/20 as designed. Rank dispersion is
`mean(abs(u - 0.5))`, 0.250 under uniformity, above it meaning the posterior is too
narrow:

| quantity | J = 3 p_unif | J = 10 p_unif | dispersion J = 3 -> J = 10 |
|---|---|---|---|
| `sigma` | 0.000109 | 0.127 | 0.259 -> 0.282 |
| `sigma_pos_field` | 0.026 | 1.7e-9 | 0.253 -> 0.292 |
| `alpha` | 0.015 | 8.4e-5 | 0.249 -> 0.299 |

All three field hyperparameters get more overconfident as the data get more
informative. The base fit's grid ESS falls 29.8 to 16.7 over the same step at fixed
nodes. Every coefficient stays near 0.250 across both.

Real runs sit on the wrong side of this: the MOTIVATE cut the coupled model is being
applied to has a median of 151 visits per site.

## Not a caller-side fix

`control$alpha.grid` is rejected when `copy()` appears in the positive formula
(`tulpaObs/R/family_cover_hurdle.R:109-113`). That guard is correct: it prevents two
sources of truth for the same axis. Supplying a raw numeric grid would also displace
the declared atom at 0 and the Exponential(1) slab the axis carries, which changes
what is integrated rather than how accurately.

## Open

1. Is the saturation in the auto placement, or downstream in the prune that drops
   low-weight cells? Not established here.
2. Should the target be a declared ESS floor the engine solves for, rather than a
   node count the caller sets per axis?

`diagnose_k` is `FALSE` on this path (`tulpaObs/R/occu_cover_joint.R:689`), so none
of the 0.2.1 to 0.2.5 Pareto-k work touches these numbers.
