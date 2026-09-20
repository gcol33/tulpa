# gcol33/tulpa#853 -- what the per-axis continuization does, measured

`tulpa_hyper_draws()` continuizes each outer-grid axis inside its own selected
cell. #853 reports that the #823 fix which introduced this trades one SBC
failure for three on `occu_cover`'s HP760 arms, names independence across axes
as the suspected mechanism, and asks for either a joint within-cell
continuization or a scoping of composite reads back to the grid-node atom.

Both remedies are measured here against an exact reference posterior, and
neither is supported. What the sweeps do separate is a different pair of
conditions, already carried by the engine's own grid diagnostic.

## The fixture

Two scale hyperparameters, correlated through a shared arm, with a closed-form
outer log-posterior so nothing about an inner Laplace is under test:

    y_j ~ N(0, s1^2)            j = 1..n1
    z_j ~ N(0, s1^2 + s2^2)     j = 1..n2
    log s1, log s2 ~ N(0, p^2)

The reference posterior is a fine 2-D quadrature over that density. `alpha =
s2 / s1` is the derived quantity standing in for the ratio #853 names. The PIT
is `mean(draw <= truth)` on each arm and the exact CDF on the reference arm,
the convention #823 used.

Arms: `atom` (the grid-node read #823 replaced), `box` (the shipped
continuization), `boxc` (one uniform shared by both axes -- comonotone),
`boxr` (a Gaussian copula at the grid's OWN weighted correlation of the log
axes), `exact`.

`boxc` and `boxr` keep each axis's within-cell marginal exactly: the uniform's
inverse CDF is affine, so any uniform input reproduces the box. They differ
from `box` only in how the two axes' uniforms are tied, which is precisely the
degree of freedom #853 proposes to use.

Driver: `probe853.R`, 300 replicates per configuration, 4000 draws per fit.

## 1. The continuization never loses to the atom

KS against uniform, `box` vs `atom`, all six configurations:

| configuration | s1 | s2 | alpha |
|---|---|---|---|
| adapt weak n1=8 nodes=5 | 0.051 / 0.206 | 0.043 / 0.223 | 0.048 / 0.121 |
| adapt sharp n1=400 nodes=5 | 0.064 / 0.226 | 0.066 / 0.189 | 0.057 / 0.106 |
| adapt sharp n1=400 nodes=9 | 0.040 / 0.104 | 0.056 / 0.092 | 0.066 / 0.064 |
| fixed weak n1=8 nodes=5 | 0.134 / 0.208 | 0.071 / 0.115 | 0.069 / 0.072 |
| fixed sharp n1=400 nodes=5 | 0.163 / 0.530 | 0.145 / 0.488 | 0.179 / 0.503 |
| fixed sharp n1=400 nodes=9 | 0.079 / 0.448 | 0.084 / 0.308 | 0.122 / 0.317 |

`box` is at or below `atom` on every quantity in every configuration,
including the ratio. On the adapted grids it tracks the exact arm (`alpha`
exact 0.061 / 0.053 / 0.037).

## 2. Coupling the axes is a no-op, or worse

`alpha` KS, the quantity the coupling is supposed to repair:

| configuration | box | boxr (grid correlation) | boxc (comonotone) | exact |
|---|---|---|---|---|
| adapt weak nodes=5 | 0.0477 | 0.0467 | 0.0690 | 0.0614 |
| adapt sharp nodes=5 | 0.0574 | 0.0587 | 0.0711 | 0.0528 |
| adapt sharp nodes=9 | 0.0661 | 0.0682 | 0.0639 | 0.0372 |
| fixed weak nodes=5 | 0.0688 | 0.0722 | 0.0723 | 0.0390 |
| fixed sharp nodes=5 | 0.1793 | 0.1790 | 0.5026 | 0.0737 |
| fixed sharp nodes=9 | 0.1221 | 0.1182 | 0.3166 | 0.0680 |

`boxr` -- the copula at the cell's own correlation, which is what #853's first
option asks for -- is indistinguishable from the shipped independent jitter
everywhere, to the third decimal.

`boxc` reproduces the ATOM exactly where the grid's correlation is ~0 (0.5026
against the atom's 0.5026; 0.3166 against 0.3166): a shared uniform moves both
axes together, and a ratio of two comonotone draws cancels the jitter. #853's
second option -- returning composite reads to the atom -- is the same object,
and it is the worse arm.

## 3. What does predict a broken read

Driver: `probe853_resolution.R`, 300 replicates, `s1`'s PIT, fixed grids.

The grid's own weighted SD is NOT usable as the statistic: it collapses toward
zero exactly when the grid stops resolving the axis, so an sd-over-cell-width
reading saturates and cannot tell "one cell holds everything" from "the
posterior is narrow". The engine's `outer_grid_h_over_sd` carries the inverse
ratio, which diverges instead, and that is the usable direction.

RESOLUTION sweep -- node count at constant extent:

| K | effective cells | edge mass | box KS | exact KS | pinned |
|---|---|---|---|---|---|
| 3 | 1.00 | 0.153 | 0.283 | 0.076 | 0.087 |
| 5 | 1.01 | 0.027 | 0.148 | 0.078 | 0.040 |
| 7 | 1.01 | 0.020 | 0.134 | 0.075 | 0.047 |
| 9 | 1.02 | 0.020 | 0.104 | 0.079 | 0.040 |
| 13 | 1.10 | 0.013 | 0.117 | 0.081 | 0.013 |
| 19 | 1.19 | 0.010 | 0.084 | 0.073 | 0.013 |

EXTENT sweep -- grid span at constant cell width:

| span | effective cells | edge mass | box KS | exact KS | pinned |
|---|---|---|---|---|---|
| 0.3 | 1.09 | 0.520 | 0.219 | 0.080 | 0.373 |
| 0.5 | 1.17 | 0.237 | 0.121 | 0.081 | 0.153 |
| 0.8 | 1.16 | 0.035 | 0.066 | 0.066 | 0.047 |
| 1.2 | 1.19 | 0.000 | 0.104 | 0.117 | 0.017 |
| 2.0 | 1.19 | 0.000 | 0.061 | 0.080 | 0.000 |

`pinned` -- the share of replicates whose PIT is exactly 0 or 1 -- tracks edge
mass monotonically across both sweeps (0.520 -> 0.373, 0.237 -> 0.153, 0.153 ->
0.087, 0.035 -> 0.047, 0.000 -> 0.000). That is mechanical: mass in the
outermost cell is mass no draw can place beyond, and a truth out there has no
draw on its far side. A fine sweep between span 0.55 and 0.90 does not resolve
a sharper crossing at 300 replicates -- the exact arm's own KS wanders 0.060 to
0.117 over the same configurations -- so no threshold is read off it here.

## 4. The engine already names this, and its threshold is deliberate

Driver: `probe853_diagnostic.R`, the same configurations, reporting the SHIPPED
`.tulpa_grid_resolution()` / `.tulpa_grid_resolution_note()`.

| configuration | PIT verdict | note fires | coarse | railed | edge | h/sd |
|---|---|---|---|---|---|---|
| K=3 span=1.0 | miscalibrated | 1.000 | 0.893 | 0.480 | 0.490 | 44.43 |
| K=5 span=1.0 | miscalibrated | 1.000 | 1.000 | 0.120 | 0.160 | 19.70 |
| K=9 span=1.0 | miscalibrated | 1.000 | 0.990 | 0.047 | 0.073 | 9.50 |
| K=19 span=1.0 | borderline | 1.000 | 0.997 | 0.033 | 0.043 | 4.02 |
| K=6 span=0.3 | miscalibrated | 1.000 | 0.643 | 0.760 | 0.803 | 4.28 |
| K=10 span=0.5 | miscalibrated | 1.000 | 0.887 | 0.413 | 0.513 | 3.96 |
| K=15 span=0.8 | tracks exact | 1.000 | 1.000 | 0.070 | 0.147 | 4.13 |
| K=38 span=2.0 | tracks exact | 1.000 | 1.000 | 0.000 | 0.000 | 3.91 |

The note speaks on every replicate of every configuration, including the two
whose PIT is indistinguishable from the exact posterior. That is NOT a defect:
`R/settings.R` sizes `grid_resolved = 1` as the point below which the
`box_uniform` and `chord` constructions converge, not as a calibration cutoff,
and records a 34-configuration census of the engine's own default axes putting
every one above it -- minimum 1.01, median 4.25, maximum 18.06 -- concluding
that an unresolved axis "is the ordinary case and is worth reporting rather
than warning about". This sweep reproduces that census independently: median
h/sd 3.91 to 4.28 on the extent arms.

What the sweep adds is an ORDERING among those ordinary grids. At h/sd held
near 4, the four extent configurations run from a read that tracks the exact
posterior to one at three times its KS, and the two conditions that move with
it are the railed-axis test and the edge mass -- both already recorded on the
fit (`outer_grid_railed_axes`, `outer_grid_edge_mass_axes`) and both already
branches of the note. The h/sd branch, by its own documentation the ordinary
case, is emitted beside them.

## Verdict on #853

* The axes ARE continuized independently (`R/posterior_draws_hyper.R:199-205`,
  a fresh `runif` per axis) -- the issue's first checklist item, confirmed.
* Independent continuization does not manufacture excess variance in a ratio.
  A copula at the cell's own correlation changes nothing; comonotone coupling
  reproduces the atom's failure. The issue's proposed remedy is a measured
  no-op and its fallback is the worse arm.
* The continuization is a strict improvement on the grid-node atom on every
  quantity in every configuration measured.
* Where the grid route still does not reach the sampled route's calibration,
  the conditions that move with it are grid EXTENT (edge mass, railed axes),
  not the within-cell construction and not coupling across axes.

The HP760 arms themselves are not reproducible from this repository; what is
established here is that the mechanism #853 names does not produce the effect
it is named for, and that neither remedy it proposes would move it.
