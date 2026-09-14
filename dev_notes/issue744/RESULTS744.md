# The outer-grid measurement fixture fits the residual variance it simulates

## Decision

A defect, not a deliberate misspecification. `ogd_fixture_sim()` drew
`y = eta + rnorm(N, 0, 0.5)` (residual variance 0.25) and `ogd_fixture_fit()`
handed the gaussian arm `phi = 0.0625`. Since 657f179 the joint door reads `phi`
as the variance, so the fit held SD 0.25; before it the door read the SD and the
fixture passed 0.25, the same SD. Nothing at either call, in #333, #599 or the
measurement files states a reason for the gap, and the files score read rules as
properties of the posterior of the model the data came from.

The same crossing had been copied into eight more places, all fixed the same
way: `test-outer-grid-dump.R` (`.ogd_sim` / `.ogd_fit`), `test-cila.R`,
`test-grid-resolution-declined.R`, `test-nl-interval-support.R`,
`test-ccd-interval-support.R` (two fits), `test-nested-laplace-joint-ccd-local.R`
and `dev_notes/issue_328/measure_fit_ranking.R`.
`test-nested-laplace-joint-prune-misrank.R` simulates at SD 0.25 and fits 0.0625,
which is consistent, and is unchanged.

## The fix

`ogd_fixture_sim(phi = 0.25)` states the residual variance in the doors'
convention, draws at `.phi_to_kernel("gaussian", phi)`, and carries `phi` on the
simulation; `ogd_fixture_fit()` passes `sim$phi`. `sqrt(0.25)` is exactly 0.5,
so every simulated data set is bit-identical to before: only the fitted
residual variance moved (0.0625 -> 0.25). The binomial arm draws and fits no
residual variance and is unchanged by construction, so its figures were not
re-run.

## Build

`out/BUILD_ID_lib744.txt`: a clean `git archive` of b4b330c with the #745 R
files overlaid, which is dd24a12's `R/` and `src/` (RcppExports line endings
aside), R 4.6.1, x86_64-w64-mingw32, no objects before the build. The sweeps
read the helpers and test files from the working tree; their blob hashes at
measurement time are in the same file. `run744.sh` runs the three groups, each
with the flat prior and `box_uniform` read first.

## What moved

Figures are flat / proper unless marked; "box" is the shipped `box_uniform`
read the fixture states, "chord" the other.

### Box-mass (`boxmass744.R`, proper, the file's own prior)

| figure | mis-specified | correct |
|---|---|---|
| L5 box rule, endpoints / widths / median against floor | 0.0745/0.1287, **0.1491/0.0882**, 0.0124/0.0184 | 0.1043/0.1496, 0.2085/0.2297, 0.0072/0.0219 |
| L5, parts resolved | widths | none |
| L4 endpoints against floor | 0.3423/0.1712 | 0.2453/0.1308 |
| L4 steepest log multiplier | 25.3 | 15.5 |
| L4 errors vs L12, shipped -> rule | 0.2014->0.1838, 0.3153->0.0955, 0.1540->0.1761 | 0.2126->0.1797, 0.3326->0.1679, 0.1427->0.1487 |

The five-level claim "the correction reaches the widths" does not survive: on the
correctly specified model the box rule is unresolved on every part at five
levels. The four-level direction (interval toward the dense read, median away)
holds.

### Outer-grid dump (`ogd744.R`)

- Tempering by 3 moved the intercept SE 0.1880 -> 0.1852; it now moves
  0.1896 -> 0.1963. The direction reverses; the slope still moves by < 1e-6.
- Tempering by 3 moved the endpoints 0.039 against a floor of 0.105 (not
  resolved); the floor is now 0.0326 and the move 0.0985 (resolved). A ladder
  puts the crossing between 1.5 (0.0268) and 2 (0.0385); the test now shows
  both verdicts on one fit.

### Barycentre (`bary327.R`)

- Five-level placement: "all three parts above the floor" (flat) no longer
  holds; widths 2.70x and median 1.75x clear, endpoints 0.90x do not. The
  "widths more than 2.5x" bound holds.
- Four-level four-arm table (flat): shipped 0.2098 / 0.3285 / 0.1275, mass
  0.1667 / 0.1437 / 0.1503, location 0.0918 / 0.1836 / 0.0568, pair 0.0788 /
  0.1420 / 0.0576. The pair is now ahead of the mass rule on the widths
  (0.1420 against 0.1437 flat, 0.1580 against 0.1679 proper) where it used to
  trail it, so the "width gain partly given back" assertion is replaced by the
  pair's endpoint lead. Over seeds 1 to 5 the pair wins all three parts 5 of 5
  under both priors.
- Five-level: each rule still costs the interval and improves the median; the
  location rule halves the median 2.95x / 2.72x.
- The chord tables no longer "reproduce the first measurement to the digit":
  that measurement was on the mis-specified model.

### Descriptor plane (`plane333.R` + `analyse333.R`, gaussian)

| figure | mis-specified (flat / proper) | correct (flat / proper) |
|---|---|---|
| three-spread rho(R_M,R_L) pooled | 0.9034 / 0.9028 | 0.9225 / 0.9224 |
| test-sweep max R_M | (three-spread 78.26) | 24.72 (three-spread 44.38) |
| A-metric pooled, three spreads | 0.9483 / 0.9487 | 0.9139 / 0.9158 |
| four-level cells resolvable, endpoints (test sweep) | 51.6% / 56.3% | 29.7% / 32.8% |
| test-sweep scored permutation combinations | 12 / 12 | 9 / 12 |
| test-sweep loc-vs-mass strata with >= 25 rows | 3 / 3 | 2 / 2 |
| three-spread selecting stratum (four-level median), p | < 0.001 | < 0.001 |
| chord five-level median loc-vs-mass (three spreads) | -0.2795 (p 6e-05) | -0.1235 (p 0.15) |
| five-level median, per-cell sum against whole-grid | -0.3463 vs +0.0175 | -0.3889 vs +0.0140 |

Four assertions were sized to the mis-specified model's steepness and are
rewritten to what the plane claim needs: `max(R_M) > 30` becomes `R_M` exceeding
ten times `R_L`'s maximum; the four-level endpoint resolvability threshold 0.4
becomes 0.2; the scored-combination count 12 becomes at least 9 with the
Bonferroni divisor taken from the realized count; `length(got) > 2` becomes
`>= 2`. Every conclusion of the file stands on the correct model: the plane does
not select a correction on the test sweep, the four-level median is the one
selecting stratum over three spreads, the loc-versus-mass association is weak and
not of one sign, and one-cell improvements do not compose. Two readings weaken:
the A metric narrows the separation within a resolution but not pooled over
spreads, and the chord five-level median's run against the partition is no
longer significant.

### #328 ranking (`measure_fit_ranking.R`, summarised by `rank744.R`)

| arm | ep / wd / md above floor (of 24) | moved ranking nearer where resolved |
|---|---|---|
| flat chord | 0 / 0 / 16 (was 0 / 0 / 14) | 0 of 16 |
| proper chord | 0 / 0 / 15 (was 0 / 0 / 15) | 0 of 15 |
| flat box | 2 / 2 / 7 (was 0 / 4 / 10) | 0 of 11 |
| proper box | 1 / 5 / 8 (was 0 / 2 / 10) | 0 of 14 |

The criterion in `R/nested_laplace_joint_ccd_local.R` stands: wherever the grid
resolves the choice, the weight ranking is nearer the reference. That comment now
quotes these figures.

## Not re-run

- The binomial arm of the descriptor-plane sweep: its data and fit read no
  residual variance, so the fix cannot reach it.
- `issue331/coverage331.R`: its fixture (`LCCD_SIM`, `RESID_VAR`) was already
  consistent.
