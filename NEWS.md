# tulpa NEWS

## 0.0.147

* The outer-grid dump / rebuild harness is in the test suite
  (`tests/testthat/helper-outer-grid-dump.R`, gcol33/tulpa#322). A candidate
  construction for the outer integration weights is pure post-processing of a
  fit that already ran, so `outer_grid_dump()` writes the grid state
  (`joint_grid`, `log_marginal`, `dnode`, `weight_kind`, the axis tags and
  domains, the support the read was taken off, and the summary the fit shipped)
  and `outer_grid_rebuild()` re-reads the per-axis summary under any weight
  vector. The read goes through the engine's own `.nl_axis_quantiles()` ->
  `.nl_summary_quantile()`, never a second copy of it, and the round-trip
  assertion in `test-outer-grid-dump.R` -- rebuild-with-own-weights equals the
  shipped read -- is what makes an offline difference attributable to the
  weights alone. It holds exactly (0.000e+00) on a tensor grid, a global CCD and
  a locally refined grid.
* `outer_grid_noise_floor()` estimates the scale below which a difference
  between two reads is not resolved by the grid, as the spread of the read under
  a weight-preserving coarsening of each axis's own atom set (consecutive atoms
  merged at their weighted mean carrying their summed weight). Total mass and
  each group's first moment are exactly preserved, so only resolution is
  removed. On a one-axis dump with a Gaussian outer log-marginal the floor
  bounds the read's true error against the closed-form quantiles at every
  resolution from 9 to 81 levels.
* The joint multi-block driver records `dnode` on the fit beside the integration
  weights it was folded into. Recovering it afterwards is a division by
  `exp(log_marginal)`, which loses the scale and is undefined on a cell whose
  inner solve returned no finite marginal.

## 0.0.146

* The local-CCD cubic misfit score now reports the whitened gradient its own
  least-squares fit already estimated, and the refinement carries it per cell as
  `offset` / `offset_declined` on `local_ccd_info` (gcol33/tulpa#321). The score
  puts the linear term in its own design columns, so a cell whose outer
  log-marginal is a perfectly good quadratic that simply is not centred on the
  cell fits exactly and scores near zero however steep the gradient across it:
  passing it certifies that the design can represent the cell, not that the
  cell's coordinate is a representative point of it. `offset` is the
  standardized displacement of the cell's own peak from the cell's coordinate,
  in units of the marginal spread the whitening used, and nothing gates on it --
  a gradient across the cell is a cross-cell estimator question, orthogonal to
  the local shape `skew_max` reads.

## 0.0.145

* Local CCD refinement of the joint outer grid now keeps a refined cell's node
  cloud only where the cell's own outer log-marginal is close to the quadratic
  the cloud was placed from, and puts the cell back as its own mass atom where it
  is not (gcol33/tulpa#318). The refinement was a large win on an outer target
  that is quadratic in the transformed coordinate (summed absolute endpoint error
  against closed-form axis quantiles 7.3118 against 24.1142 for not refining, 48
  configurations of an equicorrelated Gaussian) and a net loss on a skewed one
  (26.2467 against 23.1874 over 27 configurations of a Gaussian copula with
  Gamma(2) marginals, 42.8578 against 38.1609 over 48).

  The mechanism is the cell's own non-quadraticity, and it is measurable from the
  design rather than inferred. A central composite design identifies a full
  quadratic exactly, so the least-squares residual of the nodes' measured
  log-marginals against intercept + gradient + Hessian in the whitened offset is
  the part of the cell the design cannot represent; the nodes are evaluated
  whatever the residual says, so the score costs no inner solve.
  `.joint_local_ccd_misfit()` reports it as a standardized cubic magnitude on the
  same convention the inner-Laplace `gamma_3` uses, and on the Gaussian target it
  is identically zero in all 48 configurations while on every skewed family it
  exceeds 0.08.

  The threshold is `.NL_DIAG$gamma3_ok` (0.5), one number for the inner band and
  this gate because both are a standardized third-order departure from the
  Gaussian the approximation was placed from. Where it belongs was measured: on
  an eight-family ladder (the Gaussian target plus Gaussian copulas with
  Gamma(1), (2), (4), (8), (16), (32) and (64) marginals, 48 configurations each
  bar 32 for Gamma(1)), 0.5 is the only threshold on the ladder 0.01 to 2 that
  improves or ties every family. Per family, gated against refining
  unconditionally: 7.3118 / 7.3118 on the Gaussian, then 36.3139 / 37.2185,
  40.8376 / 42.8578, 31.9596 / 34.3650, 34.1830 / 36.8361, 44.4462 / 47.2740,
  61.8421 / 63.1718 and 83.2492 / 83.2492 down the ladder; pooled 340.1435
  against 352.2843, with 365.0831 for never refining. Lower thresholds score
  better pooled (0.175 gives 337.4142) by regressing on the two least skewed
  families. 399 of 1196 candidate cells are declined across the ladder, none of
  them on the Gaussian target.

  On the four-axis two-block fixture the gate is measurably neutral, which is
  what it has to be: over the 20 distinct refined configurations the summed
  absolute endpoint error against the converged `m = 13` reference is 2.11091
  gated against 2.12847 unconditional and 3.12915 unrefined, and the largest
  single-configuration movement is 0.00696 against that reference's own 0.01716
  endpoint noise floor. Its per-cell scores there run 0.053 to 3.822, and the
  design-dominated `m = 3` configuration reads 0.126, so it keeps its cloud.
  `control$local_ccd$skew_max` overrides the threshold; `$local_ccd_info` gains
  `misfit`, `skew_max`, `cells_declined`, `misfit_declined` and
  `n_cells_declined`, and a refinement whose every candidate declined leaves a
  plain tensor grid that reports `theta_interval_read = "density"`.

* A locally CCD-refined joint outer grid now says what its per-axis
  hyperparameter intervals were read off, and how much of the support underneath
  them is a quadrature design rather than posterior mass (gcol33/tulpa#317). It
  is the one node set carrying both kinds at once -- a carried-over base cell
  holds the mass of its own cell, a refined cell's replacement cloud holds a
  partition-of-unity share of its cell's mass placed at the design's radius --
  and it reports `integration = "grid"`, so `.nl_node_support()` read it as a
  homogeneous density grid and nothing downstream could tell. The support is now
  named `"mixed"`, the fit carries `theta_interval_read` and
  `theta_interval_design_mass`, and `diagnostic_summary()` surfaces the pair.

  The reported numbers are unchanged, on the measurement. Three replacement
  reads were scored against the converged `m = 13` tensor reference (28561
  cells) of the four-axis two-block fixture, whose own noise floor is 0.01716 on
  the endpoints and 0.03853 on the widths: splitting the read into a mass CDF
  plus a per-cell moment-matched Gaussian, collapsing each refined cell's design
  block to one atom at its own weighted mean, and the gcol33/tulpa#308 moment
  read. Summed absolute endpoint error over seven base grids (`m = 3 ... 9`,
  `design_mass` 0.930 down to 0.092) is 0.63446 for the shipped weighted
  quantile, 0.73159 for the collapse, 0.74464 for the split and 1.20402 for the
  moment read, against 1.00289 for not refining at all; over fourteen further
  configurations reached by varying `local_ccd$max_cells` on the same fixture,
  1.65810 / 1.86035 / 1.89038 / 2.82062. The collapse is ahead at the single
  `design_mass = 0.930` grid (0.12022 against 0.16409) and behind at every lower
  one, which is the pattern a fix has to avoid. On analytic outer targets whose
  axis quantiles are closed form the same ordering holds: at `design_mass >= 0.5`
  on an equicorrelated Gaussian (19 configurations) 3.18024 for the quantile
  against 15.72792 for the collapse and 17.80607 for not refining. The moment
  read wins on that target because a Gaussian moment match is its exact family
  there; on a Gaussian copula with Gamma marginals, correlated the same way, it
  is the worst of the four (39.93958 against 26.24674 over 27 configurations,
  both measured with the refinement engaged unconditionally; under the
  gcol33/tulpa#318 gate above the quantile scores 24.59641 there).

* `tulpa_re_cov_nested()` reports the median and 95% interval of every derived
  covariance quantity (`sigma_i`, `rho_ij`, `Sigma_ij`, in every block) from the
  moments its integration design reproduces, instead of from a discrete weighted
  quantile over the design's node positions (gcol33/tulpa#308). A central
  composite design is a moment rule: its nodes sit where they reproduce the
  integrand's first two moments and carry no probability mass of their own, so
  the cumulative design weight across them is not a CDF. The discrete quantile
  clamped an out-of-support probability to the extreme node, which made every
  reported interval exactly the design's own extent -- at `k` covariance
  coordinates, `theta_hat +/- 1.1 sqrt(k)` posterior SDs, whose coverage is
  capped at `2 Phi(1.1 sqrt(k)) - 1 = 0.729 / 0.880 / 0.943 / 0.972` for
  `k = 1 ... 4` no matter how much data the model is given. The interval is now
  moment-matched on each quantity's own coordinate (`log` for a scale or a
  variance, `atanh` for a correlation, the identity for a covariance) and mapped
  back, so scale intervals stay positive and asymmetric and correlation
  intervals stay inside `(-1, 1)`. Measured `sigma_1` coverage over 200 seeds
  per arm (poisson, 60 groups x 40 observations, nominal 0.95), varying only the
  random-effect block: `k = 1` 0.735 -> 0.950, `k = 2` 0.880 -> 0.945, `k = 3`
  0.950 -> 0.945 (binomial SE 0.015 - 0.031). The `mean` and `sd` columns are
  unchanged, `tulpa_re_cov_gibbs()` was never affected, and the tensor-grid
  layout (`control$integration = "grid"`), whose uniform cells do discretize the
  density, keeps the weighted quantile.

* `.nl_wtd_quantile()` takes an explicit `outside` policy for a probability
  beyond the support's own cumulative range. The default `"clamp"` is the
  cumulative-mass convention a sample uses and is unchanged; `"na"` withholds
  the number where the support is a quadrature design rather than posterior
  mass, so the clamp is a stated choice instead of a silent one.

* The nested random-effect-covariance recovery gate
  (`test-re-cov-recovery.R`) is raised from 75% to the 85% its Gibbs sibling is
  held to. The 75% was the defect above, measured and accepted rather than
  diagnosed.

* Every remaining consumer that reads a median and interval off a CCD-integrated
  outer grid now takes them from the moments the design reproduces, the same way
  `tulpa_re_cov_nested()` does since gcol33/tulpa#308. The three are the joint
  multi-block per-axis hyperparameter summary (`theta_median` / `theta_ci_lo` /
  `theta_ci_hi`, gcol33/tulpa#309), the inline `spatial()` bar field's per-block
  `sigma` / `rho` and its MCAR `Sigma` summary (gcol33/tulpa#310), and
  `spatial_range()` / `temporal_corr()` (gcol33/tulpa#312). Confirmed before
  changing anything: on a real three-block joint CCD fit the reported upper
  endpoint equalled the node maximum on every axis at `0.000e+00`, and on a real
  MCAR fit four of the six derived quantities had BOTH endpoints exactly on the
  node extent.

  Measured coverage of the reported interval, gaussian response, 40 groups per
  factor and 1200 observations, varying ONLY the number of crossed random-effect
  blocks (which is the outer dimension `d`), 200 seeds per arm, nominal 0.95:
  `d = 3` 0.9350 -> 0.9450, `d = 4` 0.9900 -> 0.9700, `d = 5` 0.9650 -> 0.9300
  (binomial SE 0.007 - 0.018). Before, the mean interval width grew with the
  design -- 0.3131 / 0.3560 / 0.3899 across the three arms, in a data regime that
  did not change -- because the endpoints were the design's `1.1 sqrt(d)`
  whitened-SD reach rather than a property of the posterior; after, it is
  0.3241 / 0.3251 / 0.3261, flat to 0.6%.

  On the MCAR bar field, whose simulated `Sigma` gives an explicit truth (8 x 8
  lattice, 20 observations per cell, 150 seeds, nominal 0.95), every derived
  quantity's coverage improves: `sigma_1` 0.9067 -> 0.9133, `sigma_2`
  0.7467 -> 0.8333, `rho_12` 0.7533 -> 0.8400, `Sigma_12` 0.8667 -> 0.9267,
  pooled 0.8211 -> 0.8767 over 900 trials. It does not reach nominal; the
  residual is the inner Laplace's attenuation of a spatial covariance at 64
  units plus the outer Gaussian grid's fit to a skewed log-Cholesky posterior,
  which this does not address.

* Each nested-Laplace hyperparameter axis names the DOMAIN its interval is
  formed on, read from the same per-axis registry the outer Pareto-k
  unconstrains with (`.joint_axis_domains()`): a positive scale on `log`, the
  BYM2 mixing weight on `logit`, an unconstrained coordinate (a copy `alpha`, an
  MCAR log-Cholesky entry) on the identity. A proper-CAR `rho_car`, whose support
  is the adjacency's eigenvalue interval, has no domain the engine will guess, so
  its interval is withheld rather than reported as the design's extent.

* `ranef()` on a `tulpa_re_cov_nested()` fit reports the SAMPLED values for a
  random-effect coordinate the subspace debias selected, instead of the Gaussian
  mixture the fit had stopped using (gcol33/tulpa#314). The node mixture is
  reused rather than redrawn, so the group effects and the fixed effects
  marginalize one weighted node set, and a `source` column says per row which
  construction produced it (`"sampled"`, `"mixture"` or `"mode"`). A fit whose
  `S` contains no random effect is untouched, consumes no random number, and
  reports what it reported before.

* The inner-k-hat identity test strips the `$skew_correction` record alongside
  `$inner_*` (gcol33/tulpa#313). Every entry of that record is derived from
  gamma_3 or from the switch itself, so it necessarily differs with
  `control$diagnose_skew` on and off; the invariant the test asserts -- that the
  diagnostic consumes no randomness and changes no non-diagnostic field -- is
  unchanged.

* Every joint outer grid reports `weight_kind`, one entry per cell, saying
  whether that cell carries the mass of its own cell or an in-cell design weight
  (gcol33/tulpa#311). A fit integrated by one rule reports one value throughout;
  a locally CCD-refined grid is the one support carrying both, and it now says so
  per cell instead of leaving a consumer to read one kind off `integration`.
  `local_ccd_info` gains `n_design_nodes` and `design_mass`, the share of the
  integration weight sitting on design-weighted nodes.

  The per-axis median and interval keep the weighted quantile on a refined grid,
  which is a measurement rather than an omission. Scored against outer targets
  whose exact axis quantiles are known in closed form, the refined grid's
  quantile beats the unrefined grid's own in 8 of 8 configurations on a
  diagonal-Gaussian target (`d = 4/5`, 5/7 levels per axis, two peak
  sharpnesses), so declining it would withhold a number strictly better than the
  one the same summary reports one refinement earlier. On a target the moment
  rule cannot fit by construction (each axis carrying a `Gamma(k, 1)` marginal on
  a log-tagged axis, 18 configurations over `d = 4/5`, 5/7/9 levels,
  `k = 1/2/8`), the moment read beats the quantile as often on an unrefined grid
  (12 of 18) as on a refined one (11 of 18), and refining improves the moment
  read in only 2 of 18 -- so that advantage belongs to coarse grids and
  near-lognormal targets, not to the mixed weights, and moving a refined fit onto
  it is a separate question about density supports. The mixing is bounded by the
  refinement itself: each node cloud is clamped to its cell's Voronoi half-box,
  so a refined cell's mass is redistributed only inside the cell the unrefined
  grid had collapsed onto one point.

* Local CCD refinement scales each node cloud by the refined cell's MARGINAL
  spread instead of its conditional one (gcol33/tulpa#316). The cloud's scale
  came from a diagonal finite-difference stencil, and `1 / sqrt(-d2_j)` is the
  spread along axis `j` with every other axis held at the cell; the per-axis
  summary reports marginal spreads, and on a correlated outer posterior -- a
  sigma-alpha copy ridge is one -- the two differ by `sqrt(H_jj (H^-1)_jj)`. The
  stencil now also differences the cell's CORNER grid neighbours, which a cell
  interior on every axis always has and which the tensor base already evaluated,
  so `sqrt(diag((-H)^-1))` costs no extra inner solve. A cell whose corners are
  missing (a grid a previous pass spliced nodes into) or whose local `-H` is not
  positive definite keeps the conditional scale.

  Only the scale changes. The design stays axis-aligned, the per-axis shrink to
  the Voronoi half-box and the node clamp are untouched, and so is the
  weight-conservation argument. That is deliberate: the summary reads a weighted
  quantile over the refined grid and on design weights a cumulative sum is not a
  CDF, so what it returns is close to the design's own per-axis EXTENT. An
  axis-aligned design puts an axial node at `f_0 sd_j` on coordinate `j`; a
  design rotated by the Cholesky factor of the same covariance puts it at
  `f_0 L[j, k]`, and measured on an equicorrelated target with unit marginal SDs
  at `rho = 0.8` the per-axis extents are 2.200 / 1.760 / 1.765 / 1.918 times the
  SD -- so a rotated design reports an interval that depends on the arbitrary
  order of the axes. Rotating was built and measured and is not what shipped.

  Scored against an equicorrelated Gaussian outer target on identity axes, whose
  axis marginals are standard normal whatever the correlation is, at four axes
  and three levels per axis (`design_mass` above 0.99), reported 95% width as a
  fraction of the exact 3.91993:

  ```
  rho     0.0     0.5     0.7     0.8     0.9
  before  1.0052  0.8587  0.6748  0.5543  0.3942
  after   1.0052  0.9740  0.9540  0.9527  0.9569
  ```

  Mean absolute endpoint error at `rho = 0.8` goes 0.58240 -> 0.06182. The target
  is quadratic, so the stencil is exact there: the recovered marginal SDs match
  `sqrt(diag(Sigma))` to 1.1e-15 and the conditional ones `1 / sqrt(diag(Q))`
  exactly.

  On the package's own 4-axis multi-block fixture, refit on axis ranges that
  bracket its posterior and scored against a converged `m = 13` tensor grid
  (28561 cells, 583 s, refinement off), mean absolute endpoint error per base
  grid, with the reference's own `m = 11` against `m = 13` movement as the noise
  floor (0.01716 on endpoints, 0.03853 on widths):

  ```
  levels m      3       4       5       6       7       8    total
  design_mass  0.931   0.289   0.222   0.254   0.169   0.121
  before       0.1797  0.1533  0.1219  0.0616  0.0537  0.0469  0.6170
  after        0.1641  0.1518  0.1219  0.0616  0.0537  0.0469  0.6000
  ```

  The improvement is concentrated where #316 is: at three levels per axis the
  reported mean width goes 0.67781 -> 0.71103 against a reference 1.09089, and
  since both sit below the reference that direction does not depend on the
  reference's exact value. It is modest there because that fixture's outer
  correlations are weak (the refined cell's local correlations are -0.460 to
  +0.158) and because at four of the six base grids the estimated `-H` is
  indefinite, so the conditional scale is kept. Four other candidates were built
  and measured against the same reference -- removing the shrink and clamp and
  deleting absorbed cells, calibrating the scale to the cell's own mass share,
  collapsing each cloud back to a mass atom, and rotating by the Cholesky factor
  -- and none of them beats this across the sweep.

  Also measured: the shrink to the Voronoi box, which #316 names as the
  mechanism, does not bind on that fixture at three levels per axis. The
  conditional scale there is (0.2120, 0.1034, 0.1954, 0.0388) against a
  `half / node_reach` of (0.3197, 0.1818, 0.3197, 0.1420), and a candidate that
  removes the shrink and the clamp entirely returns bit-identical numbers.

* `local_ccd_info` gains `cell_share`, the share of the base grid's integration
  weight each refined cell held before any node was placed. It is a different
  number from `design_mass`, which is the share the refined region holds after:
  the replacement nodes sit nearer the peak than the cell's own coordinate did,
  so refining raises it. Reading both separates how concentrated the base grid
  already was from how much the refinement concentrated it.

* A joint multi-block fit records which outer integrator the caller ASKED for
  and why the CCD did not run (gcol33/tulpa#315). `$integration` names the
  integrator that ran, and `.nl_node_support()` keys the interval construction
  off it, so `"grid"` could not distinguish a tensor grid the caller chose from
  one a declined CCD fell back to; the reason existed and was thrown away
  outside a `verbose` message. `$integration_requested` carries the request and
  `$integration_declined` the reason -- `NA_character_` when nothing was
  declined, otherwise `"axis_count"`, `"unguessable_axis"`, `"degenerate_axis"`,
  `"modefind_ridge"`, `"modefind_boundary"`, `"modefind_degenerate"`,
  `"modefind_failed"`, `"hessian_singular"` or `"hessian_not_pd"`. The
  cell-count warning drops its "set `control$integration = \"ccd\"`" advice on a
  fit that already asked for one and was turned down, and names the decline
  instead.

## 0.0.144

* **The subspace debias reaches the grid and joint nested backends**
  (gcol33/tulpa#306, the follow-up to #304). `control$subspace_debias` is now
  accepted by `tulpa_nested_laplace()` -- both the single-block kernels
  (icar / bym2 / car_proper / rw1 / rw2 / ar1) and the multi-block driver -- and
  by `tulpa_nested_laplace_joint()` on both its single-block and multi-block
  paths, with the same settings and the same meaning as on
  `tulpa_re_cov_nested()`.

  The selector costs nothing new here. `control$diagnose_skew` (on by default)
  already re-dispatches the kernel at the fitted MAP cell and attaches both
  inner scores, so `.subspace_bands()` reads the per-index gamma_3 and inner
  Pareto-k-hat off the fit instead of paying for a probe of its own -- the one
  thing it had to learn is to prefer the stored per-index k-hat over re-fitting
  the raw importance curve, which a nested fit does not retain. The correction
  itself re-runs the settled grid once with the sampler on, because the
  corrected shape is a property of each cell and which cell is the MAP is only
  known after the first pass. A corrected fit then reports `$draws` -- each
  cell's Metropolis sample for the selected coordinates, the rest from the
  Gaussian conditional given them, mixed by the grid weights through the same
  `.re_cov_nested_beta_draws()` the RE-covariance backend uses -- instead of the
  Gaussian-mixture moments, and every coefficient-facing method reads them
  through the accessor it already used.

  Measured against the exact conditional posterior, computed by two-dimensional
  quadrature outside the engine. Both fixtures put the latent block where no
  observation reaches it, so the conditional posterior factorises into the
  fixed-effect target and an independent Gaussian and the quadrature is exact
  for the reported coefficients.

  Grid backend, rare-event binomial logit (n = 120): total absolute
  interval-endpoint error 0.5229 -> 0.1883 (mean of 5 seeds, sd 0.0672), a 64.0%
  cut; the reported centre moves -2.526 -> -2.63 against an exact posterior mean
  of -2.635. S is the intercept, selected on an inner k-hat of 0.705 with
  gamma_3 = -0.375 against an exact skewness of -0.388.

  Joint multi-block, the #300 coupled occupancy fixture: 0.6189 -> 0.163 / 0.341
  / 0.313 over three seeds, the centre -0.240 -> about -0.15 against an exact
  -0.1437 and the scale 0.389 -> about 0.447 against an exact 0.479.

  Coverage, with the exact posterior's OWN coverage as the reference rather than
  the nominal level -- the correction targets that posterior, so reproducing its
  coverage is the success condition and matching nominal is not. Grid backend,
  400 seeds, per coefficient:

  | level | coef  | exact          | plain          | corrected      |
  |-------|-------|----------------|----------------|----------------|
  | 0.95  | beta0 | 0.9050 (.0147) | 0.9750 (.0078) | 0.9150 (.0139) |
  | 0.95  | beta1 | 0.9300 (.0128) | 0.9550 (.0104) | 0.9225 (.0134) |
  | 0.80  | beta0 | 0.7600 (.0214) | 0.8225 (.0191) | 0.7575 (.0214) |
  | 0.80  | beta1 | 0.7950 (.0202) | 0.8375 (.0184) | 0.7800 (.0207) |

  The corrected rate is within 0.52 SE of the exact posterior's on all four; the
  plain Laplace sits 2.4 to 4.8 SE above it. Its apparently better agreement
  with the nominal level is an over-wide, mis-shaped Gaussian, not accuracy.
  Joint multi-block, 200 seeds with S pinned to both coefficients, same
  reference: 0.9500 / 0.9600 / 0.7900 / 0.7400 exact against 0.9598 / 0.9598 /
  0.7940 / 0.7387 corrected and 0.9749 / 0.9548 / 0.8191 / 0.7739 plain (one
  seed's fit reported an NA bound, so n = 199 for the fits and 200 for the
  reference).

  Cost, over those sweeps: 0.0324 s (SE 0.0005) -> 0.1765 s (SE 0.0011) on the
  grid backend at mean |S| = 1.060, and 0.0662 s (SE 0.0012) -> 0.4019 s
  (SE 0.0045) on the joint one at |S| = 2, i.e. 5.45x and 6.07x. Roughly half of
  that is the second grid pass and the rest the sweeps themselves.

  An EMPTY selection leaves every backend bit-for-bit identical to the plain
  path, asserted per backend (log-marginal, weights, modes, the per-cell
  fixed-effect pieces, `summary()` and `vcov()`) in the new
  `tests/testthat/test-subspace-debias-backends.R`.

* **The band selector under-flags on the coupled fixture, measured.** On the
  #300 coupled occupancy fit both coefficients band `good` on both inner scores
  -- gamma_3 reads 0.256 and -0.126, the inner k-hat 0.378 and 0.329 -- while
  the exact skewness of the occupancy coefficient is 1.198, which is
  `unreliable`. gamma_3 recovers 0.21 of it there, below the 0.564 to 0.943
  range #304 measured on the separable fixtures, and the inner k-hat over 256
  one-dimensional draws does not separate that target from a Gaussian either. So
  on that fit the selector takes nothing and `control$subspace_debias$idx` is
  what pins the set; the joint numbers above are the pinned run. The `ok` band
  floor is unchanged -- #304 measured it, and one fixture where a lower-bound
  estimator undershoots harder than usual is a fact about the estimator, not a
  reason to move a threshold that was itself set on measurement.

* **Threading.** The sampler draws from R's RNG, so a debiased outer grid is
  integrated serially whatever `n_threads_outer` asked for, on both the
  single-arm driver and the sparse joint one. The cheap warm-start screen never
  runs the correction, and a joint cell whose inner solve took the s2z rank-1 or
  the PSD eigen-clamp path carries no usable factor to build the surface from
  and is left uncorrected -- the same two paths `diagnose_skew` declines on, for
  the same reason.

* **One request, one unwrap, one assignment point.** The debias request travels
  through every nested kernel entry as ONE R list (`idx` plus the sweep budget)
  rather than four parallel arguments; `unwrap_debias()` / `DebiasRequest`
  (`src/laplace_spec_fit.h`) turn it into the solver's options at each entry,
  and `run_subspace_debias()` (`src/subspace_debias.h`) guards, runs and records
  the outcome for all three Newton loops -- the single-arm spec loop and both
  joint loops -- so the "empty index set is a no-op" contract and the result
  mapping are written once. `cpp_laplace_fit_multi_re()`'s four #304 arguments
  are folded into the same list; pre-release, no shim.

* `control$subspace_debias` is left refused at `n_quad > 1` on
  `tulpa_re_cov_nested()`, which is the honest answer rather than a gap. The
  adaptive Gauss-Hermite inner marginal integrates each group's random effects
  out, so at the fitted point there is no conditional latent field: no joint
  precision to take `Sigma e_i` from, hence no Gaussian-conditional-mean surface
  and nothing for the sampler to move along. What could be corrected there is
  the outer optimum's own Laplace approximation, which is a different
  construction on a different density and would not share this machinery.

* Reporting the SAMPLED values per group for a random effect the closure pulled
  into S is split out as gcol33/tulpa#314: the draws exist per node but are not
  recorded on the fit, so `ranef()` still reports the Gaussian mixture there.

* Fixed: `tests/testthat/test-inner-pareto-k.R` asserted that a fit is
  bit-for-bit identical with `control$diagnose_skew` on and off after stripping
  `inner_*` and `timing`, which stopped holding when #302 added the
  gamma_3-derived `$skew_correction` record two commits later. The record is
  diagnostic-derived, so it is stripped alongside them (gcol33/tulpa#313).

## 0.0.143

* **The joint tier's fixed-effect block is extracted inside each cell's own
  solve** (gcol33/tulpa#307). Filling `$grid_modes` / `$grid_hessians` on a joint
  fit (#305) read the block off the cell precision, so the joint kernels ran with
  `store_Q` internally and the whole outer grid's precision was alive at once
  between the kernel call and the extraction. Both joint Newton loops now take
  the request -- the leading block size plus the field sum-to-zero groups, both
  fixed by the latent layout before the first solve -- and return the block on the
  `LaplaceResult` `re_cov` contract, which the grid driver emits per cell. The
  dense loop builds one cell's CSC, extracts, and releases it; the sparse loop
  reads the builder's own CSC, so there the block costs no precision copy at all.
  `store_Q` is once again the caller's own knob, passed straight through.

  Measured on an ICAR chain fixture (`n_fixed = 8`, 40-cell grid, R heap
  high-water over three paired runs): peak 49.9 vs 51.5 MB at `n_x = 408`,
  53.6 vs 58.2 MB at 2008, and 62.1 vs 71.9 MB at 6008 -- the saving tracking the
  grid's precision (1.63 / 4.07 / 8.54 MB, i.e. 41.7 / 104.2 / 218.7 KB per cell)
  and growing with the field where the retained block does not (868 bytes per
  cell at every size). Fit time is unchanged: 0.27 / 0.73 / 1.73 s against
  0.31 / 0.72 / 1.75 s.

  The blocks are byte-identical to what `cpp_joint_inner_vcov_blocks()` returns
  for the same cells -- same bytes in, same routine -- so `summary()`,
  `confint()` and `vcov()` report exactly what they did in 0.0.142, and draws,
  modes, weights and `log_marginal` are untouched.

* **Local-CCD refinement no longer costs a joint fit its intervals**
  (gcol33/tulpa#307). The node solves carry their own fixed-effect block through
  the splice alongside the inner modes, so a refined grid reports instead of
  recording `grid_fixed_declined = "local_ccd_refined"`. Refinement itself is
  unchanged.

* One extraction algebra behind all of it: `src/inv_block_extract.h` holds the
  conditioning-by-kriging constraint correction and the diagonal-block
  extraction, both templated on a solve oracle. `laplace_newton.h`'s
  `inv_block_layout` path drives it against the live Newton factor;
  `extract_inner_vcov_block_cell()` drives it against a factorized cell with the
  constraint; the joint loops go through the latter.

## 0.0.142

* **A joint fit reports uncertainty on its fixed effects** (gcol33/tulpa#305).
  `summary()`, `confint()` and `vcov()` on a `tulpa_nested_laplace_joint()` fit
  reported the point estimates and `NA` for every standard error and both
  bounds, on both the single-block and the multi-block path. The grid
  marginalizer `.nested_fixed_moments()` reads one representation --
  `$grid_modes` and `$grid_hessians`, the per-cell fixed-effect mode and
  marginal precision -- and the joint driver stored neither, so a joint fit
  could not put an interval on any coefficient.

  Both joint drivers now fill that same pair through one shared helper,
  `.joint_attach_grid_fixed()`, so the two tiers reach the one marginalizer
  rather than growing a second one. Both joint layouts stack every arm's
  coefficients as a contiguous prefix of the latent vector, so the extraction
  is arm-aware by construction: it takes the whole `1:n_fixed` block in one
  pass and reports each arm's coefficients under its own name. The per-cell
  block comes from `cpp_joint_inner_vcov_blocks()`, the joint tier's existing
  per-cell inner-covariance extraction, so the reported covariance is the
  field-constrained one the fit's own `tulpa_posterior_draws()` mixture is
  generated from. The reported covariance is the law of total variance over the
  outer grid and so carries both the within-cell curvature and the between-cell
  hyperparameter spread.

  Measured against references outside the engine. On the #300 coupled fixture,
  whose log posterior is written independently in R, the reported covariance
  matches the inverse numerical Hessian of that density at the mode to 1.4e-09
  relative. A one-arm joint fit and the single-block fit of the same model --
  the same data, block and grid -- now agree to 5e-08 on every coefficient and
  standard error, where the joint side previously produced `NA`; that
  equivalence is asserted for poisson, binomial and gaussian. Over a
  multi-cell ICAR grid the mixture matches the independent R implementation of
  the same law-of-total-covariance (`.joint_mixture_moments()`) to 8e-17, and
  200000 draws of the fit's own posterior mixture reproduce its standard errors
  to 0.1%. CI coverage is judged by the existing recovery sweep with the joint
  fitter substituted rather than by a second harness.

  Retention is `control$keep_grid_hessians` (default `TRUE`), and costs
  `O(n_fixed^2)` per cell: 860 bytes per cell at `n_fixed = 8`, unchanged as
  the field grows from `n_x = 408` to `n_x = 6008`, with no measurable fit-time
  overhead (-0.1% over three paired runs). Reading the block needs the cell
  precision, which the kernels now keep during the fit and drop again unless
  `control$store_Q` asked for it; that transient is the existing `store_Q`
  peak, ~64 bytes per latent per cell, and gcol33/tulpa#307 tracks removing it
  by extracting the block inside the joint Newton loop.

  Draws, modes, weights, `log_marginal` and the hyperparameter moments are
  bit-for-bit identical with the retention on and off -- this adds reporting,
  not inference. Where the retention cannot be trusted it declines with a
  reason on `$grid_fixed_declined` instead of going quiet: `"not_requested"`
  when switched off, and `"local_ccd_refined"` when local-CCD refinement
  rewrote the outer grid after the cells were stored (local-CCD keeps
  precedence, so no existing fit changes).

* `.nested_fixed_moments()` skips a grid cell that carries no integration
  weight. A pruned cell with no retained block previously turned the whole
  marginalized covariance into `NA`.

## 0.0.141

* **The reliability band is now the debias SELECTOR: exact MCMC runs on only the
  misfit directions** (gcol33/tulpa#304). Escalation used to be whole-fit and
  all-or-nothing (`tulpa_re_cov_nested` -> `tulpa_re_cov_gibbs`, or a grid
  refinement), with nothing saying which directions needed exact treatment, so
  every coordinate paid the sampler's price including the ones the Gaussian
  already fits. The per-index inner diagnostics are already a map of exactly
  that. `control$subspace_debias` (default `FALSE`) bands every probed index,
  takes `S` = the misfit set, and corrects only `x_S` by Metropolis with
  `x_{-S}` carried at its Gaussian conditional.

  The sampled surface is `x(u) = mode + V Sigma_SS^{-1} L u` with
  `V = Sigma E_S` -- the q-dimensional generalization of the one-dimensional
  conditional-mean curve both inner diagnostics already walk, reusing
  `inner_probe_column()` rather than a second solve. The Gaussian restricted to
  it is exactly `N(0, I)` in `u`, so the walk is spherical and the Laplace
  shaping lives in the coordinates. Random-walk Metropolis, not NUTS: each
  evaluation is one call of the Newton loop's own penalized objective (O(N), no
  factorization, no derivative), so a gradient sampler would buy nothing and
  would need a derivative the loop does not expose along the surface.

  Selection is at the `ok` band, one step below the `unreliable` band the
  reporting layer flags on, because `gamma_3` is a LOWER bound on the true
  skewness (0.564-0.943 of the exact value across the engine's own fixtures), so
  selecting at the reported boundary would leave genuinely misfit coordinates
  uncorrected. The inner importance k-hat (#303), which needs no derivative and
  does not undershoot the same way, is folded in as the worse of the two -- on
  the rare-event sweep below it is what bands 225 of 400 intercepts
  `unreliable` where the mean `gamma_3` is only -0.597.

  MEASURED against an exact reference (Bernoulli random intercept at fixed RE
  SD, the group intercepts integrated out by Gauss-Hermite and `p(beta | y)`
  marginalized on a grid): the exact intercept marginal is
  `mean -4.0050, 95% (-6.5082, -2.2898)`; the Laplace Gaussian gives
  `mean -3.4025, (-5.3153, -1.4897)`, total endpoint error 1.9930; the
  correction on `S = {intercept}` gives `mean -3.7675, (-6.1446, -2.0611)`,
  endpoint error 0.5923 -- a 70.3% reduction. Residual bias remains, as expected
  from a lower-bound skewness estimate and a Gaussian conditional.

  MEASURED against correcting EVERY coordinate, which is the question of whether
  a subspace is enough (200 seeds x 2 coefficients, Bernoulli random intercept,
  60 groups of 3, at the true RE SD, against the exact quadrature marginal,
  nominal 0.95): plain Laplace 0.9050 (se 0.0147), subspace debias 0.9275
  (0.0130) at a mean `|S|` of 0.945 coordinates, full-S debias over all 62
  latent coordinates 0.9225 (0.0134). Correcting about one coordinate recovers
  what correcting all 62 recovers, and costs 0.313 s against 0.461 s.

  MEASURED against the full Gibbs debias (400 seeds, rare-event binomial-logit
  with a random intercept, pooled over both coefficients): at nominal 0.95,
  plain Laplace 0.9738 (se 0.0057), subspace 0.8662 (0.0120), full Gibbs 0.8888
  (0.0111) -- subspace within 1.4 standard errors of the full debias; at nominal
  0.80, plain 0.8738, subspace 0.7175 (0.0159), full Gibbs 0.7037 (0.0161),
  within 0.6 standard errors. Cost 0.468 s against the full debias's 1.287 s,
  **2.75x cheaper**. On the small-group binary RE fixture every probed
  coordinate bands `good` (max `|gamma_3|` 0.236 over all 122 latent
  coordinates), `S` is empty, and the fit is the plain one.

  A second whole-fit sweep on a denser fixture -- Bernoulli random intercept, 60
  groups of 3 (n = 180), `beta = (-2.5, 1)`, `sigma_u = 1`, 200 seeds, all three
  backends at their defaults on the same data -- does NOT reproduce that match,
  and the reason is worth stating rather than averaging away. Beta coverage
  pooled over both coefficients at nominal 0.95: plain nested 356/400 = 0.8900
  (se 0.0156), subspace 359/400 = 0.8975 (0.0152), full Gibbs 376/400 = 0.9400
  (0.0119) -- a 2.2 standard-error gap, at 1.074 s against 8.123 s. The
  correction is not what falls short there. On the SAME 200 seeds its own layer,
  conditional coverage at the true sigma, goes 0.9050 plain -> 0.9275 subspace
  against 0.9225 for correcting every one of the 62 latent coordinates, so one
  coordinate recovers what all 62 do, at 0.313 s against 0.461 s. What is left
  is outer: the nested path's `sigma_1` interval covers 150/200 against the
  Gibbs sampler's 199/200 and its intercept interval is 26% narrower, and
  turning the correction on moves neither number. That is a different layer,
  and the two backends are not even integrating the same hyperprior (the
  conjugate `Sigma | b` draw cannot take the flat default); gcol33/tulpa#308
  separates it.

  Against exact Gauss-Hermite quadrature on that fixture (24 seeds x 2
  coefficients, max grid tail mass 8.4e-14) total absolute endpoint error is
  plain 24.2040, subspace 21.2363 (-12.3%), every-coordinate 7.7709 (-67.9%).
  So the band-selected subspace recovers the full correction's COVERAGE while
  recovering about a fifth of its endpoint accuracy: it puts the interval in the
  right place without fully fixing its shape. Closing that remainder is what the
  coupling closure below would do, and only by growing `S` to nearly the whole
  coupled block.

  The COUPLING CLOSURE (grow `S` by the precision-graph neighbours whose partial
  correlation with a member exceeds a threshold) is implemented and was measured
  both ways rather than assumed, which is what the issue asked for. At the
  default threshold it changes nothing: against the exact marginal it moves the
  endpoint error 0.5923 -> 0.5651, a difference of 0.027 against a combined seed
  standard error of 0.038, and across the 400-seed sweep it fires on 163 seeds
  yet leaves coverage identical on 1572 of 1600 seed-coefficient-levels. The
  reason is that the partial correlations between a fixed effect and the random
  effects only run about 0.09 to 0.25 on these models, so a threshold in the
  usual "strong coupling" range never bites.

  Lowering it far enough to bite does move the finer metric, and that is worth
  stating precisely rather than glossing: on a 14-coordinate fixture the total
  endpoint error against the exact marginal falls 4.43 -> 1.21 only once the
  threshold reaches 0.05, at which point `|S|` has grown to 13.4 of 14 -- the
  full debias wearing a different name rather than a subspace one. So
  conditioning `x_{-S}` on the Gaussian does NOT reproduce the exact marginal
  endpoint for endpoint; it removes about 70% of the Gaussian's endpoint error
  at `|S| = 1` and the rest is not cheaply recoverable by growing `S`.

  On the arbiter the issue actually names -- interval coverage -- that residual
  does not show: at 200 seeds the `|S| = 1` correction and the all-62-coordinate
  correction cover 0.9275 and 0.9225, indistinguishable. Coverage is the coarser
  of the two metrics, and the closure is off by default because nothing measured
  here asks for it. It stays available as `closure = TRUE` or an explicit
  threshold.

  `S` is recorded on the fit as `subspace_debias` (selected indices, the
  per-index band table they were read from, what the closure added, and the
  per-node acceptance), so the escalation is auditable rather than implicit. An
  empty `S` is not a special case of anything: the sampler is never entered, no
  random number is consumed, and the fit is bit-for-bit the plain Laplace fit --
  asserted on both the solver and the front door.

* **One random-walk Metropolis definition, not two** (`src/rwmh.h`). The
  starting scale `2.4 / sqrt(d)`, the Roberts-Gelman-Gilks target acceptance,
  the Robbins-Monro burn-in adaptation and the accept test were written out
  inline in the covariance Gibbs sweep and would have been written out again for
  the subspace debias. They are now one set of primitives both consume.
  `rw_accept()` draws its uniform unconditionally so a sweep consumes exactly
  one uniform per test whatever the ratio is, which is what keeps the migrated
  Gibbs sweep's RNG stream unchanged.

## 0.0.140

* **`gamma_3` is now consumed, not only graded: the inner-Laplace marginals can
  be skew-corrected** (gcol33/tulpa#302). The cubic term was computed, banded
  and printed, and nothing read it -- so the inner layer was nested
  approximation with no debias, the position this engine is designed against,
  one layer in from where that argument is usually made. `summary()` and
  `confint()` on a nested-Laplace fit run with `control$skew_correct = TRUE` now
  report Cornish-Fisher marginal quantiles at each coefficient's own `gamma_3`,
  gated to the `good` / `ok` bands, and the Gaussian quantiles everywhere else.
  `$skew_correction` records the per-coefficient `gamma_3`, band and
  eligibility; a `skew_applied` attribute on `summary()` / `confint()` records
  what was used at the requested level. Wired through `tulpa_nested_laplace()`
  and both `tulpa_nested_laplace_joint()` paths. (At this release a joint fit
  recorded the correction without showing it, because the joint driver retained
  no per-cell fixed-effect Hessians for the grid-marginalized covariance;
  gcol33/tulpa#305 supplies them in 0.0.142 and the correction applies there.)

  Rue, Martino & Chopin (2009) Sec 3.2.3 fit a skew normal here, under three
  constraints -- mean `gamma^(1)`, variance 1, third log-density derivative at
  the mode `gamma^(3)`. Two of those inputs exist in this engine and one does
  not: `gamma^(1)` comes from their denominator expansion, which is diagonal
  only in their augmented `x_j == eta_j` representation
  (`src/inner_laplace_skew.h` carries the reason). A skew normal fitted on the
  cubic term alone is therefore a different construction from theirs, and its
  attainable skewness saturates at `|skewness| ~ 0.995` with the shape parameter
  diverging as that bound is approached -- inside the very band the correction
  is gated to. The Cornish-Fisher expansion is the quantile-side inverse of the
  same Edgeworth series `gamma_3` is the leading term of, is linear in
  `gamma_3` so it does not saturate, and returns quantiles directly.

  The correction is skewness-only and therefore partial, which is measured
  rather than asserted. Against exact quadrature quantiles of rare-event
  binomial-logit posteriors it cuts total absolute endpoint error from 2.4931 to
  1.3837 (44.5%), improving both endpoints in every case. On CI coverage over a
  small-group Bernoulli random-effect fixture (N = 48, 200 seeds x 2
  coefficients) it is directionally right and immaterial: nominal 0.95, Gaussian
  0.9650, corrected 0.9600; nominal 0.80, 0.8050 vs 0.8075; nominal 0.50, 0.4950
  vs 0.5000 -- every difference inside one standard error. Two reasons the
  coverage gain is smaller than the marginal gain: `gamma_3` is a lower bound on
  the true skewness (0.875-0.943 of it on the cases above), and a biased Laplace
  mode stays biased because the location term is not computed. **The correction
  is therefore OFF by default** (`.NL_DIAG$skew_correct`); the coverage
  measurement does not justify defaulting it on. Draws, modes, weights and every
  other field the solve produced are bit-for-bit unchanged either way -- this is
  post-processing on the reported quantiles.

  New: `.nl_skew_marginal()`, `.nl_skew_by_fixed()`, `.nl_skew_correction_attach()`
  (`R/laplace_diagnostics.R`), `src/cornish_fisher.h` / `.cpp`,
  `tests/testthat/test-inner-skew-correction.R`, and a paired
  corrected-vs-Gaussian coverage gate in `test-nested-laplace-recovery.R`.

## 0.0.139

* **`gamma_3` now scores coupled multi-predictor likelihoods instead of
  declining on them** (gcol33/tulpa#301). The cubic Edgeworth term assumed a
  log-likelihood that is a separable sum of one-eta terms, so every unit reading
  several linear predictors at once -- a zero-inflation mixture's (count, zi)
  pair, a `CellCouplingSpec` cell's arms (tulpaObs's `occu_cover`) -- had no
  per-eta third derivative and came back `NaN` for good. The expansion is
  unchanged; only the contraction widens, to
  `sum_units sum_{a,b,c} T^{abc} u^a u^b u^c` with `T` the unit's third
  derivative in its linear predictors and `u` the eta response to `Sigma e_i`.
  The separable case is the one-coordinate special case of it.

  `T` is never materialised (`src/curvature3_contract.h`). Partition the unit's
  coordinates into `K` blocks and the contraction equals
  `sum_a d/ds [u' L''(e + s u^(a)) u]` at `s = 0`, because moving along block
  `a`'s slice of the direction differentiates exactly that block's coordinates.
  Each term is one central difference of the Hessian the likelihood already
  returns for the Newton solve, so the whole tensor costs `2K` extra evaluations
  per unit and no storage, at any block sizes. For a `CellCouplingSpec` that
  Hessian is the analytic `CellDerivs` block, so this is one finite-difference
  layer on an exact quantity, not a difference of a difference.

  The step is scaled PER BLOCK off that block's own eta magnitude, matching the
  eta-space step the scalar working-weight fallback takes. Measured against a
  five-point third derivative of the cell log-density: identical to a single
  global step while the arms share an eta scale, and 1.8x more accurate once one
  arm's `|eta|` is 67x the other's. The contraction is symmetrised over index
  permutations; for this block decomposition that is algebraically the plain sum
  (the three relabelings coincide), so it buys robustness at a block whose own
  quotient could not be formed rather than variance reduction.

  Verified against the exact posterior, not asserted: on the coupled two-arm
  occupancy fixture the engine's `gamma_3` reproduces the same quantity computed
  independently in R -- the third derivative of the exact log posterior along
  the same conditional-mean curve -- to 8e-4 relative, and the zero-inflated
  Poisson to 5e-4. Held against the two-dimensional quadrature of the same
  posteriors it has the right sign and undershoots, closely where the skewness
  is small (0.86 and 0.93 of the exact value at `|skew| ~ 0.11-0.13`) and by
  about half where it is moderate (0.299 of an exact 0.530). That last case is
  pinned in the suite because it has a consequence: `gamma_3` is a LOWER BOUND
  on the skewness, and there it bands "good" where the exact value bands "ok".
  A coupled Gaussian cell, whose Hessian is constant, reads exactly `0`.

  The scalar single-coordinate path is untouched: byte-identical across seven
  fixtures on both the family-enum and the spec entry (`identical()`, max
  absolute difference exactly 0), verified against a build of the preceding
  commit.

* `"coupled_likelihood"` is retired from the inner-skew decline vocabulary and
  from `.INNER_SKEW_STRUCTURAL` -- coupling several processes in one likelihood
  no longer describes anything permanently unscorable. What remains is
  `"curvature3_unavailable"` (a spec that ships no way to reach a third
  derivative) and `"coupled_arm"` (a coupled fit for which no cell tensor could
  be built at all). Every decline still returns `NaN`; one unreadable cell takes
  the whole contraction to `NaN` rather than silently understating the sum.

* New `cpp_cell_coupling_curvature3()` exposes the contraction at one cell,
  outside any solve, so a registered spec's tensor can be checked against a
  direct numerical third derivative of its own log-density and the step policy
  measured rather than asserted (`tests/testthat/test-cell-curvature3.R`).

## 0.0.138

* **The inner Laplace layer now has a likelihood-agnostic reliability number**
  (gcol33/tulpa#303). `gamma_3` scores the inner Gaussian by expanding the joint
  log density along the Gaussian conditional-mean curve at a probed latent
  index, which needs a per-observation third derivative -- so a coupled
  multi-process likelihood (a ZI mixture, tulpaObs's `occu_cover`) declines
  permanently and the fit has only the outer k-hat, which scores a different
  layer. `inner_pareto_k` walks the SAME curve and simply evaluates the joint
  density along it: the inner Gaussian is an importance proposal for the exact
  conditional posterior, and the Pareto-smoothed shape of that ratio scores the
  approximation directly. No likelihood derivative anywhere, so it answers
  wherever a mode was found.

  It runs on the probed subspace, not the field. Importance sampling degrades
  with dimension on its own, so a k-hat over all `n_x` coordinates would report
  `n_x` rather than the approximation; one dimension per probed index keeps
  every sampling problem 1-D and makes the number directly comparable to the
  `gamma_3` for the same index. The engine returns the draws and the joint log
  density at them (`src/inner_laplace_is.h`); the Pareto fit is the existing
  shared `.nested_is_pareto_k()` core, which now accepts an injected draw
  matrix, so there is one importance-sampling k-hat in the package rather than
  two. The conditional-curve solve `v_i = Sigma e_i` is extracted to
  `src/inner_laplace_probe.h` and shared with the cubic term; neither
  refactorizes.

  A Pareto shape index is scale-free -- it describes the SHAPE of the
  importance-weight tail and says nothing about its size. Measured on the
  engine's own fixtures at 256 draws: a gaussian-family coefficient, where the
  inner Laplace is EXACT and `gamma_3` is exactly 0, reads k-hat 0.19 / 0.26 at
  importance efficiency 1.000, and a balanced binomial intercept (N = 500,
  S = 230, `gamma_3` = -0.007) reads 0.640 at efficiency 0.99998. Both are noise
  on a proposal that needs no correction. The k-hat is therefore banded only on
  probed indices whose realized efficiency falls below
  `.NL_DIAG$inner_k_material_ess` (0.995); the raw shape is reported either way,
  and `inner_pareto_k_uniform` records that no index carried a correction worth
  describing.

  The two inner scores agree where both compute. Across a binomial-intercept
  skewness ladder ((N, S) = (500, 230), (500, 60), (100, 3), (20, 2), (15, 1)),
  `|gamma_3|` runs 0.007 to 0.897 and the importance efficiency falls
  monotonically with it (0.99998, 0.9962, 0.850, 0.807, 0.634 -- Spearman 1.00);
  the tail shape follows at Spearman 0.90, and the band verdicts agree rung by
  rung. On the coupled fixture, where `gamma_3` is NaN for every index, the arm
  with the larger exact posterior skewness (0.53 vs 0.13 by direct quadrature)
  is the arm with the lower efficiency (0.983 vs 0.997).

  `.tulpa_combined_reliability()` folds the inner layer's two scores into one
  band -- the worse of them where both computed, the one that did where only one
  did -- so a fully coupled fit reads "reliable (both layers good)" instead of
  "inner Laplace not assessed". Reported through `diagnostics()`,
  `print.laplace_diagnostics()` and `diagnostic_summary()`; declines carry a
  reason from the same closed vocabulary the outer k-hat uses.

  The draws are engine-owned and deterministic rather than taken from R's
  stream, so requesting the diagnostic leaves a fit bit-for-bit unchanged and
  the reported k-hat does not flap with the seed. Cost is one joint-density
  evaluation per draw per probed index -- no factorization -- which is why the
  budget is a fixed engine constant rather than the outer diagnostic's
  `k_samples`, whose draws each cost a full inner Laplace solve.

* Fixed: `.tulpa_inner_k_reliability()` reads its fields with `[[`. On a
  declined fit the only field carrying the `inner_pareto_k` prefix is the reason
  string, which `$` would partial-match into the k-hat.

## 0.0.137

* **The engine can now test its own coupled likelihood paths** (gcol33/tulpa#300).
  `CellCouplingSpec` has been virtual-dispatched per cell since the joint driver
  gained a coupled branch, but every genuinely non-separable implementation lived
  downstream in tulpaObs, so the cross-arm scatter, the dense-pair allocation and
  the per-cell derivative contract were only ever exercised by a consumer. A
  minimal coupled likelihood is now registered here as a test fixture:
  `test_occupancy_mixture` (`src/test_cell_coupling_occupancy_mixture.h`), a
  two-arm occupancy mixture whose cell density is
  `psi prod_v Bern(y_v | p_v) + (1 - psi) 1{no detection}`. A cell with a
  detection factorises; a cell with none puts the occupancy state and every visit
  inside one logarithm, so `d^2 log p_cell / d eta_occ d eta_det` and the
  cross-visit second derivatives are nonzero. It writes both dense cross blocks
  (`(occ, det)` and the `(det, det)` self block) rather than taking the rank-1
  self-cross shortcut, so a third-derivative tensor has an explicit Hessian to
  difference, and it declares those two through `dense_cross_pairs()` while
  omitting the one-row occupancy self block.

* **`cpp_cell_coupling_evaluate()` exposes what a spec actually writes.** The
  inner Newton chains each spec's eta-space derivatives through the design and
  scatters them immediately, so nothing a spec computes was visible from R. This
  export drives any registered spec at one cell and returns the cell log-density,
  the per-arm gradient, the per-arm negative-Hessian diagonal and every dense
  cross block, with the same buffer-allocation policy the kernel applies (pairs
  read from the spec's own `dense_cross_pairs()`, rank-1 descriptor supplied).
  It is the surface a finite-difference check of a spec's analytic derivatives
  runs on.

* **The exact-quadrature ground truth reaches the coupled case.**
  `test-inner-skew.R` held the separable scalar reference: integrate the exact
  posterior on a grid and hold `gamma_3` against its central moments. The same
  construction is now carried to two dimensions over the coupled fixture's
  intercept-only conditional posterior, with three things asserted rather than
  assumed -- the two-dimensional quadrature reproduces the trusted scalar
  reference on a product posterior, the R density agrees cell by cell with what
  the compiled spec evaluates, and the grid is converged under widening and
  refinement. The fixture's exact marginal skewness is 0.53 on the occupancy
  intercept and -0.13 on the detection intercept, so a coupled cubic term
  (gcol33/tulpa#301) checked against it has something to be wrong about. The
  joint kernel's current behaviour on it is pinned alongside: every probed index
  returns NaN with the reason `"coupled_arm"`, never a silently-wrong 0.

* New tests: `tests/testthat/test-cell-coupling-occupancy-mixture.R` (the
  per-cell contract at tier 1 -- value against the closed form, gradient against
  a difference of the value, the full cross-arm Hessian against a difference of
  the gradient, the coupled/factorising branch split, the declared dense pairs,
  the grad-only path; then at tier 2 an end-to-end joint fit landing on the exact
  mode of the posterior it claims to solve, a spatial ICAR fit whose cross-arm
  curvature is measured nonzero at its own fitted mode, and dense-versus-sparse
  agreement) and four blocks in `tests/testthat/test-inner-skew.R`. Shared
  scaffolding is in `tests/testthat/helper-coupled-fixture.R`.

## 0.0.136

* **The joint nested-Laplace grid no longer returns numbers that depend on what
  else the machine was doing.** Two identical fits could disagree in their last
  bits, because two inputs to the coupled-cell scatter's partition were read
  from live machine state rather than from the problem:

  - The scatter splits its per-cell loop into `C` chunks and reduces them in a
    fixed chunk order, which makes the reduce independent of *which* thread ran
    each chunk. `C` itself, though, came from `team / act`, where `act` was a
    count of the solves in flight at that instant (an atomic `fetch_add`). The
    chunk count sets the chunk boundaries, the boundaries set the summation
    order, and floating-point addition is not associative -- so `C` moving with
    the machine's load moved the answer. It is now read from the cell index:
    `n_grid - k_grid` bounds how many peers a cell can have and estimates the
    same tail width from grid geometry alone. `n_outer` also replaces
    `omp_get_num_threads()`, so an OMP dynamic team adjustment cannot move it
    either.

  - The outer width `n_outer` was clamped against a live `available_ram_bytes()`
    reading, so the same model fitted twice in one session could resolve
    different widths (and therefore different partitions) depending on what the
    box had allocated in between. Both memory readings are now taken once per
    session. The model-dependent term is still computed per call, so a larger
    model is still clamped harder; only the machine-state term is frozen.

  No parallelism is given up for this. The chunks are dispatched as OpenMP
  tasks, so however many threads are genuinely idle still drain them -- only the
  partition is pinned, never the number of workers executing it. In the bulk of
  the grid the budget is 1 exactly as before, so those cells stay serial and
  allocate no partial buffers.

* **`tulpa_nested_laplace_joint()` reports `n_outer`**, the outer width the
  solve actually ran at after the memory clamp. When two fits of one model
  report different widths, that is the explanation for a shift in their last
  bits.

## 0.0.135

* **A prior block missing its required fields now errors instead of segfaulting
  the session** (gcol33/tulpa#299). Each `.NL_REGISTRY` entry declares, per
  dispatch path, the fields its converter indexes; the shared
  `.nl_check_block_fields()` checks them at the four boundaries that feed the
  kernels (`.nl_dispatch()`, `.nl_block_axis_grid()`,
  `.nl_block_spec_for_cpp()`, `.joint_block_spec_for_cpp()`, plus the
  single-block joint packer). A block naming a field wrongly -- a typo, a stale
  name after a rename, a block copied from a different family -- used to reach
  the C++ side as a zero-length vector, which the kernels index with no bounds
  check; it now raises
  `prior block 'icar' is missing required field(s): spatial_idx, adj_row_ptr,
  adj_col_idx, n_neighbors.` A field present but empty counts as missing, since
  that is the same out-of-bounds read. The per-branch presence checks that had
  accumulated in the joint converter are replaced by the shared one, so the
  declaration is the single source of truth; `test-nl-required-fields.R` walks
  every registry entry on every declared path dropping one field at a time, and
  lints both converters' sources so a field read unconditionally by a branch but
  left undeclared fails the suite.

## 0.0.134

* **`fit_st_nested()`'s auto-recenter no longer switches itself off when a grid
  knob is set to the engine's own default value** (gcol33/tulpa#294). The
  spatiotemporal rescue guarded on the PRESENCE of any of `tau_lower`,
  `tau_upper`, `n_grid_spatial`, `n_grid_temporal`, `n_grid_rho`, `rho_lower`,
  `rho_upper` in `control`, so `control = list(n_grid_spatial = 4L)` -- 4L being
  the default -- returned at the first guard and left a railed grid railed. That
  is gcol33/tulpa#293 one level down: a wrapper package exposing its own
  `n_grid` argument, defaulted to the engine's value, threads it through on
  every fit and disabled the rescue for all of them. A knob is now a PIN only
  when its value differs from `.nl_st_default()` and it carries no
  [auto_grid()] mark, the same provenance question the three grid-vector
  rescues ask.

  Pinning is also PER AXIS rather than all-or-nothing: `tau_lower` / `tau_upper`
  hold the two precision axes (they build both), `n_grid_spatial` /
  `n_grid_temporal` one each, and `n_grid_rho` / `rho_lower` / `rho_upper` the
  `ar1` autocorrelation axis. A pinned axis keeps exactly the nodes its knobs
  built and is named in the new `outer_grid_pinned_axes`; the rest are recentred
  as usual. Only when EVERY axis is pinned does the rescue decline outright.

* **`auto_grid()` now marks a scalar knob or a prior specification, not just a
  grid vector.** One front door for "this value is my default, not the user's
  choice", across the three shapes that question arises in.

* **The auto-recenter's second-attempt PC prior is no longer suppressed by a
  `prior_sigma` the caller merely supplied** (gcol33/tulpa#297). The escalation
  that exists for a runaway, near-separation mode engaged only when
  `prior_sigma` was `NULL`, so a wrapper stamping a `prior_sigma` of its own
  turned attempt 2 into a second geometry recenter while the fit still reported
  `outer_grid_recenter_attempts = 2` as if the full escalation had run. The
  suppression is now decided by provenance -- an `auto_grid()`-marked spec, or
  one equal by value to the engine's own `PC(U = 3, alpha = 0.01)`, is a default
  and does not hold the prior back -- and when a genuine pin does suppress it
  the fit carries `outer_grid_prior_declined = "prior_pinned"`.

* **The outer Pareto-k now says WHY it declined** (gcol33/tulpa#295). Roughly
  two dozen distinct decline paths all arrived as the single value
  `pareto_k = NA`, and the print method admitted as much ("outer diagnostic not
  run or proposal degenerate"). "You turned it off", "this family's support can
  never be scored", "the outer Hessian came back non-finite" and "the weights
  carry no mass" are not interchangeable, and a batch reading `pareto_k` across
  many fits could not tell a permanent structural limitation from a live signal
  about the fit. Every decline now carries a reason from a closed vocabulary in
  `fit$pareto_k_declined`: `"not_requested"`, `"not_applicable"`,
  `"unguessable_axis"` (naming the axis, e.g. car_proper's `rho_car` -- read the
  quadrature ESS instead, permanently), `"draws_too_few"`, `"grid_too_small"`,
  `"no_varying_axis"`, `"degenerate_proposal"`, and
  `"internal_inconsistency"` (an engine bug, which `diagnostic_summary()` now
  WARNs on). Wired through the joint single- and multi-block paths, the
  registry path, the SPDE grid and CCD paths, and the shared IS cores;
  surfaced by `diagnostics()`, `print.laplace_diagnostics()` and
  `diagnostic_summary()`.

* **The inner-Laplace `gamma_3` diagnostic now says why it declined too**
  (gcol33/tulpa#296). `gamma_3` never returns a silently-wrong `0`
  (gcol33/tulpa#272), but its `NaN` carried no reason, so a structurally
  unscorable model -- a coupled multi-process likelihood such as tulpaObs's
  `occu_cover`, which this formula can never score -- printed as
  `control$diagnose_skew = FALSE`, attributing an impossibility to a knob the
  user had most likely left at its default `TRUE`. The reason now travels from
  the point of decline: `build_spec_curvature3_fn()` reports
  `"coupled_likelihood"` / `"curvature3_unavailable"` through an out-parameter
  rather than a second predicate that could drift from it, the per-arm oracles
  travel as a `JointCurvature3Oracles` carrying `"coupled_arm"` and which arms
  it applies to, and the R side adds `"not_requested"`, `"no_probe_indices"`,
  `"backend_unsupported"` and `"solve_failed"`. Reported on the fit as
  `inner_skew_declined` / `inner_skew_arms_declined` -- the latter also on a
  PARTIALLY scored joint fit -- and read back by the combined verdict, which
  now distinguishes a layer that was not assessed from one that is unscorable
  by construction (for those models the outer k-hat is the only reliability
  number available, permanently).

* Fixed: `.nl_inner_skew_at_theta()` guarded its probe with `return(NULL)`
  written inside a `tryCatch()` expression (gcol33/tulpa#298). That returns from
  the ENCLOSING function, so a fit hitting any of those guards had the whole
  `res` replaced by `NULL` by a diagnostic that was only meant to decline. The
  probe is now its own function.

## 0.0.133

* **Every engine default now lives in one file (`R/settings.R`).** A default
  outer hyperparameter axis used to be written where it was consumed, so the
  same numbers appeared in several places at once: the field-SD axis
  `exp(seq(log(0.1), log(3), length.out = 5))` in five (the three single-block
  joint areal backends, the multi-block copy-block axis builder, and the
  bym2 / iid registry entries), the copy-coefficient axis
  `c(0, exp(seq(log(0.1), log(3), ...)))` verbatim in two, the wide
  intrinsic-precision axis in three, `k_samples = 200L` in five, and the
  reported Pareto-k usable threshold `0.7` in seven. `.NL_GRID` now holds one
  entry per DISTINCT default axis (keyed by what the axis measures --
  `field_sd`, `gmrf_tau`, `gp_lengthscale`, `copy_alpha`, ... -- so families
  that integrate the same quantity share the entry and move together), with
  `.NL_RECENTER`, `.NL_ST_GRID` and `.NL_DIAG` alongside it for the
  auto-recenter policy, the spatiotemporal driver's own grid, and the
  diagnostic thresholds. Every call site reads an accessor
  (`.nl_grid_axis()`, `.nl_grid_par()`, `.nl_recenter()`, `.nl_st_default()`,
  `.nl_diag()`); no number is restated anywhere else. All 20 default axes and
  every registry `defaults()` closure are byte-identical to what they produced
  before, verified node by node.

* **`.NL_FAMILY_AXES` binds family + path -> axis, so a new family cannot be
  half-registered.** The registry's per-family `defaults()` closures and the
  auto-recenter's axis-provenance check (#293) now read the SAME binding, and
  a plain Cartesian default is one line
  (`.nl_fill_family_axes(p, "bym2")`) instead of a hand-written
  `is.null()` / `expand.grid()` block per family. This closes a gap the #293
  fix left: provenance carried a hand-maintained list of two fields
  (`sigma_grid`, `tau_grid`), so an engine default on any OTHER axis
  (`gp_var`, `phi_gp_grid`, `ar1_rho`, `mo_lengthscale`, ...) coming back in
  through a wrapper's prior was still read as a user pin. Every defaulted axis
  is now covered. `type` narrows the comparison to the axis that one
  path-and-family lays -- passed explicitly, never inferred from
  `block$type`, since a joint areal block carries `type = "icar"` while its
  `sigma_grid` default comes from the joint path and the icar REGISTRY entry
  defaults a precision axis instead.

* Two source-level tests keep the defaults from re-scattering: a geometric axis
  over literal bounds outside `R/settings.R` fails
  `test-settings.R`, as does a restated Pareto-k threshold or `k_samples`
  default. `test-settings.R` also pins every axis to its exact nodes -- these
  numbers are the engine's behaviour on any fit that does not name its own
  grid, so changing one is deliberate enough to update a test.

* `.default_tau_grid()` and `.nl_default_sigma_axis()` are gone; call
  `.nl_grid_axis("gmrf_tau")` / `.nl_grid_axis("field_sd")`. Pre-release, so no
  shim.

## 0.0.132

* **The #289/#290/#291 auto-recenter now fires for wrapper-package fits: axis
  provenance replaces field presence (#293).** The rescue's guard was
  `!is.null(prior$sigma_grid)` -- "the caller named a grid, so it is an
  override". A wrapper package that computes the engine's own default axis
  itself (tulpaObs's `occu_cover()` does, because it also derives the copy arm's
  amplitude axis from that vector and hands the same axis to several blocks)
  writes a non-NULL `sigma_grid` on a fit where the USER named nothing, so every
  such fit looked pinned and the recenter never ran. `SIGMA_GRID = "auto"` was
  inert for exactly the fits it exists to rescue.

  Provenance is now explicit, in one predicate (`.nl_axis_is_pinned()`,
  `R/nested_laplace_auto_grid.R`) that all four rescues share -- the joint
  single-block, the joint multi-block copy block, the standalone registry
  (`icar` `tau_grid` / `bym2` `sigma_grid`) and `fit_st_nested()`. An axis
  counts as a DEFAULT (recentre-able) when it is absent, when it is marked with
  the new `auto_grid()`, or when its node set is exactly the engine's own
  default axis for that field -- a grid identical to the default carries no
  information a pin would add. Anything else is a pin and is never moved.

  - `auto_grid(x)` / `is_auto_grid(x)` (exported) let a wrapper package declare
    an axis it defaulted rather than one the user chose. The marker is an
    attribute, recorded and stripped once at the front door
    (`.nl_grid_provenance()`), so no downstream consumer ever sees an attributed
    numeric.
  - `control$auto_recenter = FALSE` (new, on `tulpa_nested_laplace()`,
    `tulpa_nested_laplace_joint()` and `fit_st_nested()`) is the opt-out that
    integrates any grid exactly as given, the engine's default axis included.
  - `res$outer_grid_recenter_declined` reports why a `"fixed"` placement stayed
    fixed: `"grid_not_collapsed"`, `"axis_pinned"`, `"no_usable_curvature"`,
    `"auto_recenter_disabled"`, `"grid_knobs_overridden"`, `"refit_failed"`. An
    inert rescue was previously indistinguishable from one that was never
    needed, which is how #293 went unnoticed through #289 -> #292.
  - Axis-name resolution is shared (`.nl_axis_alias()`): the same axis is
    spelled `sigma` in a single-block grid, `b<k>.sigma` in a multi-block one,
    and `theta` when a single-axis vector grid is coerced for the regime
    diagnostic. The multi-block rescue matched only the prefixed spelling and
    the registry rescue carried a hard-coded `value_axis_name` for the coerced
    one; both now go through the alias set, plus the "a lone log-tagged axis IS
    the family's scale axis" fallback.
  - The default field-SD axis is single-sourced as `.nl_default_sigma_axis()`
    (it was copy-pasted across the three single-block backends, the multi-block
    copy-axis builder and the bym2 registry default).

  Contract change: handing the engine's own default axis in explicitly used to
  hold the grid fixed; it now recentres like the defaulted axis it is
  (`test-nested-laplace-registry-auto-grid.R` updated, and that fit's k-hat now
  agrees with the defaulted-grid fit's on a collapsed grid). Use
  `control$auto_recenter = FALSE` to hold a grid where it is.

## 0.0.131

* **The 0.0.130 auto-recenter now also engages under the default
  `diagnose_k = FALSE`, and covers `fit_st_nested()`'s spatiotemporal grid
  (#291, #292).** Two gaps left open by #289/#290:

  - The joint driver's rescue only fired when `control$diagnose_k = TRUE`,
    because it read `pareto_k_mode_u`/`cov_u` -- fields only the full outer
    Pareto-k diagnostic populated. Production batch runs default
    `diagnose_k = FALSE`, so a collapsed fit stayed railed regardless of the
    auto sigma grid. `.joint_attach_pareto_k_placement()` now computes the
    same (mode, covariance) via the same `.joint_pareto_prepare()` the full
    diagnostic scores its proposal from, and runs whenever the grid has
    collapsed onto an edge, independent of `diagnose_k`.
  - `fit_st_nested()`'s `tau_spatial x tau_temporal [x rho]` tensor grid was
    the one nested-Laplace family #289/#290 left out, since it had no
    mode-find machinery to reuse. `.st_auto_grid_rescue()`
    (`R/fit_st_nested_auto_grid.R`) adds one: a box-constrained L-BFGS-B
    mode-find (finite-difference gradient, no analytic one available from the
    compiled kernel) over the unconstrained per-axis coordinate -- log for
    the two precision axes, `qlogis((rho+1)/2)` for ar1's autocorrelation --
    seeded at the collapsed grid's own highest-weight cell, then a refit on a
    grid recentered at the mode. Same trigger as every other family
    (`pareto_k_regime == "collapsed_edge"`), one attempt only, bounded 6 nats
    past the default axis on each side, and declines whenever a
    grid-construction knob (`tau_lower`/`tau_upper`/`n_grid_*`/`rho_lower`/
    `rho_upper`) was set explicitly.

## 0.0.130

* **Outer hyperparameter grids auto-recenter on a collapsed boundary instead
  of railing silently (#289, #290).** Every nested-Laplace family built its
  outer grid from a fixed default axis in original coordinates (e.g.
  bym2/icar/car_proper's `sigma_grid = exp(seq(log(0.1), log(3),
  length.out = 5))`). A fit whose field-SD posterior mode sat above the top
  node collapsed every outer weight onto that boundary node
  (`pareto_k_regime = "collapsed_edge"`), silently -- on Michael Glaser's 78
  real EVA `occu_cover` fits, 10 railed the 5.0 sigma ceiling.

  The fixed grid is now a starting axis, not a ceiling: a fit that collapses
  onto a boundary re-centers via the mode-Hessian its own outer Pareto-k
  diagnostic already computes (`R/nested_laplace_auto_grid.R`) and refits,
  reusing that curvature rather than running a second optimizer. An explicit
  `sigma_grid` / `tau_grid` always wins -- auto-recenter only engages when
  left at its default. Wired through:
  - the joint driver's single-block backends (bym2/icar/car_proper) and
    multi-block copy blocks, with a second attempt composing a light default
    PC(U=3, alpha=0.01) prior on sigma for a genuinely unidentified
    (near-separation) mode that geometry alone cannot settle;
  - the standalone `.NL_REGISTRY` path (icar's `tau_grid`, bym2's
    `sigma_grid`), one recenter attempt, reusing the joint path's generic
    axis-tagging and FD-Hessian machinery;
  - `fit_spde()`'s explicit `method = "grid"` path, which now attaches the
    same `pareto_k_regime` diagnostic (visibility only -- `fit_spde()`'s
    default `control$method` is already `"ccd"`, the mode-Hessian path, so
    `"grid"` is a deliberate opt-in the fix respects rather than overrides).

  Byte-stable when the mode already sits inside the old default axis (a
  no-op branch, exercised by regression tests in
  `test-nested-laplace-joint-auto-grid.R` and
  `test-nested-laplace-registry-auto-grid.R`). car_proper (its `rho` axis is
  unguessable, same limitation the existing outer-k-hat diagnostic already
  has) and MCAR (log-Cholesky axis geometry, a materially different
  recentering problem) are out of scope; `fit_st_nested()`'s
  spatiotemporal grid got the diagnostic only, since it has no existing
  mode-find machinery to reuse -- tracked as #291.

## 0.0.129

* **`temporal_gp()` now reaches a fitter (#287).** The constructor was
  exported, documented, and carried a `tulpa()` worked example, but `tulpa()`
  rejected `type = "gp"` by name, `validate_temporal_gp()` had no caller, and
  nothing in `R/` or `src/` populated `TemporalGPData` or set
  `TemporalType::GP`. The C++ was not the gap -- `tulpa_priors_temporal.h` has
  carried a complete templated temporal-GP prior, in both parameterizations,
  the whole time. What was missing was the marshalling.

  `tulpa(y ~ x, temporal = temporal_gp("t"), mode = "hmc")` now fits: the spec
  is validated at the front door, `build_sampler_model_inputs()` accepts
  `type = "gp"` and fills `TemporalGPData` from the unique time instants, and
  the field's two hyperparameters are sampled jointly with it. They are named
  too -- `log_sigma2_temporal_gp` / `logit_phi_temporal_gp` rather than
  `param[3]` / `param[4]`. The field is sampler-path only (there is no
  nested-Laplace kernel laying a grid over a dense T x T Gaussian), and it
  cannot yet share a fit with a spatial or `latent()` block; both now say so.

* **`temporal_gp(cov =)` selects a kernel (#288).** `cov`, `nu` and `period`
  were `match.arg`-validated, documented with their closed forms, carried on
  the spec object -- and read by nothing. The live density hardcoded
  `exp(-dt/phi)`, so `cov = "gaussian"` and `cov = "periodic"` silently fit an
  exponential field: a misspecified prior with no error or warning.

  New `src/temporal_gp_kernel.h` holds the covariance templated over the scalar
  type, so the sampled `(sigma2, phi)` can be autodiff variables -- which is
  why the old plain-double kernels could never have been wired here. The
  exponential kernel (equivalently Matern `nu = 0.5`) is an Ornstein-Uhlenbeck
  process, so it keeps the exact O(T) Markov recursion and its numbers are
  unchanged; Matern 3/2 and 5/2, Gaussian and periodic have no
  finite-dimensional state-space form and are evaluated from a dense T x T
  Cholesky, in both the centered and non-centered parameterizations.

  Matern is offered at `nu` in {0.5, 1.5, 2.5} only -- the smoothnesses with a
  closed form -- and anything between them is now rejected at construction
  rather than quietly run as exponential. `test-temporal-gp-frontdoor.R`
  asserts the five kernel configurations DISAGREE (a test asserting they agree
  would have passed before this), that Matern `nu = 0.5` reproduces the
  exponential fit to the bit, and that the periodic kernel tracks its period.

## 0.0.128

* **The nested-Laplace entry points no longer each carry their own fingerprint
  and skew boilerplate (#286).** `cpp_nested_laplace_*` was already well
  factored on the part that matters -- every entry builds its latent blocks and
  hands them to a shared kernel -- but the plumbing wrapped around that call was
  copied per model.

  The structural fingerprint is the one that punished a mistake quietly: it
  keys the grid checkpoint, so a copied block that folds the wrong structure
  produces a checkpoint that MATCHES across runs it should not, and a resumed
  run then reuses cells computed under different inputs. New
  `tulpa::NlFieldIdentity` (`nested_laplace_checkpoint.h`) names each structural
  group once -- `areal()`, `nngp()`, `hsgp()`, `temporal()` -- and each entry
  point chains the groups it carries:

  ```cpp
  const std::uint64_t struct_seed =
      tulpa::NlFieldIdentity("st_icar")
          .areal(n_spatial_units, adj_row_ptr, adj_col_idx)
          .temporal(temporal_type, n_times, cyclic, temporal_idx)
          .seed();
  ```

  Fold order is part of the fingerprint, so the optional members (the BYM2
  mixing scale, the standalone temporal field's group count) sit in the slot
  they have always occupied and every seed is unchanged. New
  `test-nl-field-identity.R` checks that bit for bit against the folds the
  entry points used to write by hand, for all eleven field models, and pins
  that each structural input still moves the seed.

  The 1-based-to-0-based `skew_idx` conversion had nineteen copies across
  `src/`; all now call the `tulpa::unwrap_skew_idx` that already existed for it
  in `laplace_spec_fit.h`. The five spatiotemporal entries returned their
  temporal axes through a repeated pair of lines carrying an `ar1`-only
  conditional; that is `nl_attach_temporal_grids` now.

  No behaviour change: the entry points shed 225 lines of plumbing, and the
  fingerprint values and returned lists are identical.

## 0.0.127

* **1015 lines of unreachable C++ removed from `src/` (#284).** Four headers
  and a set of functions nothing called. Each was checked by grepping the whole
  tree for the symbol, then by compiling all 95 translation units after the
  deletion -- nothing referenced any of it, so the change cannot alter
  behaviour.

  Deleted whole: `hmc_latent_grad.h` (a closed component superseded by
  `hmc_latent.h`'s `apply_first_zero`), `hmc_tvc_autodiff.h` and
  `log_post_car_proper_det.h` (never `#include`d anywhere), and
  `hmc_temporal_gp.h`. That last one WAS included, by `hmc_sampler.h`, but
  every symbol in it was unreachable: its namespace `tulpa_temporal_gp` is
  named nowhere but its own opening and closing brace. The live temporal GP is
  `tulpa_priors_temporal.h`, which is templated for autodiff and writes the
  exponential-kernel state-space recursion and the non-centred transform
  inline; the deleted header was a plain-double implementation that could not
  serve the gradient modes the sampler uses. Its five `temporal_cov_*` kernels
  were a second, untested copy of covariance math the canonical templated
  `tulpa_svc::compute_cov` already holds, pinned by `test-cov-kernel.R`.

  Deleted in place: `gp_nngp_gradient_w_analytical` (`hmc_gp_gradients.h`),
  `multiscale_gp_log_lik` (`hmc_gp_log_lik.h`), and `hmc_tvc.h`'s dead
  hyperparameter-prior and gradient block (`log_prior_tau_pc`,
  `log_prior_rho_uniform`, `log_prior_rho_beta`, the finite-difference
  `rw2_gradient`, `parse_tvc_structure`). The live TVC priors are in
  `tulpa_priors_tvc.h` (PC prior on log-tau via `pc_prior.h`, Uniform(-1, 1) on
  rho) and the live gradients are the analytic ones in `hmc_tvc_grad.h`.

  `SelectedInverse::at` was reported as dead and is not: `implicit_diff.h:196`
  calls it as `H_inv.at(...)`, which a search for the qualified name misses. It
  stays.

## 0.0.126

* **The small-dense Cholesky core takes its storage layout as a required
  argument (#285).** `linalg_fast.h` shipped two triangular-solve pairs on
  opposite conventions with names that said neither: `chol_forward_solve` /
  `chol_back_solve` indexed row-major with an explicit leading dimension,
  `tri_solve_lower` / `tri_solve_upper_transpose` indexed column-major with `n`
  as the stride. The two are related by transposition, so a factor handed to
  the wrong pair does not crash, does not produce NaN and trips no dimension
  check -- it solves against the transpose and returns a plausible vector.
  #283 was that mistake on a cuSOLVER factor, and it corrupted every NNGP fit
  with 51 or more locations while staying finite and ordinary-looking.

  There is now one implementation of each solve, templated on
  `tulpa_linalg::TriLayout`, and the layout is a **required** template argument
  rather than something a call site inherits from argument order:
  `tri_solve_lower<TriLayout::RowMajor>(L, n, ld, b, y)` and
  `tri_solve_lower_transpose<...>`. `chol_factor_lower` and
  `nngp_moments_from_chol` -- the producer and the consumer that must agree
  with the solve -- carry the same parameter, so a call site states the whole
  convention it is asserting. `chol_log_det` does not: the diagonal sits at
  `i * ld + i` under both. The layout folds at compile time, so the emitted
  arithmetic and its summation order are unchanged.

  `tri_solve_lower`'s old column-major body had no callers and is gone; the one
  caller of `tri_solve_upper_transpose` (the dense mass matrix's momentum draw,
  which reads an Eigen factor and so really is column-major) now says so. New
  `test-tri-solve-layout.R` states the contract -- each routine reads the lower
  triangle of the matrix its declared layout spells out of the buffer -- and
  checks it on matched and mismatched buffers, including the cuSOLVER-shaped
  one whose opposite triangle still holds the input.

## 0.0.125

* **The batched CUDA Cholesky returned a column-major factor that every
  consumer read row-major (#283).** `batch_nngp_scatter` hands the NNGP
  neighbour-covariance factorizations to `cuda_batched_cholesky` once the batch
  reaches 50 locations. cuSOLVER writes `L[i][j]` at offset `j*k + i` and leaves
  the opposite triangle holding the input; `chol_forward_solve` /
  `chol_back_solve` index `i*k + j`. The accepted "factor" was therefore the
  input covariances with a Cholesky diagonal -- finite, ordinary-looking, and
  wrong. **Every NNGP fit with 51 or more spatial locations on a machine with
  usable CUDA was affected**; below 50 the CPU path ran and was always correct.
  On a 150-point unit-square fixture (exponential covariance, `sigma2 = 0.9`,
  `phi_gp = 0.4`) the conditional variances were off by up to 0.28 in absolute
  terms and 47 of 150 nodes were pushed onto the 1e-10 variance floor, against
  a true minimum of 2.5e-02. After the fix both sides of the dispatch threshold
  agree with an independently computed reference to 3e-16 and nothing floors.
* **The GPU factor is now verified before it is accepted (#283).** The batched
  call also ignored cuSOLVER's per-matrix `info` codes, so a non-positive-
  definite neighbour set came back as a partially written factor that looked
  valid. `cuda_batched_cholesky` now reads them, and `batch_nngp_scatter`
  checks the returned factor for one batch element against the CPU
  factorization before using the batch, falling back to the CPU for all of it
  on mismatch -- one extra `m^3` factorization against 50+ matrices. The
  fallback now also restores the original covariances, which a partial GPU
  write had been corrupting.
* `test-nngp-prior-scatter.R` straddles the 50-location dispatch threshold and
  compares conditional variances against a from-scratch reference, so the CPU
  and GPU paths are held to the same answer. A fixture that stays under 50
  locations exercises only the path that was already right.

## 0.0.124

* **The NNGP prior scatter is checked against the matrix it claims to build
  (#278).** `apply_nngp_full_prior_dense` and `apply_nngp_full_prior_sparse`
  were documented as the same math in different containers, and a measured
  `log|H|` gap of 2.9e-03 between them put that in doubt. Assembling
  `Lambda = (I - A)' D^-1 (I - A)` independently from the `(alpha, cv)` the
  batched scatter returns settles it: both reproduce it to ~1e-16 RELATIVE at
  every neighbour-set size from 2 to 10. The gap was an ABSOLUTE difference on
  matrix entries of magnitude 1e13. The dense twin is deleted -- it had no
  callers once `blocks_require_sparse()` pinned NNGP to the sparse Newton path
  -- and the survivor is held to the definition by the new
  `test-nngp-prior-scatter.R`, which also pins the gradient to `-Lambda w`.
* **Where the 1e13 comes from was tracked to #283.** `nngp_moments_from_chol`
  floors the conditional variance at 1e-10, and 47 of 150 nodes hit that floor
  on the reference fixture at `nn = 8`, putting `1e10` on `Lambda`'s diagonal
  and taking `cond(Lambda)` to numerically infinite. That conditioning, not a
  defective Hessian, is what stalls a Newton solve at large `nn`. The floor
  turned out to be a symptom rather than the cause -- see 0.0.125, where the
  broken CUDA factor behind it is fixed and nothing floors on that fixture.

## 0.0.123

* **The last standalone Newton loop is gone (#282).** #277 left
  `cpp_laplace_fit_spde_precomputed` -- the fixed-hyperparameter fit behind the
  fractional/rational SPDE path -- on its own solver, and with it
  `spde_run_single_fit` and `laplace_newton_solve_sparse`. It is now a one-cell
  run of the same joint multi-block driver every other SPDE/GP entry takes:
  `make_spde_block_precomputed` seeds the block's `SpdeQBuilder` from the CSC
  `.spde_rational_assemble()` hands it, makes `prep()` the `0.5 log|Q|`
  normalizer only (the precision does not move with the cell), and leaves the
  latent uncentred -- the auxiliary weights `x` are not the field `u = Pr x`,
  and the proper SPDE prior already identifies the constant mode. Everything
  else -- `obs_indices`, the H pattern, both prior scatters, `log_prior` -- is
  shared with the FEM entries through the new `spde_assemble_block`. The path
  now inherits `gamma_3`, grid-cell checkpointing and the per-cell `score_max` /
  `converged` reporting instead of needing a wiring pass each time.
  `laplace_newton_solve_sparse` and `spde_run_single_fit` are deleted.
* **The precomputed SPDE log-marginal carries the RE prior normalizer (#282).**
  The bespoke loop's log-prior omitted `0.5 G (log tau_re - log 2 pi)`, so a fit
  with an iid RE block reported a marginal that was not comparable across
  `sigma_re` (a 3.4-nat error at `G = 6`, `sigma_re = 0.7`). The shared driver
  supplies it. The mode, the fitted linear predictor and the field are unchanged
  -- measured across nine rational-SPDE fixtures, `eta` agrees to 3e-07 and the
  penalized objective to 1e-08, with two fixtures reproducing bit-for-bit; the
  residual movement is confined to the auxiliary-weight directions the rational
  precision leaves unidentified.

## 0.0.122

* **The SPDE FEM assembly builds the operator order it was asked for (#280).**
  `SpdeQBuilder::rebuild()` branched `if (alpha == 1) ... else <alpha 2>`, so
  every `alpha >= 3` was assembled as `alpha = 2` with no error -- a user
  asking for a smoother Matern field silently got the `nu = 1` operator.
  `init()` now takes the operator order and builds the chain `M_0 = C`,
  `M_1 = G`, `M_j = G (C^-1 G)^(j-1)`; `rebuild()` is the binomial expansion of
  `Q = tau^2 K (C^-1 K)^(alpha-1)` over it, so one loop covers every integer
  alpha and the sparsity pattern widens with the order instead of every `nu`
  reusing the `alpha = 2` stencil. The analytic marginal-SE mirror
  `.spde_precision_Q()` carries the same expansion. `alpha = 2` is reproduced
  term for term, so `nu = 1` results are unchanged.

* **`(range, sigma) -> (kappa, tau)` carries `nu` in both coordinates (#279).**
  The C++ conversion had `nu` entering `kappa` but not `tau`, which is the
  `nu = 1` special case; the general d = 2 relation is
  `tau = 1 / (sqrt(4 pi nu) kappa^nu sigma)` (Lindgren, Rue & Lindstrom 2011),
  as the R side already used. At `nu = 2` the old `tau` was ~14x too large, so
  a requested `sigma` mapped to a different marginal SD, and the single fit and
  the nested integrator disagreed with each other. The conversion is now one
  function, `spde_range_sigma_to_kappa_tau()` in `src/spde_qbuilder.h`, shared
  by the block factory and the implicit-differentiation entry.

* **`spatial_spde(nu = 0)` is refused at construction (#281).** The Matern
  parameterisation is degenerate there -- `kappa` is 0 and `tau` infinite -- so
  a fit could only report an infeasible cell (`log_marginal = -Inf`, no Newton
  iterations) with nothing saying why. `.validate_spde_nu()` now requires
  `nu > 0` and names both broken quantities.

* Joint-hyper NUTS refuses an integer `nu != 1`. Its non-centered transform
  differentiates the `alpha = 2` assembly, so higher orders would have been
  sampled as `nu = 1`; the fixed-hyper sampler and the nested-Laplace path both
  assemble any integer alpha. New `test-spde-nu-general.R` checks the compiled
  assembly against an independently built `K (C^-1 K)^(alpha-1)` at
  `alpha = 1..4`, the conversion against the closed-form marginal variance, and
  a `nu = 2` field end to end.

## 0.0.121

* **`cpp_laplace_fit_gp` and `cpp_laplace_fit_spde` are now one-cell runs of
  the shared machinery (#277).** Both were fixed-hyperparameter spatial kernels
  carrying their own Newton loop, so a feature added to the joint-multi driver
  had to be wired into them separately -- which is what the `gamma_3` pass
  (#273) ran into. Each is now a thin wrapper: the same `make_single_arm`, the
  same `make_nngp_block` / `make_spde_block`, the same driver at one grid cell,
  reading the result back through `nl_grid_cell_to_result_list()`. The
  equivalence is exact, asserted at `tolerance = 0` in
  `test-laplace-spatial-gp-spde-equiv.R`, and `gamma_3` is now inherited rather
  than wired.

* **The SPDE single fit no longer reports an off-mode mode.** The mesh field is
  sum-to-zero centred after the Newton loop. The bespoke path centred it
  without moving the removed constant into the intercept, which shifts `eta`
  away from the mode the loop found, and then re-evaluated `log_marginal` and
  the Hessian there: a converged fit reported a fixed-effect score of ~0.47.
  The driver folds the constant into the arm intercept, so `eta` is preserved.
  Fixed effects and `log_marginal` move for every fixed-hyperparameter SPDE fit;
  the mesh field is unchanged.

* **NNGP is pinned to the sparse Newton path.** Its prior scatters only into
  the sparse builder, and the dense route disagreed with it -- measurably at
  `nn = 5`, and at `nn = 8` a 300-iteration non-convergence returning
  `log_marginal = NaN` against a 23-iteration convergence. `blocks_require_sparse()`
  (`latent_block.h`) now reads that requirement off the blocks: a block whose
  prior has only `add_prior_sparse` forces the sparse path, instead of each
  caller remembering `force_sparse`. This closes the silent case where the dense
  path called an absent `add_prior` and contributed nothing at all. NNGP is the
  only block whose dispatch changes; MCAR, HSGP-MO and the latent factor are
  already non-`INDEXED_SINGLE` and were forced sparse before.

* **Nested fits report per-cell solve health.** `log_det_Q`, `score_max` and
  `converged` join `log_marginal` / `n_iter` / `modes` on every grid-driver
  result, so a grid fit can see which cells settled instead of only that the
  run finished.

* `cpp_nested_laplace_nngp()` accepts an `offset`, which the generic driver
  already read off `ParsedArm`; `make_spde_block()` takes its axes as
  `(kappa, tau)` directly via `direct_kappa_tau`, so a fit handed the operator
  parameters does not round-trip them through the Matern conversion.

* `fit_spde()` with `nu = 0` now reports an infeasible cell (`log_marginal =
  -Inf`, no iterations) rather than iterating on a degenerate precision. The
  Matern parameterisation has no `(kappa, tau)` at `nu = 0`
  (`kappa = sqrt(8 nu) / range` is 0 and `tau` is infinite); this is the verdict
  `cpp_nested_laplace_spde()` has always returned there.

## 0.0.120

* **Outer `pareto_k` no longer over-flags collapsed-grid fits (#276).** On a
  sharp hyperparameter posterior the outer grid collapses onto ~1 cell, the
  existing grid-mixture rescue cannot engage (its few bumps cover worse than
  the Gaussian), and the k-hat is left scored against a SYMMETRIC Gaussian
  proposal on a right-skewed variance-component marginal. A symmetric proposal
  against a skewed target has a heavy importance-ratio tail whatever the
  integration's quality, so the k tracked the grid collapse and the marginal's
  asymmetry rather than the fit. Surfaced by a collaborator's 78-species
  `occu_cover` run in which 42/78 species binned as "problematic/unreliable" on
  a bare k threshold while their point estimates were sound.

  Three changes, all on the joint nested-Laplace backend (single- and
  multi-block paths alike):

  - **A skew-normal proposal rescue.** After the Gaussian / grid-mixture
    dispatch, a k-hat still above the good band is re-scored against a product
    of univariate skew-normals in the chosen Gaussian's whitened coordinate,
    matched to the target's PSIS-weighted mean, sd and skewness -- estimated
    from the pass's own draws, so it costs no extra inner solves and is
    automatically located and scaled where the target is. Adopted only if it
    strictly lowers the k-hat. Because a skew-normal has GAUSSIAN tails on both
    sides it can absorb asymmetry but never a heavy tail, so the rescue cannot
    launder a real tail problem: `test-outer-skew-rescue.R` asserts both
    directions, including a skewed HEAVY-tailed target on which the rescue is
    built, scored and rejected.

    Engagement is screened for significance, not just magnitude. A sample
    skewness has standard error `sqrt(6/n)` (~0.17 at 200 draws), so a bare
    magnitude floor fires on noise: measured on a GAUSSIAN outer target, an
    unscreened floor adopted the skew proposal in 18% of RNG states, a two-SE
    screen in 5%, and the shipped three-SE screen in 0%. The screen reads the
    importance weights' effective sample size, not the raw draw count.

  - **A grid-regime classifier.** `pareto_k_regime` reports `"spread"` /
    `"collapsed_interior"` / `"collapsed_edge"`, with
    `pareto_k_grid_edge_axes` / `pareto_k_grid_edge_sides` naming the axes a
    collapsed mode sits against and on which side. Below two effective grid
    cells no axis carries resolved spread, so the outer integration has
    degenerated to a point evaluation and `pareto_k` is scoring a mode-Gaussian
    stand-in for the hyperparameter marginal rather than an integration. An
    interior collapse is benign; a boundary collapse is actionable (the grid may
    be too narrow). Axes the grid pins to a single value are excluded -- pinned,
    not at a boundary. Read off stored weights, so it is attached even with
    `control$diagnose_k = FALSE`.

  - **The context is surfaced**, so a downstream bare-k threshold is not the
    whole story: `diagnostics()` gains `outer_regime`, `grid_edge_axes`,
    `grid_edge_sides` and `outer_skew_max` attributes plus `outer_regime` /
    `outer_skew_max` summary columns, `print.laplace_diagnostics()` prints the
    marginal's skewness and a one-line reading of a collapsed regime, and
    `diagnostic_summary()` raises a WARN with the widen-this-axis
    recommendation on an edge collapse.

  The `pareto_k` band itself is left to speak for the number it reports: the
  fix is the number, not a verdict override. `sn_match()` remains the single
  source of truth for the cumulant inversion; the proposal path adds only a
  vectorized sampler and log-density (`sn_cdf()` / `sn_quantile()` route through
  Owen's T by quadrature per point, which is right for a few reported quantiles
  and unusable for hundreds of proposal draws).

## 0.0.119

* **`gamma_3` wired through the SPDE / GP bespoke Newton pair (#273 item
  3).** `cpp_laplace_fit_gp`, `cpp_laplace_fit_spde` and
  `cpp_laplace_fit_spde_precomputed` are standalone, fixed-hyperparameter
  single fits (`laplace_mode_gp()` / `spde_run_single_fit()`) that route
  through their own Newton implementation rather than the joint-multi driver
  #272/#273 item 1 already wired -- the nested "nngp" / "spde" registry
  entries integrate hyperparameters via the shared joint-multi machinery
  instead and were unaffected. Both the dense branch (`laplace_newton_solve`
  / `run_spde_laplace`) and the fully sparse CHOLMOD-only branch
  (`laplace_newton_solve_sparse`, `n_x >= SPARSE_THRESHOLD`) now accept
  `compute_skew` / `skew_idx`, matching the icar/bym2/car_proper/hsgp
  kernels' existing surface. #273 item 2 (the coupled non-separable
  cubic-term derivation) remains open.

## 0.0.118

* **`gamma_3` wired through the joint multi-block dispatch (#273).** The
  inner-Laplace skewness diagnostic #272 shipped for every single-arm kernel
  and the joint driver's single-block backends, but explicitly left the
  MULTI-block joint path (`nested_laplace_joint_multi.R`, used when a fit
  carries a per-group RE / trend field / arm-specific field block) unwired.
  `.nlj_multi_inner_skew_at_theta()` closes that gap: the multi-block
  counterpart of `.nlj_inner_skew_at_theta()`, re-dispatching the SAME
  `call_kernel` at the fitted MAP grid cell with `compute_skew = TRUE` (the
  C++ kernel already accepted the parameter; only the R-side threading was
  missing). Same defaults as the single-block path: every arm's
  fixed-effects coefficients scored by default, `NA` (not a silently-wrong
  `0`) for a non-separable coupled arm.

* **Combined outer/inner reliability verdict no longer conflates "not
  assessed" with "good" (#274).** `.tulpa_combined_reliability()` collapsed
  an unassessed inner layer (gamma_3 not computable for a given backend or
  likelihood) into the same verdict string as a genuinely good inner layer
  whenever the outer layer was flagged, so a batch consumer reading the
  `reliability` string off many fits couldn't tell "outer bad, inner
  genuinely fine" from "outer bad, inner never checked". Every combination
  naming an unassessed layer now says so explicitly ("... not assessed"),
  symmetric in both layers (an unassessed OUTER layer -- e.g. a multi-block,
  multi-axis grid that declines Pareto-k rather than apply a guessed support
  transform -- gets the same honest treatment).

* **Fixed a crash in the joint multi-block dispatch on a malformed per-arm
  index vector (#275, found while testing the #273 fix).** A block's
  per-arm `spatial_idx` / `temporal_idx` / `obs_idx` entry shorter than that
  arm's actual observation count (in particular, an empty vector for an arm
  the block is meant to "skip") was read out of bounds by the C++ kernel's
  per-arm index closure, crashing the R session instead of raising an error.
  There is no supported "this block excludes arm k" shorthand via a
  short/empty index vector -- every arm needs a matching-length index vector
  for every block; a block's contribution to an arm is excluded via that
  arm's `field_coef = 0` instead. `.multi_block_per_arm_idx()` now validates
  every per-arm entry's length across all 7 call sites (icar/bym2/car_proper,
  mcar, rw1/rw2/ar1, iid, miid, tgmrf, lf) and raises a clear error naming
  the block, the arm, and the expected/actual counts.

## 0.0.117

* **Inner-Laplace skewness diagnostic: score the layer outer Pareto-k-hat
  doesn't cover (#272).** `pareto_k` scores the OUTER hyperparameter-grid
  integration around a fixed inner Laplace; it read as a whole-fit verdict
  even though the inner Gaussian approximation to the latent-field
  conditional posterior is a separate, unscored layer -- an `occu_cover`
  batch flagged 42/78 species "unreliable" on outer k-hat alone when their
  point estimates, governed by the healthy inner layer, were fine. `gamma_3`
  (`src/inner_laplace_skew.h`, the leading-order Edgeworth skewness estimate
  from Rue, Martino & Chopin 2009 Sec 3.2.3's cubic correction, generalized
  from their augmented representation to tulpa's general `eta = compute_eta(x)`
  and to the joint multi-arm case) closes that gap: opt-in
  (`control$diagnose_skew`, default `TRUE`) and computed with one extra
  deterministic Newton solve at the fitted MAP grid cell, scoring every arm's
  fixed-effects coefficients by default (`control$skew_idx` extends it).
  Declines to `NA` -- never a silently-wrong `0` ("perfectly Gaussian") -- for
  a likelihood the formula cannot score (a coupled multi-process spec such as
  zero-inflation or tulpaObs's `occu_cover`, or a family with no registered
  third derivative); this also caught and fixed a real bug in the diagnostic
  as first staged, where an entirely absent oracle silently summed to
  `0 / sigma_i^3 == 0` instead of `NA`. `diagnostics()` /
  `print.laplace_diagnostics()` report a combined whole-fit verdict naming
  which layer degrades, if either does. Wired through every single-arm
  nested-Laplace kernel (icar/bym2/car_proper/temporal/nngp/hsgp/the ST
  variants/SPDE) and the joint driver's single-block backends; validated in
  `tests/testthat/test-inner-skew.R` against a direct numerically-integrated
  exact posterior skewness (a rare-event binomial intercept), not just shape
  checks. Known remaining scope (joint multi-block wiring, a genuinely
  coupled-arm cubic-term derivation, the SPDE/GP bespoke large-`n` Newton
  pair) tracked in #273.

## 0.0.116

* **Stale test fixed, no engine bug (#271).** `test-tulpa-entry-nested.R`'s
  "more than one random-intercept term alongside a block errors" test
  asserted a restriction gcol33/tulpa#265 (0.0.113-era, commit cd80b95)
  deliberately removed: every `(1 | g)` term on the nested-Laplace + `latent()`
  path now becomes its own `iid` block integrated on the outer grid, so N
  random-intercept terms beside a block are no different in kind from one --
  the same change `test-smoother-re-integrated.R` already covers beside a
  smoother. #265's commit updated `R/tulpa.R` but never touched this test, so
  it kept asserting the old `stop()` and started failing with a `NULL`
  condition once the guard it expected was gone. Bisected past the #267
  auto-mode change #271 suspected as the cause: `auto_select_mode()` checks
  `has_latent` before `has_re`, so a model combining RE terms with a
  `latent()` block already routes to `nested_laplace`, not `re_cov_gibbs`,
  regardless of #267. The test now asserts the current, intended behavior
  (routing, `re_block_index`, and bit-exact equivalence to the direct
  multi-block `tulpa_nested_laplace()` call).

## 0.0.115

* **The joint Hessian sparsity pattern now covers a latent block reached by
  only one side of a coupled arm pair (#270).** `HessianPatternGuard`
  (introduced after the tulpaObs v0.0.101 pin, so this had never been
  checked) caught `occu_cover()` dropping 10592-124160 nonzero contributions
  per fit whenever a coupled ICAR field met either a correlated random-slope
  RE block private to the detection arm, or its own detection-arm beta under
  the rank-1 s2z fold path (fields above `TULPA_S2Z_DENSIFY_MAX`, default
  256 units).

  The per-cell cross-Hessian scatter multiplies EVERY active dof of one
  coupled arm's row -- beta, RE, and any latent-block dof that row's `idx` /
  `obs_indices` resolves to -- against every active dof of another (or the
  same) coupled arm's row sharing the cell. `build_joint_hessian_pattern`'s
  cross-arm section only ever registered the beta/RE part of that product; a
  block reached by just one side (a private random-slope block, or a field
  the other arm's `field_coef = 0` decouples) had no pattern entry for its
  cross term against the other arm's beta/RE/latent dofs, even though the
  scatter produces a real nonzero value there whenever the coupling spec's
  cross-Hessian for that arm pair is nonzero. A new section walks each
  coupled cell's per-arm active-dof union (mirroring the scatter's own
  `collect_coupled_row_latents` resolver) and adds the missing cross
  entries, scoped per cell so it costs no more than the scatter already
  does. Two regression tests reproduce both shapes on tulpa's own bivariate
  test-coupling spec (`test-cell-coupling-cross-hess.R`), independent of
  tulpaObs.

## 0.0.114

* **`mode = "auto"` no longer conditions a random-effect term's SD at 1 (#267).**
  `tulpa(y ~ x + (1 | g))` on the default mode reported `sd = 1, source =
  "conditioned"`, while the same term with a slope added (`(1 + x | g)`) had
  its whole covariance inferred -- the richer model was handled better than the
  plainer one. `auto_select_mode()` took no random-effect argument at all, so a
  mixed model fell through to the same Tier-1 default a plain GLM reaches.

  `auto` now routes any random-effect term -- intercept-only or slope -- to the
  exact Metropolis-within-Gibbs covariance debias (`re_cov_gibbs`), which
  already treats a scalar `(1 | g)` as the degenerate `c = 1` covariance block,
  so no special-casing by term shape was needed. An explicit `mode = "laplace"`
  / `"mala"` / ... still conditions on `sigma_re` (defaulting to 1) when the
  caller names it directly -- only the `auto` default changes. The `message()`
  that reported the conditioning is now a `warning()`, so a script that
  promotes warnings sees it (same sub-issue as #265).

* **One hyperprior convention for a random-effect covariance's scale, chosen
  (#268).** The nested-Laplace path integrates every scale axis (`icar`, `rw1`,
  `rw2`, `ar1`'s `tau`, `iid`) flat in `log(theta)`, by construction of the grid
  and its softmax weighting; `tulpa_re_cov_nested()` / `tulpa_re_cov_gibbs()`'s
  `Sigma` estimate carried a PC + LKJ prior by default. The two paths put
  different priors on the same statistical object, undocumented in either
  direction, so the RE SD from one backend was not the RE SD from the other.

  `tulpa_eb()` and `tulpa_re_cov_nested()` now share a `hyperprior` argument,
  `"flat"` (default) or `"pc_lkj"`: `"flat"` matches the nested-Laplace
  convention everywhere else in the engine; `"pc_lkj"` opts into the
  weakly-informative PC + LKJ prior `re_cov_pc_lkj_prior()` builds, still
  available on request. `tulpa_re_cov_gibbs()`'s `Sigma | b` conjugate draw
  cannot go fully flat -- an improper prior is not a valid target for that
  step -- so it keeps its existing minimal-proper Inverse-Wishart default
  (`prior_df = n_coefs + 1`), documented as the closest analogue. See
  `vignette("priors")`.

## 0.0.113

* **A random-effect term on the nested path has its SD integrated instead of
  conditioned at 1 (#265).** `tulpa(y ~ s(x) + (1 | site))` reported
  `sd = 1, source = "conditioned"` -- on a formula whose smoother hyperparameter
  *was* integrated, the random effect was the one variance component the fit never
  estimated, and 1 is the value nobody supplied. There was no argument
  combination that estimated it: `mode = "eb"` was overridden by the smoother
  redirect (#266) and the RE-covariance integrators are reached only when a term
  carries slopes.

  Each `(1 | g)` now becomes an `iid` latent block, so its SD is one more axis of
  the outer grid beside the smoother's `tau`. The block type, its integration and
  its recovery already existed (the #86 coupled field + RE capability); only the
  front door was not using it, and `.tulpa_fitter_args()`'s own comment already
  named this as the intended treatment. Consequences:

  - Several RE terms now work. The path previously refused more than one
    outright ("supports at most one random-intercept term"); each term is its own
    block.
  - `sigma_re` supplied explicitly still conditions, as the one-point `sigma_grid`
    the `iid` registry entry documents -- conditioning is the degenerate case of
    the same path, not a second one -- and `VarCorr()` still labels it
    `conditioned` rather than claiming the data produced it.
  - `VarCorr()` reports the integrated posterior's **median**, not its mean: a
    variance component at few groups is right-skewed, so its mean sits above its
    bulk by construction.
  - `ranef()` reports every group. The RE blocks are appended LAST, which is what
    makes their latent segment addressable as the trailing `sum(n_groups)` columns
    without re-deriving any other block's width -- a second source of truth for
    something the driver already knows. Its exact-tail-width guard cannot fire
    once the RE shares the latent vector with a field block, so without this the
    accessor would have returned the empty table #264 just removed.
  - A random slope beside a smoother still refuses, now naming the reason (an
    `iid` block has no `Z` design) and pointing at the backends that do fit a
    slope covariance.

  Recovery is asserted as **coverage**, not as a point tolerance. The grid
  integrates the SD under a prior flat in `log(sigma)`, the convention every
  nested scale axis uses; that does not shrink, so at G = 15 the posterior is wide
  and its point summary runs above the truth (mean of medians 1.02 against 0.9
  over 6 seeds) while the 95% interval covers the truth 6/6. Asserting
  `|est - truth|` would encode the +0.12 as the target. Whether the engine should
  carry one hyperprior convention across the nested and RE-covariance paths is
  #268, deliberately not bundled here: a PC prior on the `iid` axis alone would
  trade the cross-path inconsistency for one inside a single fit.

## 0.0.112

* **An explicit `mode` is no longer silently overridden by a structural redirect
  (#266).** `tulpa(y ~ s(x) + (1 | site), mode = "eb")` fitted `nested_laplace`
  and reported `selection_reason` as though `mode` had been `"auto"`, so nothing
  on the fit recorded that the requested inference method had been swapped. The
  redirect machinery now carries that:

  - `select_inference_mode()` marks a selection explicit (anything but
    `mode = "auto"`) and keeps the literal request, so a later redirect can name
    what it overrode.
  - `.sel_redirect()` records the override in `sel$overridden` and appends a
    clause to the reason. The clause is re-appended to *whatever* reason the
    selection ends up carrying rather than only the one that recorded it: each
    redirect replaces `sel$reason`, so in a chain (slopes then a smoother) the
    later one used to drop the statement and leave the fit looking as though
    nothing had been overridden. Only the first override is recorded, since the
    request the user actually made is the one worth naming.
  - `tulpa()` warns, and stamps the machine-readable `fit$mode_overridden`
    (`requested` + the backend it would have used). A `warning()` rather than a
    `message()`, so a script that promotes warnings, or a chunk that traps them,
    sees it.

  The warning is opt-in per redirect site (`notify =`), because two different
  things were being conflated. A smoother sending an `eb` / `agq` request to the
  nested kernels takes away the random-effect SD those two would have estimated,
  and warns. A random slope under `mode = "laplace"` has no scalar `sigma_re` to
  condition on, a temporal field has no conditional-Laplace kernel, and an SPDE
  field redirected to the `spde` backend is the same mode and tier reaching its
  own integrator -- those are documented routes for the structure, not
  capabilities taken away, so they are recorded on the fit without warning on
  every fit.

* **Dead `src/hmc_spatiotemporal.h` removed (#261).** Unreachable: its only
  include was `hmc_sampler.h`, no external call site referenced
  `tulpa_spatiotemporal::`, everything in it lived inside that namespace, and it
  was in `src/` rather than `inst/include/tulpa/` so no `LinkingTo` consumer
  could reach it either. It had also diverged from the live path
  (`src/tulpa_priors_st.h`) in two places that were fixed only on the live side:
  it overstated the cyclic RW1/RW2 rank by one, and hardcoded `rank_space = S - 1`
  against the #241 component fix. Its `st_sum_to_zero_penalty` duplicated the
  live one. A reader grepping for the ST rank found the wrong copy first, which
  is the shape that made gcol33/tulpaRatio#12 possible.

## 0.0.111

* **`ranef()` reports the per-group posterior on both RE-covariance backends
  (#264).** It returned a 0-row data frame for a fit from either integrator --
  on exactly the fits whose free covariance over correlated slopes is the point,
  and indistinguishable from a model carrying no random effects at all. Both
  backends do hold the per-group information, and both now report it:

  - `tulpa_re_cov_gibbs()` samples `b` in its Metropolis-within-Gibbs sweep and
    threw the draws away at the end of each sweep. The compiled sweep now
    records them (`fit$re`, one column per (block, group, coefficient),
    row-aligned with the `beta` draws so a row is a joint state), and `ranef()`
    summarizes them as the exact posterior: mean, SD and 2.5%/97.5% quantiles.
    `posterior_predict()` picks the same draws up through `.re_draws_mat()`,
    so its replicates carry the random-effect uncertainty instead of falling
    back to a population-level linear predictor.
  - `tulpa_re_cov_nested()` has a Gaussian per-group posterior at every
    integration node. Each node's conditional mean and marginal variance are
    retained (`fit$re_nodes` / `fit$re_var_nodes`, the inner solve now being
    asked for its covariance blocks), and `ranef()` reports the exact moments
    and quantiles of the weighted mixture of them -- so the interval carries
    both the within-node curvature and the `Sigma` uncertainty. The interval
    inverts the mixture CDF rather than assuming normality around the mean: a
    mixture over a skewed `Sigma` posterior is itself skewed. New
    `.nl_gauss_mixture_summary()` is that summary (verified against Monte Carlo,
    and against the mixture CDF at its own returned quantiles); it is the
    continuous counterpart of the discrete `.nl_wtd_quantile()`.

  The adaptive Gauss-Hermite inner marginal (`control$re_cov = "aghq"`, or
  `n_quad > 1`) integrates each group out by quadrature and so forms no
  per-group posterior at all. It now says that, with the two modes that do
  report one, rather than returning the empty frame: `ranef()` errors on the
  stated `fit$ranef_unavailable` reason. `?ranef` documents what each backend
  reports and why.

## 0.0.110

* **`tglmm()` and `tgam()` are removed; `tulpa()` fits both model classes.**
  The two doors dispatched through `tulpa()` and returned a byte-identical fit,
  so they carried no engine of their own: what they added was a signature
  missing `spatial` / `temporal`, a refusal of the structures outside their
  model class, and a subclass that unlocked a richer `print()`. The first two
  are reach removal on a call that would otherwise fit the model correctly, and
  the third is now driven by the fit instead (below). `R/doors.R`, the
  `.TULPA_DOORS` registry, and the `tulpa_glmm` / `tulpa_gam` subclasses are
  gone; the GLMM and GAM formulas are unchanged, so `tulpa(y ~ x + (1 + x | g))`
  and `tulpa(y ~ s(x))` fit exactly what the doors did. The README now shows
  both.

* **A fit reports the structure it carries, not the function that produced
  it (#262).** The random-effect covariance and the smoother table were printed
  only for a fit that came through a door, though the metadata behind both is
  attached by `tulpa()` itself (`fit$smooth_terms` at `R/tulpa.R`, `VarCorr()`
  on any `tulpa_fit`). `print.tulpa_fit()` now ends in `.print_re_section()` +
  `.print_smooth_section()`, each silent when the fit has no such structure, and
  `plot(fit, type = "smooth")` draws the fitted curves for any fit carrying
  `s(...)` terms. `print.tulpa_nested_laplace()` composes the generic fit body
  after its own hyperparameter report, so a smoother fit reports its fixed
  effects too -- that tier's print method had replaced the generic one rather
  than extending it, so those were missing entirely.

* **`VarCorr()` reported `sd = 1, conditioned` for a fit that integrated Sigma
  (#263).** `.varcorr_from_sigma()` read only `$Sigma`, while
  `tulpa_re_cov_gibbs()` / `tulpa_re_cov_nested()` report the posterior mean
  under `$Sigma_mean`, so resolution fell through to the conditioning fallback
  and invented a `sigma_re` the user never supplied -- on the very fits whose
  free covariance is the point. Both fields are now read, with `[[` rather than
  `$`: a gibbs fit carries `Sigma_draws` alongside, making `$Sigma` an ambiguous
  partial match there while it would silently resolve on a nested fit. A
  conditioning fit is still labelled `conditioned`.

## 0.0.109

* **The intrinsic RW rank is exported, so a linking package can consume it
  instead of carrying its own copy (gcol33/tulpaRatio#12).** `rw1_rank()` and
  `rw2_rank()` move from `src/hmc_temporal.h` into the exported
  `tulpa/sum_to_zero.h`, alongside the augmented rank they feed. They already
  encoded that a cycle-graph Laplacian still annihilates only the constant, so
  a cyclic RW1 has rank `T-1` and a cyclic RW2 `T-1` as well; a consumer
  computing `cyclic ? T : T-1` inline overstates both by one and biases the
  `tau` posterior. `tulpa_temporal::rw1_rank` / `rw2_rank` keep resolving, so
  the engine's own call sites are unchanged.

## 0.0.108

* **An areal field's component partition is now set with its adjacency, so a
  consumer cannot leave the field unidentified (gcol33/tulpaRatio#19).** The
  sum-to-zero augmentation that makes an intrinsic ICAR / BYM2 prior proper
  iterates over the field's connected components, and a default-constructed
  `GraphPartition` describes zero nodes and reports zero of them. A caller that
  assigned the CSR adjacency alone therefore got an augmentation that pinned no
  direction at all, while `n_spatial_components` kept its own default of 1 and
  the rank normalizer went on crediting `+0.5 log tau` for the pin that was
  never applied.

  Nothing looked wrong from R. `icar_center_field` still centres the field on
  its way into eta, so the linear predictor stayed invariant and the field's
  shape and the fixed effects came out right; only the level was loose. On
  tulpaRatio's 8-unit chain the reported `mean(phi_spatial)` was +200.9, -342.8
  and +423.1 at three consecutive seeds against a legacy level held within 0.02
  of zero, and `tau_spatial` sat at 16.89 against 11.90 with a seed-to-seed
  spread of 0.42 -- the variance component had a different posterior, not just a
  different origin. A chain also spends its adaptation and treedepth on a flat
  direction, which is what makes the field's convergence diagnostics
  meaningless while it does.

  `ModelData::set_spatial_adjacency(n_units, row_ptr, col_idx, n_neighbors)`
  now sets the CSR arrays, the partition and the component count together; the
  engine's own loader uses it at both of its sites, and `compute_param_layout()`
  rejects a field whose partition does not describe its adjacency, alongside the
  existing PC-range-anchor guard. After the fix the same three seeds report a
  level of -0.003 and a `tau_spatial` of 12.05, inside the legacy spread.

* ABI 39 -> 40. No struct layout or callable changed; the bump makes a consumer
  built against 39 report "rebuild required" at first NUTS use rather than
  tripping the new guard.

* **The inner Laplace solve now reaches stationarity at a large random-effect
  scale (gcol33/tulpa#259, gcol33/tulpa#260).** With #255 fixed, solves at a
  moderate scale arrived; at a large one they still stopped short, returning a
  joint score of 1e-04 with `converged = TRUE` while the stationary point sat
  1e-05 away in relative terms. Nothing silently wrong shipped -- the settled-mode
  gate in `.laplace_exact_core()` refuses the exact outer gradient there -- but
  the fits did not arrive. They arrive now: the worst residual over the reported
  configurations drops from 3.4e-04 to 4.7e-10, and each returned mode agrees with
  the observed-curvature stationary point to 1e-12 relative or better.

  The cause is neither a cycle nor a conditioning floor but plain local
  divergence. The Newton weight is the working (expected / quasi-likelihood) one
  wherever it differs from the observed curvature, so the error map near the mode
  is `I - Hw^-1 Ho`, which contracts only while every eigenvalue of `Hw^-1 Ho`
  is below 2. On the `neg_binomial_1` fixture at `phi = 6`, `sigma_re = 5` the
  largest is 2.23: the iterate walks away from the mode geometrically, losing a
  few parts in 1e9 of objective per step, and the stall test then reads the
  growing step as a floor and stops. `inverse_gaussian` (Fisher weight
  `1 / (phi mu)`) does the same, so this is not one family's quirk; families whose
  Newton weight already is the observed curvature cannot reach it at all.

  No acceptance rule reading the objective can fix that. The penalized
  log-posterior is stationary at the mode, so a step gains about `decrement / 2`
  while the objective's own accumulation noise is `8 eps |obj|`; on these fixtures
  the two cross at a joint score near 1e-6, which is exactly where every affected
  solve stopped. Tightening the test makes it worse -- an Armijo sufficient-
  decrease condition rejects genuine progress as noise and left the worst
  residual at 1.4e-05, three of the reported rows unimproved and one clean
  `phi = 0.5` solve regressed from 4.9e-10 to 5.7e-07.

  So the steering moves to the Newton decrement `g' H^-1 g`, which is already
  computed every iteration, is the affine-invariant distance to the mode, and
  keeps full relative precision exactly where objective differences are noise.
  Below the near-mode gate the scale the line search opens with (`newton_trust_scale`)
  halves whenever the decrement grew -- the previous step overshot -- and relaxes
  back toward 1 by 1.5 whenever it fell. A solve whose decrement never grows below
  the gate holds the scale at 1 and takes exactly the trial sequence it always
  did, so every converging fit, the whole #255 fixture (81 iterations at
  `phi = 4`, 323 at `phi = 6`) and the rational-SPDE case the stall path exists
  for are unchanged.

* **Newton convergence reads the proposal, not the damped step.** Both the
  tolerance and the stall test now key on `max|delta|` rather than
  `max|step_scale delta|`. Convergence is a property of the iterate -- `H^-1 g` is
  small exactly when `x` is stationary -- while `step_scale` is the line search's
  choice about how much of that proposal to trust, and conflating them fails in
  the dangerous direction: a search that had to damp a large proposal to nothing
  has not arrived, it has failed to move. With the trust factor able to open at
  `2^-20` this is reachable rather than hypothetical, a proposal of 1e-6 damped to
  the floor otherwise clearing a 1e-12 tolerance. Solves that take the full step
  are unaffected, since the two quantities coincide there.

* Two probes make the mechanism testable from R rather than inferable from where
  a fit landed: `cpp_newton_trust_probe()` replays the damping schedule over a
  supplied decrement sequence, and `cpp_newton_converged_probe()` returns one
  convergence verdict at a given accepted step scale.
  `dev_notes/probe_inner_stationarity.R` gains `curvature` (the eigenvalues of
  `Hw^-1 Ho` at the mode, i.e. how far the working weight understates the true
  curvature) and `bigscale` (the residual table above).

## 0.0.106

* **The Newton stall test no longer reads slow convergence as a converged mode
  (gcol33/tulpa#255).** `newton_converged()` carried a rescue path for an
  ill-conditioned Hessian, where `H^-1` amplifies the gradient's rounding
  residual into a spurious step and `max|delta| < tol` can never fire. It
  detected that as a Newton decrement that had stopped halving -- but the
  decrement shrinks as the square of the convergence rate, so any solve
  converging linearly at rate 0.707 or slower looked identical to one that had
  hit its conditioning floor. `neg_binomial_1` is that solve: its
  quasi-likelihood Newton weight `mu / (1 + phi)` sits far below the observed
  curvature at large `phi`, and the measured rate reaches 0.707 between
  `phi = 3` (0.57) and `phi = 4` (0.70). Those fits stopped 30 iterations in at
  a joint score of 7e-04 instead of 1e-11, reported `converged = TRUE`, and cost
  the exact outer gradient five digits (relative error 1.0e-04 against a central
  difference of tulpa's own `log_marginal`, against 1e-10 at `phi <= 3`).

  What separates the two cases is whether the step is still shrinking: a
  converging solve sets a new shortest step every iteration however slowly,
  while one at its conditioning floor bounces around it (on the rational-SPDE
  fixture the step wanders over 3e-6 .. 4e-5 with per-iteration ratios from 0.25
  to 7). The stall test now runs on that, so the affected solves are left to
  finish -- 81 iterations at `phi = 4`, 323 at `phi = 6` -- and the gradient
  agrees to 1e-10 across every family and `phi` in
  `dev_notes/probe_phi_gradient_families.R`. The rational SPDE case the path
  exists for is unchanged (converged, `n_iter` 9, field correlation 0.96), and
  so is every solve that was already tripping the step criterion.

* **Every Laplace solve reports the residual it achieved, not just whether its
  stopping rule fired (gcol33/tulpa#255).** `LaplaceResult` carries `score_max`,
  the largest absolute component of the joint penalized score at the returned
  mode, surfaced on the R side as `fit$score_max`. `converged` answers a
  different question, and the difference matters because the log-marginal feels
  a mode error quadratically while its theta-gradient feels it linearly. Read
  off the final scatter every driver already performs, so it costs one pass over
  the latent vector. The three `LaplaceResult`-returning exports that hand-rolled
  their own result list (the SPDE Laplace pair and the implicit-diff gradient)
  now go through `laplace_result_to_list()`, which is what makes a new diagnostic
  reach all of them.

* **The exact outer gradient declines a mode that did not settle
  (gcol33/tulpa#255).** `.laplace_exact_core()` keeps only the `log|H|` path
  because the joint score at `x_hat` is zero; a solve that stopped short leaves
  behind exactly the term the derivation discards. The reported residual is
  mapped through the same inverse the mode motion travels
  (`max|dx| <= ||H_true^-1||_inf max|score|`), compared against the latent scale,
  and refused above 1e-6 -- which bounds the outer gradient at ~1e-5 relative,
  four orders under the 1.0e-04 above. Both sides of that threshold are measured:
  settled solves swept over five families and `n = 50 .. 18000` span 3.4e-16 to
  2.1e-11, solves stopped short span 8.8e-05 to 1.2e-02. The refusal is signalled
  as a `tulpa_unsettled_mode` condition carrying the residual, the implied mode
  error and the iteration count, so a direct call says what to change while the
  outer optimizers collapse it into one warning per fit rather than one per trial
  `theta`.

* **A declined gradient no longer lets the outer optimizer report its own
  starting value as the estimate (gcol33/tulpa#255).** When the exact gradient is
  unavailable at a trial `theta`, `optim()` receives a vector of zeros, which it
  cannot distinguish from a stationary point -- so a refusal on the first step
  ends the search there and `phi` is reported back unchanged from where it
  started. The gradient-driven branch of `tulpa_re_cov_nested()` now discards
  that run and restarts derivative-free, the same response it already had to an
  outright optimizer failure, and says so in the warning.

* Grid and chain checkpoint files carry a new payload field, so the format magic
  is `TLPACKP2`. A file written by an earlier version errors with the existing
  "point `checkpoint$path` at a fresh path" message instead of being misparsed.

## 0.0.105

* **The zero-inflation refusal names every family the gate admits
  (gcol33/tulpa#250).** The kernel guard stops on
  `compiled_zi_supported(family)` but restated the supported set beside it as
  three families, where the gate admits seven -- it omitted `neg_binomial_1`
  and both zero-truncated bases, which are exactly the hurdle models. A caller
  who reached the error for some other family was told a hurdle on a truncated
  base was unavailable. The message is now built from the gate:
  `discrete_mass_families()` is the candidate list `has_discrete_mass()` tests
  membership in, and `compiled_zi_supported_families()` runs the gate over it.

* **A mistyped `TULPA_S2Z_DENSIFY_MAX` no longer forces the Woodbury path
  (gcol33/tulpa#251).** The override was read with `atoi()`, which maps every
  unparseable string to `0` -- and `0` is a meaningful setting here, forcing
  the rank-1 storage on every intrinsic field. `=ture` or `=256a` therefore
  changed which algorithm ran for the rest of the session, with nothing in the
  output saying so. It is parsed with `strtol()` against the end pointer now:
  anything that is not a whole non-negative int is treated as unset, so the
  documented default applies. Both storages remain exact, so no fitted value
  moves. The other three `getenv` knobs in `src/` are presence or single-char
  tests whose unparseable case already lands on the default.

* **A warm start whose source supplies the wrong number of SDs is refused
  (gcol33/tulpa#252).** `.build_warm_start()` errors when a source fit's block
  does not match the sampler's layout, but two branches took `s_t[1]` and
  carried on -- seeding every coefficient of a term from the first
  coefficient's SD, for both the mass scale and the `log_sigma_re` starting
  value, and reporting that as a warm start. A single SD still broadcasts (the
  scalar-term case); any length other than one or the term's coefficient count
  now stops with the wording the coefficient placement already used.

* **`get_mode_backends()` derives its mode list from `INFERENCE_TIERS`
  (gcol33/tulpa#254).** Its error message restated the names its sibling
  `select_inference_mode()` already builds from the registry. Correct today,
  stale the moment a tier is added.

## 0.0.104

* **The dispersion convention follows the base family, not the spelling
  (gcol33/tulpa#256).** `phi` is the residual VARIANCE for the normal families
  while the compiled kernels parameterize by the SD, and the conversion tested
  the family name for exact equality. Every `<family>_<link>` spelling --
  `gaussian_log`, `gaussian_inverse`, `lognormal_log` -- therefore reached the
  kernel with the variance where the SD belongs, so a caller asking for a
  residual variance of 0.8 was fitted at 0.64. The four R -> kernel boundaries
  (`tulpa_laplace()`, the `tulpa()` front door, `fit_spde()`'s working weights,
  and the SPDE predict path) now share one pair of converters resolving the base
  family through `.family_base()`, the way every other consumer already did.

  It went unnoticed because nothing downstream distinguishes the two under the
  canonical link: `dW/deta` is identically zero for `gaussian_identity`, so a
  wrong `phi` cancels out of the mode motion and the curvature channel. It stops
  cancelling for `gaussian_log`, whose observed-minus-working delta became
  nonzero in 0.0.103. Changes fitted values for existing suffixed-normal fits.

* **The second dispersion reaches the random-effect covariance paths
  (gcol33/tulpa#257).** `tulpa_eb()` and `tulpa_re_cov_nested()` take `phi2` and
  thread it through the pilot fit, both inner solves and the exact outer
  gradient, which hard-coded `NA_real_`. `family = "t"` had been fitting at the
  compiled default of four degrees of freedom whatever the caller meant --
  reporting an ML-II `phi` conditional on a df nobody chose -- and
  `family = "tweedie"` could not be fitted on those paths at all, failing as a
  generic inner-solve error because its variance power had no way to arrive.
  The power is now required by name, a `phi2` supplied for a family that carries
  none errors instead of being ignored, and `tulpa(phi2 = )` forwards to
  `mode = "eb"` and the nested `Sigma` integrator. Supplying `phi2 = 4` for `t`
  reproduces the previous default bit for bit, which is what pins the threading
  as faithful rather than merely different. The backends carrying `phi2` are
  derived from the registry (`.phi2_backends()`) rather than restated in the
  refusal message.

## 0.0.103

* **`estimate_phi` covers every front-door family that has a dispersion
  (gcol33/tulpa#247).** It was offered for twelve and carried a derivative for
  four; the other eight hit a hard refusal at fit time. `lognormal`, `t`,
  `neg_binomial_1`, `beta`, `inverse_gaussian`, `beta_binomial` and `tweedie`
  are now registered, so `.dispersion_families()` is exactly the set of
  front-door families carrying a `phi`. Each new entry finite-differences
  against the `.FAMILY_OPS` likelihood, score and weight it claims to
  differentiate (~1e-9), and the assembled `dm/dlog_phi` matches a central
  difference of the Laplace log-marginal to the same order.

  `gaussian` and `lognormal` now share one set of derivatives read at their own
  residual -- the lognormal is the same normal density on the log scale plus a
  dispersion-free Jacobian. Tweedie's log-density derivative reads the mean
  event count off the SAME compound Poisson-gamma enumeration that produces the
  density, so the two cannot be taken over different truncations of the series.

  An exact gradient says the optimizer walks the right surface, not that the
  maximizer of that surface is the generating dispersion, so each new family
  additionally recovers its `phi` from simulated truth over several seeds
  (`test-eb-dispersion.R`). Measured mean estimates against truth: `lognormal`
  1.44 / 1.5, `neg_binomial_1` 2.11 / 2.0, `beta` 8.32 / 8.0,
  `inverse_gaussian` 0.48 / 0.5, `beta_binomial` 8.08 / 8.0, `t` 1.21 / 1.2,
  `truncated_neg_binomial_2` 2.96 / 3.0, with the random-effect SD recovered at
  0.68-0.74 against 0.7 throughout. `tweedie` is the exception, and not for a
  reason of its own: the EB path never threads `phi2`, so it cannot be fitted
  there at all (gcol33/tulpa#257).

* **The observed curvature is registered across the family registry.** Whether
  a family's dispersion derivatives assemble into the right gradient turns on
  one quantity: the mode-motion channel `q_eta = (dW/deta) s`. It is right when
  either the working weight carries no eta (gaussian, lognormal, gamma, t) or
  the mode-motion solve is on the TRUE curvature -- which needs the family's
  observed curvature `-l''(eta)`, and only six families had one. Adding the
  mu-space ladder `dgrad_mu_dmu` / `d2grad_mu_dmu2` alongside the existing
  `grad_mu` closes that generically: `-l'' = -(L'' u^2 + L' u1)` covers every
  family on the generic route in one branch, with explicit branches for
  `beta_binomial`, `tweedie` and `t`. Verified against a numerical second
  derivative of the log density for seventeen family/link pairs.

  This is what unparks `beta`, whose derivatives had been kept unregistered
  since its assembled gradient landed ~1e-4 off the objective. It also makes
  the exact mode Jacobian and the marginal fixed-effect precision available
  across the registry, where before they were declined for `gamma`, `beta`,
  `inverse_gaussian`, `beta_binomial`, `tweedie` and `t`.

* **The closed phi Hessian covers both families that were on the stencil, plus
  two more (gcol33/tulpa#248).** It was registered for two of the four families
  that carried the phi gradient. `gamma` had been deferred on the grounds that
  its Fisher working weight differs from its observed curvature; the operative
  condition is narrower than that -- the weight is free of eta, which zeroes
  every channel the two inverses could differ on -- and it holds. `lognormal`
  and `t` join on the same terms.

  `truncated_neg_binomial_2` needed the border itself fixed. It differentiates
  `u` and the phi mode motion, both formed on the true-curvature inverse, while
  reading `Hinv` -- correct only where the two weights coincide, which is what
  the old registration gate quietly relied on. The border now carries
  `dH_true/dpsi`, which needs the phi-derivative of the observed-minus-working
  correction as well as the eta-derivative the random-effect block already had;
  a family in that position supplies `dobs_weight` and `.family_dphi2()`
  withholds the whole entry without it, rather than pairing two different
  inverses. The second mode derivative's right-hand side reads the same
  correction: the score's eta-derivative is the OBSERVED curvature, and the two
  differ for exactly these families.

  The truncated family's three weight derivatives are one chain rule over
  `f(a, a_e)` and the shape's own derivatives, the same partials
  `curvature_deta2_for_family` carries, so `dweight`, `dweight2` and the mixed
  `dweight_deta` come from a single derivation instead of three. The bordered
  Hessian matches a central difference of the exact gradient to ~1e-9 for all
  six registered families. `neg_binomial_1` keeps the differencing stencil.

* **The compiled zero-inflation gate no longer rides on the observed
  curvature alone.** `compiled_zi_supported()` was defined as
  `has_observed_curvature()`, which was the right answer only while the count
  families were the only ones carrying one: the mixture's `y = 0` branch reads
  the log density at zero as `log P(Y = 0)`, which for a continuous family is a
  log-DENSITY -- finite and plausible for gaussian, infinite for gamma or beta.
  It now also requires a discrete base. The R front door reads the compiled
  predicate instead of keeping its own list of families, so the two cannot
  drift; zero-inflated `beta_binomial` is admitted by that (it has both an atom
  at zero and, now, an observed curvature).

## 0.0.102

* **Out-of-pattern Hessian writes are detected instead of silently discarded
  (gcol33/tulpa#249).** `SparseHessianBuilder::add()` dropped any contribution
  whose `(row, col)` was absent from the registered sparsity pattern, and the
  cached-slot fast paths did the same behind `if (slot >= 0)`. Nothing counted
  it. What survives a drop is a Hessian too small in one entry -- finite,
  positive definite, correctly shaped -- which the Newton step, the Laplace
  log-determinant, the marginal standard errors and the exact outer gradient all
  inherit, so the only symptom is a recovery test drifting, the hardest signal
  here to attribute. The invariant had already broken twice: unequal /
  non-contiguous areal components (#241) and weighted-entry blocks (#242).

  `src/hessian_pattern_guard.h` adds the drop counter, the `scatter_slot()`
  helper that single-sources every guarded slot write (14 sites across the
  dense-basis, indexed-cache, joint-batch and sum-to-zero scatters), and
  `HessianPatternGuard`, which snapshots the counter and raises. Detection is a
  counter rather than a throw at the write because the scatter runs inside
  OpenMP parallel regions, where an `Rcpp::stop` escaping the structured block
  is `std::terminate` -- the constraint that also makes
  `LaplaceResult::start_infeasible` a flag. The guard is a pure snapshot, so a
  driver nested inside another measures its own window without disturbing it.
  Placed in `run_nested_laplace_grid` (covering the single-block and both joint
  paths), the batched joint driver, `spde_run_single_fit` and
  `cpp_laplace_fit_gp`.

  Only a NONZERO discarded contribution counts. The scatter index caches resolve
  whole cross products up front -- every (beta_j, RE_g) pair of an arm -- and
  legitimately hold -1 for pairs no observation ever touches; those slots are
  written with a structural zero, which changes nothing whether it lands or not.
  Counting at index-resolution time instead would fire on every real fit.

  The shipping kernels scatter entirely inside their registered patterns: the
  suite runs with the check armed and zero drops, so this lands as a regression
  barrier rather than a bug fix. New `test-hessian-pattern-guard.R` covers the
  counter on both write paths, the raise, the zero-versus-nonzero distinction,
  and a real fit.

## 0.0.101

* **`portable_math.h` compiles for downstream packages on macOS again
  (bugfix).** The header returns `std::pair` from `portable_digamma_lgamma()`
  but included only `<cmath>` and `<limits>`. libstdc++ reaches `<utility>`
  transitively through those, so Linux and MinGW builds were fine and tulpa's
  own sources were fine everywhere -- they include `<utility>` ahead of it. Apple
  clang's libc++ does not, so any package that includes the header first failed
  to compile: `no template named 'pair' in namespace 'std'`, which took
  tulpaObs's macOS `R CMD check` down at `count_grouped_oracle.o`. The header
  now includes `<utility>` itself. Header-only change, no behaviour anywhere.

## 0.0.100

Warm-starting the sampler from a cheaper fit of the same model.

New:

- `estimate_phi = TRUE` now covers `truncated_neg_binomial_2`, and covers it
  alongside a zero-inflation process when the base is zero-truncated (a hurdle).
  The family was listed in `.PHI_FAMILIES` but had no entry in `.FAMILY_DPHI`,
  so the dispersion was refused for it outright. Its three derivatives are the
  untruncated ones plus the phi-derivative of the retained mass
  `P(Y > 0)`; the weight differentiated is the EXPECTED form `Var(y | y > 0)`,
  which is what the Newton solve builds `H` from for this family -- the opposite
  choice from `neg_binomial_2`, where `H` carries the observed curvature. Each
  finite-differences against the registry to ~1e-9 over `phi` in [0.5, 25], and
  the assembled `dm/dlog_phi` matches a central difference of the Laplace
  log-marginal to ~1e-9.

  Under a mixture the refusal now tests the BASE for zero-truncation rather than
  testing for a mixture at all. A hurdle's zero branch is `log(pi)` and carries
  no dispersion, so the base family's derivatives are the mixture's once masked
  at `y = 0`; genuine zero inflation on an untruncated base keeps the refusal,
  since its zero branch depends on `phi` through `P(Y = 0)` as well. Verified to
  ~1e-8 against the two-process log-marginal on a fixture with 35% zeros, and
  `phi_hat` is consistent -- 2.921, 2.871, 2.998 as the count arm grows 750,
  2269, 5057 (12 seeds each, standard error 0.144, 0.091, 0.065).

- The closed-form outer Hessian now carries every mixture the exact gradient
  does -- hurdles and genuine zero inflation alike -- where all of them
  previously fell to the gradient stencil. The curvature block is
  `-Hess(log density)`, so its nine second partials are the FIVE distinct fourth
  derivatives of one scalar, indexed by how many of the four derivatives are in
  `eta`. New `mixture_curvature_deriv2()` returns those five; a hurdle is the
  case where the three mixed ones vanish, so one assembly covers both and the
  hurdle's terms multiply by zero rather than being special-cased.

  The coupled `y = 0` branch differentiates `D = pi + (1 - pi) P(Y = 0)` a
  fourth time, which is the only genuinely new input: it needs `P(Y = 0)`'s
  fourth log-derivative and so the second eta-derivative of the observed
  curvature, which `has_zi_curvature_2nd_derivative()` gates on. That gate is no
  longer narrower than the gradient's.

  The five fields are checked along every route to them: most are reachable by
  differentiating two or three DIFFERENT third-order fields, in different
  directions, and all routes agree to ~1e-11 -- which is what tests the
  `-Hess(log D)` identification rather than the arithmetic. Against a central
  difference of the exact gradient the assembled Hessian reaches 1.5e-9
  (poisson, 53% zeros), 4.1e-9 (`neg_binomial_2`, 57%), 5.2e-9 (binomial, 60%),
  and 3.3e-11 under a hurdle -- the last unchanged to every printed digit by the
  generalization.

- The closed outer Hessian also covers the families whose Newton weight is not
  their observed curvature, which unblocks `truncated_neg_binomial_2` on the
  one-process path and `hurdle_nbinom2` on the two-process one. The assembly
  forms `u` on the observed-curvature inverse but was differentiating it through
  the working-weight inverse -- the same thing only where the two weights
  coincide. Pairing them needs `dH_true/dtheta`, and so the new
  `obs_curvature_delta_deta_for_family()`: for a zero-truncated NB2 the
  difference `W_obs - w` differentiates to the NB2 curvature derivative plus
  `d3 log P(Y > 0)` minus the truncated working one, all three of which the
  registry (and `truncation_shape()`'s `d3a`) already supplied. It
  finite-differences against `W_obs - w` to ~4e-10, and the assembled Hessian
  reaches 2.3e-09 one-process and 3.4e-09 under the hurdle.

- `estimate_phi = TRUE` now runs alongside GENUINE zero inflation as well, and
  the closed Hessian's dispersion border runs under any mixture. A hurdle's zero
  branch is `log(pi)` and carries no dispersion, which is what let the base
  family's registry stand in for the mixture's; genuine zero inflation has zero
  branch `log(pi + (1 - pi) P(Y = 0, phi))`, which carries `phi` through
  `P(Y = 0)` and couples it to BOTH linear predictors -- so the mode motion
  solves against both scores, and the explicit `dW/dphi` runs over the whole
  2 x 2 block.

  The sixteen fields that needs are sixteen CALLS on one engine, not sixteen
  formulas: the mixture density is `q(z) B(eta, phi) + C(z)` in every branch, so
  every mixed partial of `log D` follows from the multivariate
  cumulant-from-moment recursion over the alphabet (eta, z, phi), exact at any
  order and any mix of directions. The base registry covers the rows the mixture
  leaves additively separable and the engine covers the coupled `y = 0` rows;
  the two are summed over disjoint rows, and the engine returns exact zeros
  wherever the registry applies (every `y != 0` row, a hurdle's `y = 0` rows, a
  family with no free dispersion).

  Anchored on `zi_loglik` / `zi_score_eta` / `zi_neg_hessian` and the curvature
  engines, all written independently of this one: 40 field checks agree to
  1.9e-10, with the two shared fourth-order names each reached from BOTH members
  of their pair. The assembled `dm/dlog_phi` matches the two-process
  log-marginal to 5.6e-09 and the bordered Hessian a difference of the exact
  gradient to 2.1e-10, on a fixture with 57% zeros. The one-process path is not
  a second assembly: it is this one with the six z-direction fields at zero, and
  it expands term for term to the expression it replaces.

- `neg_binomial_1` is covered throughout, on the one-process closed Hessian and
  under zero inflation. Its observed curvature is
  `-s + r^2 (psi'(r) - psi'(y+r))`, so `d(W_obs - w)/deta` carries a tetragamma
  and the next rung a pentagamma; both are now in `portable_math.h` alongside the
  existing digamma / trigamma pair, in the same recurrence-plus-asymptotic form.
  The first rung finite-differences to 2.8e-10 and agrees with the same formula
  over R's own `psigamma` to 6.5e-10; the one-process closed Hessian reaches
  3.5e-11.

  The same missing derivative was also what kept the family off the mixture
  curvature gate, so that gate now asks for the property it needs -- the
  observed curvature's eta-derivative, being the registered working-weight one
  plus the correction -- rather than naming families. `beta_binomial`, `t` and
  `tweedie` stay out, having no observed form at all. Under zero inflation
  `neg_binomial_1`'s third- and fourth-order mixture fields reach 2.2e-11 and
  6.3e-11, and its closed Hessian 2.5e-11.

  Still on the stencil, for a stated reason rather than by omission:
  `estimate_phi` for `neg_binomial_1` in any model, zero-inflated or not, since
  the family has no `.FAMILY_DPHI` entry at all -- a separate gap from the one
  closed here.

- Weighted-entry intrinsic fields -- separable multivariate CAR (`mcar`) and the
  areal / temporal varying-coefficient blocks -- are now hard-constrained by the
  same augment-and-centre identification the uniformly-seen fields moved to in
  0.0.99, instead of the old soft sum-to-zero pin (gcol33/tulpa#242). MCAR carries
  the augmented precision `Sigma^-1 (x) Q_aug`
  (`Q_aug = Q + sum_c 1_c 1_c'/J_c` per field, coupled across fields by
  `Sigma^-1` and folded by the sparse solver), so its Sigma-dependent normalizer
  takes the full `n log|Sigma^-1|` per field. A weighted field's constant aliases
  with the coefficient on the covariate it rides on, not the intercept, so the
  centerer folds the field's global level into that column -- `svc_beta_offset`
  for a single field, per-field `field_beta_offset` for MCAR, or `-1` for a
  covariate with no fixed-effect counterpart (the level is then left in the field
  for the augmentation to identify). A declared column is verified against the
  design so a wrong alias cannot silently shift eta.

- `tglmm()` and `tgam()` are named front doors onto `tulpa()`: contract-narrowing
  views, not new engines. Each carries `tulpa()`'s signature minus the arguments
  its model class cannot use (`spatial`, `temporal`), requires the structure that
  defines the class in the formula (`tglmm()` a random-effect term, `tgam()` an
  `s(...)` smoother), and refuses the structures outside it with a pointer to
  `tulpa()` rather than fitting them. Dispatch, tier selection, backends and every
  `tulpa_fit` accessor are unchanged; the fit is byte-identical to the same call
  through `tulpa()`.

  The returned fit carries `"tulpa_glmm"` / `"tulpa_gam"` ahead of `"tulpa_fit"`,
  so `print()` adds the section that model class is read for -- the random-effect
  covariance (with `VarCorr()`'s estimated / sampled / conditioned label) or the
  smoother table -- and `plot()` on a GAM draws the fitted smooths. A door's
  `print()` composes the backend's own report rather than displacing it, so a
  nested-Laplace GAM still shows its integrated hyperparameters, grid size and
  outer Pareto-k.

  Doors are registry-driven (`.TULPA_DOORS`): a new one is an entry plus a stub,
  and `test-doors.R` asserts each signature stays exactly `tulpa()`'s minus its
  withheld arguments, so a new statistical argument on `tulpa()` cannot leave a
  door silently stale.

- `tulpa_check_control(control, allowed, where)` is exported. It is the
  control-list name check every front-door fitter already ran internally,
  promoted so consumer packages validate their own `control = list()` surface
  against the same rules instead of reimplementing them. tulpaRatio's
  `tratio()` is the first caller.


- `tulpa(warm_start = )` seeds the NUTS sampler from a Laplace or empirical-Bayes
  fit: `"eb"` or `"laplace"` fits one first, or pass an existing fit from either
  mode. Chains start at that mode instead of the origin. Chains after the first
  are dispersed around it, so the between-chain spread `rhat()` compares against
  is not collapsed by a shared starting point. Only the NUTS/HMC backends take
  one -- the rest error rather than sample from the default start and report a
  fit that answers a different question.

- It is offered as a capability, not as a speedup: over 10 seeds, redrawing both
  the data and the sampler seed on a crossed random-intercept Poisson model,
  warm and cold were indistinguishable in effective sample size per second
  (paired Wilcoxon p = 0.32 and p = 0.49, with the source fit's own cost charged
  to the warm runs). Single runs of that comparison disagree with each other by
  2-3x, because `min(ESS)` is an order statistic over parameters. Whether it
  pays on harder geometries than this one is untested.

- `cpp_tulpa_glmm_layout()` reports the parameter layout the sampler will use --
  block spans, per-term random-effect shape, and the column names -- without
  sampling. The warm start places values by index against it rather than
  reconstructing the sampler's naming convention, so the layout stays the single
  source of truth for where a parameter lives.

- `tulpa_sample_glmm(warm_start = )` and `cpp_tulpa_sample_glmm()`'s `init` /
  `inv_metric_diag` are the engine-level entry points. `init` is one row per
  chain. The inverse-mass diagonal is assembled but not supplied by default: the
  kernel reads every entry as a posterior variance, and today the random-effect
  entries would be prior variances and the variance components would have no
  estimate at all. `return_joint_hessian` and `tulpa_eb(marginal = TRUE)` now
  supply both, so composing it properly is the follow-up.

- `tulpa(estimate_phi = TRUE)` estimates the family's dispersion instead of
  conditioning on `phi`, which then supplies the starting value. `log(phi)`
  joins the empirical-Bayes maximization as one further coordinate carrying the
  exact derivative of the Laplace log-marginal, so it is the ML-II estimate: the
  hyperprior covers the random-effect covariances and the dispersion enters
  unpenalized. `fit$phi` is the estimate and `fit$phi_estimated` distinguishes
  it from a conditioned value. Available under `mode = "eb"` and for the
  families whose dispersion derivative is registered; any other mode errors
  rather than fitting at the starting value under a name that says otherwise.

  This is what makes a Gaussian GLMM comparable to `lme4::lmer(REML = FALSE)`.
  With the residual variance held at its default the outer maximization absorbs
  the mismatch into the random-effect scales: on a nested `(1 | g1/g2)` fit the
  inner term came out at 0.10 against lmer's 0.31. With the dispersion free the
  same fit reports 0.32, a residual SD of 0.5266 against lmer's 0.5260, and
  fixed effects within 0.006 of an lmer standard error.

- Reference tests against external maximum-likelihood implementations of the
  same likelihoods: `lme4`, `glmmTMB`, `betareg`, `pscl` and `MASS::glm.nb`
  (`glmmTMB` and `pscl` are new in `Suggests`). Every count family, the
  zero-inflation and hurdle mixtures, beta regression, and the random-intercept
  / nested / correlated-slope structures are checked against a second
  implementation on identical data.

  Conditioned on the reference's own dispersion and given a diffuse
  fixed-effect prior, the Laplace mode is the MLE of the same likelihood, so the
  tests assert agreement to a thousandth of a reference standard error rather
  than to a fraction of one. The dispersion half of each likelihood is checked
  separately by where the profile log-marginal peaks. A parameter-recovery test
  against a simulated truth cannot separate a likelihood bug from sampling noise
  at finite N; these can.

- Zero inflation and hurdle mixtures reach the random-effect backends.
  `ziformula` is now carried by `mode = "eb"` and `mode = "re_cov_nested"`, and
  `tulpa_eb()` / `tulpa_re_cov_nested()` take `X_zi` and `zi_prior_sd` directly,
  so the random-effect covariance is estimated or integrated under the mixture
  rather than under a model missing it. Their inner solve is `tulpa_laplace()`,
  which already carried the second process, so the mixture changes that solve and
  not the covariance coordinates the outer objective searches over. The mode is
  `[beta | beta_zi | random effects]` and both fixed blocks are reported:
  `coef()`, `vcov()` and `confint()` return `ncol(X) + ncol(X_zi)` entries, the
  zero-side ones named `zi_*`.

  The exact outer gradient follows the mixture. With two linear predictors the
  per-observation curvature is a 2 x 2 block over the count predictor and the
  zero predictor, and both move with the mode, so `d log|H| / d theta` picks up
  all six of its partials rather than the single `dw/deta` the one-process path
  uses. `laplace_family_zi_curvature.h` supplies them in closed form, contracted
  against the three linear-predictor (co)variances.
  `cpp_family_has_zi_curvature_derivative()` gates it, and the gate is narrower
  than zero inflation itself: an untruncated `y = 0` branch differentiates
  `P(Y = 0)` a third time, which needs the eta-derivative of the OBSERVED
  curvature, and only the families whose working weight already is that have it
  registered. A hurdle base is zero-truncated, so its `y = 0` branch is flat in
  the count predictor and needs no third derivative.

  Refused rather than fitted as a different model: `n_quad > 1` (the adaptive
  Gauss-Hermite inner marginal runs through a single-predictor compiled oracle),
  `estimate_phi = TRUE` (the registered dispersion derivative is the base
  family's, while the mixture's zero branch depends on phi through `P(Y = 0)` as
  well, so the two describe different objectives), and `mode = "re_cov_gibbs"`
  (its conditional carries no second process).

Fixed:

- Continuous GP / NNGP spatial fields (`spatial_gp()`) fitted with exact NUTS
  under-recovered the field: the centered parameterization sampled the field
  jointly with its `(sigma2, phi)` hyperparameters -- Neal's funnel -- and a
  diagonal mass matrix stalled in the neck, so the posterior-mean amplitude
  collapsed toward zero and `sigma2` was under-estimated regardless of warmup
  (gcol33/tulpa#243). The sampling path now uses the non-centered
  parameterization by default (sample `z ~ N(0, I)`, reconstruct
  `w = f(z, sigma2, phi)`), wiring the field's likelihood gradient back to
  `(z, log_sigma2, log_phi)` through the hand-derived NNGP forward/backward in
  an arena `custom_backward`; the stored draws are transformed `z -> w` on the
  way out. `spatial_gp(parameterization = "centered")` restores the old path.

  The same transform is now available on the NNGP spatially-varying-coefficient
  (`spatial_svc(parameterization = "noncentered")`, per term) and multi-scale
  (`spatial_multiscale(sampler = "noncentered")`, per scale) fields, which the
  three blocks reach through one shared per-term applier rather than a copy
  each. The HSGP variants of all three are already spectrally non-centered and
  are unchanged.

  SVC keeps the CENTERED default, unlike GP. The issue that prompted this work
  expected SVC to share the funnel but recorded that as untested; measured on
  the `test-svc-nuts-frontdoor.R` recovery fit (Poisson, n = 120, same data /
  seed / budget for both arms), the centered path is already funnel-free --
  0/700 divergent, field correlation 0.939, sd ratio 0.662, mean treedepth 6.3
  -- and it fits in 163s where non-centered needs well over ten times that for
  the same answer. The funnel that motivated the GP flip was measured on a
  weakly identified field, which is where non-centered earns its cost; a
  well-identified response does not pay it. Reach for `"noncentered"` when the
  field is weakly identified or the centered fit reports divergences.

  Correspondingly there is no new SVC amplitude test: `test-svc-nuts-frontdoor.R`
  already asserts the same sd-ratio band plus a divergence guard on
  well-identified data, which is the correct design for that quantity -- on
  weakly identified data a posterior mean shrunk toward zero is the right
  answer, not evidence of a funnel, so a sd-ratio contrast there measures
  shrinkage rather than geometry.

  The multiscale transform is verified by finite differences against the
  analytic backward on both scales, but its end-to-end amplitude test is
  skipped against gcol33/tulpa#244: that block's range prior is still a Uniform
  behind a hard `-INFINITY` wall (the defect #144 fixed for GP and SVC, which
  multiscale escaped only by being unreachable), and it leaves 82-88% of
  post-warmup draws divergent whichever parameterization is used.

  `spatial_multiscale()` reaches exact NUTS through the `tulpa()` front door
  for the first time: the prior and its parameter layout existed, but no
  spatial-type branch ever built the sampler inputs for it, so the path was
  unreachable. Both scales' fields are reported as `gp_local[i]` /
  `gp_regional[i]` draws alongside their own `(sigma2, phi)`. It has no
  nested-Laplace kernel, so `mode = "structured"` still routes it to the
  conditional Laplace path (which rejects it); `auto` routes it to exact NUTS,
  the integrator that fits it.

- The samplers started every coordinate at the origin, which puts a spatial
  range at `phi = 1`. That is inside the support of every PC-range block, but
  the multi-scale block declares hard range bounds, so bounds excluding 1 left
  the chain starting at `-Inf` -- it never moved and returned an all-zero field
  at a 100% divergence rate. Each bounded range now starts at the geometric
  mean of its own bounds; a caller-supplied `init` is left untouched, and
  unbounded blocks are unaffected.

- A Laplace fit made with no `beta_prior` reported standard errors that omitted
  the prior its mode was found under. The compiled kernels apply a built-in
  `N(0, 100^2)` ridge whether or not one is supplied, but the reported
  curvature added a penalty only when the argument was present, so the default
  fit's `vcov()` described a different posterior than its point estimate.

- Fixed-effect standard errors on a nested-Laplace fit carrying an RW1 / RW2
  temporal field -- or any `s(...)` smoother, which is an RW2 over the binned
  covariate -- were wrong for the intercept, which was reported at the
  fixed-effect prior SD (100) regardless of the data. On a 400-observation RW1
  fit the intercept SE was 99.998 against an exact-MCMC value of 0.047.

  An intrinsic field's constant null direction is jointly unidentified with the
  intercept. The spatial (ICAR / BYM2) and sampler paths identify it by
  augmenting the field precision during the solve, so the level sits under the
  field's own tau; the temporal Laplace kernels never did. They identified the
  level only by centring the mode afterwards and folding the removed mean into
  the intercept -- which fixes the point estimate and leaves that direction flat
  in the Hessian the marginal covariance is read from. Point estimates, field
  shape, log-marginals and every other coefficient were unaffected, which is why
  nothing else moved: on the same fit the slope SE was already correct to 2%.

  The RW1 / RW2 kernels now carry the same augmentation, over the field's global
  constant (the between-group level contrasts of a panel field stay improper by
  design, matching the sampler). The pin is registered for the solver to fold in
  rather than stored, so an RW chain's Hessian stays tridiagonal / pentadiagonal;
  both storages are exact and agree to five significant figures. The augmentation
  itself is now one helper shared with the ICAR kernels, whose numbers are
  unchanged.

- Fixed-effect standard errors on a conditional-Laplace fit
  (`tulpa_laplace()`) carrying an intrinsic areal field were understated for the
  intercept: 2.3x at 10 areal units, 2.7x at 40, 3.0x at 120 for ICAR, and about
  1.3x for BYM2 (whose structured half carries the field). Reported intervals
  were correspondingly too narrow.

  `summary()` / `vcov()` / `confint()` marginalize the field out of the joint
  Hessian by reconstructing the field precision in R. That reconstruction
  augmented the ICAR structure with a unit `11'`, where the kernel it is meant to
  mirror augments with `1_c 1_c'/J_c` per component -- a factor `J` too much
  weight on the constant direction, which is exactly the direction the intercept
  is aliased with, and no per-component handling on a replicated field. The
  reconstruction now matches the kernel and follows the component split
  `for_each_icar_component` walks. Slope standard errors are unchanged to five
  decimals, the pin being orthogonal to covariate contrasts.

  The reconstruction is now pinned against the kernel's own exported
  `log_prior_icar` rather than against a second R statement of the convention:
  the previous reference encoded the same unit `11'`, so it agreed with the
  helper by construction and could not have caught this.

- The sum-to-zero augmentation of an intrinsic areal field (ICAR / BYM2 / MCAR)
  pinned each connected component's constant over an equal-size contiguous split
  of the field, `n / n_components` nodes per component. This is correct for a
  single connected map and for a `spatial(by = )` replicate over the
  block-diagonal `I_L (x) Q` (equal-size, contiguous copies), but wrong for a
  genuine disconnected map whose components are unequal in size or not
  contiguous in the node ordering (a mainland plus islands, the islands sorted
  among the mainland cells -- the standard US-counties layout). The wrong node
  sets shifted both the fitted field and its variance component, on the Laplace
  and the exact-NUTS paths alike, and the marginal intercept SE with them.

  Components are now taken from the adjacency graph itself: the connected-
  component partition (`inst/include/tulpa/graph_components.h`, one DFS computed
  once per fit and carried on `ModelData`) drives every consumer -- the Laplace
  gradient / Hessian / pattern / log-prior, the autodiff NUTS prior, the MCAR
  block, and the R marginal-SE reconstruction -- so each component's constant is
  pinned over its actual nodes. The rank-1 sum-to-zero fold the sparse solver
  uses (Woodbury step and block-Schur log-determinant) now carries a node-index
  list, so a component that is too large to densify folds exactly whether its
  nodes are contiguous or not. A connected graph and an equal-size replicate are
  the trivial partition and stay byte-identical. `ModelData` grew
  `spatial_partition` and the ABI version is 37; consumer packages that
  `LinkingTo: tulpa` must be rebuilt.

- `control = list(n_warmup = )` was silently dropped on the sampler backends.
  `tulpa()`'s control surface is the union over the backends it dispatches, so
  `n_warmup` (which the NUTS-SPDE driver reads) passed the front-door check, and
  the subset that narrows `control` for the chosen fitter then discarded it
  because `tulpa_sample_glmm()` spells the same knob `warmup`. Warmup silently
  stayed at `n_iter / 2` while appearing to honour the request. Control knobs
  are now canonicalized through one alias table, so a fitter that reads the
  other spelling still receives it and an explicit value always wins.

An exact analytic gradient of the joint-field Laplace log-marginal with respect
to the random-effect covariances, and the outer optimizer and marginal
correction rebuilt on top of it.

New:

- `tulpa_laplace(return_joint_hessian = TRUE)` returns `H_joint`, the full
  joint posterior precision of `[beta | random effects]` at the mode, as a
  symmetric sparse matrix. Previously only its fixed-effect Schur complement
  (`H_beta`) was reachable, so nothing outside the kernel could differentiate
  the quantity the Laplace approximation takes the determinant of.
- `dw/deta` for every compiled family (`laplace_family_curvature.h`): the
  eta-derivative of the curvature the Newton system uses. For poisson,
  binomial, neg_binomial_2 and the truncated families that is the true third
  derivative of the log density; for the families whose Hessian carries an
  expected or working weight it is deliberately the derivative of THAT weight,
  since it is the weight the objective is built from.
  `cpp_family_has_curvature_derivative()` gates it, mirroring
  `has_observed_curvature()`.

Changed:

- The outer optimization over the random-effect covariances now runs BFGS
  against the analytic gradient where one is available, instead of Brent
  (`k == 1`) or Nelder-Mead (`k >= 2`). Measured 1.6x fewer inner Laplace
  solves for a scalar block and 2.8x for a correlated one; the gain grows with
  the number of hyperparameter coordinates, since the simplex is what scales
  badly. Falls back to the derivative-free path for a family with no exact
  curvature derivative, for the AGHQ inner marginal (`n_quad > 1`, a different
  objective), and if a gradient-driven run fails outright.
- `tulpa_eb(marginal = TRUE)` builds its correction from the analytic gradient:
  the mode Jacobian `J` is now closed-form rather than finite-differenced, and
  `H_theta` comes from differencing an exact gradient (`2k` solves) rather than
  second-differencing the objective (`1 + 2k^2` solves). Same correction, 2 to
  2.8x fewer solves.

Notes:

- The Fisher-identity gradient the AGHQ path supplies is not a substitute at
  the Laplace case: it omits the term that flows through the mode into
  `log|H|`, which is wrong by 9 to 58 percent on the checked cases. It vanishes
  only for a Gaussian response, where the curvature does not move with the
  linear predictor. Derivation and numerical confirmation in
  `dev_notes/laplace_exact_gradient.md`.

Identification of intrinsic fields:

- Intrinsic fields are now identified by AUGMENTING the precision rather than
  penalising the field's sum: `Q_aug = Q + sum_c 1_c 1_c' / J_c`, with the field
  centred on its way into the linear predictor. The constant direction then
  carries the field's own precision and never reaches the likelihood -- a free
  draw that integrates out -- instead of being held by a stiff penalty the
  sampler still has to traverse (gcol33/tulpa#241). This is what INLA's
  `constr=TRUE` and Stan's `sum_to_zero_vector` do. Applied to ICAR and BYM2 on
  both the exact-NUTS and Laplace paths, and to the multiscale temporal trend
  and seasonal arms. On the reference ICAR fit the correlation between the field
  sum and the intercept goes from -0.996 to 0.035, the intercept SD from 0.31 to
  0.026, and the field sum sits at the `sqrt(J/tau)` the augmented prior
  predicts.
- The rank moves with it. `Q_aug` fills one direction per pinned component, so
  ICAR -- whose null space is exactly its component constants -- becomes full
  rank and the normalizer takes `J log tau`. Carrying the augmentation while
  keeping the deficient rank would bias the variance component low. The rank
  helper takes `rank(Q)` and the number of directions filled rather than the
  field length, because that identity is not universal: a non-cyclic RW2 also
  has a linear null direction a sum-to-zero augmentation does not touch, so it
  stays deficient by one.
- `temporal_multiscale()` is reachable from `tulpa(mode = "hmc")`. The block's
  layout, prior and gradient all existed but nothing in the package set the flag
  they key on, so it could only be driven from a consumer package. Its
  parameters are named (`trend[i]`, `seasonal[i]`, `short_term[i]`, and the
  matching scales) instead of positional. Recovery through the front door:
  `cor(trend, truth) = 0.97` on a 36-point RW1 trend.

Fixed:

- The multiscale temporal Gibbs sampler discarded the level it removed when
  centring the seasonal arm, and never centred the trend at all -- unlike every
  spatial Polya-Gamma kernel, which folds the removed mean into the intercept so
  eta is unchanged. Discarding it shifted eta every sweep, lagging the intercept
  behind the field and inflating the variance. The intercept is now identified,
  which the recovery test previously could not assert.
- Every ICAR block factory on the Laplace paths left the connected-component
  count at its default of 1, so a disconnected field (what `spatial(by=)`
  produces) was pinned once over the whole vector and normalized at rank
  `J - 1`, while the sampler and the Polya-Gamma kernels derived the true count
  from the same adjacency. The count is now derived there too.
- The sum-to-zero fold carries an optional dense coupling over its rank-1
  vectors, so a Kronecker-structured field can express the augmentation its
  precision implies (a multivariate CAR block's `Sigma^-1 (x) 11'/J` couples
  fields, which independent rank-1 terms cannot represent). `LatentBlock::center`
  reports where each removed constant belongs rather than assuming the arm
  intercept, since a block reached through a per-observation weight aliases with
  that covariate's coefficient instead.
- `tulpa_laplace(return_hessian = TRUE)` assembled the marginal fixed-effect
  precision by hand as `X'WX - X'WZ (Z'WZ + D^-1)^-1 Z'WX` from
  `glmm_weights()`. That form reaches ONE linear predictor, so it sliced the
  random effects out of the mode at `ncol(X)` -- inside the `beta_zi` block when
  a zero-inflation process is present -- and it carried the R-side Fisher weight
  rather than the curvature the kernel found the mode under, measured 1.8% apart
  on `neg_binomial_2`. It is now the Schur complement of the joint curvature over
  the random-effect block, corrected to the observed weight for the two families
  whose Newton weight is not it. The joint Hessian already carries the
  fixed-effect prior and the random-effect penalty the mode was found under, so
  nothing is added back.
- The exact outer gradient's `dW/dtheta` channel formed its mode-motion solve on
  the working-weight inverse. That channel is `v_r' (dx_hat/dtheta)`, and the
  mode follows the true stationarity condition, so it is governed by the observed
  curvature -- the same inverse the mode Jacobian already used. The two coincide
  wherever the working weight is the observed one, which is why a poisson or
  gaussian check could not see the difference; on `truncated_neg_binomial_2` the
  inverses differ by ~9% and the gradient was off ~0.8%.
- The closed-form outer theta-Hessian declines for `neg_binomial_1` and
  `truncated_neg_binomial_2` instead of returning an inexact value. Its
  `du/dtheta` differentiates `u` through the working-weight inverse while `u`
  itself is now formed on the observed-curvature one, and forced on it misses the
  exact Hessian by 2.6e-2 and 2.7e-4 respectively. Those two families take the
  gradient stencil instead -- a central difference of the exact gradient, 2k
  gradient evaluations rather than the `1 + 2k^2` objective re-solves of the full
  fallback -- which is checked against a second difference of the objective.
- `marginal` is a formal argument of `tulpa_eb()` rather than one of its control
  knobs, so the front door had no route to the hyperparameter-uncertainty
  correction. `control$marginal` on `tulpa()` / `tglmm()` now forwards to it, and
  passing it in `control` to `tulpa_eb()` directly errors rather than being
  accepted and ignored.

## 0.0.97

The soft sum-to-zero constant that identifies intrinsic latent fields, moved
onto the reference idiom and single-sourced (#241).

Fixed:

- The soft sum-to-zero penalty on intrinsic fields is now derived from
  `sd(sum phi) = kappa * n` with `kappa = 0.001` (Morris et al. 2019; the
  constant brms and the Stan ICAR case study use), via the new exported
  `tulpa/soft_sum_to_zero.h`. The sampler's ICAR and BYM2 branches previously
  applied a flat precision of `0.01`, i.e. `sd(sum phi) = 10`, which leaves the
  field's constant direction free and aliased with the intercept. On a 36-unit
  ICAR binomial fit against a known intercept of 0.5, over 5 seeds:
  `cor(sum(phi), intercept)` moves from -0.996 to -0.037, `sd(sum(phi))` from
  11.1 to 0.036 (= `kappa * n`, the target), and the intercept posterior SD from
  0.310 to 0.027. Correlation between the posterior-mean field and truth is
  0.989 either way: the penalty pins the level without shrinking the pattern.
- The sampler's ICAR branch pinned one direction for the whole field while its
  rank normalizer already used `J - n_components`. It now pins one direction per
  connected component, matching the rank term and the Laplace path, so a
  disconnected graph (`spatial(by =)` replicated CAR) is fully identified.
- The multiscale temporal trend (RW1/RW2) and seasonal (cyclic RW1) arms carried
  no sum-to-zero term at all. Both are intrinsic and both enter the same linear
  predictor, so each had a null direction aliased with the intercept and with
  the other. Both are now pinned; the proper short-term arm (AR1/IID) is not.
- The spatiotemporal, TVC and SVC penalties took the constant as a defaulted
  argument that call sites filled with a bare literal. The precision is now
  derived inside each penalty from the length of the sum it pins, so no call
  site can pass a kappa where a precision is meant. The interaction margins take
  their own constants rather than sharing one (a space margin sums `S` terms, a
  time margin `T`). The SVC penalty is written on the sum rather than the mean;
  the two forms are the same penalty, related by a factor of `n_obs`.

Internal:

- `for_each_icar_component()` moved from `laplace_spatial_priors.cpp` into the
  shared `icar_kernel.h`, so both engines walk the null space identically.
- The MCAR block factory's own copy of the constant is gone; it takes the shared
  per-component precision. `test-mcar-prior.R`'s reference now reads the same
  helper the C++ does rather than hard-coding the value.

## 0.0.96

A settable prior on the zero-inflation coefficients, and the correctness fix
that finding it turned up.

New:

- `tulpa(zi_prior = list(sd =))` sets the prior SD on the zero-inflation
  coefficients, `beta_zi ~ N(0, sd^2)`; the default is 2.5. One scalar applies
  to the whole block and the mean is fixed at 0, matching what the compiled
  kernels carry -- a `mean` entry is refused rather than silently dropped, and
  the list form leaves room for a per-coefficient prior later. `sd = Inf`
  removes the penalty. `tulpa_laplace()` takes the same value as the scalar
  `zi_prior_sd`.

- The prior is worth setting rather than accepting: where a ZI design level
  contributes no zeros the likelihood is monotone in that level's coefficient
  and alone would send it to `-Inf`, so the prior is what makes the mode exist.
  `zi_prior` is how that scale is now chosen.

Fixed:

- The Laplace path applied the zero-inflation prior only when the caller also
  supplied `beta_prior`. Those govern different blocks -- `beta_prior` is the
  count block -- so the ZI coefficients silently carried a prior SD of 2.5 with
  a fixed-effect prior present and the weak built-in 100 without one, while
  every sampler path applied 2.5 unconditionally. The same `ziformula`
  therefore fit a different model depending on `mode`, and on an unrelated
  argument. The prior is now applied whenever the ZI block exists. A second
  edge in the same branch: supplying only `beta_prior$mean` left the precision
  vector empty, so the guard that resized it skipped the ZI tail.

- `ziformula` on `tulpa()` and `X_zi` on `tulpa_laplace()` appeared in the usage
  blocks with no `\item`, an undocumented-argument check warning carried since
  the zero-inflation port. Both are now documented.

- The hurdle-factorization test compared a joint fit's zi block against a
  standalone binomial without matching their priors, and absorbed the resulting
  gap in a 1e-4 tolerance -- the same prior discrepancy described above, hidden
  rather than caught. Matched, the factorization is exact to solver precision,
  so the two zi assertions are now pinned at 1e-9, and a companion test repeats
  the identity at a prior far from the default to show it is a property of the
  likelihood and not of the default scale.

## 0.0.95

One diagnostic front door, selected by draws provenance.

New:

- `diagnostics()` is the entry point for posterior diagnostics on any fit. It
  reads how the draws were produced and returns the reliability question that
  applies: chain mixing (improved Rhat, bulk / tail / mean / sd / quantile ESS,
  MCSE) for MCMC draws; the PSIS approximation-reliability table (`pareto_k`,
  grid quadrature ESS) for i.i.d. draws from a deterministic backend; `NULL`
  with a message for a point summary that carries no sample.

- The routing is a registry keyed by provenance kind, so a new engine class is
  one entry plus its table builder rather than another branch. This replaces the
  hand-rolled `if (!is_chain)` dispatch that previously sat inside
  `mcmc_diagnostics()`.

Deprecated:

- `mcmc_diagnostics()` and `laplace_diagnostics()` are deprecated in favour of
  `diagnostics()`. Both still work and return exactly what they always did;
  `mcmc_diagnostics()` in particular still routes an i.i.d. fit to the
  reliability table, which is the behaviour that made its name wrong. The
  `laplace_diagnostics` class and its `print()` method are unchanged, so code
  that inspects the returned object keeps working.

Fixed:

- The `laplace_diagnostics()` example fitted with `mode = "laplace"`, which
  returns a mode plus covariance and carries no draws, so the example printed a
  "no posterior draws" message instead of the table it documents. Both it and
  the new `diagnostics()` example now use `mode = "smc"`, a deterministic
  backend that does emit draws.

## 0.0.94

Empirical Bayes over random-effect covariances, and the lme4 / posterior
accessor surface.

New:

- `mode = "eb"` and `tulpa_eb()`: estimate one or more random-effect covariances
  by maximizing the Laplace marginal likelihood over them, then report the fixed
  effects conditional on the maximizer. This is the plug-in counterpart of
  `tulpa_re_cov_nested()`, and deliberately not a second implementation of it:
  both call the extracted `.re_cov_theta_fit()`, so they share the objective, the
  inner solve and the optimizer, and their `theta_hat` values are identical on
  the same data (asserted with `expect_identical`, not a tolerance). EB stops at
  the mode; the nested integrator carries on and marginalizes around it.
  Registered as a Tier-2 backend and opt-in by name -- conditioning on
  `Sigma_hat` drops the hyperparameter uncertainty, so `auto` never selects it.
- `fixef()` and the `as_draws()` / `as_draws_array()` / `as_draws_matrix()` /
  `as_draws_df()` / `as_draws_rvars()` family on `tulpa_fit`. `.onLoad()` also
  registers these methods (and the existing `ranef()`) on `lme4::fixef`,
  `nlme::fixef`, `lme4::ranef`, `nlme::ranef` and `posterior::as_draws*` when
  those packages are installed, so `lme4::fixef(fit)` dispatches without any of
  them entering Imports -- and without tulpa masking their generics on attach.
- A Gaussian-approximation fit carries no draws, so `as_draws()` on one errors by
  default and names the alternative. `as_draws(fit, n_draws = )` opts in to
  sampling `N(coef, vcov)`; that is a modelling decision (every downstream
  `posterior` summary would treat the approximation as a posterior sample), so it
  is never taken silently.

Correctness:

- **`$` is now exact on a `tulpa_fit`** (new `$.tulpa_fit`). A fit is a list, so
  `$`'s default partial matching let an ABSENT field resolve to any longer field
  it prefixes -- and since the accessors decide which posterior shape a fit
  carries by testing whether `$draws` / `$mode` / `$modes` / `$cov` is NULL, a
  partial match there reads the wrong object outright. Live collisions:
  `$draws -> draws_kind` (the string `"iid"`, on every Laplace/EB fit),
  `$mode -> model_matrix` (the design matrix, on every sampler fit),
  `$sigma -> sigma_re` (AGQ fits), `$theta -> theta_hat` (EB fits).

  Three symptoms were live: `posterior_sample()` returned `"iid"` and
  `tulpa_draws_array()` built a 1x1x1 array from it for every Laplace-shaped fit;
  `laplace_diagnostics()`, which exists for exactly those fits, could never reach
  its "no posterior draws" branch; and `print()` on an AGQ fit reported the
  random-effect standard deviation under the `sigma:` label, where it reads as
  the dispersion. The remaining collisions were latent -- masked by a branch
  ordered ahead of them or by a companion `&& !is.null(...)` guard -- so **no
  coefficient, standard error or interval changes**: `coef()`, `vcov()`,
  `confint()` and `summary()` return exactly what they did before on every
  backend. The draws accessors additionally route through one `.fit_draws()`
  helper.

  This reaches model packages that set `class = c("<model>_fit", "tulpa_fit")`:
  they inherit exact `$` too, so any of their code that was relying on a partial
  match now gets NULL. That reliance was always a bug, but it will surface here.
  Cost is ~2 us per `$` read (an S3 dispatch); `coef()` and `summary()` are ~0.5
  ms, so it is not measurable at the accessor level.
- **`offset()` was silently dropped by the RE-covariance backends.** `tulpa()`
  never threaded the offset into `tulpa_re_cov_nested()` (which had no `offset`
  argument at all), so a rate model reached under `mode = "re_cov_nested"` -- or
  via the automatic random-slope redirect off `mode = "laplace"` -- fitted counts
  instead. On a simulated rate model with exposure spanning 50x and a true
  intercept of -1.5, it returned +1.66. `tulpa_re_cov_nested()` and `tulpa_eb()`
  now take `offset` and thread it through the inner `tulpa_laplace()` solve,
  which always supported it. Where an offset genuinely cannot be carried it now
  errors instead of dropping: `n_quad > 1` (the compiled per-group AGHQ oracle
  has no offset term) and the `re_cov_gibbs` backend.
- The outer optimization over the random-effect covariance(s) now warns when it
  does not converge, in both `tulpa_eb()` and `tulpa_re_cov_nested()`. It
  previously returned wherever the optimizer stopped without a word -- for EB
  that is the estimate, and for the nested path it is the centre the integration
  grid is placed around. `control$outer_maxit` (default 500) sets the budget, and
  `tulpa_eb()` reports the code as `$outer_convergence`.
- One-dimensional outer optimization uses Brent rather than Nelder-Mead, which R
  warns is unreliable there. `k == 1` is the common case (a scalar `(1 | g)`
  block), so this affects `tulpa_re_cov_nested()` as well as EB. Brent reports
  success at a bracket endpoint, so a variance component pinned at the bracket
  now warns rather than being reported as a fitted value -- the low end is the
  classic empirical-Bayes collapse to `sigma = 0`.

## 0.0.93

Zero inflation as a composition over the count families, and compiled kernels
for the last three families that had none.

New:

- `ziformula` builds a second linear predictor and forms the mixture at the
  likelihood level, so zero inflation is a composition over the base families
  rather than a set of families of its own. The math lives once, in
  `R/family_zi.R` and `src/builtin_family_zi.h`, mirrored term for term.

- The two backend classes reach it through different hooks because they differ
  structurally. The sampler paths take the `logit_zi` callback argument the
  engine already plumbed; the Laplace spec shim hardcodes that argument to zero
  and instead rides the zi predictor as process 1, so the mixture's cross term
  reuses the existing row-major `n_processes x n_processes` curvature block
  rather than introducing a second curvature contract. Coverage is derived from
  `BACKEND_REGISTRY`, so a new sampler backend inherits ZI with its fitter, and
  backends that would silently drop the mixture refuse instead.

- Hurdle models fall out of this as ZI over a zero-truncated base: with no atom
  at zero the mixture degenerates exactly, so they need no families of their
  own. The degenerate case is reached by taking the `p0 = 0` limit rather than
  by evaluating a density at a point the family does not define -- the general
  branch is correct in double arithmetic but yields NaN under AD. Both
  directions are pinned, including the converse that an untruncated base keeps
  a non-zero cross term.

- Compiled kernels for `neg_binomial_1`, `truncated_poisson` and
  `truncated_neg_binomial_2` on both the Laplace and the AD / sampler paths.
  These three were registered in R but absent from C++; `.R_ONLY_FAMILIES` is
  now empty, so every registered family is fittable. Density, score, working
  weight and observed curvature agree with the R registry to machine precision,
  and each family recovers its generating parameters through the Laplace path.

- The two zero-truncated families share their retained-mass term
  `log P(Y > 0)` and its first two eta-derivatives, differing only in `a` and
  its derivatives, so `truncation_term()` computes the pieces once and the
  density, the weight and the observed curvature all read from it.

- Two curvatures are now distinguished rather than conflated.
  `grad_hess_for_family()` returns the Newton working weight, chosen for
  positive-definiteness; `obs_grad_hess_for_family()` returns the observed
  curvature at the realized `y`, mirroring `.family_obs_weight()` in R. The
  zero-inflation mixture needs the latter, because its `y = 0` branch
  differentiates the density rather than taking an expectation. The distinction
  is not cosmetic: `neg_binomial_1`'s observed curvature turns negative once
  `y` sits well above the mean, which would break Newton, while its working
  weight `mu / (1 + phi)` never does. The mixture only ever evaluates the
  observed form at `y = 0`, where it is positive.

- Compiled ZI coverage follows from that: a family becomes ZI-fittable exactly
  when its observed curvature is registered, which is now `poisson`,
  `binomial`, `neg_binomial_2`, `neg_binomial_1`, `truncated_poisson` and
  `truncated_neg_binomial_2`. The remaining count families keep the R-level
  composition (`zi_loglik()` and friends) for density work and are refused at
  the front door for fitting.

- `arena::Var` and `ad::Var` gain `expm1` (`fwd::Dual` already had it), and
  `autodiff_utils.h` gains `expm1_fn` plus a `log1m_exp_fn` that splits at
  `log 2`. Both branches differentiate, so the truncated AD densities are exact
  rather than falling back.

- `cpp_family_obs_terms()` and `cpp_family_ad_terms()` give the two new
  surfaces test probes. The latter evaluates the AD density at `fwd::Dual`, so
  its value checks against the independent double implementation and its
  derivative against the analytic score.

Correctness:

- The silent Poisson fall-through is gone. `variance_fn`, `grad_mu` and
  `log_lik_mu` each ended in the Poisson branch, so a family known to R but not
  to C++ fitted, and fitted the wrong likelihood -- which is what
  `.R_ONLY_FAMILIES` existed to guard against. They now raise, and that is what
  makes emptying the list safe.

- `.family_base()` stripped link suffixes in registration order and so resolved
  `beta_binomial` to `beta`. Every validator built on it was applying beta's
  rules to beta-binomials, including `.validate_family_counts`, which silently
  skipped integer-count validation for that family. It now matches exactly
  first, then by longest prefix.

- `src/autodiff_fwd.h` was a byte-equivalent copy of
  `inst/include/tulpa/autodiff_fwd.h` serving a single include, and is deleted
  rather than left to diverge once `expm1` landed.

## 0.0.92

Audit fixes (0.0.91 review, issues #228-#239).

Correctness:

- The batched joint nested-Laplace driver (`compute_eta_species`) now adds the
  per-observation `offset`, which it silently dropped -- multi-arm joint fits
  with an exposure offset on a coupled arm no longer disagree with the
  single-species path (#228).
- Cyclic RW2 is now honored on the joint multi-block kernel: the `cyclic` flag
  reached the block but the rw2 precision / pattern / log-prior calls were
  hardcoded acyclic, so a cyclic seasonal RW2 in a joint prior fit as acyclic.
  This mirrors the single-arm fix in #218 (#229).
- The spatiotemporal temporal rank is now single-sourced through
  `rw1_rank` / `rw2_rank`; the centered Type-IV path had drifted to a full-rank
  cyclic normalizer inconsistent with the non-centered path (#230).
- The SPDE Matern `(range, sigma) <-> (kappa, tau)` conversion is now
  nu-general (`sigma = 1 / (sqrt(4*pi*nu) * kappa^nu * tau)`) and single-sourced
  via `.spde_kappa_tau` / `.spde_range_sigma`; the six copies hardcoded the
  `nu = 1` normalizer, mis-calibrating fractional-nu SPDE fits. The default
  `nu = 1` path is byte-identical (#231).
- `select_main_params()` now strips comma-separated multi-index latent names
  (`factor[i,j]`), so latent-factor fields no longer flood the diagnostic
  display (#232).
- The simulation diagnostics (`test_dispersion`, `test_zero_inflation`,
  `check_model`) error via a shared `.resolve_obs()` when the observed response
  cannot be found, instead of fabricating a "0 observed zeros" result (#233).
- `tulpa_hyper_grid(var_of_means_consistency = TRUE)` now recomputes the
  per-cell log-prior after the consistency pass grows the grid, fixing a
  length mismatch that returned `NA` reweighting (#234).

Validation & docs:

- `spatial_gp()` / `spatial_svc()` / `spatial_multiscale()` reject unsupported
  covariances (`gaussian`, `spherical`) and Matern `nu` outside `{1.5, 2.5}` at
  construction rather than deep in the fit (#238).
- Corrected the `tulpa_em_laplace(damping=)` and `agq_fit(sigma_eps=)`
  documentation to match the implementation (#239).

Clean-up:

- Removed process/status/refactor-history comments from committed source and a
  stale issue reference (#235); deleted an orphaned duplicate SVC roxygen block
  in `spatial_gp.R` (#236); single-sourced the natural-scale hyperparameter
  transforms shared by `spatial_range()` / `temporal_corr()` (#237).

## 0.0.91

Audit fixes (0.0.90 review, issues #218-#227).

* **Cyclic RW2 dropped on the multi-block nested-Laplace path (#218).** A
  `temporal_rw2(cyclic = TRUE)` block was honored on the single-block and
  exact-NUTS paths but silently ignored once a second latent block routed the
  fit through the multi-block driver: the wrap-around second-difference penalty
  and the `T-1` (vs `T-2`) rank normalizer were hardcoded to the acyclic form.
  The C++ multi driver and the joint-multi spec builder now thread `cyclic` for
  RW2 as well as RW1.

* **NNGP marginal-SE precision builder over-allocated (#219).** The triplet
  accumulator pre-size squared the grand total of neighbour counts
  (`sum(...)^2`) instead of the per-row `sum((...)^2)`, allocating
  `O((n_spatial * nn)^2)` integers -- tens of GB at a few thousand locations, so
  `summary()` / `vcov()` / `confint()` on an NNGP fit could OOM. The result was
  always numerically correct; only the allocation was quadratic.

* **`tulpa()` control surface omitted the joint keyset (#220).** Inline
  `spatial()` / `temporal()` field fits route through the joint nested-Laplace
  driver, but `tulpa()`'s control whitelist did not union its keys and the field
  fitter forwarded the raw control unmasked. Legitimate joint knobs
  (`adaptive_grid`, `prune`, ...) were rejected at the front door, and some
  keys valid elsewhere hard-errored inside the joint driver. The union now
  includes `nested_laplace_joint`, and the field fitter subsets control to the
  joint keys like every other backend route.

* **Outer Pareto-k target carried a spurious Jacobian on the single-block grid
  path (#221).** The default positive-scale grid is geometric (uniform in
  `u = log theta`) and the integrator weights it with plain
  `softmax(log_marginal)` and no volume element, so `exp(log_marginal)` is
  already the `u`-space posterior density. Both the single-block
  (`.nested_grid_pareto_k`) and joint / multi-axis (`.joint_pareto_inv`) paths
  added a `+ sum(u)` change-of-variables term on the `log` axes, tilting the
  certified target away from the posterior the fit reports (a false reliable /
  unreliable verdict; draws and moments were unaffected). Both now drop it on the
  `log` axes, matching the integrator and the SPDE Pareto-k path. A
  target-agnostic ground truth (PSIS of the grid-node `log(w_k) - log q(u_k)`)
  confirms the correction: with the Jacobian the outer k-hat overstated the truth
  by ~0.25-0.55, enough to flip a verdict near 0.7. The correlation axis
  (`logit01`) keeps its logit Jacobian, correct for the grid uniform in the
  natural `rho`.

* **`auto` mode errored on SVC / TVC (#222).** With the default
  `mode = "auto"`, a spatially- or temporally-varying-coefficient model fell to
  a Laplace / size heuristic and then hit the varying-coefficient guard, which
  only the exact ModelData NUTS backend clears. `auto_select_mode()` now routes
  SVC / TVC to the exact backend so the default mode fits end to end.

* **Gradient-check fallback leaked across fits (#223).** A failed warmup
  gradient check flipped the process-global gradient mode to numerical and never
  reset it, so every later fit in the session silently ran slower
  central-difference gradients. The fallback is now scoped to a single fit
  (restored on return), while a mode set explicitly via `set_gradient_mode()` is
  preserved.

* **`find_reasonable_epsilon` ignored the active integrator (#227).** The
  warmup step-size seed always used a first-order leapfrog step regardless of the
  selected scheme (yoshida4/6/8, minerror2, adaptive, mts). It now walks the
  same SIMP op sequence the trajectory integrator uses, so the seed epsilon
  matches the scheme's per-step energy error.

* **Diagnostic and marginal-SE test coverage (#224, #225).** Added known-answer
  tests for the generic `moran_i` / `durbin_watson` diagnostics (hand-computed,
  plus `spdep` / `lmtest` cross-checks where installed), and a correctness test
  pinning the continuous-spatial (NNGP) marginal fixed-effect SE against an exact
  dense-GP penalized-IRLS Schur reference.

* **Comment cleanup (#226).** Removed residual meta / status comments from the
  sampler and autodiff sources and repaired truncated comment fragments in the
  joint Pareto-k module.

## 0.0.90

Audit fixes (0.0.89 review, issues #207-#217).

* **Nested-Laplace crash on a length-1 multi-block prior (#207).** The outer
  Pareto-k diagnostic was invoked with `type = NULL` on the multi-block path;
  the single-block decline guard did not fire for a length-1 block list, so the
  `.NL_REGISTRY[[NULL]]` lookup errored. A single latent block wrapped in a list
  (the `tgmrf()` / custom-latent front door) or a single block with a
  model-supplied `likelihood` (the tulpaObs consumer path) now resolves the type
  from the block itself. Default-config crash on both paths.

* **Self-loop adjacency corrupted the CAR/ICAR precision (#208).** A user
  adjacency with a non-zero diagonal was fit on a self-referential `Q` after only
  a warning. `.validate_adjacency_arg()` now zeroes the diagonal (the graph is an
  off-diagonal adjacency everywhere downstream), and the CSR builder excludes a
  node from its own neighbour list as defence in depth.

* **`temporal_ar1(rho_prior = )` is now wired end to end (#209).** Supply a
  `prior_beta(alpha, beta)` to place a Beta prior on `u = (rho + 1)/2`. The
  compiled sampler kernel adds `a*log(u) + b*log(1-u)` (new ModelData
  `ar1_rho_prior_a` / `ar1_rho_prior_b`, ABI 35 -> 36) and the nested-Laplace
  outer grid is reweighted by the Beta density. Default `Beta(1, 1)` reproduces
  the previous `Uniform(-1, 1)`. Previously the argument was accepted and
  documented but silently ignored.

* **Nested-Laplace fit accessors (#210).** `ranef()` now grid-marginalizes the
  random-effect tail of a nested fit (was empty); `spatial_range()` /
  `temporal_corr()` summarize the spatial / temporal axes of a mixed or
  spatiotemporal nested fit instead of erroring on the `all(...)` type check; and
  `diagnostic_summary()` unwraps a `$joint_fit` wrapper like the shared
  reliability readers, so a model-package subclass surfaces its Pareto-k.

* **Input-robustness fixes (#211).** SVC / TVC `terms = ~ f` / character
  interfaces expand a factor covariate to its contrast columns (shared
  `.resolve_varying_coef_columns()`) instead of failing a bare-name match; the
  inline `temporal()` field enforces the RW2 >= 3 time-point guard;
  `moran_i()` drops self explicitly under coincident coordinates; and
  `check_diagnostics()` returns `NA` on a too-short chain rather than a spurious
  "checks passed".

* **C++ defensive fixes (#212).** `TapeScope` move-assignment re-points the
  active thread-local tape; a zero-byte file at a checkpoint path is treated as
  fresh rather than a bad-magic error.

* **GP / NNGP Laplace pinned to serial (#217).** Running the GP / NNGP Laplace
  kernel multi-threaded triggered a flaky heap corruption under the mingw OpenMP
  toolchain (a hard crash the second time it ran in a session). The
  observation-scatter is the only OpenMP region there and its speedup is
  negligible (small `n_spatial`, serial Vecchia prior scatter dominates), so the
  kernel now runs on one thread; the crash is eliminated.

* **Docs and internal single-sourcing (#213).** Removed the `tulpa_psis()`
  `tail_points` "(with a warning)" doc that never fired; corrected the default
  `phi` PC-prior description to match its behaviour (`phi` is the NB2 size, so
  the default keeps `phi` finite / allows overdispersion, with `phi -> Inf` the
  Poisson limit); `.tulpa_param_layout()` builds RE names through the single
  `.re_names_from_layout()` source; grammar fix in `priors_default()`.

* **Front-door hyperparameter recovery + CI coverage (#214).** New
  `test-hyperparameter-coverage.R` gates the hyperparameter posteriors of the
  nested-Laplace paths against simulated truth with >= 20-seed CI-coverage
  checks: temporal AR1 (rho, precision), proper CAR (sigma, rho), GP / NNGP
  (sigma, range), HSGP (sigma), BYM2 (total spatial SD), and the free-Sigma
  random-slope correlation. Previously only field shapes and fixed slopes were
  recovered; the variance / range / correlation hyperparameters had no coverage
  test.

* **Test quality (#215, #216).** The SVC / TVC exact-NUTS front door gained a
  divergence guard and scale recovery (the configuration the #144 divergence bug
  survived in). The debias is now shown as a differential: on small binary
  groups the exact Gibbs draw of Sigma lifts the under-dispersed nested sigma
  toward truth. `tgmrf_cpp()` recovery is checked across seeds (cpp == R to 1e-6
  per seed, inheriting the R closure's verified recovery). `test-gpu-nngp` was
  retitled to the path it exercises; `test-hsgp-recovery` renamed to
  `test-hsgp-density-identity` (it is an analytic identity, not a fit); the
  leapfrog-drift baseline now fails loudly if missing; the inference-tier test
  checks every backend's tier; and the native Rhat/ESS/MCSE reference tolerance
  was tightened from 1e-4 to 1e-12 (the estimators match `posterior::` to
  ~8e-16).

## 0.0.89

Audit fixes (0.0.88 review, issues #193-#206).

* **Field-aware marginal SE for intrinsic areal fields (#196).** ICAR / CAR /
  BYM2 conditional-Laplace fits computed `H_beta` in the dense-Hessian branch
  that ignores the spatial field, so `summary()` / `vcov()` / `confint()`
  reported fixed-effect SEs at the wrong linear predictor and without
  marginalizing the field. Added `.marginal_H_beta_icar` / `_bym2` (field
  precision `L + 11'`; the BYM2 two-block convolution), validated against an
  independent penalized-IRLS reference (`test-marginal-se-areal.R`).
* **Front-door `control` knobs no longer dropped (#194).** The spde /
  re_cov_nested / re_cov_gibbs / gibbs branches hand-built the inner `control`
  list, silently discarding valid knobs (`diagnose_k`, `k_samples`,
  `checkpoint`, `n_threads`, ...); they now forward the validated subset.
* **Random-slope guard single-sourced (#195).** A no-intercept single slope
  `(0 + x | g)` was silently fit as a random intercept on the group-index-only
  paths; `.is_scalar_re_intercept()` now gates AGQ / Gibbs / SPDE / nested-RE
  consistently. Plain Laplace still fits it (carries the slope column as Z).
* **ESS sampler bounds (#193, #201).** `n_save` floored while the store loop
  fires `ceil(post / n_thin)` times (out-of-bounds write); multi-term RE with
  distinct sigma now hard-errors instead of freezing the extra terms.
* **Spatiotemporal summary labels (#197).** The `format = "summary"` (s, t)
  index was t-fastest while draws are s-fastest, mispairing labels when S != T.
* **Nested-Laplace diagnostics (#198, #203, #204).** `tulpa_hyper_grid`
  refreshes the per-cell log-prior on the refined grid (was stale-length -> NA
  into power-scaling); the single-block outer Pareto-k is computed on the
  default grid (not only when the user named it); a nested-prior EM block
  attaches a grid-marginalized `H_beta` so the MI / Gibbs correction reports a
  real pooled SE instead of NaN.
* **Cross-tier hyperparameter summaries (#199).** `spatial_range()` /
  `temporal_corr()` returned raw grid axes (tau, phi_gp, sigma2) on a
  nested-Laplace fit but interpretable range / sigma / rho on a sampler fit. The
  nested path now maps each axis to the same interpretable quantity, computed
  per grid cell then weighted-summarized (sigma = 1/sqrt(tau) or sqrt(sigma2);
  range = 3 * lengthscale).
* **Generic diagnostics (#200, #205).** `plot_pairs()` selects fixed effects
  from `fit$fixed_names` (was hard-coded to ratio's `beta_num` / `beta_denom`);
  `plot_diagnostics()` guards an all-NA Rhat; NNGP neighbour builders no longer
  index a non-existent row at `N == 1`; `ranef()` no longer emits the field /
  hyperparameter tail as random effects; `plot_acf()` / `geweke_test()` handle a
  3-D `[iter, chain, param]` fit; `(x - 1 | g)` drops the intercept;
  `spatial_multiscale()` `sampler` narrowed to the documented modes.
* **Cleanup (#206).** Issue tokens and `dev_notes/` pointers removed from
  user-facing `stop()` messages and comments, refactor-history narrative
  reworded to describe current behavior, a dead no-op and a duplicated MCSE body
  removed, and the multi-block CAR_proper log-det cache made cell-keyed
  (`NlCellCache`) to match the single-block path.

## 0.0.88

* **SoftAbs divergence-retry invariance test (#189).** The post-warmup SoftAbs
  retry re-runs a diverged NUTS trajectory under a frozen Hessian-based metric
  and takes that proposal instead -- a state-dependent kernel mixture whose
  target-invariance was argued but never tested. A new test-only entry point
  (`cpp_test_funnel_nuts`, `src/tulpa_test_funnel.cpp`) fits Neal's funnel
  through the exact production NUTS path with the retry forced on or off, and
  `test-softabs-retry-invariance.R` checks that toggling the retry on removes
  divergences without shifting the `v` marginal (seed-averaged paired
  mean/sd/tail differences within tolerance over 10 seeds). Verdict:
  invariance holds empirically -- 24-seed paired |t| < 0.4 on every summary,
  divergences 735 -> 8, no posterior shift.

## 0.0.87

Bug fixes and cleanups from a second whole-repo audit (issues #176-192).

* **Cyclic RW1 / RW2 rank normalizer on the exact-NUTS, multiscale, TVC, and
  spatiotemporal paths (#176).** The production templated log-posterior used
  `rank = T` in the cyclic branch; a cycle-graph intrinsic GMRF has a single
  null direction, so the rank is `T-1` for both RW1 and RW2. The 0.0.86 fix had
  landed only on the double-precision twin. The rank is now single-sourced
  through `tulpa_temporal::rw1_rank` / `rw2_rank`, consumed by every site.
* **Cyclic RW2 now fits cyclically on the Laplace / nested-Laplace path (#177).**
  The RW2 kernels dropped the `cyclic` flag and the front door only propagated
  it for RW1, so `temporal_rw2(cyclic = TRUE)` silently fit a non-cyclic RW2.
  The dense / sparse / pattern kernels now add the two ring-closing second
  differences and the flag is threaded through the RW2 front door.
* **`car_proper` nested grid (#178).** Supplying only one grid axis no longer
  discards the other, and the correlation axis is accepted under both `rho_grid`
  and the joint-API `rho_car_grid` spelling.
* **Grid-checkpoint fingerprint folds the RE group assignment `re_idx` (#180)**
  so a resume onto a checkpoint written for a different grouping errors instead
  of loading stale cells.
* **`get_compute_layout_fn()` verifies the ABI version (#181)** like the sibling
  registered-callable getters, so an ABI-mismatched consumer gets a clear error
  rather than silent memory corruption.
* **tgmrf joint / outer NUTS count energy divergences (#182).** Divergences were
  only flagged on a gradient-resync failure; a genuine energy divergence now
  sets the flag, and the R adapter surfaces `divergent` / `n_divergent`.
* **`ranef()` on sampler-tier fits (#183)** returns exactly the random-effect
  coefficients (the `re[...]` draw columns), no longer re-including the latent
  field and a spurious `log_sigma_re` row.
* **BYM2 default scaling (#184).** `compute_bym2_scale()` now returns the
  Riebler et al. (2016) generalized-variance factor `1 / sqrt(geomean(diag(Q^+)))`
  instead of the geometric mean of the ICAR eigenvalues, so the spatial fraction
  `rho` stays interpretable across graphs.
* **`spatial_range()` / `temporal_corr()` (#186)** now summarise the outer
  hyperparameter grid on a nested-Laplace fit rather than erroring on the
  primary spatial / temporal fitting path.
* **`sigma_re` ignored-argument warning (#187)** also fires when the covariance
  backend is selected by name (`mode = "re_cov_*"`).
* **`tulpa_simulate(theta = <engine fit>)` (#188)** consumes a single-process
  matrix-draws fit instead of erroring.
* Gradient-mode documentation reconciled with the dispatcher: `A` / `A_t` alias
  the arena reverse-mode path rather than claiming separate kernels (#185).
* Single-sourced the duplicated NUTS trajectory loop (primary + SoftAbs retry)
  and aligned the SVC gradient Cholesky pivot floor with the value kernel (#190).
* Removed dead code, stale rename artifacts, and leftover issue-tracker / history
  comments; completed the ABI changelog (#191). Minor: `.default_tau_grid`
  argument cleanup and the spatiotemporal long-format s/t ordering (#192).

The nested-Laplace tensor-grid integrator was investigated (#179) and left
unchanged: CAR_proper `(tau, rho)` parameter recovery confirms the marginal is
already in the internal log-scale parameterization, so equal-weight grid
integration is correctly calibrated and adding a user-scale Jacobian biases the
scale posterior.

## 0.0.86

Bug fixes from a whole-repo audit.

* **Random slopes on the sampler modes now integrate the RE covariance.**
  `tulpa(y ~ (1 + x | g), mode = "mala" / "pathfinder" / "imh_laplace")`
  previously fell through the covariance redirect (gated on the Laplace backend)
  and fit the term with a single scalar `sigma_re` per block -- dropping the
  intercept/slope correlation, forcing the SDs equal, and conditioning at
  `sigma_re = 1`. These modes now route to the exact Metropolis-within-Gibbs
  `Sigma` debias, matching the documented coverage.
* **Cyclic RW1 / RW2 temporal (and spatiotemporal) rank normalizer.** The
  cyclic branch used `rank = T`; a cycle-graph GMRF has a single null direction
  (the constant), so the correct generalized-determinant rank is `T - 1` for
  both RW1 and RW2. The old value biased the integrated / sampled precision
  `tau` upward (oversmoothing), scaled by `S - 1` in Type-IV interactions.
* **Plain-ICAR exact-NUTS prior** now adds the soft sum-to-zero pin the BYM2 and
  Laplace paths already carry, so the constant field direction is no longer
  jointly unidentified with the intercept.
* **`compare_models()` weights.** The Akaike / pseudo-BMA weight on the elpd
  scale used `exp(0.5 * delta)`; corrected to `exp(delta)` (weights were
  systematically too uniform).
* **`check_model()`** asked for `residuals(type = "deviance")`, which is not a
  supported residual type and errored on the base fit; panel 2 now uses Pearson
  residuals.
* **Chain diagnostics on non-chain fits.** `geweke_test()` and `plot_acf()` now
  respect the draws-provenance gate (they returned a vacuous "converged" result
  on i.i.d. nested-Laplace / VI draws); `plot_pairs()`, `plot_divergences()`,
  and `plot_energy()` no longer error when `$backend` is `NULL`.
* **`spatial_range()` / `temporal_corr()`** named the quantile columns `q025` /
  `q975` regardless of `probs`; the columns are now derived from `probs`
  (e.g. `q2.5`, `q97.5`).
* **Non-finite fitting inputs.** `tulpa()` now rejects NA/NaN/Inf in the
  response or model matrix with the offending row, instead of letting a `NaN`
  propagate silently into the kernels (the model is built with `na.pass`).
* **`svc(approx = "nngp")`** now rejects duplicated coordinates (a distance-0
  neighbour gave a singular per-node covariance) rather than failing silently
  downstream.

Warnings and cleanups.

* **Silently-ignored arguments.** `tulpa()` now warns when `sigma_re` is passed
  for a random-slope model (the covariance is integrated, not conditioned on a
  scalar SD), and `spatial_multiscale()` surfaces its default range-prior bounds
  when they are left unset.
* **Dead code removed.** The unreachable non-centered AR1 gradient helpers (one
  carried a wrong logit Jacobian), five unused NUTS helpers, the unwired
  spatiotemporal-interaction helpers (`validate_spatiotemporal`,
  `prepare_spatiotemporal_for_hmc`, `build_st_index`, and the two precision
  builders -- superseded by `fit_st_nested()`), the internal
  `has_implicit_intercept()`, and a no-effect temporal-Gibbs selection gate were
  deleted; the proper-CAR `rho`-bounds eigenvalue roles were corrected (they were
  swapped, though the (0, 1) clamp masked it).
* **Minor consistency fixes.** `predict(type = "response")` documents that a
  binomial fit returns the per-trial probability (vs `fitted()`'s trial-scaled
  count); the k-quality band index now shares the reliability bands the
  band-confidence flag used (at the realised finite-draw count) rather than
  recomputing them at the draw budget; the fractional-SPDE rational-order default
  is single-sourced to 2; and two split-message errors and a couple of stale
  comments were tidied.
* **Duplicated computations single-sourced.** `spatial_range()` and
  `temporal_corr()` now share one hyperparameter-summary scaffold; the three RNG
  snapshot/restore helpers share one snapshot + restore-closure factory; the
  outer Pareto-k importance core delegates to the batched core (identical draws);
  and the temporal-GP PC prior routes through the shared `pc_prior.h` form
  instead of a hand-rolled Jacobian. The GP and SVC NNGP analytic gradients now
  share one Vecchia conditional-gradient assembler (`nngp_cond.h`; each keeps its
  own solver and distance source), and the fast leapfrog drift carries the same
  sparse-GMRF mass-block range override as `inv_mass_times_p()` so the integrator
  and U-turn check cannot use different metrics.
* **Meta-comments removed.** Issue-tracker references (`gcol33/tulpa#NNN` in all
  forms, parenthetical `(#NNN)` lists, and bare issue numbers) and stale version
  tokens ("Phase 1.3") were stripped from code comments and roxygen, keeping the
  domain rationale. References inside error messages, `test_that()` labels, and
  design-principle / improvement enumerations (`principle #5`, `improvement #1`)
  were preserved.

## 0.0.85

Per-block quadrature order and an optional variance-component prior on the AGHQ
path.

* **`tulpa_re_aghq(n_quad = ...)`.** `n_quad` now accepts an integer vector of
  length `length(re_terms)` giving a per-block node count, alongside the existing
  single integer broadcast to every covariance block. The tensor grid then uses
  `n_quad[b]` nodes along every dimension of block `b`
  (`prod_b n_quad[b]^(dim_b)` total nodes); a scalar reproduces the uniform grid
  exactly (byte-identical). Per-block orders let a heterogeneous stack spend fewer
  nodes on cheap scalar nuisance blocks (a dispersion or zero-inflation random
  effect) than on the correlated coefficient blocks. R-only, no ABI change.

* **`tulpa_re_aghq(sigma_prior = ...)`.** A Penalized-Complexity prior on the
  marginal standard deviations of one or more random-effect covariance blocks,
  added to the ML-II objective (and hence the marginal Hessian). `NULL` (default)
  is pure ML on the covariances, byte-identical to before. A `c(U, alpha)` pair
  (`P(sigma_i > U) = alpha`, the `re_cov_pc_lkj_prior()` convention) applies to
  every block, or `list(blocks = <indices>, prior_sigma = c(U, alpha))` to named
  blocks only. Reuses the exact PC log-prior + Jacobian of `re_cov_pc_lkj_prior()`
  (single source of truth), so the `+ log sigma` Jacobian repels `sigma -> 0` and
  the `- lambda sigma` term caps inflation. A weakly-identified variance component
  (e.g. a scalar dispersion / zero-inflation random effect at few groups) can
  drift to the boundary and flatten the marginal Hessian; a weak PC prior adds
  curvature there, keeping the joint optimum non-singular without materially
  shifting an identified fit. R-only, no ABI change.

## 0.0.84

Checkpoint fix (#161).

* **Diagnostic re-solve no longer aborts a resumable joint fit.** In a joint
  nested-Laplace fit, the outer Pareto-k diagnostic re-solves the inner marginal
  with its own cheaper solver knobs (`max_iter`, `tol`, `inner_refresh`). Those
  knobs are part of the checkpoint fingerprint, so with `control$checkpoint` set
  and `diagnose_k = TRUE` the diagnostic pass computed a fingerprint that did not
  match the file the main outer grid had written, and the fit stopped with a
  "fingerprint mismatch" error after the main grid had already completed. The
  diagnostic solve now runs checkpoint-free (via the same quiet-options path the
  CCD / adaptive probes already use), so only the main outer grid owns the
  checkpoint. A resumed fit stays byte-identical to an uninterrupted one; fits
  with `diagnose_k = FALSE` are unaffected.

## 0.0.83

Front-door API convention cleanup (#156, fully closed) and the
missing-front-door features (#158, fully closed). **ABI break**
(`TULPA_ABI_VERSION` 34 -> 35) for the proper-CAR exact-NUTS eigenvalue
log-determinant; downstream packages must rebuild.

* **Unified fixed-effect prior.** `beta_prior = list(mean, sd)` is now the one
  interface across `tulpa_ep`, `tulpa_multinomial`, `tulpa_ordinal`,
  `tulpa_gibbs`, `tulpa_gaussian`, `tulpa_re_cov_gibbs`, `tulpa_nuts_beta`, and
  `tulpa_nuts_spde`, replacing the four names (`beta_prior_sd`, `sigma_beta`,
  `prior_beta_sd`, `beta_prior_mean`/`_sd`) it had before.
* **Statistical hyperpriors leave `control`.** New statistical `re_prior =
  list()` argument on `tulpa()` carries the random-effect / variance-component
  hyperpriors (`prior_sigma`, `eta`, `prior_df`, `prior_scale`,
  `prior_sigma_scale`, `sigma_re_scale`); `sigma_beta` folds into `beta_prior`.
  `tulpa(control = list(prior_sigma = ))` now errors and points at `re_prior`.
* **Control doctrine.** `tulpa_gibbs`, `tulpa_gaussian`, `tulpa_nuts_beta`, and
  `tulpa_nuts_spde` move their perf/sampler knobs into `control = list()`;
  `tulpa_gibbs` threads `thin` and seeds via the session RNG.
* **Inference as an argument.** `tulpa_laplace_beta(mode = "nuts")` and
  `fit_spde(mode = "nuts")` delegate to the NUTS engines, folding the
  `tulpa_nuts_beta` / `tulpa_nuts_spde` verb variants behind a `mode=` switch.
* **Flagship input validation.** `tulpa_nested_laplace`, `tulpa_gibbs`, and the
  two `re_cov` fitters now check `nrow(X) == length(y)`, `n_trials` length, and
  `re_idx` / `group` range up front.
* **Hard-error silently ignored inputs.** `tulpa_em_laplace()` rejects the
  never-consumed `spatial=` / `re_list=`; `tulpa_re_cov_gibbs()` errors when a
  `prior_scale` matches no RE block; `plot.tulpa_st_summary()` implements
  `type = "spatial_map"` and errors on an unknown type.
* **EP is a registered backend** (#158): `tulpa(mode = "ep")` fits a
  fixed-effect GLM by Expectation Propagation (Tier 2 structured).
* **SPDE + random intercept** (#158): a single `(1 | g)` term now rides
  alongside a Matern SPDE field through `fit_spde()` and `tulpa()`
  (integer-nu, conditioned on `sigma_re`).
* **`predict()` kriging for continuous fields** (#158): `predict()` now
  interpolates the posterior-mean field to new coordinates for HSGP
  (`spatial_gp(approx = "hsgp")`) and GP/NNGP (`spatial_gp()`) fits, not just
  SPDE. Both are marginalised over the hyperparameter grid via new native
  kernels (`cpp_hsgp_field_predict`, `cpp_gp_field_predict`); held-out recovery
  cor > 0.9 against a known surface.
* **FIX**: a `tulpa()` GP/NNGP nested-Laplace fit aborted with "'a' is
  computationally singular" on ordinary data (Matrix's condition guard on the
  ill-conditioned joint precision); the fixed-effect marginal-Hessian solve now
  retries with a negligible jitter, so gp/nngp fits complete.
* **Tier-1 exact SPDE from the front door** (#158): `tulpa(spatial = <spde>,
  mode = "exact")` routes to NUTS over the Matern field + hyperparameters.
* **Spatiotemporal nested Laplace** (#158): new exported `fit_st_nested()` wires
  the `cpp_nested_laplace_st_{icar,bym2,car_proper}` kernels (additive areal
  field + rw1/rw2/ar1 temporal field, integrated jointly over the spatial /
  temporal precisions and the ar1 autocorrelation), previously reachable only
  from consumer packages. Recovers both fields (cor > 0.85) on simulated data.
* `latent_factor()` roxygen corrected: it is a ratio / multi-arm construct for
  consumer packages, not the single-response `tulpa()` front door (#158).
* **Continuous / areal fields from the exact-NUTS front door** (#158):
  `tulpa(spatial = <field>, mode = "exact")` now samples the latent field, its
  variance, and (where present) its range / correlation jointly with Tier-1
  NUTS for GP / NNGP (`spatial_gp()`), HSGP (`spatial_gp(approx = "hsgp")`), and
  proper CAR (`spatial_car_proper()`), joining the SPDE exact path. The generic
  ModelData sampler keys each block on `spatial_type`, so the field enters the
  parameter vector (previously only the fixed effects were sampled). The
  proper-CAR log-determinant `log|D - rho W|` is evaluated in the
  autodiff-friendly closed form `sum_i log(1 - rho mu_i)` from the precomputed
  adjacency eigenvalues, not a per-gradient Cholesky. Each field's NUTS
  posterior-mean recovers the simulated truth and matches its nested-Laplace
  counterpart (cor > 0.88).
* **Spatially- and temporally-varying coefficients from the front door** (#158):
  `tulpa(spatial = spatial_svc(...), mode = "exact")` and
  `tulpa(temporal = temporal_tvc(...), mode = "exact")` fit NNGP spatially-
  varying and RW1/RW2/AR1 temporally-varying coefficients, threaded into the
  generic sampler via new `svc_spec` / `tvc_spec` inputs. Exact-only: a nested /
  laplace / auto mode errors rather than silently dropping the varying field.
  Recovers the varying-coefficient surface / trajectory on simulated data
  (cor > 0.94).

## 0.0.82 (2026-07-15)

The sampled spatial range gets a real prior. **ABI break**
(`TULPA_ABI_VERSION` 33 -> 34); downstream packages must rebuild, and any
package placing a `gp()` / `svc()` NNGP block must now supply the range
anchors (see the last bullet).

* FIX (statistical): the GP and SVC NNGP paths placed a Uniform prior on the
  spatial range `phi` behind a hard `return -INFINITY` outside
  `(phi_prior_lower, phi_prior_upper)`. Two defects compounded
  (gcol33/tulpa#144). The rejection sits inside an autodiff log-posterior, so
  a step landing outside the box produced no usable gradient and NUTS reported
  it as a divergence; and the `+ log_phi` Jacobian made the flat density in the
  sampled `log_phi` a Uniform on `phi` itself, whose default `(0.01, 10)`
  carries mean ~5. Under a weakly informative binary likelihood the posterior
  collapsed onto that mean. Measured downstream on `occu() + svc()` with a
  truth of `phi = 0.25` on unit-square coordinates: 72-83% of post-warmup draws
  divergent and `phi` at ~4 across every seed (gcol33/tulpaObs#118). Both paths
  now sample `log_phi` unconstrained under a PC prior on the range, which is
  proper on `(0, inf)` and needs no bounding box -- the same shape the SPDE
  field has always used for `log_kappa`.

* FIX (statistical): the SVC NNGP block's half-Cauchy prior on the marginal SD
  was improper on the coordinate it is sampled on, so nothing bounded the SVC
  marginal SD from above. The density is written on `sigma`
  (`-log(1 + sigma^2 / scale^2)`) but carried to the sampled `log_sigma2` with
  that coordinate's *variance*-row Jacobian (a full `log_sigma2`) rather than
  its own (`-log(2) + 0.5 * log_sigma2`). The surplus `0.5 * log_sigma2` flattens
  the tail exactly: `log p` tends to a constant as `sigma2` grows, so the mass
  below a bound grows linearly in that bound instead of converging, and only the
  likelihood pulled `sigma` back. The GP block was never affected -- its base
  density (`log_prior_sigma2_pc_t`) is written on `sigma2`, so the full
  `log_sigma2` is the right Jacobian there. Both scale priors now come from
  named helpers in `pc_prior.h` (`log_prior_sigma_half_cauchy`,
  `log_prior_log_sigma2_half_cauchy`) rather than being spelled out at the
  sampling site, which is the failure mode that header exists to prevent.
  `test-pc-prior.R` asserts the density integrates to the half-Cauchy
  normalizer `scale * pi/2` and that doubling the integration bound leaves the
  accumulated mass unchanged.

* FIX: every NNGP fit whose locations carry replicates returned
  `H_beta = NULL`, so all fixed-effect standard errors were lost -- degraded to
  a warning rather than an error, and covered by a test that guarded its own
  assertions behind `if (!is.null(fit$H_beta))`, which passes precisely when
  the builder fails. The obs -> unit design read the map from `spatial_idx`,
  the field an *areal* spec carries; a GP spec calls it `obs_to_loc`. Absent,
  the map defaulted to `seq_len(n_obs)`, whose column index runs past
  `n_spatial` as soon as one location holds more than one observation. The
  helper now takes the map and the unit count from its caller, and rejects a
  map that is missing, short, or out of range instead of defaulting.

* FIX (statistical): `spatialRange()` reported the reciprocal of the range. It
  computed the effective range as `3 / phi`, commented as a decay parameter,
  but every kernel this engine ships is `exp(-d / phi)` (`cov_exponential`,
  `cov_matern32`, `cov_gaussian`), so `phi` is the range and the effective
  range is `-phi * log(0.05) ~= 3 * phi`. A fit at `phi = 0.25` was reported at
  12 rather than 0.75.

* FIX (statistical): the unwired `log_prior_phi_pc` set its rate to
  `-log(alpha) / U` while its own derivation, `exp(-rate / U) = alpha`, gives
  `-U * log(alpha)`, and it disagreed with both written PC range priors. It had
  no call sites, so it never reached a fit; it is removed rather than wired.
  `test-pc-prior.R` pins the contract it violated by integrating the density:
  `P(range < U) = alpha`.

* REFACTOR: the d = 2 PC range density now lives once, in `pc_prior.h`
  alongside the PC scale densities, and the SPDE hyper-prior consumes it
  instead of restating the closed form. `test-pc-prior.R` checks it against
  `pc_prior_log_density()`, the independently written R twin the SPDE nested
  path integrates against. The stale block comment in `tulpa_priors_spde.h`
  still documenting the superseded d = 1 form is corrected.

* REFACTOR: the logit-bounded parameterization (map + exact Jacobian) is
  extracted from the temporal GP path into `bounded_from_logit()` /
  `log_jacobian_bounded()` in `autodiff_utils.h`. Behaviour is unchanged; the
  temporal range keeps its Uniform prior, since moving it off Uniform needs the
  d = 1 PC density that `pc_prior.h` deliberately does not provide.

* BREAKING: `ModelData` carries `gp_phi_prior_U` / `gp_phi_prior_alpha` and
  `svc_phi_prior_U` / `svc_phi_prior_alpha` (encoding `P(range < U) = alpha`)
  in place of the `_lower` / `_upper` bounds. They ship unset (`-1.0`) and
  `compute_param_layout()` errors once, before sampling, on any NNGP block that
  fails to set them -- matching the SPDE field's existing `prior_range`
  contract. The engine does not default them: a silent default is a prior, and
  an unanchored range under a weakly informative likelihood is precisely how
  `phi` came to sit at the old Uniform's mean. The SVC HSGP path is unaffected;
  it puts a `LogNormal(0, 1)` on an unbounded log-lengthscale and reads neither.

* The internal `tulpa_version()` reads DESCRIPTION via
  `utils::packageVersion()` instead of returning a version restated in C++. The
  literal dated to the first commit and had never been updated, so it disagreed
  with DESCRIPTION across the whole 0.0.x line; the test asserted the same
  literal, which is why nothing caught it. The C++ entry point is removed rather
  than kept in sync, and the test now reads DESCRIPTION directly.
  `TULPA_ABI_VERSION` remains a compiled constant by design: it describes the
  DLL a model package linked against, not the metadata beside it.

## 0.0.81 (2026-07-15)

Third deep-audit pass: statistical, memory-safety, and backend-consistency
fixes surfaced by a fan-out code audit.

* FIX (statistical): the GP NNGP autodiff kernel and its double twin
  conditioned the neighbour covariance differently -- the double copy added a
  `1e-8` nugget to every diagonal, the autodiff copy added it only to an
  already-degenerate pivot, so on well-conditioned data it added none at all.
  The analytic GP gradients are finite-differenced from the double copy, so the
  value and the gradient described different models on ordinary input, not just
  degenerate input. (`hmc_gp_autodiff.h` carried a "known heisenbug with
  autodiff" note.) Both twins now share one kernel and one pair of constants.
* FIX: the SVC NNGP double kernel and its autodiff twin ran different
  conditioning -- 1e-6 vs 1e-4. The 0.0.75 consolidation (#109) routed the double
  kernel through a `double`-only core the autodiff twin could not use, so the
  twin kept its own literals; and the function it consolidated has no callers,
  while the live SVC path runs the autodiff one. Both now read
  `tulpa_svc::kSvcJitter` / `kSvcVarFloor`, pinned at the live path's values, so
  behaviour on the reachable path is unchanged.
* The Vecchia/NNGP conditional (factor, krige, floor, accumulate) is now one
  templated kernel in `nngp_cond.h`, shared by the SVC and GP kernels on both the
  `double` and autodiff paths -- the previous shared core was `double`-only,
  which is precisely why the autodiff copies were hand-written and drifted. The
  per-kernel conditioning constants stay explicit arguments (the SVC kernel
  deliberately runs looser than the GP one), but a kernel can no longer differ
  from its own twin. New `test-nngp-twin.R` asserts each autodiff kernel,
  instantiated at `T = double`, returns its double twin's value -- including on
  near-duplicate coordinates, where the conditioning actually bites
  (gcol33/tulpa#142 A3).
* FIX (statistical): the GP NNGP gradient returned half the true derivative of
  the gaussian covariance with respect to the range -- `dcov_dphi` in
  `hmc_gp_gradients.h` dropped the factor of 2 in `k * 2*d^2/phi^3`. The SVC
  copy of the same function had it. A `spatial_gp(cov = "gaussian")` HMC/NUTS
  fit therefore explored the range on a mis-scaled gradient.
* FIX (statistical): neither copy implemented the spherical range-derivative, so
  `spatial_gp(cov = "spherical")` computed its covariance from the spherical
  kernel and its gradient from the exponential one -- the value and its
  derivative described different kernels. The spherical derivative is now
  implemented (it needs `sigma2`, since unlike the others it is not proportional
  to `k(d)`, which is why it had been left to fall through).
* `dcov_dphi` is single-sourced: the GP copy delegates to the canonical
  `tulpa_svc::dcov_dphi_svc`. New `test-cov-kernel.R` checks every covariance
  type's range-derivative against a numerical derivative of the value function
  it differentiates, so a copy cannot drift from its own kernel again
  (gcol33/tulpa#142 A5).
* FIX: two out-of-bounds accesses in the analytic temporal gradient kernels --
  `rw1_grad_w` read one past the end of `w` at a single time point, and
  `rw2_grad_w` wrote one past the end of its second-difference buffer for fewer
  than two. The sigma2-parameterized twins guarded both cases; the precision
  twins, written separately, did not. The RW2 and AR1 gradients are now thin
  wrappers over the canonical precision kernels (as RW1 already was), and the
  AR1 `rho` gradient floors only the dividing `1 - rho^2` so it stays finite at
  the stationarity boundary without biasing the stationary precision. New
  `test-temporal-grad-equiv.R` pins the wrappers against the canonical kernels
  and both against numerical derivatives of their priors -- neither header was
  reached from a compiled translation unit, so nothing would previously have
  caught a wrong wrapper (gcol33/tulpa#142 A7).
* FIX: `control$hessian` was parsed and `match.arg`-validated on the
  multi-block joint nested-Laplace path and then never passed to the kernel, so
  `"psd"` and `"fisher"` silently ran as LM / observed curvature. The
  single-block path forwarded it correctly, so the setting worked or not
  depending only on how many blocks the prior had. The seven `.cpp_joint_multi`
  call sites now go through one fit-scoped factory
  (`.joint_multi_call_factory`) that supplies the invariant arguments once, so
  a per-fit setting cannot be honoured at one solve and dropped at another; each
  site drops from ~19 arguments to a handful (gcol33/tulpa#142 A6).
* FIX: `temporal_ar1()` and `spatial()` silently accepted `shared = FALSE`
  while every sibling constructor warned about unshared confounding structure.
  The warning was copied into 12 constructors, two of which had also drifted to
  a shorter one-sentence form; all now call one `.warn_nonshared()` helper.
* FIX: `spatial_car()`, `spatial_bym2()` and `spatial()` validated a raw
  adjacency matrix only for squareness and exact symmetry -- strictly weaker
  than the validator `adjacency()` / `check_adjacency()` already use. A graph
  with self-loops, non-binary weights or isolated nodes was reported by
  `check_adjacency()` and accepted silently by the constructors, which then
  built an improper field. They now share `.validate_adjacency_arg()`, so the
  same graph reports the same way whichever door it enters. This also stops the
  constructors densifying a sparse graph to test symmetry (O(n^2) memory) and
  accepts float-rounded symmetric matrices that `check_adjacency()` accepts.
  New `test-field-constructor-shared.R` pins the cross-constructor consistency
  (gcol33/tulpa#142 A8).
* FIX: the Laplace binomial log-likelihood dropped the `lchoose(n, y)`
  normalizer, so it was not a true log-density: a binomial `logLik` / WAIC /
  cross-backend comparison was off by `sum(lchoose(n_i, y_i))` whenever
  `n > 1`, while the autodiff and GLMM-oracle paths (and `dbinom()`) kept it.
  The term is eta-independent, so no mode, gradient or normalized grid weight
  moves; only the reported absolute value changes. Bernoulli data (`n = 1`) is
  unaffected.
* New `test-family-cross-path.R` evaluates every per-family (loglik, grad,
  curvature) kernel maintained in parallel -- the Laplace/Newton dispatch, the
  compiled GLMM oracle, and the explicit triplets -- at a shared `(y, eta, phi)`
  and pins them against each other and against R's own densities. It guards the
  0.0.73 bug directly: `phi` is the residual SD in the Laplace kernels and the
  residual VARIANCE in the GLMM oracle, bridged only by convention in R, so the
  test asserts `glmm_elt(phi = s^2) == log_lik_for_family(phi = s)` and checks
  that passing the wrong scale is actually detectable. Adds the three probes
  that made this constructible (`cpp_family_terms`, `cpp_glmm_elt_terms`,
  `cpp_test_laplace_gaussian` -- the gaussian triplet carrying the phi
  convention previously had no callable surface) (gcol33/tulpa#142 A9).
* FIX (statistical): the PC prior on the TVC log-precision carried an excess
  `+2*log_tau`, tilting the prior by `tau^2` toward large precision (improper as
  `tau -> Inf`) and biasing time-varying-coefficient SDs low -- coefficients were
  over-smoothed toward constant. The identical error had been found and fixed in
  the spatiotemporal block but was never propagated to the TVC copy.
* FIX (statistical): the PC prior on the HSGP log-variance mixed two
  change-of-variables conventions and lost `+log(sigma)`, leaving the density
  flat (improper) at the origin and biasing the HSGP amplitude low. The same
  applied to the HSGP-ST block.
* The PC prior is now single-sourced for real: `pc_prior.h` is templated over the
  autodiff types and provides the density on every scale a sampler parameterizes
  (`sigma`, `log_sigma`, `sigma2`, `log_sigma2`, `tau`, `log_tau`), each derived
  once from the base exponential plus its Jacobian. Every site delegates. The
  header was previously `double`-only, which is why each autodiff path re-derived
  the algebra by hand -- and why two of them got it wrong. New
  `test-pc-prior-scales.R` pins each scale against the base density plus a
  numerical Jacobian and checks each integrates to one (gcol33/tulpa#142 A4).
* FIX (statistical): RW1/RW2 single-block Laplace deleted the field's level --
  the post-Newton centering discarded the field mean instead of folding it into
  the intercept, so an intrinsic temporal fit reported an intercept of ~0.
  `spec_inner_solve` now compensates process 0's intercept like the joint
  driver; RW1/RW2 intercept recovery tests added.
* FIX (statistical): the LKJ prior on correlated random slopes double-counted
  the Cholesky -> correlation Jacobian (effective LKJ(2.5) not LKJ(2)); removed
  the spurious term in `tulpa_priors_re.h` / `lkj_chol_helpers.h`. The test now
  cross-checks the Stan Cholesky lpdf plus an independent
  `det(R)^(eta-1) * |dR/draw|` pushforward.
* FIX (statistical): the temporal-GP non-centered branch added a spurious
  `z -> f` Jacobian (the NC target is `-0.5 z'z`); the spatiotemporal PC-prior
  Jacobian had the wrong sign for the log-precision parameterization; the SPDE
  range PC prior flipped its tail in the nested path and used the d=1 shape for a
  2-D field in the NUTS path. All corrected to the d=2 Fuglstad et al. (2019)
  form; the stale d=1 test reference is updated.
* FIX (statistical): temporal AR(1) under NUTS was restricted to `rho` in (0,1);
  it now maps to (-1,1) to match the documented Uniform(-1,1) prior and
  `build_ar1_precision`.
* FIX (statistical): the sampler ICAR/BYM2/ST rank normalizer counted a single
  graph component; it now counts connected components (`S - k`) like the Laplace
  path, via a shared `count_graph_components` helper (ABI 32 -> 33).
* FIX (statistical): the single-point NNGP Laplace attached observation `i` to
  field node `i`, mis-mapping and dropping observations when coordinates repeat
  (`n_spatial < N`); it now uses the per-observation `obs_to_loc` map.
* FIX (draws): BYM2 posterior draws constrained the unstructured `theta`
  component to sum-to-zero (it has a proper N(0,1) prior and no centerer);
  `tulpa_re_aghq()` reported `log_marginal` including the LKJ penalty; and a
  nested-Laplace mixture node with a failed Cholesky became a zero-variance
  point mass that deflated fixed-effect CIs. All corrected.
* FIX (memory / concurrency): the nested-Laplace per-cell block cache could
  collide worker threads onto one slot when `n_threads_outer` exceeded the
  environment thread count (e.g. `OMP_NUM_THREADS=1` with `n_threads_outer > 1`),
  corrupting the shared CHOLMOD factor; the outer width is now clamped. Also
  fixed: an out-of-bounds write in the NNGP non-centered forward, a missing
  exception barrier around a `tweedie` `Rcpp::stop` in a parallel region,
  per-thread reduction buffers allocated inside a parallel region, and an
  untrusted checkpoint record length driving a multi-GiB allocation.
* FIX (joint driver): local-CCD refinement combined with `store_Q` left the
  stored per-cell `Q` misaligned with the refined grid and crashed
  `tulpa_posterior_draws`; local-CCD is now skipped when `store_Q` is set. The
  `k_quality` grid-refinement verdict no longer over-claims on multi-block fits.
* CHANGE (tiers): `sgld`, `sghmc`, and (unadjusted) `mclmc` are reclassified
  from tier "exact" to "optimized" -- they carry discretization / minibatch
  bias, so `mcmc_diagnostics()` no longer certifies them as exact and auto-mode
  never selects them silently. `smc` stays exact.
* FIX (single-source): `print.tulpa_spatial` was defined twice (the SPDE copy
  shadowed the areal ICAR/CAR/BYM2 formatter); merged into one method. A dead
  duplicate `.with_preserved_seed` was removed.
* FIX (CRAN): `tulpaRatio` example blocks moved from `\donttest` to `\dontrun`;
  `tools` declared in `Imports`; the `ggplot2` vignette chunks gated on
  `requireNamespace()`.
* FIX (previously unwired PG kernels): the six binomial Polya-Gamma spatial /
  temporal Gibbs kernels (ICAR, GP, multiscale GP, temporal AR1, BYM2, RSR) --
  exported but not wired to a front door, and previously untested -- carried real
  statistical defects. Fixed and validated with a new parameter-recovery suite
  (`test-pg-spatial-recovery.R`): the ICAR tau shape now counts connected
  components and the field mean is absorbed into the intercept; the single- and
  multi-scale GP kernels use the NNGP-correct conjugate sigma2 and MH phi
  (shared `update_nngp_scale`, DRY) with the field level anchored (they diverged
  before -- sigma2 railed, phi did a data-free random walk); the temporal AR1
  conditional now uses both neighbours and rho is sampled (it was frozen at its
  init yet reported as a posterior); and the BYM2 component updates remove only
  the other component's contribution. The two-process negbin kernel now draws the
  exact real Polya-Gamma shape (`rpg_real`).

## 0.0.80 (2026-07-14)

CRAN-preparation release.

* FIX (statistical, Polya-Gamma Gibbs): the shared fixed-effect update drew
  from `N(m, (L'L)^{-1})` instead of `N(m, (LL')^{-1})` -- a forward
  substitution where the transpose back-substitution was required -- so every
  PG Gibbs backend (binomial ICAR/BYM2/RSR/GP/multiscale/temporal, negbin,
  the two-process negbin) sampled beta with the wrong covariance whenever
  `p >= 2` and the design was correlated. Means were unaffected. Draw
  covariance is now pinned against the asymptotic `glm` vcov in a regression
  test.
* FIX (statistical, Laplace SEs): the fixed-effect curvature `H_beta` (and
  the RE / spatial-field Schur complements) evaluated the GLM weights at
  `eta` WITHOUT the observation offset, so `vcov()` / SEs / `confint()`
  described a different model on any non-gaussian offset fit (poisson probe:
  link-scale SEs off ~2.3x). The offset now enters the Hessian eta on every
  path.
* FIX (statistical, negbin Gibbs): the PG augmentation `omega ~ PG(y + r,
  eta)` rounded its real shape to an integer, biasing the chain worst at
  zero counts with small `r`. New `rpg_real()` draws the exact real-shape
  Polya-Gamma (truncated sum-of-gammas with tail-mean correction). Same
  pass: the spatial negbin kernel's ICAR `tau` shape counts adjacency
  components, its sum-to-zero centering absorbs the field mean into the
  intercept (posterior-invariant), and all three negbin kernels share the
  exact half-Cauchy auxiliary scheme for `sigma_re`.
* FIX (concurrency): the hsgp / hsgp_mo / spde / nngp latent-block factories
  kept one shared `prep`-rebuilt cache; under the joint driver's parallel
  outer grid (`control$n_threads_outer > 1`) concurrent cells silently
  solved at each other's hyperparameters. Per-cell state is now published
  under the cell id (`NlCellCache`), safe under the coupled-cell scatter's
  work-stealing tasks; joint and single-arm car_proper use the same cache.
  Regression test: joint fit identical under `n_threads_outer = 2` and `1`.
* Robustness (C++): exception barriers on the nested-Laplace grid's OpenMP
  cell loops and the parallel NUTS chain loop (a worker exception was
  `std::terminate`); the remaining six registered C callables carry the shim
  exception guard; every OpenMP region is sized by an explicit
  `num_threads(...)` clause through the `OMP_THREAD_LIMIT`-aware clamp and
  the process-global `omp_set_num_threads()` mutations are gone; allocation
  sizes compute in `size_t`.
* API: every front door validates its `control` names against a canonical
  whitelist (a misspelled knob was a silent no-op) and `tulpa()` errors on
  stray `...` arguments (`familly = "poisson"` used to fit a gaussian). The
  joint fitter's `diagnose_draws` knob is renamed to `k_samples`, the one
  name every nested-Laplace fitter uses (hard rename, no alias).
* API: front-door SPDE fits answer `coef` / `summary` / `confint` / `vcov` /
  `predict` (the nested return now carries an anchored mode plus the
  field-marginal `H_beta`); direct `tulpa_nested_laplace()` /
  `tulpa_nested_laplace_joint()` fits slice `coef()` to the actual fixed
  effects instead of relabelling the full latent vector; `tulpa_gibbs()`
  returns a classed `tulpa_fit`; `fitted()` includes the offset and trial
  scaling so `y - fitted(fit)` equals `residuals(fit, "response")`;
  `tidy` / `glance` re-export the `generics` generics (no masking next to
  broom).
* The backend registry is the single source for per-backend family support
  (spde gained its gaussian entry; the agq entry declares its three
  families); `compare_models()` / `model_average()` on Laplace-tier fits are
  deterministic (the Gaussian-synthesis draws behind the reconstructed
  log-likelihood are seed-pinned and RNG-neutral).
* FIX (heap corruption): lazily-initialized `thread_local` C++ objects inside
  OpenMP parallel regions corrupt the process heap under the mingw toolchain.
  One call to the NNGP CG/PCG solver (`spatial_gp(solver = "cg")`) was enough
  to crash the R session minutes later (Windows fail-fast 0xC0000374/09), and
  the joint coupled-cell scatter / parallel-chain NC-GP storage carried the
  same pattern -- the long-standing "OpenMP joint-parallel test heap-corrupts
  under parallel testthat" instability. All four sites now use either
  function-local buffers or constant-initialized POD-pointer TLS (the safe
  pattern the autodiff arena already used). The full test suite completes in
  a single process again.
* FIX: inline `temporal(~ ... || time)` varying-coefficient fields reported
  empty per-time field means (`fit$temporal_fields[[...]]$mean` had length 0):
  the shared bar-field core read the joint layout's `field_starts`, which
  registers only areal block types. It now reads the per-block `block_start`,
  which is aligned for every block type.
* Documentation: every exported Rd documents its return value; stale ratio-era
  examples (two-arm formulas with `tulpa_poisson_gamma()` / `tulpa_binomial()`
  and pre-`control` arguments) are rewritten as runnable single-response
  engine examples (plot_rhat, spatial_bym2, spatial_hsgp, spatial_rsr,
  temporal_rw2, temporal_ar1) or marked `\dontrun` with their tulpaRatio
  provenance; all 318 Rd example sets now run clean including `\donttest`.
  Slow-but-runnable examples moved from `\dontrun` to `\donttest` (`\dontrun`
  kept only on experimental-path examples); new runnable examples for the core verbs
  (`tulpa_criteria()`, `compare_models()`, `model_average()`,
  `mcmc_diagnostics()`, `tidy()`, `glance()`, `ranef()`, `moran_i()`,
  `posterior_sample()`, `tulpa_em_laplace()`); R sources and Rd are
  ASCII-clean; plumbing exports (`ccd_grid()`, `hyper_axis_spec()`, `sn_*()`,
  `findbars()`, ...) are `@keywords internal`.
* New generics: `residuals.tulpa_fit` (`"pearson"` / `"response"`, computed
  from the family registry) and `nobs.tulpa_fit`; `print.tulpa_fit` rewritten
  to be shape-robust across the sampler / Laplace / nested tiers (it printed
  Gaussian-only fields before).
* WAIC / LOO on engine fits: when a fit does not carry `$draws$log_lik`, the
  pointwise log-likelihood is computed from the linear-predictor posterior
  draws and the family registry, so `compare_models()`, `model_average()` and
  the criteria layer work on `tulpa()` fits from any tier.
* Simulation diagnostics (`pit_residuals()`, `test_dispersion()`,
  `test_outliers()`, `test_zero_inflation()`, ...) read the engine fit's
  stored response (`$y`); the default residual type for `moran_i()` /
  `durbin_watson()` is `"pearson"`, matching the new `residuals()` method.
* RNG hygiene: user-supplied seeds are applied through a scoped helper that
  restores `.Random.seed` when the fitter exits (9 fitters plus `bayes_R2()`),
  so a seeded fit no longer clobbers the session RNG stream.
* `control$re_cov = "aghq"`: random-slope fits can route the nested `Sigma`
  integrator through the AGHQ inner marginal (`control$n_quad`, default 9
  there).
* `tulpa_re_aghq()`: `maxit` renamed `max_iter` (hard rename, no shim). The
  EM drivers report progress via `message()` (suppressible); `tulpa_gibbs()`
  is quiet by default.
* Every `extern "C"` C-callable shim (the model-package ABI) now catches C++
  exceptions at the boundary and converts them to R errors instead of
  unwinding across the C ABI.
* OpenMP team sizing honors `OMP_THREAD_LIMIT` via the shared
  `tulpa_omp_team_size()` helper (CRAN 2-core test compliance).
* DESCRIPTION: `Remotes` dropped (tulpaMesh is on CRAN), Description expanded
  with method references (DOIs), `cph` role added; new `inst/WORDLIST`,
  `inst/COPYRIGHTS`, and a `cleanup` script; `inst/CITATION` reads
  `meta$Version`. Vendored `src/simp/` headers carry license lines.
* Test suite: heavy fit / sampler blocks are `skip_on_cran()`-gated; vignette
  sampler chunks shrunk to CRAN-friendly iteration counts.

## 0.0.79 (2026-07-12)

* `integration = "grid_adaptive"` now declines to the dense tensor BEFORE any
  inner solve when the outer grid is small (fewer than `control$adaptive_grid_min_cells`
  cells, default 48), instead of paying a coarse-seed pass first. On a small
  tensor the coarse seed is already most of the grid, so there is no tail worth
  skipping; the guard means the adaptive integrator can never be slower than the
  dense tensor on a small grid. Measured on the real MOTIVATE 25 km shared-trend
  fit (a 22-cell outer grid), `grid_adaptive` and `grid` are now the same fit at
  the same wall-clock. The adaptive integrator earns its speedup only on large
  outer grids whose hyperparameter posterior concentrates.

## 0.0.78 (2026-07-12)

* New outer-grid integrator `integration = "grid_adaptive"` for the multi-block
  joint nested-Laplace driver, the low-dimensional companion to the CCD. It seeds
  a coarse subsample of the hyperparameter tensor lattice (latent block axes and
  phi axes together), floods outward from the posterior mode on the fine lattice,
  and evaluates only the cells within a log-density cutoff of the peak. The kept
  cells are a strict, uniform-weight subset of the dense tensor, so the posterior
  matches the dense grid to that cutoff (each omitted cell carries dense-grid
  weight `< exp(-cutoff)`) while evaluating far fewer inner solves when the
  hyperparameter posterior concentrates -- the fine-grid regime where the mass
  sits in a handful of cells. On a diffuse posterior it declines back to the dense
  tensor after only the coarse seed, so it never costs accuracy. Tuned by
  `control$adaptive_grid_cutoff` / `adaptive_grid_stride` / `adaptive_grid_max_frac`;
  the kept-cell / dense / solve counts are returned as `$adaptive_grid_info`. The
  main kernel call re-evaluates the selected cells with the full contract
  (`store_Q`, phi tensor, tile warm start), so the fixed-effect and per-cell
  posteriors are unchanged from the dense path. Fills the gap the CCD leaves at
  `1 <= d <= 3` latent axes, where the CCD mode-find is not worth it but a dense
  tensor still pays a full inner solve for every near-zero-weight cell.

## 0.0.77 (2026-07-10)

* The outer-grid thread memory clamp in the sparse joint nested-Laplace driver
  now budgets against the memory that is actually **free**, not the installed
  total. The previous clamp (tulpa#64) reserved half of total RAM for the
  replicated per-thread working set, which over-provisions on a loaded machine:
  half of a 64 GB install is 32 GB even when only 30 GB is free and the rest is
  already committed, so a wide fit (a fine-resolution SPDE field at many outer
  threads) still spilled into swap or ran out of memory. The clamp now sizes the
  per-thread pool at a safety fraction (0.6) of currently-available RAM, leaving
  the remainder for the grid results, CHOLMOD fill-in, and OS headroom, and
  falls back to half of total (then a fixed 2 GB) only when the platform memory
  query is unavailable. The budget arithmetic is factored into `mem_budget.h`
  (`outer_thread_mem_budget()`, `outer_thread_cap()`), single-sourced and
  unit-tested.
* The per-thread footprint the clamp budgets is now **measured**, not guessed.
  The dominant term is the CHOLMOD factor each concurrent solve carries, and its
  fill-in on a 2D SPDE mesh is superlinear, so the previous flat "2x nnz(Q)"
  allowance under-counted a fine field (3.9x low at a 16x16 lattice, widening to
  7x low by 96x96 -- and worse still at production resolution). The clamp now
  runs a one-time supernodal symbolic analysis of the joint Hessian pattern (the
  same `as_cholmod` view and supernodal `cholmod_common` the inner solve uses)
  and reads the factor's true `L->xsize`, via the new
  `SparseCholeskySolver::analyzed_factor_bytes()`; the analysis is symbolic (no
  numeric values, runs once in serial setup) and falls back to the old estimate
  only if it produces no factor. The O(n_x) Newton scratch is now counted too.
* When the clamp has to reduce the requested outer thread count it now emits an
  R **warning** naming the old and new thread counts and the memory budget, so a
  memory-driven slowdown is visible rather than silent. In the floor case -- a
  single cell's working set larger than the whole budget -- it warns that the
  fit runs best-effort at one thread and may page or run out of memory, and
  points at the real remedies (a coarser grid/resolution, freeing RAM, or
  `control$checkpoint` to make the run resumable) rather than crashing.
* New platform memory queries `available_ram_bytes()` (Windows `ullAvailPhys`,
  Linux `/proc/meminfo` `MemAvailable`, macOS Mach free + inactive pages)
  alongside the existing `total_ram_bytes()`, exposed to R for diagnostics as
  `cpp_available_ram_bytes()` / `cpp_total_ram_bytes()`.

## 0.0.76 (2026-07-10)

* The coupled-cell scatter in the sparse joint nested-Laplace driver (the
  `occu_cover()` hot loop, about 94% of runtime on a large fit) runs in parallel
  when the outer hyperparameter grid is under-saturated: the tail of any grid,
  small grids, or a many-core server where more outer threads are free than there
  are active grid cells. Those idle team threads steal per-cell scatter chunks,
  each accumulating into a private partial Hessian that is reduced in a fixed
  chunk order, so no grid cell's scatter runs single-threaded while freed cores
  sit idle. When the grid saturates the thread pool (the bulk of a large fit) the
  scatter stays serial and byte-identical to before; only the under-saturated
  cells chunk, and they stay within the thread-invariance tolerance (means agree
  to about 1e-14 and are reproducible run to run via the fixed-order reduce). Set
  `TULPA_GRID_WORKSTEAL=0` to force the serial scatter for exact reproducibility.

* The documented starting point for the PC prior on the copy coefficient
  `alpha` is now `c(U = 8.0, alpha = 0.01)`. A sweep over prior strengths on the
  alpha-recovery fixture showed the earlier `U = 2.0` recommendation over-shrinks
  the copy coefficient below its truth and, through the `alpha * sigma` copy
  axis, lifts the coupled donor amplitude `sigma` above its truth; `U = 8.0`
  regularizes the tail without that bias. The sigma-pos-prior recovery test
  asserts the retuned, measured behaviour.

## 0.0.75 (2026-07-08)

* SPDE field prediction standard errors (`predict(se.fit = TRUE)` with the field
  included) are computed by a streaming C++ kernel (`cpp_spde_field_se`) that
  factorizes the joint (beta, field) precision once and solves one query cell at
  a time. The dense working set is `O(p + n_mesh)`, independent of the number of
  query cells, so the per-cell SE over a large prediction grid holds a bounded
  amount of memory. Numerical results are identical.

* The sparse joint nested-Laplace driver shares one per-worker resource pool
  (Hessian builder, scatter cache, Newton scratch, arm specs, dense-basis
  buffers) across the cheap-screen and full-solve passes, which run in disjoint
  phases. This halves the pre-grid setup working set at a given
  `control$n_threads_outer`; fits are numerically identical.

## 0.0.74 (2026-07-07)

* `adjacency()` gains a settable neighbourhood. `order = k` extends the grid /
  raster stencil to the k-th ring: queen keeps every cell within Chebyshev
  distance `k` (`(2k+1)^2 - 1` neighbours: 8, 24, 48, ...), rook every cell
  within Manhattan distance `k` (`2k(k+1)`: 4, 12, 24, ...). The advanced
  `offsets` argument takes a custom stencil (a two-column integer matrix or a
  list of `c(dx, dy)` lattice offsets) for any anisotropic / off-axis
  neighbourhood; because an ICAR / CAR field is undirected, an asymmetric
  stencil is symmetrized to an undirected graph with a message. `order = 1`
  (default) and `offsets = NULL` reproduce the previous queen / rook graphs
  byte-for-byte.

## 0.0.73 (2026-07-07)

* Posterior prediction: `posterior_predict()` draws replicated responses from
  the posterior predictive (per-draw linear predictor -- fixed and random
  effects jointly from the draws, or the Gaussian approximation on the
  Laplace tier -- through new per-family sampling functions);
  `simulate.tulpa_fit()` is the base-R alias, and `pp_check()` falls back to
  generated replicates when the fit stores no `y_rep`.
* Observation weights: `tulpa(weights =)` scales each row's log-likelihood on
  the non-spatial Laplace path and the log-posterior samplers; other backends
  refuse loudly. A weight of 2 reproduces the duplicated-row fit exactly.
* Covariate smoothers: `y ~ s(x, k =, structure = "rw2"/"rw1")` puts an RW
  GMRF over the binned covariate and integrates its smoothness
  hyperparameter through the nested-Laplace temporal kernels (single block
  alone; the joint multi-block stack alongside an areal spatial field,
  temporal field, or a second smoother). `smooth_effects()` extracts the
  fitted smooth at the nodes.
* Second dispersion channel: `phi2` threads through the family registry, the
  compiled kernels, and every fitter surface. The Student-t degrees of
  freedom are now configurable (`family = "t"`, `phi2 = df`; default 4).
* Gaussian dispersion unified: `tulpa()`'s `phi` is the residual VARIANCE for
  gaussian/lognormal on every backend, as documented. Previously the
  compiled kernels (Laplace, ModelData samplers) read it as the residual SD,
  so `mode = "laplace"` and `mode = "mala"` fit different models at phi != 1;
  the conversion now happens once at each R-to-kernel boundary. The direct
  doors (`fit_spde()`, `tulpa_nested_laplace_joint()`, `tulpa_sample_glmm()`)
  keep their documented SD parameterization.
* New families: `lognormal` (variance convention, front-door fittable on the
  Laplace and sampler tiers) and `tweedie` (compound Poisson-gamma, `phi2` =
  power in (1, 2); Dunn-Smyth series density in R and the Laplace kernel,
  pinned against `tweedie::dtweedie`).
* Categorical responses through the front door: `tulpa(family =
  "multinomial" / "ordinal" / "ordinal_probit")` routes to the Laplace
  drivers; `tulpa_ordinal()` gains `link = "probit"` (pinned against
  `MASS::polr`).
* Model criticism: `bayes_R2()` (per-draw R^2 with model-based residual
  variance from new per-family variance functions); `tulpa_reloo()`
  (PSIS-LOO with exact refits at observations above the Pareto k-hat
  threshold, sharing the kfold refit machinery); `tulpa_kfold()` now threads
  stored `n_trials` / `weights` per training partition (previously a stored
  `n_trials` expression evaluated full-length against the subset data).
* Expectation Propagation returns its approximate marginal likelihood
  (`log_marginal`, exact for gaussian), so EP fits enter model comparison via
  `logLik()`.
* Power-scaling sensitivity gains a `hyperparameter` column: the
  nested-Laplace mixture paths record per-draw hyperparameter log-priors at
  draw-synthesis time and `tulpa_powerscale_sensitivity()` reweights them.
* SPDE kriging uncertainty: `predict(se.fit = TRUE)` with the field included
  propagates the joint (fixed-effect, field) posterior precision at the
  fitted `(range, sigma)`, including the cross term (integer-nu, no-RE fits).
* Temporal AR(p): `temporal_ar(time_idx, p =)` generalizes `temporal_ar2()`
  via the Levinson-Durbin PACF parameterization (always stationary) and the
  exact Yule-Walker banded precision.
* Boundary decision recorded: general censored/survival responses are
  observation processes owned by model packages via `LikelihoodSpec`; the
  engine keeps only the generic interval/truncated gaussian kernels.

## 0.0.72 (2026-07-07)

* New approximation-layer fitters:
  - `tulpa_ep()`: Expectation Propagation for GLMs with a Gaussian coefficient
    prior -- one Gaussian site per observation on the linear predictor, tilted
    moments by Gauss-Hermite quadrature. Exact for a Gaussian likelihood;
    matches marginal moments rather than mode curvature, so it is typically
    more accurate than Laplace on skewed GLM likelihoods.
  - `tulpa_multinomial()`: baseline-category (unordered K-class) multinomial
    logistic regression as a Laplace fit driven by the validated native kernel
    (`cpp_multinomial_logit_terms`) -- a joint Newton solve over the K-1
    coefficient blocks, no engine change.
  - `tulpa_ordinal()`: cumulative-logit / proportional-odds ordinal regression
    (L-BFGS mode + Laplace); cutpoint ordering is guaranteed by the
    log-increment reparameterization.
* `tulpa_kfold()`: refit-based K-fold cross-validation, the exact counterpart
  to PSIS-LOO in `tulpa_criteria()` for fits whose outer Pareto k-hat flags
  importance sampling as unreliable. Fixed-effect / GLMM fits only --
  subsetting observations would break a spatial or temporal field's structure,
  so those are rejected loudly.
* `tulpa_powerscale_sensitivity()`: power-scaling prior / likelihood
  sensitivity (Kallioinen et al. 2024) by importance-reweighting the existing
  draws through the native PSIS smoother -- no refits. The cumulative
  Jensen-Shannon distance and its gradient reproduce the priorsense reference
  implementation.
* `temporal_ar2()`: exact stationary AR(2) temporal GMRF as a user-defined
  latent block (`latent(temporal_ar2(time_idx))`) -- pentadiagonal precision
  from the Yule-Walker autocovariances, with the PACF parameterization mapping
  the open square bijectively onto the stationarity triangle.
* New builtin families `gamma`, `inverse_gaussian`, `beta_binomial`, and
  Student-`t` (fixed df = 4, matching the C++ `kStudentTDf`) in both the
  R-closure family ops and the C++ AD path, so they run under Laplace and NUTS
  alike (`test-family-ad-nuts.R`).
* SPDE: `fit_spde()` gains `family = "gaussian"` (continuous-field
  geostatistics; `phi` is the observation-noise SD), and `predict()` kriges a
  fitted Matern field to arbitrary `newdata` coordinates by re-projecting the
  posterior-mean mesh-node field through the spec's mesh (`include_field =
  FALSE` gives the fixed-effect / population prediction).
* API cleanup:
  - `fit_spde()`, `tulpa_re_cov_nested()`, and `tulpa_re_cov_gibbs()` move
    their perf / numerical knobs (`method`, `n_grid`, `integration`,
    `n_per_axis`, `span`, `n_draws`, `seed`, `max_iter`, `tol`, `n_threads`,
    `diagnose_k`, `k_samples`) into the single `control = list()`, matching
    the front-door convention. Statistical arguments stay in the signature.
  - snake_case renames (hard, no shims): `modelAverage` -> `model_average`,
    `postHocLM` -> `post_hoc_lm`, `spatialRange` -> `spatial_range`,
    `temporalCorr` -> `temporal_corr`. `getSVCSamples()` is removed -- use
    `svc(fit, summary = TRUE)`. The internal formula helpers
    `find_latent_terms`, `no_latent_terms`, and `parse_bar_term` are no longer
    exported.
* Internal reorganization: the nested-Laplace posterior-moment machinery moved
  to `R/nested_laplace_moments.R`, and the Polya-Gamma Gibbs dispatch plus the
  `tulpa_gibbs()` front door to `R/fit_gibbs.R`.

## 0.0.71 (2026-07-06)

* `cpp_laplace_fit_spatial()` and `cpp_laplace_fit_bym2()` gain a `force_sparse`
  argument (0 = size threshold, 1 = force sparse, -1 = force dense) threaded to
  the shared Newton solver. It lets the same problem run through both
  factorization paths for a byte-level dense-vs-sparse equivalence gate on the
  single-response path, the analogue of the joint path's `force_sparse` control.
  Default behaviour (0) is unchanged.
* Tests: a byte-level dense == sparse equivalence gate on the single-response
  ICAR / BYM2 Laplace path (`test-sparse-cholesky.R`), a `tulpa_pit()`
  calibration check (Uniform-under-correct-model, with a powered misspecification
  counter-case, `test-pit-calibration.R`), and a stored-draws reference for the
  default leapfrog integrator so a change to the stepper fails loudly rather than
  silently reproducing its own new draws (`test-integrator.R`,
  `tools/gen_leapfrog_ref.R`).

## 0.0.70 (2026-07-06)

* `tulpa_integrator("adaptive2")` / `"adaptive3"` cap the warmup-end curvature
  estimate at `DENSE_MAX_PARAMS` (200) parameters. Above that bound the
  step-adapted coefficient resolves at the well-adapted operating point
  (`nu_max = epsilon`, `omega_max = 1`) rather than building the dense `p x p`
  finite-difference Hessian, matching the cap the dense mass matrix already
  applies. A high-dimensional latent field (thousands of BYM2 / ICAR cells)
  selecting an adaptive integrator no longer allocates an O(p^2) Hessian or runs
  `p + 1` gradient sweeps at warmup end. Default leapfrog, the fixed schemes, and
  any model at or below the bound are unchanged.

## 0.0.69 (2026-07-06)

* `tulpa_integrator("adaptive2")` and `"adaptive3"` wire the SIMP step-adapted
  minimum-error integrators into NUTS. Each chain resolves its multistage
  coefficient at the end of warmup for its own operating point: the coefficient
  that minimizes the worst-case energy error over the band of dimensionless
  steps `(0, nu_max]` the chain actually takes. `nu_max = omega_max * eps`
  follows from the adapted mass matrix and the local posterior curvature
  (`omega_max` is the square root of the largest eigenvalue of `M^{-1}` times the
  curvature, found by power iteration on a warmup-end finite-difference Hessian),
  so the integrator is tuned to the target rather than to a fixed compromise --
  the nested-approximation-informs-the-sampler synthesis. The coefficient is
  per-chain (carried on the NUTS workspace, not a shared global); warmup runs a
  fixed placeholder of the same stage family so the dual-averaged step size
  transfers. Default leapfrog and the fixed schemes are byte-for-byte unchanged.

* `tulpa_integrator("mts", mts_substeps = )` adds a multiple-time-stepping
  (RESPA / Verlet-I) integrator to NUTS. The trajectory leaf splits the force
  into a stiff but cheap prior part (the Gaussian latent structure, taken with
  `mts_substeps` inner leapfrog substeps) and a smooth but expensive likelihood
  part (one full gradient per leaf, as leapfrog pays), so a larger outer step
  handles the stiff prior without evaluating the likelihood at the inner rate.
  The split reuses the existing `skip_obs_loop` decomposition: a new prior-only
  gradient path (arena-AD or central differences, folding in any model-package
  prior) supplies the fast force, and `grad_full - grad_prior` gives the slow
  (likelihood) force -- exact when the full and prior gradients are additive,
  which holds unless a model-package gradient hook alters the full gradient
  non-additively. The leaf stays symplectic and time reversible, so the U-turn /
  tree machinery is unchanged. Helps most when the latent field is stiff relative
  to a comparatively flat likelihood.

* Refreshed the vendored SIMP snapshot, which adds the step-adapted
  minimum-error integrators (an exact harmonic energy-error analysis picks the
  multistage coefficient for a target's step band, from `omega_max * eps`) and
  the parametric `two_stage` / `three_stage` constructors. These ship in
  `src/simp/`; the fixed `tulpa_integrator()` schemes are unchanged. The
  integrator decls now include only `simp/scheme.h`, so the Eigen-heavy headers
  do not enter every translation unit.

## 0.0.67 (2026-07-03)

* The SIMP integrator headers are now vendored into `src/simp/` (snapshot via
  `vendor_simp.sh`) instead of pulled in through `LinkingTo: SIMP`. tulpa builds
  self-contained -- no dependency on a non-CRAN package and no
  `Additional_repositories` -- while SIMP (gcol33/SIMP) stays the upstream
  development home for the integrator core. Re-run `vendor_simp.sh` to refresh
  the snapshot after updating SIMP. No behaviour change: `tulpa_integrator()`
  and the schemes are identical.

## 0.0.66 (2026-07-03)

* `tulpa_integrator("minerror2")` selects the two-stage minimum-error
  integrator from SIMP 0.2.0. Its coefficient cancels the leading energy error
  on a Gaussian target, so near the mass-adapted optimum it conserves energy
  well and adapts to larger step sizes without the stability cliff of the
  high-order Yoshida schemes. It recovers with zero divergences on the
  linear-Gaussian recovery test and is the recommended advanced integrator for
  near-Gaussian posteriors. Requires `SIMP (>= 0.2.0)`.

## 0.0.65 (2026-07-03)

* The HMC / NUTS trajectory integrator is now backed by the SIMP symplectic
  integrator library (`LinkingTo: SIMP`). Both leapfrog steppers (the in-place
  NUTS step and the fixed-trajectory HMC step) walk a SIMP scheme's op
  sequence with tulpa's own fused mass-matrix kernels, so the integrator
  identity has a single source of truth while the hot path keeps its
  specialised drift kernels. The default, leapfrog, is byte-identical to the
  previous step (verified against the bit-for-bit chain checkpoint and the
  existing NUTS recovery / reproducibility tests).
* `tulpa_integrator()` selects the integrator process-wide: `"leapfrog"`
  (default) or the higher-order `"yoshida4"` / `"yoshida6"` / `"yoshida8"`,
  generated from leapfrog by SIMP's triple-jump composition. `"yoshida4"`
  samples reliably; the higher orders are experimental for NUTS (a sharp
  step-size stability threshold interacts poorly with dual-averaging
  adaptation). See `?tulpa_integrator` and `test-integrator.R`.

## 0.0.64 (2026-07-01)

* `tulpa_pit()` runs in C++ (`cpp_tulpa_pit`), and the leave-one-out PIT
  weighting is exposed as `cpp_psis_loo_pit` -- per-observation PSIS leave-one-out
  weights (reusing the deterministic PSIS core) applied to predictive-CDF limits,
  with the PSIS columns parallelised and the single uniform jitter drawn in index
  order. Both draw from R's RNG stream in the same order as their former R
  bodies, so results are byte-identical under a fixed seed (test-pit-cpp.R).

## 0.0.63 (2026-07-01)

* `tulpa_psis()` runs its deterministic core -- the Zhang-Stephens generalized-
  Pareto tail fit and the Pareto smoothing of the upper-tail log weights -- in a
  C++ kernel (`cpp_tulpa_psis`). The R helpers `.tulpa_gpd_fit()` /
  `.tulpa_qgpd()` are kept as the reference oracle (test-psis-cpp.R). The tail
  size (with its expert-control cap and warning) stays in R, and the bootstrap
  k-uncertainty still resamples with R's RNG, so results are unchanged and
  reproducible; each per-observation LOO fit and each bootstrap refit is now the
  C++ path. Byte-close to the former R body (~1e-12).

## 0.0.62 (2026-06-30)

* `tulpa_nested_laplace_joint()` gains `prior_phi`, a regularizing hyperprior on
  the per-arm dispersion axes declared through `phi_grid` (a Beta precision, a
  negbin size, a Gaussian residual SD). Mirrors `prior_sigma` / `prior_alpha`:
  `NULL` (flat over the phi grid, default), `list("pc.prec", c(U, alpha))`, or
  `list("half_normal", scale)`. A single spec re-weights every `phi_<arm>` axis
  by its density at the kernel-call boundary, so refinement and Pareto-k passes
  see the regularized posterior; with no `phi_grid` it is a no-op. Threads
  through the single- and multi-block paths (gcol33/tulpa#139).

## 0.0.61 (2026-06-23)

* New `control$local_ccd` refines a multi-block tensor outer grid with local
  central-composite-design node clouds (`R/nested_laplace_joint_ccd_local.R`):
  a few high-weight, mutually non-adjacent interior cells are each replaced by a
  small curvature-aware CCD design, so a coarse base grid resolves the
  sharply-peaked hyperparameter directions without the `k^d` tensor blow-up. The
  local curvature is a diagonal finite difference of the outer log-marginal over
  each cell's own grid neighbours -- no mode-find, only the off-centre nodes are
  new inner solves, warm-started from the cell's mode. Refined cells carry
  partition-of-unity design weights, so the total integration weight is conserved
  exactly (no double-count); the design scale is shrunk per cell so the cloud
  fits the cell's Voronoi box (the local-Gaussian mass beyond it belongs to the
  neighbouring cells, which carry their own mass). Engages only on the tensor
  path at `>= 4` transformable latent axes with no active `phi_grid` -- the
  regime where a uniformly fine tensor is `k^d`-expensive and grid densification
  is the wrong tool; below it, the tensor grid is already dense and boundary /
  interior grid refinement covers a too-narrow grid. The applied refinement is
  summarised on the result as `$local_ccd_info`.

* `control$k_refine` gains a `"ccd"` rung alongside `"grid"`. Under
  `k_quality = "ok"` / `"good"`, a bad outer Pareto-k-hat now escalates by
  refining high-weight cells with local CCD node clouds (forcing a tensor base so
  the curvature stencil has axis neighbours), the right response when the grid is
  too coarse to resolve a sharp direction rather than too narrow at the boundary.
  Each round refines more cells; the verdict never silently downgrades and reports
  when local CCD finds no peaked interior cell to act on. Recovery, weight
  conservation, and the escalation path are pinned by
  `test-nested-laplace-joint-ccd-local.R`.

## 0.0.60 (2026-06-23)

* Joint nested-Laplace CCD outer integration is now robust to a sharply-peaked,
  ill-conditioned hyperparameter posterior -- the shape a joint occu_cover fit
  with an observation-arm random effect produces, where a narrow field-SD axis
  sits at a grid edge alongside a wide, weakly-identified axis. The CCD mode-find
  (`.joint_ccd_grid`) keeps the grid-median seed with a fixed finite-difference
  step as the default and, when that declines, falls forward to a rescue that
  warm-starts at the best latent grid cell (`.joint_ccd_grid_seed` evaluates the
  latent Cartesian grid in one batched call and takes the joint argmax) and uses
  a per-axis step calibrated to the local curvature (`.joint_ccd_calibrate_step`
  spans ~1 posterior sd per axis: small on a sharp axis, wide on a weakly-curved
  one, since one fixed step cannot resolve both). A ridge-safe coordinate-ascent
  seed (`.joint_ccd_pilot_seed`) is the fallback when the latent grid is too large
  to evaluate. Well-conditioned and sigma-alpha-ridge fits are unchanged (they
  engage on the default path); previously a sharply-peaked posterior declined to
  the full tensor grid. A recovery test pins the rescue.

* The CCD mode-find's backtracking line search now evaluates all `max_halve`
  step lengths in one batched call, so the candidates run across the outer-grid
  threads rather than one full-field inner solve at a time. On a large field (an
  expensive inner Laplace) the line search no longer serialises the mode-find;
  the accepted step is the same the sequential backtrack would take.

## 0.0.59 (2026-06-22)

* New `adjacency()` front door builds the symmetric graph that `spatial()` and
  `spatial_car()` consume, so areal models no longer need a hand-coded
  adjacency matrix. It is one generic that dispatches on the layout: a
  `data.frame` of cell centroids (queen / rook contiguity over the inferred
  lattice, no extra dependency), an `sf` polygon layer (shared-boundary
  contiguity via DE-9IM; queen = point or edge, rook = edge), and a raster
  (a terra `SpatRaster` or a `stars` object) whose non-`NA` cells become the
  nodes. It returns a printable `tulpa_adjacency` object carrying
  the sparse 0/1 matrix (`$adjacency`), the per-node cell identifier (`$ids`),
  the inferred cell size, and the node count -- the model still receives an
  explicit `graph = g$adjacency`, so the graph stays inspectable before
  fitting; nothing is guessed from coordinates silently.
* `node_index(graph, ids)` maps original cell identifiers to 1-based node
  indices by key, for remapping observation data (many rows per cell, different
  row order) onto the graph without assuming row alignment.
* `check_adjacency()` validates a hand-built matrix in one pass -- square,
  symmetric, zero diagonal, 0/1 valued, isolated nodes, and matching unique
  ids -- and `adjacency()` runs the same checks on the graphs it constructs.
* `spatial()`, `spatial_car()`, and `spatial_bym2()` now accept a
  `tulpa_adjacency` object directly (unwrapping its `$adjacency`), in addition
  to a bare matrix.

## 0.0.57 (2026-06-22)

* Joint nested-Laplace cell loop: a `CellCouplingSpec` can now declare, via the
  new `dense_cross_pairs(n_coupled, rank1_self_supported)` virtual, which
  `(kk, ll)` arm-pair cross-Hessian slabs it actually writes densely. The
  single-response cell loop allocates a dense `rc_kk * rc_ll` slab only for those
  pairs; every other pair keeps a `nullptr` buffer (the scatter already guards
  null). This bounds a cell with `J` observations on a self-coupled arm to `O(J)`
  rather than `O(J^2)`: a self block emitted through the rank-1 self-cross
  descriptor, or a cross a factorising likelihood never writes, no longer
  reserves a `J x J` slab. The default returns every pair, so a spec that does
  not override it is unchanged; the change is numerically inert for the
  `occu_cover` spec (the dropped slabs were allocated, zeroed, and never written
  even before). Removes a `std::bad_alloc` on grids with a very high-visit cell
  (e.g. an all-undetected cell with tens of thousands of plots).

## 0.0.56 (2026-06-21)

* `k_quality` escalation is now driven by adaptive integration-grid refinement
  rather than diagnostic-draw doubling (gcol33/tulpa#131). When the outer
  Pareto-k is bad the integration grid does not faithfully represent the
  hyperparameter posterior, so each escalation round now REFINES THE GRID
  (`adaptive_grid` boundary extension / interior densification, one more pass per
  round) and re-diagnoses, driven by the bad k, until the band is reached or the
  round budget is spent. Doubling `diagnose_draws` only re-scores the same grid,
  so it is no longer the escalation lever; `diagnose_draws` stays the separate
  knob that sharpens the k ESTIMATE (still auto-raised for `"ok"` / `"good"` so
  the bootstrap CI resolves).
* `control$k_refine` value `"mixture"` is renamed to `"grid"` (it refines the
  integration grid), and the default for `k_quality = "ok"` / `"good"` changes
  from `"none"` to `"grid"`: asking for a reliable band now chases it by
  refining. `k_refine = "none"` opts out (the band is reported but not chased).
* The #130 dispatcher invariant -- a grid-width deficiency stays unreliable and
  the reported k is never the moment-matched single Gaussian's optimistic value
  -- is now pinned by a direct unit test, not only indirectly via the band
  threshold.

## 0.0.55 (2026-06-21)

* The outer Pareto-k proposal refinement (moment matching, gcol33/tulpa#119) is
  allowed more passes. Proposal refinement is a separate step from the bare
  diagnostic and is not under the diagnostic's cost target, so the moment-matching
  cap `.K_DIAG_MM_MAX` is raised from `3` (a cost throttle added in #127) to `8`,
  a runaway-loop backstop rather than a budget. The loop self-limits well below the
  cap: it still stops as soon as the k-hat reaches the usable band, and now also
  stops once a refined pass no longer improves on the proposal it was estimated
  from, so the extra passes are spent only on a stubborn k that is still above the
  usable band and still falling. A fit that already fits, or one whose moment
  matching has plateaued, pays the same as before.

## 0.0.54 (2026-06-20)

* `control$k_quality` now climbs the reliability ladder (gcol33/tulpa#131). When
  an `"ok"` / `"good"` target is not confidently reached on the first fit, the
  engine escalates: it doubles `diagnose_draws` each round and, with the new
  `control$k_refine = "mixture"`, refines the integration grid (`adaptive_grid`)
  on the final round when more draws alone do not resolve the band, re-fitting and
  re-diagnosing up to `control$k_max_rounds` (default `2`) times. `k_quality_rounds`
  reports how many re-fits were used, and the verdict stays honest -- the request
  is a target, not a promise. The k_quality verdict (`k_quality_requested` /
  `reached` / `best` / `reason` / `rounds`) is now attached for BOTH the single-
  and multi-block paths (previously single-block only). The single fit is factored
  into an internal engine driven by the escalation front door, with no change to
  the fit itself for the default `k_quality = "report"`.

## 0.0.53 (2026-06-20)

* Fixed: the outer Pareto-k no longer under-reports when the hyperparameter
  posterior is heavier or wider than the integration grid (gcol33/tulpa#130). The
  moment-matching refinement could widen the single-Gaussian proposal past the
  grid and report a deceptively low k -- a target the grid cannot cover read
  ~0.39 ("good") instead of ~0.76 ("unreliable"), exactly the grid-width
  deficiency the diagnostic exists to flag. The dispatcher now compares the grid
  mixture (the faithful within-grid proposal) against the GRID-MOMENT k rather
  than the moment-matching-refined one, so a refinement that lowers k only by
  escaping the grid can no longer win. A near-collapsed grid, where the mixture is
  degenerate and does not improve on the grid-moment proposal, still keeps the
  moment-matched Gaussian, so the collapsed-grid path (gcol33/tulpa#117) is
  unaffected. The skipped bad-case recovery test is restored.

## 0.0.52 (2026-06-20)

* New `control$k_quality` reliability front door for the joint nested-Laplace
  outer Pareto-k (gcol33/tulpa#129). A single statement of the reliability the fit
  should report: `"report"` (default) computes the diagnostic and reports the
  achieved band; `"ok"` / `"good"` name a target band (the k-hat confidently
  usable, resp. good) and raise the default `diagnose_draws` (to `800L` / `2000L`,
  unless `diagnose_draws` / `k_samples` is set) so the bootstrap CI can resolve
  it; `"none"` disables the diagnostic. The fit carries an honest verdict --
  `k_quality_requested`, `k_quality_reached`, `k_quality_best`, `k_quality_reason`
  -- and never silently downgrades: if the requested band is not confidently met
  it reports the band actually reached and why. The adaptive draw-escalation loop
  and the integration-refinement rung (`k_refine`) are tracked in
  gcol33/tulpa#131. The reliability vignette covers the new front door.

## 0.0.51 (2026-06-20)

* Outer Pareto-k reliability bands are now sample-size dependent (gcol33/tulpa#128).
  The usable upper boundary is `min(1 - 1/log10(S), 0.7)` for `S` importance draws
  (Vehtari et al. 2024; matches loo's `ps_khat_threshold`): about 0.565 at the
  small-S end (`S = 200`), reaching the fixed 0.7 cap only past `S` ~ 2154. The
  good cut stays at 0.5. `pareto_k_band_confident` now tests the bootstrap CI
  against `c(0.5, min(1 - 1/log10(S), 0.7))` at the realised draw count, so a
  k-hat near the upper band at a modest `diagnose_draws` is correctly read as
  not-yet-usable rather than acceptable. `control$k_conf_bands` defaults to `NULL`
  (the size-dependent bands); pass a strictly-increasing vector (e.g.
  `c(0.5, 0.7)`) to fix the boundaries. New vignette section and the helpers
  `.ps_khat_threshold` / `.ps_conf_bands`.
* Known issue (gcol33/tulpa#130): for a hyperparameter posterior heavier or wider
  than the integration grid, moment matching can widen the diagnostic proposal
  past the grid and under-report the outer Pareto-k (reading it usable when the
  grid is in fact too narrow). A fix that bounds the refinement to the grid is
  tracked; the bad-case recovery test is skipped meanwhile.

## 0.0.50 (2026-06-19)

* Joint nested-Laplace outer Pareto-k: replaced the adaptive batched reporting
  (#123/#124) with a single-batch + bootstrap uncertainty (gcol33/tulpa#127). The
  batched/adaptive design could not honour a sane cost budget: one importance
  batch costs about one fit (each draw is an off-grid Laplace re-solve), so
  growing the batch count to resolve a borderline k drove the diagnostic to 10-30x
  the fit. The chosen proposal is now scored ONCE over `control$diagnose_draws`
  importance draws, and the k-hat's sampling uncertainty is estimated by
  bootstrapping its raw importance log-ratios (`control$k_bootstrap` replicates,
  re-fitting the GPD tail; no new inner solves). New control / output:
    - `diagnose_draws` (default `500L`; legacy `k_samples` accepted as an alias) is
      the precision knob -- a tighter k needs MORE actual tail ratios, so raise it,
      NOT `k_bootstrap`. The bootstrap only quantifies how unstable the current
      estimate is; it cannot create tail information.
    - `pareto_k_se_boot`, `pareto_k_ci_low` / `pareto_k_ci_high` (2.5% / 97.5%
      bootstrap quantiles), `pareto_k_se_formula` (the closed-form GPD-shape MLE
      asymptotic SE `(1 + k) / sqrt(M)`, a cross-check), and
      `pareto_k_band_confident` (TRUE iff the bootstrap CI lies within one
      reliability band).
    - `k_tail_points` (default `NULL` = the automatic PSIS rule
      `ceil(min(0.2 N, 3 sqrt(N)))`) is an EXPERT tail-threshold control, capped at
      the 20%-of-draws ceiling with a warning; the used / requested counts are
      reported in `pareto_k_tail_points` / `pareto_k_tail_points_requested`.
    - `k_conf_bands` (default `c(0.5, 0.7)`) sets the reliability-band boundaries,
      with intervals `(-Inf, 0.5] (0.5, 0.7] (0.7, Inf)`.
    - `diagnose_cost_ratio` (and `diagnose_draws`) attached at the top level: the
      diagnostic's wall-clock cost relative to the fit it certifies.
  The per-arm k (`diagnose_k = "by_arm"`) carries the matching per-arm fields. A
  borderline k is reported with its honest (wide) CI rather than chased to false
  precision; for a tighter estimate, raise `diagnose_draws`. `tulpa_psis()` gains
  a `tail_points` argument and returns `tail_len`.
* New vignette `reliability-pareto-k`: what the outer Pareto-k certifies (the
  integration, not the posterior), the reliability bands and their
  draw-count-dependent usable boundary `min(1 - 1/log10(S), 0.7)`, the bootstrap
  band-confidence flag, and the reliability ladder from reporting to debiasing.

## 0.0.49 (2026-06-19)

* Joint nested-Laplace outer Pareto-k: adaptive batched reporting with a proper
  Monte Carlo standard error and a band-resolution flag (gcol33/tulpa#124). The
  true outer k-hat is a single fixed number for a given fit and proposal, so all
  batch-to-batch variation is estimator sampling error -- min/max conflates a
  good/bad seed with too few iterations and WIDENS with the batch count `B`, so it
  cannot signal convergence. The batched diagnostic (`control$k_batches > 1`) now
  reports, alongside the median `pareto_k`:
    - `pareto_k_mcse` = `sd(batch k-hats) / sqrt(B)`, the Monte Carlo standard
      error of the estimate (shrinks as `1/sqrt(B)`);
    - `pareto_k_band_confident` = `TRUE` iff `pareto_k +/- 2 * pareto_k_mcse` lies
      within one reliability band (does not cross 0.5 or 0.7);
    - `pareto_k_n_batches` and the secondary observed `pareto_k_lo` / `pareto_k_hi`
      range (kept as QA fields).
  The per-arm k (`diagnose_k = "by_arm"`) carries the same fields
  (`pareto_k_by_arm_mcse`, `pareto_k_by_arm_band_confident`), over the same number
  of batches the joint loop settled on.
* New opt-in adaptive mode `control$k_adapt = TRUE`: starting at `k_batches`
  batches (defaulting to `4L` rather than the off sentinel), the diagnostic adds
  batches until `pareto_k_band_confident` becomes `TRUE` or the
  `control$k_batches_max` cap (default `20L`) is reached. A fit on the wrong side
  of a band boundary keeps sampling until the good/ok/unreliable verdict resolves;
  a fit whose true k sits ON a boundary (whose interval always straddles) stops at
  the cap with the honest `band_confident = FALSE`, so the cap is both a cost
  bound and the correct classification for an on-the-line fit. The seed pool is
  drawn for the cap, so an adaptive run is a reproducible prefix of the full-cap
  run. `k_batches` must be `>= 2` when `k_adapt = TRUE` (an MCSE needs at least
  two batches). The `1/sqrt(B)` rule reduces the seed-to-seed variance, not the
  GPD k-hat's small-sample bias (controlled by `k_samples`, kept `>= 200`).

## 0.0.48 (2026-06-19)

* Joint nested-Laplace outer Pareto-k: opt-in batched reporting
  (gcol33/tulpa#123). The outer k-hat is a noisy estimator -- a generalized
  Pareto shape fit to the upper tail of one batch of importance weights, with
  real Monte Carlo error that shrinks only slowly with the draw count -- so a
  single reported value is seed-dependent and can mislead near a reliability-band
  boundary (`< 0.5` good, `0.5-0.7` ok, `>= 0.7` unreliable). The new
  `control$k_batches` (default `1L` = OFF, byte-identical to the prior
  single-value behaviour and cost) evaluates the CHOSEN proposal's k over that
  many independent importance batches and reports `pareto_k` as the MEDIAN plus
  the observed `pareto_k_lo` / `pareto_k_hi` range and `pareto_k_n_batches`; the
  reliability band is classified off the median. The opt-in per-arm k
  (gcol33/tulpa#120) is batched the same way (`pareto_k_by_arm_lo` /
  `pareto_k_by_arm_hi`). The proposal SELECTION (single Gaussian vs grid-mixture,
  gcol33/tulpa#121) is made once on the canonical pass so every batch scores the
  SAME proposal (no per-batch source flip); per-batch seeds are drawn up front
  from the restored RNG state, so the diagnostic stays reproducible and the fit's
  draws are bit-for-bit unchanged. The spread is the Monte Carlo uncertainty of
  the PSIS k-hat across independent importance samples -- NOT a posterior credible
  interval and NOT a coverage-calibrated CI; the per-cell estimates and
  coefficients do not move. Cost is `k_batches` times the scoring, gated behind
  the already opt-in / slow diagnostic; a handful of batches (5-10) gives an
  honest min/max range.

## 0.0.47 (2026-06-19)

* New built-in `truncated_gaussian` family (gcol33/tulpa#122): an
  upper-truncated Gaussian latent, the bounded-support sibling of the
  `lognormal` arm. On the natural scale it is an upper-truncated lognormal --
  a positive response known to lie below a ceiling, modelled as a Gaussian on
  the log-response conditioned on `log y <= u` for a per-row bound `u`
  (`+Inf` => no truncation). The likelihood is the Gaussian density divided by
  the retained mass `Phi((u - eta)/sigma)`, the continuous-density counterpart
  of the `interval_gaussian` (ordinal) family. It is log-concave in `eta` so
  the inner Newton needs no Fisher fallback, and reduces exactly to the
  `gaussian` arm as `u -> +Inf`. Wired through the joint nested-Laplace path
  via a per-arm `trunc_upper` ceiling (`src/laplace_family_link.h`,
  `src/laplace_builtin_family_spec.h`, `R/nested_laplace_joint_helpers.R`).
  FD gradient/Hessian, deep-truncation stability, the lognormal reduction, and
  truncated-normal moment identities are checked in
  `test-truncated-gaussian.R`.

* New baseline-category multinomial-logit kernel (`src/multinomial_logit.h`,
  gcol33/tulpaObs#106): a nominal (unordered) K-class likelihood with K-1
  coupled linear predictors sharing the softmax denominator. The per-observation
  negative Hessian is the full (K-1)x(K-1) multinomial information
  (`diag(p) - p p'`), positive semidefinite for any `eta`, so the inner Newton
  needs no Fisher fallback; the softmax is formed overflow-safe. This is the
  engine primitive backing tulpaObs's `occu_categorical()` positive arm. FD
  gradient/Hessian, the PSD data-free information identity, and overflow safety
  are checked in `test-multinomial-logit.R`.

## 0.0.46 (2026-06-18)

* Joint nested-Laplace outer Pareto-k: grid-mixture (basin) importance proposal
  (gcol33/tulpa#121). The engine represents the hyperparameter posterior as the
  weighted integration grid and draws hyperparameters from it (a grid cell
  proportional to its weight, then that cell's latent Laplace), never from a
  single continuous Gaussian. The diagnostic, however, scored a single
  grid-moment Gaussian fit to the grid's mean and covariance. On a skewed or
  multi-node hyperparameter posterior, which the grid covers through its nodes,
  that symmetric Gaussian underweights the off-mode mass: importance draws
  landing there carry runaway weights and the k-hat reads unreliable even though
  the grid representation, and the per-cell estimates drawn from it, are fine.
  The spread-tensor `grid_moment` path now also scores the proposal the engine
  actually samples, a defensive mixture of local Gaussian bumps at the grid cells
  mixed by the grid weights (`proposal_source = "grid_mixture"`); each bump's
  per-axis SD is `getOption("tulpa.kdiag.mix_bw", 0.5)` times the largest grid gap
  on that axis. Because that mixture is confined to the grid, it is adopted only
  when the grid actually covers the posterior (the grid-moment Gaussian's
  importance weight stays inside the mixture's coverage hull) and it lowers the
  k-hat: a target tail beyond the grid keeps the single Gaussian's higher k so the
  grid-width deficiency is still flagged, a near-collapsed grid keeps the
  moment-matched Gaussian, and no fit that already read usable can regress. A true
  delta collapse still uses the finite-difference mode-Hessian and a supplied CCD
  mode-Hessian proposal still uses the single Gaussian, both unchanged. The fit,
  its per-cell estimates and coefficients are untouched; only the reliability
  diagnostic changes. On a real EVA-scale occu_cover fit whose occupancy field-SD
  posterior is right-skewed the reported k-hat goes from a seed lottery (0.10 to
  0.81, peaks flagged unreliable) to a stable usable band (max ~0.47 across seeds,
  median ~0.16) with a roughly two-fold higher importance-sampling effective size.

## 0.0.45 (2026-06-18)

* `tulpa_criteria()` gains a `group` argument that sets the LOO unit explicitly
  (gcol33/tulpa#118). A column of `log_lik` is the leave-one-out fold; with
  `group = NULL` (the default) the result is byte-identical to before
  (leave-one-row-out, e.g. per plot / per visit). When supplied, the per-draw
  pointwise log-likelihoods are summed within group to a `[n_draws x n_groups]`
  matrix *before* PSIS, so each fold is a whole group (leave-one-group-out
  cross-validation, LOGO-CV) -- e.g. leave out a whole cell rather than one of
  its rows. The grouping streams over the (possibly EVA-scale) input once and
  never materialises it; `lppd`, `p_waic`, `elpd_loo`, `cpo` and `pareto_k` are
  all computed on the grouped matrix and the standard-error multipliers follow
  the fold count. DIC is a plug-in deviance over all observations and is
  unaffected. The result reports `n_groups` and prints the fold count, and the
  pointwise data frame is keyed by `group`.

* Joint nested-Laplace outer Pareto-k: opt-in per-arm reporting
  (gcol33/tulpa#120). `control$diagnose_k = "by_arm"` computes, in addition to
  the single joint k over the whole hyperparameter posterior, a k-hat restricted
  to each arm's hyperparameter axes (the other arms held at their posterior
  mean), so a tail-heavy joint k can be localised to one arm rather than reported
  as one pooled number. Each axis is attributed to the arm(s) whose linear
  predictor it enters -- a latent block's axes to the arms the block loads on, a
  copy coefficient `alpha` to the recipient arm, a `phi_<arm>` dispersion axis to
  that arm -- reusing the same proposal-build + moment-matching + PSIS path as
  the joint k over the arm's axis subspace. Reported in `pareto_k_by_arm` (named
  by arm) with `pareto_k_by_arm_is_ess` / `pareto_k_by_arm_scope`, surfaced in
  `diagnostic_summary()`. OFF by default: the joint k stays the default behaviour
  and cost, and is bit-for-bit unchanged by the opt-in (it is scored first, and
  the diagnostic is RNG-restored). Defined for the multi-block layout with two or
  more arms; the single-block shared-field layout declines rather than
  mis-attribute its axes.

## 0.0.44 (2026-06-18)

* New built-in `interval_gaussian` family: an interval-censored Gaussian latent
  (an ordered probit with KNOWN thresholds). The latent value is
  `Normal(eta, sigma^2)` and the observation records only that it fell in the
  half-open interval `(lower, upper]` on the linear-predictor scale, with
  `-Inf` / `+Inf` the open outer classes. The log-density is the class
  probability MASS, `log(Phi((upper - eta)/sigma) - Phi((lower - eta)/sigma))`
  -- a genuine PMF over classes with no change-of-variable Jacobian, so the
  score is comparable across arms. The mass is differenced in the accurate tail
  to avoid catastrophic cancellation, and `P(eta)` is log-concave so the
  analytic `-d2 logP/d eta2 >= 0` needs no Fisher fallback. The family is read
  through `tulpa_nested_laplace_joint()` (`family = "interval_gaussian"`, with
  per-arm `lower` / `upper` in place of the point response `y`); it backs
  tulpaObs's `cover(positive = "ordinal")` Braun-Blanquet cover arm. The kernel
  is FD-gradient tested via the internal `cpp_interval_gaussian_terms()`
  (`test-interval-gaussian.R`).

* Joint nested-Laplace outer Pareto-k: the importance-sampling proposal now
  carries an optional moment-matching refinement (gcol33/tulpa#119, after
  Paananen, Piironen, Burkner & Vehtari 2021). When the integration grid is
  sharply concentrated the node-covariance proposal can mis-scale -- too wide
  scatters draws to extreme hyperparameters where the inner Laplace log-marginal
  inflates, too narrow leaves the target tail uncovered -- so the k-hat reads
  unreliable even on a fine fit. The proposal is now re-estimated from the
  PSIS-smoothed importance-weighted moments of its own draws and re-scored, up
  to a few passes, keeping the lowest-k-hat proposal and stopping early once the
  k-hat reaches the usable band (`<= 0.7`); the smoothed weights bound any single
  draw's influence so a sharp posterior is matched in a couple of passes
  (`proposal_source = "moment_matched"`). The fit's RNG is restored so the
  posterior draws are bit-for-bit unchanged. The collapsed-grid FD mode-Hessian
  rescue (gcol33/tulpa#116, #117) is now reserved for a TRUE delta collapse (no
  grid-weighted spread on ANY axis); a partial collapse where some axis still
  carries weighted spread keeps the grid-moment proposal and lets moment matching
  refine it, rather than over-widening the proposal with the local mode curvature
  of a non-Gaussian outer marginal.

## 0.0.43 (2026-06-18)

* Joint nested-Laplace outer Pareto-k: the collapsed-grid mode-Hessian rescue
  (gcol33/tulpa#116) now engages when a hyperparameter axis is pinned
  (gcol33/tulpa#117). When the integration grid concentrates (`ess_grid <= d`)
  the diagnostic reconstructs a Laplace-at-mode Gaussian proposal from a
  finite-difference Hessian of the outer target. Previously it differenced over
  **all** `d` axes, so a pinned axis (zero weighted variance: a `copy()` `alpha`
  fixed at 0, or a one-point dispersion grid) made the FD Hessian singular, the
  conditioning guard rejected it, and the k fell back to the `grid_moment`
  proposal #116 was meant to supersede -- which can be spuriously high and label
  an otherwise-fine fit "unreliable". `.joint_pareto_mode_cov()` now restricts
  the stencil to the varying axes (the same `var_tol` set the proposal build
  uses, factored into the shared `.joint_pareto_vary_axes()`) and embeds the
  inverse curvature block-diagonally, pinning the zero-variance axes at `u_hat`.
  Excluding a pinned axis from the curvature is exact, not an approximation.
  Affected any joint `occu_cover` fit with an uncoupled (no-`copy()`) cover arm
  plus a concentrated hyperparameter posterior.

* Per-cell warm start for the outer Pareto-k re-solves (gcol33/tulpa#118
  follow-up). Each importance draw's inner solve now starts from the converged
  latent mode of its NEAREST integration cell (`.joint_nearest_grid_mode`,
  threaded through `cpp_nested_laplace_joint_multi(x_init_per_cell=)` and the
  grid driver) instead of the single broadcast modal mode. Unlike the 0.0.42
  near-neighbour re-order -- which only helps the serial chain -- this also warms
  the PARALLEL pilot-mode path, so a threaded diagnostic (`n.threads.outer > 1`)
  gets it too: on a real EVA occu_cover fit (402 cells, beta) the per-cell warm
  cuts the parallel diagnostic a further ~1.5x on top of threading. The k-hat is
  byte-stable (each draw converges to the same mode regardless of start;
  validated == the broadcast-mode path and `loo::psis`). Knob
  `tulpa.kdiag.percell` (default TRUE; falls back to the re-order, then the
  broadcast mode, when grid modes are unavailable).

## 0.0.42 (2026-06-18)

* Faster joint nested-Laplace outer Pareto-k diagnostic (gcol33/tulpa#118).
  Profiling the joint occu_cover diagnostic showed the dominant cost is the
  per-Newton-iteration Hessian/gradient scatter (the beta cover arm's
  per-observation digamma/trigamma curvature fill, 73-83%), not the sparse
  Cholesky factorize (8-12%). Two changes attack that without moving the k-hat:
    * **Shamanskii (chord) factor reuse on the re-solves** (`.K_DIAG_REFRESH`):
      the diagnostic's inner re-solves run with `inner_refresh = 4`, so off-factor
      steps re-apply the cached factor to a refreshed gradient and scatter
      `grad_only` (skipping the curvature fill). The final mode-pass always
      re-factorizes with the true Hessian, so the converged log-marginal -- and
      thus the k-hat -- is unchanged; only the path to the mode uses a stale
      curvature, which the diagnostic (no per-draw SEs) does not need.
    * **Near-neighbour chain ordering of the importance batch**
      (`.joint_is_chain_order` / `.joint_is_solve_reordered`): the serial
      outer-grid driver warm-starts each cell from the previous cell's converged
      mode, so the random-order proposal draws were each starting from a
      random-neighbour mode (8-16 inner-Newton steps/draw). The batch is re-ordered
      into a standardised near-neighbour chain seeded at the modal cell
      (`.joint_modal_theta`), so each draw corrects only the small drift from its
      neighbour; the result is un-permuted before the PSIS layer, which is
      unaffected. The per-cell parallel path warm-starts from the pilot mode, so
      the order is then immaterial.
    * **Loosened inner-Newton tol on the re-solves** (`.K_DIAG_TOL = 1e-4`): a
      large share of the per-draw steps was intrinsic convergence to the fit's
      own tol (~1e-6), which the diagnostic does not need -- the Laplace
      log-marginal error from stopping at gradient norm `t` is O(t^2),
      immaterial to the tail-shape k-hat. Never tighter than the fit's tol.
  Combined, the three cut the diagnostic 3-4x on the beta arm with the k-hat
  byte-stable (validated vs `loo::psis` / `posterior::pareto_khat` on real EVA
  occu_cover importance ratios: identical to 4 decimals). Each is overridable
  via `tulpa.kdiag.refresh` / `tulpa.kdiag.tol` / `tulpa.kdiag.reorder` for the
  byte-for-byte exact diagnostic, and `tulpa.kdiag.capture` exposes the importance
  log-ratios for an external cross-check.

## 0.0.41 (2026-06-18)

* Mode-Hessian outer Pareto-k proposal for the joint nested-Laplace backend
  (gcol33/tulpa#116). The outer `pareto_k` diagnostic built its importance
  proposal from the grid-weighted covariance of the integration nodes; when the
  hyperparameter posterior is sharp the grid concentrates on ~1 cell, the
  grid-weighted covariance is then driven by negligible-weight far cells, and the
  too-narrow proposal yields a spurious high k-hat even though the fit is fine.
  The proposal is now built from a mode Hessian, on both integration paths.
    * **CCD path.** The integrator already places its design from the analytic
      curvature at the outer mode; that Gaussian (`u_hat`, `L_scale`) is captured
      and spliced into the Pareto-k driver over the axes it spans, block-diagonal
      with the grid-weighted spread on the independently tensor-crossed phi axes.
    * **Tensor path.** When the grid collapses (effective grid ESS `<= d`) and no
      CCD proposal exists, the Laplace-at-mode covariance is reconstructed from a
      finite-difference Hessian of the outer target at the modal cell (reusing the
      CCD stencil / conditioning helpers). Degenerate or ridged curvature falls
      back gracefully to the grid estimate.
    * New return field `pareto_k_proposal_source` in
      `{"mode_hessian", "grid_moment", NA}` (documented on
      `tulpa_nested_laplace_joint`) flags which regime a fit is in.
    * Known boundary: on a genuinely ridged outer posterior (a weakly-identified
      copy `alpha`) there is no PD Hessian along the ridge, so the FD Hessian
      declines and the path reports a `grid_moment` k. A ridge gives a wide
      proposal and hence a low k, so the spurious-high-k failure does not arise
      there; a principled diagnostic for non-Gaussian hyperparameter posteriors
      is left as future work.
    * A matched proposal is a byte-exact no-op on well-resolved grids, so the
      diagnostic is unchanged where it already worked. Decisive before/after on a
      well-conditioned tensor collapse (quad-ESS = 1.00): `grid_moment` k = 0.895
      (unreliable) -> `mode_hessian` k = 0.543 (reliable).

## 0.0.40 (2026-06-17)

* Outer-grid progress reporter (`tulpa_progress::GridProgress`, the
  nested-Laplace grid and parallel NUTS sampler): two fixes for long detached /
  redirected runs (gcol33/tulpa#115).
    * **ETA from realised throughput, not the serial pilot.** The ETA used to
      project the central, warm, serial pilot cell's per-cell rate across the
      whole grid. The parallel cells are extreme-hyperparameter, take more inner
      Newton steps, and run under memory-bandwidth contention, so each costs
      well more than the pilot -- the projection ran badly optimistic
      (observed ~10x low). It now rests on the mean wall time of completed
      POST-pilot cells (the throughput the remaining cells actually run at, the
      outer width already folded in). While only the pilot has been timed the
      projection is shown as a lower bound (`ETA >=`) rather than a point
      estimate (`ETA ~`), and the printed `s/cells` tracks the running average.
      At width 1 (serial) this reduces to the plain extrapolation.
    * **Live console in a detached parallel run.** The console line used to
      freeze at the serial pilot (`1/N`) because every parallel tick suppressed
      the console (worker threads must not touch the R print API), leaving only
      the heartbeat file advancing. The master thread (thread 0, the R main
      thread) now emits the newline-terminated line from inside the parallel
      region on its own throttle clock, so a redirected log shows the grid
      advancing cell by cell. Worker threads still only update the counter and
      heartbeat file; the heartbeat-file wire format
      (`<done> <total> <elapsed_s> <eta_s>`) is unchanged, so existing readers
      keep parsing.


* Joint nested-Laplace multi-block driver: two random-effect capabilities for
  random **slopes** on an observation arm (gcol33/tulpa#114), so a downstream
  consumer can fit correlated and uncorrelated slopes, not just intercepts.
    * An optional per-row design weight (`svc_weight`) on the `iid` block,
      mirroring the areal / temporal SVC path: the field's contribution to arm
      `k` row `i` is row-scaled by `svc_weight[[k]][i]`
      (`eta_i += svc_weight[[k]][i] * sigma * u[obs_idx_i]`). An uncorrelated
      slope `(0 + x | g)` / `(x || g)` is then one weighted `iid` block per
      coefficient, each with its own `sigma` axis. Unset `svc_weight` is
      byte-identical to the plain random-intercept `iid` block.
    * A new multivariate-IID block, `type = "miid"`: the non-spatial sibling of
      `mcar` with `Q = I`. Per group `g` a coefficient vector
      `b_g ~ N(0, Sigma)` (block dim `n_fields = 1 + n_slopes`), with the free
      cross-coefficient `Sigma` integrated over the same `p(p+1)/2` log-Cholesky
      outer-grid axes as `mcar`. The precision `Sigma^-1 (x) I` is full rank, so
      (unlike `mcar`) there is no sum-to-zero pinning and the normalizer carries
      the full `n` (`n log|Sigma^-1|`). This expresses a correlated random slope
      `(1 + x | g)`. Copy (`alpha`) semantics compose as for `mcar`. A `miid`
      block with `n_fields = 1` is the centered counterpart of the scalar `iid`
      block (Laplace-invariant). Direct-algebra prior assembly
      (`Sigma^-1 (x) I`, gradient, log-prior) and parameter recovery are tested
      (`test-miid-prior.R`, `test-miid-recovery.R`). The shared MCAR/MIID obs
      scatter and copy `arm_scale` are single-sourced in `mcar_block_factory.h`.

* `control$k_threads`: outer-thread width for the joint nested-Laplace fit's
  Pareto-k diagnostic importance batch. The `k_samples` re-solves run after the
  grid (every core free), each solved single-threaded once the batch saturates
  the pool, so widening the pool is a bit-identical wall-clock speedup with an
  unchanged k-hat. `NULL` (default) follows the fit's own thread grant, `"auto"`
  uses the physical performance-core count (capped at 2 under R CMD check), and
  an integer pins the width.

## 0.0.38 (2026-06-17)

* `laplace_diagnostics()`: a front-door diagnostic for deterministic
  (i.i.d.-draw) nested-Laplace fits, the class `mcmc_diagnostics()` declines to
  treat as MCMC. It returns a per-parameter table (posterior mean / sd, plus the
  i.i.d.-draw bulk / tail effective sample size and split-Rhat of the draws,
  labelled as Monte-Carlo diagnostics rather than chain mixing) and attaches the
  reliability headline as attributes and a `summary` row: the PSIS Pareto-k-hat
  of the outer hyperparameter integration scored against the exact inner-Laplace
  marginal (Vehtari et al. 2024; Yao et al. 2018), and the grid quadrature
  effective sample size `ess_grid = 1 / sum(w_k^2)`. `mcmc_diagnostics()` now
  dispatches an i.i.d.-draw fit to `laplace_diagnostics()`.

* Pareto-k diagnostic on the joint engine: the importance-sampling proposal is
  now built on the grid axes that actually vary, so an outer grid that pins an
  axis (for example `alpha.grid = 0`) no longer yields a singular `chol(Su)` that
  silently skips the diagnostic; the k-hat is computed on the remaining axes.

* Speed: the per-sample marginal re-solves of the Pareto-k diagnostic refit now
  honour `n_threads_outer` (previously hardcoded to one thread), giving a
  5.8-7.7x speedup on `diagnose.k = TRUE` fits with k-hat unchanged.

## 0.0.37 (2026-06-16)

* build: the `tulpaMesh` dependency floor is raised to `tulpaMesh (>= 0.1.3)`,
  locking it to the current tulpaMesh release. The `Remotes` install reference was
  already `gcol33/tulpaMesh@v0.1.3`.

## 0.0.36 (2026-06-16)

* Maintenance release. No user-facing changes; the version is bumped to keep the
  tulpa and tulpaObs release tags in step.

## 0.0.35 (2026-06-16)

* test(tiers): the test suite is now organized into three explicit cost tiers,
  single-sourced in `tests/testthat/helper-tiers.R`: tier 1 structural (ungated,
  runs on CRAN), tier 2 recovery (`skip_on_cran()`), tier 3 full samplers and
  coverage (`skip_if_not_slow()`). About 100 test files were re-gated and a tier
  table added in `tests/testthat/README.md`.
* test(tiers): `TULPA_FAST=1` is a fast smoke profile folded into both tier
  gates, collapsing the suite to the tier-1 structural tests only (heavy fits and
  samplers reported as skips, never dropped) for sub-minute plumbing iteration.
  The default and CRAN runs are unchanged.
* build: bump `LinkingTo: gcol33/tulpaMesh` to `v0.1.3`.

## 0.0.34 (2026-06-12)

* feat(nested-laplace-joint): `cpp_joint_inner_vcov_blocks` now defaults
  `field_marginal = TRUE` (and `n_threads = 1`). The cheap selected-inversion
  recipe -- the betas block and betas x field cross solved exactly, the field
  marginal variances from one Takahashi pass -- is the default per-cell
  extraction; the full `p x p` block is opt-in via `field_marginal = FALSE`.
* perf(nested-laplace-joint): the cell-coupling per-cell scatter
  (`scatter_cell_coupling_branch_impl`, `src/nested_laplace_joint_multi.h`) gained
  an optional per-arm rank-1 self-cross descriptor on `CellDerivs`
  (`arm_cross_rank1_coef` / `arm_cross_rank1_vec`, `inst/include/tulpa/cell_coupling.h`).
  When a coupled arm's (k, k) off-diagonal cross-Hessian is the symmetric rank-1
  `a v v^T` -- every cross-row second derivative factoring through one scalar, as
  in tulpaObs's all-undetected occupancy mixture -- the kernel collapses it to a
  single `a u u^T` in joint-dof space (`u = sum_r v[r] chain(row_r)`, accumulated
  by `accumulate_self_rank1_u` and scattered by `scatter_self_rank1_{dense,sparse}`)
  instead of the O(rc^2) dense `arm_cross_hess[k][k]` loop, dropping the scatter
  from O(sum rc^2) to O(sum rc) (gcol33/tulpaObs#94). Single-response path
  (`n_batch_ == 1`) only; cross-arm blocks and the dense `arm_cross_hess` path are
  unchanged, and the spec folds the rank-1's own diagonal into
  `arm_neg_hess_diag` so the assembled Hessian matches the dense path to machine
  precision.

## 0.0.33 (2026-06-12)

* perf(nested-laplace-joint): the joint post-grid inner-covariance extraction is
  now a single parallel C++ primitive, `cpp_joint_inner_vcov_blocks`
  (`src/joint_inner_vcov.{h,cpp}`), replacing the serial-R per-cell
  `solve(Qk, E)` over ~`n_betas + n_field` right-hand sides (gcol33/tulpa#112,
  #113; gcol33/tulpaObs#93). For the field-marginal summary it solves only the
  `n_dense` fixed-effect columns of `Qk^-1` (the betas block and the betas x
  field cross, exact) and recovers the field marginal variances from one
  Takahashi selected-inversion pass (`selected_inversion_diagonal`,
  `sparse_cholesky.h`); the field x field off-diagonal -- read by neither the SD
  summary nor the `Q_k`-direct predict path -- is not formed. Cells run
  concurrently over the supplied thread budget. A `field_marginal = FALSE` mode
  forms the full block (the betas-only callers). Numerically identical to the
  former dense path on the read sub-blocks (`test-joint-inner-vcov.R`).

## 0.0.32 (2026-06-12)

* refactor(linalg): the small-dense lower-Cholesky factorization, triangular
  solves, log-determinant, and NNGP conditional (kriging) moments are now a
  single `linalg_fast.h` core (`chol_factor_lower` / `chol_forward_solve` /
  `chol_back_solve` / `chol_log_det` / `nngp_conditional_moments`), replacing 8
  hand-rolled copies across the Laplace, PG-Gibbs, SVC, spatiotemporal,
  temporal-GP, GPU-fallback, and proper-CAR paths (#109). The default `1e-10`
  pivot jitter is named `kCholJitter`; the SVC kernel's `1e-6` is now an explicit
  argument rather than a drifted literal. Behavior-preserving.
* refactor(temporal): the RW1/RW2 quadratic and cross forms and the AR1
  log-density are single templated kernels shared by the double, autodiff, and
  generic-sampler paths (TVC and multiscale-temporal included), replacing three
  drifted implementations (#110). Where the copies disagreed the merged kernel
  keeps the live generic-sampler behavior (cyclic honored, AR1 stationary guard
  `1e-10`); the dead `*_autodiff.h` twins are removed. Numerics shift at most at
  the `1e-10` / ULP level (AR1 stationary term, summation order).
* refactor(joint): `.normalise_joint_arm()` and `.normalise_joint_arm_multi()`
  delegate to a shared `.normalise_joint_arm_core()`, collapsing ~50 duplicated
  lines of arm-spec validation (#111). The single-block path still requires an
  arm-level `spatial_idx`; the multi-block path still fills a zero placeholder --
  the only behavioral split, now an explicit policy argument.

## 0.0.31 (2026-06-12)

* fix(laplace): `tulpa_laplace(..., weights=)` now scales the log-likelihood by
  the per-observation weight, matching the already-weighted score and Fisher
  Hessian (#108). The Newton globalization backtracks on the log-likelihood, so a
  weighted-optimal step was judged against an unweighted objective, halved toward
  zero, and stalled -- any non-uniform weighting returned a non-converged mode
  shrunk toward the prior. The fix is a no-op when weights are absent or uniform
  (every unweighted path is byte-identical); weighted gaussian / binomial fits now
  match `lm()` / `glm(weights=)` to ~1e-5. `test-laplace-weights.R` locks it in.
* perf(nested-laplace-joint): when the outer grid has fewer cells than the outer
  thread pool, the surplus threads are now handed to the inner per-observation
  solve via nested OpenMP instead of idling (#107). A shared
  `joint_inner_thread_budget()` splits the pool so `outer_used * inner <= n_outer`
  (never oversubscribed) and is a no-op when the grid saturates the pool. Verified
  result-invariant (theta means byte-identical) with a ~1.12x single-fit speedup on
  a 4-cell surplus grid; `test-nested-laplace-joint-threading.R` locks in the
  invariance.

## 0.0.30 (2026-06-10)

* refactor(spde): the fractional rSPDE Laplace marginal moves from R to C++
  (`cpp_spde_fractional_logmarginal`, Eigen). The well-conditioned `B` /
  matrix-determinant-lemma method -- built through the operator factor `Pl`
  (cond = sqrt cond(Q)), never an explicit ill-conditioned `Q` inverse -- is
  preserved: a direct precision-space marginal drifts by O(10) nats at large range
  where cond(Q) ~ 1e10+, mis-identifying range. `.spde_nested_logmarginal_at` and
  the single-point `.spde_laplace_fractional_at` delegate to it; the R det-lemma /
  closed-form / family-weight code is removed. Reproduces the former R marginal to
  ~1e-10. The gaussian fractional `phi` is now the residual SD (variance `phi^2`),
  consistent with the integer path (was the variance).
* fix(spde): the nested-Laplace `(range, sigma)` marginal now carries the GMRF
  prior normalizer `0.5 log|Q(theta)|` (#98). It was dropped on the integer-alpha
  SPDE path -- the Occam term that bends the marginal down at large `sigma` -- so
  `fit_spde()` (and the `tulpa(..., spatial = spatial_spde())` front door) railed
  `sigma` to the prior boundary and collapsed `range`, while the Tier-1 NUTS-joint
  path recovered. The two integrators now agree: on a true 0.35/0.80 field the CCD
  weighted means recover `range ~ 0.38`, `sigma ~ 0.76` (matching NUTS), validated
  by a new multi-seed recovery + CI-coverage gate in `test-spde-ccd.R`. The
  normalizer is single-sourced in C++ (`src/spde_logdet.h`, a CHOLMOD log|Q|);
  the fractional path's R-side fold is removed accordingly. Exposed and fixed a
  latent gap where the SPDE CCD refit returned a `NULL` `beta` (the mode was never
  split into fixed effects / field), now done once in `laplace_spde_at()`.
* fix(diagnostics): the outer Pareto-k radius cap no longer biases k-hat
  downward in the heavy-tail regime it exists to flag (#100). When the importance
  log-ratio is still rising at the cap boundary (the target is heavier-tailed than
  the Gaussian proposal) the dropped far-radius draws are folded back in, so the
  GPD fits the genuine uncapped tail; a flat/light tail leaves the cost cap in
  force. The re-cov path's no-cap choice is documented as consistent in
  correctness.
* fix(s3): every exported front-door fitter (`tulpa_re_cov_nested` / `_gibbs`,
  `tulpa_laplace`, `fit_spde`, `tulpa_nested_laplace[_joint]`, `tulpa_tgmrf`)
  routes its return through a shared `.finalize_fit()` so a directly-called fit
  carries the same `tulpa_fit` class, fixed-effect layout, and explicit
  `draws_kind` provenance tag as a `tulpa()`-dispatched one. The chain-vs-iid
  diagnostic gate no longer computes a vacuous Rhat on a directly-called iid fit
  (#102).
* fix(validate): `tulpa()` now validates `phi > 0` for every dispersion-carrying
  family (not just beta) and rejects non-integer / negative `y` for count
  families at the front door; the `inverse` / `1mu2` links clamp `eta` off their
  singularities (#104).
* refactor: removed the single-block `copy=` back-compat shim from
  `tulpa_nested_laplace_joint()` (declare the copy coefficient on the arm via
  `field_coef`, #105); consolidated the comparison / averaging verbs to native
  PSIS-backed `compare_models()` / `modelAverage()` (stacking + pseudo-BMA, no
  `loo` dependency -- `loo` moved to Suggests), and the four `tulpa_tgmrf_*`
  fitters into one `tulpa_tgmrf(mode=)`; standardized RE-Gibbs / PG-Gibbs
  iteration arguments on `n_iter` + `warmup` (#103).
* test: parameter-recovery / CI-coverage gates on the deterministic Tier-2 hot
  paths -- nested-Laplace spatial hyperparameters (ICAR/BYM2/CAR_proper/NNGP/HSGP,
  #97), the Pareto-k diagnostic's discriminating power on real engine output
  (#99), and the assembled generic-NUTS and VI estimators (#101). The SPDE
  cross-integrator test (#98) makes explicit that the deterministic `fit_spde`
  CCD path does NOT recover `(range, sigma)` (the Laplace-marginalized SPDE
  likelihood is prior-dominated in both); only the Tier-1 NUTS-joint path does.
* docs: runnable `\examples` and method `@references` on the front doors,
  `inst/CITATION`, internal issue tokens stripped from rendered help, and stale
  version / example strings refreshed (#106).

* feat(frontdoor): `mode = "agq"` now reaches the adaptive Gauss-Hermite
  quadrature fitter through `tulpa()`. A single random-intercept `(1 | g)` model
  with a `binomial` / `poisson` / `gaussian` family routes to `agq_fit()`, with
  `control$n_quad` (default 7; `1` recovers Laplace) selecting the quadrature
  order and `phi` mapped to the gaussian residual sd. The front-door fit equals a
  direct `agq_fit()` call. Previously the backend was registered but the design
  dispatch fell through to a "reachable but not yet wired" error. Random slopes,
  multiple RE terms, other families, and a `beta_prior` (AGQ is a
  marginal-likelihood fit) are rejected with guidance.
* fix(methods): the generic fixed-effect accessors (`coef`, `summary`, `confint`,
  `vcov`) now read a Gaussian fit's full-parameter `$cov` when it carries no
  draws, grid moments, or `$H_beta` (the AGQ shape), and `.fixed_draws_mat()`
  treats a zero-row `$draws` matrix as "no draws" rather than empty draws. Fits
  with real draws / grid moments / `H_beta` are unaffected.
* feat(spatial): the single-arm multi-block nested-Laplace driver now honours a
  per-observation design weight on an areal (icar) block, exposed as an optional
  `svc_weight` field in the block spec. When present, observation i's eta
  contribution is `svc_weight[i] * z[spatial_idx[i]]` rather than
  `z[spatial_idx[i]]` -- a spatially-varying coefficient (the areal
  `f(cell, weight, ...)`), the single-arm analogue of the `row_weight` the joint
  multi-arm driver already carries. The weight enters at one layer (the
  block-local weight resolved alongside the node index); the gradient inherits it
  and the block Hessian its square through the chain rule, in `compute_eta_spec`
  / `scatter_spec` and the `fitted_eta` / predictive-variance reconstruction. A
  block without `svc_weight` is byte-identical to before. This lets a standalone
  occupancy fit carry a cell-indexed varying-coefficient field (consumed by
  tulpaObs `occu()`'s spatial bar).

* fix(joint): the coupled cell-coupling per-cell scatter now handles
  `INDEXED_MULTI` prior blocks (a separable-MCAR block's several latent dofs per
  row), not only `INDEXED_SINGLE`. Previously `scatter_one_arm_row_{dense,sparse}`
  and `build_arm_row_chain` resolved a row's active latent dofs through
  `block.idx` alone, so a multi-field block (e.g. a free-Sigma MCAR field) coupled
  onto a `coupled = TRUE` arm received no gradient/Hessian from the cell-coupling
  likelihood and stayed pinned at its prior mean. The active-latent resolution is
  now a single `collect_coupled_row_latents()` helper shared by all three, so the
  free-Sigma MCAR field couples through the joint occupancy mixture (consumed by
  tulpaObs `occu_cover()`'s correlated `|` spatial bar). `INDEXED_SINGLE` coupled
  fits (the areal / SVC trend path) are byte-identical.
* feat(spatial): `spatial()` gains a `by =` argument for replicated CAR -- one
  independent copy of the whole varying-coefficient field per level of a factor,
  with the hyperparameters shared across levels (`INLA`'s `replicate =` /
  `mgcv`'s `s(cell, by = ...)`, generalised to the bar). It is orthogonal to the
  bar character: `|` / `||` sets the covariance among the coefficient columns
  within a field, while `by` sets how many replicates exist. A `by` factor with
  `L` levels builds the field over the block-diagonal Kronecker graph
  `I_L (x) Q` (`L` disjoint copies, the node index offset into each level's
  copy), so the replicates are independent and share one precision -- the outer
  integration grid stays one axis. Supported for `||` and `|` (intrinsic) and
  `||` + `proper = TRUE`; `|` + `proper` stays out of scope, with or without
  `by`. The new `tulpa_bar_field_replicate()` exposes the Kronecker remap for
  consumer packages (the graph-side sibling of `tulpa_bar_field_specs()`).
* fix(spatial): the intrinsic-CAR kernels (ICAR and the separable MCAR) are now
  connected-component aware. The rank-deficiency treatment -- the sum-to-zero
  null-space pin and the `(n - 1)` log-determinant normaliser -- assumed a single
  connected graph; over a disconnected graph (the `L`-component block-diagonal
  field a replicated CAR builds) the field constant of each component is its own
  null direction. The kernels now apply one sum-to-zero pin per component and
  normalise with `(n - n_components)`, so a block-diagonal `L`-component
  log-prior equals the sum of the `L` independent single-component log-priors.
  A connected graph is the `n_components = 1` case, byte-identical to before.

## 0.0.28 (2026-06-09)

* feat(spatial): the separable-MCAR areal block can now be COPIED across arms in
  the joint multi-block driver. A correlated `(intercept, slope)` field sharing a
  free cross-covariance `Sigma (x) Q^-1` (the within-arm covariance among the
  fields) is copied onto a second linear predictor with one estimated amplitude
  `alpha` (the cross-arm transfer): the donor arm sees the natural-parameter
  field at amplitude 1, the copy arm at `alpha`, with `alpha` integrated over the
  outer grid as a trailing axis alongside the `Sigma` log-Cholesky coordinates.
  `copy = list(arm =, block =, alpha_grid =)` now accepts a `type = "mcar"`
  block; the copy amplitude rides on the block's `arm_scale` (the natural-
  parameter field stays the latent, so a single per-arm scalar carries the whole
  correlated field). Previously MCAR rejected copy semantics. This is the engine
  half of the cover-hurdle correlated-field consumer (gcol33/tulpaObs#64). The
  single-arm MCAR path and every existing copy block are byte-identical
  (no copy = empty `arm_scale`). Recovers `Sigma` (both SDs + the cross-
  correlation) and the copy `alpha` against simulated truth
  (test-nested-laplace-joint-multi-copy.R).

## 0.0.27 (2026-06-09)

* feat(formula): the varying-coefficient bar column-expansion is now public.
  `tulpa_bar_field_specs(~ 1 + w || node, data)` expands an lme4-style bar into
  one spec per design-matrix column -- `(column_name, weight, is_intercept)`,
  where the intercept column is the unweighted (all-ones) field and each
  covariate column carries its per-observation design value as the field weight.
  `tulpa_is_spatial_bar()` recognizes such a bar. Both surface the single
  expansion `spatial()` and the inline `temporal()` field constructor already
  use internally (the one bar column-expansion helper the two paths share), so a
  downstream package can offer a one-term spatial / temporal bar without
  re-parsing the `~ 1 + w || node` grammar. `spatial()` and the temporal field
  constructor are refactored onto the shared bar recognizer so the engine and
  any consumer cannot drift (gcol33/tulpa#93).

## 0.0.26 (2026-06-08)

* fix(spatial): `fit_spde(method = "ccd")` now actually runs the central-
  composite design. `fit_spde_nested_ccd()` took `optimHess()` of the negative
  log-posterior -- already the positive-definite precision of the Laplace
  approximation -- and then negated it, so the degeneracy guard rejected every
  usable mode and the integrator fell back to the rectangular grid on all
  inputs. The negation is removed; the precision is used directly to orient the
  design. `method = "ccd"` is the `fit_spde()` default, so fits over an
  identified SPDE hyperparameter posterior now use the 9-node mode-centred
  design instead of the 25-node grid. A weakly-identified axis still falls back
  to the grid via the existing mode and Hessian guards (gcol33/tulpa#92).

## 0.0.25 (2026-06-08)

* feat(spatial): the correlated areal field (separable MCAR,
  `spatial(graph, ~ ... | cell)`) now integrates its cross-covariance `Sigma`
  on a **mode-centred central-composite design** over the log-Cholesky
  coordinates, the same outer-integration recipe `tulpa_re_cov_nested()` uses
  for random-effect covariances. The log-Cholesky coordinates (`log L_ii` on
  the diagonal, raw strict-lower `L_ij`) are already unconstrained on all of R,
  so they enter the joint CCD as identity axes: the integrator mode-finds the
  marginal-likelihood mode in `Sigma`-space, orients the design by the Cholesky
  of the posterior covariance, and weights with the corrected R-INLA design
  weights -- no new mode-find or CCD code, the existing joint CCD machinery
  drives it. This replaces the fixed log-Cholesky tensor that could land on the
  nearest node and miss a sharp likelihood mode, and scales to general `p` at a
  polynomial node count (`1 + 2k + 2^k` for `k = p(p+1)/2` axes) where the fixed
  tensor was exponential (p = 3 is 77 nodes vs ~1700 cells; p >= 4 no longer
  overruns the grid cap). When the cross-correlation is weakly identified the
  outer curvature is ill-conditioned and the CCD declines back to the fixed
  log-Cholesky tensor, the correct net there. The outer Pareto-k accuracy
  diagnostic (`fit$pareto_k`) is now reported for MCAR fits (it was `NA` while
  the fit declined the CCD); a high k-hat on a small-group, weakly-identified
  cross-correlation is a correct signal, not a defect. Recovers the fields and
  every cross-correlation `rho_ij` with covering CIs at p = 2 and p = 3
  (test-spatial-mcar.R).

## 0.0.24 (2026-06-08)

* feat(spatial): `spatial(graph, ~ 1 + x | cell)` (a single bar `|`) builds
  correlated areal varying-coefficient fields -- a separable multivariate CAR
  (MCAR) where the per-cell coefficient vector shares a cross-covariance
  `Sigma`, with joint latent covariance `Sigma (x) Q^-1` (#89). One coupled
  block over the `p` fields assembles the Kronecker precision `Sigma^-1 (x) Q`
  in the inner Laplace solve (the `p` per-field sum-to-zero constants are pinned
  and folded by the block-Schur path), with the `p` design columns entering the
  linear predictor as `eta_i += sum_c X_{ic} u^{(c)}_{cell_i}` via an
  INDEXED_MULTI block. The outer grid integrates over `Sigma` in log-Cholesky
  coordinates; the cross-field correlation `rho` and the per-field `sigma`s are
  derived quantities, reconstructed per grid cell and weighted-quantiled (the
  marginalize-derived-quantities rule). `print()` and `fit$mcar_summary` report
  the marginalized `Sigma` (`sigma_1`, `sigma_2`, `rho_12`, ... with 95% CIs).
  Recovers the fields and `rho` vs simulated truth with CI coverage
  (test-spatial-mcar.R); the `Sigma^-1 (x) Q` assembly, gradient, and log-prior
  (incl. the `(n-1) log|Sigma^-1|` normalizer) are locked by a direct algebra
  check against `kronecker(Sigma^-1, Q)` (test-mcar-prior.R). The single bar `|`
  no longer errors. A single `|` with `proper = TRUE` (correlated proper CAR) is
  out of scope and errors. Tested for `p = 2` (the headline intercept-plus-slope
  case); general `p > 2` fits through the same path with a coarser raw
  log-Cholesky grid.

## 0.0.23 (2026-06-08)

* feat(temporal): `temporal(formula = ~ 1 + x || time, structure = "rw1")`
  declares inline temporally varying-coefficient fields in a `tulpa()` model
  formula, the temporal mirror of `spatial()` (#91). The bar's right-hand side
  names the time index; the left-hand side expands (via `model.matrix`) into one
  temporal field per design column -- the intercept column is a smooth temporal
  level, a covariate column is a temporally varying slope on it
  (`eta_i += x_i * f(time_i)`). `structure` selects the temporal GMRF: `"rw1"`
  (default), `"rw2"`, or `"ar1"` (which estimates its own correlation `rho`).
  `||` (independent fields) only; a single `|` (correlated) is reserved (the
  temporal counterpart of the spatial MCAR). `print()` and
  `fit$temporal_field_hypers` report each field's structure, `sigma`, and (ar1)
  `rho`. The `temporal()` accessor on a fitted model and the bare `temporal(col)`
  naming term are unchanged -- the constructor is reached only when `temporal()`
  is given a formula.
* The temporal nested-Laplace blocks (`rw1` / `rw2` / `ar1`) now carry the per-
  row design weight (`svc_weight`) that the areal blocks already had, so a
  covariate column scales the field per observation. The spatial and temporal
  inline-field fitters share one engine (`.bar_field_fit_core`) and one bar
  column-expansion helper, so the two paths cannot drift.

## 0.0.22 (2026-06-08)

* feat(spatial): `spatial(graph, ~ ... || cell, proper = TRUE)` builds proper
  CAR varying-coefficient fields, where each field's precision is
  `Q = D - rho_car W` with the spatial autocorrelation `rho_car` estimated from
  the data instead of the intrinsic `rho = 1` (#90). Each field stays
  independent (`||`) but gains its own `(sigma, rho_car)` pair, so the per-field
  outer grid is 2D (the `car_proper` registry derives the `(tau, rho_car)` axes
  from the eigenvalue interval of `D^-1 W`); two proper fields give 4 axes and
  CCD engages automatically. `print()` reports each field's structure (`ICAR`
  vs `proper CAR`) plus the marginalized `sigma` and `rho_car` (median + 95% CI,
  each a derived quantity weighted-quantiled over the outer grid, never a
  plug-in of the modal hyperparameter); `fit$spatial_field_hypers` exposes them.
  `proper = FALSE` (default) is unchanged (intrinsic ICAR, `rho` fixed at 1). A
  single `|` with `proper = TRUE` (correlated proper CAR) remains a separate
  model.

## 0.0.21 (2026-06-08)

* perf/fix(nested-laplace): the sparse sum-to-zero large-field inner solve now
  takes the exact block-Schur Newton step instead of escalating a
  Levenberg-Marquardt ridge until the pinned Hessian factors. For
  `B = A + sum_k coef_k 1_k 1_k'` (the intrinsic field plus its sum-to-zero
  rank-1 pins) the field sub-block `A_FF` is factored once, the pins fold via the
  matrix-determinant lemma, and the field<->scalar coupling closes with a small
  dense Schur complement -- the true Newton step with no perturbing ridge, so the
  inner iteration converges quadratically and no longer drops ill-conditioned
  high-`(sigma, alpha)` grid cells to `-Inf` (gcol33/tulpa#69). The Laplace
  log-determinant uses the same partition (`log|B| = log|B_FF| + log|Schur|`),
  which never factors the unpinned near-singular full matrix and so avoids the
  matrix-determinant-lemma cancellation along the constant direction. The single-
  species sparse oracle and the fused batched multi-response driver
  (`run_multi_block_nested_laplace_joint_batch`) route their inner step through
  one shared `s2z_newton_step`, so they stay bit-identical per species; the LM
  ridge + Woodbury path remains the fallback when `A_FF` or the Schur complement
  is indefinite far from the mode. Validated bit-identical (batched vs single,
  `max|dmode| = max|dlogmarg| = max|dQ| = 0` at a 412-cell field) and locked by a
  direct block-Schur-vs-dense-LLT log-determinant + step unit test.

## 0.0.20 (2026-06-08)

* feat(spatial): `spatial(graph, formula = ~ 1 + time || cell)` declares areal
  varying-coefficient fields inline in a `tulpa()` model formula, the way a
  random-effect bar is written. The bar's left-hand side expands (via
  `model.matrix`) into one independent CAR / Besag field per design column: the
  intercept column is the spatial intercept field `u_cell`, a covariate column
  is a spatially varying slope on it (a per-region trend `time * s_cell`). The
  intercept is just the all-ones column, so the unweighted and weighted cases
  share one path; factors, `I(time^2)`, and splines expand for free. Each field
  carries the sum-to-zero constraint and its own precision (independent fields).
  A single response is fit through the single-arm joint nested-Laplace path,
  which threads the per-row design weight (`svc_weight`), and `summary()` /
  `coef()` report the marginalized fixed effects; the per-field posterior means
  are on `fit$spatial_fields`.
* The bar grammar is strict: `||` (independent fields) only -- a single `|`
  (correlated fields, a multivariate CAR) errors as not-yet-implemented, as do
  nested (`a / b`) or interaction grouping (the grouping must be a single
  graph-node index; add ordinary nested random effects such as `(1 | site)`
  separately), a missing bar, `by =` (reserved for replicated CAR), and
  `proper = TRUE` (reserved). The bare `spatial(col)` areal-naming term and the
  `spatial =` constructor path are unchanged.

## 0.0.19 (2026-06-08)

* feat(progress): the nested-Laplace outer-grid progress line now shows the
  active outer-thread count, e.g. `... | 0.06s/cells | 28 threads`, whenever the
  grid runs more than one cell at once, so "ran on N cores" is a property of the
  fit log itself rather than something to read out separately (#88). The count
  is the realised outer width stamped on the reporter by `run_nested_laplace_grid`
  (after the sparse path's memory clamp), so it covers every model routing
  through the joint engine -- the cover hurdle and `occu_cover()` included -- plus
  the parallel NUTS sampler, which reports its concurrent-chain count the same
  way. The mirrored R-side reporter (`.tulpa_iter_progress()`) gains an optional
  `threads` argument carrying the same field. Serial loops leave the count at 1
  and omit the field; their lines are byte-for-byte unchanged.

## 0.0.18 (2026-06-08)

* feat(spde): fractional-nu SPDE fields gain a fixed-hyperparameter NUTS path
  (#85, #87). The rational (BRASIL) approximation is sampled with the
  hyperparameters held fixed and the latent field non-centered, marginal
  fixed-effect standard errors are reported, and the precompute path passes the
  full both-triangle precision `Q` on the fractional NUTS route (#87).
* fix(numerics): the s2z rank-1 log-determinant no longer cancels catastrophically
  for fields above 256 nodes. The matrix-determinant-lemma update lost ~2.7 nats
  to cancellation at `n_x > 256`; the log-determinant structure is now cached
  across grid cells and species, restoring the densify-vs-rank-1 log-marginal
  equivalence (guarded downstream by tulpaObs).
* perf(laplace): the fused batched occu_cover scatter is sparse-native and
  bit-identical to the dense path.
* test(joint): the coupled-cell path composes a shared field with a per-group
  random effect (#86).

## 0.0.17 (2026-06-07)

* docs(vignettes): correctness pass against the current API. Corrected the
  nested-fit `logLik()` / `compare_models()` description (a nested fit returns the
  integrated-evidence scalar, not a per-grid vector) in the spatial and temporal
  vignettes, refreshed the stale temporal front-door scope note (rw1/rw2/ar1 plus
  panel and areal space-time are all wired), replaced an unexported
  `is_connected()` reference with a base-R graph-Laplacian connectivity check, and
  moved `tgmrf_cpp()` from a "forthcoming" framing to its shipped present-tense
  description.
* build: drop the precompiled-header mechanism; each translation unit parses
  RcppEigen directly.

## 0.0.16 (2026-06-07)

* Tagged release of the grouped beta sufficient-statistic joint interface
  (`slog_y` / `slog_1my`, added in 0.0.15) so consumer packages can pin a
  released engine; consumed by tulpaObs `aggregate.pos`. No engine code change.

## 0.0.15 (2026-06-07)

* feat(laplace): the joint nested-Laplace engine accepts optional grouped beta
  sufficient statistics on a built-in `beta` arm. When an arm carries `slog_y` /
  `slog_1my` (the within-group sums of `log(y)` and `log(1 - y)`, with
  `n_trials` the per-row group count), a row collapses `n` exchangeable beta
  observations sharing one linear predictor into a single row. The beta
  log-density is linear in `log(y)` and `log(1 - y)`, so the log-likelihood,
  gradient and Fisher Hessian are pointwise unchanged and the aggregated fit is
  byte-identical to the per-observation path (`n = 1` reduces exactly to the
  ungrouped branch). Read by the single-block and multi-arm joint drivers; the
  shared `log_lik_beta_grouped` / `grad_hess_beta_grouped` helpers in
  `laplace_family_link.h` are the single source. This is the engine backing for
  tulpaObs's `aggregate.pos` cover-arm reduction (gcol33/tulpaObs#49).

## 0.0.14 (2026-06-06)

* refactor: remove structural duplication flagged by a code-rot scan, with no
  change in behavior. `.is_multi_block_prior` is now a single predicate (the
  byte-identical `_joint` copy is dropped); the spatially- and
  temporally-varying-coefficient `print`/`summary` methods delegate to shared
  `.print_varying_coef` / `.summary_varying_coef` helpers; the column-major
  matrix builder is one `template` over the element type (NumericMatrix vs
  IntegerMatrix); and the internal CAR-proper and PC-variance log-prior helpers
  follow the dominant `log_prior_*` naming (`log_prior_car_proper`,
  `log_prior_sigma2_pc`), retiring a dead wrapper.

## 0.0.13 (2026-06-06)

* fix(check): clears every `R CMD check --as-cran` ERROR and WARNING. The
  joint-NUTS fractional-nu test now asserts the rejection at the fit call --
  the spec constructor accepts fractional nu since the rational SPDE landed
  (gcol33/tulpa#71); the performance-core test drops an assertion comparing a
  hardware core count against the OpenMP-capped thread limit. Rd: a dangling
  `\link` and lost-brace math in the rational-SPDE docs are fixed; NAMESPACE
  gains `importFrom(utils, flush.console)` / `importFrom(stats, vcov)`; the
  `tulpa_sample_glmm()` offset is passed by its full name (`offset_nullable`).
* fix(build): the Windows precompiled header now rebuilds when `Makevars.win`
  changes, so a `.gch` left from an earlier flag set is no longer silently
  rejected ("created and used with differing settings") and re-parsed in every
  translation unit -- restoring the PCH speedup (cold compile ~70s -> ~61s).

## 0.0.12 (2026-06-06)

* feat(progress): unified iteration progress + ETA across every fitting loop
  -- the C++ outer grids (nested-Laplace, joint, sparse SPDE), the NUTS
  sampler, and the R-side EM loop (gcol33/tulpaObs#43). Two independently
  gated channels: a console bar (the noisy TTY channel) and a heartbeat file
  written whenever `progress.file` is set, the robust liveness signal for
  detached runs where an Rcout flush does not survive the stdout buffer.
  `GridProgress` gains `emit_console` + `unit`; an R-level
  `.tulpa_iter_progress()` mirrors the same `<done> <total> <elapsed_s>
  <eta_s>` wire format so a detached reader sees one file regardless of which
  loop produced it. NUTS ticks a shared reporter across chains (the console
  line self-suppresses inside the OpenMP region, the heartbeat file is the
  parallel channel). The nested-Laplace console default flips ON; inner
  refinement / EM / CCD-probe call sites pass `progress = FALSE` so only the
  top-level fit ticks.
* docs(api): document `tulpa_profile()`, the inner sparse-Laplace phase timer.

## 0.0.11 (2026-06-06)

* feat(samplers): thread random-effect, areal-spatial, and temporal latent
  structure through the model-agnostic ModelData sampler kernels --
  hmc / ess / sghmc / sgld / mclmc / smc / vi (gcol33/tulpa#75). These kernels
  previously fit a fixed-effect GLM only; a structured formula was routed away
  to the conditional logpost / nested-Laplace paths. They now sample the full
  latent vector `compute_param_layout()` lays out: random effects (intercept,
  slopes, correlated, multi-term), an areal spatial field (ICAR / BYM2), and a
  temporal field (RW1 / RW2 / AR1). The variance-component hyperparameters
  (`sigma_re` per term/coef, spatial `tau` / BYM2 `sigma`+`rho`, temporal `tau`
  / AR1 `rho`) are sampled JOINTLY with the latent and fixed effects -- full
  Bayes over the variance components, the exact-MCMC counterpart of the Laplace
  / logpost backends that condition on them. The four default-link families
  (gaussian / poisson / binomial / neg_binomial_2) get an analytic reverse-mode
  AD gradient; other families fall back to the numerical gradient, which still
  scores the full latent log-posterior. `tulpa(mode = "hmc" / "smc" / "vi" /
  ...)` reaches the new path for a structured formula; `ess` carries random
  effects but declines a structured spatial / temporal field (its isotropic
  Gaussian-prior block cannot hold the graph precision). The multi-term RE
  ModelData marshalling is now shared (`re_structure.h`) between the Laplace
  multi-RE fit and the sampler builder. Continuous spatial (gp / nngp / hsgp),
  CAR_proper, and SPDE fields keep their dedicated paths.

## 0.0.10 (2026-06-05)

* Fix a data race in the threaded sparse joint outer-grid nested-Laplace driver
  (`run_multi_block_nested_laplace_joint_sparse_impl`). The cell-coupling
  (coupled) arm's per-cell dispersion was read lock-free from the shared `arms`
  during the inner Newton solve while a concurrent grid cell's `prep_at_grid`
  rewrote it under the `nl_sparse_phi` critical -- every other arm already read
  a thread-local snapshot, but the coupled arm did not. On a gridded coupled-arm
  dispersion (e.g. the beta-cover precision on the phi axis) this corrupted
  per-cell values across `n_threads_outer > 1`, causing wrong log-marginals /
  non-convergence and the intermittent native crashes reported downstream
  (gcol33/tulpaObs#42). Now the coupled arms' dispersion is snapshotted under the
  same critical and read via a `phi_override` in the coupled scatter / log-lik;
  serial and dense callers pass `nullptr` and are byte-unchanged. Verified: a
  220-region BYM2 beta-cover fit is identical serial vs `n_threads_outer = 6` to
  ~1e-10.

## 0.0.9 (2026-06-03)

* feat(offset): thread `offset()` terms through the SPDE, spatial-Laplace, and
  ModelData-sampler paths of `tulpa()` (gcol33/tulpa#72). A fixed log-exposure /
  log-effort offset (the standard way to model rates) now enters the linear
  predictor `eta = offset + X beta + field` on every fitter, where it was
  previously honoured only on the non-spatial GLMM / EM paths and hard-errored
  on the rest. The offset is carried as the per-process `ProcessData::offset`
  (areal ICAR/CAR/BYM2 + NNGP Laplace and all seven sampler kernels --
  hmc/ess/sghmc/sgld/mclmc/smc/vi -- which already consumed it via
  `compute_eta_spec` / `precompute_generic_fixed_eta`), as a per-arm
  `ParsedArm::offset` in the nested-Laplace joint engine (the nested SPDE path),
  and as a raw additive vector in the single-point SPDE kernel. The three
  `tulpa()` guards are removed; `fit_spde()` and `tulpa_sample_glmm()` gain an
  `offset` argument.

* fix(spde): gate fractional Matern smoothness (`nu`) instead of silently
  fitting a mis-specified field (gcol33/tulpa#71). The wired rational assembly
  builds `Q = tau^2 sum_k w_k (L + r_k C)' C^{-1} (L + r_k C)`, whose spectral
  symbol `tau^2 sum_k w_k (l + r_k)^2` is a single quadratic in `l` for any
  number of poles -- so no choice of poles/weights recovers a fractional
  `alpha`; the construction collapses to an `alpha = 2` field regardless of the
  coefficients (verified to machine precision). The coefficient generator
  (`rational_spde_coefficients()`) previously returned a self-derived
  log-uniform approximation while documenting the published BRASIL / Bolin et
  al. (2023) method. `spatial_spde()` / `spatial_spde_custom()` now require an
  integer `nu` (0, 1, 2, ...) -- the exact FEM construction -- and reject
  fractional `nu` with a clear error across every fit path (Laplace, nested,
  NUTS, joint). A faithful rational SPDE precision assembly remains tracked
  under gcol33/tulpa#71.

* perf(nested-laplace-joint): budget the replicated per-outer-thread state
  (the sparse-Hessian builders + numeric factor) against detected physical RAM
  instead of a fixed 2 GB cap (gcol33/tulpa#64). The old cap clamped the outer
  grid to ~15 threads at EVA scale on a 64 GB box even when 28 were requested;
  a new standalone `sysmem` translation unit (`total_ram_bytes()`, Windows /
  macOS / POSIX) lets a wide field use every requested outer thread when the
  memory is there, falling back to the 2 GB cap only when the RAM query fails.

* fix(nested-laplace-joint): the outer-grid progress ETA is now computed from
  the realised parallel width, not a serial extrapolation of the pilot rate
  (gcol33/tulpa#64). A ~21 min serial pilot on a 48-cell grid previously
  projected `ETA ~16 h`; the ETA now estimates per-cell wall time from completed
  *waves* (one serial pilot wave + `(done-1)/width` parallel waves) and projects
  the remaining cells over `ceil(remaining/width)` waves. At outer width 1 this
  is exactly the previous serial formula.

* fix(nested-laplace): retire the unguarded `.nl_normalise_weights` softmax
  entirely -- every outer-grid weight normaliser now routes through the
  finite-guarded `.nl_normalise_weights_safe` (gcol33/tulpa#65). The remaining
  unguarded call sites (the single-block grid, the multi-block grid, the
  adaptive-refinement reweight, and the cheap-screen ESS gate) could still
  collapse `fit$weights` to all-NaN when one outer cell returned a non-finite
  `log_marginal`, breaking `tulpa_posterior_draws()` / `predict()` / WAIC on the
  finite, precision-carrying cells. `logLik()`'s grid log-sum-exp is guarded the
  same way so a non-finite cell no longer poisons the integrated evidence (and
  `AIC` / `compare_models` downstream). Behaviour is unchanged on all-finite grids.

* fix(nested-laplace-joint): the single-block joint path normalises the outer-grid
  weights with the NaN-safe `.nl_normalise_weights_safe` (drop non-finite cells,
  renormalise) instead of the bare `max(lm)` softmax. A single non-finite
  `log_marginal` (an inner solve that diverges at an extreme hyperparameter cell)
  previously poisoned the whole `weights` vector to NaN, so `theta_*` summaries
  had to work around it and `tulpa_posterior_draws()` failed with "no outer-grid
  cell has positive weight". Degenerate cells now get zero weight and the
  remaining cells carry the mass, matching the multi-block path and the
  hyper-grid path. Behaviour is unchanged on all-finite grids.

* feat(gauss-hermite): export `gauss_hermite.h` (probabilist Golub-Welsch nodes)
  under `inst/include/tulpa/` so `LinkingTo` consumer packages reuse the engine's
  one implementation instead of re-deriving the quadrature.

* feat(nested-laplace): the multi-block joint path announces the engaged outer
  integrator under `control$verbose = TRUE`, in one line at selection time
  before the inner solves (gcol33/tulpa#63): e.g. `outer integration: CCD
  (4 latent axes, 25 nodes)`, `tensor grid (72 cells)`, or `CCD declined ->
  tensor grid (72 cells)`. Previously the `"auto"` switch to the CCD at `>= 4`
  latent axes was silent -- a consumer who omitted `integration` from the
  control could end up on the CCD path (and, on a ridged posterior, in the
  #62 thrash) with no signal, the only post-hoc tell being `fit$...$integration`.
  The resolved integrator is still returned on the joint result as `$integration`.
* fix(nested-laplace): the joint multi-block CCD outer mode-find now declines
  **fast** on a flat / ridged hyperparameter posterior (gcol33/tulpa#62).
  Previously a sigma-alpha ridge produced a near-singular outer Hessian, a huge
  Newton step, and a deeply backtracking line search of full-field inner solves
  (hours, before the post-hoc guard could decline). The mode-find now (a)
  pre-checks the centre Hessian conditioning and declines to the tensor grid
  immediately on a ridge -- the same verdict, minus the line search; (b)
  trust-clamps the Newton step so a candidate never leaps to an extreme
  hyperparameter; (c) caps the line-search backtracking; and (d) advances the
  inner warm start to each accepted point so probes solve in a few Newton steps
  instead of cold from the box centre.
* feat(nested-laplace): `control$integration` for a multi-block joint prior
  gains `"auto"` (the new default) and now takes `"auto"` / `"ccd"` / `"grid"`
  (gcol33/tulpa#59). `"auto"` uses the CCD only at `>= 4` transformable axes,
  where the tensor product's `k^d` blow-up bites hardest, and keeps the cheaper,
  more ridge-robust tensor grid at `<= 3` axes; `"ccd"` lowers the CCD threshold
  to `>= 3` axes; `"grid"` always forces the tensor product. (Previously the
  default engaged the CCD at `>= 3` axes.)

## 0.0.8 (2026-06-02)

* feat(nested-laplace): CCD outer integration for the joint multi-block path
  (gcol33/tulpa#59). `control$integration = "ccd"` (default for >= 3
  transformable axes) integrates the joint hyperparameter posterior on a central
  composite design around its mode -- far fewer inner solves than the k^d tensor
  product. Auto-declines to the tensor grid for <= 2 axes, an unguessable axis
  (CAR_proper `rho_car` / non-BYM2 `rho`), or a degenerate mode-find.
* feat(nested-laplace): CCD now rides an active `phi_grid` (gcol33/tulpa#61).
  An active per-arm dispersion axis no longer disables CCD: the design is built
  over the `>= 3` latent axes and the `phi` tensor is Cartesian-crossed on top,
  with the CCD node weights replicated across the `phi` cells. A two-field beta
  `occu_cover()` / `cover()` joint fit (4 latent axes + a `phi` grid) integrates
  on `25 x phi` cells instead of the `81 x phi` dense tensor.
* feat(samplers): generic R fitter for NUTS + log-posterior kernels; negbin
  spatial (areal ICAR) Gibbs.
* refactor(spatial/joint/pg): share the ICAR prior (one structured quadratic +
  Besag sum-to-zero penalty across the dense / sparse paths), thread an optional
  informative per-coefficient Gaussian beta prior into the joint arms (breaks the
  occupancy psi-p identifiability ridge), and fix the AR1 gradient.
* fix(nested-laplace): exact sparse ICAR / BYM2 sum-to-zero (gcol33/tulpa#60).
  The Besag sum-to-zero penalty has a dense rank-1 (`1 1'`) Hessian; the sparse
  path previously stored only its diagonal, so `force_sparse` / large-field ICAR
  and BYM2 joint fits had a log-marginal off from the dense path by the missing
  rank-1 log-det term. Now exact, size-gated: small fields densify the field
  block and store `1 1'` directly; large fields fold the rank-1 in at solve time
  (Sherman-Morrison step + matrix-determinant-lemma log-det) via one reuse-solve
  per field. `TULPA_S2Z_DENSIFY_MAX` tunes the cutoff.
* chore: finish Phase D of the tulpaRatio migration -- remove the dead legacy
  ratio C++ (`gibbs_spatial*`, `hmc_zi.h`) and the trailing `ModelType` /
  `LegacyRatio` references in the exported headers; rename stale `numdenom`
  references to `tulpaRatio` in the docs.
* tests: `TULPA_FAST` dev tier (`skip_if_fast()`) runs the structural /
  closed-form / gradient unit tests only; the default still runs the full
  recovery suite.

## 0.0.7 (2026-06-01)

* feat: grid-cell / per-unit **checkpoint/resume across every fitter with an
  expensive outer loop** (gcol33/tulpa#50). A killed or rebooted fit reloads the
  completed units and runs only the rest. One content-addressed binary append
  log (`src/checkpoint_io.h`, `CheckpointLog<Payload>`) owns the file format,
  load/append/torn-tail logic and fingerprinting once; a torn final record is
  truncated and re-run, and a header fingerprint mismatch (different data /
  settings / grid) errors rather than resuming onto a stale result. Wired
  through:
  - the single-block nested-Laplace kernels (icar / bym2 / car_proper /
    temporal / the ST variants / nngp / hsgp), `tulpa_nested_laplace()`
    multi-block, and `fit_spde()` -- `control$checkpoint = list(path =,
    resume =)` (a `checkpoint =` arg on `fit_spde()`);
  - `tulpa_re_cov_nested()`'s CCD node integration (`checkpoint =` arg, an
    atomic-RDS node cache);
  - the per-chain NUTS producer (`cpp_tulpa_fit_generic_chains(checkpoint_path=)`):
    a chain is deterministic in `(seed, chain_id, data, settings)`, so a resumed
    chain is bit-for-bit identical to the uninterrupted one.
  Extends the joint-fit checkpoint shipped earlier to the rest of the engine;
  the joint path was refactored onto the shared core with no file-format change.
  Tests: `test-checkpoint-universal.R`.
* fix(nested-laplace): the outer Pareto-k accuracy diagnostic no longer
  dominates runtime or runs solves it then discards (gcol33/tulpa#51). Three
  changes, all on the diagnostic path -- the integration result is unchanged:
  - The importance-sampling cores decline up front when `k_samples` is below
    the GPD-fit floor (`.PSIS_MIN_EVAL`, 25): a sub-floor budget can never reach
    enough finite evaluations, so it now returns `NA` without paying a single
    inner solve instead of evaluating the whole budget and discarding it.
  - Each diagnostic re-evaluation solve is warm-started from the modal cell's
    converged latent mode, so the draws carrying the bulk of the importance
    weight converge in a few Newton steps rather than from a cold start.
  - The diagnostic solves cap their inner iterations at `min(max_iter, 25)`. A
    draw at an implausible hyperparameter (where the inner Newton would
    otherwise stall to the full budget) carries negligible importance weight, so
    the cap bounds its cost without moving the k-hat -- converged draws keep
    their exact log-marginal; only the negligible-weight tail is truncated.
  Covers `tulpa_nested_laplace_joint()` (single- and multi-block); the
  sub-floor decline also covers `tulpa_nested_laplace()`, `tulpa_re_cov_nested()`
  and `fit_spde()` through the shared PSIS cores.

## 0.0.6

* `tulpa_re_cov_nested()` documents its outer accuracy-diagnostic controls
  `diagnose_k` (default `TRUE`) and `k_samples` (default 200) in the help page.
* Requires tulpaMesh (>= 0.1.2), which fixes SPDE mesh construction returning a
  zero-triangle mesh on some constrained inputs (gcol33/tulpaMesh#3).
* The two-arm community N-mixture oracle equivalence/recovery test moves to its
  model-adapter home in tulpaObs (`ms_abun`); the generic engine retains the
  structure-agnostic `make_site`/`make_group` equivalence checks.

## 0.0.5

* feat(nested-laplace): fits now record wall-clock runtime on the returned
  object as `fit$timing` (gcol33/tulpa#48). A named numeric of seconds carrying
  `total` plus a phase breakdown -- `setup` (validation / encoding / grid
  construction), `grid` (the inner Laplace solves that scale with grid size and
  core count, including adaptive-refinement and consistency passes), `postproc`
  (weight / moment / marginal assembly), and `diagnostics` (the outer
  Pareto-k-hat). Covers `tulpa_nested_laplace()` (single- and multi-block) and
  `tulpa_nested_laplace_joint()` (single- and multi-block dispatch); consumer
  fits riding on the joint object inherit it. A new `print` method for the
  nested-Laplace classes surfaces a one-line summary (`"fit in 5h 25m (grid 2h
  09m)"`) alongside the hyperparameters, grid size, and outer Pareto-k-hat.
* perf(nested-laplace): the **sparse** joint outer grid now runs in parallel
  under `control$n_threads_outer` (gcol33/tulpa#46, lever 2). It previously
  forced a serial outer loop, so on a large field only the inner per-observation
  OpenMP parallelised and most cores idled. The sparse driver now allocates a
  per-outer-thread pool (Hessian builder, Newton scratch, arm specs, scatter
  index cache, DENSE_BASIS scratch) and dispatches grid cells across
  `n_threads_outer`, matching the dense driver. The phi-grid dispersion axis is
  parallel-safe: it rewrites the shared `arms` dispersion per cell, so each cell
  snapshots it into its own thread's specs under a short critical before the
  lock-free Newton solve. A memory guard clamps the thread count when the
  replicated builders would be too large (very wide fields fall back to fewer
  outer threads). Parallel-vs-serial parity (with and without a phi axis) is
  covered in `tests/testthat/test-nested-laplace-joint-sparse-parallel.R`.
* perf(nested-laplace): grad-only inner-Newton steps under `inner_refresh`
  (gcol33/tulpa#46, lever 1b). On a factor-reuse step the Hessian is discarded,
  so the cell-coupling scatter now passes a `grad_only` request to the
  `CellCouplingSpec` (a new ABI-appended `CellDerivs::grad_only`): a spec may
  skip its negative-Hessian work (e.g. a beta arm's digamma/trigamma) and emit
  only the exact gradient, and the kernel skips the cross-arm Hessian scatter.
  Specs that do not implement it write the full Hessian as before (correct, no
  saving). The gradient stays exact on every step, so the converged mode is
  unchanged -- validated by a grad-only-honoring cell-coupling reuse test in
  `tests/testthat/test-nested-laplace-joint-inner-refresh.R`.
* perf(nested-laplace): `control$inner_refresh` (default `1L`) adds
  Shamanskii / chord-method Cholesky factor reuse to the sparse joint inner
  Newton (gcol33/tulpa#46). For a non-quadratic positive arm (e.g. a beta cover
  arm) the latent Hessian changes every inner iteration, so the default
  re-factorizes the sparse Cholesky on each step -- the dominant per-grid-cell
  cost. `inner_refresh = m > 1` re-factorizes only every `m`-th inner step and
  reuses the cached factor in between (refreshing early if a reused solve
  fails). The gradient is exact on every step and each step is line-search
  safeguarded, so the converged mode is unchanged and the final mode-pass
  Hessian (`log_det`, SEs) is always fresh -- only the path to the mode uses a
  stale curvature. Applies to the sparse LM path; the dense small-`n_x` path
  re-factorizes cheaply and ignores it. Bit-equivalence to the every-step
  default is covered in `tests/testthat/test-nested-laplace-joint-inner-refresh.R`.
* feat(nested-laplace): `tulpa_posterior_draws(fit, idx, n)` -- a generic
  posterior sampler for the grid-integrated joint nested-Laplace backend (the
  `inla.posterior.sample()` analogue, gcol33/tulpa#44). Draws from the outer-grid
  mixture `sum_k w_k N(m_k, V_k)`: each draw picks a grid cell from the
  integration weights, then samples the inner latent vector from that cell's
  constrained Gaussian via the stored sparse precision `Q_csc_*_per_grid`
  (requires `control$store_Q = TRUE`). The ICAR / BYM2 field sum-to-zero
  constraint is imposed by conditioning on kriging (Rue & Held 2005), so the
  per-cell marginal matches the constrained inner-Laplace covariance exactly;
  single-block and multi-block (multi-field trend) layouts are both handled.
  Sampling the mixture -- rather than a single moment-matched Gaussian -- is the
  faithful primitive for marginalizing nonlinear derived quantities (change in
  occupancy, expected-cover products). Draws are tagged `iid`. Tests in
  `tests/testthat/test-posterior-draws-joint.R`.
* fix(nested-laplace): the joint outer-grid cheap-pass prune
  (`control$prune = TRUE`) no longer mis-ranks grid cells or drops the true
  posterior mode (gcol33/tulpa#43). The screen previously ran a single Newton
  step from one global pilot mode for every cell; when the inner latent mode
  moves substantially across the outer grid (large spatial fields, wide
  sigma/rho/alpha ranges) the one-step approximation mis-estimated far cells by
  O(1e5) log-units and inverted the ranking, so the prune could skip the full
  solve on the actual mode. The screen is now a rank-faithful chained sweep over
  the lattice: each cell runs a short Newton run warm-started from the previous
  screened cell's quasi-mode, so every cheap mode stays near its cell's true
  mode and the cheap ranking agrees with the full-solve ranking.
* fix(nested-laplace): added a safety gate to the joint cheap-pass prune. If the
  cheap-screen argmax disagrees with the full-solve argmax, or the kept
  posterior collapses onto a cell whose cheap-vs-full log-marginal gap is large,
  the fitter warns and falls back to the full grid (`$prune_fallback_triggered`,
  `$prune_fallback_reason`) rather than silently returning a pruned answer. A
  silently-wrong pruned posterior is now impossible.
* feat(spde): `fit_spde()` reports an outer Pareto-k-hat accuracy diagnostic
  (`$pareto_k`) over the integrated `(range, sigma)` hyperparameters -- the
  iid-fit counterpart of Rhat. k-hat < 0.7 means the Gaussian proposal the
  integrator orients its CCD/grid with fits the hyperparameter posterior;
  >= 0.7 flags a skewed / heavy-tailed posterior the grid misfits. Controlled
  by `diagnose_k` (default TRUE) / `k_samples` (default 200), RNG-restored so
  the fit's draws are unchanged.
* feat(nested-laplace): the joint backend (`tulpa_nested_laplace_joint`,
  single- and multi-block) reports the same outer Pareto-k-hat over its
  heterogeneous hyperparameter space (gcol33/tulpa#42). A block-type-aware
  per-axis transform unconstrains each axis -- positive scales by `log`, the
  BYM2 mixing weight by logit, the copy coefficient by identity -- and the
  summed log-Jacobians enter the importance target. A CAR_proper `rho_car` axis
  (support is the adjacency eigenvalue interval, not guessable) declines to
  quadrature ESS rather than apply a wrong transform.
* feat(nested-laplace): `control$hessian` selects the inner-Newton curvature for
  the joint mixture Hessian -- `"lm"` (default, diagonal-ridge escalation until
  CHOLMOD factorizes), `"psd"` (eigen-clamp the dense observed Hessian), or
  `"fisher"` (complete-data expected information, PSD by construction).
* feat(joint-laplace): cross-arm coupling via a `CellCouplingSpec` registry
  (gcol33/tulpa#32). Consumer packages register a per-cell coupling spec (e.g.
  tulpaObs's cover-hurdle) so two arms share a latent field; the cross-arm block
  is assembled into the joint Hessian (`tulpa_register_cell_coupling` C
  callable, default `"separable"` always available).

## 0.0.2

* feat: `tulpa_re_aghq()` -- a callback-driven adaptive Gauss-Hermite refinement
  of a grouped random-effect covariance. Generalizes `agq_fit()` (intercept-only
  RE, built-in `binomial`/`poisson`/`gaussian`) to **random slopes and
  correlated multi-coefficient blocks** sharing one grouping factor, with the
  per-observation marginal likelihood supplied by the caller through a
  `make_site` callback. This lets a downstream package refine a custom marginal
  (e.g. a latent-state-integrated occupancy / detection likelihood) through the
  same quadrature. Reuses the existing log-Cholesky covariance parametrization
  (`.re_cov_*`), `gauss_hermite_prob()`, and an optional LKJ correlation penalty;
  the fixed parameters and `chol(Sigma)` are optimized jointly on the
  exact-marginal log-likelihood, with SEs from the marginal Hessian. Recovery
  tests in `tests/testthat/test-re-aghq.R`.

## 0.0.3 (2026-05-28)

* feat(nested-laplace): `tulpa_nested_laplace_joint()` now reports the outer
  Pareto-k-hat accuracy diagnostic (`$pareto_k`, `$pareto_k_is_ess`) over its
  heterogeneous hyperparameter space, completing the nested-Laplace k-hat family
  alongside the re-cov, generic single-axis, and SPDE paths (gcol33/tulpa#42). A
  block-type-aware per-axis transform registry unconstrains each axis -- positive
  scales (`sigma`, `tau`, `phi_*`, ...) by `log`, the BYM2 mixing weight (`rho`)
  by logit, the copy coefficient (`alpha`) by identity -- with the summed
  log-Jacobians in the importance target; the inner marginal is re-evaluated
  through the same kernel the integrator used. A fit carrying an axis whose
  support is the adjacency eigenvalue interval (`CAR_proper`'s `rho_car`) declines
  to the quadrature-ESS fallback rather than apply a guessed transform. Gated by
  `control$diagnose_k` (default `TRUE`) / `control$k_samples` (`200`), RNG-restored
  so draws are unchanged. Recovery + plumbing tests in
  `tests/testthat/test-nested-laplace-joint-pareto-k.R`.

* refactor(aghq): one compiled adaptive-Gauss-Hermite engine behind the whole
  ML-II optimize family. The per-group marginal -- mode-find, quadrature grid,
  log-Cholesky `Sigma` packing, LKJ penalty and marginal Hessian -- now lives in
  C++ (`src/aghq_re*.{h,cpp}`, `inst/include/tulpa/aghq_oracle.h`), driven through
  one structure-agnostic per-group oracle. `tulpa_re_aghq()` and `agq_fit()` are
  thin wrappers over it (their R integration loops are gone; the optimizer takes
  finite differences of the compiled objective, consistent at every `n_quad`).
  The mode-find is a globally-convergent modified Newton (prefers the true
  observed-info Hessian where PD; falls back to a caller-supplied PSD Fisher or an
  eigenvalue-reflected curvature otherwise), so a latent-variable marginal whose
  observed information is indefinite away from the mode no longer breaks it.
  `tulpa_re_aghq()` gains `theta_prior_sd` (a Gaussian ridge on the fixed
  parameters) and returns `log_marginal`.

* refactor(aghq): `agq_fit()` builds its per-group marginal from the native
  GLMM oracle (`cpp_glmm_oracle_make`, `src/glmm_oracle.h`) instead of an
  R-closure oracle over an R family density. The built-in `binomial` /
  `poisson` / `gaussian` densities now have a single C++ source of truth shared
  with `tulpa_re_aghq()`, `tulpa_re_cov_nested(n_quad > 1)` and the Gibbs sweep;
  this removes `.agq_loglik_elt()` / `.agq_score_info()`. Estimates,
  covariances and `n_quad`-convergence are unchanged; the reported
  `log_marginal` now carries the full likelihood normalizing constants (the
  binomial coefficient and Poisson `lgamma` the R density previously dropped),
  matching the other AGHQ fitters.

* refactor(gibbs): the exact-target random-effect-covariance sampler
  (`tulpa_re_cov_gibbs()`) runs its Metropolis-within-Gibbs sweep in compiled
  code (`src/re_cov_gibbs.cpp`, `src/re_cov_gibbs_sweep.h`), driven by one native
  per-row GLMM likelihood (`src/glmm_oracle.h`) rather than a duplicated R
  density. This removes `.re_obs_loglik`: the family densities (binomial /
  poisson / gaussian / negative-binomial-2) now have a single source, and the
  engine owns the shared linear predictor with the cross-block eta coupling for
  several terms. The estimator is unchanged -- the C++ sweep keeps the R
  sampler's RNG-draw order, so a seeded run reproduces the previous sampler's
  draws bit-for-bit (verified across the correlated, diagonal and multi-term
  `test-re-cov-gibbs.R` cases). The R wrapper keeps the pilot Laplace solve
  (starting values + proposal shapes) and the weighted-quantile summary.

* feat(re-cov): `tulpa_re_cov_nested()` gains `n_quad` -- an adaptive
  Gauss-Hermite refinement of the inner marginal. `n_quad = 1` (default) keeps
  the joint-field Laplace inner solve unchanged; `n_quad > 1` routes the inner
  solve through the shared compiled AGHQ engine (`cpp_glmm_oracle_make` +
  `cpp_aghq_objective`), so each per-group integral inside the `Sigma`
  integration is debiased by quadrature (the `tulpa_re_aghq()` correction applied
  under the grid), reducing the small-cluster variance attenuation for binary /
  low-count data. The fixed effects are integrated (profiled out + a fixed-effect
  Laplace term), so the reported fixed-effect posterior is the marginal (ML-II)
  one rather than the joint-mode (PQL) estimate. The per-group integral only
  factorizes over one shared grouping factor, so AGHQ requires that; crossed RE
  terms keep the joint-field Laplace (`n_quad > 1` errors). Recovery + the
  crossed-factor guard in `test-re-cov-nested.R`.

* feat(nmix): community / multispecies N-mixture (`tulpa_nmix_laplace_re()`, the
  spAbundance `msNMix` model) now fits through that shared engine -- it wraps
  `tulpa_nmix_site_marginal()` as the per-species oracle (the marginal, the
  abundance/detection score, and both the observed-info block with the
  `Var[N|y]` coupling and the PSD complete-data Fisher for the mode-find) and
  integrates the per-species coefficient random effects at `n_quad = 1` (joint
  Laplace). This replaces and removes the bespoke C++ Laplace-EM
  (`src/nmix_re.cpp`); the community fit verified against an independent Laplace
  marginal and recovered over seeds (`tests/testthat/test-nmix-re.R`).
  Fixed-effect SEs are now the joint marginal Hessian (marginalizing the
  community-covariance uncertainty, closer to the spAbundance posterior) rather
  than a `Sigma`-plug-in Schur complement. The numerical knobs `tol` / `inner_max`
  / `inner_tol` are dropped (the engine owns the mode-find); `sigma_beta` is kept
  as the fixed-effect ridge. Poisson only for now.

* feat(nmix): `tulpa_nmix_site_marginal()` exposes the per-site N-mixture
  marginal as a composable random-effect callback (`eval` / `eval_beta` /
  `obs_info_block`), and `tulpa_re_aghq()` gained a `make_group` path for the
  general / multi-arm case (a per-group `b`-space oracle), so a custom marginal
  with coupled arms at different granularities -- the abundance / detection arms
  of an N-mixture site sharing a species grouping -- integrates through the same
  quadrature. (The C++ `tulpa_nmix_laplace_re()` above is the production fitter;
  this is the composable / AGHQ-refinement path.)

* perf(nmix): the per-site kernel (`nmix_kernel.h`) caches its eta-independent
  `lgamma` combinatorial terms (`NMixSiteCache` / `nmix_precompute_site` /
  `compute_nmix_site_cached`), so an iterative fitter that evaluates a site many
  times at changing linear predictors skips the `lgamma` recompute (the dominant
  cost). The single-shot `compute_nmix_site()` Poisson path delegates to the
  cached helper -- single source of truth -- so existing single-species / spatial
  fits are byte-identical (nmix regression suite unchanged).

* refactor(nested-laplace)!: collapsed the 3 single-block temporal entries
  (`*_{rw1,rw2,ar1}`) to one `*_temporal` entry that selects the kernel at
  runtime via a `temporal_type` argument through the shared `make_temporal_ops`
  registry -- the same collapse the spatio-temporal entries already use, so
  adding a temporal kernel is O(1) at every layer (Rcpp entry, extern-C shim,
  exported ABI). **ABI break** (`TULPA_ABI_VERSION` 26 -> 27):
  `tulpa_nested_laplace_{rw1,rw2,ar1}` + their `NestedLaplace{Rw1,Rw2,Ar1}Fn`
  typedefs become `tulpa_nested_laplace_temporal` / `NestedLaplaceTemporalFn`;
  downstream packages rebuild. The R-level `rw1` / `rw2` / `ar1` block types are
  unchanged.

* refactor(laplace)!: removed the 8 dead family-enum single-point Laplace
  C-callables (`tulpa_laplace_mode_{dense,spatial,dense_multi_re,bym2,gp,
  multiscale_gp,multiscale_temporal,rsr}`) and their `LaplaceMode*Fn` typedefs.
  No package consumes them -- every model package routes single-point Laplace
  through the `LikelihoodSpec` path (`tulpa_laplace_spec_*`). **ABI break**
  (`TULPA_ABI_VERSION` 25 -> 26); downstream packages must rebuild. The shared
  `LaplaceShimResult` POD is retained (reused by the spec shims).

* refactor(nested-laplace)!: collapsed the 15 spatio-temporal nested-Laplace
  entries (`*_st_<spatial>_<temporal>`) to 5 per-spatial-family entries
  (`*_st_{icar,car_proper,bym2,hsgp,nngp}`) that select the temporal kernel
  (rw1 / rw2 / ar1) at runtime via a `temporal_type` argument, dispatched
  through a single `make_temporal_ops` registry. Adding a temporal kernel is
  now O(1) -- one registry branch, no new cross-product function at any layer
  (Rcpp entry, extern-C shim, or exported ABI). **ABI break** (`TULPA_ABI_VERSION`
  24 -> 25): the 15 `tulpa_nested_laplace_st_*` registered callables + their
  `NestedLaplaceSt*Fn` typedefs became 5; downstream packages must rebuild.
  Dense/sparse per-kernel equivalence preserved (`test-nested-laplace-st-sparse-equivalence.R`).

* feat(nmix): `tulpa_nmix_laplace()` gains `mixture = c("P", "NB")` -- a
  negative-binomial abundance mixing distribution
  (`N_i ~ NegBin(mean = lambda_i, size = r)`, `neg_binomial_2` convention) in
  addition to the Royle (2004) Poisson kernel. The per-site marginal, its
  scores (including the analytic dispersion score `d log L / d log r`), and the
  full joint observed-information Hessian are closed form; the dispersion
  `log_r` is profiled by block coordinate ascent outside the inner beta-Newton
  and reported with its standard error in `vcov`. Matches
  `unmarked::pcount(mixture = "NB")` on coefficients, log-likelihood, and
  standard errors to machine precision, with the usual analytic-derivative
  speed advantage. Poisson remains the default and is unchanged.

* feat(nmix): the spatial nested-Laplace N-mixture fits
  (`tulpa_nmix_laplace_icar()`, `tulpa_nmix_laplace_car_proper()`,
  `tulpa_nmix_laplace_bym2()`) gain `mixture = "NB"`. The NB size `r` is
  integrated as an additional outer grid dimension alongside the spatial
  hyperparameters (`tau` / `rho` / `sigma`); the posterior `r_mean` / `r_sd`
  are reported from the grid weights. The inner `(beta, z)` / `(beta, v, w)`
  Newton is unchanged in dimension -- only the likelihood pieces and the
  NB-aware `Var[N|y]` rank-1 correction depend on `r`. Poisson remains the
  default with identical behaviour and grid shape.

* refactor(api): `tulpa_nested_laplace()` and `tulpa_nested_laplace_joint()`
  collapse their perf/numerical knobs into a single `control = list()` argument,
  matching `tulpa()`. The top-level signatures now carry only statistical
  arguments (`y`/`n_trials`/`X`/`prior`/`spec`/`family`/`phi`/`likelihood`/...;
  `responses`/`prior`/`copy`/`phi_grid`/`prior_sigma`/`prior_alpha`). Tuning
  knobs move into `control`: single-arm `max_iter`, `tol`, `n_threads`, `x_init`,
  `keep_grid_hessians`; joint additionally `n_threads_outer`, `tile_warm`,
  `prune`, `prune_tol`, `store_Q`, `adaptive_grid`,
  `adaptive_grid_edge_thresh`, `adaptive_grid_max_passes`,
  `var_of_means_consistency`, `force_sparse`, `verbose`. Pre-release breaking
  change -- pass these inside `control = list(...)` (no deprecation shim). The
  dead single-arm `verbose` knob was dropped. Internal callers (`em_laplace`,
  the tgmrf pilots) and the shipped examples were migrated.

* feat(laplace): `tulpa_laplace_beta()` gains a `beta_prior` argument, forwarded
  to the inner `tulpa_laplace()` fits (both the outer `phi` search and the final
  refit) so the beta arm can carry a Gaussian fixed-effect penalty. Pure R
  passthrough -- `tulpa_laplace()` already applied `beta_prior` for
  `family = "beta"`. Enables penalised beta-regression arms downstream
  (tulpaObs `cover_priors()` positive arm). Rejected with `spatial`, matching
  `tulpa_laplace()`.

* feat(laplace): correlated random slopes `(1 + x | g)` on the Laplace engine
  (gcol33/tulpa#28). `tulpa_laplace()` RE terms accept a per-term covariance via
  `L` (lower-triangular Cholesky, `Sigma = L L'`) or `cov`; the off-diagonal now
  enters both the joint Hessian (mode finding) and the marginal fixed-effect SE.
  Previously a multi-coefficient RE block could only carry a per-coefficient
  marginal-sigma vector (a diagonal covariance), so `(1 + x | g)` was
  inexpressible under Laplace and downstream packages routed it to NUTS. The C++
  multi-RE kernel already consumed a packed Cholesky; this wires the R API to it
  through a single `.re_cov_spec()` helper and rebuilds the marginal Schur
  complement with a block-diagonal precision. It also fixes a pre-existing bug
  in the marginal-SE linear predictor -- the eta reconstruction treated every RE
  term as intercept-only, so the returned `H_beta` silently ignored random
  slopes (this affected `(x || g)` as well). Validated against an independent
  full-precision Schur in `tests/testthat/test-laplace-corr-re.R`. Estimating the
  covariance itself (the EM M-step for a full `Sigma`) is the follow-up; the
  engine now fits correlated slopes at a supplied covariance.

* feat(laplace): `tulpa_laplace(return_re_cov = TRUE)` returns per-group
  posterior covariance blocks `cov_blocks` -- one `n_coefs x n_coefs` matrix per
  (RE term, group) in term-major then group order, each a diagonal block of the
  *full* inverse Hessian (fixed effects and other groups marginalized out), i.e.
  `Cov(u_g | y, Sigma)`, not the inverse of a diagonal block. Built by reusing
  the Cholesky factor from the log-determinant (one back-solve per block column,
  no refactorization). This is the primitive a full-covariance EM M-step
  consumes to update `Sigma_k <- mean_g [u_g u_g' + Cov(u_g)]`; tulpaObs#11 uses
  it to fit `(1 + x | g)` deterministically. Non-spatial multi-RE path only
  (rejected with `spatial`).

* feat(nuts): expose tulpa's across-chain OpenMP runner through the model-facing
  C ABI (gcol33/tulpa#30). New registered callable `tulpa_run_nuts_chains`
  (header accessor `tulpa::get_nuts_chains_fn()`) runs `n_chains` chains in one
  call and fills a caller-allocated array of `NUTSResult`, so downstream packages
  stop re-implementing chain orchestration (offset-seed loops / PSOCK clusters)
  in R and get the engine's thread-parallel path for free. `init` and the
  optional `inv_metric_diag` are chain-major `[n_chains * n_params]`, so a fresh
  fit broadcasts one init while a resume passes each chain's `final_position` +
  `inv_metric_out` (with `n_warmup = 0`) — composing with #29 to continue a whole
  multi-chain fit. The OpenMP loop now lives in one pure-C++ core
  (`run_hmc_parallel_chains_cpp`) shared by the C ABI and the existing
  Rcpp-returning `run_hmc_parallel_chains`. New generic R entry point
  `cpp_tulpa_fit_generic_chains()` returns draws stacked chain-major with a
  `chain_id` vector — the layout `mcmc_diagnostics()` (#26) consumes directly —
  plus per-chain `epsilon` / `inv_metric` / `final_position`. Validated in
  `tests/testthat/test-generic-sampler.R`, including a cross-chain Rhat/ESS check
  through `mcmc_diagnostics()`. **ABI bump 23 -> 24** (new callable only; no
  struct layout change).

* feat(nuts): the NUTS C-ABI now returns the state needed to resume or
  warm-start a chain (gcol33/tulpa#29). `NUTSResult` gains `inv_metric_out`
  (the adapted inverse-mass diagonal at end of warmup) and `final_position`
  (the last raw sampler state); `epsilon` was already returned. Feeding them
  back as `init` + `inv_metric_diag` with `n_warmup = 0` continues the chain
  from the previous fit's geometry instead of rediscovering it. The inputs
  already existed on `NUTSFn`; only the result fields were missing. The
  generic R entry point `cpp_tulpa_fit_generic()` gains optional `init` /
  `inv_metric_init` arguments and returns `inv_metric` / `final_position`,
  exercised in `tests/testthat/test-generic-sampler.R`. **ABI bump 22 -> 23**
  (two trailing pointers appended to `NUTSResult`; `NUTSFn` unchanged).

* feat(diagnostics): extend the native MCMC convergence surface
  (`R/convergence.R`) toward posterior parity (gcol33/tulpa#26).
  `mcmc_diagnostics()` gains `measures` and `probs` arguments selecting from
  improved `rhat` (now the maximum of rank-normalized split-Rhat and folded
  split-Rhat, matching `posterior::rhat`), `rhat_bulk`, `rhat_fold`,
  `ess_bulk`, `ess_tail`, `ess_mean`, `ess_sd`, `mcse_mean`, `mcse_sd`, and
  per-probability `ess_quantile` / `mcse_quantile`. The default columns
  (`rhat`, `ess_bulk`, `ess_tail`) are unchanged. New measures are registered
  in one table (`.tulpa_diag_measures`), so adding a statistic is a one-line
  change. Two estimator bugs are fixed so the native code reproduces
  `posterior` to machine precision (~1e-12): the rank-normalization divisor is
  now the Blom `S + 1/4` (was `S - 1/4`) and the Geyer `tau_hat` tail term no
  longer double-counts. New exported helpers: `tulpa_draws_array()` (an
  `as_draws_array()`-style `[iter, chain, param]` accessor), `n_divergent()`,
  and `check_diagnostics()`. The plotting / summary layer (`plot_rhat`,
  `plot_ess`, `diagnostic_summary`, `plot_diagnostics`, `plot_acf`,
  `plot_pairs`) now resolves the previously undefined `get_draws_array()`,
  `grep_params()`, and `n_divergent()` helpers and runs end-to-end on a
  multi-chain fit. Validated in `tests/testthat/test-convergence.R`.

* feat(re): random-effect blocks support an optional per-term intercept.
  `ModelData::re_has_intercept` (default: all terms carry the implicit group
  intercept) lets a term be slope-only (lme4 `(0 + x | g)`): every
  `re_n_coefs[t]` coefficient is a slope read from the slope design matrix and
  there is no `z = 1` column. The change is threaded through the design lookup
  (`slope_at()` / `obs_re_contrib()` in `laplace_spec.cpp`) and the autodiff
  RE contribution (`log_post_generic_impl.h`), so the value and its gradient
  stay consistent. `re_term_has_intercept()` (in `model_data.h`) centralises
  the per-term test. **ABI bump 21 -> 22** (new `ModelData` field). Enables
  gcol33/tulpaObs#10 slope-only bar syntax.

* fix(spatial): `spatial_car()` / `spatial_bym2()` /
  `spatial_car_proper()` with `level = "group"` now accept datasets that
  cover only a subset of adjacency cells. Closes gcol33/tulpa#25. The
  Besag / ICAR / BYM2 / proper-CAR field is well-defined on every node
  of the graph regardless of whether each node has an observation;
  unobserved cells simply contribute no likelihood term (matching
  INLA's `f(cell, model = "besag", graph = g)`). `validate_spatial()`
  and `prior_from_spec()` now resolve `group_var` to 1-based adjacency
  row indices via a new `.resolve_spatial_idx()` helper:
  - integer / numeric `group_var` -> 1-based row indices,
    validated against `[1, n_spatial_units]`.
  - character / factor `group_var` with `rownames(adjacency)` set ->
    matched by name (preserves cell identity for sparse subsets).
  - character / factor `group_var` without rownames -> legacy
    `as.integer(as.factor(.))`, retained for back-compat; errors with
    an actionable message when level count differs from adjacency
    size.

* refactor(joint-laplace): unify single-block and multi-block joint
  dispatch (Phase J-E). `tulpa_nested_laplace_joint()`'s single-block
  path (`prior = list(type = "bym2"/"icar"/"car_proper", ...)`) now
  packs the prior into a length-1 `blocks_spec` and dispatches through
  the same `cpp_nested_laplace_joint_multi` entry that drives the
  list-of-blocks API. The three legacy `cpp_nested_laplace_joint_bym2`
  / `_icar` / `_car_proper` R-facing wrappers (524 lines in
  `src/nested_laplace_joint.cpp`) are deleted; the inner Newton driver
  (`run_multi_block_nested_laplace_joint`) was already shared, so the
  refactor is a routing change with bit-identical log_marginal on every
  joint test (147/147 pass). User-facing R API and result shape are
  unchanged. **C ABI:** the `tulpa_nested_laplace_joint_bym2` shim
  (`R_RegisterCCallable` entry in `tulpa_shims.cpp`) is removed;
  external embedders should call `tulpa_nested_laplace_joint()` from
  R or build a shim on top of `cpp_nested_laplace_joint_multi`.
  `TULPA_ABI_VERSION` bumped **19 → 20**.

* fix: joint nested-Laplace reparam `(sigma, alpha)` → `(sigma_occ, sigma_pos)`.
  Closes gcol33/tulpa#18. The BYM2 / ICAR / CAR_proper backends of
  `tulpa_nested_laplace_joint()` previously parameterized the joint
  outer grid as `(sigma, rho/rho_car, alpha)`, where `sigma` was the
  shared field amplitude and `alpha` scaled the copy arm's contribution.
  At small `n_pos` and low cover-arm sample fraction (e.g. d7 Cell B,
  `n_s = 25`, `n_pos ≈ 46`), the cover-arm likelihood identified only the
  *product* `alpha * sigma`; sigma was pulled toward its prior and alpha
  inflated to compensate (~ −15% sigma bias, ~ +27% alpha bias on 30
  seeds). The reparam scales each arm's contribution to a unit-precision
  latent by its own sigma — `eta_arm = X beta + sigma_arm * z_s`, with
  `sigma_arm = sigma_occ` on donor arms and `sigma_arm = sigma_pos` on
  the copy arm. Each axis is now anchored by its own arm's likelihood,
  so the posterior ridge along constant `alpha * sigma` disappears.
  `alpha = sigma_pos / sigma_occ` is recovered post-hoc and attached to
  `theta_grid` / `theta_mean` / `theta_sd`. **API:** `copy$alpha_grid` is
  superseded by `copy$sigma_pos_grid`; `alpha_grid` still works with a
  deprecation warning that translates it to
  `alpha_grid * median(prior$sigma_grid)`. ICAR / CAR_proper joint
  kernels now take `sigma_grid` (donor sigma in sigma-space) instead of
  `tau_grid`; tulpaObs callers translate `tau_grid` to
  `sigma_grid = 1/sqrt(tau_grid)` internally. **C ABI:**
  `tulpa_nested_laplace_joint_bym2_impl` switches its grid args from
  `sigma_spatial_grid` / `alpha_grid` to `sigma_occ_grid` /
  `sigma_pos_grid`. `TULPA_ABI_VERSION` bumped **18 → 19**.

* feat: EM+Laplace MI and Gibbs corrections. `tulpa_em_laplace()` gains two
  post-EM correction modes (`correction = "mi"` / `"gibbs"`) that replace
  the previous "not yet implemented" stub. MI draws `n_imputations` hard
  `z`'s from the converged posterior weights `P(z|y, theta_hat)`, refits
  each block on the hard draws, and pools per-submodel coefficients via
  `rubins_pool()`. Gibbs runs a warm-started `z|theta -> theta|z` Markov
  chain of length `n_gibbs` starting from the EM fits — every step
  refreshes weights via the user's `e_step`, draws hard z, refits — and
  pools the chain via Rubin's rules. The fixed-effect `(beta, se)`
  extraction is now a shared helper (`.attach_beta_se`) consumed by both
  the new corrections and the existing `tulpa_em_mc()` MCEM driver, so
  there is one source of truth for "Laplace fit -> Rubin pool input".
  Bernoulli is the default per-observation hard-z draw; multi-class
  latent structures supply their own via the new `draw_z` callback.
  Return shape gains `correction`, `pooled`, and `draws` fields when a
  correction is requested. No ABI bump (R-side only); closes
  `TODO.md` P3.8.

* feat: ABI v14 — SPDE nested-Laplace upgraded to the v10-style universal
  shim (store_modes, store_Q, paired range/sigma grids, formula-side iid-RE
  block). Replaces the v0 `cpp_nested_laplace_spde` entry and folds the
  SPDE C-callable into the shared `NestedLaplaceShimResult` block used by
  ICAR / BYM2 / NNGP / HSGP. The dedicated `SpdeNestedLaplaceShimResult`
  struct is removed. Latent layout:
  `[beta (p)] [re (n_re_groups)] [w_mesh (n_mesh)]`.
  `TULPA_ABI_VERSION` bumped **13 → 14**; downstream packages must rebuild.

## 2026-05-13 — ABI v13: Phase D — delete legacy ratio path

Closes the tulpaRatio migration tracker (gcol33/tulpa#15). After v12
gated the legacy ratio body of `compute_log_post_impl<T>` behind a
generic-layout check, Phase D removes the body itself and every
consumer of `LegacyRatioData` / `LegacyRatioLayout`.

* `TULPA_ABI_VERSION` bumped **12 → 13**. Downstream packages must
  rebuild against the v13 headers.
* **Removed exported types.** `ModelData::LegacyRatioData legacy`
  (`inst/include/tulpa/model_data.h`) and
  `ParamLayout::LegacyRatioLayout legacy`
  (`inst/include/tulpa/param_layout.h`) are gone. `n_processes > 0`
  with a non-null `data.likelihood_spec` is now the only supported
  configuration.
* **Removed Rcpp entry points** (D-1): `cpp_hmc_fit`, `cpp_hmc_fit_gp`,
  `cpp_hmc_fit_gp_v2`, `cpp_ess_fit`, `cpp_ess_get_n_params`,
  `cpp_vi_fit`, `cpp_vi_get_n_params`, `cpp_sghmc_fit`, `cpp_sgld_fit`,
  `cpp_compute_log_post_test`, `cpp_compute_log_prior_test`,
  `cpp_compute_log_lik_only_test`, `cpp_log_post_split_n_params`.
  Internal samplers (`run_ess_sampler`, `run_sghmc_sampler`, `fit_vi`,
  `run_mclmc_sampler`) and their C-callable shims
  (`tulpa_run_ess_sampler`, `tulpa_sghmc_fit`, `tulpa_fit_vi`,
  `tulpa_mclmc_fit`) remain — downstream packages reach them via the
  generic ModelData/ParamLayout API. Dev tools
  `tools/icar_collapsed_check.R` and `tools/bym2_gradient_check.R`
  also removed.
* **Removed dispatcher branches** (D-2). `resolve_gradient_fn`
  (`src/hmc_gradient_dispatch.h`) now only resolves the generic
  `spec->gradient_fn` / `compute_gradient_generic_arena` /
  `compute_gradient_generic_numerical` paths. Mode overrides
  (`AUTODIFF_TAPE`, `AUTODIFF_ARENA`, `AUTODIFF_FWD`) and the H-mode
  specialized fallthroughs are gone. Callers reaching the dispatcher
  with `n_processes == 0` get `Rcpp::stop` with a pointer to this
  entry. `hmc_gradient_dispatch_predicates.h` deleted.
* **Removed log-posterior orchestrators' legacy body** (D-2).
  `compute_log_post`, `compute_log_prior`, `compute_log_lik_only`
  (`src/hmc_sampler.cpp`) now forward to
  `compute_log_post_generic_spec_double`; the
  `accumulate_log_prior_and_state` / `accumulate_obs_log_lik` body
  and its 5 `hmc_sampler_log_prior_*.h` fragments are gone, along
  with `hmc_log_posterior_split.h`. `compute_log_post_impl<T>`
  (`src/log_post_impl.h`) reduces to the same forward for
  `T = double` and a defensive `T(0)` no-op for autodiff `T` (arena
  AD now routes through `compute_log_post_generic<Var>`).
* **Removed gradient kernels** (D-3, ~30 files, ~17 KLOC). All
  hand-coded H-mode kernels (composite + 4 phases, vectorized +
  5 fragments, analytical, autodiff, feature, gp, hsgp, msgp, svc,
  tvc, st, temporal_gp, ms_temporal, latent), the collapsed-spatial
  machinery (`hmc_icar_collapsed_*` ×9, `hmc_gp_collapsed_*` ×5),
  the legacy ratio likelihood (`hmc_likelihood.h`,
  `hmc_observation_likelihood.h`), and the legacy fallback gradients
  (`compute_gradient_numerical` / `_autodiff` / `_arena` / `_forward`
  / `_numerical_impl`) are deleted. The 6 `log_post_impl_*_block.h`
  fragments and the 2 Rcpp ModelData populators
  (`model_data_rcpp.h`, `hmc_modeldata_builders.h`) follow.
  `verify_gradient_runtime` now always uses
  `compute_gradient_generic_numerical` as the reference.
* **Simplified samplers** (D-4). `compute_param_layout`
  (`src/hmc_param_layout.cpp`) requires `n_processes > 0`; model
  packages place model-specific scalars (overdispersion etc.) in the
  LikelihoodSpec extra-parameter block at `layout.extra_offset`.
  ESS's `build_gaussian_priors` and `get_non_gaussian_params`
  (`src/ess_sampler.h`) walk `process_beta_start` and
  `extra_offset` only.
  `hmc_nuts_mass_init.cpp` drops the family-specific block-spec
  heuristics (NB+ICAR / Bin+ICAR forced DENSE, NB phi-pair 2×2
  block) — re-introducing them would need a LikelihoodSpec hint.
* **ST_IV mass-matrix override disabled.** The precision-informed
  diagonal mass setup at warmup end
  (`src/hmc_nuts_chain_iter_nuts.h`) reconstructed `eta` from
  `data.legacy.X_num_flat` and branched on the legacy `ModelType`.
  ST_IV chains now fall back to the adapted DIAG mass matrix until
  the override is re-expressed through `spec->eta_weights_fn`. One
  no-op per chain at warmup end; practical impact on sampling
  efficiency is small.
* **Removed skipped tests.** `tests/testthat/test-log-post-split.R`
  and `tests/testthat/test-hmc-modeldata-builders.R` are deleted (every
  test was a Phase-D skip). The legacy-ratio gradient-check test in
  `test-spatial-car-proper.R` is removed; the two R-side
  `spatial_car_proper()` construction tests stay.
* **Cumulative numbers.** Phase D-1..D-5 deletes ~57 files and
  ~18 000 lines of legacy ratio infrastructure across `src/`,
  `inst/include/tulpa/`, `tests/`, and `tools/`. Net code reduction
  before the v13 maintenance window starts.

Downstream rebuild notes:
* `tulpaRatio` already routes through the generic LikelihoodSpec
  path via `tulpa_bridge.cpp` + per-family payloads in `lik_specs/`
  (B1+B2 of the migration); rebuild against v13 headers, no logic
  changes needed.
* `tulpaObs` never used the legacy ratio path; rebuild against v13.
* `tulpaGlmm` Day-22+ already targets the generic path; rebuild
  against v13.

## 2026-05-12 — ABI v12: generic-layout safety in compute_log_post_impl + ESS port

* `TULPA_ABI_VERSION` bumped **11 → 12**. Downstream packages must
  rebuild against the v12 headers.
* **Critical fix.** `compute_log_post_impl<T>` (`src/log_post_impl.h`)
  now early-returns to `compute_log_post_generic_spec_double` when
  the caller built `ModelData` with `n_processes > 0` and a non-null
  `likelihood_spec`. Previously the function reached lines 83-84 and
  unconditionally read `params[layout.legacy.beta_num_start]`, which
  is `params[-1]` for generic-layout callers — segfault. This was the
  blocker for tulpaGlmm Day-22 `inference = "ess"` (see deferred
  `fix.md` entry from 2026-05-06). The early-return makes the function
  safe for both layouts; the legacy ratio body remains in place for
  `n_processes == 0` callers (i.e. nobody outside this file at the
  moment, but tulpaRatio's `hmc_sampler.cpp` keeps its own copy).
* **ESS generic-layout port.** `tulpa_ess::build_gaussian_priors`
  (`src/ess_sampler.h`) now walks every process's β block
  (`layout.process_beta_start[k]` for `k` in `0..n_processes`) when
  `data.n_processes > 0`, instead of only `layout.legacy.beta_num_start
  / beta_denom_start`. Previously generic-layout ESS produced an
  empty β prior block and β was never sampled.
* **ESS RWMH coverage of model-specific extras.**
  `tulpa_ess::get_non_gaussian_params` now appends every parameter
  in `[layout.extra_offset, layout.extra_offset + n_extra_params)`
  to the RWMH list. LikelihoodSpec authors pack their model-specific
  scalars (e.g. log_phi for negative-binomial, log_sigma for Gaussian)
  into that block; ESS now walks them. Legacy ratio `log_phi_num /
  log_phi_denom` indices remain in the list for `n_processes == 0`.
* `LegacyRatioData` / `LegacyRatioLayout` (`inst/include/tulpa/model_data.h`,
  `param_layout.h`) are still exported but stay deprecated — the
  in-engine consumers are the H-mode gradient kernels, the legacy
  AD fallback (`hmc_gradient_fallback.cpp`), the composite gradient,
  and `tulpa_hmc::compute_log_post` inside `hmc_sampler.cpp`. None
  of those are reached when the dispatcher (`hmc_gradient_dispatch.h`)
  sees `n_processes > 0`. Full removal is a follow-up cut after the
  collapsed-spatial double-evaluator (MCLMC / SGHMC consumer) is
  reworked.
* Downstream rebuild notes: tulpaRatio uses the generic-LikelihoodSpec
  path via `tulpa_bridge.cpp` + per-family payloads in
  `lik_specs/`; it never touched `ModelData::legacy` and rebuilds
  cleanly against v12 headers. tulpaGlmm Day-22 ESS shim can now
  call `tulpa::get_ess_fn()(...)` end-to-end on a generic-layout
  `ModelData` without segfaulting.

## 2026-05-12 — ABI v11: caller-supplied inv-mass diagonal for NUTS

* `TULPA_ABI_VERSION` bumped **10 → 11**. Downstream packages must
  rebuild against the v11 headers.
* Registered C-callable `tulpa_run_nuts_generic` (`nuts_api.h`
  `NUTSFn`) gains a new positional parameter `const double*
  inv_metric_diag` immediately before `NUTSResult* result_out`. Pass
  `nullptr` to keep the v10 behaviour (default structural warm-start
  of the mass matrix). Pass a length-`n_params` vector to seed the
  diagonal inverse-mass — useful for warm-starting NUTS from an
  analytical-approximation method (Laplace, VI, etc.).
* `run_hmc_chain_cpp` / `run_hmc_chain` (`hmc_sampler_funcs.h`) take a
  matching trailing `inv_metric_init` `std::vector<double>` (default
  empty). Within `run_hmc_chain_cpp` (`hmc_nuts_chain_setup.h`), a
  non-empty caller diagonal overrides the structural diagonal set by
  `warm_start_mass_matrix`. Values are clamped to `[1e-3, 1e3]`
  before being installed via `mass.set_diagonal`, then
  `find_reasonable_epsilon` re-runs against the seeded metric.
* Mass-matrix *adaptation* is unchanged: the standard dual-averaging
  + expanding-windows path still refines the diagonal during warmup.
  The caller's vector is the *initial* metric, not a frozen one.
* Internal callers of `run_hmc_chain_cpp` (`hmc_nuts_parallel.cpp` ×3,
  `tulpa_generic_sampler.cpp` ×1) continue to pass no `inv_metric_init`
  via the default empty vector; the local forward declaration in
  `tulpa_generic_sampler.cpp` was updated to match the canonical
  declaration's parameter list.
* Downstream rebuild notes: `tulpaGlmm` exercises this end-to-end via
  Day-32's `hmc_warm_start = "laplace"` argument. `tulpaObs` and
  `tulpaRatio` need to be reinstalled against v11 — both already
  updated to pass `nullptr` for the new parameter (no logic change).

## 2026-05-11 — NNGP Laplace: full off-diagonal precision scatter

* `laplace_mode_gp` (and the spatial-only / ST-combo NNGP entries in
  `nested_laplace.cpp`) now assemble the **full NNGP precision matrix**
  `Λ = (I - A)' D⁻¹ (I - A)` in every Newton iteration, replacing the
  diagonal-on-w approximation that only kept `1/v_i` on the focal
  diagonal of each row.
* What was missed before: the gradient contribution to neighbours
  (`+a_{i,k}·q_i/v_i`), the off-diagonal Hessian entries
  (focal, neighbour_k) and (neighbour_k, neighbour_kp), and the
  pairwise precision between members of every conditioning set. The
  Newton mode for `w` was therefore shrunk toward zero and pointwise
  field recovery on smooth latent fields collapsed (cor ≈ 0).
* New helpers in `gpu_nngp_laplace.h`:
    - `batch_nngp_scatter(..., alpha_out = nullptr)` — backward-compatible
      extra optional output capturing the per-row conditional regression
      weights (already computed internally; just exposed).
    - `apply_nngp_full_prior_dense` — scatters the full precision
      contribution into a dense `(grad, H)` pair via the alpha + cv
      bundle.
    - `apply_nngp_full_prior_sparse` — same, into a `SparseHessianBuilder`.
    - `make_nngp_prior_sparsity_pattern` — emits the `(row, col)` pairs
      required to back the sparse path.
* Wired into the four scatter call sites: `laplace_mode_gp` dense Newton,
  `laplace_mode_gp` sparse Newton (with pattern expansion), the
  spatial-only `cpp_nested_laplace_nngp` scatter lambda, and
  `make_nngp_spatial_ops::add_prior_at_k` (the ST-combo NNGP block).
  Log-prior calls (`log_prior` lambdas) are unchanged — they only need
  `cm` and `cv`, and the existing `batch_nngp_scatter` signature still
  supports that without `alpha_out`.
* Effect (downstream measurement from tulpaGlmm Day-31 smoke):
  `cor(w_mean, f_true)` jumps from near zero to **≈ 0.81 Pearson** on
  a 120-location Poisson + smooth-GP simulation. β recovery unchanged.
* Additive: no `TULPA_ABI_VERSION` bump (still v10). Public shim
  signatures are unchanged; only the inner Laplace scatter is upgraded.
  Downstream packages must rebuild against this commit to pick up the
  new behaviour (no source changes required).

## 2026-05-11 — nested-Laplace ST family: 5 more indexed × indexed combos

* Adds five additional joint spatial × temporal nested-Laplace shims,
  built on the same `run_two_indexed_nested_laplace` driver and joint
  inner Newton introduced earlier today:
    - `tulpa_nested_laplace_st_icar_rw1`
    - `tulpa_nested_laplace_st_icar_rw2`
    - `tulpa_nested_laplace_st_car_proper_rw1`
    - `tulpa_nested_laplace_st_car_proper_rw2`
    - `tulpa_nested_laplace_st_car_proper_ar1`
  Each routes through a per-combo Rcpp entry plus a C-callable
  `_impl` wrapper; matching typedefs + getters live in
  `nested_laplace_api.h`.
* New internal factory pattern `IndexedPriorOps` in `nested_laplace.cpp`
  with per-kind builders `make_icar_ops`, `make_car_proper_ops`,
  `make_rw1_ops`, `make_rw2_ops`, `make_ar1_ops`. The shared
  `run_two_indexed_nested_laplace` driver now consumes
  `std::function`-typed callbacks, so adding the next indexed × indexed
  combination is a few lines of Rcpp glue rather than a re-derivation.
* Refactored `cpp_nested_laplace_st_icar_ar1` to use the new factories
  (identical behavior; just dropped the inline lambdas).
* Additive: no `TULPA_ABI_VERSION` bump (still v9). The new shims are
  resolved via `R_GetCCallable` at first use; downstream packages
  rebuilt against ABI v9 pick them up automatically.

## 2026-05-11 — nested-Laplace joint spatial × temporal (ICAR × AR1)

* New shim `tulpa_nested_laplace_st_icar_ar1` for joint nested-Laplace
  inference with an ICAR spatial field AND an AR1 temporal field in the
  same fit. The joint inner Newton solves over the full latent vector
  `[beta] [re] [w_spatial (n_s)] [w_temporal (n_t)]` at each grid point;
  the cross-block `H[w_s, w_t]` is non-zero, so the two fields cannot be
  Laplace-marginalized separately. The hyperparameter grid is supplied
  caller-side as paired vectors of length `n_grid` (Cartesian product of
  `τ_spatial × τ_temporal × ρ_temporal` built on the R side).
* New C-callable typedef `NestedLaplaceStIcarAr1Fn` + getter
  `get_nested_laplace_st_icar_ar1_fn()` in `nested_laplace_api.h`.
* New internal building blocks in `nested_laplace.cpp`: the templated
  `run_two_indexed_nested_laplace` driver and helpers
  `nl_compute_eta_two_indexed` / `nl_scatter_obs_two_indexed`. These are
  the shared substrate for the remaining 11 (spatial_kind × temporal_kind)
  combinations.
* `TULPA_ABI_VERSION` bumped 8 → 9. Downstream packages must be rebuilt
  against this header set.

## 2026-05-11 — nested-Laplace HSGP returns modes + store_Q

* `tulpa_nested_laplace_hsgp` now sets `store_modes = 1` (was 0) and
  gained a `store_Q` flag matching the rest of the nested-Laplace family.
  The basis-coefficient latent `[beta] [re] [beta_M (n_basis)]` is
  returned per `(σ², ℓ)` grid point, and with `store_Q = 1` the joint Q
  at the mode is retained in the standard `NestedLaplaceShimResult::Q_*_flat`
  slots.
* `cpp_nested_laplace_hsgp` gained a trailing `bool store_Q = false`
  argument and now passes `store_modes = true` to the grid driver. The
  C-callable `tulpa_nested_laplace_hsgp_impl` signature picks up the
  matching `int store_Q` parameter; the public typedef
  `NestedLaplaceHsgpFn` in `nested_laplace_api.h` is updated to match.
* This unblocks HSGP mixture-of-MVN sampling in tulpaGlmm — the
  observation-level spatial effect `f_i = Σ_j Φ_ij · √S(λ_j; σ²_k, ℓ_k) · β_M_j`
  can be reconstructed caller-side from modes + posterior draws over the
  basis coefficients plus the per-draw grid index.
* `TULPA_ABI_VERSION` bumped 7 → 8. Downstream packages (tulpaGlmm,
  tulpaObs) must be rebuilt against the updated headers.

## 2026-05-11 — nested-Laplace BYM2 returns modes + store_Q

* `tulpa_nested_laplace_bym2` now sets `store_modes = 1` (was 0) and
  gained a `store_Q` flag matching the rest of the nested-Laplace family.
  The reparameterised latent `[beta] [re] [phi (n_spatial)] [theta
  (n_spatial)]` is returned per grid point, and with `store_Q = 1` the
  joint Q at the mode is retained in the standard
  `NestedLaplaceShimResult::Q_*_flat` slots.
* `cpp_nested_laplace_bym2` gained a trailing `bool store_Q = false`
  argument and now passes `store_modes = true` to the grid driver. The
  C-callable `tulpa_nested_laplace_bym2_impl` signature picks up the
  matching `int store_Q` parameter; the public typedef
  `NestedLaplaceBym2Fn` in `nested_laplace_api.h` is updated to match.
* This unblocks BYM2 mixture-of-MVN sampling in tulpaGlmm — the total
  spatial effect `w_s = σ·(√ρ · scale · φ_s + √(1−ρ) · θ_s)` can be
  reconstructed caller-side from modes + posterior draws over the
  (σ, ρ) grid.
* `TULPA_ABI_VERSION` bumped 6 → 7. Downstream packages (tulpaGlmm,
  tulpaObs) must be rebuilt against the updated headers.

## 2026-05-11 — nested-Laplace store_Q on RW1/RW2/AR1/CAR_proper

* `tulpa_nested_laplace_rw1`, `tulpa_nested_laplace_rw2`,
  `tulpa_nested_laplace_ar1`, and `tulpa_nested_laplace_car_proper` now
  accept a `store_Q` flag (matching the ICAR shim added in v5). When set
  the shim retains the joint negative-Hessian Q at each grid point's mode
  in `NestedLaplaceShimResult::Q_*_flat`, so downstream packages can draw
  mixture-of-MVN posteriors `sum_k w_k · N(mode_k, Q_k^{-1})` without
  re-doing the Newton assembly R-side.
* The underlying `cpp_nested_laplace_<rw1|rw2|ar1|car_proper>` entries
  gained a trailing `bool store_Q = false` argument. Default is `false`,
  so existing callers that don't ask for Q keep the previous behaviour
  and footprint.
* `TULPA_ABI_VERSION` bumped 5 → 6. Downstream packages (tulpaGlmm,
  tulpaObs) must be rebuilt against the updated headers.

## 2026-05-06 — Takahashi partial inverse as a registered C-callable

* New free function `tulpa::takahashi_partial_inverse_dense(n, Lp, Li, Lx,
  Z_out)` in `sparse_cholesky.{h,cpp}` runs the Takahashi recursion on a
  caller-supplied lower-triangular `L` (CSC) and writes a dense column-major
  `n*n` `Z` with `Q^{-1}` on `pattern(L + L^T)` and zeros elsewhere. A
  matching `takahashi_partial_inverse_csc` returns just the `Zx` values on
  pattern(L). The existing `SparseCholeskySolver::selected_inversion_diagonal`
  now routes through the new helper so there is one source of truth for the
  recursion (no copy-paste).
* Registered C-callable `tulpa_takahashi_partial_inverse_dense` exposes the
  pure-function variant to downstream packages. Resolved via
  `tulpa::get_takahashi_partial_inverse_dense_fn()` in
  `inst/include/tulpa/sparse_solver_api.h`; the getter `Rf_error`s if the
  symbol is missing (i.e. caller built against newer headers than the loaded
  tulpa).
* No struct layout changes; `TULPA_ABI_VERSION` stays at 4. Downstream
  packages that want the new shim need only rebuild against the updated
  `sparse_solver_api.h`.

## 2026-05-05 — multi-term + slope REs on the spec-Laplace path

* `tulpa_laplace_spec_dense` (and its public C ABI shim) now accepts the
  full multi-term, multi-coefficient RE structure populated by `populate_re`
  in downstream model packages: K = `data.n_re_terms` random-effect terms,
  each with `q_t = re_n_coefs[t]` coefficients per group (intercept-only
  when `q_t == 1`, intercept + slopes when `q_t > 1`), uncorrelated
  (`(x||g)`) or correlated (`(x|g)`) prior covariance. Per-process
  sharing is uniform across terms via `data.sharing.re`.
* The legacy single-term path (`layout.re_start` / `layout.re_end` /
  `log_sigma_re_idx` with `data.n_re_terms == 0`) is preserved and stays
  bit-identical numerically.
* The `result_out->mode` writeout from `tulpa_laplace_spec_dense_impl`
  now concatenates every RE term's block in term order
  (`re_start_multi[t]..re_end_multi[t]`), matching the new
  `SpecLatentLayout` ordering.
* New internal test fixture `cpp_laplace_spec_test_multi_re` exercises
  the multi-term path end-to-end against hand-derived linear-Gaussian
  reference solutions; new tests in `tests/testthat/test-laplace-spec.R`
  cover (a) two crossed intercept-only terms, (b) one correlated random
  slope, (c) one uncorrelated random slope.
* `TULPA_ABI_VERSION` bumped from 3 → 4. Downstream packages that ship
  with the spec-Laplace dispatcher (e.g. tulpaGlmm) must be rebuilt
  against the new headers.
