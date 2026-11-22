#' Nested Laplace approximation for latent Gaussian models
#'
#' @description
#' Generic outer-grid nested Laplace driver. Builds a grid over the
#' hyperparameters of one latent prior block or a list of them (spatial,
#' temporal, iid, custom), runs an inner Laplace at each grid point with
#' warm-starting, and integrates over the grid to give proper hyperparameter
#' marginals. The fit is the one-arm fit of [tulpa_nested_laplace_joint()]:
#' its outer axes are labelled by block, `b<k>.<hyperparameter>` (`b1.tau` for
#' the precision of the first block), a single block included.
#'
#' Supported priors:
#'  * Spatial (areal): `"icar"` (1D grid on tau), `"bym2"`
#'    (2D on (sigma, rho)), `"car_proper"` (2D on (tau, rho); rho lives
#'    in the eigenvalue interval (1/lambda_min, 1/lambda_max) of
#'    `D^{-1} W`).
#'  * Spatial (continuous): `"nngp"` (2D on (sigma2, phi_gp)),
#'    `"hsgp"` (2D on (sigma2, lengthscale)).
#'  * Temporal: `"rw1"`, `"rw2"` (1D grid on tau), `"ar1"` (2D on (tau, rho))
#'  * SPDE continuous spatial: see \code{cpp_nested_laplace_spde()} (separate
#'    entry, rebuilds Q via SPDE Q-builder).
#'
#' @param y Response vector.
#' @param n_trials Trial sizes (binomial). Pass `1L`-vector otherwise.
#' @param X Fixed-effects design matrix.
#' @param prior A list describing the latent prior block. Required field
#'   `type` one of \{"icar", "bym2", "car_proper", "rw1", "rw2", "ar1"\}.
#'   Type-specific fields:
#'   * icar:  `spatial_idx`, `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'           `n_neighbors` (CSR adjacency, 0-based); optional `tau_grid`.
#'
#'   Any block accepts an optional `svc_weight`, one weight per observation,
#'   which makes it a varying coefficient: observation `i` contributes
#'   `svc_weight[i] * f_i` instead of `f_i`, where `f_i` is the block's field
#'   at that row (`z[spatial_idx[i]]` for an areal block, `(A u)_i` for an
#'   SPDE one). Absent, the field enters unweighted.
#'   * bym2:  same adjacency; `scale_factor`; optional `sigma_grid`, `rho_grid`.
#'   * car_proper: same adjacency; optional `tau_grid`, `rho_grid`,
#'           `rho_bounds = c(lower, upper)` (defaults to (0, 1)).
#'   * rw1/rw2: `temporal_idx` (1-based), `n_times`; optional `tau_grid`,
#'             `cyclic` (default FALSE).
#'   * ar1:   `temporal_idx`, `n_times`; optional `tau_grid`, `rho_grid`
#'           (each `rho_grid` value strictly inside (-1, 1)).
#'
#'   An `icar`, `bym2`, `car_proper` or `nngp` block may carry `projector`, an
#'   `N x n_spatial_units` (`N x n_spatial` for `nngp`) matrix (dense or a
#'   \pkg{Matrix} sparse matrix), in place of `spatial_idx`: observation `i`
#'   then receives `(A z)_i`, so restricted spatial regression is
#'   `A = S P_perp` with `S` the observation-to-unit incidence. The block is
#'   fitted by the multi-block driver, alone or in a list of blocks. An
#'   intrinsic field's level is placed from `A 1`: removed with no fold where
#'   it is zero, folded into the intercept where it is the intercept column,
#'   otherwise left to the augmentation. The `nngp` and `car_proper` priors are
#'   proper and carry their level themselves.
#'
#'   Grids of a two-axis type (`bym2`, `car_proper`, `ar1`) are PAIRED when
#'   both are supplied: the i-th entries of `tau_grid` / `sigma_grid` and
#'   `rho_grid` form the i-th outer-grid cell, so the two must have equal
#'   length, and a tensor grid is written as `expand.grid()` columns. A single
#'   supplied axis is kept as an axis (its distinct values) and crossed with
#'   the default of the other.
#'
#'   A default grid axis is a starting axis, not a hard ceiling: for `icar`
#'   (`tau_grid`) and `bym2` (`sigma_grid`) a posterior mode that rails a
#'   boundary node (`pareto_k_regime = "collapsed_edge"`) triggers one
#'   mode-Hessian recenter-and-refit, reported through
#'   `outer_grid_placement` / `outer_grid_recenter_declined` -- see
#'   [tulpa_nested_laplace_joint()]'s return docs. An axis the caller pinned is
#'   never moved; mark a grid your own code defaulted with [auto_grid()] to keep
#'   the recenter live on it.
#' @param re_idx Optional 1-based RE group index per obs (defaults to no RE).
#' @param n_re_groups RE group count (default 0).
#' @param sigma_re RE standard deviation (default 1).
#' @param family `"binomial"`, `"poisson"`, `"neg_binomial_2"`, etc.
#' @template phi
#' @param offset Optional per-observation offset added to the linear predictor
#'   (length `length(y)`), as an `offset()` term in a model formula. Every
#'   single- and multi-block kernel carries it, and `fitted_eta` includes it.
#' @template device
#' @param control Optional list of perf/numerical tuning knobs (statistical
#'   arguments stay top-level), following the `control` convention of
#'   [tulpa()]. Recognised elements (defaults in parentheses):
#'   * `max_iter` (`50L`), `tol` (`1e-6`) -- inner Newton iteration budget and
#'     tolerance.
#'   * `n_threads` (`1L`) -- inner-loop OpenMP threads.
#'   * `n_threads_outer` (`1L`) -- concurrent outer-grid cells, each retaining
#'     its own Newton state. Equal-order CUDA factors can share batched launches.
#'   * `x_init` (`NULL`) -- warm-start for the first grid point's inner solve.
#'   * `keep_grid_hessians` (`TRUE`) -- when `TRUE`, retain per-grid-point
#'     fixed-effects marginal Hessian \eqn{H_\beta} and mode \eqn{\hat{\beta}}
#'     on the return list as `$grid_hessians` (list of dense \eqn{p\times p}
#'     matrices) and `$grid_modes` (list of length-\eqn{p} vectors). Used
#'     downstream by simplified-Laplace (SLA) callers to assemble skew-aware
#'     marginals -- see the cumulant pooling in [rubins_pool()].
#'   * `diagnose_k` (`TRUE`), `k_samples` (`500L`) -- compute the outer
#'     Pareto-\eqn{\hat{k}} accuracy diagnostic (`$pareto_k`) by importance
#'     sampling the hyperparameter posterior against the Gaussian proposal
#'     fitted to the grid, drawing `k_samples` extra inner-marginal evaluations.
#'     `k_samples` is a precision knob: the GPD tail size is held at the
#'     fraction the default budget implies, so a larger budget sharpens the
#'     same k-hat instead of moving it to a deeper quantile of the weight
#'     distribution (gcol33/tulpa#631). `k_tail_points` overrides that tail
#'     size directly (capped at 20% of the draws).
#'     Scored as [tulpa_nested_laplace_joint()] scores it; an axis without a
#'     guessable coordinate is declined with the grid's quadrature ESS as the
#'     fallback diagnostic. See [tulpa_psis()].
#'   * `diagnose_skew` (`TRUE`), `skew_idx` (`NULL`) -- compute the inner-Laplace
#'     skewness diagnostic (`$inner_skew`, gamma_3, Rue Martino & Chopin 2009
#'     Sec 3.2.3) at the fitted MAP grid cell: one extra Newton solve, scoring
#'     the `p` fixed-effects latent indices by default (pass `skew_idx`, 1-based
#'     latent indices, to score additional ones, e.g. specific spatial units --
#'     the full latent field is not scored by default since it costs one linear
#'     solve per index). This is the complementary layer to `diagnose_k`: the
#'     outer diagnostic scores the hyperparameter-grid integration around a
#'     FIXED inner Laplace, this scores whether that inner Gaussian
#'     approximation is itself a good fit to the latent-field conditional
#'     posterior. See [diagnostics()] for the combined whole-fit verdict.
#'   * `within_cell` (`"log_quadratic"`) -- the WITHIN-CELL construction the
#'     reported per-axis hyperparameter intervals are read with. The outer
#'     grid's weights say how much mass each cell holds; they do not say how it
#'     is spread inside the cell, and a quantile needs both.
#'     `"log_quadratic"` joins each row's node log densities by overlapping
#'     quadratics, exact for a Gaussian at any spacing, and continues the end
#'     quadratic past the outer node instead of stopping at the box edge; it
#'     runs on the rows that resolve their own conditional (`h / sd <= 2`) and
#'     reads every other row as boxes, declining to `"box_uniform"` where no
#'     row qualifies. `"box_uniform"` puts the cumulative FULL mass at each
#'     cell EDGE and interpolates between edges; `"chord"` puts the cumulative
#'     MID-mass at each cell coordinate and interpolates between coordinates.
#'     The default moved from `"box_uniform"` at gcol33/tulpa#932: a placed
#'     axis sits at 1.25 posterior SDs per cell, where the box read is narrow
#'     by construction, and against dense references of a two-arm ICAR
#'     fixture the log-quadratic read's summed error is 0.038 against the box
#'     read's 0.110 on the same grids. `outer_grid_h_over_sd` is how wide a
#'     cell is on each axis, and `theta_within_cell` is what each axis was
#'     actually read with. Only a `"density"` support admits the box and
#'     log-quadratic reads -- a CCD design, a locally refined grid and a
#'     posterior sample are not cell partitions that tile -- and an axis they
#'     decline on reports the read that ran with a reason rather than erroring.
#'     Nothing else moves: point estimates, moments, draws and weights are
#'     untouched.
#'   * `skew_correct` (`TRUE`) -- consume the inner-Laplace expansion instead of
#'     only grading it: report
#'     Cornish-Fisher marginal quantiles at each coefficient's own `gamma_3`,
#'     about the centre `gamma_1 + gamma_3 / 2` that Rue, Martino & Chopin
#'     (2009) eq. (22) implies, from `summary()` / `confint()` wherever the
#'     combined inner band says the leading-order expansion is in its regime, and
#'     the Gaussian quantiles everywhere else. It is post-processing on the
#'     reported quantiles: draws, modes and weights are untouched, so a fit run
#'     with it off is bit for bit the fit it was before. A coefficient whose
#'     location term could not be formed declines rather than reading it as zero.
#'     The band that bounded the relocation itself (`centre_unreliable`) is
#'     off (`Inf`): scored over seven fixtures with an exact
#'     reference, every finite cutoff declines the coefficients the correction
#'     helps most, because a large centre carrying a small `gamma_3` is uniformly
#'     weak correlation rather than an expansion out of its regime.
#'     `$skew_correction` records the per-coefficient `gamma_3`, `gamma_1` and
#'     the centre they form, the band, the inner importance k-hat, the combined
#'     band, the eligibility and the reason behind it; the `skew_applied`
#'     attribute on `summary()` / `confint()` records what was actually used at
#'     the requested level. RMC fit a skew normal here instead; the series
#'     correction is the same-order alternative, and unlike a skew normal its
#'     skewness does not saturate inside the band it is applied on.
#'
#'     MEASURED. Against exact quadrature
#'     quantiles of rare-event binomial-logit posteriors it cuts total absolute
#'     endpoint error 69.2%, improving both endpoints in every case. Scored over
#'     the WHOLE marginal -- paired CRPS against the exact posterior in a
#'     400-replicate prior-predictive experiment -- it reads -0.01643 against
#'     the uncorrected Laplace at t = -1.89, essentially all of the -0.01662 the
#'     exact posterior itself achieves, and its PIT re-enters the simultaneous
#'     SBC band. Applied about the Laplace mode instead of about
#'     `gamma_1 + gamma_3 / 2` the same reshaping scored +0.00775 at t = +3.54,
#'     a net loss; the location term is what supplies that centre.
#'
#'     IT IS ON BY DEFAULT, so `summary()` / `confint()` on a
#'     nested-Laplace fit report the corrected quantiles wherever the combined
#'     inner band admits the coefficient; `skew_correct = FALSE` restores the
#'     uncorrected report exactly. Scored against the mixture read a
#'     correction-off fit gives, the flip is t = -1.895 on the rare-event
#'     intercept and -3.765 / -3.201 on the small-group Bernoulli design. Across
#'     twelve model classes read off one solve per seed, pooled 95% coverage
#'     moves 0.9510 -> 0.9542 at a standard error of 0.0070, with every class
#'     inside the acceptance the shipped gates use. A fit the correction cannot
#'     help -- a coupled one, whose location term is unreachable -- reports what
#'     it reported before, to the bit.
#'   * `subspace_debias` (`FALSE`) -- latent Metropolis correction, supplied as
#'     `TRUE` or a list. `method = "fast"` (default) samples the true joint
#'     density on the Gaussian conditional-mean surface of the selected
#'     coordinates, carries Gaussian nuisance conditionals, and keeps Laplace
#'     cell weights (Tier 2). Its empty selection consumes no randomness.
#'     `method = "exact"` proposes the full latent vector: a symmetric move in
#'     the selected coordinates and a fresh Gaussian conditional nuisance draw.
#'     The joint proposal density enters the MH ratio, and rejection retains
#'     the complete state. Each cell's kernel targets `p(x | theta_k, y)`.
#'     Independent full-dimensional t importance samples estimate the ratio of
#'     the true latent integral to its Laplace approximation. Independent pilots
#'     calibrate the particle count using the shared IS settings; the final
#'     unbiased integral estimate reweights the cell. The existing PSIS k
#'     diagnostic and gate assess the stratified joint correction. Finite
#'     importance and outer-grid errors
#'     remain, so the nested result stays Tier 2. Hard constraints and unusable
#'     Cholesky factors decline explicitly. Choose one of this exact composition
#'     and `cila`.
#'     Other settings are `band` (default `"ok"`), `idx` (explicit selected
#'     indices), `closure` / `closure_max` (precision-graph neighbours),
#'     `n_iter` (recorded sweeps), `warmup`, `thin`, and `n_draws` (reported
#'     mixture rows). The selector reads `diagnose_skew`'s per-index bands.
#'     The correction re-solves the settled grid. `$subspace_debias` records
#'     the selected set, targets, acceptance, IS diagnostics and decline reason.
#'     An applied exact correction retains each full latent chain, its cell mode
#'     and paired row indices; `tulpa_posterior_draws()` samples those rows.
#'   * `cila` (`FALSE`) -- corrected integrated Laplace, the second inner-layer
#'     debias (after Lai, Margossian and Sheldon,
#'     arXiv:2605.20345). Where
#'     `subspace_debias` samples a selected Gaussian-conditioned subspace,
#'     this selects nothing: at every outer cell it draws `n_points` points from
#'     the whole inner Gaussian, weights each by the exact joint density it came
#'     from, and reports the weighted particles. Increasing particle effort
#'     corrects integration within each fixed cell; finite outer-grid error
#'     remains. This fit remains Tier 2. `TRUE` takes the defaults; a
#'     list overrides `n_points` (`1024L`), `variant` (`"qmc"`, a Sobol net;
#'     `"is"` for iid draws, `"rqmc"` for the net under `n_shift` random
#'     shifts), `n_shift` (`8L`), `n_draws` and `seed`. Below 512 points a
#'     cell's particle set is too coarse to be a marginal at all and the request
#'     is refused. The corrected per-cell masses become the fit's own
#'     `weights` / `log_marginal` and `weights_source` reports `"cila"`; the
#'     pre-correction pair is kept as `$cila$laplace`. A cell whose inner solve
#'     factorized sparsely draws through the CHOLMOD factor's own triangular and
#'     permutation solves; an LDL' factor has no square root to draw with and is
#'     declined with `"sparse_factor_not_ll"`.
#'   * `auto_recenter` (`TRUE`) -- outer-grid placement. `TRUE` runs the
#'     placement pass of [tulpa_nested_laplace_joint()]: it re-centres the
#'     movable default axes on the posterior mode and refits when the grid either
#'     RAILS (an axis's own marginal is maximal at one of its own endpoints) or
#'     does not RESOLVE its posterior (an axis's node spacing exceeds 2 posterior
#'     SDs in its own coordinate). Both tests read the weights the fit already
#'     stored, so a grid that brackets and resolves its mode costs nothing beyond
#'     them; when the pass does fire it is a second full grid solve plus a
#'     finite-difference mode/Hessian stencil. `FALSE` integrates over the grid
#'     exactly as given and records
#'     `outer_grid_recenter_declined = "auto_recenter_disabled"`. Any other value
#'     is an error.
#'
#'     Measured over 200 fixed-truth seeds on each of six configurations (icar
#'     chain / icar lattice / rw1 / bym2 / iid / nngp), the rail-or-resolve rule
#'     moves mean |coverage - nominal| from 0.043 to 0.030 at the 95% level,
#'     0.171 to 0.084 at 80% and 0.243 to 0.129 at 50% against re-centring on a
#'     rail alone, at 0.63 times the 95% interval width and 0.76 times the
#'     median bias, for 1.71 times the wall clock.
#'   * `max_grid_cells` (`NULL`, no ceiling) -- a ceiling on a multi-block
#'     outer grid's cell count, one inner Newton solve per cell; a grid past it
#'     is refused before any solve, naming the axes that produced the count.
#'     Unset, a grid of any size is solved, and one past 2048 cells is announced
#'     with a warning first.
#'   * `checkpoint` (`list(path =, resume =)`) -- grid-cell checkpoint /
#'     resume. Each solved outer cell is appended to `path`, keyed by its
#'     hyperparameter coordinate; `resume = TRUE` loads the finished cells and
#'     solves only the rest, and a fingerprint mismatch (different data,
#'     settings or grid) errors rather than resuming onto a stale result. See
#'     `?tulpa_nested_laplace` "Checkpoint / resume".
#'   * `progress`, `progress.every`, `progress.file`, `progress.throttle` --
#'     the outer-grid progress reporter: whether to print, how often (in
#'     cells), where to, and the minimum seconds between lines.
#'   * `prune` (`TRUE`), `prune_tol` (`1e-4`), `screen_iters`
#'     (`.NL_SCREEN$iters`, 2 -- the depth gcol33/tulpa#640 measured the screen
#'     down to) -- cheap-pass screening of the outer grid. The driver first
#'     sweeps the lattice running a `screen_iters`-step inner Newton per cell,
#'     each warm-started from the previous screened cell's quasi-mode, computes
#'     a screening Laplace log-marginal, softmax-normalises it, and skips the
#'     full inner Newton on every cell whose screened weight is below
#'     `prune_tol`. The neighbour warm-start keeps each cheap mode near its
#'     cell's own mode, so the cheap ranking tracks the full-solve ranking even
#'     where the latent mode moves across the grid. Cell-dependent basis
#'     loadings use the full `max_iter` budget and the fit's ordinary inner
#'     convergence tolerance, since their warm-start coordinates change between
#'     cells. Pruned cells get
#'     `log_marginal = -Inf`, `n_iter = 0` and inherit the pilot mode; the pilot
#'     cell is never pruned, and at least `prune_min_keep` cells (5) are solved
#'     in full whatever the tolerance says -- the highest-ranked dropped cells
#'     are restored up to that floor, because the outer grid placement pass
#'     reads a finite-difference curvature off the cells that were SOLVED and a
#'     kept set of one carries none. After the full pass the posterior mass the
#'     dropped cells may carry is bounded by lifting each by the worst screening
#'     error measured on a kept cell (`prune_dropped_mass_bound`); past 1% the
#'     driver solves the dropped cells that contribute most to it until it is
#'     under 1% (`prune_n_repaired`), and a safety gate falls back to the full
#'     grid (with a warning) if the bound still fails. The bound assumes a
#'     dropped cell is screened no worse than the worst kept one. On by default
#'     (gcol33/tulpa#925); `FALSE` solves every cell.
#'     `prune_tol` must be a finite numeric in `[0, 1)` and `screen_iters` a
#'     single integer `>= 1`; both are validated whatever `prune` says.
#'     The screen is the one [tulpa_nested_laplace_joint()] runs.
#'   * `prune_log_gap` (unset) -- the screening cut stated in nats instead of as
#'     a normalised weight: keep every cell within this many nats of the best
#'     screened cell. `prune_tol` is a weight, so what it cuts at is the gap
#'     `-(log(prune_tol) + log(Z))`, and the whole range a caller would type
#'     (`1e-3` to `1e-12`) is 7 to 28 nats of it -- on an outer surface whose
#'     log-marginal spans thousands of nats every setting returns the same kept
#'     set. Stating the gap directly keeps the knob's resolution wherever the
#'     surface is steep. It reaches the kernel as `prune_tol = exp(-gap)`, so
#'     the realised cut is the requested gap less `log(Z)`, at most `log(n_grid)`
#'     nats narrower; the fit reports the realised value. Setting both this and
#'     `prune_tol` is an error -- they state one cut in two units.
#'   * `fitted_var` (`TRUE`) -- fill the per-row predictive variance
#'     `fitted_eta_var`, the companion of `fitted_eta` a caller marginalises the
#'     per-row linear predictor with. It costs one back-solve per distinct
#'     loading vector per cell, which on a design with few repeated rows is the
#'     dominant cost of a cell, so a caller reading only `fitted_eta` or the
#'     coefficient summaries sets `FALSE` and the fit then carries no
#'     `fitted_eta_var`. Every block reports it, the NNGP, HSGP, SPDE and
#'     spatiotemporal fields included, each read off its cell's own precision
#'     at the mode.
#'   * `var_of_means_consistency` (`TRUE`) -- the consistency pass of
#'     [tulpa_nested_laplace_joint()]: a field SD (`tau` of an `icar`, `rw1` or
#'     `rw2` block, `sigma` of an `iid`, `bym2` or `spde` block) whose marginal
#'     has collapsed onto too few nodes to carry a spread has levels added in
#'     every row of the other axes that holds the posterior until it resolves, and the new cells are
#'     solved and carried like the rest (`var_of_means_consistency_info`
#'     records the pass). Other axes keep the nodes they were laid on.
#'   * `adaptive_grid` (`FALSE`), `adaptive_grid_edge_thresh` (`0.02`),
#'     `adaptive_grid_max_passes` (`1L`) -- the opt-in boundary / interior pass,
#'     on the same axes.
#'   * `integration` (`"aghq"` for a local/regional HSGP field, `"qmc"` for
#'     a local/regional NNGP field, `"grid"` otherwise) -- `"aghq"` uses
#'     nested Genz-Keister quadrature in jointly standardized coordinates.
#'     A smooth logit shift centres the regional prior-CDF coordinates at
#'     their placement posterior medians. A normal quantile transform maps
#'     their endpoints to infinity. A joint affine map uses the placement
#'     posterior mean and covariance in these coordinates.
#'     Observation-space HSGP sums use separable regional prior-CDF coordinates
#'     and nested Patterson rules, preserving repeated spectral projections.
#'     Additional HSGP and independent Gaussian blocks use mode-scaled log
#'     coordinates. Every varying axis must freely extend. With `"aghq"`,
#'     grids seed placement and integration covers the positive prior support.
#'     The local mode and covariance use the regional marginal evaluated with
#'     a five-point regional rule. Posterior contours two log-density units
#'     below that mode set a smooth per-side scale on each whitened coordinate.
#'     The transport includes its Jacobian and preserves both infinite tails.
#'     Dimension-adaptive Smolyak refinement reuses every solved node.
#'     Its combination of tensor cubic splines interpolates the log posterior
#'     ratio. Exponentiating that interpolant supplies positive posterior mass.
#'     Positive recombination fits conditional Gaussian CDFs, fixed-effect
#'     moments and hyperparameter moments to the interpolated posterior.
#'     Matrix scaling couples these
#'     conditional-cell masses to the continuous hyperparameter read.
#'     Hyperparameter draws condition on this positive coupling.
#'     `quad_max_level` (`5L`, between three and five) bounds nested refinement.
#'     Normal-rule sizes are 1, 3, 9, 19 and 35; bounded-rule sizes are
#'     1, 3, 7, 15 and 31.
#'     `quad_read_nodes` (`65536L`, divisible by eight) starts inexpensive
#'     shifted Sobol reads of the log spline through a seven-degree Student
#'     proposal on the Gaussian axes, with twice the fitted covariance.
#'     A stalled read refits its proposal once per grid; independent shifts
#'     check the refit. Each doubling tests
#'     whether uncertainty decreases at the sampling rate.
#'     `quad_read_max_nodes` (`524288L`, divisible by eight) bounds each read.
#'     The sampling-rate forecast at that bound directs refinement when
#'     a larger read cannot close the remaining error budget.
#'     A stalled read directs work to grid refinement. Reads use no inner solves.
#'     A coarse read fits the proposal moments; independent full reads check
#'     its precision once the quantile interpolation budget is resolved,
#'     independently of the evidence target.
#'     Read uncertainty uses a simultaneous delete-scramble quantile envelope.
#'     `quad_tol` (`0.05` posterior SD)
#'     and `quad_evidence_tol` (`0.02` nats) control refinement.
#'     The quantity with the largest error relative to its target directs
#'     both read allocation and grid refinement.
#'     Two attained evidence budgets give a log Bayes factor budget of 0.04 nats.
#'     Quantile indicators include changes in posterior CDFs from active subspaces,
#'     conditional-mixture resolution and shifted-read uncertainty.
#'     Evidence is the positive interpolant integral. Its indicator combines
#'     active differential integrals, their discrepancy from that integral,
#'     and shifted-read uncertainty.
#'     Three refinements precede a finite estimate. `outer_quadrature` records
#'     the estimates, unique solve count, index set and refinement history.
#'
#'     `"user"` uses `outer_design = list(points =, weights =)`.
#'     `points` is a finite numeric matrix with the fitted outer-axis column
#'     names; columns may be reordered.
#'     Values use the kernel's physical coordinates. Weights are finite,
#'     nonnegative integration masses in the axes' integration coordinates
#'     (log coordinates for positive scale axes). They multiply the stored
#'     joint density including its hyperprior. They need not sum to one.
#'     Supplying `outer_design` selects `"user"` when `integration` is omitted.
#'     Evidence, summaries and predictions use these same weighted cells.
#'
#'     `"qmc"` integrates continuous hyperparameters with
#'     quasi Monte Carlo importance quadrature in transformed coordinates.
#'     Each free covariance block uses its declared grid coordinates for the proposal.
#'     A mixture of diagonal product Student distributions is fitted to the
#'     placement grid and is adapted in two
#'     pilot rounds. The final points carry inverse-proposal-density weights.
#'     `qmc_nodes` (`65536L`) bounds the final proposal count; each pilot proposes
#'     at most `1024L` points. Independent scrambled sequences grow in blocks
#'     until the estimated error in log evidence (nats) and every reported
#'     hyperparameter and fixed-effect quantile (posterior SD units) is below
#'     `qmc_tol` (`0.05`). Hyperparameter errors use the proposal coordinates,
#'     so unbounded range axes use their finite log-coordinate SD. This design follows separate variance arms without
#'     crossing each new axis level with every other axis. Axes whose nodes
#'     the caller supplied retain their integration bounds; points outside
#'     those bounds carry zero mass and are discarded. The reported
#'     `outer_qmc` contains the proposal, evaluation count, effective sample size,
#'     error estimates, stopping history and convergence flag. A simultaneous
#'     95% Student-t critical value calibrates the endpoint family and evidence.
#'     Scramble residuals of pooled CDFs supply simultaneous probability bands.
#'     Empirical CDF bands include every knot. Inverting each band gives
#'     endpoint uncertainty on the corresponding parameter's scale.
#'     The relative normalizer uncertainty is
#'     transformed through the logarithm to bound evidence uncertainty.
#'     A stopping candidate and the point ceiling receive the complete
#'     fixed-effect check. Earlier history entries record `complete = 0` and
#'     `error = Inf` when a failed hyperparameter or evidence criterion makes
#'     that check unnecessary; `hyper_error` retains the hyperparameter result.
#'     The importance points receive full inner solves. The placement grid
#'     uses the pruning controls.
#'   * `axis_refine` (`NULL`) -- per-axis override of how far those axes may be
#'     moved, named by the grid's axes (`"b1.tau"`): `"none"`, `"densify"` (inside the declared span) or `"extend"`.
#'     Nodes the caller wrote down densify by default and an engine-placed axis
#'     extends, as in [tulpa_nested_laplace_joint()].
#'
#' @return A list with:
#'   * `theta_grid`: matrix or vector of grid hyperparameter values.
#'   * `log_marginal`: log p(y, mode | theta_k) at each grid point.
#'   * `weights`: integration weights normalising to sum 1.
#'   * `theta_mean`, `theta_sd`: posterior moments per hyperparameter.
#'     A range or precision moment that diverges is `Inf`: a precision
#'     under the PC prior on its SD keeps the prior's `tau^(-3/2)` tail, so it
#'     has neither a mean nor an SD. `moment_existence` records each such
#'     axis's prior tail, likelihood boundary decay, fixed or bounded support,
#'     and the reason for its moment-existence result.
#'     `theta_log_mean` and `theta_log_sd` retain the finite log-range moments.
#'   * `n_iter`: inner Newton iterations per grid point.
#'   * `modes`: matrix `[n_grid x n_x]` of inner modes, when stored.
#'   * `pareto_k`, `pareto_k_is_ess`: outer Pareto-\eqn{\hat{k}} and its
#'     importance-sampling ESS (`NA` when not computed for the grid; see
#'     `control$diagnose_k`).
#'   * `pareto_k_proposal_source`, `pareto_k_first_pass`: which proposal family
#'     the reported k-hat came from (the outer k-hat is scored against several
#'     candidates and the best is kept), and the k-hat of the FIRST pass -- the
#'     proposal exactly as this backend placed it, before any candidate refined
#'     or replaced it. A large gap between the two says the nodes are badly
#'     scaled around the hyperparameter posterior even though the verdict is
#'     fine; no gap says the placement was already right.
#'   * `inner_skew`, `inner_skew_idx`, `inner_skew_dropped`: the inner-Laplace
#'     skewness diagnostic (gamma_3) at each scored latent index and its
#'     1-based index, plus a count of (index, observation) contributions
#'     dropped for a non-finite third derivative (see `control$diagnose_skew`).
#'   * `timing`: named numeric of wall-clock seconds (`total`, `setup`,
#'     `grid`, `postproc`, `diagnostics`); the `grid` phase is the inner
#'     Laplace pass that scales with grid size. Surfaced one-line in `print`.
#'   * `checkpoint_n_loaded`: present when `control$checkpoint` is set -- the
#'     number of outer-grid cells this fit read back from the checkpoint file
#'     instead of solving, over every grid it solved (a recentred refit and
#'     refinement rounds included). `0` on a fresh run.
#'   * `prune_cheap_log_marginal`, `prune_mask`, `prune_n_pruned`, `prune_tol`:
#'     present when the grid was screened and the safety gate did not
#'     trip -- the cheap-screen log-marginal per cell (the truncated solve's
#'     value plus half its Newton decrement, the second-order estimate of the
#'     converged value the cells are ranked by; the decrement itself is
#'     `prune_screen_decrement`), the logical mask of
#'     pruned cells, the pruned-cell count, and the threshold actually applied.
#'   * `prune_log_gap_cut`, `prune_cheap_lm_spread`, `prune_min_keep`,
#'     `prune_n_floor_restored`: the same screen read on the scale it operates
#'     on -- how many nats below the best screened cell the tolerance cut at,
#'     how many nats the screened surface spans, the kept-cell floor applied,
#'     and how many cells that floor put back. A cut that is a sliver of the
#'     spread is a tolerance with no resolution on this grid.
#'   * `prune_dropped_mass_bound`, `prune_n_repaired`, `prune_gate_mass`: the
#'     posterior mass the dropped cells may carry, with each lifted by the worst
#'     screening error measured on a kept cell; how many dropped cells the
#'     driver solved in full to bring that under the gate; and the gate.
#'   * `prune_fallback_triggered`, `prune_fallback_reason`: present only when
#'     the gate did trip; the rest of the fit is then the full-grid result and
#'     the reason string records which check fired.
#'   * `prior`: echoed input.
#'
#' @param spec Optional `tulpa_temporal` or `tulpa_spatial` spec object
#'   (output of [temporal_rw1()], [temporal_rw2()], [temporal_ar1()],
#'   [spatial_car()], [spatial_bym2()], etc.). When supplied alongside
#'   `data`, the `prior` list is built automatically via
#'   [prior_from_spec()] -- pass either `prior` or `spec`, not both.
#' @param data Data frame used to validate `spec` and resolve
#'   time/group/site indices. Required when `spec` is supplied.
#' @param likelihood Optional model-supplied likelihood, replacing the built-in
#'   `family`. Pass an external pointer to a `tulpa::NestedLikelihood` (built in
#'   a model package's own C++ from a `LikelihoodSpec`); the inner Laplace solve
#'   then reads the per-observation score, Fisher weight, and log-likelihood from
#'   that spec instead of `family`, so `family`/`phi` are ignored. Used by model
#'   packages to fit a custom response without adding a family to tulpa -- for
#'   example tulpaObs threads its marginalized single-season occupancy
#'   likelihood (a scaled Bernoulli, with the latent occupancy state integrated
#'   out) through this. Default `NULL` (use `family`).
#' @template hyperprior
#' @param beta_prior Optional Gaussian prior on the fixed effects,
#'   `list(mean = , sd = )` (or [prior_normal()]), each scalar or of length
#'   `ncol(X)`; `sd = Inf` leaves a coefficient unpenalized. Every grid cell
#'   is solved under it. `NULL` (default) takes the engine default
#'   `N(0, 2.5^2)`, as in [tulpa_laplace()] and [tulpa()].
#'
#' @references
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations.
#' \emph{JRSS-B} 71(2):319-392.
#' @examples
#' \donttest{
#' set.seed(1)
#' S <- 30L                                   # spatial units arranged in a chain
#' nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
#' nn <- lengths(nb)
#' field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))   # smooth spatial field
#' idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
#' y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
#' prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
#'               adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
#'               n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
#' fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
#'                             family = "binomial")
#' fit$theta_mean        # marginalized ICAR precision
#' }
#' @export
tulpa_nested_laplace <- function(y, n_trials, X, prior = NULL,
                            spec = NULL, data = NULL,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            family = "binomial", phi = 1.0, offset = NULL,
                            likelihood = NULL,
                            hyperprior = c("proper", "flat"),
                            beta_prior = NULL,
                            control = list()) {

  tulpa_check_control(control, .CONTROL_KEYS$nested_laplace, "tulpa_nested_laplace")
  design_cfg <- .nl_outer_design_control(control)
  # A typo is named here, against the family registry, rather than by the C++
  # family lookup, whose message names a header (gcol33/tulpa#886). A consumer
  # `likelihood` supplies its own density and bypasses the builtin family.
  if (is.null(likelihood)) .family_or_stop(.canonical_family(family))
  hyperprior <- .hp_choice(match.arg(hyperprior))

  if (!is.null(spec)) {
    if (!is.null(prior)) {
      stop("Pass either `spec` or `prior`, not both.", call. = FALSE)
    }
    if (is.null(data)) {
      stop("`data` is required when `spec` is supplied.", call. = FALSE)
    }
    # An HSGP spec whose basis is left to the data is fitted through the HSGP
    # basis loop; the fit records the basis it was made at as `hsgp_basis`.
    if (inherits(spec, "tulpa_hsgp")) {
      v <- validate_hsgp(spec, data)
      if (.hsgp_basis_auto(v)) {
        cl <- match.call()
        env <- parent.frame()
        run <- .hsgp_fit_basis_loop(v, function(s) {
          cl$spec <- NULL
          cl$data <- NULL
          cl$prior <- .spatial_spec_to_nl_prior(s)
          eval(cl, env)
        })
        run$fit$hsgp_basis <- run$spec
        return(run$fit)
      }
    }
    prior <- prior_from_spec(spec, data)
  }
  if (inherits(prior, "tulpa_latent_block")) prior <- list(prior)
  if (!is.list(prior)) {
    stop("`prior` must be a list (single block) or list-of-lists (multi-block).",
         call. = FALSE)
  }
  # This door's outer grid is spanned by the latent blocks' hyperparameters
  # alone, so a prior with no block leaves nothing to integrate. The joint door
  # integrates per-arm dispersion axes as well and does take an empty prior.
  if (!length(prior)) {
    stop("`prior` holds no latent block, so tulpa_nested_laplace() has no ",
         "hyperparameter to integrate. Fit a fixed-effects model with ",
         "tulpa(mode = \"laplace\"), or integrate a dispersion axis with ",
         "tulpa_nested_laplace_joint(prior = list(), phi_grid = ).",
         call. = FALSE)
  }
  # One block is the block list holding it: every fit is the block list's
  # one-arm joint fit, whose axes are labelled by block (`b1.tau`).
  if (!.is_multi_block_prior(prior)) {
    if (is.null(prior$type)) {
      stop("Single-block `prior` must have a `type` field, ",
           "or pass a list of blocks for multi-block.", call. = FALSE)
    }
    prior <- list(prior)
  }
  ms <- all(c("local", "regional") %in%
              vapply(prior, function(p)
                if (tolower(p$type %||% "") %in% c("nngp", "hsgp"))
                  p$name %||% "" else "", character(1)))
  integration <- .nl_outer_integration(control$integration, design_cfg, prior,
    if (ms && .nl_aghq_supported(prior, control)) "aghq" else if (ms) "qmc" else "grid",
    c("grid", .NL_OUTER_DESIGNS))
  vd <- .validate_glm_design(y, X, n_trials, "tulpa_nested_laplace",
                             missing_response = TRUE)
  N  <- vd$N
  n_trials <- vd$n_trials
  if (is.null(re_idx)) re_idx <- rep(0L, N)
  re_idx <- .validate_re_idx(re_idx, n_re_groups, N, "tulpa_nested_laplace")
  if (n_re_groups > 0L && !(length(sigma_re) == 1L && is.finite(sigma_re) &&
                           sigma_re > 0)) {
    stop("`sigma_re` must be a positive number (the random-intercept SD).",
         call. = FALSE)
  }

  if (!is.null(offset)) {
    offset <- as.numeric(offset)
    if (length(offset) != length(y) || any(!is.finite(offset))) {
      stop("`offset` must be a finite numeric vector of length length(y) (",
           length(y), ").", call. = FALSE)
    }
  }

  keep_grid_hessians <- isTRUE(control$keep_grid_hessians %||% TRUE) ||
    !is.null(.subspace_debias_config(control$subspace_debias, control$cila))
  cargs <- .nl_kernel_args(
    y = y, n_trials = n_trials, X = X, offset = offset, re_idx = re_idx,
    n_re_groups = n_re_groups, sigma_re = sigma_re, family = family,
    phi = phi, max_iter = control$max_iter %||% 50L, tol = control$tol %||% 1e-6,
    n_threads = control$n_threads %||% 1L, x_init = control$x_init,
    store_Q = keep_grid_hessians, beta_prior = beta_prior,
    hyperprior = hyperprior,
    n_threads_outer = .nl_outer_threads(control$n_threads_outer %||% 1L,
                                        "tulpa_nested_laplace()")$realised)

  .nl_registry_multi_fit(
    prior, cargs, control, likelihood, hyperprior, integration,
    keep_grid_hessians = keep_grid_hessians, X = X,
    data = list(y = y, n_trials = n_trials, model_matrix = X,
                family = family, offset = offset, phi = phi))
}

# --- cheap-pass screening knobs ---------------------------------------------
#
# The kernels read a tolerance in [0, 1) and a screening depth of at least one
# Newton step. A value outside either range is a caller error rather than
# something to clamp: a clamped tolerance reads back off the fit as a setting
# that was applied.
.nl_check_prune_tol <- function(x) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x < 0 || x >= 1) {
    stop("`prune_tol` must be a finite numeric in [0, 1).", call. = FALSE)
  }
  as.numeric(x)
}

.nl_check_screen_iters <- function(x) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x < 1 ||
      x > .Machine$integer.max || x != round(x)) {
    stop("`screen_iters` must be a single integer >= 1.", call. = FALSE)
  }
  as.integer(x)
}

# A constraint row as a driver hands it back (`constraint_cols`): an integer
# vector of 1-based latent indices is a sum-to-zero group (every weight 1); a
# list(cols, w) is a weighted row, a hard linear constraint a block declares
# or a weighted level. `.constraint_row_dense()` is the row as a length-n_x
# vector; `.constraint_rows_for_cpp()` the list in the form
# read_constraint_rows() takes.
.constraint_row_dense <- function(cc, n_x) {
  a <- numeric(n_x)
  if (is.list(cc)) {
    cols <- as.integer(cc$cols)
    a[sort(unique(cols))] <- as.numeric(rowsum(as.numeric(cc$w), cols))
  } else {
    a[unique(as.integer(cc))] <- 1
  }
  a
}

# The rows as the columns of a sparse n_x x k matrix (A').
.constraint_rows_matrix <- function(constr, n_x) {
  ij <- lapply(seq_along(constr), function(g) {
    cc <- constr[[g]]
    if (is.list(cc)) {
      data.frame(i = as.integer(cc$cols), j = g, x = as.numeric(cc$w))
    } else {
      u <- unique(as.integer(cc))
      data.frame(i = u, j = rep(g, length(u)), x = rep(1, length(u)))
    }
  })
  ij <- do.call(rbind, ij)
  Matrix::sparseMatrix(i = ij$i, j = ij$j, x = ij$x,
                       dims = c(n_x, length(constr)))
}

.constraint_rows_for_cpp <- function(constr) {
  lapply(constr %||% list(), function(cc) {
    if (is.list(cc)) list(cols = as.integer(cc$cols), w = as.numeric(cc$w))
    else as.integer(cc)
  })
}

# The constraint every cell's inner Gaussian carries, as the engine solved it,
# read off a fit or a kernel result `src`: the hard rows `constraint_cols` the
# Gaussian is conditioned on (shared by every cell), and the sum-to-zero pins
# the stored precision leaves off (`Q_soft_cols`, folded at solve time), with
# each cell's precision for them (`Q_soft_D_per_grid[[k]]`, NULL where the
# cell's precision holds every pin). The Gaussian of cell k has precision
# Q_k + S' D_k S, conditioned on the hard rows. `cells` subsets the per-cell
# part to the precisions a caller holds.
.inner_constraint <- function(src, cells = NULL) {
  D <- src$Q_soft_D_per_grid
  if (!is.null(D) && !is.null(cells)) D <- D[cells]
  list(hard = src$constraint_cols %||% list(),
       soft = if (!is.null(D)) src$Q_soft_cols,
       soft_D = D)
}

# Cell k of an `.inner_constraint()`.
.inner_constraint_cell <- function(cons, k) {
  Dk <- if (!is.null(cons$soft_D)) cons$soft_D[[k]]
  list(hard = cons$hard, soft = if (!is.null(Dk)) cons$soft, soft_D = Dk,
       soft_D_per_grid = list(Dk))
}

# The constraint arguments every stored-precision entry takes
# (read_vcov_grid_inputs): the hard rows, the soft rows and the per-cell
# precision list aligned with the precisions handed over.
.inner_constraint_cpp <- function(cons) {
  list(A_cols_list = .constraint_rows_for_cpp(cons$hard),
       soft_cols_list = if (!is.null(cons$soft_D))
         .constraint_rows_for_cpp(cons$soft),
       soft_D_per_grid = cons$soft_D)
}

# --- Per-prior registry & dispatcher ---
#
# Each entry describes one supported prior `type`: the fields its converters
# index (`required`), how its default outer axes are filled (`defaults`) and
# the axes it contributes to the outer grid (`theta`). The block itself is
# written for the joint driver by `.nl_block_as_arm()`.

# --- Required block fields, per dispatch path -------------------------------
#
# Each registry entry declares which fields its converter indexes, keyed by the
# path that consumes them. The C++ kernels read those vectors positionally with
# no bounds check, so a block naming a field wrongly -- a typo, a stale name
# after a rename, a block copied from a different family -- arrives with a
# zero-length vector and takes the session down. Checking
# presence at the R boundary turns that into a message naming the fields.
#
# Path keys:
#   axis   read by defaults() / theta() while the outer grid is built, so
#          checked on every path.
#   multi  extra fields .nl_block_as_arm() indexes to write a registry block
#          in the one-arm form the joint converter reads.
#   joint  extra fields .joint_block_spec_for_cpp() indexes.
#   projected
#          the fields a projected block carries on a multi-block path: an areal
#          block's graph or an NNGP block's neighbour structure, with
#          `projector` in place of `spatial_idx`.
# A key absent from an entry means the type is not dispatchable on that path;
# the path's own converter raises its "only supported inside ..." message.
.NL_REQ_AREAL_GRAPH <- c("n_spatial_units", "adj_row_ptr", "adj_col_idx",
                         "n_neighbors")
.NL_REQ_AREAL <- c("spatial_idx", .NL_REQ_AREAL_GRAPH)
.NL_REQ_TEMPORAL <- c("temporal_idx", "n_times")
.NL_REQ_NNGP_FIELD <- c("coords", "nn_idx", "nn_dist", "n_spatial", "nn")
.NL_REQ_HSGP_ARM <- c("m_total", "phi", "n_obs_per_arm", "eigenvalues")
.NL_REQ_SPDE_MESH <- c("n_mesh", "A_x", "A_i", "A_p",
                       "C0_diag", "G1_x", "G1_i", "G1_p", "nu")

# `paths` is one or more keys above; their union is checked. A field that is
# present but empty counts as missing -- a zero-length vector is exactly what
# the kernel indexes out of bounds.
.nl_check_block_fields <- function(p, paths, block_index = NULL) {
  type <- tolower(p$type %||% "")
  spec <- .NL_REGISTRY[[type]]
  if (is.null(spec) || is.null(spec$required)) return(invisible(p))
  req <- unique(unlist(spec$required[paths], use.names = FALSE))
  if (!length(req)) return(invisible(p))
  bad <- req[vapply(req, function(f) length(p[[f]]) == 0L, logical(1))]
  if (!length(bad)) return(invisible(p))
  stop("prior block ", if (is.null(block_index)) "" else paste0(block_index, " "),
       "'", type, "' is missing required field(s): ",
       paste(bad, collapse = ", "), ".", call. = FALSE)
}

# The log-Cholesky coordinate names of a p x p Sigma, in the order the grid
# columns run: column by column down the lower triangle (L11, L21, ..., Lp1,
# L22, ...).
.mcar_logchol_names <- function(p) {
  unlist(lapply(seq_len(p), function(j) sprintf("L%d%d", j:p, j)))
}

# A block's `logchol_grid`, the default one where none is given; a supplied
# grid without column names takes the layout's names, so every reader of the
# outer grid finds its axes by name.
.mcar_logchol_grid <- function(p) {
  n_fields <- as.integer(p$n_fields)
  g <- p$logchol_grid
  if (is.null(g)) return(.mcar_default_logchol_grid(n_fields))
  nm <- .mcar_logchol_names(n_fields)
  if (!is.matrix(g) || ncol(g) != length(nm)) {
    stop("`logchol_grid` has ", NCOL(g), " column(s); ", n_fields,
         " coupled fields have ", length(nm), " log-Cholesky coordinates (",
         paste(nm, collapse = ", "), ").", call. = FALSE)
  }
  if (is.null(colnames(g))) colnames(g) <- nm
  g
}

# Default outer grid for a p-field separable MCAR, returned in column-major
# lower-triangle log-Cholesky coordinates of Sigma = L L' (the columns the
# integrator reads). The nodes are laid as a tensor in the marginal SDs and the
# canonical partial correlations -- the "cpc" design (`.hp_logchol_design()`),
# where the correlation nodes sit at sensible correlation values and a field SD
# the data hold weakly has no log-Cholesky funnel to run down -- and mapped to
# the columns by `.hp_logchol_from_coords()`. At p = 2 the one partial
# correlation is the correlation itself.
.mcar_default_logchol_grid <- function(p) {
  p <- as.integer(p)
  sig <- .nl_grid_axis(if (p == 2L) "mcar_sd" else "mcar_sd_wide")
  cor <- .nl_grid_axis(if (p == 2L) "mcar_rho" else "mcar_pcor")
  nm <- .mcar_logchol_names(p)
  is_diag <- unlist(lapply(seq_len(p), function(j) (j:p) == j))
  axes <- lapply(is_diag, function(d) if (d) log(sig) else atanh(cor))
  C <- as.matrix(do.call(expand.grid, c(axes, list(KEEP.OUT.ATTRS = FALSE))))
  out <- .hp_logchol_from_coords(C, list(design = "cpc", p = p))
  attr(out, "log_jac") <- NULL
  dimnames(out) <- list(NULL, nm)
  out
}

.NL_REGISTRY <- list(
  icar = list(
    required = list(multi = .NL_REQ_AREAL,
                    joint = .NL_REQ_AREAL,
                    projected = .NL_REQ_AREAL_GRAPH),
    defaults = function(p, a) .nl_fill_family_axes(p, "icar"),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  bym2 = list(
    required = list(multi = .NL_REQ_AREAL,
                    joint = .NL_REQ_AREAL,
                    projected = .NL_REQ_AREAL_GRAPH),
    defaults = function(p, a) {
      # The mixing weight enters as sqrt(rho) and sqrt(1 - rho).
      .nl_check_axis_interval(p, "rho_grid", 0, 1, "a BYM2 mixing weight",
                              closed = TRUE)
      .nl_fill_family_axes(p, "bym2")
    },
    theta = function(p) list(
      grid  = cbind(sigma = p$sigma_grid, rho = p$rho_grid),
      names = c("sigma", "rho")
    )
  ),

  car_proper = list(
    required = list(multi = .NL_REQ_AREAL,
                    joint = .NL_REQ_AREAL,
                    projected = .NL_REQ_AREAL_GRAPH),
    defaults = function(p, a) {
      # `rho_car_grid` is the joint-API spelling of the correlation axis; accept
      # it as an alias so a grid passed under either name is honoured.
      p$rho_grid <- p$rho_grid %||% p$rho_car_grid
      # Both axes supplied -> treat as the pre-paired integration grid (the
      # theta builder pairs them with cbind), matching bym2 / ar1.
      if (!is.null(p$tau_grid) && !is.null(p$rho_grid)) {
        return(.nl_check_paired_axes(p, c("tau_grid", "rho_grid")))
      }
      # Otherwise default each missing axis independently and cross the two, so
      # supplying only one axis keeps it instead of discarding it.
      rb <- p$rho_bounds %||% .nl_grid_par("car_rho", "bounds")
      # Margin in from the eigenvalue endpoints -- Q goes singular at the
      # boundary, so anchoring the grid in (lower + eps, upper - eps) avoids
      # NaN log-determinants from the per-grid-point Cholesky.
      eps <- .nl_grid_par("car_rho", "margin") * (rb[2] - rb[1])
      g_tau <- if (is.null(p$tau_grid))
        .nl_grid_axis("car_tau") else sort(unique(p$tau_grid))
      g_rho <- if (is.null(p$rho_grid))
        seq(rb[1] + eps, rb[2] - eps,
            length.out = .nl_grid_par("car_rho", "n")) else sort(unique(p$rho_grid))
      gr <- expand.grid(tau = g_tau, rho = g_rho)
      p$tau_grid <- gr$tau; p$rho_grid <- gr$rho
      p
    },
    theta = function(p) list(
      grid  = cbind(tau = p$tau_grid, rho = p$rho_grid),
      names = c("tau", "rho")
    )
  ),

  mcar = list(
    # Separable multivariate CAR (p coupled areal fields sharing Sigma (x) Q^-1).
    # Multi-block-only (like iid): the coupled inner solve lives in the joint
    # driver. The outer axes are the p(p+1)/2 log-Cholesky coordinates of Sigma.
    required = list(axis = "n_fields",
                    joint = c(.NL_REQ_AREAL, "n_fields", "field_weight")),
    defaults = function(p, a) {
      p$logchol_grid <- .mcar_logchol_grid(p)
      p
    },
    theta = function(p) {
      g <- p$logchol_grid
      list(grid = g, names = colnames(g))
    }
  ),

  miid = list(
    # Multivariate IID: p coupled per-group coefficient
    # fields sharing a free Sigma -- the non-spatial sibling of mcar (Q = I, so
    # no graph). Multi-block-only (like mcar / iid): the coupled inner solve
    # lives in the joint driver. The outer axes are the same p(p+1)/2
    # log-Cholesky coordinates of Sigma; only the per-field precision (I vs
    # D - W) differs, which does not change the outer grid, so the default grid
    # is reused from mcar.
    required = list(axis = "n_fields",
                    joint = c("obs_idx", "n_groups", "n_fields", "field_weight")),
    defaults = function(p, a) {
      p$logchol_grid <- .mcar_logchol_grid(p)
      p
    },
    theta = function(p) {
      g <- p$logchol_grid
      list(grid = g, names = colnames(g))
    }
  ),

  nngp = list(
    # spatial_idx / nn_order / cov_type are defaulted on the single-block path,
    # so they are not required there; the multi-block path reads spatial_idx.
    required = list(multi = c(.NL_REQ_NNGP_FIELD, "spatial_idx"),
                    joint = c(.NL_REQ_NNGP_FIELD, "spatial_idx"),
                    projected = .NL_REQ_NNGP_FIELD),
    defaults = function(p, a) .nl_fill_family_axes(p, "nngp"),
    theta = function(p) list(
      grid  = cbind(sigma2 = p$sigma2_grid, phi_gp = p$phi_gp_grid),
      names = c("sigma2", "phi_gp")
    )
  ),

  spde = list(
    # Continuous Matern SPDE field on a shared FEM mesh. INDEXED_MULTI: each
    # observation projects onto ~3 mesh nodes through the FEM matrix A, so the
    # block is consumed via obs_indices (the A CSC slots), not a one-node
    # spatial_idx. Same SpdeQBuilder and make_spde_block factory the
    # single-Laplace occupancy SPDE path (cpp_nested_laplace_spde) uses.
    # The registry block reads a scalar `n_obs`; the joint one reads a per-arm
    # `n_obs_per_arm` alongside per-arm A CSC triples.
    required = list(multi = c(.NL_REQ_SPDE_MESH, "n_obs"),
                    joint = c(.NL_REQ_SPDE_MESH, "n_obs_per_arm")),
    defaults = function(p, a) {
      if (is.null(p$range_grid) || is.null(p$sigma_grid)) {
        # Geometric grid centred on the PC-prior mode (prior_range[1] /
        # prior_sigma[1]). The span is kept tight (mode / SPAN .. mode * SPAN)
        # rather than the wide [mode*0.3, mode*3] window: a rectangular
        # Cartesian grid (no CCD mode-find here) that runs to a very small
        # range / large sigma corner lets the binary-occupancy field over-fit
        # there, and the corner cell -- being the highest inner likelihood --
        # dominates the weighted field even though the PC prior disfavours it.
        # Anchoring the grid near the regularised prior mode keeps the
        # integrated field in the well-identified region the single-Laplace
        # SPDE path fits at the fixed mode, while still integrating modest
        # hyperparameter uncertainty.
        n_g <- as.integer(p$n_grid %||% .nl_grid_par("spde_registry", "n"))
        span <- as.numeric(p$grid_span %||% .nl_grid_par("spde_registry", "span"))
        rng_mode <- (.hp_range_anchor(p)$anchor %||%
                       .nl_grid_par("spde_registry", "centre_range"))[1]
        sig_mode <- .hp_scale_anchor(p)[1]
        rg <- exp(seq(log(rng_mode / span), log(rng_mode * span), length.out = n_g))
        sg <- exp(seq(log(sig_mode / span), log(sig_mode * span), length.out = n_g))
        gr <- expand.grid(range = rg, sigma = sg)
        p$range_grid <- gr$range
        p$sigma_grid <- gr$sigma
      }
      p
    },
    theta = function(p) list(
      grid  = cbind(range = as.numeric(p$range_grid),
                    sigma = as.numeric(p$sigma_grid)),
      names = c("range", "sigma")
    )
  ),

  st_interaction = list(
    # Knorr-Held space-time interaction (types I-IV), built by
    # .st_interaction_block() from a spatiotemporal() spec: precision tau R on
    # the S x T cells, conditioned on the rows spanning null(R). Multi-block
    # only (the hard constraints live in the sparse joint Newton).
    required = list(multi = c("obs_idx", "n_latent", "S_p", "S_i", "S_x",
                              "log_pdet"),
                    joint = c("obs_idx", "n_latent", "S_p", "S_i", "S_x",
                              "log_pdet")),
    defaults = function(p, a) .nl_fill_family_axes(p, "st_interaction"),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  iid = list(
    # Unstructured (iid) Gaussian latent, integrated over a single sigma axis.
    # The C++ joint block parser builds the block with d_fac(k) = sigma_k and an
    # N(0, 1) prior. A one-point sigma_grid conditions the field at a FIXED
    # sigma (an EM-estimated random-effect term on a nested block carries its
    # own grid this way; see .fit_block_via_nested_laplace).
    required = list(multi = c("obs_idx", "n_units"),
                    joint = c("obs_idx", "n_units")),
    defaults = function(p, a) .nl_fill_family_axes(p, "iid"),
    theta = function(p) list(grid = as.numeric(p$sigma_grid), names = "sigma")
  ),

  hsgp = list(
    # The single-block kernel takes the basis as one matrix (phi_basis) with its
    # eigenvalues (lambda_eig); the joint one takes a per-arm list (phi) with
    # `eigenvalues` and the basis / row counts it needs to slice them.
    required = list(multi = c("phi_basis", "lambda_eig"),
                    joint = .NL_REQ_HSGP_ARM),
    defaults = function(p, a) .nl_fill_family_axes(p, "hsgp"),
    theta = function(p) list(
      grid  = cbind(sigma2 = p$sigma2_grid, lengthscale = p$lengthscale_grid),
      names = c("sigma2", "lengthscale")
    )
  ),

  hsgp_mo = list(
    # Multi-output (co-regionalization) HSGP block. K = n_arms
    # correlated latent fields share basis + eigenvalues, cross-output
    # coefficients live in Sigma. First ship: K = 2 with raw axes
    # (sigma_1, sigma_2, rho, ell) and the natural Cartesian product over
    # paired per-axis grids. Multi-block-only -- there is no single-arm
    # version of multi-output HSGP.
    required = list(joint = .NL_REQ_HSGP_ARM),
    defaults = function(p, a) {
      # All four grids treated as paired axes (each row of theta_grid is
      # one (sigma_1, sigma_2, rho, ell) tuple). When any grid is missing
      # we fill in a small Cartesian default; when the user supplies grids
      # they must already be aligned (use expand.grid() on the R side to
      # form a Cartesian, then pass the four aligned columns).
      .nl_fill_family_axes(p, "hsgp_mo")
    },
    theta = function(p) list(
      grid  = cbind(sigma_1 = as.numeric(p$sigma_1_grid),
                    sigma_2 = as.numeric(p$sigma_2_grid),
                    rho     = as.numeric(p$rho_grid),
                    ell     = as.numeric(p$lengthscale_grid)),
      names = c("sigma_1", "sigma_2", "rho", "ell")
    )
  ),

  rw1 = list(
    required = list(multi = .NL_REQ_TEMPORAL,
                    joint = .NL_REQ_TEMPORAL, projected = "n_times"),
    defaults = function(p, a) .nl_fill_family_axes(p, "rw1"),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  rw2 = list(
    required = list(multi = .NL_REQ_TEMPORAL,
                    joint = .NL_REQ_TEMPORAL, projected = "n_times"),
    defaults = function(p, a) .nl_fill_family_axes(p, "rw2"),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  ar1 = list(
    required = list(multi = .NL_REQ_TEMPORAL,
                    joint = .NL_REQ_TEMPORAL, projected = "n_times"),
    defaults = function(p, a) {
      # The temporal kernel takes rho as given, so an AR1 correlation outside
      # (-1, 1) -- a non-stationary process -- is refused here, by name.
      .nl_check_axis_interval(p, "rho_grid", -1, 1, "an AR1 correlation")
      .nl_fill_family_axes(p, "ar1")
    },
    theta = function(p) list(
      grid  = cbind(tau = p$tau_grid, rho = p$rho_grid),
      names = c("tau", "rho")
    )
  ),

  lf = list(
    # Latent factor block, one factor per block. No outer-grid axes: the
    # inner solve integrates the scores and places the loadings at their
    # score-marginal mode. Returns a 1 x 0 grid so the joint Cartesian
    # product treats this block as a "no axis" contributor.
    required = list(joint = c("obs_idx", "n_latent")),
    defaults = function(p, a) p,
    theta = function(p) list(
      grid  = matrix(numeric(0), nrow = 1L, ncol = 0L),
      names = character(0)
    )
  ),

  tgmrf = list(
    # User-defined GMRF block (see ?tgmrf). Only available inside a multi-
    # block prior. The outer theta-grid is the Cartesian product of
    # per-axis grids: `block$n_grid[j]` (default `.NL_GRID$tgmrf_axis$n`)
    # equispaced points between block$grid_span$lower[j] and
    # block$grid_span$upper[j], or block$init[j] +/- `half_width` if the block
    # states no span. Q_k / logdet / log p(theta_k) at the cells of every
    # solve are evaluated from the block's own closures and packed into the
    # C++ block spec (`.joint_tgmrf_at_cells()`, `.nl_tgmrf_per_grid()`).
    required = list(
      axis  = c("theta_dim", "theta_names"),
      multi = c("n_latent", "obs_idx"),
      joint = c("n_latent", "obs_idx")),
    defaults = function(p, a) {
      if (is.null(p$theta_grid_built)) {
        d <- p$theta_dim
        axes <- vector("list", d)
        n_ax <- rep_len(as.integer(p$n_grid %||%
                                     .nl_grid_par("tgmrf_axis", "n")), d)
        for (j in seq_len(d)) {
          if (!is.null(p$grid_span)) {
            lo <- p$grid_span$lower[j]
            hi <- p$grid_span$upper[j]
          } else {
            hw <- .nl_grid_par("tgmrf_axis", "half_width")
            lo <- p$init[j] - hw
            hi <- p$init[j] + hw
          }
          axes[[j]] <- seq(lo, hi, length.out = n_ax[j])
        }
        names(axes) <- p$theta_names
        gr <- do.call(expand.grid, axes)
        p$theta_grid_built <- as.matrix(gr)
      }
      p
    },
    theta = function(p) list(
      grid  = p$theta_grid_built,
      names = p$theta_names
    )
  )
)

# Per-cell log weight the cheap screen ranks with: the hyperprior R folds into
# `log_marginal` after the kernel returns, plus the cell measure the grid's
# weights are built with (`.nl_attach_outer_integration()`), in the grid's cell
# order. Both are fixed by the grid before any cell is solved, and together with
# the kernel's log-marginal they are the posterior weight a cell ends up with,
# so the screen and the full pass rank the same quantity.
#
# `screen` is the record a refinement round hands its kernel
# (`.hyper_refine_screen()`): the cells are a batch joining a solved grid, so
# their measure is read on the merged grid (`log_measure`) rather than on the
# batch, and the grid's own mass (`log_ref`) travels as the offset's `log_ref`
# attribute, which the kernel's screen normalises against.
.nl_screen_log_offset <- function(tg, hp_parts, specs = NULL, screen = NULL) {
  tg <- as.matrix(tg)
  n  <- nrow(tg)
  lp <- numeric(n)
  axes <- character(0)
  support <- list()
  for (hp in hp_parts) {
    lp <- lp + hp$lp
    axes <- union(axes, hp$axes)
    support[names(hp$support)] <- hp$support
  }
  lq <- if (!is.null(screen)) {
    if (length(screen$log_measure) != n) {
      stop("A refinement screen measures ", length(screen$log_measure),
           " cells; the kernel call holds ", n, ".", call. = FALSE)
    }
    screen$log_measure
  } else {
    if (is.null(specs) && !is.null(colnames(tg))) {
      specs <- .joint_axis_specs_from_grid(tg, folded_axes = axes,
                                           axis_support = support)
    }
    .hyper_log_quad_weights(tg, specs)
  }
  off <- lp + (if (length(lq) == n) lq else 0)
  off[is.na(off)] <- -Inf
  if (!is.null(screen)) attr(off, "log_ref") <- as.numeric(screen$log_ref)
  off
}

# Grid columns a block declares on their own natural coordinate whatever their
# names: a tgmrf() block's theta, laid evenly in theta by the registry
# (`.NL_REGISTRY$tgmrf$theta`) and carrying the user's prior as a density on
# theta. Read off the result's block layout (`blocks`, `axis_offsets`), so the
# cell measure of such an axis is its width in theta rather than undeclared.
.NL_LINEAR_AXIS_BLOCKS <- "tgmrf"

.nl_linear_axes <- function(res) {
  tg <- .nl_theta_matrix(res)
  blocks <- res$blocks
  ao <- as.integer(res$axis_offsets)
  if (is.null(tg) || !is.list(blocks) || length(ao) != length(blocks) + 1L) {
    return(NULL)
  }
  cn <- colnames(tg)
  out <- character(0)
  for (b in seq_along(blocks)) {
    if (ao[b + 1L] <= ao[b]) next
    if (tolower(blocks[[b]]$type %||% "") %in% .NL_LINEAR_AXIS_BLOCKS) {
      out <- c(out, cn[(ao[b] + 1L):ao[b + 1L]])
    }
  }
  out
}

# The outer grid's cell measure, its weights and its log evidence, set together
# from the grid and the log-marginals the fit carries. `log_marginal` already
# holds the hyperprior density the dispatch folded in (`R/hyperprior_default.R`),
# so a folded axis is measured by its cell widths alone and its support is not
# clipped to the span the nodes were declared over.
.nl_attach_outer_integration <- function(res, what = "outer grid") {
  tg    <- .nl_theta_matrix(res)
  if (!is.null(res$outer_log_measure)) {
    res$log_quad <- res$outer_log_measure
    res$weights <- .nl_normalise_weights_safe(res$log_marginal, what,
                                               log_quad = res$log_quad)
    if (!is.null(res$outer_qmc)) res$outer_qmc$ess <- 1 / sum(res$weights^2)
    if (!is.null(res$outer_quadrature)) {
      res$outer_quadrature$ess <- 1 / sum(res$weights^2)
    }
    return(.nl_attach_evidence(res, tg, NULL, log_measure = res$log_quad))
  }
  # A node a refinement pass appended was not declared, so the specs are read
  # off the base cells and the cells' tags carry the rest.
  refining <- res$refining_axis
  base  <- !nzchar(.hyper_slice_home(refining, nrow(tg)))
  specs <- .joint_axis_specs_from_grid(
    tg[base, , drop = FALSE], folded_axes = res$log_hyperprior_axes,
    linear_axes = .nl_linear_axes(res),
    axis_support = res$log_hyperprior_support)
  res$log_quad     <- .hyper_log_quad_weights(tg, specs, refining = refining)
  res$axis_support <- .hyper_grid_supports(tg, specs, refining = refining)
  res$outer_axis_specs <- specs
  res$weights      <- .nl_normalise_weights_safe(res$log_marginal, what,
                                                 log_quad = res$log_quad)
  res <- .nl_prune_merged_record(res)
  .nl_attach_evidence(res, tg, specs)
}

# Normalise log-marginals to integration weights summing to 1. The single
# weight normaliser for every outer hyperparameter grid: it drops non-finite
# nodes (an inner Newton diverging in a grid corner returns +Inf / NaN) from
# the max-shift, zeroes their weight, and renormalises over the finite cells;
# all-NA (with a warning) when no node carries mass, never NaN. There is no
# unguarded variant -- a single `max(lm)` over a grid with one +Inf / NaN cell
# would collapse every weight to NaN and break the grid sampler. `what` names
# the grid in the degenerate-case warning.
# `log_quad` adds the per-cell log quadrature weight of the outer grid (see
# R/hyper_quadrature.R), so posterior weight is likelihood times prior mass
# rather than likelihood alone. Callers integrating over a hyperparameter grid
# pass it; callers weighting something that is not a quadrature rule leave it
# NULL.
#' Normalise outer-grid log-marginals to posterior cell weights
#'
#' The single weight normaliser for every outer hyperparameter grid: softmax
#' of `lm` (optionally plus a per-cell log quadrature weight) into weights
#' summing to 1. Non-finite nodes (an inner Newton diverging in a grid
#' corner) are dropped from the max-shift and zeroed before renormalising
#' over the finite cells; an all-non-finite grid returns all-`NA` with a
#' warning rather than `NaN`. Intended for reconstructing a fit's outer-grid
#' posterior weights (with [tulpa_theta_matrix()] and
#' [tulpa_grid_log_quad()]) when the fit doesn't already carry `fit$weights`.
#'
#' @param lm Numeric vector of per-cell log-marginal-likelihood values.
#' @param what Label for the grid, used only in the degenerate-case warning.
#' @param log_quad Optional per-cell log quadrature weight (prior mass) of
#'   the outer grid, e.g. from [tulpa_grid_log_quad()]; `NULL` normalises the
#'   likelihood alone.
#' @return Numeric vector of posterior cell weights summing to 1, the same
#'   length as `lm`, or all-`NA` if no cell carries finite mass.
#' @seealso [tulpa_theta_matrix()], [tulpa_grid_log_quad()]
#' @examples
#' tulpa_normalise_weights_safe(c(-1, 0, -2))
#' # With per-cell log quadrature weights (cell prior masses):
#' tulpa_normalise_weights_safe(c(-1, 0, -2), log_quad = log(c(0.2, 0.5, 0.3)))
#' @export
tulpa_normalise_weights_safe <- function(lm, what = "grids / data",
                                          log_quad = NULL) {
  .nl_normalise_weights_safe(lm, what = what, log_quad = log_quad)
}

.nl_normalise_weights_safe <- function(lm, what = "grids / data",
                                       log_quad = NULL) {
  if (!is.null(log_quad)) {
    if (length(log_quad) != length(lm)) {
      stop(sprintf("`log_quad` has length %d but `lm` has length %d (%s).",
                   length(log_quad), length(lm), what), call. = FALSE)
    }
    lm <- lm + log_quad
    lm[is.na(lm)] <- -Inf
  }
  finite_lm <- lm[is.finite(lm)]
  if (length(finite_lm) == 0L) {
    warning(sprintf("All grid points returned non-finite log_marginal -- check %s.", what),
            call. = FALSE)
    return(rep(NA_real_, length(lm)))
  }
  w <- exp(lm - max(finite_lm))
  w[!is.finite(w)] <- 0
  if (sum(w) == 0) return(rep(NA_real_, length(lm)))
  w / sum(w)
}

# Log evidence of an outer hyperparameter grid.
#
# `log_marginal` carries a normalised hyperprior density on each axis's
# integration coordinate and `log_measure` is each cell's ABSOLUTE log volume on
# that coordinate (its widths, and a declared atom's probability), so
#
#   log p(y) = log sum_k exp(log_marginal_k + log_measure_k)
#
# is the evidence under the prior itself, not under a copy of it restricted to
# the grid and renormalised there. Where the grid covers the posterior the
# value does not move with where the nodes were laid or how many there are.
#
# A cell whose inner solve failed (`log_marginal` NaN, or `+Inf` from a non-PD
# Hessian) contributes nothing: its likelihood is unknown and is not guessed.
# A cell screened out by the pruning pass carries `-Inf` and contributes
# nothing either, which is what screening it asserted.
.nl_outer_log_evidence <- function(log_marginal, log_measure) {
  lq <- as.numeric(log_measure)
  if (length(lq) != length(log_marginal)) {
    stop(sprintf(paste0("Outer evidence: `log_marginal` has length %d but the ",
                        "measure has %d."),
                 length(log_marginal), length(lq)), call. = FALSE)
  }
  known <- !is.na(log_marginal) & log_marginal < Inf & !is.na(lq)
  if (!any(known)) return(NA_real_)
  ev <- .tulpa_logsumexp(log_marginal[known] + lq[known])
  if (is.finite(ev)) ev else NA_real_
}

# Record a grid's evidence on `res`, or why it has none. The evidence exists
# only when every integrated axis carries a normalised density; otherwise it is
# `NA` with `log_evidence_declined = "improper_hyperprior"` and
# `log_evidence_declined_axes` naming each such axis and its reason.
.nl_attach_evidence <- function(res, tg, specs,
                                log_measure = .hyper_log_quad_weights(
                                  tg, specs, absolute = TRUE,
                                  refining = res$refining_axis,
                                  log_marginal = res$log_marginal)) {
  uncovered <- .nl_evidence_uncovered(tg, specs, res)
  res$log_evidence_declined_axes <- if (length(uncovered)) uncovered else NULL
  if (length(uncovered)) {
    res$log_evidence <- NA_real_
    res$log_evidence_declined <- "improper_hyperprior"
    return(res)
  }
  n <- length(res$log_marginal)
  if (is.null(res$outer_log_measure) && !is.null(tg) && nrow(as.matrix(tg)) == n &&
      !length(.hp_integrated_axes(as.matrix(tg)))) {
    # Every axis is pinned, so each is part of the model and the grid integrates
    # nothing: its cells are copies of one point, which together carry measure 1.
    log_measure <- rep(-log(n), n)
  }
  res$log_evidence <- .nl_outer_log_evidence(res$log_marginal, log_measure)
  res$log_evidence_declined <- NULL
  if (is.na(res$log_evidence)) {
    # Every axis is covered and the evidence is still not a number: either some
    # integrated axis carries no declared coordinate to measure a cell by, or no
    # cell holds both a finite marginal and a finite measure.
    undeclared <- Filter(function(s) isTRUE(s$unweighted), specs %||% list())
    undeclared <- vapply(undeclared, `[[`, character(1), "name")
    if (length(undeclared)) {
      res$log_evidence_declined <- "undeclared_axis_coordinate"
      res$log_evidence_declined_axes <- stats::setNames(
        rep("undeclared_axis_coordinate", length(undeclared)), undeclared)
    } else {
      res$log_evidence_declined <- "no_finite_cell"
    }
  }
  res
}


`%||%` <- function(x, y) if (is.null(x)) y else x

# The arguments every registry kernel and the multi-block driver are called
# with: the response and design, the conditioned random-intercept block, the
# family on the kernel's dispersion scale, the inner-solve controls and the
# fixed-effect prior (the caller's, or NULL for the kernels' own weak default).
# `hyperprior` is read by the dispatchers, which fold the outer hyperprior in
# R, and is never handed to a kernel.
.nl_kernel_args <- function(y, n_trials, X, offset, re_idx, n_re_groups,
                            sigma_re, family, phi, max_iter, tol, n_threads,
                            x_init = NULL, store_Q = FALSE,
                            checkpoint_path = "", prune_tol = 0,
                            screen_iters = .nl_screen("iters"),
                            fitted_var = TRUE, beta_prior = NULL,
                            hyperprior = .hp_choice("proper"), n_threads_outer = 1L) {
  list(
    y = as.numeric(y),
    n = as.integer(n_trials),
    offset_nullable = offset,
    X = X,
    re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups),
    sigma_re = as.numeric(sigma_re),
    family = family,
    phi = .phi_to_kernel(family, as.numeric(phi)),
    max_iter = as.integer(max_iter),
    tol = as.numeric(tol),
    n_threads = as.integer(n_threads),
    n_threads_outer = as.integer(n_threads_outer),
    x_init_nullable = x_init,
    store_Q = isTRUE(store_Q),
    checkpoint_path = checkpoint_path,
    prune_tol = prune_tol,
    screen_iters = screen_iters,
    compute_fitted_var = fitted_var,
    beta_prior = .normalize_beta_prior(beta_prior, ncol(X)),
    hyperprior = hyperprior
  )
}

# The record a cell of a grid fit is solved again from when its latent
# posterior is drawn (`.field_cell_solve()`): the dispatcher arguments the grid
# ran under, less the response and the design the fit carries itself, and the
# door (`"registry"`, a single block's kernel, with its `type`; `"multi"`, the
# multi-block driver; `"spde"`, the SPDE grid kernel). `prior` is kept on the
# record when the fit does not carry its block itself (a conditional-Laplace
# fit). `key` names the record in the session's cache of solved cells
# (`.field_cell_solve()`). NULL for a fit through a model-supplied likelihood,
# whose spec is not kept.
.nl_field_solve_record <- function(door, cargs, prior = NULL, likelihood = NULL,
                                   type = NULL, keep_prior = FALSE) {
  if (!is.null(likelihood)) return(NULL)
  cargs[c("y", "X", "checkpoint_path", "screen_refine")] <- NULL
  list(door = door, type = type, args = cargs,
       prior = if (isTRUE(keep_prior)) prior, key = .field_solve_key())
}

# ----------------------------------------------------------------------------
# Per-grid-point fixed-effects Hessian extraction.
#
# When tulpa_nested_laplace() is called with keep_grid_hessians = TRUE the
# inner Laplace solves are run with store_Q = TRUE, so each grid point's full
# posterior precision (over the joint latent vector x = (beta, RE, latent))
# is returned in CSC lower-triangle form. Here we
#
#   1. Reconstruct sparse symmetric Q_k from the CSC triple at grid point k.
#   2. Compute Sigma_{beta beta} = (Q_k^{-1})[1:p, 1:p] via a single sparse
#      solve with the first p columns of the identity as RHS.
#   3. Invert that small p x p block to get the marginal fixed-effects
#      Hessian H_beta = Sigma_{beta beta}^{-1}.
#
# Per-grid memory cost: O(p^2). The dominant compute is the sparse solve,
# already required for the integrated log-marginal -- so adding it here is
# essentially free.
# ----------------------------------------------------------------------------
# `keep_joint` additionally retains the MODAL cell's joint precision as
# `res$H_joint`, which is what the subspace-debias coupling closure reads to
# grow the corrected set over strongly coupled neighbours
# (`.subspace_closure()`). One cell, not `n_grid`: the closure is a SELECTION
# decision, and the selection is made once. Off by default, so a fit that did
# not ask for closure retains nothing extra (gcol33/tulpa#862).
# The MODAL cell's joint posterior precision, assembled from the per-cell CSC
# scratch the inner solves return under `store_Q`. This is what the
# subspace-debias coupling closure reads (`.subspace_closure()`): a coordinate
# strongly coupled to a member of the corrected set and left OUT of it is being
# carried linearly, which is precisely the error the correction removes, so the
# closure needs the precision graph to find it.
#
# ONE cell, not `n_grid`. The closure is a selection decision made once, and
# the modal cell is the one the reported mixture is centred on. Both drivers
# read this -- the single/multi-block grid through
# `.nl_attach_grid_hessians()`, the joint through `.joint_finalize_grid_fixed()`
# -- so the two cannot disagree about which cell the coupling came from
# (gcol33/tulpa#862). NULL when the fit retained no precision, which the caller
# records as a declined closure rather than as "no neighbours found".
.nl_modal_joint_precision <- function(res) {
  Q_p <- res$Q_csc_p_per_grid
  Q_i <- res$Q_csc_i_per_grid
  Q_x <- res$Q_csc_x_per_grid
  n_x <- res$Q_csc_n
  if (is.null(Q_p) || is.null(Q_i) || is.null(Q_x) || is.null(n_x)) return(NULL)
  lm <- res$log_marginal
  if (is.null(lm) || !any(is.finite(lm))) return(NULL)
  k <- which.max(replace(lm, !is.finite(lm), -Inf))
  if (k > length(Q_p) || length(Q_p[[k]]) != n_x + 1L) return(NULL)
  L <- Matrix::sparseMatrix(i = Q_i[[k]], p = Q_p[[k]], x = Q_x[[k]],
                            dims = c(n_x, n_x), index1 = FALSE)
  L + Matrix::t(L) - Matrix::Diagonal(n_x, Matrix::diag(L))
}

#
# `var_cols` names latent coordinates whose per-cell MARGINAL variance,
# diag(Q_k^{-1}), is retained alongside (`grid_field_var`, n_grid x
# length(var_cols), with the columns in `grid_field_var_cols`). The same solve
# delivers it: those unit columns join E. It defaults to the temporal field (the
# `temporal =` block, or a latent(temporal_ar*()) block), so
# temporal() reports each cell's Gaussian -- mode AND curvature -- mixed over
# the grid rather than the cell modes alone, which carried only the spread
# between cells and covered the truth at 57% for a nominal 95%
# (gcol33/tulpa#904). Only a field whose read-back needs it is retained: one
# column per coordinate per cell, and a large spatial field is not asked for.
#
# The field's full within-cell covariance (`grid_field_cov`, a list of
# n_var x n_var matrices parallel to the cells, NULL where a cell was pruned)
# is retained too, so temporal()'s draws carry the correlation BETWEEN time
# points a cell's Gaussian implies and not only each point's marginal: a
# derived quantity over time (a trend, a difference between two years, a
# cumulative sum) read off draws with independent coordinates has the wrong
# spread. Capped at `.NL_FIELD_COV_MAX` doubles over the grid; past the cap only
# the marginals are kept and the draws fall back to independent coordinates.
.NL_FIELD_COV_MAX <- 4e6
#
# `fixed = FALSE` adds the field reads alone and leaves the fixed-effect blocks
# a fit already carries: a joint fit extracts them inside its own solves.

.nl_attach_grid_hessians <- function(res, p_fixed, keep_joint = FALSE,
                                     var_cols = c(
                                       .nl_role_cols(res$blocks,
                                                     res$block_latent_offsets,
                                                     res$Q_csc_n, "temporal",
                                                     all = TRUE),
                                       .nl_temporal_latent_block(res)$cols),
                                     units = .nl_unit_loads(
                                       res$blocks, res$block_latent_offsets,
                                       res$Q_csc_n, res$theta_grid),
                                     budget_cells = length(res$Q_csc_p_per_grid),
                                     fixed = TRUE) {
  Q_p <- res$Q_csc_p_per_grid
  Q_i <- res$Q_csc_i_per_grid
  Q_x <- res$Q_csc_x_per_grid
  n_x <- res$Q_csc_n

  if (is.null(Q_p) || is.null(Q_i) || is.null(Q_x) || is.null(n_x)) {
    warning("keep_grid_hessians = TRUE but the inner solve did not return Q. ",
            "This is a tulpa bug; please report.",
            call. = FALSE)
    return(res)
  }

  n_grid <- length(Q_p)
  # A non-finite marginal has zero integration weight and no conditional
  # Gaussian to retain. Its Newton scratch precision can contain overflow.
  unavailable <- !is.finite(res$log_marginal)
  Q_p[unavailable] <- list(NULL)
  Q_i[unavailable] <- list(NULL)
  Q_x[unavailable] <- list(NULL)
  grid_hessians <- vector("list", n_grid)
  grid_modes    <- vector("list", n_grid)

  var_cols <- as.integer(var_cols)
  var_cols <- var_cols[var_cols > p_fixed & var_cols <= n_x]
  n_var <- length(var_cols)
  field_var <- if (n_var) matrix(NA_real_, n_grid, n_var) else NULL
  # The field's whole within-cell covariance is the lower-right block of the
  # same S, so retaining it costs no further solve -- only the storage, which
  # grows as n_var^2 per cell and is capped by `.NL_FIELD_COV_MAX`.
  keep_cov <- n_var > 0L && as.numeric(budget_cells) * n_var^2 <= .NL_FIELD_COV_MAX
  field_cov <- if (keep_cov) vector("list", n_grid)

  modes_mat <- res$modes  # n_grid x n_x

  # An intrinsic block (ICAR, BYM2's structured part, RW1 / RW2) stands for a
  # field whose constant is a HARD sum-to-zero constraint, and a Knorr-Held
  # interaction carries hard rows of its own. Unconditioned, the inverse puts
  # the level's variance into the intercept's standard error (3-4x the exact
  # posterior SD on a 50-node ICAR, gcol33/tulpa#901) and into an intrinsic
  # temporal field's marginal variance. The driver names every constraint row
  # (`constraint_cols`), and both reads are the block of
  #   Cov(x | A x = 0) = V - W (A W)^-1 W',  V = Q^-1 E,  W = Q^-1 A'
  # (Rue & Held 2005) over the fixed effects and the read field columns.
  #
  # The stored precision need not be invertible on its own: the sparse joint
  # Newton folds the sum-to-zero pins at solve time and leaves them off the
  # stored Q, so two intrinsic fields loading the same rows (an ICAR and an RW1
  # beside a space-time interaction) leave their opposite shift, +c on one
  # level and -c on the other, with no curvature at all. Q^-1 is then of order
  # 1 / round-off along it, A W carries that in one direction, and the O(1)
  # answer is the difference of such terms. The block is therefore read by
  # the engine's own extraction (cpp_joint_inner_vcov_blocks), which folds the
  # conditioning against a precision anchored on the constraint rows' own
  # nodes (InvBlockConstraint, inv_block_extract.h) and is the read a joint
  # fit's retained block uses, and it adds the pins the stored Q leaves off as
  # their soft precision (`.inner_constraint()`). The mode needs no
  # correction: the centring already put it on the constraint.
  cons <- .inner_constraint(res)
  ccpp <- .inner_constraint_cpp(cons)
  sel_idx <- c(seq_len(p_fixed), var_cols)
  n_sel <- length(sel_idx)
  solved <- vapply(Q_p, function(qp) length(qp) == n_x + 1L, logical(1))
  blocks_k <- vector("list", n_grid)
  compact <- n_var > 0L && !keep_cov
  if (any(solved)) {
    blocks_k[solved] <- cpp_joint_inner_vcov_blocks(
      Q_p_per_grid = Q_p[solved], Q_i_per_grid = Q_i[solved],
      Q_x_per_grid = Q_x[solved], n_x = as.integer(n_x),
      idx = as.integer(sel_idx), n_dense = if(compact) p_fixed else n_sel,
      A_cols_list = ccpp$A_cols_list,
      field_marginal = compact,
      n_threads = 1L,
      soft_cols_list = ccpp$soft_cols_list,
      soft_D_per_grid = if (!is.null(ccpp$soft_D_per_grid))
        ccpp$soft_D_per_grid[solved], compact = compact)
  }

  for (k in seq_len(n_grid)) {
    # A cell the cheap screen pruned is never solved, so it holds no precision
    # to retain and its CSC slot is empty. Skip the write and the slot stays
    # blank, which is what `.nested_fixed_moments()` already reads: it drops any
    # cell whose block is absent, and a pruned cell carries log_marginal = -Inf
    # and hence zero weight, so nothing the report reads was in it. Skipping
    # rather than assigning NULL is what keeps `grid_modes` / `grid_hessians`
    # parallel to `weights` -- assigning NULL removes the element and shortens
    # the list, which NAs the whole coefficient table (gcol33/tulpa#345).
    if (!solved[k]) next
    grid_modes[[k]] <- if (!is.null(modes_mat)) {
      as.numeric(modes_mat[k, seq_len(p_fixed)])
    } else {
      rep(NA_real_, p_fixed)
    }
    # A cell whose precision factors neither bare nor with the extraction's
    # relative ridge holds no covariance to read; its slot stays blank and the
    # moments drop it the way they drop a pruned cell.
    S <- blocks_k[[k]]
    if (is.null(S)) next
    Sigma_bb <- if(compact) S$fixed else S[seq_len(p_fixed), seq_len(p_fixed), drop = FALSE]
    grid_hessians[[k]] <- .spd_inverse(Sigma_bb)
    if (n_var) {
      fi <- p_fixed + seq_len(n_var)
      field_var[k, ] <- if(compact) S$field_diag else diag(S)[fi]
      if (keep_cov) {
        Sf <- S[fi, fi, drop = FALSE]
        field_cov[[k]] <- (Sf + t(Sf)) / 2
      }
    }
  }
  res <- .nl_attach_unit_loads(res, units, Q_p, Q_i, Q_x, n_x, cons)

  if (isTRUE(fixed)) {
    res$grid_hessians <- grid_hessians
    res$grid_modes    <- grid_modes
  }
  if (n_var) {
    res$grid_field_var      <- field_var
    res$grid_field_var_cols <- var_cols
    if (keep_cov) res$grid_field_cov <- field_cov
  }
  if (isTRUE(keep_joint)) res$H_joint <- .nl_modal_joint_precision(res)
  # Strip the verbose CSC scratch fields once Hessians are assembled -- except
  # under a Knorr-Held interaction, whose joint draws over the S x T cells
  # (spatiotemporal_effects()) are taken from each cell's precision.
  if (is.null(units$spatiotemporal)) {
    res$Q_csc_p_per_grid <- NULL
    res$Q_csc_i_per_grid <- NULL
    res$Q_csc_x_per_grid <- NULL
    res$Q_csc_n          <- NULL
    res$Q_soft_cols       <- NULL
    res$Q_soft_D_per_grid <- NULL
  }
  res
}

# --- Multi-block dispatch ---
#
# A multi-block prior is `list(block1, block2, ...)` where each blockN is a
# list with a `type` field (same shape as a single-block prior). At each
# outer-grid point all blocks share a Newton solve; the joint grid is the
# Cartesian product of per-block axes.

# Block types whose field may reach the observations through a projector: arm
# k's predictor receives A_k z in place of the unit gather z[spatial_idx] or
# z[temporal_idx] (restricted spatial or temporal regression, A_k = S_k P_perp
# with S_k the observation-to-unit incidence). Both multi-block drivers read
# one; an intrinsic block's (ICAR, BYM2, RW1, RW2) level is placed from A_k 1,
# a proper block's (proper CAR, NNGP, AR1) prior carries its own.
.NL_PROJECTABLE_AREAL <- c("icar", "bym2", "car_proper")
.NL_PROJECTABLE_TEMPORAL <- c("rw1", "rw2", "ar1")
.NL_PROJECTABLE_TYPES <- c(.NL_PROJECTABLE_AREAL, "nngp",
                           .NL_PROJECTABLE_TEMPORAL)

# The per-observation index a projector replaces on a block of `type`.
.nl_projector_idx_field <- function(type) {
  if (type %in% .NL_PROJECTABLE_TEMPORAL) "temporal_idx" else "spatial_idx"
}

.nl_block_is_projected <- function(p) {
  is.list(p) && tolower(p$type %||% "") %in% .NL_PROJECTABLE_TYPES &&
    !is.null(p$projector)
}

# A projected block's per-arm projectors as the column-compressed triple
# the C++ block builders read (0-based `A_i` / `A_p`, the SPDE projector's
# layout). `p$projector` is a list of n_arms matrices, each N_k x n_units
# (dense or a Matrix sparse matrix), or one matrix shared by every arm; a NULL
# entry is an arm the field does not reach. The projector replaces the unit
# gather, so a block carrying its index (`spatial_idx` / `temporal_idx`) as
# well, or a per-row `svc_weight` that belongs inside A_k, is refused.
.nl_block_projector <- function(p, n_arms, block_index, arm_n_obs) {
  type <- tolower(p$type)
  lab <- paste0("Block ", block_index, " (type '", type, "')")
  idx_field <- .nl_projector_idx_field(type)
  if (!is.null(p[[idx_field]])) {
    stop(lab, ": pass either `projector` or `", idx_field, "`, not both; a ",
         "projector already maps the observations to the units.",
         call. = FALSE)
  }
  if (!is.null(p$svc_weight)) {
    stop(lab, ": `projector` and `svc_weight` cannot be combined; scale ",
         "the projector's rows by the weight.", call. = FALSE)
  }
  if (is.null(arm_n_obs)) {
    stop(lab, ": a projector needs the arms' observation counts.",
         call. = FALSE)
  }
  n_units <- as.integer(p$n_spatial_units %||% p$n_spatial %||% p$n_times)
  projector <- p$projector
  if (!is.list(projector)) projector <- rep(list(projector), n_arms)
  if (length(projector) != n_arms) {
    stop(lab, ": `projector` must be a list of length n_arms (", n_arms,
         "), or one matrix shared by every arm.", call. = FALSE)
  }
  csc <- lapply(seq_len(n_arms), function(k) {
    A <- projector[[k]]
    if (is.null(A)) {
      return(list(x = numeric(0), i = integer(0), p = integer(n_units + 1L)))
    }
    if (!(is.matrix(A) || methods::is(A, "Matrix")) ||
        nrow(A) != arm_n_obs[k] || ncol(A) != n_units) {
      stop(lab, ": `projector[[", k, "]]` must be a ", arm_n_obs[k],
           " x ", n_units, " matrix (arm ", k, "'s observations by the ",
           "block's units).", call. = FALSE)
    }
    A <- methods::as(methods::as(Matrix::Matrix(A, sparse = TRUE),
                                 "generalMatrix"), "CsparseMatrix")
    if (!all(is.finite(A@x))) {
      stop(lab, ": `projector[[", k, "]]` holds a non-finite entry.",
           call. = FALSE)
    }
    list(x = as.numeric(A@x), i = as.integer(A@i), p = as.integer(A@p))
  })
  list(A_x = lapply(csc, `[[`, "x"),
       A_i = lapply(csc, `[[`, "i"),
       A_p = lapply(csc, `[[`, "p"))
}

# An NNGP block's neighbour structure as the C++ block builders read it, on
# either the single-block kernel or the multi-block driver. nn_order is passed
# 0-based, the convention cpp_laplace_fit_gp also uses.
.nl_nngp_field_args <- function(p) {
  n_spatial <- as.integer(p$n_spatial)
  list(
    coords    = as.matrix(p$coords),
    nn_idx    = as.matrix(p$nn_idx),
    nn_dist   = as.matrix(p$nn_dist),
    nn_order  = as.integer((p$nn_order %||% seq_len(n_spatial)) - 1L),
    n_spatial = n_spatial,
    nn        = as.integer(p$nn),
    cov_type  = as.integer(p$cov_type %||% 2L)  # default Matern-5/2
  )
}

.is_multi_block_prior <- function(p) {
  is.list(p) && is.null(p$type) &&
    all(vapply(p, function(x) is.list(x) && !is.null(x$type), logical(1)))
}

# What each block of a nested prior is FOR: "spatial", "temporal" or "other".
# tulpa() tags its `spatial =` / `temporal =` blocks with `role`, and on a tagged
# prior every untagged block (an s(x) smoother, a latent() block, a `(1 | g)`
# iid block) is "other" -- a type does not say what a block is for, since a
# smoother is an rw1 / rw2 block and a random intercept an iid one. A prior
# built by hand carries no tags, and its blocks are classified by type
# (`.SPATIAL_NL_TYPES` / `.TEMPORAL_NL_TYPES`). `prior` is a single block or a
# list of blocks; the result has one entry per block.
.nl_block_roles <- function(prior) {
  if (is.null(prior) || !is.list(prior) || !length(prior)) return(character(0))
  blocks <- if (!is.null(prior$type)) list(prior) else prior
  tagged <- vapply(blocks, function(b) {
    r <- if (is.list(b)) b$role else NULL
    if (is.character(r) && length(r) == 1L) r else NA_character_
  }, character(1))
  if (any(!is.na(tagged))) return(ifelse(is.na(tagged), "other", tagged))
  types <- vapply(blocks, function(b) tolower(b$type %||% ""), character(1))
  ifelse(types %in% .SPATIAL_NL_TYPES, "spatial",
         ifelse(types %in% .TEMPORAL_NL_TYPES, "temporal", "other"))
}

# Columns of a nested fit's latent vector (`modes`, the per-cell precision)
# holding the first block whose role is `role`, or NULL. A multi-block solve
# reports where each block starts (`block_latent_offsets`, from the C++ block
# builders, which are what decide a block's width); a single-block solve lays
# out [fixed, RE, field], so its field is the tail. `blocks` is the fit's block
# list (or single block), `n_x` the latent length. `all = TRUE` returns the
# columns of every block of that role, in block order (the components of a
# multi-scale field share one role).
.nl_role_cols <- function(blocks, offsets, n_x, role, all = FALSE) {
  roles <- .nl_block_roles(blocks)
  ks <- which(roles == role)
  if (!length(ks) || is.null(n_x)) return(NULL)
  if (length(roles) == 1L && !is.null(blocks$type)) {
    w <- .nl_single_block_width(blocks)
    if (is.null(w) || w < 1L || w > n_x) return(NULL)
    return(as.integer(seq.int(n_x - w + 1L, n_x)))
  }
  if (!isTRUE(all)) ks <- ks[1L]
  cols <- lapply(ks, function(k) .nl_block_cols(offsets, n_x, k, length(roles)))
  if (any(vapply(cols, is.null, logical(1)))) return(NULL)
  unlist(cols)
}

# Columns of block `k` of an `n_blocks`-block multi-block fit, read off the
# offsets the solve reported; NULL when they do not locate it.
.nl_block_cols <- function(offsets, n_x, k, n_blocks) {
  if (length(offsets) != n_blocks + 1L || is.null(n_x)) return(NULL)
  lo <- offsets[k]
  size <- attr(offsets, "sizes", exact = TRUE)
  hi <- if (length(size) == n_blocks) lo + size[k] else offsets[k + 1L]
  if (!is.finite(lo) || !is.finite(hi) || hi <= lo || hi > n_x) return(NULL)
  as.integer(seq.int(lo + 1L, hi))
}

# Width of a single block's field in the latent vector, for the fields a
# read-back slices: a temporal block (n_times per group) and an areal one (one
# coordinate per unit; BYM2 lays its structured then its unstructured
# component, two per unit). NULL when the block does not say.
.nl_single_block_width <- function(p) {
  nt <- p$n_times
  if (!is.null(nt)) return(as.integer(nt) * max(as.integer(p$n_groups %||% 1L), 1L))
  nu <- p$n_spatial_units
  type <- tolower(p$type %||% "")
  if (is.null(nu) || !type %in% c("icar", "car", "car_proper", "bym2")) return(NULL)
  as.integer(nu) * if (type == "bym2") 2L else 1L
}

# The per-unit loading of a fit's areal spatial field on its latent vector:
# which coordinates make up unit s's contribution to eta, and with what weight
# at cell k. An ICAR / proper-CAR block's coordinate IS the unit's field
# (weight 1). A BYM2 block's unit field is
#   sigma_k * (sqrt(rho_k) * s * phi_s + sqrt(1 - rho_k) * theta_s),
# s the scale factor, so its weights move with the cell's (sigma, rho). The
# unit field of a multi-scale field (two or more spatial NNGP blocks) is the
# SUM of its scales at a distinct location, each block's coordinate being its
# field there: a loading on every block at once, weight 1 each. NULL when the
# fit carries no such field or its columns cannot be located.
.nl_spatial_field_load <- function(blocks, offsets, n_x, theta_grid) {
  cols <- .nl_role_cols(blocks, offsets, n_x, "spatial")
  if (is.null(cols)) return(NULL)
  roles <- .nl_block_roles(blocks)
  kb <- which(roles == "spatial")[1L]
  single <- !is.null(blocks$type)
  blk <- if (single) blocks else blocks[[kb]]
  if (!is.null(blk$projector)) return(NULL)
  type <- tolower(blk$type %||% "")
  if (type == "nngp") {
    ks <- which(roles == "spatial")
    if (single || length(ks) < 2L) return(NULL)
    sp <- blocks[ks]
    if (!all(vapply(sp, function(b) identical(tolower(b$type %||% ""), "nngp") &&
                      is.null(b$projector), logical(1)))) return(NULL)
    n_u <- as.integer(blk$n_spatial %||% NA_integer_)
    cl <- lapply(ks, function(k) .nl_block_cols(offsets, n_x, k, length(roles)))
    if (is.na(n_u) || any(vapply(cl, length, integer(1)) != n_u)) return(NULL)
    return(list(type = "nngp", n_units = n_u, cols = cl,
                weights = function(k) as.list(rep(1, length(cl)))))
  }
  n_u <- as.integer(blk$n_spatial_units %||% NA_integer_)
  if (is.na(n_u)) return(NULL)
  if (type %in% c("icar", "car", "car_proper")) {
    if (length(cols) != n_u) return(NULL)
    return(list(type = type, n_units = n_u, cols = list(cols),
                weights = function(k) 1))
  }
  if (type != "bym2" || length(cols) != 2L * n_u) return(NULL)
  tg <- as.matrix(theta_grid)
  pre <- if (single) "" else .axis_prefix(kb)
  cs <- paste0(pre, c("sigma", "rho"))
  if (is.null(colnames(tg)) || !all(cs %in% colnames(tg))) return(NULL)
  sf <- as.numeric(blk$scale_factor %||% 1)
  list(type = "bym2", n_units = n_u,
       cols = list(cols[seq_len(n_u)], cols[n_u + seq_len(n_u)]),
       weights = function(k) {
         sg <- tg[k, cs[1L]]; rh <- tg[k, cs[2L]]
         c(sg * sqrt(rh) * sf, sg * sqrt(1 - rh))
       })
}

# The random-effect loading of a multi-block fit: every `iid` block, in block
# order (the order ranef() names the groups in). The block is non-centred, an
# N(0, 1) coordinate entering eta as `sigma_k * z` (the joint block parser's
# d_fac), so the random effect a cell implies is its coordinate times that
# cell's sigma. NULL when the fit carries no iid block.
.nl_re_field_load <- function(blocks, offsets, n_x, theta_grid) {
  if (!is.list(blocks) || !is.null(blocks$type) ||
      length(offsets) != length(blocks) + 1L || is.null(n_x)) return(NULL)
  tg <- if (!is.null(theta_grid)) as.matrix(theta_grid)
  cols <- integer(0)
  sig <- list()
  # An iid block tagged as a field (the short-term component of a multi-scale
  # temporal field) is that field's, not a random effect.
  roles <- .nl_block_roles(blocks)
  for (b in seq_along(blocks)) {
    if (!identical(tolower(blocks[[b]]$type %||% ""), "iid") ||
        roles[b] != "other") next
    cb <- .nl_block_cols(offsets, n_x, b, length(blocks))
    if (is.null(cb)) return(NULL)
    nm <- .axis_label(b, "sigma")
    s_fixed <- as.numeric(blocks[[b]]$sigma_grid)[1L]
    cols <- c(cols, cb)
    sig[[length(sig) + 1L]] <- list(n = length(cb), nm = nm, fixed = s_fixed)
  }
  if (!length(cols)) return(NULL)
  list(type = "iid", n_units = length(cols), cols = list(cols),
       weights = function(k) {
         list(unlist(lapply(sig, function(s) {
           v <- if (!is.null(tg) && s$nm %in% colnames(tg)) tg[k, s$nm]
                else s$fixed
           rep(v, s$n)
         })))
       })
}

# The coordinates of every block of `type` -- a `latent()` block (a tgmrf()
# field) or a Knorr-Held interaction (st_interaction), both of whose
# coordinates enter eta as they are, `eta_i += x[obs_idx[i]]` -- concatenated
# in block order; `sizes` holds each block's width, keyed by its position in
# the block list, so an accessor can split them back. NULL when the fit
# carries no such block.
.nl_coord_field_load <- function(blocks, offsets, n_x, type) {
  if (!is.list(blocks) || !is.null(blocks$type) ||
      length(offsets) != length(blocks) + 1L || is.null(n_x)) return(NULL)
  cols <- integer(0)
  sizes <- integer(0)
  for (b in seq_along(blocks)) {
    if (!identical(tolower(blocks[[b]]$type %||% ""), type)) next
    cb <- .nl_block_cols(offsets, n_x, b, length(blocks))
    if (is.null(cb)) return(NULL)
    cols <- c(cols, cb)
    sizes <- c(sizes, stats::setNames(length(cb), as.character(b)))
  }
  if (!length(cols)) return(NULL)
  list(type = type, n_units = length(cols), cols = list(cols),
       sizes = sizes, weights = function(k) 1)
}

# The per-unit fields a nested fit retains as loadings of each cell's latent
# Gaussian, by role: the areal spatial field, the random effects, the
# latent() blocks and the Knorr-Held interactions, and each named scale of a
# multi-scale spatial field on its own (`spatial_<name>`).
.nl_unit_loads <- function(blocks, offsets, n_x, theta_grid) {
  c(list(spatial = .nl_spatial_field_load(blocks, offsets, n_x, theta_grid),
         re      = .nl_re_field_load(blocks, offsets, n_x, theta_grid),
         latent  = .nl_coord_field_load(blocks, offsets, n_x, "tgmrf"),
         spatiotemporal = .nl_coord_field_load(blocks, offsets, n_x,
                                               "st_interaction")),
    .nl_spatial_component_loads(blocks, offsets, n_x))
}

# One loading per named NNGP scale of a multi-scale spatial field, keyed
# `spatial_<name>`: the block's coordinate at each distinct location. Empty
# unless the fit carries more than one spatial-role NNGP block.
.nl_spatial_component_loads <- function(blocks, offsets, n_x) {
  if (!is.list(blocks) || !is.null(blocks$type)) return(list())
  roles <- .nl_block_roles(blocks)
  ks <- which(roles == "spatial")
  if (length(ks) < 2L) return(list())
  out <- list()
  for (k in ks) {
    b <- blocks[[k]]
    nm <- b$name
    if (!identical(tolower(b$type %||% ""), "nngp") || !is.character(nm) ||
        !is.null(b$projector)) next
    cols <- .nl_block_cols(offsets, n_x, k, length(blocks))
    if (is.null(cols) || length(cols) != as.integer(b$n_spatial)) next
    out[[paste0("spatial_", nm)]] <- list(type = "nngp", n_units = length(cols),
                                          cols = list(cols),
                                          weights = function(k) 1)
  }
  out
}

# Per-cell mean and variance of every unit loading in `units`
# (`.nl_unit_loads()`), stored as `grid_<role>_mean` / `grid_<role>_var`
# (n_grid x n_units). The mean is the loading of the cell mode; the variance is
# the loading's quadratic form in the cell's inverse precision, conditioned on
# the sum-to-zero groups `constr`, read off one selected inversion per cell
# (`cpp_loading_marginal_var()`), so it costs nothing per unit beyond the
# loading itself and needs no dense solve against the units' columns.
.nl_attach_unit_loads <- function(res, units, Q_p, Q_i, Q_x, n_x, cons) {
  ccpp <- .inner_constraint_cpp(cons)
  M <- res$modes
  n_grid <- length(Q_p)
  for (role in names(units)) {
    ld <- units[[role]]
    if (is.null(ld)) next
    n_u <- as.integer(ld$n_units)
    n_c <- length(ld$cols)
    idx <- as.integer(do.call(rbind, ld$cols))
    W <- matrix(NA_real_, n_grid, n_u * n_c)
    mu <- matrix(NA_real_, n_grid, n_u)
    for (k in seq_len(n_grid)) {
      wk <- lapply(ld$weights(k), function(w) rep_len(as.numeric(w), n_u))
      W[k, ] <- as.numeric(do.call(rbind, wk))
      if (is.matrix(M) && all(is.finite(M[k, ]))) {
        mu[k, ] <- Reduce(`+`, lapply(seq_len(n_c), function(c)
          wk[[c]] * M[k, ld$cols[[c]]]))
      }
    }
    W[!is.finite(W)] <- 0
    v <- cpp_loading_marginal_var(
      Q_p_per_grid = Q_p, Q_i_per_grid = Q_i, Q_x_per_grid = Q_x,
      n_x = as.integer(n_x), load_idx = idx,
      load_ptr = as.integer(seq(0L, n_u * n_c, by = n_c)),
      load_w = W, A_cols_list = ccpp$A_cols_list,
      n_threads = 1L, soft_cols_list = ccpp$soft_cols_list,
      soft_D_per_grid = ccpp$soft_D_per_grid)
    res[[paste0("grid_", role, "_mean")]] <- mu
    res[[paste0("grid_", role, "_var")]]  <- v
    if (!is.null(ld$sizes)) res[[paste0("grid_", role, "_sizes")]] <- ld$sizes
  }
  res
}

# A registry block in the one-arm form the joint block parser reads
# (`.joint_block_spec_for_cpp()`). The registry spells a few single-response
# fields as one object where the joint parser takes a per-arm list -- an HSGP
# basis (`phi_basis`, `lambda_eig`) and an SPDE projector (`A_x`, `A_i`, `A_p`,
# `n_obs`). A tgmrf block keeps its closures, which the joint driver evaluates
# at the cells of every solve (`.joint_tgmrf_at_cells()`).
.nl_block_as_arm <- function(p, block_index) {
  type <- tolower(p$type)
  # An NNGP block given without an observation map reads one observation per
  # location.
  if (type == "nngp" && is.null(p$spatial_idx) && is.null(p$projector) &&
      !is.null(p$n_spatial)) {
    p$spatial_idx <- seq_len(as.integer(p$n_spatial))
  }
  # A scale or precision axis reaches a logarithm. A copy scale's zero level is
  # a declared point mass, not a node of its log continuum.
  for (f in setdiff(grep("_grid$", names(p), value = TRUE), "alpha_grid")) {
    if (isTRUE(.hyper_axis_scale(sub("_grid$", "", f))) && is.numeric(p[[f]])) {
      .nl_check_axis_interval(p, f, 0, Inf, "a scale or precision")
    }
  }
  .nl_check_block_fields(
    p, c("axis", if (.nl_block_is_projected(p)) "projected" else "multi"),
    block_index)
  arm <- .nl_block_arm_fields(p, type)
  p[names(arm)] <- arm
  p
}

# The per-arm fields `.nl_block_as_arm()` writes onto a registry block, NULL
# where the block already reads the same in both forms.
.nl_block_arm_fields <- function(p, type) {
  if (type == "hsgp" && is.null(p[["phi"]])) {
    phi <- as.matrix(p$phi_basis)
    storage.mode(phi) <- "double"
    if (length(p$lambda_eig) != ncol(phi)) {
      stop("HSGP block: lambda_eig must have length ncol(phi_basis) (",
           ncol(phi), "), got ", length(p$lambda_eig), ".", call. = FALSE)
    }
    return(list(phi = list(phi), m_total = ncol(phi), n_obs_per_arm = nrow(phi),
                eigenvalues = as.numeric(p$lambda_eig)))
  }
  if (type == "spde" && !is.list(p$A_x)) {
    return(list(A_x = list(as.numeric(p$A_x)), A_i = list(as.integer(p$A_i)),
                A_p = list(as.integer(p$A_p)), n_obs_per_arm = as.integer(p$n_obs)))
  }
  NULL
}

# A tgmrf block's precision at every row of its columns of the outer grid, as
# CSC triples with log|Q| and log p(theta). Iterating over the block's columns
# of the JOINT grid (rather than the block's own grid) is what lets a tgmrf
# compose with other blocks: where other blocks vary across joint rows and the
# tgmrf does not, the duplicated Q land at the right joint index. Q and the
# log-prior come from the user's R closures (`backend == "r"`, the default) or
# from the registered C++ spec (`backend == "cpp"`).
.nl_tgmrf_per_grid <- function(p, block_joint_grid) {
  tg <- as.matrix(block_joint_grid)
  n_cells   <- nrow(tg)
  n_latent  <- p$n_latent
  Q_csc_p   <- vector("list", n_cells)
  Q_csc_i   <- vector("list", n_cells)
  Q_csc_x   <- vector("list", n_cells)
  logdet_Q  <- numeric(n_cells)
  log_pi_th <- numeric(n_cells)
  use_cpp <- isTRUE(identical(p$backend, "cpp"))
  if (use_cpp) {
    if (is.null(p$cpp_id) || !nzchar(p$cpp_id)) {
      stop("Internal error: tgmrf block has backend = 'cpp' ",
           "but no cpp_id.", call. = FALSE)
    }
    if (!cpp_tgmrf_registry_has(p$cpp_id)) {
      stop("tgmrf C++ spec '", p$cpp_id, "' is not registered in the ",
           "current R session. Was the user .cpp file sourceCpp'd? ",
           "(Re-load via tgmrf_cpp(cpp_file = ..., id = '", p$cpp_id,
           "').)", call. = FALSE)
    }
  }
  for (k in seq_len(n_cells)) {
    th <- as.numeric(tg[k, ])
    names(th) <- p$theta_names
    if (use_cpp) {
      evk <- cpp_tgmrf_eval(p$cpp_id, th)
      if (as.integer(evk$n) != n_latent) {
        stop("Registered tgmrf_cpp spec '", p$cpp_id,
             "' returned Q of size ", evk$n, " at grid row ", k,
             " but the block was registered with n_latent = ", n_latent,
             ".", call. = FALSE)
      }
      Q_csc_p[[k]] <- as.integer(evk$p)
      Q_csc_i[[k]] <- as.integer(evk$i)
      Q_csc_x[[k]] <- as.numeric(evk$x)
      Qk <- Matrix::sparseMatrix(
        i = Q_csc_i[[k]], p = Q_csc_p[[k]], x = Q_csc_x[[k]],
        dims = c(n_latent, n_latent), index1 = FALSE
      )
      lp <- as.numeric(evk$log_prior)
    } else {
      Qk <- p$Q(th)
      Qk <- methods::as(methods::as(Qk, "generalMatrix"), "CsparseMatrix")
      if (nrow(Qk) != n_latent || ncol(Qk) != n_latent) {
        stop("`Q(theta)` returned ", nrow(Qk), " x ", ncol(Qk),
             " at grid row ", k,
             " but the block was registered with n_latent = ", n_latent, ".",
             call. = FALSE)
      }
      Q_csc_p[[k]] <- as.integer(Qk@p)
      Q_csc_i[[k]] <- as.integer(Qk@i)
      Q_csc_x[[k]] <- as.numeric(Qk@x)
      lp <- as.numeric(p$prior(th))
    }
    # sparse Cholesky log-determinant; falls back to dense det for tiny
    # blocks where Matrix::Cholesky balks.
    ld <- tryCatch({
      ch <- Matrix::Cholesky(methods::as(Qk, "symmetricMatrix"), LDL = FALSE)
      # log|Q| = 2 * sum(log(diag(L))) for the LL' factor.
      2 * Matrix::determinant(ch, sqrt = TRUE, logarithm = TRUE)$modulus
    }, error = function(e) {
      as.numeric(determinant(Qk, logarithm = TRUE)$modulus)
    })
    logdet_Q[k] <- as.numeric(ld)
    if (!is.finite(lp)) {
      stop("`prior(theta)` returned a non-finite value at grid row ", k,
           " (theta = ", paste(format(th, digits = 4), collapse = ", "),
           ").", call. = FALSE)
    }
    log_pi_th[k] <- lp
  }
  list(
    Q_csc_p_per_grid         = Q_csc_p,
    Q_csc_i_per_grid         = Q_csc_i,
    Q_csc_x_per_grid         = Q_csc_x,
    logdet_Q_per_grid        = as.numeric(logdet_Q),
    log_prior_theta_per_grid = as.numeric(log_pi_th)
  )
}

# The registry's response, design and fixed-effect prior (`cargs`) as the one
# arm a joint fit reads: every multi-block registry fit is a one-arm fit on the
# joint driver, its latent blocks owning their observation maps.
.nl_registry_arm <- function(cargs, likelihood = NULL) {
  .normalise_joint_arm(list(
    y           = cargs$y,
    n_trials    = cargs$n,
    X           = cargs$X,
    family      = cargs$family,
    phi         = cargs$phi,
    re_idx      = if (isTRUE(cargs$n_re_groups > 0L)) cargs$re_idx,
    n_re_groups = cargs$n_re_groups,
    sigma_re    = cargs$sigma_re,
    offset      = cargs$offset_nullable,
    likelihood  = likelihood,
    beta_prior  = cargs$beta_prior), 1L)
}

# A multi-block registry fit is the one-arm fit it is on the joint door
# (`tulpa_nested_laplace_joint()`): the registry's response, design and
# fixed-effect prior are the arm, its blocks the joint prior, and the joint
# door's placement pass lays every axis, its refinement and outer designs
# integrate them. What the registry reports beyond a joint fit is added to the
# result -- each cell's field loadings and marginal field variances, read off the
# per-cell precision the joint fit keeps for them (`.nl_attach_grid_hessians()`),
# the record a cell is solved again from (`field_solve`), the response and the
# fixed-effect prior -- and the fit is the registry's (`backend =
# "nested_laplace"`).
.nl_registry_multi_fit <- function(prior, cargs, control, likelihood, hyperprior,
                                   integration, keep_grid_hessians, X, data) {
  prior <- .nl_tgmrf_default_obs_idx(prior, length(cargs$y))
  prior <- lapply(seq_along(prior), function(b) .nl_block_as_arm(prior[[b]], b))
  ctl <- control
  ctl$integration <- integration
  ctl$store_Q <- isTRUE(keep_grid_hessians)
  # The joint door takes a response's dispersion on its R-level convention.
  arm <- .nl_registry_arm(cargs, likelihood)
  arm$phi <- data$phi
  fit <- tulpa_nested_laplace_joint(
    stats::setNames(list(arm), .JOINT_ARM_NAME),
    prior = prior, hyperprior = hyperprior, control = ctl)

  oldClass(fit) <- NULL
  fit[c("responses", "arm_layout", "eta_layout", "N", "cell_coupling",
        "copy", "fixed_names")] <- NULL
  fit$backend <- "nested_laplace"
  fit$prior <- .nl_grid_provenance(fit$prior)$prior
  if (isTRUE(keep_grid_hessians)) {
    fit <- .nl_attach_grid_hessians(fit, ncol(X), fixed = FALSE)
  }
  fit$beta_prior <- cargs$beta_prior
  fit$field_solve <- .nl_field_solve_record(
    "multi", utils::modifyList(cargs, list(checkpoint_path = "", prune_tol = 0)),
    fit$prior, likelihood)
  for (nm in names(data)) fit[nm] <- list(data[[nm]])
  .finalize_fit(fit, backend = "nested_laplace",
                n_fixed = ncol(X), fixed_names = colnames(X),
                extra_class = c("tulpa_nested_laplace", "list"), data = data)
}

# A tgmrf block given without an observation map reads one observation per
# latent slot, the identity map; a block whose latent size is not the
# observation count has to state its map.
.nl_tgmrf_default_obs_idx <- function(prior_list, n_obs) {
  for (b in seq_along(prior_list)) {
    blk <- prior_list[[b]]
    if (identical(tolower(blk$type %||% ""), "tgmrf") && is.null(blk$obs_idx)) {
      if (blk$n_latent != n_obs) {
        stop("tgmrf block has n_latent = ", blk$n_latent,
             " but data has N = ", n_obs, " observations. Provide an explicit ",
             "`obs_idx` argument to tgmrf() mapping observations to latent slots.",
             call. = FALSE)
      }
      prior_list[[b]]$obs_idx <- seq_len(n_obs)
    }
  }
  prior_list
}

# Per-block axis grid as a matrix `[n_block_cells x n_axes_for_block]`.
# Single-axis types (icar / rw1 / rw2 / iid) get a 1-column matrix; 2-axis
# types (bym2 / car_proper / ar1) get a 2-column matrix.
.nl_block_axis_grid <- function(p) {
  spec <- .NL_REGISTRY[[tolower(p$type)]]
  if (is.null(spec)) {
    stop("Unknown block type: ", p$type, call. = FALSE)
  }
  .nl_check_block_fields(p, "axis")
  p <- spec$defaults(p, list())
  th <- spec$theta(p)
  if (is.matrix(th$grid)) {
    grid <- th$grid
    if (is.null(colnames(grid))) colnames(grid) <- th$names
  } else {
    grid <- matrix(as.numeric(th$grid), ncol = 1)
    colnames(grid) <- th$names
  }
  list(grid = grid, names = colnames(grid), prepared = p)
}

# Cell count above which a dense multi-block outer tensor is announced with a
# warning BEFORE its solve starts. Each cell is one inner Laplace solve, so the
# count is the run's size in solves; it does not know the per-solve cost, which
# is why it warns rather than refuses. The warning is a function of the grid
# alone, so the same call warns or stays silent on every machine and every run.
.NL_DENSE_GRID_WARN_CELLS <- 2048L

# Shared by both multi-block dispatchers: the dense tensor about to be solved
# has `n_cells` inner solves, `layout` (`.nl_grid_crossing()`) names the axes
# that produced them and `remedy` is the dispatcher's advice. An internal batch
# the caller did not size (`.nl_internal_batch()`) stays quiet.
.nl_dense_grid_warn <- function(n_cells, remedy, layout = "") {
  if (isTRUE(n_cells > .NL_DENSE_GRID_WARN_CELLS) && !.nl_internal_batch()) {
    warning(sprintf(
      paste0("Multi-block outer grid is a dense tensor of %s cells, one inner ",
             "solve each.%s %s"),
      format(n_cells, scientific = FALSE, trim = TRUE), layout, remedy
    ), call. = FALSE)
  }
  invisible(NULL)
}

# Whether the dispatch currently running is an INTERNAL batch rather than the
# grid the caller asked for. The dense-grid warning above is advice addressed to
# whoever chose the cell count, and on an internal batch that is not the caller:
# the outer Pareto-k diagnostic re-evaluates `log_marginal` at `k_samples`
# importance draws by substituting them for the block's grid axis, so a 7-node
# fit warned about a cell count the diagnostic's own budget set and advised
# reducing per-block grid sizes, which does not reach the number in the message
# (gcol33/tulpa#614).
#
# Transported on an option rather than an argument, the same reason
# `tulpa.nl_progress` and `tulpa.nl_checkpoint` are options: the warning sits
# behind dispatchers whose signatures do not carry it.
.nl_internal_batch <- function() {
  isTRUE(getOption("tulpa.nl_internal_batch", FALSE))
}

# Run `expr` with that flag set. Nests and restores, so an internal batch that
# itself re-dispatches stays internal.
.nl_with_internal_batch <- function(expr) {
  op <- options(tulpa.nl_internal_batch = TRUE)
  on.exit(options(op), add = TRUE)
  force(expr)
}

# The caller's ceiling on a dense multi-block outer tensor, in inner solves.
# Unset there is none: a cell count does not measure the run's cost, so the
# default is the pre-solve warning above, and a caller who wants a hard stop
# (a batch of fits on a shared node, say) sets `control$max_grid_cells`. The
# front doors publish the resolved value on the scoped
# `tulpa.nl_max_grid_cells` option, so every grid built inside the fit -- the
# multi-block dispatch, the joint multi-block dispatch, the k-hat
# re-evaluations -- reads one ceiling without carrying it in each signature,
# the transport `tulpa.nl_progress` and `tulpa.nl_checkpoint` already use.
.nl_max_grid_cells <- function(control = list()) {
  v <- control$max_grid_cells
  if (is.null(v)) v <- getOption("tulpa.nl_max_grid_cells", NULL)
  if (is.null(v)) return(Inf)
  if (!is.numeric(v) || length(v) != 1L || is.na(v) || v < 1) {
    stop("`control$max_grid_cells` must be a single number >= 1 ",
         "(the cell-count ceiling on a multi-block outer grid).", call. = FALSE)
  }
  as.numeric(v)
}

# The single enforcement of that ceiling, shared by both multi-block
# dispatchers, before any inner solve. `layout` is `.nl_grid_crossing()`'s
# account of the axes that produced the count and `remedy` the dispatcher's
# advice.
.nl_check_grid_cap <- function(n_cells, max_cells, remedy, layout = "") {
  if (n_cells <= max_cells) return(invisible(n_cells))
  fmt <- function(x) format(x, scientific = FALSE, trim = TRUE)
  stop(sprintf(
    paste0("Multi-block outer grid has %s cells, past control$max_grid_cells ",
           "= %s.%s The cap is read on the grid each pass is about to solve, ",
           "the declared grid first; a placement or refinement pass that runs ",
           "afterwards can re-lay the axes, so a fitted grid can hold a ",
           "different number of cells. %s"),
    fmt(n_cells), fmt(max_cells), layout, remedy
  ), call. = FALSE)
}

# The axes a dense outer tensor crosses, as the sentence the timing warning
# carries: " It crosses b1 (...) x phi_pos 4." `block_grids` are the per-block
# axis grids, so a fit that set no grid learns which default axes produced its
# count (gcol33/tulpa#913); `phi_axes` are the active named per-arm dispersion
# axes crossed on top, each a factor on the number of inner solves.
.nl_grid_crossing <- function(block_grids = NULL, phi_axes = NULL) {
  parts <- c(if (length(block_grids)) .nl_grid_layout(block_grids),
             if (length(phi_axes))
               paste0("phi_", names(phi_axes), " ", lengths(phi_axes)))
  if (length(parts)) paste0(" It crosses ", paste(parts, collapse = " x "), ".")
  else ""
}

# One block's grid as "b<k> (<rows> rows: <axis> <levels> x ...)", the levels
# being each axis's distinct node count. A block grid is a tensor of its axes
# for the grid-axis families and a paired node list for others (BYM2's
# (sigma, rho) nodes), so the row count is stated alongside the per-axis
# levels rather than derived from them.
.nl_grid_layout <- function(block_grids) {
  parts <- vapply(seq_along(block_grids), function(b) {
    g <- as.matrix(block_grids[[b]])
    nm <- colnames(g) %||% paste0("axis", seq_len(ncol(g)))
    lv <- vapply(seq_len(ncol(g)), function(j)
      length(unique(as.numeric(g[, j]))), integer(1))
    sprintf("b%d (%d rows: %s)", b, nrow(g),
            paste(nm, lv, collapse = " x "))
  }, character(1))
  paste(parts, collapse = " x ")
}

# The registry arm's inner solve at given cells: `theta` is a matrix whose rows
# are outer cells, its columns the blocks' axes in block order. A conditional
# fit (one cell at fixed hyperparameters), a cell solved again for its
# precision, and a custom block's log marginal at a set of points read this; a
# grid fit is the joint door's. `compute_skew` / `skew_idx` request the
# inner-skew terms at the cells, `debias` / `cila` the kernel-facing correction
# requests; `force_sparse` takes the joint sparse Newton whatever the latent
# size (a dense == sparse check).
.nl_solve_cells <- function(cargs, prior_list, theta, likelihood = NULL,
                            compute_skew = FALSE, skew_idx = NULL,
                            debias = NULL, cila = NULL, force_sparse = FALSE) {
  prior_list <- .nl_tgmrf_default_obs_idx(prior_list, length(cargs$y))

  per_block <- lapply(prior_list, .nl_block_axis_grid)
  prepared  <- lapply(per_block, function(x) x$prepared)
  block_grids <- lapply(per_block, function(x) x$grid)

  axis_counts  <- vapply(block_grids, ncol, integer(1))
  axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
  axis_names <- unlist(lapply(seq_along(block_grids), function(b)
    .axis_label(b, colnames(block_grids[[b]]))))

  joint_grid <- as.matrix(theta)
  if (ncol(joint_grid) != axis_offsets[length(axis_offsets)]) {
    stop("`theta` must have ", axis_offsets[length(axis_offsets)],
         " columns (sum of block axes), got ", ncol(joint_grid), ".",
         call. = FALSE)
  }
  colnames(joint_grid) <- axis_names

  arm <- .nl_registry_arm(cargs, likelihood)
  blocks_spec <- .joint_tgmrf_at_cells(
    .joint_blocks_spec_for_cpp(
      lapply(seq_along(prepared), function(b) .nl_block_as_arm(prepared[[b]], b)),
      1L, arms = list(arm)),
    prepared, joint_grid, axis_offsets)

  # Every block's hyperprior over its own columns, so a stencil re-evaluating
  # the marginal differences the target the grid integrates. The integrated
  # columns are read off the block's declared grid, never off `theta`, whose
  # rows hold a fixed axis at other values or an integrated one at a single
  # value (gcol33/tulpa#760).
  hp_parts <- lapply(seq_along(prepared), function(b) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    tg_b <- joint_grid[, cols, drop = FALSE]
    colnames(tg_b) <- colnames(block_grids[[b]])
    blk <- prepared[[b]]
    blk$log_prior_theta_per_grid <- blocks_spec[[b]]$log_prior_theta_per_grid
    .hp_prefix(.nl_block_log_hyperprior(blk, tg_b, .hp_choice(cargs$hyperprior),
                                        declared = .hp_declare(block_grids[[b]])),
               .axis_prefix(b))
  })
  progress <- .nl_progress_args(list(progress = FALSE))

  out <- cpp_nested_laplace_joint_multi(
    arms_list   = list(arm),
    copy_arms   = integer(0),
    copy_blocks = integer(0),
    blocks_spec = blocks_spec,
    theta_grid  = joint_grid,
    axis_offsets = axis_offsets,
    max_iter    = cargs$max_iter,
    tol         = cargs$tol,
    n_threads   = cargs$n_threads,
    x_init_nullable = cargs$x_init_nullable,
    store_Q     = isTRUE(cargs$store_Q),
    prune_tol   = 0,
    force_sparse = isTRUE(force_sparse),
    progress          = isTRUE(progress$progress),
    progress_every    = as.integer(progress$progress_every),
    progress_throttle = as.numeric(progress$progress_throttle),
    progress_file     = as.character(progress$progress_file),
    checkpoint_path   = "",
    compute_skew      = isTRUE(compute_skew),
    skew_idx          = skew_idx,
    debias            = debias,
    cila              = cila,
    screen_log_offset = NULL,
    compute_fitted_var = isTRUE(cargs$compute_fitted_var %||% TRUE),
    n_threads_outer = as.integer(cargs$n_threads_outer %||% 1L)
  )

  out$theta_grid   <- joint_grid
  out$theta_names  <- axis_names
  out$axis_offsets <- axis_offsets
  out$blocks       <- prepared
  .nl_fold_hyperprior(out, hp_parts)
}

#' Build a `prior` list for [tulpa_nested_laplace()] from a tulpa spec object
#'
#' @description
#' Validates a `tulpa_temporal` or `tulpa_spatial` specification against
#' `data`, then converts it to the prior list shape consumed by
#' [tulpa_nested_laplace()]. Mainly an internal helper for callers that already
#' have a fitted spec; users typically pass `spec` + `data` directly to
#' `tulpa_nested_laplace()` instead.
#'
#' Supported spec types:
#'  * `tulpa_temporal` with `type in {"rw1", "rw2", "ar1"}`
#'  * `tulpa_spatial`, areal `type in {"car", "icar", "car_proper", "bym2"}`.
#'    An areal spec returns the graph fields only -- `type`, `spatial_idx`,
#'    `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`, `n_neighbors` -- plus
#'    `scale_factor` for bym2 and the eigenvalue-derived `rho_bounds` for
#'    proper CAR. No grid field is returned for any areal type: the outer axes
#'    are built later by the family's registry `defaults()` closure, and proper
#'    CAR's `(tau, rho)` grid is built there from those bounds.
#'  * `tulpa_gp` / `tulpa_hsgp`, continuous `type in {"gp", "nngp", "hsgp"}`:
#'    validated against `data` and routed to the `nngp` / `hsgp` nested kernel
#'    through the shared converter.
#'
#'  SPDE is the one continuous field not built here -- it carries its own
#'  (range, sigma) FEM integrator; call [fit_spde()] with the `spatial_spde()`
#'  spec.
#'
#' @param spec A `tulpa_temporal` or `tulpa_spatial` object.
#' @param data Data frame the spec resolves time/group/site indices against.
#' @return A `prior` list ready for [tulpa_nested_laplace()].
#' @examples
#' d <- data.frame(year = rep(1:10, each = 3))
#' pr <- prior_from_spec(temporal_ar1("year"), d)
#' str(pr)
#' @export
prior_from_spec <- function(spec, data) {
  if (inherits(spec, "tulpa_temporal")) {
    return(.prior_from_temporal_spec(spec, data))
  }
  if (inherits(spec, "tulpa_spatial")) {
    return(.prior_from_spatial_spec(spec, data))
  }
  stop("`spec` must inherit from 'tulpa_temporal' or 'tulpa_spatial'.",
       call. = FALSE)
}

.prior_from_temporal_spec <- function(spec, data) {
  spec <- validate_temporal(spec, data)
  type <- tolower(spec$type)
  if (!type %in% c("rw1", "rw2", "ar1")) {
    stop("tulpa_nested_laplace() supports temporal types rw1, rw2, ar1; got '",
         type, "'.", call. = FALSE)
  }
  out <- list(
    type = type,
    temporal_idx = as.integer(spec$time_index),
    n_times = as.integer(spec$n_times)
  )
  if (type %in% c("rw1", "rw2")) out$cyclic <- isTRUE(spec$cyclic)
  out
}

.prior_from_spatial_spec <- function(spec, data) {
  type <- tolower(spec$type)

  # Continuous coordinate-addressed fields (gp/nngp/hsgp). Validate to populate
  # the spec's resolved coordinate structure (NNGP neighbour info / HSGP basis),
  # then build the nested prior block through the shared converter
  # `.spatial_spec_to_nl_prior` -- the single source of truth for spec -> nested
  # prior, also used by the tulpa() front door. gp/nngp become an `nngp` block,
  # hsgp an `hsgp` block.
  if (type %in% c("gp", "nngp")) {
    return(.spatial_spec_to_nl_prior(validate_gp(spec, data)))
  }
  if (type == "hsgp") {
    return(.spatial_spec_to_nl_prior(validate_hsgp(spec, data)))
  }
  if (type == "spde") {
    stop("tulpa_nested_laplace() does not host an SPDE field: SPDE carries its ",
         "own (range, sigma) FEM integrator. Call fit_spde() with the ",
         "spatial_spde() spec.", call. = FALSE)
  }

  if (!type %in% .NL_FRONTDOOR_AREAL) {
    stop("Unknown spatial type '", type, "'. Areal: ",
         paste(.NL_FRONTDOOR_AREAL, collapse = "/"),
         "; continuous: gp/nngp/hsgp; SPDE via fit_spde().", call. = FALSE)
  }
  validate_spatial(spec, data)

  # Areal field: resolve the per-observation spatial_idx (1-based row indices
  # into the adjacency) onto the spec, then build the block through the shared
  # converter. Empty cells (no data) are allowed -- the ICAR / CAR / BYM2 prior
  # is well-defined on every graph node; unobserved cells contribute no
  # likelihood term. See `.resolve_spatial_idx()` for the resolution rules.
  adj <- spec$adjacency
  n_spatial_units <- nrow(as.matrix(adj))
  if (!is.null(spec$level) && spec$level == "group" &&
      !is.null(spec$group_var)) {
    if (!(spec$group_var %in% names(data))) {
      stop("Spatial group variable '", spec$group_var,
           "' not found in data.", call. = FALSE)
    }
    spec$spatial_idx <- .resolve_spatial_idx(
      values = data[[spec$group_var]],
      n_spatial_units = n_spatial_units,
      adjacency = adj,
      group_var = spec$group_var
    )
  } else {
    if (nrow(data) != n_spatial_units) {
      stop("Observation-level spatial spec requires nrow(data) == ",
           "nrow(adjacency).", call. = FALSE)
    }
    spec$spatial_idx <- seq_len(n_spatial_units)
  }
  .spatial_spec_to_nl_prior(spec)
}

# Normalize the outer-grid progress knobs from a `control` list into the four
# scalars the C++ entry points accept. On by default; `progress = FALSE` turns
# off the flushed cell-k/n_grid + ETA reporter (see the GridProgress reporter in
# inst/include/tulpa/nested_progress.h). `progress.file`
# adds a heartbeat file for detached runs where Rcout flushing is unreliable.
# The inner refinement / EM / CCD-probe call sites pass `progress = FALSE`
# explicitly so only the top-level fit ticks.
.nl_progress_args <- function(control) {
  # Control keys win; otherwise fall back to the scoped `tulpa.nl_progress`
  # option a caller (e.g. tulpaObs) may have set for a whole fit, so progress
  # reaches grids run through fixed-control internal paths (the EM per-block
  # nested solve) without threading the knobs through every layer.
  # `[[` (exact) not `$` on `progress`: it is a strict prefix of the other three
  # keys, so a `$` read returns whichever ONE of them is set (and NULL when two
  # are, the match being ambiguous). `isTRUE()` of a file path or a cadence is
  # FALSE, which turned asking for the heartbeat file into switching the console
  # bar off.
  has_ctrl <- !is.null(control[["progress"]]) || !is.null(control$progress.every) ||
              !is.null(control$progress.throttle) || !is.null(control$progress.file)
  if (!has_ctrl) {
    opt <- getOption("tulpa.nl_progress", NULL)
    if (is.list(opt)) return(opt)
  }
  list(
    progress          = isTRUE(control[["progress"]] %||% TRUE),
    progress_every    = as.integer(control$progress.every    %||% 0L),
    progress_throttle = as.numeric(control$progress.throttle %||% 2),
    progress_file     = as.character(control$progress.file    %||% "")
  )
}

# Resolve a checkpoint spec to a normalized `list(path, resume)`.
# `control$checkpoint = list(path = , resume = TRUE)` enables it; `resume =
# FALSE` starts over (the caller removes any prior file before the first
# kernel call). Like `.nl_progress_args`, an absent control key falls back to
# the scoped `tulpa.nl_checkpoint` option (when `use_option`, the default) so
# the checkpoint reaches grids run through fixed-control internal paths;
# `tulpa_sample_glmm()`'s per-chain checkpoint is a distinct namespace from
# the nested-Laplace grid-cell one this option serves, so it reads
# `use_option = FALSE` and ignores it.
.nl_checkpoint_args <- function(control, use_option = TRUE) {
  cp <- control$checkpoint
  if (is.null(cp)) {
    if (use_option) {
      opt <- getOption("tulpa.nl_checkpoint", NULL)
      if (is.list(opt)) return(opt)
    }
    return(list(path = "", resume = TRUE))
  }
  if (!is.list(cp) || is.null(cp$path) || !is.character(cp$path) ||
      length(cp$path) != 1L || is.na(cp$path) || !nzchar(cp$path)) {
    stop("`control$checkpoint` must be a list with a single non-empty ",
         "character `path` (e.g. ",
         "list(path = \"fit.ckpt\", resume = TRUE)).", call. = FALSE)
  }
  resume <- cp$resume %||% TRUE
  if (!is.logical(resume) || length(resume) != 1L || is.na(resume)) {
    stop("`control$checkpoint$resume` must be a single TRUE or FALSE.",
         call. = FALSE)
  }
  list(path = path.expand(cp$path), resume = isTRUE(resume))
}

# Outer-grid cells read back from the checkpoint over the process; a door
# reports the difference across its own run, which spans every kernel call the
# fit makes on the path (first grid, recentred refit, refinement rounds).
.nl_checkpoint_loaded <- function() cpp_nl_checkpoint_loaded_total()

# Stamp `checkpoint_n_loaded` on a fit run with a checkpoint: the number of
# grid cells this fit took from the file rather than solving. A fresh run
# reads 0; a full resume reads every cell its grids carried.
.nl_attach_checkpoint_loaded <- function(res, ckpt, loaded0) {
  if (!is.list(res) || !nzchar(ckpt$path)) return(res)
  res$checkpoint_n_loaded <- as.integer(round(.nl_checkpoint_loaded() - loaded0))
  res
}
