#' Nested Laplace approximation for latent Gaussian models
#'
#' @description
#' Generic outer-grid nested Laplace driver. Builds a grid over the
#' hyperparameters of a single latent prior block (spatial or temporal),
#' runs an inner Laplace at each grid point with warm-starting, and
#' integrates over the grid to give proper hyperparameter marginals.
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
#'           `n_neighbors` (CSR adjacency, 0-based); optional `tau_grid`;
#'           optional `svc_weight` (one weight per observation) to make it a
#'           spatially-varying coefficient whose eta contribution is
#'           `svc_weight[i] * z[spatial_idx[i]]` rather than `z[spatial_idx[i]]`.
#'   * bym2:  same adjacency; `scale_factor`; optional `sigma_grid`, `rho_grid`.
#'   * car_proper: same adjacency; optional `tau_grid`, `rho_grid`,
#'           `rho_bounds = c(lower, upper)` (defaults to (0, 1)).
#'   * rw1/rw2: `temporal_idx` (1-based), `n_times`; optional `tau_grid`,
#'             `cyclic` (default FALSE).
#'   * ar1:   `temporal_idx`, `n_times`; optional `tau_grid`, `rho_grid`.
#' @param re_idx Optional 1-based RE group index per obs (defaults to no RE).
#' @param n_re_groups RE group count (default 0).
#' @param sigma_re RE standard deviation (default 1).
#' @param family `"binomial"`, `"poisson"`, `"neg_binomial_2"`, etc.
#' @param phi Dispersion (negbin/gamma).
#' @param control Optional list of perf/numerical tuning knobs (statistical
#'   arguments stay top-level), following the `control` convention of
#'   [tulpa()]. Recognised elements (defaults in parentheses):
#'   * `max_iter` (`50L`), `tol` (`1e-6`) -- inner Newton iteration budget and
#'     tolerance.
#'   * `n_threads` (`1L`) -- inner-loop OpenMP threads.
#'   * `x_init` (`NULL`) -- warm-start for the first grid point's inner solve.
#'   * `keep_grid_hessians` (`FALSE`) -- when `TRUE`, retain per-grid-point
#'     fixed-effects marginal Hessian \eqn{H_\beta} and mode \eqn{\hat{\beta}}
#'     on the return list as `$grid_hessians` (list of dense \eqn{p\times p}
#'     matrices) and `$grid_modes` (list of length-\eqn{p} vectors). Used
#'     downstream by simplified-Laplace (SLA) callers to assemble skew-aware
#'     marginals -- see the cumulant pooling in [rubins_pool()].
#'   * `diagnose_k` (`TRUE`), `k_samples` (`200L`) -- compute the outer
#'     Pareto-\eqn{\hat{k}} accuracy diagnostic (`$pareto_k`) by importance
#'     sampling the hyperparameter posterior against the Gaussian proposal
#'     fitted to the grid, drawing `k_samples` extra inner-marginal evaluations.
#'     Computed for a single-block, single positive-scale-axis grid; left `NA`
#'     (with the grid's quadrature ESS as the fallback diagnostic) for
#'     multi-block, multi-axis, or bounded-parameter grids. See [tulpa_psis()].
#'
#' @return A list with:
#'   * `theta_grid`: matrix or vector of grid hyperparameter values.
#'   * `log_marginal`: log p(y, mode | theta_k) at each grid point.
#'   * `weights`: integration weights normalising to sum 1.
#'   * `theta_mean`, `theta_sd`: posterior moments per hyperparameter.
#'   * `n_iter`: inner Newton iterations per grid point.
#'   * `modes`: matrix `[n_grid x n_x]` of inner modes, when stored.
#'   * `pareto_k`, `pareto_k_is_ess`: outer Pareto-\eqn{\hat{k}} and its
#'     importance-sampling ESS (`NA` when not computed for the grid; see
#'     `control$diagnose_k`).
#'   * `timing`: named numeric of wall-clock seconds (`total`, `setup`,
#'     `grid`, `postproc`, `diagnostics`); the `grid` phase is the inner
#'     Laplace pass that scales with grid size. Surfaced one-line in `print`.
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
#'   out) through this. Multi-block `prior` only. Default `NULL` (use `family`).
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
                            family = "binomial", phi = 1.0,
                            likelihood = NULL,
                            control = list()) {

  .check_control(control, .CONTROL_KEYS$nested_laplace, "tulpa_nested_laplace")
  tm <- .tulpa_timer()

  # Perf/numerical knobs live in `control = list()` (matching tulpa()); the
  # top-level signature carries only statistical arguments.
  max_iter           <- control$max_iter %||% 50L
  tol                <- control$tol %||% 1e-6
  n_threads          <- control$n_threads %||% 1L
  x_init             <- control$x_init
  keep_grid_hessians <- isTRUE(control$keep_grid_hessians)
  diagnose_k         <- isTRUE(control$diagnose_k %||% TRUE)
  k_samples          <- as.integer(control$k_samples %||% 200L)

  # Grid-cell checkpoint/resume. `control$checkpoint =
  # list(path =, resume =)` makes every grid cell append to `path`; a resume
  # loads the finished cells and solves only the rest. A fresh (resume = FALSE)
  # run removes any stale file once here, before the first kernel call, so the
  # several within-fit kernel calls (initial grid, adaptive refinement) all
  # append rather than truncate each other. The path threads into `cargs` and
  # reaches both the single-block (.nl_dispatch) and multi-block
  # (.nl_dispatch_multi) kernels by name; the k-hat diagnostic re-evaluations
  # are run with it stripped so they do not pollute the file.
  .ckpt <- .nl_checkpoint_args(control)
  if (nzchar(.ckpt$path) && !isTRUE(.ckpt$resume) && file.exists(.ckpt$path)) {
    file.remove(.ckpt$path)
  }

  if (!is.null(spec)) {
    if (!is.null(prior)) {
      stop("Pass either `spec` or `prior`, not both.", call. = FALSE)
    }
    if (is.null(data)) {
      stop("`data` is required when `spec` is supplied.", call. = FALSE)
    }
    prior <- prior_from_spec(spec, data)
  }
  # Allow passing a single tgmrf S3 object directly: wrap it in a length-1
  # block list so the multi-block dispatch picks it up. This keeps the
  # ergonomic API close to the formula sketch (one block,
  # one call) while still routing through the validated multi-block path.
  if (inherits(prior, "tulpa_latent_block")) {
    prior <- list(prior)
  }
  if (!is.list(prior)) {
    stop("`prior` must be a list (single block) or list-of-lists (multi-block).",
         call. = FALSE)
  }
  # A model-supplied `likelihood` is only honoured by the spec-driven multi-block
  # driver (run_multi_block_nested_laplace); the single-kernel path (.nl_dispatch)
  # has no spec hook. Route a single LatentBlock-type prior through the
  # multi-block driver by wrapping it as a length-1 block list -- the per-cell
  # math is identical (every single-block kernel already calls the same driver).
  if (!is.null(likelihood) && !is.null(prior$type)) {
    prior <- list(prior)
  }
  vd <- .validate_glm_design(y, X, n_trials, "tulpa_nested_laplace")
  N  <- vd$N
  n_trials <- vd$n_trials
  if (is.null(re_idx)) re_idx <- rep(0L, N)
  re_idx <- .validate_re_idx(re_idx, n_re_groups, N, "tulpa_nested_laplace")

  cargs <- list(
    y = as.numeric(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups),
    sigma_re = as.numeric(sigma_re),
    family = family,
    phi = as.numeric(phi),
    max_iter = as.integer(max_iter),
    tol = as.numeric(tol),
    n_threads = as.integer(n_threads),
    x_init_nullable = x_init,
    store_Q = isTRUE(keep_grid_hessians),
    checkpoint_path = .ckpt$path
  )

  # cargs without the checkpoint, for the k-hat diagnostic re-evaluations.
  cargs_no_ckpt <- utils::modifyList(cargs, list(checkpoint_path = ""))

  p_fixed <- ncol(X)

  if (.is_multi_block_prior(prior)) {
    tm$mark("setup")
    res <- .nl_dispatch_multi(cargs, prior, likelihood = likelihood,
                              progress = .nl_progress_args(control))
    tm$mark("grid")
    if (isTRUE(keep_grid_hessians)) {
      res <- .nl_attach_grid_hessians(res, p_fixed)
    }
    tm$mark("postproc")
    res <- .nl_attach_pareto_k(res, prior, cargs_no_ckpt, "multi", NULL,
                               likelihood, k_samples, compute = diagnose_k)
    tm$mark("diagnostics")
    res$prior <- prior
    res$timing <- tm$timing()
    return(.finalize_fit(res, backend = "nested_laplace",
                         n_fixed = p_fixed, fixed_names = colnames(X),
                         extra_class = c("tulpa_nested_laplace", "list")))
  }

  if (!is.null(likelihood)) {
    stop("`likelihood` (model-supplied spec) needs a latent prior block. ",
         "Pass `prior` (a block or list of blocks).", call. = FALSE)
  }

  if (is.null(prior$type)) {
    stop("Single-block `prior` must have a `type` field, ",
         "or pass a list of blocks for multi-block.", call. = FALSE)
  }
  type <- tolower(prior$type)
  tm$mark("setup")
  res <- .nl_dispatch(type, cargs, prior)
  res <- .nl_apply_ar1_rho_prior(res, type, prior)
  tm$mark("grid")

  # Integrate exp(log_marginal) over the outer grid. `log_marginal` is the
  # inner marginal in the internal (log-scale) hyperparameter parameterization
  # the grid is laid out on, so the grid nodes are equally-weighted quadrature
  # points and no user-scale volume element is applied -- adding one biases the
  # scale-hyperparameter posterior (confirmed by CAR_proper (tau, rho) recovery).
  res$weights <- .nl_normalise_weights_safe(res$log_marginal, "outer grid")
  res <- .nl_posterior_moments(res, type)
  if (isTRUE(keep_grid_hessians)) {
    res <- .nl_attach_grid_hessians(res, p_fixed)
  }
  tm$mark("postproc")
  res <- .nl_attach_pareto_k(res, prior, cargs_no_ckpt, "single", type, NULL,
                             k_samples, compute = diagnose_k)
  tm$mark("diagnostics")
  res$prior <- prior
  res$timing <- tm$timing()
  .finalize_fit(res, backend = "nested_laplace",
                n_fixed = p_fixed, fixed_names = colnames(X),
                extra_class = c("tulpa_nested_laplace", "list"))
}

# Positive-scale hyperparameter grid fields: one log-transform unconstrains
# them, so the outer Pareto-k-hat (`.nested_grid_pareto_k`) is well-defined and
# needs no bounded-parameter Jacobian. A block carrying a bounded axis (e.g. a
# correlation `rho_grid`) is absent from this set and is declined to the
# quadrature-ESS fallback rather than transformed with a guessed support.
.NL_POS_GRID <- c("sigma_grid", "sigma2_grid", "tau_grid", "phi_gp_grid",
                  "phi_grid", "range_grid", "scale_grid", "sigma_spatial_grid",
                  "lambda_grid", "kappa_grid")

# Attach res$pareto_k for a single-block, single positive-scale-axis nested fit.
# Leaves res$pareto_k = NA (the diagnostic layer then reports the grid's
# quadrature ESS) for multi-block, multi-axis, or bounded-parameter grids.
# `refit` re-evaluates the inner marginal at a substituted grid through the SAME
# driver `res` came from; run with the RNG restored so the fit is unperturbed.
.nl_attach_pareto_k <- function(res, prior, cargs, dispatch_kind, type,
                                likelihood, n_samples, compute = TRUE) {
  res$pareto_k        <- NA_real_
  res$pareto_k_is_ess <- NA_real_
  res$pareto_k_scope  <- "outer (hyperparameter) Gaussian proposal"
  if (!compute) return(res)                                  # field present, not computed

  blocks <- if (is.list(prior) && is.null(prior$type)) prior else list(prior)
  if (length(blocks) != 1L) return(res)                      # multi-block: decline
  blk <- blocks[[1L]]
  # The multi-block dispatch does not carry a `type`, so resolve it from the block
  # itself for the length-1 case (a single latent block wrapped in a list, or the
  # model-supplied `likelihood` path). Without this the registry lookup below is
  # `.NL_REGISTRY[[NULL]]`, an error rather than NULL.
  type <- type %||% tolower(blk$type)
  # A fit that relied on the default grid carries no `*_grid` field on the input
  # prior, but the realised res$theta_grid is a valid single positive-scale axis
  # regardless. Fill the type's defaults (the same ones .nl_dispatch applied) so
  # eligibility keys on the actual grid, not on whether the user named it. A
  # multi-axis type then exposes >1 grid field and still declines below.
  spec <- .NL_REGISTRY[[type]]
  if (!is.null(spec) && is.function(spec$defaults)) {
    blk <- tryCatch(spec$defaults(blk, cargs), error = function(e) blk)
  }
  gfs <- grep("_grid$", names(blk), value = TRUE)
  gfs <- gfs[vapply(gfs, function(f) is.numeric(blk[[f]]) && length(blk[[f]]) >= 2L,
                    logical(1))]
  if (length(gfs) != 1L || !(gfs %in% .NL_POS_GRID)) return(res)  # multi-axis / bounded
  tg <- as.numeric(res$theta_grid)
  if (length(tg) != length(res$weights) || any(tg <= 0)) return(res)

  refit <- function(theta_mat) {
    blk2 <- blk; blk2[[gfs]] <- as.numeric(theta_mat[, 1L])
    prior2 <- if (is.list(prior) && is.null(prior$type)) list(blk2) else blk2
    lm <- tryCatch(
      if (dispatch_kind == "multi")
        .nl_dispatch_multi(cargs, prior2, likelihood = likelihood)$log_marginal
      else .nl_dispatch(type, cargs, prior2)$log_marginal,
      error = function(e) rep(-Inf, nrow(theta_mat)))
    if (length(lm) != nrow(theta_mat)) rep(-Inf, nrow(theta_mat)) else lm
  }

  kd <- .with_preserved_seed(tryCatch(
    .nested_grid_pareto_k(matrix(log(tg), ncol = 1L), res$weights, refit, n_samples),
    error = function(e) NULL))
  if (!is.null(kd)) { res$pareto_k <- kd$pareto_k; res$pareto_k_is_ess <- kd$is_ess }
  res
}

# --- Per-prior registry & dispatcher ---
#
# Each entry describes one supported prior `type`. To add a new prior:
#   1. Add a row here with cpp_fn, defaults, pack, theta.
#   2. Make sure cpp_nested_laplace_<variant> exists in RcppExports.
# No changes needed to .nl_dispatch() or the public driver.

# Areal CSR adjacency block -- shared by icar/bym2/car_proper.
.nl_adj_args <- function(p) list(
  spatial_idx     = as.integer(p$spatial_idx),
  n_spatial_units = as.integer(p$n_spatial_units),
  adj_row_ptr     = as.integer(p$adj_row_ptr),
  adj_col_idx     = as.integer(p$adj_col_idx),
  n_neighbors     = as.integer(p$n_neighbors)
)

# Default outer grid for a p-field separable MCAR, returned in column-major
# lower-triangle log-Cholesky coordinates of Sigma = L L' (the columns the
# integrator reads). For the common p = 2 case the nodes are placed in the
# interpretable (sigma_1, sigma_2, rho) space -- with nodes AT sensible
# correlation values -- and converted to log-Cholesky, so the rho axis is
# covered evenly (raw log-Cholesky off-diagonal nodes map to awkward, uneven
# correlations and can miss the mode under sharp likelihoods). For p > 2 a raw
# log-Cholesky tensor is used (coarser; the off-diagonal correlations are not
# separately interpretable).
.mcar_default_logchol_grid <- function(p) {
  if (p == 2L) {
    sig <- c(0.4, 0.7, 1.1, 1.7)
    rho <- c(-0.8, -0.4, 0, 0.4, 0.7, 0.9)
    g <- expand.grid(s1 = sig, s2 = sig, rho = rho,
                     KEEP.OUT.ATTRS = FALSE)
    # Sigma = [[s1^2, rho s1 s2], [rho s1 s2, s2^2]] -> lower Cholesky L:
    # L11 = s1, L21 = rho s2, L22 = s2 sqrt(1 - rho^2).
    out <- cbind(L11 = log(g$s1),
                 L21 = g$rho * g$s2,
                 L22 = log(g$s2 * sqrt(1 - g$rho^2)))
    colnames(out) <- c("L11", "L21", "L22")
    return(out)
  }
  diag_nodes <- log(c(0.4, 0.8, 1.5, 2.5))
  off_nodes  <- c(-1.2, 0.0, 1.2)
  m <- p * (p + 1L) / 2L
  axes <- vector("list", m)
  nm <- character(m)
  t <- 1L
  for (j in seq_len(p)) for (i in j:p) {
    axes[[t]] <- if (i == j) diag_nodes else off_nodes
    nm[t] <- sprintf("L%d%d", i, j)
    t <- t + 1L
  }
  g <- as.matrix(do.call(expand.grid, c(axes, list(KEEP.OUT.ATTRS = FALSE))))
  colnames(g) <- nm
  g
}

.NL_REGISTRY <- list(
  icar = list(
    cpp_fn = "cpp_nested_laplace_icar",
    defaults = function(p, a) {
      if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid()
      p
    },
    pack = function(p) c(.nl_adj_args(p), list(
      tau_grid = as.numeric(p$tau_grid)
    )),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  bym2 = list(
    cpp_fn = "cpp_nested_laplace_bym2",
    defaults = function(p, a) {
      if (is.null(p$sigma_grid) || is.null(p$rho_grid)) {
        sg <- exp(seq(log(0.1), log(3), length.out = 5))
        rg <- c(0.2, 0.5, 0.8, 0.95)
        gr <- expand.grid(sigma = sg, rho = rg)
        p$sigma_grid <- gr$sigma; p$rho_grid <- gr$rho
      }
      p
    },
    pack = function(p) c(.nl_adj_args(p), list(
      scale_factor       = as.numeric(p$scale_factor %||% 1.0),
      sigma_spatial_grid = as.numeric(p$sigma_grid),
      rho_grid           = as.numeric(p$rho_grid)
    )),
    theta = function(p) list(
      grid  = cbind(sigma = p$sigma_grid, rho = p$rho_grid),
      names = c("sigma", "rho")
    )
  ),

  car_proper = list(
    cpp_fn = "cpp_nested_laplace_car_proper",
    defaults = function(p, a) {
      # `rho_car_grid` is the joint-API spelling of the correlation axis; accept
      # it as an alias so a grid passed under either name is honoured.
      p$rho_grid <- p$rho_grid %||% p$rho_car_grid
      # Both axes supplied -> treat as the pre-paired integration grid (the
      # theta builder pairs them with cbind), matching bym2 / ar1.
      if (!is.null(p$tau_grid) && !is.null(p$rho_grid)) return(p)
      # Otherwise default each missing axis independently and cross the two, so
      # supplying only one axis keeps it instead of discarding it.
      rb <- p$rho_bounds %||% c(0, 1)
      # Margin in from the eigenvalue endpoints -- Q goes singular at the
      # boundary, so anchoring the grid in (lower + eps, upper - eps) avoids
      # NaN log-determinants from the per-grid-point Cholesky.
      eps <- 0.05 * (rb[2] - rb[1])
      g_tau <- if (is.null(p$tau_grid))
        exp(seq(log(0.3), log(30), length.out = 5)) else sort(unique(p$tau_grid))
      g_rho <- if (is.null(p$rho_grid))
        seq(rb[1] + eps, rb[2] - eps, length.out = 5) else sort(unique(p$rho_grid))
      gr <- expand.grid(tau = g_tau, rho = g_rho)
      p$tau_grid <- gr$tau; p$rho_grid <- gr$rho
      p
    },
    pack = function(p) c(.nl_adj_args(p), list(
      tau_grid = as.numeric(p$tau_grid),
      rho_grid = as.numeric(p$rho_grid)
    )),
    theta = function(p) list(
      grid  = cbind(tau = p$tau_grid, rho = p$rho_grid),
      names = c("tau", "rho")
    )
  ),

  mcar = list(
    # Separable multivariate CAR (p coupled areal fields sharing Sigma (x) Q^-1).
    # Multi-block-only (like iid): the coupled inner solve lives in the joint
    # driver. The outer axes are the p(p+1)/2 log-Cholesky coordinates of Sigma.
    cpp_fn = NULL,
    defaults = function(p, a) {
      if (is.null(p$logchol_grid)) {
        p$logchol_grid <- .mcar_default_logchol_grid(as.integer(p$n_fields))
      }
      p
    },
    pack = function(p) stop(
      "MCAR is only supported inside a multi-block prior (a coupled areal ",
      "field declared by spatial(graph, ~ ... | cell)).", call. = FALSE
    ),
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
    cpp_fn = NULL,
    defaults = function(p, a) {
      if (is.null(p$logchol_grid)) {
        p$logchol_grid <- .mcar_default_logchol_grid(as.integer(p$n_fields))
      }
      p
    },
    pack = function(p) stop(
      "miid is only supported inside a multi-block prior (a correlated ",
      "random-slope term, e.g. (1 + x | group)).", call. = FALSE
    ),
    theta = function(p) {
      g <- p$logchol_grid
      list(grid = g, names = colnames(g))
    }
  ),

  nngp = list(
    cpp_fn = "cpp_nested_laplace_nngp",
    defaults = function(p, a) {
      if (is.null(p$sigma2_grid) || is.null(p$phi_gp_grid)) {
        s2 <- exp(seq(log(0.05), log(2), length.out = 5))
        pg <- exp(seq(log(0.05), log(1.5), length.out = 5))
        gr <- expand.grid(sigma2 = s2, phi_gp = pg)
        p$sigma2_grid <- gr$sigma2
        p$phi_gp_grid <- gr$phi_gp
      }
      p
    },
    pack = function(p) {
      # nn_order in cpp_nested_laplace_nngp expects 0-based indices, matching
      # the convention in cpp_laplace_fit_gp (see R/fit_laplace.R:433).
      n_spatial <- as.integer(p$n_spatial)
      # spatial_idx (1-based, length N) maps obs -> spatial unit. Default to
      # 1..n_spatial when N == n_spatial (the legacy one-obs-per-location case).
      spatial_idx <- if (!is.null(p$spatial_idx)) {
        as.integer(p$spatial_idx)
      } else {
        seq_len(n_spatial)
      }
      list(
        spatial_idx = spatial_idx,
        coords      = as.matrix(p$coords),
        nn_idx      = as.matrix(p$nn_idx),
        nn_dist     = as.matrix(p$nn_dist),
        nn_order    = as.integer((p$nn_order %||% seq_len(n_spatial)) - 1L),
        n_spatial   = n_spatial,
        nn          = as.integer(p$nn),
        sigma2_grid = as.numeric(p$sigma2_grid),
        phi_gp_grid = as.numeric(p$phi_gp_grid),
        cov_type    = as.integer(p$cov_type %||% 2L)  # default Matern-5/2
      )
    },
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
    cpp_fn = NULL,   # multi-block only; reached via .nl_dispatch_multi
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
        n_g <- as.integer(p$n_grid %||% 5L)
        span <- as.numeric(p$grid_span %||% 1.4)
        rng_mode <- (p$prior_range %||% c(1, 0.5))[1]
        sig_mode <- (p$prior_sigma %||% c(1, 0.5))[1]
        rg <- exp(seq(log(rng_mode / span), log(rng_mode * span), length.out = n_g))
        sg <- exp(seq(log(sig_mode / span), log(sig_mode * span), length.out = n_g))
        gr <- expand.grid(range = rg, sigma = sg)
        p$range_grid <- gr$range
        p$sigma_grid <- gr$sigma
      }
      p
    },
    pack = function(p) stop(
      "spde is only supported inside a multi-block latent prior on the ",
      "nested-Laplace path. Pass it as a one-element list to ",
      "tulpa_nested_laplace(prior = list(blk), ...).",
      call. = FALSE
    ),
    theta = function(p) list(
      grid  = cbind(range = as.numeric(p$range_grid),
                    sigma = as.numeric(p$sigma_grid)),
      names = c("range", "sigma")
    )
  ),

  iid = list(
    # Unstructured (iid) Gaussian latent, integrated over a single sigma axis.
    # Multi-block-only: there is no single-arm iid kernel (a lone iid field is a
    # plain random intercept). Reached via .nl_dispatch_multi -- the C++
    # build_blocks_from_spec builds the block with d_fac(k) = sigma_k and an
    # N(0, 1) prior. A one-point sigma_grid conditions the field at a FIXED
    # sigma (an EM-estimated random-effect term on a nested block carries its
    # own grid this way; see .fit_block_via_nested_laplace).
    cpp_fn = NULL,
    defaults = function(p, a) {
      if (is.null(p$sigma_grid)) {
        p$sigma_grid <- exp(seq(log(0.1), log(3), length.out = 5))
      }
      p
    },
    pack = function(p) stop(
      "iid is only supported inside a multi-block latent prior on the ",
      "nested-Laplace path. Pass it as one element of a block list to ",
      "tulpa_nested_laplace(prior = list(...), ...).",
      call. = FALSE
    ),
    theta = function(p) list(grid = as.numeric(p$sigma_grid), names = "sigma")
  ),

  hsgp = list(
    cpp_fn = "cpp_nested_laplace_hsgp",
    defaults = function(p, a) {
      if (is.null(p$sigma2_grid) || is.null(p$lengthscale_grid)) {
        s2 <- exp(seq(log(0.05), log(2), length.out = 5))
        ls <- exp(seq(log(0.05), log(1.5), length.out = 5))
        gr <- expand.grid(sigma2 = s2, ell = ls)
        p$sigma2_grid <- gr$sigma2
        p$lengthscale_grid <- gr$ell
      }
      p
    },
    pack = function(p) list(
      phi_basis        = as.matrix(p$phi_basis),
      lambda_eig       = as.numeric(p$lambda_eig),
      sigma2_grid      = as.numeric(p$sigma2_grid),
      lengthscale_grid = as.numeric(p$lengthscale_grid)
    ),
    theta = function(p) list(
      grid  = cbind(sigma2 = p$sigma2_grid, lengthscale = p$lengthscale_grid),
      names = c("sigma2", "lengthscale")
    )
  ),

  hsgp_mo = list(
    # Multi-output (co-regionalization) HSGP block (Stage 1.7). K = n_arms
    # correlated latent fields share basis + eigenvalues, cross-output
    # coefficients live in Sigma. First ship: K = 2 with raw axes
    # (sigma_1, sigma_2, rho, ell) and the natural Cartesian product over
    # paired per-axis grids. Multi-block-only -- there is no single-arm
    # version of multi-output HSGP.
    cpp_fn = NULL,
    defaults = function(p, a) {
      # All four grids treated as paired axes (each row of theta_grid is
      # one (sigma_1, sigma_2, rho, ell) tuple). When any grid is missing
      # we fill in a small Cartesian default; when the user supplies grids
      # they must already be aligned (use expand.grid() on the R side to
      # form a Cartesian, then pass the four aligned columns).
      if (is.null(p$sigma_1_grid) || is.null(p$sigma_2_grid) ||
          is.null(p$rho_grid) || is.null(p$lengthscale_grid)) {
        s1 <- exp(seq(log(0.3), log(1.5), length.out = 3))
        s2 <- exp(seq(log(0.3), log(1.5), length.out = 3))
        rh <- c(-0.4, 0.0, 0.4)
        ls <- exp(seq(log(0.1), log(1.0), length.out = 3))
        gr <- expand.grid(sigma_1 = s1, sigma_2 = s2, rho = rh, ell = ls,
                          KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
        p$sigma_1_grid     <- gr$sigma_1
        p$sigma_2_grid     <- gr$sigma_2
        p$rho_grid         <- gr$rho
        p$lengthscale_grid <- gr$ell
      }
      p
    },
    pack = function(p) stop(
      "hsgp_mo is only supported inside a multi-block joint prior. ",
      "Wrap the block in list() and pass to tulpa_nested_laplace_joint(prior = list(blk), ...).",
      call. = FALSE
    ),
    theta = function(p) list(
      grid  = cbind(sigma_1 = as.numeric(p$sigma_1_grid),
                    sigma_2 = as.numeric(p$sigma_2_grid),
                    rho     = as.numeric(p$rho_grid),
                    ell     = as.numeric(p$lengthscale_grid)),
      names = c("sigma_1", "sigma_2", "rho", "ell")
    )
  ),

  rw1 = list(
    cpp_fn = "cpp_nested_laplace_temporal",
    defaults = function(p, a) {
      if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid()
      p
    },
    pack = function(p) list(
      temporal_idx  = as.integer(p$temporal_idx),
      n_times       = as.integer(p$n_times),
      temporal_type = "rw1",
      tau_grid      = as.numeric(p$tau_grid),
      rho_grid      = numeric(0),
      cyclic        = isTRUE(p$cyclic),
      n_groups      = as.integer(p$n_groups %||% 1L)
    ),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  rw2 = list(
    cpp_fn = "cpp_nested_laplace_temporal",
    defaults = function(p, a) {
      if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid()
      p
    },
    pack = function(p) list(
      temporal_idx  = as.integer(p$temporal_idx),
      n_times       = as.integer(p$n_times),
      temporal_type = "rw2",
      tau_grid      = as.numeric(p$tau_grid),
      rho_grid      = numeric(0),
      cyclic        = FALSE,
      n_groups      = as.integer(p$n_groups %||% 1L)
    ),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  ar1 = list(
    cpp_fn = "cpp_nested_laplace_temporal",
    defaults = function(p, a) {
      if (is.null(p$tau_grid) || is.null(p$rho_grid)) {
        g_tau <- exp(seq(log(0.5), log(20), length.out = 5))
        g_rho <- c(0.0, 0.4, 0.7, 0.9, 0.97)
        gr <- expand.grid(tau = g_tau, rho = g_rho)
        p$tau_grid <- gr$tau; p$rho_grid <- gr$rho
      }
      p
    },
    pack = function(p) list(
      temporal_idx  = as.integer(p$temporal_idx),
      n_times       = as.integer(p$n_times),
      temporal_type = "ar1",
      tau_grid      = as.numeric(p$tau_grid),
      rho_grid      = as.numeric(p$rho_grid),
      cyclic        = FALSE,
      n_groups      = as.integer(p$n_groups %||% 1L)
    ),
    theta = function(p) list(
      grid  = cbind(tau = p$tau_grid, rho = p$rho_grid),
      names = c("tau", "rho")
    )
  ),

  lf = list(
    # Latent factor block (Stage 1.6a). F = 1 only. No outer-grid axes:
    # identifiability is handled by tight Gaussian anchors on (u_1,
    # lambda_1) inside the C++ factory; everything else is jointly
    # optimized by the inner Newton. Returns a 1 x 0 grid so the joint
    # Cartesian product treats this block as a "no axis" contributor.
    cpp_fn = NULL,
    defaults = function(p, a) p,
    pack = function(p) stop(
      "lf is only supported inside a multi-block prior. ",
      "Wrap the block in list(): tulpa_nested_laplace_joint(prior = list(blk), ...).",
      call. = FALSE
    ),
    theta = function(p) list(
      grid  = matrix(numeric(0), nrow = 1L, ncol = 0L),
      names = character(0)
    )
  ),

  tgmrf = list(
    # User-defined GMRF block (see ?tgmrf). Only available inside a multi-
    # block prior. The outer theta-grid is the Cartesian product of
    # per-axis grids; per-axis defaults are 5 equispaced points between
    # block$bounds$lower[j] and block$bounds$upper[j], or block$init[j] +/-
    # 2 if no bounds are supplied. Precomputed Q_k / logdet / log p(theta_k)
    # at every grid row are packed into the C++ block spec by
    # `.nl_block_spec_for_cpp`.
    cpp_fn = NULL,
    defaults = function(p, a) {
      if (is.null(p$theta_grid_built)) {
        d <- p$theta_dim
        axes <- vector("list", d)
        for (j in seq_len(d)) {
          if (!is.null(p$bounds)) {
            lo <- p$bounds$lower[j]
            hi <- p$bounds$upper[j]
          } else {
            lo <- p$init[j] - 2
            hi <- p$init[j] + 2
          }
          axes[[j]] <- seq(lo, hi, length.out = 5L)
        }
        names(axes) <- p$theta_names
        gr <- do.call(expand.grid, axes)
        p$theta_grid_built <- as.matrix(gr)
      }
      p
    },
    pack = function(p) stop(
      "tgmrf is only supported inside a multi-block prior. ",
      "Wrap the block in list(): tulpa_nested_laplace(prior = list(blk), ...).",
      call. = FALSE
    ),
    theta = function(p) list(
      grid  = p$theta_grid_built,
      names = p$theta_names
    )
  )
)

.nl_dispatch <- function(type, a, p) {
  spec <- .NL_REGISTRY[[type]]
  if (is.null(spec)) {
    stop("Unknown prior type: ", type,
         ". Supported: ", paste(names(.NL_REGISTRY), collapse = ", "), ".",
         call. = FALSE)
  }
  p   <- spec$defaults(p, a)
  out <- do.call(spec$cpp_fn, c(spec$pack(p), a))
  th  <- spec$theta(p)
  out$theta_grid  <- th$grid
  out$theta_names <- th$names
  out
}

# Reweight the AR1 nested-Laplace outer grid by a Beta(a, b) prior on
# u = (rho + 1)/2. The inner marginal (log_prior_ar1) carries no rho prior, so an
# equal-weight grid implies a uniform rho; adding the Beta log-density at each
# natural-scale grid rho makes the outer integration honor `rho_prior`. There is
# no logit Jacobian here -- that belongs to the sampler's unconstrained
# parameterization, not the grid. Default Beta(1, 1) is uniform and a no-op.
.nl_apply_ar1_rho_prior <- function(res, type, prior) {
  if (type != "ar1" || is.null(prior$rho_prior)) return(res)
  ab <- .ar1_rho_beta_ab(prior$rho_prior)
  if (ab[1L] == 1 && ab[2L] == 1) return(res)
  tg <- res$theta_grid
  if (is.null(tg) || !is.matrix(tg) || !("rho" %in% colnames(tg))) return(res)
  u  <- pmin(pmax(0.5 * (tg[, "rho"] + 1), 1e-12), 1 - 1e-12)
  res$log_marginal <- res$log_marginal +
    (ab[1L] - 1) * log(u) + (ab[2L] - 1) * log1p(-u)
  res
}

# Default 1D log-spaced tau grid: a 9-point search over a wide precision range,
# shared by the icar / rw1 / rw2 intrinsic-GMRF kernels.
.default_tau_grid <- function() {
  exp(seq(log(0.3), log(30), length.out = 9))
}

# Normalise log-marginals to integration weights summing to 1. The single
# weight normaliser for every outer hyperparameter grid: it drops non-finite
# nodes (an inner Newton diverging in a grid corner returns +Inf / NaN) from
# the max-shift, zeroes their weight, and renormalises over the finite cells;
# all-NA (with a warning) when no node carries mass, never NaN. There is no
# unguarded variant -- a single `max(lm)` over a grid with one +Inf / NaN cell
# would collapse every weight to NaN and break the grid sampler. `what` names
# the grid in the degenerate-case warning.
.nl_normalise_weights_safe <- function(lm, what = "grids / data") {
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


`%||%` <- function(x, y) if (is.null(x)) y else x

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
.nl_attach_grid_hessians <- function(res, p_fixed) {
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
  grid_hessians <- vector("list", n_grid)
  grid_modes    <- vector("list", n_grid)

  E <- matrix(0, nrow = n_x, ncol = p_fixed)
  for (j in seq_len(p_fixed)) E[j, j] <- 1

  modes_mat <- res$modes  # n_grid x n_x

  for (k in seq_len(n_grid)) {
    L <- Matrix::sparseMatrix(
      i = Q_i[[k]], p = Q_p[[k]], x = Q_x[[k]],
      dims = c(n_x, n_x), index1 = FALSE
    )
    diag_L <- Matrix::diag(L)
    Qk <- L + Matrix::t(L) - Matrix::Diagonal(n_x, diag_L)

    # V = Qk^{-1} E selects the fixed-effect columns of the joint inverse; its
    # top p_fixed rows are the fixed-effect marginal covariance. A dense GP /
    # NNGP field precision at close locations makes the joint precision
    # ill-conditioned, which trips Matrix's condition-number guard. Retry with a
    # negligible diagonal jitter (scaled to the precision) so the marginal-SE
    # extraction stays available rather than aborting the whole fit.
    V <- tryCatch(
      as.matrix(Matrix::solve(Qk, E)),
      error = function(e) {
        jit <- 1e-8 * mean(Matrix::diag(Qk))
        as.matrix(Matrix::solve(Qk + Matrix::Diagonal(n_x, jit), E))
      }
    )
    Sigma_bb <- V[seq_len(p_fixed), , drop = FALSE]
    grid_hessians[[k]] <- solve(Sigma_bb)

    grid_modes[[k]] <- if (!is.null(modes_mat)) {
      as.numeric(modes_mat[k, seq_len(p_fixed)])
    } else {
      rep(NA_real_, p_fixed)
    }
  }

  res$grid_hessians <- grid_hessians
  res$grid_modes    <- grid_modes
  # Strip the verbose CSC scratch fields once Hessians are assembled.
  res$Q_csc_p_per_grid <- NULL
  res$Q_csc_i_per_grid <- NULL
  res$Q_csc_x_per_grid <- NULL
  res$Q_csc_n          <- NULL
  res
}

# --- Multi-block dispatch ---
#
# A multi-block prior is `list(block1, block2, ...)` where each blockN is a
# list with a `type` field (same shape as a single-block prior). At each
# outer-grid point all blocks share a Newton solve; the joint grid is the
# Cartesian product of per-block axes.

.is_multi_block_prior <- function(p) {
  is.list(p) && is.null(p$type) && length(p) > 0 &&
    all(vapply(p, function(x) is.list(x) && !is.null(x$type), logical(1)))
}

# Structural spec for one block sent to cpp_nested_laplace_multi. Grid values
# go in theta_grid separately; this function returns only the type-specific
# data needed to build a LatentBlock (idx vector, sizes, adjacency, etc.).
#
# `block_joint_grid` is the joint-grid sub-matrix (n_joint x block$theta_dim)
# already realised from the Cartesian product. tgmrf needs it to precompute
# Q(theta_k) at every joint-grid row; built-in types ignore the argument
# because they read theta values directly from theta_grid in the C++ side.
.nl_block_spec_for_cpp <- function(p, block_joint_grid = NULL) {
  type <- tolower(p$type)
  if (type %in% c("icar", "bym2", "car_proper")) {
    out <- list(
      type            = type,
      spatial_idx     = as.integer(p$spatial_idx),
      n_spatial_units = as.integer(p$n_spatial_units),
      adj_row_ptr     = as.integer(p$adj_row_ptr),
      adj_col_idx     = as.integer(p$adj_col_idx),
      n_neighbors     = as.integer(p$n_neighbors)
    )
    if (type == "bym2") {
      out$scale_factor <- as.numeric(p$scale_factor %||% 1.0)
    }
    # Optional per-observation design weight: an areal varying-coefficient
    # (SVC) field whose eta contribution is svc_weight[i] * z[spatial_idx[i]].
    # The icar block builder reads it as LatentBlock::row_weight. Only icar
    # carries it on this path; absent -> a plain intercept field.
    if (type == "icar" && !is.null(p$svc_weight)) {
      out$svc_weight <- as.numeric(p$svc_weight)
    }
    out
  } else if (type %in% c("rw1", "rw2", "ar1")) {
    out <- list(
      type         = type,
      temporal_idx = as.integer(p$temporal_idx),
      n_times      = as.integer(p$n_times)
    )
    if (type %in% c("rw1", "rw2")) out$cyclic <- isTRUE(p$cyclic)
    out
  } else if (type == "iid") {
    list(
      type    = "iid",
      obs_idx = as.integer(p$obs_idx),
      n_units = as.integer(p$n_units)
    )
  } else if (type == "spde") {
    # FEM mesh + projection. The C++ side wraps the single A CSC in a length-1
    # per-arm list and builds the LatentBlock via make_spde_block. Optional
    # rational coefficients (fractional nu) pass through when present.
    out <- list(
      type    = "spde",
      n_mesh  = as.integer(p$n_mesh),
      n_obs   = as.integer(p$n_obs),
      A_x     = as.numeric(p$A_x),
      A_i     = as.integer(p$A_i),
      A_p     = as.integer(p$A_p),
      C0_diag = as.numeric(p$C0_diag),
      G1_x    = as.numeric(p$G1_x),
      G1_i    = as.integer(p$G1_i),
      G1_p    = as.integer(p$G1_p),
      nu      = as.numeric(p$nu)
    )
    if (!is.null(p$rational_poles) && !is.null(p$rational_weights)) {
      out$rational_poles   <- as.numeric(p$rational_poles)
      out$rational_weights <- as.numeric(p$rational_weights)
    }
    out
  } else if (type == "tgmrf") {
    # Precompute Q(theta_k) for every joint-grid row and pack CSC triples +
    # logdet + log p(theta_k). The C++ side reads them directly at grid
    # point k - no R callback during the inner Newton loop. Iterating over
    # block_joint_grid (rather than the block's own grid) is what lets
    # tgmrf compose with other blocks in a multi-block prior: when other
    # blocks vary across joint rows but tgmrf does not, the duplicated Q
    # matrices land at the right joint index.
    #
    # The Q / log_prior(theta) evaluation either goes through the user's R
    # closure (`backend == "r"`, the default) or through the registered C++
    # spec (`backend == "cpp"`). Wire format downstream is identical in both
    # cases.
    if (is.null(block_joint_grid)) {
      stop("Internal error: tgmrf block_spec called without block_joint_grid.",
           call. = FALSE)
    }
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
      type                     = "tgmrf",
      n_latent                 = as.integer(n_latent),
      obs_idx                  = as.integer(p$obs_idx),
      Q_csc_p_per_grid         = Q_csc_p,
      Q_csc_i_per_grid         = Q_csc_i,
      Q_csc_x_per_grid         = Q_csc_x,
      logdet_Q_per_grid        = as.numeric(logdet_Q),
      log_prior_theta_per_grid = as.numeric(log_pi_th)
    )
  } else {
    stop("Block type '", type, "' is not supported in multi-block priors.",
         call. = FALSE)
  }
}

# Per-block axis grid as a matrix `[n_block_cells x n_axes_for_block]`.
# Single-axis types (icar / rw1 / rw2 / iid) get a 1-column matrix; 2-axis
# types (bym2 / car_proper / ar1) get a 2-column matrix.
.nl_block_axis_grid <- function(p) {
  spec <- .NL_REGISTRY[[tolower(p$type)]]
  if (is.null(spec)) {
    stop("Unknown block type: ", p$type, call. = FALSE)
  }
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

# Soft cap on joint-grid cell count. CCD integration around a pilot mode is
# the standard fix for k >= 3 blocks (see R/ccd_grid.R); the multi-block driver
# uses the Cartesian outer integration, which is correct but scales
# multiplicatively in the block count, so warn past the soft cap and proceed.
.NL_MULTI_GRID_WARN <- 50L
.NL_MULTI_GRID_HARD_CAP <- 2048L

.nl_dispatch_multi <- function(cargs, prior_list, likelihood = NULL,
                               progress = .nl_progress_args(list(progress = FALSE))) {
  # Inject a default obs_idx for any tgmrf block that didn't supply one.
  # The C++ scatter needs obs_idx[i] -> latent slot for each observation;
  # the canonical "one obs per latent slot" case has N == n_latent and the
  # identity mapping. Heterogeneous (N != n_latent) layouts must supply
  # obs_idx explicitly at tgmrf() construction.
  N_obs <- length(cargs$y)
  for (b in seq_along(prior_list)) {
    blk <- prior_list[[b]]
    if (identical(tolower(blk$type %||% ""), "tgmrf") && is.null(blk$obs_idx)) {
      if (blk$n_latent != N_obs) {
        stop("tgmrf block has n_latent = ", blk$n_latent,
             " but data has N = ", N_obs, " observations. Provide an explicit ",
             "`obs_idx` argument to tgmrf() mapping observations to latent slots.",
             call. = FALSE)
      }
      prior_list[[b]]$obs_idx <- seq_len(N_obs)
    }
  }

  per_block <- lapply(prior_list, .nl_block_axis_grid)
  prepared  <- lapply(per_block, function(x) x$prepared)
  block_grids <- lapply(per_block, function(x) x$grid)

  # Cartesian product of per-block row indices.
  row_counts <- vapply(block_grids, nrow, integer(1))
  idx <- do.call(expand.grid, lapply(row_counts, seq_len))
  n_cells <- nrow(idx)
  if (n_cells > .NL_MULTI_GRID_HARD_CAP) {
    stop(sprintf(
      "Joint multi-block grid has %d cells (hard cap %d). Reduce per-block grid sizes or wait for CCD integration support.",
      n_cells, .NL_MULTI_GRID_HARD_CAP
    ), call. = FALSE)
  }
  if (n_cells > .NL_MULTI_GRID_WARN) {
    warning(sprintf(
      "Joint multi-block grid has %d cells (>%d). Each cell costs one inner Newton solve; reduce per-block grid sizes if this is slow. CCD integration is a follow-up.",
      n_cells, .NL_MULTI_GRID_WARN
    ), call. = FALSE)
  }

  # Concatenate per-block axis grids into the joint theta_grid.
  joint_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
    block_grids[[b]][idx[[b]], , drop = FALSE]
  }))
  axis_counts  <- vapply(block_grids, ncol, integer(1))
  axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
  # Block-prefix the axis names so they're unambiguous (e.g. block1.tau).
  axis_names <- unlist(lapply(seq_along(block_grids), function(b) {
    paste0("b", b, ".", colnames(block_grids[[b]]))
  }))
  colnames(joint_grid) <- axis_names

  blocks_spec <- lapply(seq_along(prepared), function(b) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    block_joint <- joint_grid[, cols, drop = FALSE]
    .nl_block_spec_for_cpp(prepared[[b]], block_joint)
  })

  out <- cpp_nested_laplace_multi(
    y           = cargs$y,
    n           = cargs$n,
    X           = cargs$X,
    re_idx      = cargs$re_idx,
    n_re_groups = cargs$n_re_groups,
    sigma_re    = cargs$sigma_re,
    blocks_spec = blocks_spec,
    theta_grid  = joint_grid,
    axis_offsets = axis_offsets,
    family      = cargs$family,
    phi         = cargs$phi,
    max_iter    = cargs$max_iter,
    tol         = cargs$tol,
    n_threads   = cargs$n_threads,
    x_init_nullable = cargs$x_init_nullable,
    store_Q     = isTRUE(cargs$store_Q),
    likelihood  = likelihood,
    progress          = isTRUE(progress$progress),
    progress_every    = as.integer(progress$progress_every),
    progress_throttle = as.numeric(progress$progress_throttle),
    progress_file     = as.character(progress$progress_file),
    checkpoint_path   = as.character(cargs$checkpoint_path %||% "")
  )

  out$theta_grid   <- joint_grid
  out$theta_names  <- axis_names
  out$axis_offsets <- axis_offsets

  # Fold each SPDE block's PC prior on (range, sigma) into the per-cell
  # log-marginal. The areal / temporal / iid blocks carry their hyperprior
  # inside the C++ block log_prior (it folds into log|Q|); the SPDE factory
  # drops the log|Q|/2 normalizer (it is recovered from the Laplace Hessian
  # log-determinant) and carries no hyperprior, so the proper Matern PC prior
  # is added here -- the same prior + closed form the single-Laplace SPDE path
  # uses (pc_prior_log_density), so both paths integrate the same posterior.
  for (b in seq_along(prepared)) {
    pb <- prepared[[b]]
    if (!identical(tolower(pb$type %||% ""), "spde")) next
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    rng  <- as.numeric(joint_grid[, cols[1L]])
    sig  <- as.numeric(joint_grid[, cols[2L]])
    lp <- pc_prior_log_density(rng, sig,
                               pb$prior_range %||% c(1, 0.5),
                               pb$prior_sigma %||% c(1, 0.5))
    out$log_marginal <- out$log_marginal + lp
  }

  out$weights      <- .nl_normalise_weights_safe(out$log_marginal, "multi-block outer grid")
  out <- .nl_posterior_moments_multi(out, prepared, axis_offsets, joint_grid)
  out$blocks       <- prepared
  out
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
#'    Proper CAR (rho estimated) is converted to a 2D grid over (tau, rho)
#'    using the spec's eigenvalue-derived `rho_bounds`.
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
  # prior, also used by the tulpa() front door. gp/nngp drive the `nngp` kernel,
  # hsgp the `hsgp` kernel; both are single-block (cpp_fn present in .NL_REGISTRY).
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
  has_ctrl <- !is.null(control$progress) || !is.null(control$progress.every) ||
              !is.null(control$progress.throttle) || !is.null(control$progress.file)
  if (!has_ctrl) {
    opt <- getOption("tulpa.nl_progress", NULL)
    if (is.list(opt)) return(opt)
  }
  list(
    progress          = isTRUE(control$progress %||% TRUE),
    progress_every    = as.integer(control$progress.every    %||% 0L),
    progress_throttle = as.numeric(control$progress.throttle %||% 2),
    progress_file     = as.character(control$progress.file    %||% "")
  )
}

# Resolve the grid-cell checkpoint spec to a normalized
# `list(path, resume)`. `control$checkpoint = list(path = , resume = TRUE)`
# enables it; `resume = FALSE` starts over (the front door removes any prior
# file before the first kernel call). Like `.nl_progress_args`, an absent
# control key falls back to the scoped `tulpa.nl_checkpoint` option so the
# checkpoint reaches grids run through fixed-control internal paths.
.nl_checkpoint_args <- function(control) {
  cp <- control$checkpoint
  if (is.null(cp)) {
    opt <- getOption("tulpa.nl_checkpoint", NULL)
    if (is.list(opt)) return(opt)
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
