# Corrected integrated Laplace as an inner-layer debias (gcol33/tulpa#351).
#
# The arbiters here are outside the estimator: a family whose inner Laplace is
# exact (so the correction must reproduce the uncorrected fit to machine
# accuracy), and a two-dimensional latent whose cell marginal is available by
# direct quadrature (so the correction must converge to it while the Laplace
# does not).

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A gaussian arm with one grouping factor. The joint log density is quadratic
# in the latent, so the inner Gaussian IS the conditional posterior and every
# importance ratio along the grid is the same number.
cila_gaussian_fit <- function(cila = NULL, seed = 11L, n_group = 8L,
                              per_group = 4L, force_sparse = FALSE) {
  set.seed(seed)
  xg <- stats::rnorm(n_group)
  region <- rep(seq_len(n_group), each = per_group)
  X <- cbind(1, xg[region])
  u <- stats::rnorm(n_group, 0, 0.7)
  y <- as.numeric(X %*% c(0.5, -0.8)) + u[region] +
    stats::rnorm(n_group * per_group, 0, 0.5)
  ctl <- list(max_iter = 300L, tol = 1e-11, n_threads = 1L,
              keep_grid_hessians = TRUE, diagnose_k = FALSE,
              diagnose_skew = FALSE, auto_recenter = FALSE, progress = FALSE,
              force_sparse = force_sparse)
  if (!is.null(cila)) ctl$cila <- cila
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(list(y = y, X = X, family = "gaussian", phi = 0.25,
                          beta_prior_prec = rep(0.16, 2))),
    prior = list(list(type = "iid", obs_idx = as.integer(region),
                      n_units = n_group,
                      sigma_grid = exp(seq(log(0.25), log(1.4),
                                           length.out = 5)))),
    control = ctl))
}

# A binomial arm with ONE group, so the latent is (beta, u) and the cell
# marginal p(y | sigma) is a two-dimensional integral this file evaluates
# directly. B_PREC is the arm's own beta_prior_prec, so the reference and the
# fit condition on the same prior.
CILA_PREC <- 0.16
CILA_SIGMA_GRID <- c(0.4, 0.8, 1.6)

cila_binom_data <- function(seed = 4L, n_trial = 6L) {
  set.seed(seed)
  y <- c(rep(1, n_trial))            # a saturated single group: maximally
  list(y = y, n_trial = n_trial,     # non-Gaussian in the inner layer
       X = matrix(1, n_trial, 1L),
       obs_idx = rep(1L, n_trial))
}

cila_binom_fit <- function(d, cila = NULL) {
  ctl <- list(max_iter = 300L, tol = 1e-11, n_threads = 1L,
              keep_grid_hessians = TRUE, diagnose_k = FALSE,
              diagnose_skew = FALSE, auto_recenter = FALSE, progress = FALSE)
  if (!is.null(cila)) ctl$cila <- cila
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(list(y = d$y, n_trials = rep(1L, d$n_trial), X = d$X,
                          family = "binomial", phi = 1,
                          beta_prior_prec = CILA_PREC)),
    prior = list(list(type = "iid", obs_idx = d$obs_idx, n_units = 1L,
                      sigma_grid = CILA_SIGMA_GRID)),
    control = ctl))
}

# log p(y | sigma) by direct two-dimensional quadrature over (beta, u), on a
# grid wide and fine enough that refining it does not move the answer.
cila_binom_exact_log_marg <- function(d, sigma, half = 14, ng = 1601L) {
  b <- seq(-half, half, length.out = ng)
  u <- seq(-half * sigma, half * sigma, length.out = ng)
  eta <- outer(b, u, "+")
  ll <- sum(d$y) * eta - d$n_trial * log1p(exp(eta))
  lp <- ll - 0.5 * CILA_PREC * b^2 + 0.5 * log(CILA_PREC / (2 * pi)) -
    rep(0.5 * (u / sigma)^2, each = ng) - log(sigma) - 0.5 * log(2 * pi)
  m <- max(lp)
  m + log(sum(exp(lp - m))) + log(diff(b[1:2])) + log(diff(u[1:2]))
}

# ---------------------------------------------------------------------------
# 1. The request is validated before anything runs
# ---------------------------------------------------------------------------

test_that("control$cila rejects settings it does not have", {
  expect_error(.cila_config(list(npoints = 1024L)), "Unknown")
  expect_error(.cila_config(list(variant = "sobol")), "variant")
  expect_error(.cila_config(list(n_points = 0L)), "positive")
  expect_null(.cila_config(NULL))
  expect_null(.cila_config(FALSE))
  expect_equal(.cila_config(TRUE)$variant, "qmc")
  expect_equal(.cila_config(TRUE)$n_points, 1024L)
})

test_that("the effort floor names the measurement it comes from", {
  # gcol33/tulpa#341 measured every variant leaving the simultaneous
  # calibration band at M = 64 and the iid one still leaving it at M = 256:
  # Proposition 5's recovered posterior is a weighted particle set, and below
  # the floor it is too coarse to be a marginal at all.
  expect_error(.cila_config(list(n_points = 64L)), "512")
  expect_error(.cila_config(list(n_points = 256L)), "particle set")
  expect_silent(.cila_config(list(n_points = 512L)))
})

test_that("the kernel request is NULL where there is nothing to retain", {
  cfg <- .cila_config(TRUE)
  expect_null(.cila_request(cfg, 0L))
  expect_null(.cila_request(NULL, 3L))
  req <- .cila_request(cfg, 3L)
  expect_equal(req$n_fixed, 3L)
  expect_equal(req$variant, 0L)
})

# ---------------------------------------------------------------------------
# 2. A fit that did not ask for the correction is untouched
# ---------------------------------------------------------------------------

test_that("requesting the correction leaves the uncorrected fit alone", {
  skip_on_cran()
  f0 <- cila_gaussian_fit()
  f1 <- cila_gaussian_fit(cila = list(n_points = 512L))
  expect_null(f0$cila)
  expect_null(f0$draws)
  expect_false(is.null(f1$cila))
  # The correction is post-processing on a second pass over the same grid: the
  # first pass's own numbers cannot move. The corrected fit ADOPTS the corrected
  # grid read (gcol33/tulpa#367), so the pair to compare against is the one it
  # kept, not the one it now reports.
  expect_identical(f0$log_marginal, f1$cila$laplace$log_marginal)
  expect_identical(f0$weights, f1$cila$laplace$weights)
  expect_identical(f0$grid_modes, f1$grid_modes)
  expect_null(f0$weights_source)
})

# ---------------------------------------------------------------------------
# 2b. One grid weighting per fit (gcol33/tulpa#367)
# ---------------------------------------------------------------------------

test_that("the corrected fit carries ONE grid weighting", {
  skip_on_cran()
  f <- cila_gaussian_fit(cila = list(n_points = 512L))
  # Not "equal to within a tolerance" -- the same numbers. A caller pairing
  # `weights` with `cila$cell_weights` cannot pair them wrongly because there is
  # nothing to pair: they are one vector reported under two names.
  expect_identical(f$weights, f$cila$cell_weights)
  expect_identical(f$log_marginal, f$cila$cell_log_marginal)
  expect_equal(f$weights_source, "cila")
  expect_true(f$cila$weights_adopted)
  expect_equal(f$cila$retained_mass, 1)
  # The only other weighting on the fit is the pre-correction one, under a name
  # that says so.
  expect_false(identical(f$cila$laplace$weights, f$weights))
  expect_equal(as.numeric(f$cila$laplace$weights), as.numeric(f$weights),
               tolerance = 1e-9)
})

test_that("the hyperparameter summary is recomputed from the adopted read", {
  skip_on_cran()
  f <- cila_gaussian_fit(cila = list(n_points = 512L))
  # theta_mean is the fitter's own weighted moment, so it must be the weighted
  # moment of the weights the fit reports -- not of the ones it replaced.
  expect_equal(as.numeric(f$theta_mean),
               as.numeric(crossprod(f$weights, f$theta_grid)),
               tolerance = 1e-12, ignore_attr = TRUE)
})

test_that("a declined correction leaves the Laplace read in place, and says so", {
  cfg <- .cila_config(TRUE)
  res <- .nl_cila_attach(list(log_marginal = c(-1, -2), weights = c(0.6, 0.4)),
                         cfg, function(req) stop("boom"), p_fixed = 2L)
  expect_equal(res$weights_source, "laplace_grid")
  expect_equal(res$weights, c(0.6, 0.4))
  expect_false(res$cila$weights_adopted)
  expect_equal(res$cila$declined, "redispatch_failed")
})

# ---------------------------------------------------------------------------
# 3. The exactness arbiter: a gaussian inner layer has nothing to correct
# ---------------------------------------------------------------------------

test_that("on a gaussian arm the correction reproduces the Laplace grid", {
  skip_on_cran()
  f <- cila_gaussian_fit(cila = list(n_points = 512L, variant = "qmc"))
  expect_true(is.na(f$cila$declined))
  expect_equal(f$cila$variant_used, "qmc")
  # The inner Gaussian is the exact conditional posterior here, so every
  # importance ratio in a cell is the same number and the corrected cell mass
  # is the uncorrected one.
  expect_equal(as.numeric(f$cila$cell_weights), as.numeric(f$weights),
               tolerance = 1e-9)
  # ... and the corrected cell marginal is the Laplace one, cell by cell.
  expect_equal(as.numeric(f$cila$cell_log_marginal),
               as.numeric(f$log_marginal), tolerance = 1e-9)
  # A constant weight set is a perfect importance sample.
  expect_lt(f$cila$pareto_k, 0.5)
  # The reported posterior is now draws, and they carry the Laplace moments.
  mom <- .nested_fixed_moments(f)
  expect_equal(colMeans(f$draws), as.numeric(mom$mean), tolerance = 0.05,
               ignore_attr = TRUE)
  expect_equal(apply(f$draws, 2, stats::sd),
               sqrt(diag(mom$cov)), tolerance = 0.08, ignore_attr = TRUE)
})

test_that("every variant reproduces the gaussian grid", {
  skip_on_cran()
  for (v in c("qmc", "is", "rqmc")) {
    f <- cila_gaussian_fit(cila = list(n_points = 512L, variant = v))
    expect_equal(as.numeric(f$cila$cell_log_marginal),
                 as.numeric(f$log_marginal), tolerance = 1e-9,
                 info = v)
    expect_equal(f$cila$variant_used, v)
  }
})

# ---------------------------------------------------------------------------
# 4. The convergence arbiter: quadrature the Laplace cannot reach
# ---------------------------------------------------------------------------

test_that("the corrected cell marginal converges to the exact one", {
  skip_on_cran()
  d <- cila_binom_data()
  exact <- vapply(CILA_SIGMA_GRID,
                  function(s) cila_binom_exact_log_marg(d, s), numeric(1))
  # The reference is converged in its own right.
  coarse <- vapply(CILA_SIGMA_GRID,
                   function(s) cila_binom_exact_log_marg(d, s, half = 12,
                                                         ng = 1201L),
                   numeric(1))
  expect_lt(max(abs(exact - coarse)), 1e-5)

  f_lap <- cila_binom_fit(d)
  lap_err <- mean(abs(as.numeric(f_lap$log_marginal) - exact))
  expect_gt(lap_err, 1e-2)   # the fixture is genuinely non-Gaussian

  # Scored on the MEAN error over the cells, not the worst one. A QMC estimate
  # is not monotone in the point count cell by cell -- the net is a different
  # point set at each M rather than a refinement of the last one -- so a single
  # cell can sit further off at a higher rung while the estimator as a whole
  # converges. What has to fall is the error of the grid the fit integrates.
  errs <- vapply(c(512L, 262144L), function(M) {
    f <- cila_binom_fit(d, cila = list(n_points = M, variant = "qmc"))
    mean(abs(as.numeric(f$cila$cell_log_marginal) - exact))
  }, numeric(1))
  expect_lt(errs[2], 0.3 * errs[1])
  expect_lt(errs[2], 0.15 * lap_err)
})

test_that("the corrected latent marginal moves toward the exact posterior", {
  skip_on_cran()
  d <- cila_binom_data()
  # The exact posterior of beta, pooling the sigma cells by their own exact
  # marginals under the flat grid prior the fit uses.
  half <- 14; ng <- 1601L
  b <- seq(-half, half, length.out = ng)
  post <- rep(0, ng)
  lm_ex <- vapply(CILA_SIGMA_GRID,
                  function(s) cila_binom_exact_log_marg(d, s), numeric(1))
  wc <- exp(lm_ex - max(lm_ex)); wc <- wc / sum(wc)
  for (k in seq_along(CILA_SIGMA_GRID)) {
    s <- CILA_SIGMA_GRID[k]
    u <- seq(-half * s, half * s, length.out = ng)
    eta <- outer(b, u, "+")
    lp <- sum(d$y) * eta - d$n_trial * log1p(exp(eta)) -
      0.5 * CILA_PREC * b^2 - rep(0.5 * (u / s)^2, each = ng)
    m <- max(lp)
    dens <- exp(m) * rowSums(exp(lp - m))
    post <- post + wc[k] * dens / sum(dens)
  }
  post <- post / sum(post)
  exact_mean <- sum(b * post)

  f_lap <- cila_binom_fit(d)
  lap_mean <- as.numeric(.nested_fixed_moments(f_lap)$mean[1])
  f_cor <- cila_binom_fit(d, cila = list(n_points = 131072L, variant = "qmc"))
  cor_mean <- mean(f_cor$draws[, 1])

  expect_lt(abs(cor_mean - exact_mean), 0.5 * abs(lap_mean - exact_mean))
})

# ---------------------------------------------------------------------------
# 5. The auxiliary points are engine-owned, so the correction does not flap
# ---------------------------------------------------------------------------

test_that("the correction is reproducible and does not consume R's stream", {
  skip_on_cran()
  set.seed(1)
  f1 <- cila_gaussian_fit(cila = list(n_points = 512L))
  set.seed(99)
  stats::runif(37)
  f2 <- cila_gaussian_fit(cila = list(n_points = 512L))
  # Everything the auxiliary point set determines is identical across two runs
  # in different RNG states; only the final resample of the reported draws
  # reads R's stream.
  expect_identical(f1$cila$cell_log_marginal, f2$cila$cell_log_marginal)
  expect_identical(f1$cila$cell_weights, f2$cila$cell_weights)
  expect_identical(f1$cila$pareto_k, f2$cila$pareto_k)
})

test_that("a different seed is a different realization of the same estimator", {
  skip_on_cran()
  a <- cila_binom_fit(cila_binom_data(),
                      cila = list(n_points = 512L, variant = "rqmc"))
  b <- cila_binom_fit(cila_binom_data(),
                      cila = list(n_points = 512L, variant = "rqmc",
                                  seed = 20260351))
  expect_false(isTRUE(all.equal(a$cila$cell_log_marginal,
                                b$cila$cell_log_marginal)))
  # Two realizations of an unbiased estimator, not two different estimators.
  expect_lt(max(abs(a$cila$cell_log_marginal - b$cila$cell_log_marginal)), 0.05)
})

# ---------------------------------------------------------------------------
# 5b. The non-joint fitter carries the same correction (gcol33/tulpa#368)
# ---------------------------------------------------------------------------

# A gaussian arm on an RW1 field, through tulpa_nested_laplace() rather than the
# joint driver. Same exactness arbiter: the inner Laplace is the conditional
# posterior, so the corrected cell marginal must reproduce the uncorrected one,
# and at the joint file's own 1e-9 (gcol33/tulpa#371).
#
# It read 3.4e-07 while this loop reported its log-marginal at the post-centring
# point rather than at the Newton mode, and 1e-05 on a proper AR1 field, which
# is the same defect at the scale of how far the field's prior is from
# shift-invariant. `score_max` is the independent arbiter for that reading: it
# measured 1.6e-05 on this fixture and 5.7e-02 on the AR1 one, and both are
# ~1e-14 now.
cila_nonjoint_fit <- function(cila = NULL, seed = 3L, n_time = 10L,
                              per_time = 3L) {
  set.seed(seed)
  tidx <- rep(seq_len(n_time), each = per_time)
  xv <- stats::rnorm(n_time * per_time)
  X <- cbind(1, xv)
  fld <- cumsum(stats::rnorm(n_time, 0, 0.4))
  y <- as.numeric(X %*% c(0.3, -0.6)) + fld[tidx] +
    stats::rnorm(n_time * per_time, 0, 0.5)
  ctl <- list(max_iter = 300L, tol = 1e-11, n_threads = 1L,
              keep_grid_hessians = TRUE, diagnose_k = FALSE,
              diagnose_skew = FALSE, auto_recenter = FALSE)
  if (!is.null(cila)) ctl$cila <- cila
  suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, length(y)), X = X,
    prior = list(type = "rw1", temporal_idx = as.integer(tidx),
                 n_times = n_time,
                 tau_grid = exp(seq(log(0.5), log(12), length.out = 4))),
    family = "gaussian", phi = 0.5, control = ctl))
}

test_that("tulpa_nested_laplace() accepts the correction", {
  expect_silent(tulpa_check_control(list(cila = TRUE),
                                    .CONTROL_KEYS$nested_laplace,
                                    "tulpa_nested_laplace"))
})

test_that("on the non-joint fitter a gaussian arm reproduces the Laplace grid", {
  skip_on_cran()
  f0 <- cila_nonjoint_fit()
  f  <- cila_nonjoint_fit(cila = list(n_points = 512L, variant = "qmc"))
  expect_true(is.na(f$cila$declined))
  expect_equal(f$cila$variant_used, "qmc")
  expect_equal(as.numeric(f$cila$cell_log_marginal),
               as.numeric(f0$log_marginal), tolerance = 1e-9)
  expect_equal(as.numeric(f$weights), as.numeric(f0$weights),
               tolerance = 1e-8)
  # The reported point IS the mode, which is what makes the line above hold.
  expect_lt(max(f0$score_max), 1e-10)
  expect_lt(f$cila$pareto_k, 0.5)
  expect_equal(f$weights_source, "cila")
  expect_false(is.null(f$draws))
  expect_equal(ncol(f$draws), 2L)
})

# ---------------------------------------------------------------------------
# 5c. A sparsely factorized cell draws through its own factor
#     (gcol33/tulpa#366)
# ---------------------------------------------------------------------------

test_that("a sparse cell is corrected, and agrees with the dense route", {
  skip_on_cran()
  # `force_sparse` straddles the dispatch at a fixture small enough to run BOTH
  # ways, so the two routes are compared on the same model rather than the
  # sparse one being compared only against itself.
  f0 <- cila_gaussian_fit(force_sparse = TRUE)
  fs <- cila_gaussian_fit(cila = list(n_points = 512L), force_sparse = TRUE)
  expect_true(is.na(fs$cila$declined))
  # The exactness arbiter first: on a gaussian arm the sparse cell's corrected
  # marginal is its own uncorrected one.
  expect_equal(as.numeric(fs$cila$cell_log_marginal),
               as.numeric(f0$log_marginal), tolerance = 1e-9)
  # ... and the same model run BOTH ways agrees, so the sparse draw is the same
  # Gaussian the dense back-substitution draws from.
  fd <- cila_gaussian_fit(cila = list(n_points = 512L))
  expect_equal(as.numeric(fs$cila$cell_log_marginal),
               as.numeric(fd$cila$cell_log_marginal), tolerance = 1e-8)
  expect_equal(as.numeric(fs$weights), as.numeric(fd$weights),
               tolerance = 1e-8)
  # ... and they are not the SAME numbers, which is what says the sparse route
  # ran at all: a fit that silently took the dense one would be byte-identical.
  expect_false(identical(fs$cila$cell_log_marginal, fd$cila$cell_log_marginal))
})

# ---------------------------------------------------------------------------
# 6. Declines say why
# ---------------------------------------------------------------------------

test_that("a fit with no fixed effects to retain declines with a reason", {
  cfg <- .cila_config(TRUE)
  res <- .nl_cila_attach(list(), cfg, function(req) NULL, p_fixed = 0L)
  expect_equal(res$cila$declined, "no_fixed_effects")

  res <- .nl_cila_attach(list(log_marginal = 1:3), cfg,
                         function(req) NULL, p_fixed = 2L)
  expect_equal(res$cila$declined, "no_grid_weights")

  res <- .nl_cila_attach(list(log_marginal = c(-1, -2), weights = c(0.6, 0.4)),
                         cfg, function(req) stop("boom"), p_fixed = 2L)
  expect_equal(res$cila$declined, "redispatch_failed")

  res <- .nl_cila_attach(
    list(log_marginal = c(-1, -2), weights = c(0.6, 0.4)), cfg,
    function(req) list(cila_log_w_per_grid = list(NULL, NULL),
                       cila_fixed_per_grid = list(NULL, NULL),
                       cila_declined = c("degenerate_proposal", NA)),
    p_fixed = 2L)
  expect_equal(res$cila$declined, "degenerate_proposal")
  expect_equal(res$cila$n_cells_declined, 2L)
})
