# The corrected-integrated-Laplace correction under a parallel outer grid
# (gcol33/tulpa#424).
#
# compute_inner_cila used to allocate `Rcpp::NumericVector present(n_x)` before
# its draw loop. Rf_allocVector is not thread-safe and pushes onto R's
# protection stack, and the routine is reached from every per-cell inner solve,
# which the grid driver runs inside `#pragma omp parallel for` when
# n_threads_outer > 1. The file's own header claims the correction leaves the
# outer grid parallelisable, and the sibling inner-layer diagnostics
# (compute_inner_is_curve, inner_skew_probe_scan, inner_eta_var_scan) all take
# their buffers from the caller for exactly this reason.
#
# The failure mode was nondeterministic heap or protection-stack corruption
# surfacing inside a later GC, so a green single-threaded suite said nothing
# about it. These fits run the supported configuration and pin the parallel
# result to the serial one -- the correction is deterministic given the grid, so
# equality is the right assertion and any per-thread scratch that leaked between
# cells would break it.

cila_par_fit <- function(cila, n_threads_outer, seed = 11L,
                         n_group = 12L, per_group = 5L) {
  set.seed(seed)
  xg <- stats::rnorm(n_group)
  region <- rep(seq_len(n_group), each = per_group)
  X <- cbind(1, xg[region])
  u <- stats::rnorm(n_group, 0, 0.7)
  eta <- as.numeric(X %*% c(0.4, -0.7)) + u[region]
  y <- stats::rbinom(n_group * per_group, 1L, stats::plogis(eta))
  ctl <- list(max_iter = 300L, tol = 1e-11, n_threads = 1L,
              n_threads_outer = n_threads_outer,
              keep_grid_hessians = TRUE, diagnose_k = FALSE,
              diagnose_skew = FALSE, auto_recenter = FALSE, progress = FALSE)
  if (!is.null(cila)) ctl$cila <- cila
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(list(y = y, n_trials = rep(1L, length(y)), X = X,
                          family = "binomial", phi = 1,
                          beta_prior_prec = rep(0.16, 2))),
    prior = list(list(type = "iid", obs_idx = as.integer(region),
                      n_units = n_group,
                      sigma_grid = exp(seq(log(0.25), log(1.6),
                                           length.out = 9)))),
    control = ctl))
}

test_that("a cila fit runs on a parallel outer grid and matches the serial one", {
  skip_on_cran()
  req <- list(n_points = 512L)
  serial <- cila_par_fit(req, 1L)
  expect_true(is.na(serial$cila$declined))

  for (nt in c(2L, 4L)) {
    par <- cila_par_fit(req, nt)
    expect_true(is.na(par$cila$declined), info = paste("threads", nt))
    expect_equal(par$cila$cell_log_marginal, serial$cila$cell_log_marginal,
                 info = paste("threads", nt))
    expect_equal(par$cila$cell_weights, serial$cila$cell_weights,
                 info = paste("threads", nt))
    expect_equal(par$means, serial$means, info = paste("threads", nt))
  }
})

test_that("the correction still moves the fit on the parallel grid", {
  # Without this the equality above would also hold for a correction that
  # declined on every cell.
  skip_on_cran()
  off <- cila_par_fit(NULL, 4L)
  on  <- cila_par_fit(list(n_points = 512L), 4L)
  expect_false(isTRUE(all.equal(on$cila$cell_weights,
                                on$cila$laplace$weights)))
  expect_equal(on$cila$laplace$weights, off$weights, tolerance = 1e-10)
})

test_that("repeated parallel cila fits are bit-for-bit reproducible", {
  # An allocation from R's heap on a worker thread can corrupt state that only
  # shows up later, so the same fit is run twice in one session.
  skip_on_cran()
  req <- list(n_points = 512L)
  a <- cila_par_fit(req, 4L)
  b <- cila_par_fit(req, 4L)
  expect_identical(a$cila$cell_log_marginal, b$cila$cell_log_marginal)
  expect_identical(a$means, b$means)
  # Force a collection between the two, which is where the protection-stack
  # damage used to surface.
  gc()
  d <- cila_par_fit(req, 4L)
  expect_identical(a$means, d$means)
})
