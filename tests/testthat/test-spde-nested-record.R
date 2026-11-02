# A fit_spde() fit integrating (range, sigma) carries the outer-grid record
# every nested-Laplace fit carries (`$nested`: theta_grid, weights,
# log_marginal, modes, per-node field variances), which tulpa_outer_grid(),
# field_effects() and predict() read.

skip_on_cran()

.spde_record_fixture <- local({
  fit <- NULL
  function() {
    if (is.null(fit)) {
      set.seed(2)
      n <- 200
      d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
      sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.1, 0.3))
      d$y <- rpois(n, exp(1 + 0.5 * d$x + sin(3 * d$lon)))
      X <- model.matrix(~ x, d)
      fit <<- list(d = d, sp = sp, X = X,
                   fit = fit_spde(d$y, X, sp, family = "poisson",
                                  control = list(diagnose_k = FALSE)))
    }
    fit
  }
})

test_that("tulpa_outer_grid() reads a fit_spde() fit's (range, sigma) grid", {
  skip_on_cran()
  fx <- .spde_record_fixture()
  g <- tulpa_outer_grid(fx$fit)
  expect_equal(colnames(g$nodes), c("range", "sigma"))
  expect_equal(unname(g$nodes[, "range"]), fx$fit$nested$range_grid)
  expect_equal(sum(g$weights), 1)
  expect_equal(nrow(g$modes), nrow(g$nodes))
  expect_equal(ncol(g$modes), ncol(fx$X) + fx$sp$n_mesh)
  expect_identical(g$n_fixed, ncol(fx$X))
  expect_true(all(is.finite(g$log_marginal)))
})

test_that("field_effects() on a nested SPDE fit is the grid's Gaussian mixture", {
  skip_on_cran()
  fx <- .spde_record_fixture()
  fe <- field_effects(fx$fit)
  expect_identical(unique(fe$source), "mixture")
  expect_equal(nrow(fe), fx$sp$n_mesh)
  expect_true(all(is.finite(fe$sd) & fe$sd > 0))
  rec <- fx$fit$nested
  mesh <- rec$grid_field_var_cols
  w <- rec$weights / sum(rec$weights)
  mu <- rec$modes[, mesh, drop = FALSE]
  expect_equal(fe$estimate, as.numeric(crossprod(w, mu)), tolerance = 1e-10)
  # Law of total variance over the cells.
  tv <- as.numeric(crossprod(w, rec$grid_field_var)) +
    as.numeric(crossprod(w, sweep(mu, 2L, fe$estimate)^2))
  expect_equal(fe$sd, sqrt(tv), tolerance = 1e-8)
})

test_that("the per-node field variance is the exact Gaussian posterior variance", {
  skip_on_cran()
  # Gaussian response at one (range, sigma): the inner Laplace is exact, so the
  # mesh-node variances must equal diag of the inverse joint precision of
  # [beta, w], [X A]'[X A] / phi + blockdiag(tau_beta I, Q).
  set.seed(5)
  n <- 120
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.15, 0.4))
  d$y <- 1 + 0.5 * d$x + sin(3 * d$lon) + rnorm(n, 0, 0.5)
  X <- model.matrix(~ x, d)
  phi <- 0.25; r <- 0.4; s <- 0.8
  res <- tulpa:::cpp_nested_laplace_spde(
    y = d$y, n_trials = rep(1L, n), X = X, re_idx = rep(0, n),
    n_re_groups = 0L, sigma_re = 1, A_x = sp$A_x, A_i = sp$A_i, A_p = sp$A_p,
    n_obs = n, n_mesh = sp$n_mesh, C0_diag = sp$C0_diag, G1_x = sp$G1_x,
    G1_i = sp$G1_i, G1_p = sp$G1_p, range_grid = r, sigma_grid = s, nu = 1,
    family = "gaussian", phi = sqrt(phi), store_Q = TRUE)
  rec <- tulpa:::.spde_nested_record(res, r, s, 0, sp, "grid")
  kappa <- sqrt(8) / r
  Q <- tulpa_spde_precision_Q(sp, kappa, 1 / (sqrt(4 * pi) * kappa * s))
  D <- cbind(X, as.matrix(sp$A))
  P <- crossprod(D) / phi
  P[1:2, 1:2] <- P[1:2, 1:2] + diag(1e-4, 2)
  P[-(1:2), -(1:2)] <- P[-(1:2), -(1:2)] + as.matrix(Q)
  expect_equal(rec$grid_field_var[1, ], unname(diag(solve(P))[-(1:2)]),
               tolerance = 1e-6)
})


test_that("spatial_range() reads a fit_spde() CCD as a moment rule", {
  skip_on_cran()
  fx <- .spde_record_fixture()
  fit <- fx$fit
  rec <- fit$nested
  expect_identical(rec$integration, rec$method)
  sr <- spatial_range(fit)
  ref <- tulpa:::.nested_hyper_summary(rec, c(0.025, 0.975),
                                       transform = tulpa:::.SPATIAL_HYPER_TRANSFORM)
  expect_equal(sr, ref)
  if (identical(rec$method, "ccd")) {
    # A weighted quantile of the nine design nodes stops at the outermost node,
    # about 1.6 posterior SDs out; the moment-rule read reaches past it.
    expect_gt(sr["range", "q97.5"], max(rec$range_grid))
    expect_gt(sr["sigma", "q97.5"], max(rec$sigma_grid))
  }
})
