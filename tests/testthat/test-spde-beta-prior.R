# A fixed-effect prior on the SPDE Laplace path, against the exact gaussian
# reference.
#
# For y = X beta + A u + e with beta ~ N(m, diag(s^2)), u ~ N(0, Sigma_u) and
# e ~ N(0, phi I), the marginal of y is N(X m, V) with
# V = A Sigma_u A' + X diag(s^2) X' + phi I, and the joint mode of (beta, u) is
# the posterior mean: beta_hat = m + diag(s^2) X' V^-1 (y - X m), and the
# linear predictor X beta_hat + A u_hat = y - phi V^-1 (y - X m). Every SPDE
# solve (integer and fractional nu, single point and nested cell, fit_spde() and
# tulpa_laplace()) has to reproduce the marginal, the linear predictor and the
# slope under a supplied prior. The integer-nu path reports its mesh field
# centred, with the removed level folded into the intercept, so the intercept
# alone is read through the linear predictor.

skip_on_cran()

.bp_fixture <- function(seed = 5L, n = 70L) {
  set.seed(seed)
  xy <- cbind(runif(n), runif(n))
  x <- rnorm(n)
  X <- cbind(1, x)
  D <- as.matrix(stats::dist(xy))
  u <- as.numeric(t(chol(0.6^2 * exp(-D / 0.2) + diag(1e-9, n))) %*% rnorm(n))
  phi <- 0.1
  list(xy = xy, X = X, phi = phi, n = n,
       y = as.numeric(0.4 - 0.3 * x + u + rnorm(n, 0, sqrt(phi))))
}

# Field covariance at the observations: the exact lumped-FEM Matern field for an
# integer nu, the assembled rational field for a fractional one.
.bp_field_cov <- function(sp, range, sigma) {
  if (tulpa:::.spde_nu_is_fractional(sp$nu)) {
    asm <- tulpa:::.spde_assemble_at(sp, range, sigma)
    return(as.matrix(asm$A_eff %*% Matrix::solve(asm$Q, Matrix::t(asm$A_eff))))
  }
  C0 <- as.numeric(sp$C0_diag)
  keep <- which(C0 > 1e-12)
  C0k <- C0[keep]
  S <- as.matrix(sp$G[keep, keep]) / sqrt(outer(C0k, C0k))
  e <- eigen(S, symmetric = TRUE)
  nu <- sp$nu
  kappa <- sqrt(8 * nu) / range
  tau2inv <- 4 * pi * nu * kappa^(2 * nu) * sigma^2
  W <- e$vectors / sqrt(C0k)
  Ak <- as.matrix(sp$A[, keep])
  Ak %*% (W %*% (t(W) * (kappa^2 + e$values)^(-(nu + 1))) * tau2inv) %*% t(Ak)
}

.bp_exact <- function(fx, sp, range, sigma, m, s) {
  V <- .bp_field_cov(sp, range, sigma) + fx$X %*% (s^2 * t(fx$X)) +
    diag(fx$phi, fx$n)
  r <- fx$y - as.numeric(fx$X %*% m)
  R <- chol(V)
  z <- backsolve(R, r, transpose = TRUE)
  Vir <- backsolve(R, z)
  list(beta = as.numeric(m + s^2 * crossprod(fx$X, Vir)),
       eta = fx$y - fx$phi * Vir,
       log_marginal = -0.5 * sum(z^2) - sum(log(diag(R))) -
         0.5 * fx$n * log(2 * pi))
}

.bp_eta <- function(fx, sp, fit) {
  as.numeric(fx$X %*% fit$beta + sp$A %*% fit$spatial_effects)
}

.bp_m <- c(0.7, -0.2)
.bp_s <- c(0.4, 1.5)

test_that("a supplied beta_prior is the prior of the single-point SPDE solve", {
  fx <- .bp_fixture()
  for (nu in c(1, 0.5)) {
    sp <- spatial_spde(fx$xy, nu = nu, max_edge = c(0.08, 0.3))
    fit <- fit_spde(fx$y, fx$X, sp, family = "gaussian", phi = fx$phi,
                    range = 0.25, sigma = 0.7, nested_laplace = FALSE,
                    beta_prior = list(mean = .bp_m, sd = .bp_s),
                    control = list(tol = 1e-10))
    ex <- .bp_exact(fx, sp, 0.25, 0.7, .bp_m, .bp_s)
    expect_true(fit$converged)
    expect_equal(fit$beta[2], ex$beta[2], tolerance = 1e-6)
    expect_equal(.bp_eta(fx, sp, fit), ex$eta, tolerance = 1e-6)
    expect_equal(fit$log_marginal, ex$log_marginal, tolerance = 1e-6)

    # NULL is the engine default, N(0, 2.5^2) on every coefficient.
    sb <- tulpa:::cpp_test_default_sigma_beta()
    f0 <- fit_spde(fx$y, fx$X, sp, family = "gaussian", phi = fx$phi,
                   range = 0.25, sigma = 0.7, nested_laplace = FALSE,
                   control = list(tol = 1e-10))
    e0 <- .bp_exact(fx, sp, 0.25, 0.7, c(0, 0), c(sb, sb))
    expect_equal(f0$beta[2], e0$beta[2], tolerance = 1e-6)
    expect_equal(.bp_eta(fx, sp, f0), e0$eta, tolerance = 1e-6)
    expect_equal(f0$log_marginal, e0$log_marginal, tolerance = 1e-6)
  }
})

test_that("tulpa_laplace() carries beta_prior onto an SPDE field and refuses it elsewhere", {
  fx <- .bp_fixture()
  sp <- spatial_spde(fx$xy, nu = 1, max_edge = c(0.08, 0.3))
  bp <- list(mean = .bp_m, sd = .bp_s)
  tl <- tulpa_laplace(y = fx$y, n_trials = rep(1L, fx$n), X = fx$X,
                      family = "gaussian", phi = fx$phi, spatial = sp,
                      beta_prior = bp, tol = 1e-10)
  ex <- .bp_exact(fx, sp, sp$prior_range[1], sp$prior_sigma[1], .bp_m, .bp_s)
  expect_equal(tl$mode[2], ex$beta[2], tolerance = 1e-6)
  expect_equal(tl$log_marginal, ex$log_marginal, tolerance = 1e-6)

  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- adj[2, 3] <- adj[3, 2] <- 1
  icar <- list(type = "icar", adjacency = adj, spatial_idx = 1:4)
  expect_error(
    tulpa_laplace(y = c(0L, 1L, 1L, 0L), n_trials = rep(1L, 4),
                  X = cbind(1, c(0.1, -0.3, 0.2, 0.5)), family = "binomial",
                  spatial = icar, beta_prior = list(mean = 0, sd = 1)),
    "not supported on the spatial Laplace path for a 'icar' field")
})

test_that("every nested SPDE cell solves under a supplied beta_prior", {
  fx <- .bp_fixture()
  sp <- spatial_spde(fx$xy, nu = 1, max_edge = c(0.08, 0.3))
  ranges <- c(0.1, 0.3); sigmas <- c(0.5, 1)
  res <- tulpa:::cpp_nested_laplace_spde(
    y = fx$y, n_trials = rep(1L, fx$n), X = fx$X,
    re_idx = rep(0L, fx$n), n_re_groups = 0L, sigma_re = 1,
    A_x = sp$A_x, A_i = sp$A_i, A_p = sp$A_p,
    n_obs = fx$n, n_mesh = sp$n_mesh, C0_diag = sp$C0_diag,
    G1_x = sp$G1_x, G1_i = sp$G1_i, G1_p = sp$G1_p,
    range_grid = ranges, sigma_grid = sigmas, nu = 1,
    family = "gaussian", phi = sqrt(fx$phi), tol = 1e-10,
    beta_prior = list(mean = .bp_m, sd = .bp_s))
  for (k in seq_along(ranges)) {
    ex <- .bp_exact(fx, sp, ranges[k], sigmas[k], .bp_m, .bp_s)
    expect_equal(res$log_marginal[k], ex$log_marginal, tolerance = 1e-6)
  }
})

test_that("tulpa() hands beta_prior to the SPDE path and reports it", {
  fx <- .bp_fixture()
  sp <- spatial_spde(fx$xy, nu = 1, max_edge = c(0.08, 0.3))
  df <- data.frame(y = fx$y, x = fx$X[, 2])
  bp <- list(mean = 0, sd = 0.05)
  fit <- suppressWarnings(
    tulpa(y ~ x, data = df, family = "gaussian", phi = fx$phi, spatial = sp,
          beta_prior = bp, control = list(diagnose_k = FALSE)))
  expect_identical(fit$beta_prior$sd, 0.05)
  # The front door runs fit_spde() under the prior it was handed.
  fs <- fit_spde(fx$y, model.matrix(~ x, df), sp, family = "gaussian",
                 phi = fx$phi, beta_prior = bp,
                 control = list(diagnose_k = FALSE))
  expect_equal(unname(coef(fit)), unname(coef(fs)), tolerance = 1e-8)
  # and the prior binds: the tight prior shrinks the slope the default leaves.
  f0 <- suppressWarnings(
    tulpa(y ~ x, data = df, family = "gaussian", phi = fx$phi, spatial = sp,
          control = list(diagnose_k = FALSE)))
  expect_lt(abs(coef(fit)[["x"]]), 0.5 * abs(coef(f0)[["x"]]))
})
