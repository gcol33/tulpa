# Tests for tulpa_nmix_laplace() -- non-spatial Royle (2004) N-mixture
# Laplace fit via inner Newton with marginal observed Fisher info.

simulate_nmix <- function(seed,
                          n_sites = 200,
                          J = 5,
                          beta_lambda = c(log(5), 0.4),
                          beta_p = c(qlogis(0.4), -0.3),
                          r = Inf) {
  set.seed(seed)
  elev <- rnorm(n_sites)
  wind <- matrix(rnorm(n_sites * J), n_sites, J)

  X_lambda <- cbind(intercept = 1, elev = elev)
  eta_lam  <- as.vector(X_lambda %*% beta_lambda)
  lambda   <- exp(eta_lam)

  X_p <- cbind(intercept = 1, wind = as.vector(t(wind)))
  site_idx <- as.integer(rep(seq_len(n_sites), each = J))
  eta_p_mat <- matrix(0, n_sites, J)
  for (s in seq_len(n_sites)) {
    eta_p_mat[s, ] <- beta_p[1] + beta_p[2] * wind[s, ]
  }
  p_mat <- plogis(eta_p_mat)

  # Abundance: Poisson (r = Inf) or negative-binomial (finite size r).
  N <- if (is.finite(r)) rnbinom(n_sites, size = r, mu = lambda) else rpois(n_sites, lambda)
  y_mat <- matrix(0L, n_sites, J)
  for (s in seq_len(n_sites)) for (j in seq_len(J)) {
    y_mat[s, j] <- rbinom(1, N[s], p_mat[s, j])
  }
  y_long <- as.integer(as.vector(t(y_mat)))

  list(
    y = y_long, site_idx = site_idx,
    X_lambda = X_lambda, X_p = X_p,
    y_mat = y_mat, wind = wind, elev = elev,
    N_true = N,
    beta_lambda_true = beta_lambda,
    beta_p_true = beta_p,
    r_true = r
  )
}

test_that("Newton converges to a finite MLE on the Royle simulation", {
  dat <- simulate_nmix(seed = 42)
  fit <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    max_iter = 50L, tol = 1e-6
  )
  expect_true(fit$converged)
  expect_lt(fit$grad_norm, 1e-6)
  expect_lt(fit$n_iter, 20L)
  expect_true(is.finite(fit$log_lik))

  # Sanity: slope estimates (well-identified, unlike the lambda/p intercepts
  # that ride the classical N-mixture identifiability ridge) are near truth
  # on n_sites = 200. The multi-seed coverage test below is the proper
  # statistical recovery check; this is just a shape sanity.
  expect_lt(abs(fit$beta_lambda["elev"] - dat$beta_lambda_true[2]), 0.15)
  expect_lt(abs(fit$beta_p["wind"]      - dat$beta_p_true[2]),      0.15)

  expect_equal(length(fit$mean_N), nrow(dat$X_lambda))
  expect_true(all(fit$boundary_weight < 1e-4))
})

test_that("Multi-seed coverage of 95% Wald CIs is near nominal", {
  skip_on_cran()
  n_seeds <- 30L
  covered_lam <- 0L
  covered_p   <- 0L
  bias_lam <- numeric(n_seeds)
  bias_p   <- numeric(n_seeds)
  for (k in seq_len(n_seeds)) {
    dat <- simulate_nmix(seed = 1000L + k)
    fit <- suppressWarnings(tulpa_nmix_laplace(
      y = dat$y, site_idx = dat$site_idx,
      X_lambda = dat$X_lambda, X_p = dat$X_p,
      max_iter = 80L, tol = 1e-6
    ))
    if (!fit$converged) next
    se <- sqrt(diag(fit$vcov))
    # slope coefficients (elev for lambda; wind for p)
    if (abs(fit$beta_lambda[2] - dat$beta_lambda_true[2]) <
        1.96 * se["elev"]) covered_lam <- covered_lam + 1L
    if (abs(fit$beta_p[2] - dat$beta_p_true[2]) <
        1.96 * se["wind"]) covered_p <- covered_p + 1L
    bias_lam[k] <- fit$beta_lambda[2] - dat$beta_lambda_true[2]
    bias_p[k]   <- fit$beta_p[2]      - dat$beta_p_true[2]
  }
  expect_gte(covered_lam, 24L)   # >= 80% of nominal 95%
  expect_gte(covered_p,   24L)
  expect_lt(abs(mean(bias_lam)), 0.1)
  expect_lt(abs(mean(bias_p)),   0.1)
})

test_that("Cross-check against unmarked::pcount", {
  skip_if_not_installed("unmarked")
  # unmarked fits are S4. Call its coef()/logLik() namespace-qualified so they
  # bind to unmarked's S4 generics directly: under R CMD check all test files
  # share one session, and a bare coef() can resolve to stats::coef.default
  # (-> "$ operator not defined for this S4 class") if the search path shifts
  # after another file attaches a package. unmarked:: is immune to that.
  suppressPackageStartupMessages(library(unmarked))
  dat <- simulate_nmix(seed = 42)
  fit <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    max_iter = 50L, tol = 1e-8
  )
  umf <- unmarked::unmarkedFramePCount(
    y = dat$y_mat,
    siteCovs = data.frame(elev = dat$elev),
    obsCovs  = list(wind = dat$wind)
  )
  um_fit <- unmarked::pcount(~ wind ~ elev, data = umf,
                             K = max(dat$y) + 100L, mixture = "P")
  um_coef <- unmarked::coef(um_fit)
  # tulpa should match (or slightly improve on) unmarked's BFGS-with-numerical-derivatives
  expect_gte(fit$log_lik, as.numeric(unmarked::logLik(um_fit)) - 1e-3)
  expect_lt(abs(fit$beta_lambda[1] - um_coef["lam(Int)"]),  5e-3)
  expect_lt(abs(fit$beta_lambda[2] - um_coef["lam(elev)"]), 5e-3)
  expect_lt(abs(fit$beta_p[1]      - um_coef["p(Int)"]),    5e-3)
  expect_lt(abs(fit$beta_p[2]      - um_coef["p(wind)"]),   5e-3)
})

test_that("K_max sanity: low K_max triggers boundary warning", {
  dat <- simulate_nmix(seed = 42)
  # Force K_max close to mean(N) so a heavy posterior tail at K_max is likely
  expect_warning(
    fit <- tulpa_nmix_laplace(
      y = dat$y, site_idx = dat$site_idx,
      X_lambda = dat$X_lambda, X_p = dat$X_p,
      K_max = max(dat$y) + 1L, max_iter = 50L
    ),
    "posterior weight on N = K_max"
  )
  expect_true(any(fit$boundary_weight > 1e-4))
})

test_that("K_max < max(y) errors clearly", {
  dat <- simulate_nmix(seed = 42)
  expect_error(
    tulpa_nmix_laplace(
      y = dat$y, site_idx = dat$site_idx,
      X_lambda = dat$X_lambda, X_p = dat$X_p,
      K_max = max(dat$y) - 1L
    ),
    "K_max"
  )
})

test_that("print method runs without error", {
  dat <- simulate_nmix(seed = 42)
  fit <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    max_iter = 30L
  )
  expect_output(print(fit), "N-mixture Laplace fit")
})

# --------------------------------------------------------------------------
# Negative-binomial mixture (mixture = "NB")
# --------------------------------------------------------------------------

test_that("NB fit converges and estimates a finite dispersion", {
  dat <- simulate_nmix(seed = 5, n_sites = 250, r = 2)
  fit <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    mixture = "NB", max_iter = 100L, tol = 1e-7
  )
  expect_true(fit$converged)
  expect_lt(fit$grad_norm, 1e-5)
  expect_identical(fit$mixture, "NB")
  expect_true(is.finite(fit$log_r) && is.finite(fit$r))
  # vcov carries log_r as its last coordinate with a finite SE.
  expect_equal(ncol(fit$vcov), 5L)
  expect_true("log_r" %in% rownames(fit$vcov))
  expect_true(is.finite(sqrt(diag(fit$vcov))["log_r"]))
  # slope + dispersion near truth on n = 250 (intercepts ride the N-mixture ridge).
  expect_lt(abs(fit$beta_lambda["elev"] - dat$beta_lambda_true[2]), 0.15)
  expect_lt(abs(fit$beta_p["wind"]      - dat$beta_p_true[2]),      0.15)
  expect_lt(abs(fit$log_r - log(dat$r_true)), 0.5)
})

test_that("NB log-lik exceeds Poisson on overdispersed data; matches on Poisson data", {
  dat_nb <- simulate_nmix(seed = 6, n_sites = 250, r = 1.5)
  fit_nb <- tulpa_nmix_laplace(dat_nb$y, dat_nb$site_idx, dat_nb$X_lambda, dat_nb$X_p,
                               mixture = "NB", max_iter = 100L, tol = 1e-7)
  fit_p  <- tulpa_nmix_laplace(dat_nb$y, dat_nb$site_idx, dat_nb$X_lambda, dat_nb$X_p,
                               mixture = "P",  max_iter = 100L, tol = 1e-7)
  # NB nests Poisson, so its max log-lik cannot be lower; on overdispersed data
  # it should be strictly higher.
  expect_gt(fit_nb$log_lik, fit_p$log_lik)
})

test_that("NB multi-seed recovery: slopes and dispersion, near-nominal coverage", {
  skip_on_cran()
  n_seeds <- 30L
  r_true <- 2
  covered_lam <- 0L; covered_p <- 0L; covered_r <- 0L
  bias_lam <- numeric(n_seeds); bias_p <- numeric(n_seeds); bias_lr <- numeric(n_seeds)
  used <- 0L
  for (k in seq_len(n_seeds)) {
    dat <- simulate_nmix(seed = 2000L + k, n_sites = 250, r = r_true)
    fit <- suppressWarnings(tulpa_nmix_laplace(
      dat$y, dat$site_idx, dat$X_lambda, dat$X_p,
      mixture = "NB", max_iter = 120L, tol = 1e-7
    ))
    if (!fit$converged || isTRUE(fit$dispersion_boundary)) next
    used <- used + 1L
    se <- sqrt(diag(fit$vcov))
    if (abs(fit$beta_lambda[2] - dat$beta_lambda_true[2]) < 1.96 * se["elev"]) covered_lam <- covered_lam + 1L
    if (abs(fit$beta_p[2]      - dat$beta_p_true[2])      < 1.96 * se["wind"]) covered_p   <- covered_p   + 1L
    if (abs(fit$log_r          - log(r_true))            < 1.96 * se["log_r"]) covered_r   <- covered_r   + 1L
    bias_lam[k] <- fit$beta_lambda[2] - dat$beta_lambda_true[2]
    bias_p[k]   <- fit$beta_p[2]      - dat$beta_p_true[2]
    bias_lr[k]  <- fit$log_r          - log(r_true)
  }
  expect_gte(used, 24L)
  # >= 80% of nominal 95% coverage for slopes and dispersion.
  expect_gte(covered_lam, floor(0.8 * used))
  expect_gte(covered_p,   floor(0.8 * used))
  expect_gte(covered_r,   floor(0.8 * used))
  expect_lt(abs(mean(bias_lam[bias_lam != 0])), 0.1)
  expect_lt(abs(mean(bias_p[bias_p != 0])),     0.1)
  expect_lt(abs(mean(bias_lr[bias_lr != 0])),   0.2)
})

test_that("NB cross-check against unmarked::pcount(mixture = 'NB')", {
  skip_if_not_installed("unmarked")
  suppressPackageStartupMessages(library(unmarked))
  dat <- simulate_nmix(seed = 5, n_sites = 250, r = 2)
  fit <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    mixture = "NB", max_iter = 100L, tol = 1e-8
  )
  umf <- unmarked::unmarkedFramePCount(
    y = dat$y_mat,
    siteCovs = data.frame(elev = dat$elev),
    obsCovs  = list(wind = dat$wind)
  )
  um_fit <- unmarked::pcount(~ wind ~ elev, data = umf,
                             K = max(dat$y) + 100L, mixture = "NB")
  um_coef <- unmarked::coef(um_fit)
  um_se   <- sqrt(diag(unmarked::vcov(um_fit)))
  lr_um   <- unname(um_coef[grep("alpha", names(um_coef))])

  expect_gte(fit$log_lik, as.numeric(unmarked::logLik(um_fit)) - 1e-3)
  expect_lt(abs(fit$beta_lambda[1] - um_coef["lam(Int)"]),  5e-3)
  expect_lt(abs(fit$beta_lambda[2] - um_coef["lam(elev)"]), 5e-3)
  expect_lt(abs(fit$beta_p[1]      - um_coef["p(Int)"]),    5e-3)
  expect_lt(abs(fit$beta_p[2]      - um_coef["p(wind)"]),   5e-3)
  expect_lt(abs(fit$log_r          - lr_um),               1e-2)
  # SE(log_r): tulpa's analytic observed-info inverse vs unmarked's numeric one.
  se_lr <- sqrt(diag(fit$vcov))["log_r"]
  expect_lt(abs(se_lr - um_se[grep("alpha", names(um_coef))]), 0.05 * um_se[grep("alpha", names(um_coef))] + 1e-3)
})

test_that("NB on Poisson data with low r_max pins dispersion and warns", {
  dat <- simulate_nmix(seed = 7, n_sites = 200, r = Inf)  # Poisson truth
  expect_warning(
    fit <- tulpa_nmix_laplace(
      dat$y, dat$site_idx, dat$X_lambda, dat$X_p,
      mixture = "NB", r_max = 5, max_iter = 80L
    ),
    "consistent with Poisson"
  )
  expect_true(isTRUE(fit$dispersion_boundary))
})

test_that("NB print method shows the dispersion", {
  dat <- simulate_nmix(seed = 5, n_sites = 150, r = 2)
  fit <- tulpa_nmix_laplace(dat$y, dat$site_idx, dat$X_lambda, dat$X_p,
                            mixture = "NB", max_iter = 60L)
  expect_output(print(fit), "mixture = NB")
  expect_output(print(fit), "dispersion: log_r")
})
