# Built-in family path through the unified single-point Laplace export
# (cpp_laplace_fit). builtin_family_spec() routes each family's per-observation
# likelihood through grad_hess_for_family / log_lik_for_family, and the spec
# solver (laplace_mode_spec_dense_solve) is now the only inner Newton (B2-live
# retired the family-enum bodies). The weak default prior (sigma_beta = 100,
# tau = 1e-4) is effectively flat at these sample sizes, so the MAP coincides
# with the GLM MLE for every family stats::glm fits natively -- an independent
# reference. Broader multi-family parameter recovery + CI coverage lives in
# test-nested-laplace-recovery.R.

# cpp_laplace_fit MAP beta (the leading p of the mode) vs stats::glm MLE.
expect_glm_match <- function(family, glm_family, y, n_trials, X, phi,
                             tol = 1e-2) {
  p <- ncol(X)
  fit <- tulpa:::cpp_laplace_fit(
    y = y, n = n_trials, X = X,
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = family, phi = phi,
    max_iter = 300L, tol = 1e-11, n_threads = 1L
  )
  beta_hat <- fit$mode[seq_len(p)]
  # glm reference (no intercept term -- X already carries the intercept column).
  ref <- if (family == "binomial") {
    stats::glm(cbind(y, n_trials - y) ~ 0 + X, family = glm_family)
  } else {
    stats::glm(y ~ 0 + X, family = glm_family)
  }
  expect_true(fit$converged, info = paste0(family, ": converged"))
  expect_equal(unname(beta_hat), unname(coef(ref)), tolerance = tol,
               info = paste0(family, ": MAP == glm MLE"))
}

test_that("cpp_laplace_fit MAP matches the glm MLE for GLM-native families", {
  set.seed(2026L)
  N <- 400L
  p <- 3L
  X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
  beta <- c(0.2, -0.3, 0.4)
  eta <- as.numeric(X %*% beta)
  ones <- rep(1L, N)

  expect_glm_match("gaussian", gaussian(),            eta + rnorm(N, 0, 0.7),
                   ones, X, phi = 0.7)
  expect_glm_match("poisson",  poisson(),             rpois(N, exp(eta)),
                   ones, X, phi = 1.0)
  expect_glm_match("binomial", binomial(),            rbinom(N, 10L, plogis(eta)),
                   rep(10L, N), X, phi = 1.0)
  expect_glm_match("gamma",    Gamma(link = "log"),
                   rgamma(N, shape = 2, rate = 2 / exp(eta)), ones, X, phi = 2.0)
})

test_that("cpp_laplace_fit converges to a finite mode for every built-in family", {
  set.seed(2026L)
  N <- 300L
  p <- 3L
  X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
  beta <- c(0.2, -0.3, 0.4)
  eta <- as.numeric(X %*% beta)
  ones <- rep(1L, N)
  clamp01 <- function(v) pmin(pmax(v, 1e-4), 1 - 1e-4)

  cases <- list(
    list(f = "gaussian",         y = eta + rnorm(N, 0, 0.7),              n = ones, phi = 0.7),
    list(f = "poisson",          y = rpois(N, exp(eta)),                  n = ones, phi = 1.0),
    list(f = "binomial",         y = rbinom(N, 10L, plogis(eta)),         n = rep(10L, N), phi = 1.0),
    list(f = "neg_binomial_2",   y = rnbinom(N, mu = exp(eta), size = 2), n = ones, phi = 2.0),
    list(f = "gamma",            y = rgamma(N, 2, 2 / exp(eta)),          n = ones, phi = 2.0),
    list(f = "beta",             y = clamp01(rbeta(N, plogis(eta) * 5, (1 - plogis(eta)) * 5)),
                                                                          n = ones, phi = 5.0),
    list(f = "lognormal",        y = exp(eta + rnorm(N, 0, 0.5)),         n = ones, phi = 0.5),
    list(f = "inverse_gaussian", y = exp(eta) * exp(rnorm(N, 0, 0.3)),    n = ones, phi = 1.0)
  )
  for (cs in cases) {
    fit <- tulpa:::cpp_laplace_fit(
      y = cs$y, n = cs$n, X = X,
      re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
      family = cs$f, phi = cs$phi, max_iter = 300L, tol = 1e-10, n_threads = 1L
    )
    expect_true(fit$converged, info = paste0(cs$f, ": converged"))
    expect_true(all(is.finite(fit$mode)), info = paste0(cs$f, ": finite mode"))
    expect_equal(length(fit$mode), p, info = paste0(cs$f, ": mode length"))
  }
})

test_that("cpp_laplace_fit agrees with cpp_laplace_fit_multi_re on a single iid RE", {
  # Two independent marshalling paths into the one spec solver (the basic
  # single-RE setup vs the multi-term builder); their modes must coincide.
  set.seed(7L)
  N <- 300L
  p <- 2L
  n_re_groups <- 12L
  sigma_re <- 0.5
  X <- cbind(1, rnorm(N))
  beta <- c(0.3, -0.4)
  re_idx <- sample.int(n_re_groups, N, replace = TRUE)
  u <- rnorm(n_re_groups, sd = sigma_re)
  eta <- as.numeric(X %*% beta) + u[re_idx]

  for (fam in c("poisson", "binomial")) {
    nt <- if (fam == "binomial") rep(8L, N) else rep(1L, N)
    y  <- if (fam == "binomial") rbinom(N, 8L, plogis(eta)) else rpois(N, exp(eta))
    basic <- tulpa:::cpp_laplace_fit(
      y = y, n = nt, X = X, re_idx = as.numeric(re_idx),
      n_re_groups = n_re_groups, sigma_re = sigma_re, family = fam, phi = 1.0,
      max_iter = 300L, tol = 1e-11, n_threads = 1L
    )
    multi <- tulpa:::cpp_laplace_fit_multi_re(
      y = y, n = nt, X = X,
      re_idx_list = list(as.integer(re_idx)),
      re_ngroups = n_re_groups,
      re_sigma_list = list(sigma_re),
      family = fam, phi = 1.0, max_iter = 300L, tol = 1e-11, n_threads = 1L
    )
    expect_equal(basic$mode, multi$mode, tolerance = 1e-6,
                 info = paste0(fam, ": basic vs multi_re mode"))
  }
})
