# Built-in family LikelihoodSpec adapter (L1 of the spec-driven-solver
# unification, see clean_migration.md). builtin_family_spec() routes the
# per-observation likelihood through the same grad_hess_for_family /
# log_lik_for_family closed forms the family-enum reference uses, so the
# spec-driven Laplace mode must equal the family-enum mode (cpp_laplace_fit)
# on identical data. The data need only be in-support for each family; this
# is an equivalence check between two solver scaffolds over the same per-obs
# math, not a parameter-recovery check.

# Cross-check one family: spec-driven mode == family-enum mode.
expect_family_match <- function(family, y, n_trials, X, phi,
                                re_idx = integer(0), n_re_groups = 0L,
                                sigma_re = 1.0, tol = 1e-5) {
  ref <- tulpa:::cpp_laplace_fit(
    y = y, n = n_trials, X = X,
    re_idx = if (n_re_groups > 0L) as.numeric(re_idx) else numeric(0),
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    max_iter = 300L, tol = 1e-11, n_threads = 1L
  )
  spec <- tulpa:::cpp_laplace_spec_test_family(
    y = y, n = n_trials, X = X,
    re_idx = if (n_re_groups > 0L) as.integer(re_idx) else integer(0),
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    sigma_beta = 100, family = family, phi = phi,
    max_iter = 300L, tol = 1e-11, n_threads = 1L
  )
  expect_true(spec$converged, info = paste0(family, ": converged"))
  expect_equal(spec$mode, ref$mode, tolerance = tol,
               info = paste0(family, ": mode"))
}

test_that("builtin_family_spec reproduces the family-enum mode (no RE)", {
  set.seed(2026L)
  N <- 300L
  p <- 3L
  X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
  beta <- c(0.2, -0.3, 0.4)
  eta <- as.numeric(X %*% beta)
  ones <- rep(1L, N)
  clamp01 <- function(v) pmin(pmax(v, 1e-4), 1 - 1e-4)

  expect_family_match("gaussian", eta + rnorm(N, 0, 0.7), ones, X, phi = 0.7)
  expect_family_match("poisson", rpois(N, exp(eta)), ones, X, phi = 1.0)
  expect_family_match("binomial", rbinom(N, 10L, plogis(eta)),
                      rep(10L, N), X, phi = 1.0)
  expect_family_match("neg_binomial_2", rnbinom(N, mu = exp(eta), size = 2),
                      ones, X, phi = 2.0)
  expect_family_match("gamma", rgamma(N, shape = 2, rate = 2 / exp(eta)),
                      ones, X, phi = 2.0)
  expect_family_match("beta",
                      clamp01(rbeta(N, plogis(eta) * 5, (1 - plogis(eta)) * 5)),
                      ones, X, phi = 5.0)
  expect_family_match("lognormal", exp(eta + rnorm(N, 0, 0.5)), ones, X,
                      phi = 0.5)
  expect_family_match("inverse_gaussian", exp(eta) * exp(rnorm(N, 0, 0.3)),
                      ones, X, phi = 1.0)
})

test_that("builtin_family_spec reproduces the family-enum mode (with iid RE)", {
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
  ones <- rep(1L, N)

  expect_family_match("poisson", rpois(N, exp(eta)), ones, X, phi = 1.0,
                      re_idx = re_idx, n_re_groups = n_re_groups,
                      sigma_re = sigma_re)
  expect_family_match("binomial", rbinom(N, 8L, plogis(eta)),
                      rep(8L, N), X, phi = 1.0,
                      re_idx = re_idx, n_re_groups = n_re_groups,
                      sigma_re = sigma_re)
})
