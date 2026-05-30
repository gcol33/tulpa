# Pareto-smoothed importance sampling: the native PSIS core, its agreement with
# loo, and the outer Pareto-k-hat diagnostic for the nested-Laplace path.

test_that("tulpa_psis reproduces loo::psis pareto_k", {
  skip_if_not_installed("loo")
  set.seed(42)

  cases <- list(
    light = rnorm(3000),                 # lognormal weights: all moments finite
    heavy = log(1 / runif(3000)),        # Pareto(1) weights: k-hat ~ 1
    mixed = rnorm(2000) + rexp(2000)     # moderately heavy
  )
  for (nm in names(cases)) {
    lr  <- cases[[nm]]
    ours <- tulpa_psis(lr)$pareto_k
    ref  <- suppressWarnings(
      loo::psis(matrix(lr, ncol = 1), r_eff = NA)$diagnostics$pareto_k)
    expect_equal(ours, ref, tolerance = 0.02,
                 info = sprintf("case '%s': ours=%.4f loo=%.4f", nm, ours, ref))
  }
})

test_that("pareto_k separates light from heavy tails; is_ess is bounded", {
  set.seed(1)
  light <- tulpa_psis(rnorm(3000))
  heavy <- tulpa_psis(log(1 / runif(3000)))

  expect_lt(light$pareto_k, 0.5)         # finite-moment proposal: reliable
  expect_gt(heavy$pareto_k, 0.7)         # Pareto(1): not correctable
  expect_lt(light$is_ess, 3000 + 1e-6)   # ESS never exceeds the sample size
  expect_gt(light$is_ess, 0)
  expect_lt(heavy$is_ess, light$is_ess)  # heavy weights concentrate -> lower ESS
})

test_that(".nested_outer_pareto_k is low when the proposal covers the target", {
  set.seed(7)
  # Gaussian target N(mu, Sigma); proposal slightly wider (1.5 * Sigma) so the
  # importance weights are bounded -> small / negative k-hat, high IS-ESS.
  mu    <- c(0.4, -0.2)
  Sigma <- matrix(c(1, 0.3, 0.3, 0.5), 2)
  Sinv  <- solve(Sigma)
  log_target <- function(th) {
    d <- th - mu
    -0.5 * as.numeric(t(d) %*% Sinv %*% d)            # up to a constant
  }
  L_scale <- t(chol(1.5 * Sigma))                     # proposal scale > target

  kd <- .nested_outer_pareto_k(log_target, theta_hat = mu,
                               L_scale = L_scale, n_samples = 2000L)
  expect_true(is.finite(kd$pareto_k))
  expect_lt(kd$pareto_k, 0.5)
  expect_gt(kd$is_ess, 0)
  expect_equal(kd$n_eval, 2000L)
})

test_that(".nested_outer_pareto_k rises when the target is heavier than the proposal", {
  set.seed(9)
  mu <- c(0, 0)
  # Student-t(df = 2) target (infinite variance) against a unit-Gaussian
  # proposal of matching scale: tails the Gaussian cannot cover -> high k-hat.
  log_target_t <- function(th) sum(stats::dt(th - mu, df = 2, log = TRUE))
  kd <- .nested_outer_pareto_k(log_target_t, theta_hat = mu,
                               L_scale = diag(1, 2), n_samples = 4000L)
  expect_gt(kd$pareto_k, 0.5)                          # heavier target -> not correctable
})

test_that("tulpa_re_cov_nested reports a Pareto-k-hat without disturbing draws", {
  skip_on_cran()
  sim <- function(seed, G = 50L, npg = 10L) {
    set.seed(seed)
    N <- G * npg; grp <- rep(seq_len(G), each = npg)
    x <- rnorm(N); X <- cbind(1, x); Z <- cbind(1, x)
    Sig <- matrix(c(0.64, 0.24, 0.24, 0.36), 2)
    u <- t(t(chol(Sig)) %*% matrix(rnorm(2 * G), 2))
    eta <- as.numeric(X %*% c(-0.3, 0.7)) + rowSums(Z * u[grp, ])
    list(y = rbinom(N, 1L, plogis(eta)), X = X, Z = Z, grp = grp, G = G, N = N)
  }
  d  <- sim(11L)
  rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)

  fit <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             seed = 11L, k_samples = 200L)
  # k-hat is a finite, data-dependent reading (a sparse binary RE-covariance
  # posterior is skewed, so a HIGH k-hat here is a correct signal, not a bug --
  # the estimator's correctness is pinned by the loo-equivalence test above and
  # the synthetic-target helper test). Assert the plumbing and ESS range only.
  expect_true(is.finite(fit$pareto_k))
  expect_true(is.finite(fit$pareto_k_is_ess))
  expect_gt(fit$pareto_k_is_ess, 0)
  expect_lte(fit$pareto_k_is_ess, 200 + 1e-6)
  expect_equal(fit$pareto_k_scope, "outer (hyperparameter) Gaussian proposal")

  # diagnose_k must not perturb the fixed-effect draws (RNG state restored).
  off <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             seed = 11L, diagnose_k = FALSE)
  on  <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             seed = 11L, diagnose_k = TRUE, k_samples = 150L)
  expect_true(is.na(off$pareto_k))
  expect_equal(off$draws, on$draws)
})

test_that("tulpa_nested_laplace reports an outer k-hat for a positive-scale block", {
  skip_on_cran()
  set.seed(3)
  nr <- 60L; spr <- 10L; N <- nr * spr
  region <- rep(seq_len(nr), each = spr)
  X <- cbind(1, rnorm(N))
  u <- rnorm(nr, 0, 0.5)
  y <- rbinom(N, 10L, plogis(as.numeric(X %*% c(-0.2, 0.7)) + u[region]))
  sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
  mk <- function(grid) list(list(type = "iid", obs_idx = region,
                                 n_units = nr, sigma_grid = grid))

  fit <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(10L, N), X = X, prior = mk(sg),
    family = "binomial", phi = 1,
    control = list(max_iter = 100L, tol = 1e-8, k_samples = 200L)))
  # Single positive-scale (RE-SD) axis: k-hat is computed via the log transform.
  # The value is data-dependent (a sparse binary RE-SD posterior is skewed, so a
  # high reading is correct) -- assert plumbing + ESS range, as in the re_cov case.
  expect_true(is.finite(fit$pareto_k))
  expect_true(is.finite(fit$pareto_k_is_ess))
  expect_gt(fit$pareto_k_is_ess, 0)
  expect_lte(fit$pareto_k_is_ess, 200 + 1e-6)
  expect_equal(fit$pareto_k_scope, "outer (hyperparameter) Gaussian proposal")

  off <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(10L, N), X = X, prior = mk(sg),
    family = "binomial", phi = 1,
    control = list(max_iter = 100L, tol = 1e-8, diagnose_k = FALSE)))
  expect_true(is.na(off$pareto_k))                  # gated off
})

test_that("outer k-hat declines (NA) for a multi-block nested fit", {
  skip_on_cran()
  set.seed(4)
  nr <- 40L; spr <- 8L; N <- nr * spr
  region  <- rep(seq_len(nr), each = spr)
  region2 <- rep(seq_len(nr), times = spr)
  X <- cbind(1, rnorm(N))
  y <- rbinom(N, 5L, plogis(as.numeric(X %*% c(-0.1, 0.5))))
  sg <- exp(seq(log(0.2), log(1.2), length.out = 5))
  prior <- list(
    list(type = "iid", obs_idx = region,  n_units = nr, sigma_grid = sg),
    list(type = "iid", obs_idx = region2, n_units = nr, sigma_grid = sg))
  fit <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(5L, N), X = X, prior = prior,
    family = "binomial", phi = 1, control = list(max_iter = 80L, tol = 1e-7)))
  expect_true(is.na(fit$pareto_k))                  # multi-block: declined, not guessed
  expect_gt(length(fit$weights), 1L)                # quadrature-ESS fallback available
})
