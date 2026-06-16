# test-nested-laplace-occupancy.R
# Recovery and calibration tests for the marginalized single-season occupancy
# likelihood routed through the multi-block nested-Laplace dispatch as a
# model-supplied LikelihoodSpec (dev_notes/plans/clean_migration.md L5). tulpa no longer ships
# the occupancy family; the scaled Bernoulli is built by the reference C++
# harness cpp_nested_laplace_test_occupancy_likelihood() and passed via the
# `likelihood =` external pointer -- the same path (and very nearly the same
# code) tulpaObs uses.
#
# Model. Each site i contributes one Bernoulli on the detection indicator
# D_i = 1{>= 1 detection}, with the latent occupancy state integrated out:
#     D_i ~ Bernoulli(mu_i),   mu_i = q_i * sigma(eta_i),
# where sigma(eta_i) = psi_i is the occupancy probability and
#     q_i = 1 - (1 - p)^{J_i}
# is the per-site probability of at least one detection given occupancy (J_i
# visits, detection prob p). q_i is supplied as `det_prob`; q_i = 0 (no visits)
# drops the site from the likelihood but keeps its latent value (held-out), and
# q_i = 1 reduces the likelihood to a standard logit Bernoulli. Because the
# occupancy state is marginalized analytically, the converged Hessian carries
# the expected (Fisher) information q*sigma*(1-sigma)^2/(1-q*sigma) -- the true
# marginal curvature -- so the per-row predictive variance fitted_eta_var is
# calibrated directly, with no post-hoc rescaling.

# A region-grouped IID latent keeps the random effect identified (several sites
# per region) while leaving each site one detection observation, as in a real
# single-season design.
sim_occu <- function(seed, n_regions = 40L, sites_per_region = 5L,
                     beta = c(-0.1, 0.8), sigma_u = 0.6, p_det = 0.4) {
  set.seed(seed)
  n_sites <- n_regions * sites_per_region
  region  <- rep(seq_len(n_regions), each = sites_per_region)
  x       <- rnorm(n_sites)
  X       <- cbind(1, x)
  u       <- rnorm(n_regions, 0, sigma_u)
  eta     <- as.numeric(X %*% beta) + u[region]
  psi     <- plogis(eta)
  J       <- sample.int(4L, n_sites, replace = TRUE)   # 1..4 visits per site
  q       <- 1 - (1 - p_det)^J                          # per-site >=1 detection
  z       <- rbinom(n_sites, 1, psi)                    # true occupancy
  D       <- z * rbinom(n_sites, 1, q)                  # detected iff occupied & visited
  list(y = as.integer(D), X = X, det_prob = q,
       region = as.integer(region), n_regions = as.integer(n_regions),
       eta_true = eta, psi = psi, q = q, n_sites = n_sites)
}

# Fit through the model-supplied occupancy LikelihoodSpec (the L5 path):
# build the scaled-Bernoulli spec from (D_i, q_i) and pass it as `likelihood`.
fit_occu <- function(d, det_prob = d$det_prob, sigma_grid = c(0.3, 0.6, 1.0)) {
  prior <- list(list(type = "iid", obs_idx = d$region,
                     n_units = d$n_regions, sigma_grid = sigma_grid))
  lik <- tulpa:::cpp_nested_laplace_test_occupancy_likelihood(
    y = as.numeric(d$y), det_prob = as.numeric(det_prob))
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$n_sites), X = d$X,
    prior = prior, likelihood = lik,
    control = list(max_iter = 50L, tol = 1e-7, n_threads = 1L)
  ))
}

# Built-in binomial fit (n_trials = 1) -- the reference the q = 1 occupancy
# spec must reproduce exactly.
fit_binom <- function(d, sigma_grid = c(0.3, 0.6, 1.0)) {
  prior <- list(list(type = "iid", obs_idx = d$region,
                     n_units = d$n_regions, sigma_grid = sigma_grid))
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$n_sites), X = d$X,
    prior = prior, family = "binomial",
    control = list(max_iter = 50L, tol = 1e-7, n_threads = 1L)
  ))
}

# Marginal posterior of each eta_i as a Gaussian mixture over the grid:
# component k is N(fitted_eta[k,i], fitted_eta_var[k,i]) with weight w_k. Returns
# the mixture mean and the law-of-total-variance variance per row.
post_eta_moments <- function(fit) {
  w  <- fit$weights
  fe <- fit$fitted_eta
  fv <- fit$fitted_eta_var
  m  <- as.numeric(crossprod(w, fe))
  v  <- pmax(0, as.numeric(crossprod(w, fv + fe^2)) - m^2)
  list(mean = m, var = v)
}

eta_coverage <- function(fit, eta_true, keep = NULL, level = 0.95) {
  mom <- post_eta_moments(fit)
  if (is.null(keep)) keep <- rep(TRUE, length(eta_true))
  z  <- stats::qnorm(1 - (1 - level) / 2)
  lo <- mom$mean - z * sqrt(mom$var)
  hi <- mom$mean + z * sqrt(mom$var)
  mean(eta_true[keep] >= lo[keep] & eta_true[keep] <= hi[keep])
}

test_that("occupancy fit exposes calibrated fitted_eta / fitted_eta_var", {
  d  <- sim_occu(seed = 21L)
  f  <- fit_occu(d)
  ng <- nrow(f$theta_grid)

  expect_s3_class(f, "tulpa_nested_laplace")
  expect_true(is.matrix(f$fitted_eta))
  expect_true(is.matrix(f$fitted_eta_var))
  expect_equal(dim(f$fitted_eta), c(ng, d$n_sites))
  expect_equal(dim(f$fitted_eta_var), c(ng, d$n_sites))
  expect_true(all(is.finite(f$fitted_eta)))
  expect_true(all(is.finite(f$fitted_eta_var)))
  expect_true(all(f$fitted_eta_var >= 0))
  expect_true(all(is.finite(f$log_marginal)))
})

test_that("occupancy with det_prob == 1 reduces to a logit Bernoulli", {
  d  <- sim_occu(seed = 22L)
  fo <- fit_occu(d, det_prob = rep(1, d$n_sites))
  fb <- fit_binom(d)

  # q = 1 makes mu = sigma(eta): the model spec and the built-in binomial must
  # agree on mode, evidence, and curvature.
  expect_equal(fo$log_marginal, fb$log_marginal, tolerance = 1e-7)
  expect_equal(fo$fitted_eta, fb$fitted_eta, tolerance = 1e-7)
  expect_equal(fo$fitted_eta_var, fb$fitted_eta_var, tolerance = 1e-6)
})

test_that("occupancy recovers occupancy above the raw detection rate", {
  d   <- sim_occu(seed = 23L, n_regions = 60L)
  f   <- fit_occu(d)
  mom <- post_eta_moments(f)

  # The marginalized fit reconstructs the latent occupancy predictor.
  expect_gt(stats::cor(mom$mean, d$eta_true), 0.6)

  # Occupancy correction: because E[D_i] = q_i * psi_i with q_i < 1, the fitted
  # occupancy probability must sit above the raw detection rate and track the
  # true mean occupancy -- the whole point of separating detection from state.
  psi_hat <- plogis(mom$mean)
  expect_gt(mean(psi_hat), mean(d$y) + 0.05)
  expect_lt(abs(mean(psi_hat) - mean(d$psi)), 0.12)
})

test_that("held-out (det_prob = 0) sites get finite prior-driven predictions", {
  d <- sim_occu(seed = 24L)
  # Mark one site per region as unvisited: q = 0, no detection possible.
  held <- which(!duplicated(d$region))
  d$det_prob[held] <- 0
  d$y[held] <- 0L

  f <- fit_occu(d)
  expect_true(all(is.finite(f$fitted_eta[, held])))
  expect_true(all(is.finite(f$fitted_eta_var[, held])))
  expect_true(all(f$fitted_eta_var[, held] >= 0))
  # An unvisited site carries no likelihood information of its own, so its
  # predictive variance is at least as large as a typical visited site's.
  expect_gte(median(f$fitted_eta_var[, held]),
             median(f$fitted_eta_var[, -held]) * 0.95)
})

test_that("occupancy predictive intervals reach nominal coverage", {
  skip_if_not_slow()
  if (!isTRUE(as.logical(Sys.getenv("TULPA_SLOW_TESTS", "false")))) {
    skip("Slow 20-seed coverage test. Set TULPA_SLOW_TESTS=true to run.")
  }

  n_seeds <- 20L
  cov <- numeric(n_seeds)
  for (s in seq_len(n_seeds)) {
    d <- sim_occu(seed = 300L + s)
    f <- fit_occu(d)
    cov[s] <- eta_coverage(f, d$eta_true, keep = d$q > 0)
  }
  # The marginal occupancy curvature should give near-nominal 95% coverage of
  # the realized linear predictor. Allow Laplace + coarse-grid slack on the
  # lower side; flag gross over-coverage too.
  expect_gte(mean(cov), 0.86)
  expect_lte(mean(cov), 0.99)
})
