# Negative-binomial community N-mixture: end-to-end recovery via the joint_grad
# debias path (the analytic gradient now carries the global log_r dispersion as
# an extra fixed effect). The gradient correctness itself is locked in
# test-aghq-gradient.R; this confirms the wired pipeline recovers sensible
# community means / covariances / dispersion from simulated NB truth.

.sim_nb_community <- function(seed, S, R, J, mu_l, mu_p, sd_l, sd_p, r_true) {
  set.seed(seed)
  X_lambda <- matrix(1, R, 1L)
  y <- integer(0); site_idx <- integer(0); species_idx <- integer(0)
  for (s in seq_len(S)) {
    lam_s <- exp(mu_l + stats::rnorm(1, 0, sd_l))
    p_s   <- stats::plogis(mu_p + stats::rnorm(1, 0, sd_p))
    for (i in seq_len(R)) {
      N <- stats::rnbinom(1, size = r_true, mu = lam_s)
      for (j in seq_len(J)) {
        y           <- c(y, stats::rbinom(1, N, p_s))
        site_idx    <- c(site_idx, i)
        species_idx <- c(species_idx, s)
      }
    }
  }
  list(y = y, site_idx = site_idx, species_idx = species_idx,
       X_lambda = X_lambda, X_p = matrix(1, length(y), 1L), R = R, S = S)
}

test_that("NB community N-mixture recovers community means / variances / dispersion", {
  skip_on_cran()
  mu_l <- 1.2; mu_p <- 0.3; sd_l <- 0.5; sd_p <- 0.4; r_true <- 8
  d <- .sim_nb_community(seed = 321L, S = 14L, R = 15L, J = 4L,
                         mu_l = mu_l, mu_p = mu_p, sd_l = sd_l, sd_p = sd_p,
                         r_true = r_true)
  fit <- tulpa_nmix_laplace_re(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                               d$R, d$S, mixture = "NB", n_quad = 5L)

  expect_identical(fit$mixture, "NB")
  expect_identical(fit$optimizer, "joint_grad")
  expect_true(isTRUE(fit$converged))
  expect_true(is.finite(fit$r) && fit$r > 0)

  # Recovery (single seed; the community means are tighter than the
  # weakly-identified dispersion and variance components).
  expect_lt(abs(as.numeric(fit$mu_lambda) - mu_l), 0.4)
  expect_lt(abs(as.numeric(fit$mu_p)      - mu_p), 0.4)
  expect_gt(fit$r, r_true / 3); expect_lt(fit$r, r_true * 3)
  expect_lt(abs(sqrt(fit$Sigma_lambda[1, 1]) - sd_l), 0.35)
  expect_lt(abs(sqrt(fit$Sigma_p[1, 1])      - sd_p), 0.35)
})

test_that("mixture = 'NB' rejects the EM optimizer (Poisson-only solver)", {
  skip_on_cran()
  d <- .sim_nb_community(seed = 7L, S = 3L, R = 5L, J = 2L,
                         mu_l = 1.0, mu_p = 0.2, sd_l = 0.3, sd_p = 0.3, r_true = 6)
  expect_error(
    tulpa_nmix_laplace_re(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                          d$R, d$S, mixture = "NB", optimizer = "em"),
    "joint optimizer")
})
