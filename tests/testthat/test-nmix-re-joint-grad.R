# joint_grad (analytic Fisher-identity gradient) vs joint_fd (finite-difference
# of the objective) for the community N-mixture. Both drive the same native
# oracle through the AGHQ engine and target nearly-coincident optima at the AGHQ
# default n_quad = 9 (they differ only by the O(AGHQ-truncation) node-placement
# terms the analytic gradient omits), so the refined community means / covariances
# / marginal log-lik must agree closely. Also checks the n_quad = 1 guard.

.sim_small_community <- function(seed, S, R, J, mu_lambda, mu_p, Sig_l, sd_p) {
  set.seed(seed)
  p_lam <- length(mu_lambda)
  Ll <- t(chol(Sig_l))
  X_lambda <- cbind(1, as.numeric(scale(rnorm(R))))[, seq_len(p_lam), drop = FALSE]
  y <- integer(0); site_idx <- integer(0); species_idx <- integer(0)
  for (s in seq_len(S)) {
    cf_l <- mu_lambda + as.numeric(Ll %*% rnorm(p_lam))
    cf_p <- mu_p + sd_p * rnorm(1)
    pij  <- stats::plogis(cf_p)
    for (i in seq_len(R)) {
      N <- stats::rpois(1, exp(sum(X_lambda[i, ] * cf_l)))
      for (j in seq_len(J)) {
        y           <- c(y, stats::rbinom(1, N, pij))
        site_idx    <- c(site_idx, i)
        species_idx <- c(species_idx, s)
      }
    }
  }
  list(y = y, site_idx = site_idx, species_idx = species_idx,
       X_lambda = X_lambda, X_p = matrix(1, length(y), 1), R = R, S = S)
}

test_that("joint_grad matches joint_fd at n_quad = 9 (Poisson community N-mixture)", {
  skip_on_cran()
  d <- .sim_small_community(
    seed = 77L, S = 6L, R = 8L, J = 3L,
    mu_lambda = c(1.0, 0.3), mu_p = 0.2,
    Sig_l = matrix(c(0.25, 0.08, 0.08, 0.18), 2, 2), sd_p = 0.4)

  args <- list(y = d$y, site_idx = d$site_idx, species_idx = d$species_idx,
               X_lambda = d$X_lambda, X_p = d$X_p, n_sites = d$R, n_species = d$S,
               n_quad = 9L)
  fd <- do.call(tulpa_nmix_laplace_re, c(args, optimizer = "joint_fd"))
  gr <- do.call(tulpa_nmix_laplace_re, c(args, optimizer = "joint_grad"))

  expect_identical(gr$optimizer, "joint_grad")
  expect_true(fd$converged && gr$converged)

  expect_lt(max(abs(gr$mu_lambda - fd$mu_lambda)),        0.05)
  expect_lt(abs(gr$mu_p - fd$mu_p),                        0.05)
  expect_lt(max(abs(gr$Sigma_lambda - fd$Sigma_lambda)),  0.05)
  expect_lt(abs(as.numeric(gr$Sigma_p) - as.numeric(fd$Sigma_p)), 0.05)
  expect_lt(abs(gr$log_lik - fd$log_lik),                  0.5)
})

test_that("joint_grad requires n_quad > 1 (points to the EM)", {
  skip_on_cran()
  d <- .sim_small_community(
    seed = 9L, S = 3L, R = 5L, J = 2L,
    mu_lambda = c(0.8, 0.0), mu_p = 0.1,
    Sig_l = matrix(c(0.2, 0, 0, 0.2), 2, 2), sd_p = 0.3)
  expect_error(
    tulpa_nmix_laplace_re(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                          d$R, d$S, optimizer = "joint_grad", n_quad = 1L),
    "n_quad > 1")
})
