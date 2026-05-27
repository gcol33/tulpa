# Community / multispecies N-mixture (tulpa_nmix_laplace_re), routed through the
# shared AGHQ engine (tulpa_re_aghq at n_quad = 1) with tulpa_nmix_site_marginal
# as the per-species oracle. Two gates:
#   * correctness  -- the engine's community Laplace marginal equals an
#     independent R computation (mode via curvature-free BFGS, then the closed
#     Laplace term) that shares no code with the compiled objective beyond the
#     per-species marginal primitive;
#   * recovery     -- the community means and variance components are recovered
#     from simulated data over several seeds.

simulate_community <- function(seed, S, n_sites, J, mu_l, mu_p, sig_l, sig_p) {
  set.seed(seed)
  b_l <- rnorm(S, 0, sig_l); b_p <- rnorm(S, 0, sig_p)
  y <- integer(0); si <- integer(0); sp <- integer(0)
  for (s in seq_len(S)) {
    lam <- exp(mu_l + b_l[s]); p <- plogis(mu_p + b_p[s])
    N  <- rpois(n_sites, lam)
    ym <- matrix(rbinom(n_sites * J, rep(N, times = J), p), n_sites, J)
    y  <- c(y, as.integer(as.vector(t(ym))))
    si <- c(si, rep(seq_len(n_sites), each = J))
    sp <- c(sp, rep(s, n_sites * J))
  }
  list(y = y, site_idx = si, species_idx = sp, n_sites = n_sites, n_species = S,
       X_lambda = matrix(1, n_sites, 1), X_p = matrix(1, length(y), 1))
}

# Independent community Laplace marginal (intercept-only, n_quad = 1) at the
# given community means / covariances. Uses the public per-species marginal but
# an independent mode-find and Laplace assembly.
indep_community_laplace <- function(d, mu, sig_l, sig_p, K) {
  orc <- tulpa:::.nmix_re_oracle(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                                 d$n_sites, d$n_species, 1L, 1L, K)
  P <- diag(c(1 / sig_l, 1 / sig_p))
  logdetS <- log(sig_l) + log(sig_p)
  tot <- 0
  for (s in seq_len(d$n_species)) {
    nll <- function(b) {
      gh <- orc$grad_hess(s, mu + b)
      -(gh$logL - 0.5 * sum(b * (P %*% b)))
    }
    bh <- stats::optim(c(0, 0), nll, method = "BFGS",
                       control = list(reltol = 1e-12, maxit = 500))$par
    gh <- orc$grad_hess(s, mu + bh)
    tot <- tot + gh$logL - 0.5 * sum(bh * (P %*% bh)) - 0.5 * logdetS -
           0.5 * as.numeric(determinant(gh$negH + P, logarithm = TRUE)$modulus)
  }
  tot
}

test_that("community N-mixture marginal matches an independent Laplace", {
  d <- simulate_community(1L, S = 12L, n_sites = 12L, J = 6L,
                          mu_l = log(8), mu_p = qlogis(0.6), sig_l = 0.6, sig_p = 0.5)
  K <- max(d$y) + 100L
  fit <- tulpa_nmix_laplace_re(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                               d$n_sites, d$n_species, K_max = K)
  expect_false(is.null(fit))
  indep <- indep_community_laplace(d, c(fit$mu_lambda, fit$mu_p),
                                   fit$Sigma_lambda[1, 1], fit$Sigma_p[1, 1], K)
  # The engine's reported marginal is the Laplace value at its own optimum.
  expect_equal(fit$log_lik, indep, tolerance = 1e-3)
})

test_that("community N-mixture returns the documented contract", {
  d <- simulate_community(2L, S = 8L, n_sites = 10L, J = 5L,
                          mu_l = log(6), mu_p = qlogis(0.5), sig_l = 0.5, sig_p = 0.4)
  fit <- tulpa_nmix_laplace_re(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                               d$n_sites, d$n_species)
  expect_s3_class(fit, "tulpa_nmix_re_fit")
  expect_length(fit$mu_lambda, 1L); expect_length(fit$mu_p, 1L)
  expect_equal(dim(fit$vcov), c(2L, 2L))
  expect_equal(dim(fit$b_lambda), c(8L, 1L))
  expect_equal(dim(fit$b_p), c(8L, 1L))
  expect_true(fit$Sigma_lambda[1, 1] > 0 && fit$Sigma_p[1, 1] > 0)
  expect_true(all(is.finite(fit$vcov)))
  expect_true(is.finite(fit$log_lik))
})

test_that("community N-mixture recovers means and variance components", {
  skip_on_cran()
  mu_l <- log(7); mu_p <- qlogis(0.55); sig_l <- 0.5; sig_p <- 0.4
  seeds <- 1:6
  est <- t(vapply(seeds, function(sd) {
    d <- simulate_community(sd, S = 20L, n_sites = 12L, J = 8L,
                            mu_l = mu_l, mu_p = mu_p, sig_l = sig_l, sig_p = sig_p)
    fit <- tulpa_nmix_laplace_re(d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p,
                                 d$n_sites, d$n_species)
    c(mu_l = fit$mu_lambda, mu_p = fit$mu_p,
      s_l = sqrt(fit$Sigma_lambda[1, 1]), s_p = sqrt(fit$Sigma_p[1, 1]))
  }, numeric(4)))
  m <- colMeans(est)
  # Community means recovered near truth (averaged over seeds).
  expect_lt(abs(m["mu_l"] - mu_l), 0.20)
  expect_lt(abs(m["mu_p"] - mu_p), 0.25)
  # Variance-component SDs in the right ballpark (not collapsed / not blown up).
  expect_gt(m["s_l"], 0.25); expect_lt(m["s_l"], 0.85)
  expect_gt(m["s_p"], 0.15); expect_lt(m["s_p"], 0.75)
})
