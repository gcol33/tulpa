# Item-5 recovery gate (in-suite, trimmed). Certifies that tulpa_re_cov_gibbs
# recovers a known random-effect covariance with calibrated intervals.
#
# The authoritative gate -- N = 20 seeds, binomial AND poisson, small + identified
# regimes, strict >= 85% coverage -- is dev_notes/recovery_gate.R (PASS). Here we
# run a faster subset (fewer seeds, slightly looser floors) so the property is
# guarded in CI without a multi-minute run. The variance-component debias is the
# Bias-1 target; the intercept-slope correlation rho needs rich per-group designs
# to be identified (dev_notes/binary_identifiability.R), so the binary rho check
# uses the identified regime.

sim_recov <- function(seed, family, G, npg, ntr = 1L, beta = c(0, 0.3),
                      Sigma = matrix(c(0.7^2, 0.4 * 0.7 * 0.5,
                                       0.4 * 0.7 * 0.5, 0.5^2), 2)) {
  set.seed(seed)
  N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N); X <- cbind(1, x); Z <- cbind(1, x)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), 2))
  eta <- as.numeric(X %*% beta) + rowSums(Z * u[grp, ])
  y <- if (family == "binomial") rbinom(N, ntr, plogis(eta)) else rpois(N, exp(eta))
  list(y = y, X = X, Z = Z, grp = grp, G = G, N = N, ntr = rep(ntr, N))
}

# Fit `n_seed` data sets and return per-seed posterior medians + coverage counts.
recov_sweep <- function(family, G, npg, n_seed, seed_off = 0L) {
  truth <- c(sigma_1 = 0.7, sigma_2 = 0.5, rho_12 = 0.4)
  med <- matrix(NA_real_, n_seed, 3, dimnames = list(NULL, names(truth)))
  cov <- setNames(integer(3), names(truth))
  for (s in seq_len(n_seed)) {
    d  <- sim_recov(seed_off + s, family, G, npg)
    rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
    res <- tulpa_re_cov_gibbs(d$y, d$ntr, d$X, rt, family = family,
                              n_iter = 1200L, n_burnin = 600L, seed = 300L + s)
    for (nm in names(truth)) {
      row <- res$posterior[res$posterior$parameter == nm, ]
      med[s, nm] <- row$median
      if (truth[[nm]] >= row$ci_lo && truth[[nm]] <= row$ci_hi)
        cov[[nm]] <- cov[[nm]] + 1L
    }
  }
  list(med = med, cov = cov, truth = truth)
}


test_that("poisson small groups: variance components recover, intervals calibrated", {
  skip_on_cran()
  skip_if_fast()
  n_seed <- 12L
  R <- recov_sweep("poisson", G = 60L, npg = 10L, n_seed = n_seed)
  # Variance components (the Bias-1 debias target) recover in small groups.
  for (nm in c("sigma_1", "sigma_2")) {
    # point recovery within 18% (looser than the 15% N=20 gate for small n)
    expect_lt(abs(mean(R$med[, nm]) - R$truth[[nm]]) / R$truth[[nm]], 0.18,
              label = sprintf("poisson %s relative bias", nm))
  }
  # rho (intercept-slope correlation) is only weakly identified at npg=10 for
  # BOTH families -- it needs rich per-group covariate designs (npg ~ 40) to be
  # identified, so its point estimate is batch-variable here while the interval
  # stays calibrated. Point recovery of rho is gated in the identified regime
  # (the binomial-npg=40 test below); see dev_notes/binary_identifiability.R.
  # Coverage of every parameter is not grossly miscalibrated (>= 9/12 = 75%);
  # the strict >= 85% @ N >= 20 gate lives in dev_notes/recovery_gate.R.
  for (nm in names(R$truth)) expect_gte(R$cov[[nm]], 9L)
})


test_that("binomial identified: variance debias and rho recovery", {
  skip_on_cran()
  skip_if_fast()
  n_seed <- 8L
  R <- recov_sweep("binomial", G = 60L, npg = 40L, n_seed = n_seed,
                   seed_off = 4000L)
  # variance components: the Bias-1 target, recovered (no gross downward shrink)
  expect_lt(abs(mean(R$med[, "sigma_1"]) - 0.7) / 0.7, 0.15)
  expect_lt(abs(mean(R$med[, "sigma_2"]) - 0.5) / 0.5, 0.18)
  # rho recovers once the per-group design identifies the cross-moment
  expect_lt(abs(mean(R$med[, "rho_12"]) - 0.4) / 0.4, 0.20)
  # coverage of every parameter (>= 6/8)
  for (nm in names(R$truth)) expect_gte(R$cov[[nm]], 6L)
})
