# Item-5 recovery gate for tulpa_re_cov_gibbs. Certifies that the exact-target
# Sigma debias recovers a known random-effect covariance with calibrated
# intervals. Two layers, both tier 3:
#   * trimmed sweeps (12- / 8-seed) -- the routine full-validation run;
#   * the authoritative N = 20-seed strict gate (>= 85% coverage, 15% bias on
#     the gated parameters, binomial + poisson, small + identified regimes) --
#     the release gate, in-suite so it cannot rot against the API.
# The variance-component debias is the Bias-1 target; the intercept-slope
# correlation rho needs rich per-group designs to be identified
# (dev_notes/binary_identifiability.R), so in the SMALL regimes rho is gated
# on coverage only and its point recovery is gated in the identified regime.

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
# `fitter` selects the RE-covariance backend: "gibbs" (the exact debias) or
# "nested" (tulpa_re_cov_nested, the DEFAULT random-slope path).
recov_sweep <- function(family, G, npg, n_seed, seed_off = 0L, fitter = "gibbs") {
  truth <- c(sigma_1 = 0.7, sigma_2 = 0.5, rho_12 = 0.4)
  med <- matrix(NA_real_, n_seed, 3, dimnames = list(NULL, names(truth)))
  cov <- setNames(integer(3), names(truth))
  for (s in seq_len(n_seed)) {
    d  <- sim_recov(seed_off + s, family, G, npg)
    rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
    res <- if (fitter == "nested") {
      tulpa_re_cov_nested(d$y, d$ntr, d$X, rt, family = family)
    } else {
      tulpa_re_cov_gibbs(d$y, d$ntr, d$X, rt, family = family,
                         control = list(n_iter = 1200L, warmup = 600L,
                                        seed = 300L + s))
    }
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
  skip_if_not_slow()
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
  # the strict >= 85% @ N = 20 gate is the release-gate test below.
  for (nm in names(R$truth)) expect_gte(R$cov[[nm]], 9L)
})


test_that("binomial identified: variance debias and rho recovery", {
  skip_if_not_slow()
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


# ---------------------------------------------------------------------------
# The authoritative N = 20-seed strict gate (formerly dev_notes/recovery_gate.R,
# ported in-suite so it tracks the API). Gate criteria per regime:
# |mean(median) - truth| / truth <= 0.15 on the gated parameters AND 95%-CI
# coverage >= 85% on all parameters.
# ---------------------------------------------------------------------------

strict_gate <- function(R, n_seed, gate_bias, label) {
  for (nm in names(R$truth)) {
    cvg <- R$cov[[nm]] / n_seed
    expect_gte(cvg, 0.85,
               label = sprintf("%s %s coverage (%d/%d)", label, nm,
                               R$cov[[nm]], n_seed))
    if (nm %in% gate_bias) {
      bias <- abs(mean(R$med[, nm]) - R$truth[[nm]]) / R$truth[[nm]]
      expect_lte(bias, 0.15,
                 label = sprintf("%s %s relative bias", label, nm))
    }
  }
}

test_that("strict N=20 gate: binomial small groups (variances gated, rho coverage)", {
  skip_if_not_slow()
  n_seed <- 20L
  R <- recov_sweep("binomial", G = 60L, npg = 20L, n_seed = n_seed)
  strict_gate(R, n_seed, gate_bias = c("sigma_1", "sigma_2"),
              label = "binomial-small")
})

test_that("strict N=20 gate: poisson small groups (variances gated, rho coverage)", {
  skip_if_not_slow()
  n_seed <- 20L
  R <- recov_sweep("poisson", G = 60L, npg = 10L, n_seed = n_seed,
                   seed_off = 2000L)
  strict_gate(R, n_seed, gate_bias = c("sigma_1", "sigma_2"),
              label = "poisson-small")
})

test_that("strict N=20 gate: binomial identified (all three gated)", {
  skip_if_not_slow()
  n_seed <- 20L
  R <- recov_sweep("binomial", G = 60L, npg = 40L, n_seed = n_seed,
                   seed_off = 4000L)
  strict_gate(R, n_seed, gate_bias = c("sigma_1", "sigma_2", "rho_12"),
              label = "binomial-identified")
})


# The DEFAULT random-slope backend: tulpa() auto-selects tulpa_re_cov_nested for
# a correlated (1 + x | g) term, but that path had only interval-containment
# checks (no point-bias or coverage gate) while the gibbs sibling above carries
# the strict N=20 gate. Close the gap on the path most users actually hit
# (gcol33/tulpa#155). Poisson small groups, where the nested Laplace is
# well-behaved; calibrated at bias < 15% and coverage >= 75% on the variance
# components.
test_that("nested backend recovers variance components with coverage (default path)", {
  skip_if_not_slow()
  n_seed <- 20L
  R <- recov_sweep("poisson", G = 60L, npg = 12L, n_seed = n_seed,
                   fitter = "nested")
  for (nm in c("sigma_1", "sigma_2")) {
    expect_lt(abs(mean(R$med[, nm]) - R$truth[[nm]]) / R$truth[[nm]], 0.15,
              label = sprintf("nested %s relative bias", nm))
    expect_gte(R$cov[[nm]] / n_seed, 0.75,
               label = sprintf("nested %s coverage", nm))
  }
})
