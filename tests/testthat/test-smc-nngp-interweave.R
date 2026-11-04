test_that("the NNGP quadrature marginal equals the dense Gaussian reference", {
  for (name in paste0("F", 1:4)) {
    f <- smc_nngp_fixture(name)
    for (pair in list(c(-30, -7), c(0, 0), c(5, 12))) {
      K <- exp(pair[1]) * exp(-f$D / exp(pair[2])) + diag(1e-8, nrow(f$D))
      V <- f$noise * diag(length(f$y)) + 4 * tcrossprod(f$X) +
        f$A %*% K %*% t(f$A)
      R <- chol(V); z <- forwardsolve(t(R), f$y)
      want <- -length(f$y) * log(2 * pi) / 2 - sum(log(diag(R))) - sum(z^2) / 2
      expect_equal(as.numeric(smc_nngp_log_marginal(f, pair[1], pair[2])), want,
                   tolerance = 1e-9)
    }
  }
})

test_that("noncentred NNGP SMC evidence agrees with exact quadrature across regimes", {
  skip_if_not_slow()
  for (name in c("F3", "F1", "F2", "F4")) {
    f <- smc_nngp_fixture(name)
    exact <- smc_nngp_exact(f)
    fine <- smc_nngp_exact(f, h = 0.05)
    message(sprintf("NNGP %s exact logZ %.10f, grid change %.3g, edge %.3g",
                    name, fine$logZ, exact$logZ - fine$logZ, fine$edge))
    expect_lt(abs(exact$logZ - fine$logZ), 1e-6, label = name)
    expect_lt(fine$edge, 1e-9, label = name)
    error <- se <- numeric(8)
    for (seed in 1:8) {
      fit <- tulpa_sample_glmm(f$y, NULL, f$X, "gaussian", "smc", phi = f$noise,
        sigma_beta = 2, spatial_spec = f$sp,
        control = list(n_particles = 1000L, seed = seed))
      error[seed] <- fit$log_evidence - fine$logZ
      se[seed] <- fit$log_evidence_se
      message(sprintf("NNGP %s seed %d error %.8f reported_se %.8f",
                      name, seed, error[seed], se[seed]))
    }
    message(sprintf("NNGP %s mean_error %.8f empirical_se %.8f",
                    name, mean(error), sd(error) / sqrt(length(error))))
    expect_lt(abs(mean(error)), 2 * sd(error) / sqrt(length(error)), label = name)
    expect_lt(sd(error), 2 * sqrt(mean(se^2)), label = name)
  }
})
