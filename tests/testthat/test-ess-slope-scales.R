# gcol33/tulpa#877: the ESS backend's RWMH list named log_sigma_re_idx alone --
# the FIRST coefficient's log-SD of a slope term -- so the slope's log-SD and
# the correlated term's Cholesky coordinates were never proposed. Every draw
# carried their initial value (slope SD exactly 1), and the fixed effects were
# conditioned on it. A within-term variant of gcol33/tulpa#201.

test_that("ESS moves every scale and correlation coordinate of a slope term", {
  skip_on_cran()
  set.seed(1); J <- 30; n <- 300; gi <- sample(J, n, TRUE); x <- rnorm(n)
  S <- matrix(c(1, -.3, -.3, .25), 2)
  B <- MASS::mvrnorm(J, c(0, 0), S)
  d <- data.frame(y = 1 + .5 * x + B[gi, 1] + B[gi, 2] * x + rnorm(n, 0, .3),
                  x, g = factor(gi))

  for (form in c("y ~ x + (1 + x | g)", "y ~ x + (1 + x || g)")) {
    # The ESS block updates still cross the intercept / group-effect ridge
    # slowly at the default length; that is flagged by the fit-time
    # convergence check, not asserted away here.
    f <- suppressWarnings(tulpa(stats::as.formula(form), d, family = "gaussian",
                                phi = .09, mode = "ess",
                                control = list(seed = 1L)))
    hyper <- f$draws[, grep("^log_sigma_re|^L_re", colnames(f$draws)),
                     drop = FALSE]
    expect_equal(ncol(hyper), if (grepl("||", form, fixed = TRUE)) 2L else 3L,
                 label = form)
    n_unique <- apply(hyper, 2, function(v) length(unique(v)))
    expect_true(all(n_unique > 10L), info = paste(form, toString(n_unique)))

    # The slope SD is sampled, not pinned at its initial value of 1: its draws
    # spread over the posterior (truth 0.5) instead of repeating one number.
    sd_slope <- exp(hyper[, "log_sigma_re[t1.c2]"])
    expect_gt(diff(range(sd_slope)), 0.1, label = form)
    expect_lt(stats::quantile(sd_slope, 0.05), 0.9, label = form)
  }
})
