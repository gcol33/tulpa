# gcol33/tulpa#877: the ESS backend sliced the fixed effects and the group
# effects one block at a time and random-walked the RE log-SD with the
# non-centered effects held fixed. Two directions were then crossed only at the
# rate of the narrowest conditional step: the level the intercept shares with
# the group intercepts (eta is unchanged by beta_0 + d, b_g - d), and the RE
# scale against its effects. At the default length the intercept read
# split-Rhat around 2 and bulk-ESS in single digits. The sweep now draws the
# shared level exactly and moves each log-SD holding the effects fixed.

test_that("ESS mixes the intercept and the RE scale of a (1 | g) model (gcol33/tulpa#877)", {
  skip_on_cran()
  pars <- c("(Intercept)", "x", "log_sigma_re")
  for (fam in c("gaussian", "poisson")) {
    set.seed(877)
    J <- 30L; m <- 8L; g <- rep(seq_len(J), each = m); x <- rnorm(J * m)
    eta <- 0.5 + 0.7 * x + rnorm(J, 0, 0.8)[g]
    y <- if (fam == "gaussian") eta + rnorm(J * m) else rpois(J * m, exp(eta))
    d <- data.frame(y = y, x = x, g = factor(g))

    set.seed(1)
    fit <- tulpa(y ~ x + (1 | g), d, family = fam, mode = "ess")
    expect_true(fit$convergence$ok, info = fam)
    dg <- diagnostics(fit, measures = c("rhat", "ess_bulk"))
    dg <- dg[match(pars, dg$parameter), ]
    expect_true(all(dg$rhat < 1.05), info = paste(fam, toString(dg$rhat)))
    expect_true(all(dg$ess_bulk > 100), info = paste(fam, toString(dg$ess_bulk)))

    # And it samples the posterior NUTS samples: every mean within half a
    # posterior SD of the NUTS mean.
    ref <- suppressWarnings(tulpa(y ~ x + (1 | g), d, family = fam,
                                  mode = "hmc", control = list(seed = 1L)))
    sds <- apply(ref$draws[, pars], 2, stats::sd)
    expect_true(all(abs(fit$means[pars] - ref$means[pars]) < 0.5 * sds),
                info = fam)
  }
})
