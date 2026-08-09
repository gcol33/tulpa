# test-nested-fixed-mixture-interval.R
# Fixed-effect credible bounds from the Gaussian mixture the outer grid defines
# (gcol33/tulpa#336).
#
# The grid gives coefficient j the mixture sum_k w_k N(mu_kj, V_kjj). Its mean
# and variance are linear functionals of that mixture and survive being
# collapsed to one Gaussian; a quantile is nonlinear and does not. Before this,
# `.fit_fixed_table()` collapsed first and read `mu +/- z sigma` off the
# collapse, while `ranef()` on the SAME fit inverted the mixture CDF through
# `.nl_gauss_mixture_summary()` -- two constructions for one posterior, and the
# lossy one on the fixed effects.
#
# Everything here is post-processing on state the fit already retains, so none
# of it needs a model: each fixture hands the reporting path the (w, mu, V) a
# grid would have left behind. The cross-checks against a real fit's own
# posterior draws live in test-nested-laplace-joint-fixed-moments.R, where the
# fitted fixture is.

# A nested-Laplace-shaped fit carrying exactly the components given. `var` is
# per-cell MARGINAL variance, so the per-cell precision is its diagonal inverse
# and `.nested_fixed_moments()` reads the same variances back.
mx_fit <- function(mu, var, w, gamma3 = NULL) {
  mu <- as.matrix(mu); var <- as.matrix(var)
  p  <- ncol(mu)
  f <- list(
    modes         = mu,
    weights       = as.numeric(w),
    n_fixed       = p,
    fixed_names   = paste0("beta", seq_len(p)),
    grid_modes    = lapply(seq_len(nrow(mu)), function(k) mu[k, ]),
    grid_hessians = lapply(seq_len(nrow(mu)),
                           function(k) diag(1 / var[k, ], p, p))
  )
  if (!is.null(gamma3)) {
    f$inner_skew     <- as.numeric(gamma3)
    f$inner_skew_idx <- seq_len(p)
    f <- tulpa:::.nl_skew_correction_attach(f, p, enabled = TRUE)
  }
  structure(f, class = c("tulpa_nested_laplace", "tulpa_fit"))
}

# The bounds as a bare matrix. `confint()` carries the provenance attributes,
# which `unname()` does not strip and `expect_equal()` compares; the numbers are
# what these fixtures are about, and the attributes are asserted on their own.
mx_q <- function(...) {
  ci <- confint(...)
  matrix(as.numeric(ci), nrow(ci), ncol(ci))
}

# The interval the collapsed Gaussian reports: the pre-#336 read, rebuilt here
# from the same moments so the two differ only in the quantile step.
mx_gauss_ci <- function(fit, level = 0.95) {
  mom <- tulpa:::.nested_fixed_moments(fit)
  a   <- (1 - level) / 2
  est <- mom$mean
  se  <- sqrt(pmax(diag(mom$cov), 0))
  cbind(est + stats::qnorm(a) * se, est + stats::qnorm(1 - a) * se)
}

mx_mu <- rbind(c(-0.40, 1.20), c(0.10, 1.00), c(0.60, 0.70))
mx_vr <- rbind(c(0.09, 0.04), c(0.12, 0.05), c(0.16, 0.07))
mx_w  <- c(0.2, 0.5, 0.3)

# --------------------------------------------------------------------------- #
# (1) The bounds ARE the mixture's own quantiles                               #
# --------------------------------------------------------------------------- #

test_that("the fixed-effect bounds invert the mixture CDF", {
  fit <- mx_fit(mx_mu, mx_vr, mx_w)
  ci  <- confint(fit)
  ref <- tulpa:::.nl_gauss_mixture_summary(mx_mu, mx_vr, mx_w,
                                           probs = c(0.025, 0.975))
  expect_equal(mx_q(fit), unname(ref$quantiles), tolerance = 1e-12)
  expect_identical(attr(ci, "interval_source"), "mixture_cdf")
  expect_true(is.na(attr(ci, "interval_declined")))

  # Read directly off the defining CDF, with no engine helper in the loop.
  cdf <- function(b, j) {
    sum(mx_w * stats::pnorm(b, mx_mu[, j], sqrt(mx_vr[, j]))) / sum(mx_w)
  }
  for (j in 1:2) {
    expect_equal(cdf(ci[j, 1], j), 0.025, tolerance = 1e-9)
    expect_equal(cdf(ci[j, 2], j), 0.975, tolerance = 1e-9)
  }
})

test_that("the requested level reaches the mixture read", {
  fit <- mx_fit(mx_mu, mx_vr, mx_w)
  ci  <- confint(fit, level = 0.80)
  ref <- tulpa:::.nl_gauss_mixture_summary(mx_mu, mx_vr, mx_w,
                                           probs = c(0.10, 0.90))
  expect_equal(mx_q(fit, level = 0.80), unname(ref$quantiles), tolerance = 1e-12)
  expect_identical(colnames(ci), c("10.0%", "90.0%"))
})

# --------------------------------------------------------------------------- #
# (2) Nothing moment-based moved                                               #
# --------------------------------------------------------------------------- #

test_that("the estimate, standard error and vcov are the same moments as before", {
  fit <- mx_fit(mx_mu, mx_vr, mx_w)
  mom <- tulpa:::.nested_fixed_moments(fit)

  expect_equal(unname(coef(fit)), mom$mean, tolerance = 1e-14)
  expect_equal(summary(fit)$std.error, sqrt(diag(mom$cov)), tolerance = 1e-14)
  expect_equal(unname(vcov(fit)), unname(mom$cov), tolerance = 1e-14)

  # The two moments are the law of total variance by hand, so the marginalizer
  # is checked against arithmetic rather than against itself.
  w <- mx_w / sum(mx_w)
  expect_equal(mom$mean, as.numeric(crossprod(w, mx_mu)), tolerance = 1e-14)
  expect_equal(sqrt(diag(mom$cov)),
               sqrt(as.numeric(crossprod(w, mx_mu^2 + mx_vr)) -
                      as.numeric(crossprod(w, mx_mu))^2),
               tolerance = 1e-12)

  # And the mixture summarizer's own moments are those same two numbers, so the
  # quantiles it returns belong to the distribution `estimate` / `std.error`
  # describe.
  ref <- tulpa:::.nl_gauss_mixture_summary(mx_mu, mx_vr, mx_w,
                                           probs = c(0.025, 0.975))
  expect_equal(ref$mean, mom$mean, tolerance = 1e-12)
  expect_equal(ref$sd, sqrt(diag(mom$cov)), tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# (3) Reduction: where the mixture IS one Gaussian, the old interval returns   #
# --------------------------------------------------------------------------- #

test_that("a Gaussian-equivalent mixture reproduces the collapsed-Gaussian interval", {
  # One cell.
  f1 <- mx_fit(rbind(c(0.3, -1.1)), rbind(c(0.25, 0.04)), 1)
  expect_equal(mx_q(f1), unname(mx_gauss_ci(f1)), tolerance = 1e-10)

  # Several cells that are copies of one another at unequal weights: the
  # mixture is that one Gaussian whatever the weights are.
  mu <- mx_mu[c(2, 2, 2), ]; vr <- mx_vr[c(2, 2, 2), ]
  fk <- mx_fit(mu, vr, c(0.15, 0.60, 0.25))
  expect_equal(mx_q(fk), unname(mx_gauss_ci(fk)), tolerance = 1e-10)
  expect_identical(attr(confint(fk), "interval_source"), "mixture_cdf")

  # A zero-weight cell is not a dropped cell: it carries no mass, so the read
  # stays the mixture one.
  fz <- mx_fit(mx_mu, mx_vr, c(0.4, 0.6, 0))
  expect_identical(attr(confint(fz), "interval_source"), "mixture_cdf")
  expect_equal(mx_q(fz),
               unname(tulpa:::.nl_gauss_mixture_summary(
                 mx_mu[1:2, ], mx_vr[1:2, ], c(0.4, 0.6),
                 probs = c(0.025, 0.975))$quantiles),
               tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# (4) Symmetry is not Gaussianity                                             #
# --------------------------------------------------------------------------- #

test_that("a symmetric mixture stays symmetric and is free to differ in width", {
  # 0.5 N(-2, 1) + 0.5 N(2, 1) is symmetric about 0 and is not a Gaussian.
  # Matching its mean and variance gives sd = sqrt(5), so the collapsed read
  # reports +/- 1.96 sqrt(5) = +/- 4.38 while the mixture's own 95% interval
  # sits near +/- 3.64. Requiring equality on a merely SYMMETRIC fixture would
  # be asserting the shape away again, so the gate here is symmetry plus a
  # difference -- equality is required only of the Gaussian-equivalent
  # fixtures above.
  fit <- mx_fit(rbind(-2, 2), rbind(1, 1), c(0.5, 0.5))
  ci  <- mx_q(fit)
  expect_equal(ci[1, 1], -ci[1, 2], tolerance = 1e-9)

  g <- unname(mx_gauss_ci(fit))
  expect_equal(g[1, 2], stats::qnorm(0.975) * sqrt(5), tolerance = 1e-12)
  expect_lt(ci[1, 2], g[1, 2] - 0.5)

  cdf <- function(b) 0.5 * stats::pnorm(b, -2, 1) + 0.5 * stats::pnorm(b, 2, 1)
  expect_equal(cdf(ci[1, 2]), 0.975, tolerance = 1e-10)
  expect_equal(cdf(ci[1, 1]), 0.025, tolerance = 1e-10)
})

# --------------------------------------------------------------------------- #
# (5) Against draws from the same mixture                                      #
# --------------------------------------------------------------------------- #

test_that("the analytic quantiles are the ones draws from the same mixture carry", {
  skip_on_cran()
  fit <- mx_fit(mx_mu, mx_vr, mx_w)
  set.seed(7L)
  n  <- 400000L
  k  <- sample.int(nrow(mx_mu), n, replace = TRUE, prob = mx_w)
  dr <- cbind(stats::rnorm(n, mx_mu[k, 1], sqrt(mx_vr[k, 1])),
              stats::rnorm(n, mx_mu[k, 2], sqrt(mx_vr[k, 2])))
  emp <- t(apply(dr, 2L, stats::quantile, c(0.025, 0.975)))
  expect_equal(mx_q(fit), unname(emp), tolerance = 0.02)

  # The collapsed read is the one that misses, so this is a discriminating
  # check rather than a tolerance wide enough to pass either way.
  expect_gt(max(abs(mx_gauss_ci(fit) - emp)),
            5 * max(abs(mx_q(fit) - emp)))
})

# --------------------------------------------------------------------------- #
# (6) The #302 correction keeps its own read, and says why                     #
# --------------------------------------------------------------------------- #

test_that("an enabled skew correction reports the MAP-cell read it was measured on", {
  g   <- c(0.35, -0.28)
  fit <- mx_fit(mx_mu, mx_vr, mx_w, gamma3 = g)
  ci  <- confint(fit)

  expect_identical(attr(ci, "interval_source"), "skew_map_cell")
  expect_match(attr(ci, "interval_declined"), "MAP cell")
  expect_true(all(attr(ci, "skew_applied")))

  mom <- tulpa:::.nested_fixed_moments(fit)
  ref <- tulpa:::.nl_skew_marginal(mom$mean, sqrt(diag(mom$cov)), g,
                                   c(0.025, 0.975), enabled = TRUE)
  expect_equal(mx_q(fit), unname(ref$q), tolerance = 1e-14)

  # It is a different read from the mixture one, not the same numbers relabelled.
  expect_false(isTRUE(all.equal(mx_q(fit), mx_q(mx_fit(mx_mu, mx_vr, mx_w)))))
})

# --------------------------------------------------------------------------- #
# (7) An incomplete grid, and declines that name the gate they fell at         #
# --------------------------------------------------------------------------- #

test_that("a repaired grid reads as the posterior over the cells that remain", {
  # A positive-weight cell that retained no block. The moments renormalize over
  # the cells that did, so the mixture carries the same weighting `estimate` and
  # `std.error` were formed under and the read runs (gcol33/tulpa#342). What it
  # reports is the posterior conditional on those cells: the same fit as a grid
  # that never held the dropped cell, on every number.
  fit <- mx_fit(mx_mu, mx_vr, mx_w)
  fit$grid_hessians[2] <- list(NULL)
  kept <- mx_fit(mx_mu[c(1, 3), ], mx_vr[c(1, 3), ], mx_w[c(1, 3)])

  ci <- confint(fit)
  ck <- confint(kept)
  expect_identical(attr(ci, "interval_source"), "mixture_cdf")
  expect_true(is.na(attr(ci, "interval_declined")))
  expect_equal(mx_q(fit), mx_q(kept), tolerance = 1e-12)
  expect_equal(unname(coef(fit)), unname(coef(kept)), tolerance = 1e-14)
  expect_equal(unname(vcov(fit)), unname(vcov(kept)), tolerance = 1e-14)

  # The bounds are the mixture's own, read off the defining CDF over the two
  # surviving components rather than off the Gaussian matching their moments.
  w2 <- mx_w[c(1, 3)] / sum(mx_w[c(1, 3)])
  cdf <- function(b, j) {
    sum(w2 * stats::pnorm(b, mx_mu[c(1, 3), j], sqrt(mx_vr[c(1, 3), j])))
  }
  for (j in 1:2) {
    expect_equal(cdf(ci[j, 1], j), 0.025, tolerance = 1e-9)
    expect_equal(cdf(ci[j, 2], j), 0.975, tolerance = 1e-9)
  }
  expect_gt(max(abs(mx_q(fit) - unname(mx_gauss_ci(fit)))), 1e-6)

  # Every number agrees; the completeness report is what separates them. The
  # dropped mass is gone, not recovered, and the fit says so.
  expect_equal(attr(ci, "retained_mass"), sum(mx_w[c(1, 3)]) / sum(mx_w),
               tolerance = 1e-14)
  expect_equal(attr(ck, "retained_mass"), 1, tolerance = 1e-14)
  expect_equal(attr(summary(fit), "retained_mass"),
               attr(ci, "retained_mass"), tolerance = 1e-14)
  expect_equal(tulpa:::.nested_fixed_moments(fit)$mass,
               sum(mx_w[c(1, 3)]) / sum(mx_w), tolerance = 1e-14)
})

test_that("a fit with no retained components declines with its own reason", {
  fit <- mx_fit(mx_mu, mx_vr, mx_w)
  mom <- tulpa:::.nested_fixed_moments(fit)
  bare <- mom; bare$mu <- NULL; bare$var <- NULL
  iv <- tulpa:::.nl_fixed_interval(
    bare, 1:2, mom$mean, sqrt(diag(mom$cov)), c(0.025, 0.975),
    list(enabled = FALSE, gamma3 = rep(NA_real_, 2)))
  expect_identical(iv$source, "gaussian_moment")
  expect_match(iv$declined, "no retained mixture components")
  expect_equal(iv$q, unname(mx_gauss_ci(fit)), tolerance = 1e-14)
})
