# An outer design on the joint door. A multi-block registry fit is a one-arm
# fit on the joint driver, so a design laid on the same cells reads the same
# inner marginals and hyperprior through either door: a supplied design, whose
# cells do not depend on where either door placed its grid, has to report the
# same evidence, weights and summaries.

.jod_gaussian <- function() {
  y <- c(-.6, .2, .5, 1.2, -.1, .7)
  u <- seq(-5, 1, length.out = 129)
  quadrature <- rep(diff(u)[1], length(u))
  quadrature[c(1, length(u))] <- quadrature[1] / 2
  list(y = y, u = u, quadrature = quadrature,
       points = matrix(exp(u), ncol = 1, dimnames = list(NULL, "b1.sigma")),
       block = list(type = "iid", n_units = length(y), obs_idx = seq_along(y)))
}

.jod_ctl <- function(...) c(list(progress = FALSE, diagnose_skew = FALSE), list(...))

test_that("a supplied design on the joint door reports the exact Gaussian evidence", {
  skip_on_cran()
  f <- .jod_gaussian(); y <- f$y; u <- f$u; n <- length(y)
  fit <- tulpa_nested_laplace_joint(
    list(y = list(y = y, X = matrix(1, n, 1), family = "gaussian", phi = .2)),
    prior = list(f$block),
    control = .jod_ctl(outer_design = list(points = f$points,
                                           weights = f$quadrature)))
  v <- .2 + exp(2 * u)
  anchor <- tulpa:::.nl_scale_anchor(); rate <- -log(anchor[2]) / anchor[1]
  lp <- -.5 * ((n - 1) * log(v) + log(v + n * 2.5^2) +
    sum((y - mean(y))^2) / v + n * mean(y)^2 / (v + n * 2.5^2) + n * log(2 * pi)) +
    log(rate) + u - rate * exp(u)
  lz <- tulpa:::.tulpa_logsumexp(lp + log(f$quadrature))
  w <- exp(lp + log(f$quadrature) - lz)
  expect_identical(fit$integration, "user")
  expect_identical(fit$integration_requested, "user")
  expect_equal(unname(fit$theta_grid), unname(f$points))
  expect_equal(fit$log_evidence, lz, tolerance = 1e-7)
  expect_equal(fit$weights, w, tolerance = 1e-7)
  variance <- 1 / (n / v + 1 / 2.5^2)
  mu <- variance * sum(y) / v
  exact <- vapply(c(.025, .5, .975), function(p)
    uniroot(function(b) sum(w * pnorm(b, mu, sqrt(variance))) - p,
      c(-10, 10), tol = 1e-12)$root, numeric(1))
  fixed <- tulpa:::.nested_fixed_moments(fit)
  got <- tulpa:::.nl_gauss_mixture_summary(fixed$mu, fixed$var, fixed$w)
  expect_equal(as.numeric(got$quantiles), exact, tolerance = 1e-6)
})

test_that("the joint and registry doors agree on a supplied two-block design", {
  skip_on_cran()
  set.seed(11)
  S <- 12L
  nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- lengths(nb)
  idx <- rep(seq_len(S), each = 5L); n <- length(idx)
  g <- rep(1:6, length.out = n)
  x <- rnorm(n)
  y <- rbinom(n, 1L, plogis(-.2 + .5 * x + sin(idx / 2) + rnorm(6, 0, .4)[g]))
  icar <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
               adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
               n_neighbors = nn)
  iid <- list(type = "iid", n_units = 6L, obs_idx = g)
  pts <- as.matrix(expand.grid(b1.tau = c(.5, 1.5, 4), b2.sigma = c(.2, .6, 1.4)))
  design <- list(points = pts, weights = rep(1, nrow(pts)))
  X <- cbind(1, x)
  reg <- tulpa_nested_laplace(y, rep(1L, n), X, prior = list(icar, iid),
                              family = "binomial",
                              control = .jod_ctl(outer_design = design))
  jnt <- tulpa_nested_laplace_joint(
    list(y = list(y = y, X = X, family = "binomial")), prior = list(icar, iid),
    control = .jod_ctl(outer_design = design))
  expect_identical(jnt$integration, "user")
  expect_equal(unname(jnt$theta_grid), unname(reg$theta_grid))
  expect_equal(jnt$log_marginal, reg$log_marginal, tolerance = 1e-8)
  expect_equal(jnt$weights, reg$weights, tolerance = 1e-8)
  expect_equal(jnt$log_evidence, reg$log_evidence, tolerance = 1e-8)
  expect_equal(unname(jnt$theta_mean), unname(reg$theta_mean), tolerance = 1e-8)
  expect_equal(unname(jnt$theta_median), unname(reg$theta_median), tolerance = 1e-8)
  rf <- tulpa:::.nested_fixed_moments(reg)
  jf <- tulpa:::.nested_fixed_moments(jnt)
  expect_equal(jf$mu, rf$mu, tolerance = 1e-6)
  expect_equal(jf$var, rf$var, tolerance = 1e-6)
})

test_that("importance quadrature on the joint door reads the registry's posterior", {
  skip_on_cran()
  f <- .jod_gaussian(); n <- length(f$y)
  ctl <- .jod_ctl(integration = "qmc", qmc_tol = .05, qmc_nodes = 4096L)
  fit <- tulpa_nested_laplace_joint(
    list(y = list(y = f$y, X = matrix(1, n, 1), family = "gaussian", phi = .2)),
    prior = list(f$block), control = ctl)
  reg <- tulpa_nested_laplace(f$y, rep(1L, n), matrix(1, n, 1),
                              prior = list(f$block), family = "gaussian",
                              phi = .2, control = ctl)
  exact <- tulpa_nested_laplace_joint(
    list(y = list(y = f$y, X = matrix(1, n, 1), family = "gaussian", phi = .2)),
    prior = list(f$block),
    control = .jod_ctl(outer_design = list(points = f$points,
                                           weights = f$quadrature)))
  expect_identical(fit$integration, "qmc")
  expect_identical(fit$weight_kind, rep("importance", nrow(fit$theta_grid)))
  expect_identical(nrow(fit$theta_grid), nrow(reg$theta_grid))
  expect_equal(fit$log_evidence, reg$log_evidence, tolerance = 1e-6)
  expect_equal(unname(fit$theta_mean), unname(reg$theta_mean), tolerance = 1e-6)
  expect_lt(abs(fit$log_evidence - exact$log_evidence), 0.05)
})

test_that("an outer design is refused where the joint door cannot lay it", {
  f <- .jod_gaussian(); n <- length(f$y)
  arm <- list(y = list(y = f$y, X = matrix(1, n, 1), family = "gaussian", phi = .2))
  expect_error(tulpa_nested_laplace_joint(
    arm, prior = list(f$block), phi_grid = list(y = c(.1, .2, .4)),
    control = .jod_ctl(integration = "qmc")), "phi_grid")
  expect_error(tulpa_nested_laplace_joint(
    arm, prior = list(f$block),
    control = .jod_ctl(integration = "qmc", k_quality = "good")), "k_quality")
  expect_error(tulpa_nested_laplace_joint(
    arm, prior = list(f$block), control = .jod_ctl(integration = "user")),
    "outer_design")
  expect_error(tulpa_nested_laplace_joint(
    arm, prior = list(f$block), control = .jod_ctl(integration = "aghq")),
    "four free axes")
})
