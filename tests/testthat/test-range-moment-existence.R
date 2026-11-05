.range_truth_fixture <- function() {
  set.seed(1L)
  n <- 60L
  lon <- runif(n, 0, 10); lat <- runif(n, 0, 10)
  D <- as.matrix(dist(cbind(lon, lat)))
  draw <- function(s, r) as.numeric(t(chol(s^2 * exp(-D / r) + diag(1e-9, n))) %*% rnorm(n))
  y <- .3 + draw(.63, .5) + draw(1, 3) + rnorm(n, sd = sqrt(.1))
  d <- data.frame(lon, lat, y)
  sp <- validate_gp(spatial_multiscale(~lon + lat, nn_local = n - 1L,
    nn_regional = n - 1L, range_local = c(.2, 1.5),
    range_regional = c(1.5, 10), scale_coords = FALSE), d)
  blocks <- tulpa:::.multiscale_nl_blocks(sp)
  blocks[[1]]$sigma2_grid <- .63^2
  blocks[[1]]$phi_gp_grid <- .5
  blocks[[2]]$sigma2_grid <- 1
  list(d = d, D = D, blocks = blocks)
}

.range_truth_fit <- function(s, top, integration = "grid") {
  blocks <- s$blocks
  blocks[[2]]$phi_gp_grid <- exp(seq(log(.02), log(top), by = .075))
  suppressWarnings(tulpa_nested_laplace(s$d$y, rep(1L, nrow(s$d)),
    matrix(1, nrow(s$d), 1), prior = blocks, family = "gaussian", phi = .1,
    control = list(progress = FALSE, integration = integration, qmc_nodes = 4096L,
      auto_recenter = FALSE, prune = FALSE, axis_refine = "none",
      diagnose_k = FALSE, diagnose_skew = FALSE)))
}

# The complete neighbour set gives the dense exponential covariance. The
# intercept is integrated analytically under its N(0, 2.5^2) prior.
.range_truth_loglik <- function(s, r) {
  n <- nrow(s$d)
  V <- matrix(2.5^2, n, n) + diag(.1, n) + .63^2 * exp(-s$D / .5) + exp(-s$D / r)
  L <- chol(V)
  -.5 * (n * log(2 * pi) + 2 * sum(log(diag(L))) +
           sum(backsolve(L, s$d$y, transpose = TRUE)^2))
}

.range_truth_log_moments <- function(s) {
  lambda <- -log(tulpa:::.MSGP_RANGE_ALPHA) * 1.5 / tulpa:::.SPATIAL_RANGE_FACTOR
  shift <- .range_truth_loglik(s, 3)
  integrand <- function(v, p) vapply(v, function(h) {
    if (!is.finite(h) || h == 0) return(0)
    exp(.range_truth_loglik(s, 1 / h) - shift - lambda * h) * lambda * (-log(h))^p
  }, numeric(1))
  z <- integrate(integrand, 0, Inf, p = 0, rel.tol = 1e-9)$value
  m <- integrate(integrand, 0, Inf, p = 1, rel.tol = 1e-9)$value / z
  v <- integrate(integrand, 0, Inf, p = 2, rel.tol = 1e-9)$value / z - m^2
  c(mean = m, sd = sqrt(v))
}

test_that("moment orders follow the prior and likelihood boundary powers", {
  model <- list(prior_power = 1, likelihood_power = 0)
  read <- function(...) tulpa:::.hyper_moment_read(model, 7, 11, ...)
  expect_identical(read()$mean, Inf)
  expect_identical(read()$sd, Inf)
  expect_equal(read(power = .5)$mean, 7)
  expect_identical(read(power = .5)$sd, Inf)
  expect_equal(read(power = .25)[c("mean", "sd")], list(mean = 7, sd = 11))
  expect_equal(read(logarithm = TRUE)[c("mean", "sd")], list(mean = 7, sd = 11))
  for (setting in c("fixed", "bounded")) {
    bounded <- model; bounded[[setting]] <- TRUE
    expect_equal(tulpa:::.hyper_moment_read(bounded, 7, 11)[c("mean", "sd")],
                 list(mean = 7, sd = 11))
  }
  model$likelihood_power <- 1
  expect_true(tulpa:::.hyper_moment_rule(model)$mean_exists)
  expect_false(tulpa:::.hyper_moment_rule(model)$sd_exists)
  model$prior_power <- 3
  expect_true(tulpa:::.hyper_moment_rule(model)$sd_exists)
})

test_that("existing PC moments converge to the closed form as the window widens", {
  a <- 3; lambda <- 1
  exact_mean <- gamma(1 - 1 / a)
  exact_sd <- sqrt(gamma(1 - 2 / a) - exact_mean^2)
  evaluate <- function(top) {
    u <- seq(-4, log(top), by = .005)
    r <- exp(u)
    w <- exp(-a * u - lambda * exp(-a * u)); w <- w / sum(w)
    tulpa:::.hyper_moment_read(list(prior_power = a, likelihood_power = 0),
      sum(w * r), sqrt(sum(w * (r - sum(w * r))^2)))
  }
  m <- lapply(c(1e4, 1e6), evaluate)
  for (x in m) {
    expect_equal(x$mean, exact_mean, tolerance = 1e-6)
    expect_equal(x$sd, exact_sd, tolerance = .0002)
  }
  expect_equal(m[[1]]$mean, m[[2]]$mean, tolerance = 1e-6)
  expect_equal(m[[1]]$sd, m[[2]]$sd, tolerance = .0002)
})

test_that("explicitly bounded range axes retain their finite moments", {
  specs <- list(hyper_axis_spec("range", grid = seq(.01, 1.99, by = .01),
    bounds = c(0, 2), log_prior = function(x) rep(-log(2), length(x))))
  fit <- tulpa_hyper_grid(specs, function(hypers) list(log_marginal = 0), combine = "none")
  expect_true(fit$moment_existence$range$mean_exists)
  expect_true(fit$moment_existence$range$sd_exists)
  expect_identical(fit$moment_existence$range$reason, "bounded prior support")
  expect_equal(fit$theta_mean[["range"]], 1, tolerance = 1e-8)
  expect_equal(fit$theta_sd[["range"]], 2 / sqrt(12), tolerance = .01)
})

test_that("every sampled range uses the model declaration and retains log moments", {
  set.seed(4)
  columns <- list(gp = c("log_phi_gp"), hsgp = "log_lengthscale_hsgp",
    multiscale = c("log_phi_gp_local", "log_phi_gp_regional"),
    svc = c("log_phi_svc[1]", "log_phi_svc[2]"))
  for (backend in c("hmc_nuts", "mala", "imh_laplace", "ess", "mclmc", "smc")) {
    for (type in names(columns)) {
      dr <- matrix(rnorm(400 * length(columns[[type]])), 400,
        dimnames = list(NULL, columns[[type]]))
      fit <- tulpa:::.finalize_fit(list(draws = dr, family = "gaussian",
        spatial = list(type = type, n_svc = 2L)), backend = backend)
      s <- spatial_range(fit)
      expect_true(all(s$mean == Inf))
      expect_true(all(s$sd == Inf))
      expect_true(all(is.finite(s$q2.5) & is.finite(s$q97.5)))
      expect_true(all(vapply(attr(s, "log_moments"), function(x) all(is.finite(x)), TRUE)))
      expect_true(all(vapply(fit$moment_existence, function(x)
        identical(x$mean_exists, FALSE), TRUE)))
      if (type == "svc") expect_equal(nrow(s), 2)
    }
  }
})

test_that("the finite-mesh SPDE boundary follows its observed constant mode", {
  C <- Matrix::Diagonal(3)
  G <- Matrix::Matrix(matrix(c(1, -1, 0, -1, 2, -1, 0, -1, 1), 3), sparse = TRUE)
  sp <- spatial_spde_custom(C, G, Matrix::Diagonal(3), prior_range = c(1, .5),
                            prior_sigma = c(1, .1))
  object <- list(family = "gaussian", y = c(0, 1, 2), spatial = sp,
    joint = TRUE, draws = cbind(log_kappa = c(-1, 0, 1)),
    range_draws = c(8, 3, 1), range_summary = c(mean = 4))
  fit <- tulpa:::.hyper_attach_moments(object)
  expect_true(fit$moment_existence$log_kappa$mean_exists)
  expect_false(fit$moment_existence$log_kappa$sd_exists)
  expect_equal(fit$range_summary[["mean"]], 4)
  expect_identical(fit$range_summary[["sd"]], Inf)
  expect_identical(spatial_range(fit)$sd, Inf)
  spd <- spatial_spde_custom(C, C, Matrix::Diagonal(3), prior_range = c(1, .5),
                             prior_sigma = c(1, .1))
  object$spatial <- spd
  expect_false(tulpa:::.hyper_attach_moments(object)$moment_existence$log_kappa$mean_exists)
  object$spatial <- sp; object$family <- "poisson"; object$y <- c(0, 0, 0)
  expect_false(tulpa:::.hyper_attach_moments(object)$moment_existence$log_kappa$mean_exists)
})

test_that("the multiscale Gaussian posterior reports divergence against its analytic limit", {
  skip_on_cran()
  s <- .range_truth_fixture()
  expect_true(is.finite(.range_truth_loglik(s, Inf)))
  expect_equal(.range_truth_loglik(s, 1e8), .range_truth_loglik(s, Inf), tolerance = 1e-5)
  exact <- .range_truth_log_moments(s)
  fits <- lapply(c(1e3, 1e5), function(top) .range_truth_fit(s, top))
  for (fit in fits) {
    expect_identical(unname(fit$theta_mean["b2.phi_gp"]), Inf)
    expect_identical(unname(fit$theta_sd["b2.phi_gp"]), Inf)
    expect_equal(fit$block_moments[[2]]$mean[["phi_gp"]], Inf)
    expect_identical(spatial_range(fit)["range[regional]", "mean"], Inf)
    expect_true(isTRUE(fit$moment_existence$b1.phi_gp$fixed))
    expect_equal(fit$theta_mean[["b1.phi_gp"]], .5)
    expect_equal(fit$theta_log_mean[["b2.phi_gp"]], exact[["mean"]], tolerance = .001)
    expect_equal(fit$theta_log_sd[["b2.phi_gp"]], exact[["sd"]], tolerance = .002)
    expect_match(fit$moment_existence$b2.phi_gp$reason, "moment order must be < 1")
    row <- summary(fit)$hyperparameters
    expect_identical(row$mean[row$parameter == "phi_gp[regional]"], Inf)
    log_summary <- tulpa:::.nested_hyper_summary(fit, c(.025, .5, .975),
      transform = list(phi_gp = list(name = "log_range", fn = log,
        domain = "unbounded", moment_log = TRUE)))
    expect_true(all(is.finite(log_summary$mean)))
  }
  expect_equal(fits[[1]]$theta_log_mean, fits[[2]]$theta_log_mean, tolerance = .001)
  expect_equal(fits[[1]]$theta_log_sd, fits[[2]]$theta_log_sd, tolerance = .002)
})

test_that("importance integration obeys the same analytic range moment limit", {
  skip_on_cran()
  s <- .range_truth_fixture()
  fit <- .range_truth_fit(s, 1e5, "qmc")
  exact <- .range_truth_log_moments(s)
  expect_identical(fit$integration, "qmc")
  expect_identical(fit$theta_mean[["b2.phi_gp"]], Inf)
  expect_identical(fit$theta_sd[["b2.phi_gp"]], Inf)
  expect_equal(fit$theta_log_mean[["b2.phi_gp"]], exact[["mean"]], tolerance = .02)
  expect_equal(fit$theta_log_sd[["b2.phi_gp"]], exact[["sd"]], tolerance = .02)
  expect_true(all(is.finite(fit$theta_median)))
})
