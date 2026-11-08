test_that("log tensor interpolation reproduces cubics and partitions positive mass", {
  axes <- list(seq(-2, 2, length.out = 5), seq(-1, 3, length.out = 6))
  grid <- as.matrix(expand.grid(axes))
  f <- function(x) 1 + x[, 1]^3 - x[, 2]^2 + x[, 1] * x[, 2]
  query <- cbind(seq(-1.9, 1.9, length.out = 31), seq(2.9, -.9, length.out = 31))
  log_density <- tulpa:::cpp_outer_tensor_interpolate(axes, matrix(f(grid)), query)
  expect_equal(as.numeric(log_density), f(query), tolerance = 1e-11)
  expect_true(all(exp(log_density) > 0))
  mass <- seq_len(nrow(query)); mass <- mass / sum(mass)
  P <- tulpa:::cpp_outer_tensor_partition(axes, query, mass)
  expect_true(all(P@x >= 0))
  expect_equal(as.numeric(Matrix::rowSums(P)), mass, tolerance = 1e-14)
  expect_equal(sum(P), 1, tolerance = 1e-14)
  expect_equal(unname(as.matrix(P %*% grid)), query * mass, tolerance = 1e-14)
})

test_that("regional prior coordinates centre their medians with the declared Jacobian", {
  prior <- list(prior_sigma = c(1, .1), prior_range = c(3, .2))
  rate <- -log(.1)
  range_rate <- -log(.2) * 3 / 2
  cdf <- cbind(c(.02, .07, .3, .9), c(.04, .12, .4, .85))
  points <- cbind((-log1p(-cdf[, 1]) / rate)^2,
                  range_rate / (-log1p(-cdf[, 2])))
  weights <- c(.1, .4, .3, .2)
  transport <- tulpa:::.nl_gp_regional_transport(points, weights, prior)
  center <- transport$physical(matrix(0, 1, 2))
  got <- as.numeric(center)
  expected <- vapply(seq_len(2), function(j)
    exp(tulpa:::.nl_wtd_quantile(log(points[, j]), weights, .5, outside = "clamp")), numeric(1))
  expect_equal(got, unname(expected), tolerance = 1e-12)
  query <- rbind(c(-2, -1), c(0, 0), c(2, 1))
  h <- 1e-6
  derivative <- sapply(seq_len(2), function(j) {
    plus <- minus <- query
    plus[, j] <- plus[, j] + h; minus[, j] <- minus[, j] - h
    (log(transport$physical(plus)[, j]) - log(transport$physical(minus)[, j])) / (2 * h)
  })
  expect_equal(transport$log_density(query), rowSums(dnorm(query, log = TRUE)) -
    rowSums(log(abs(derivative))), tolerance = 1e-8)
  tail <- rbind(c(-9, 9), c(9, -9))
  physical <- transport$physical(tail)
  expect_true(all(is.finite(physical) & physical > 0))
  expect_equal(transport$inverse(physical), tail, tolerance = 1e-10)
  extreme <- cbind(rep((50 / rate)^2, 3), rep(range_rate / 55, 3))
  transport <- tulpa:::.nl_gp_regional_transport(extreme, c(.2, .6, .2), prior)
  center <- transport$physical(matrix(0, 1, 2))
  expect_true(all(is.finite(center)))
  expect_equal(as.numeric(center), unname(extreme[2, ]), tolerance = 1e-12)
  expect_true(all(is.finite(transport$log_density(query))))
})

test_that("joint transport integrates a correlated Gaussian with its full Jacobian", {
  rule <- tulpa:::gauss_hermite_prob(3L)
  standard <- as.matrix(expand.grid(rep(list(rule$nodes), 4L)))
  weights <- Reduce(`*`, as.data.frame(expand.grid(rep(list(rule$weights), 4L))))
  center <- c(.2, -.4, .6, -.1)
  L <- rbind(c(.5, 0, 0, 0), c(.4, .3, 0, 0),
             c(-.2, .1, .7, 0), c(.3, -.2, .4, .6))
  coordinates <- sweep(standard %*% t(L), 2L, center, "+")
  base <- list(physical = exp, inverse = log,
    log_density = function(x) rowSums(dnorm(x, log = TRUE)),
    read = function(query) list(query = query, correction = rep(0, nrow(query))))
  transport <- tulpa:::.nl_outer_affine_transport(base, exp(coordinates), weights)
  expect_equal(transport$coordinate_center, center, tolerance = 1e-14)
  expect_equal(transport$coordinate_L, L, tolerance = 1e-14)
  query <- rbind(c(-2, .2, 1, -.3), c(.5, -.7, .9, 2))
  expected <- sweep(query %*% t(L), 2L, center, "+")
  expect_equal(log(transport$physical(query)), expected, tolerance = 1e-14)
  expect_equal(transport$inverse(exp(expected)), query, tolerance = 1e-14)
  expect_equal(transport$log_density(query), rowSums(dnorm(query, log = TRUE)) -
    sum(log(diag(L))), tolerance = 1e-14)
  read <- transport$read(query)
  expect_equal(log(transport$physical(read$query)), query, tolerance = 1e-14)
  expect_equal(read$correction, transport$log_density(read$query) -
    rowSums(dnorm(query, log = TRUE)), tolerance = 1e-14)
})

test_that("log softplus retains finite tail coordinates", {
  x <- c(-35, -5, 0, 5, 35)
  expect_equal(tulpa:::.nl_log_softplus(x), log(log1p(exp(x))), tolerance = 1e-14)
  expect_equal(tulpa:::.nl_log_softplus(c(-1000, 1000)), c(-1000, log(1000)))
})

test_that("an outer user design is validated and uses the shared evidence measure", {
  points <- matrix(c(.2, .5, 1, 2), ncol = 1, dimnames = list(NULL, "b1.sigma"))
  design <- list(points = points, weights = c(.3, .4, .2, .1))
  bad <- list(list(points = points), list(points = unname(points), weights = 1:4),
    list(points = points, weights = c(0, 0, 0, 0)),
    list(points = points, weights = c(1, -1, 1, 1)),
    list(points = points, weights = c(1, NA, 1, 1)))
  for (d in bad) expect_error(tulpa:::.nl_validate_outer_design(d), "outer_design")
  seen <- NULL
  solve <- function(cells, retain) {
    seen <<- cells
    list(log_marginal = dnorm(log(cells[, 1]), log = TRUE),
         log_hyperprior_axes = colnames(cells))
  }
  res <- list(theta_grid = points, weights = rep(.25, 4))
  fit <- tulpa:::.nl_user_integrate(res, solve, design, "log")
  expected <- dnorm(log(points[, 1])) * design$weights
  expect_identical(seen, points)
  expect_identical(fit$integration, "user")
  expect_equal(fit$weights, expected / sum(expected), tolerance = 1e-14)
  expect_equal(fit$log_evidence, log(sum(expected)), tolerance = 1e-14)
  singleton <- list(points = points[3, , drop = FALSE], weights = .7)
  one <- tulpa:::.nl_user_integrate(res, solve, singleton, "log")
  expect_equal(one$log_evidence, dnorm(0, log = TRUE) + log(.7), tolerance = 1e-14)
  expect_error(tulpa:::.nl_validate_outer_design(design, "other"), "match the outer axes")
  design$points[1, 1] <- 0
  expect_error(tulpa:::.nl_user_integrate(res, solve, design, "log"), "positive")
  for (tag in c("atanh", "logit01")) {
    outside <- list(points = points[1:2, , drop = FALSE], weights = c(1, 1))
    outside$points[1, 1] <- 1
    expect_error(tulpa:::.nl_user_integrate(res, solve, outside, tag), "domain")
  }
})

test_that("invalid user axes fail before an inner solve", {
  local_mocked_bindings(.nl_dispatch_multi = function(...) stop("solver reached"),
                        .package = "tulpa")
  block <- list(type = "iid", n_units = 2L, obs_idx = 1:2)
  run <- function(point, name = "b1.sigma", prior = block)
    tulpa_nested_laplace(c(0, 1), c(1L, 1L), matrix(1, 2, 1), prior = prior,
      control = list(outer_design = list(points = matrix(point, ncol = 1,
        dimnames = list(NULL, name)), weights = rep(1, length(point)))))
  expect_error(run(1, "other"), "match the outer axes")
  expect_error(run(0), "positive")
  block$sigma_grid <- 2
  expect_error(run(1, prior = block), "changes fixed axis")
})

test_that("hyperparameter draws preserve the positive conditional cell mixture", {
  axes <- list(c(1, 2, 3), c(2, 4))
  points <- matrix(c(1.25, 2.75, 2.5, 3.5), 2, 2,
    dimnames = list(NULL, c("b1.sigma", "b2.sigma")))
  P <- tulpa:::cpp_outer_tensor_partition(axes, points, c(.3, .7))
  fit <- list(theta_grid = as.matrix(expand.grid(axes)),
    integration = "aghq",
    outer_posterior = list(points = points, weights = c(.3, .7),
      partition = tulpa:::.nl_outer_coupling(P, as.numeric(Matrix::colSums(P)))))
  set.seed(137)
  cells <- sample.int(ncol(P), 40000, replace = TRUE, prob = Matrix::colSums(P))
  draws <- tulpa:::.nl_outer_hyper_draws(fit, cells)
  left <- abs(draws[, 1] - points[1, 1]) < 1e-12
  expect_lt(abs(mean(left) - .15), .01)
  expect_true(all(draws[, 1] >= points[1, 1] - 1e-12 & draws[, 1] <= points[2, 1] + 1e-12))
  for (cell in unique(cells)) {
    expected <- .5 * P[1, cell] / sum(P[, cell])
    expect_lt(abs(mean(left[cells == cell]) - expected), .04)
  }
  expect_identical(attr(draws, "cells"), cells)
})

test_that("unsupported adaptive designs are rejected before an inner solve", {
  local_mocked_bindings(.nl_dispatch_multi = function(...) stop("inner reached"),
    .package = "tulpa")
  expect_error(tulpa_nested_laplace(c(0, 1), c(1L, 1L), matrix(1, 2, 1),
    prior = list(type = "iid", n_units = 2L, obs_idx = 1:2),
    control = list(integration = "aghq")), "four free axes")
})

test_that("outer coupling preserves both masses with unvisited conditional cells", {
  P <- Matrix::sparseMatrix(i = 1:2, j = 1:2, x = c(.3, .7), dims = c(2L, 3L))
  coupled <- tulpa:::.nl_outer_coupling(P, c(.2, .4, .4))
  expect_equal(tulpa:::.nl_outer_coupling_mass(coupled, "row"), c(.3, .7), tolerance = 1e-12)
  expect_equal(tulpa:::.nl_outer_coupling_mass(coupled, "column"), c(.2, .4, .4), tolerance = 1e-12)
  expect_true(all(c(coupled$local@x, coupled$row, coupled$column) >= 0))
  expect_lte(coupled$residual, 1e-13)
})

test_that("nearest-cell coupling preserves the entropy equations when rows split", {
  source <- c(.1, .2, .3, .4, 0)
  target <- c(.2, .4, .4)
  P <- Matrix::sparseMatrix(i = 1:4, j = c(1L, 1L, 2L, 2L),
    x = source[1:4], dims = c(5L, 3L))
  coupled <- tulpa:::.nl_outer_coupling(P, target)
  mass <- as.numeric(Matrix::colSums(P))
  retained <- as.numeric(Matrix::colSums(coupled$local))
  global <- sum(coupled$row)
  completion <- (sum(abs(mass - target)) / 2)^2
  expect_equal(retained * completion * target * global,
    (mass - retained) * (target - retained), tolerance = 1e-13)
  expect_equal(tulpa:::.nl_outer_coupling_mass(coupled, "row"), source,
    tolerance = 1e-13)
  expect_equal(tulpa:::.nl_outer_coupling_mass(coupled, "column"), target,
    tolerance = 1e-13)
  compact <- tulpa:::.nl_outer_coupling(Matrix::sparseMatrix(
    i = seq_along(mass), j = seq_along(mass), x = mass), target)
  expect_equal(as.numeric(Matrix::colSums(compact$local)), retained,
    tolerance = 1e-13)
  expect_equal(compact$column * sum(compact$row), coupled$column * global,
    tolerance = 1e-13)
})

test_that("a supplied design reports exact Gaussian evidence and fixed-effect mixtures", {
  skip_on_cran()
  y <- c(-.6, .2, .5, 1.2, -.1, .7)
  u <- seq(-5, 1, length.out = 129)
  points <- matrix(exp(u), ncol = 1, dimnames = list(NULL, "b1.sigma"))
  quadrature <- rep(diff(u)[1], length(u)); quadrature[c(1, length(u))] <- quadrature[1] / 2
  fit <- tulpa_nested_laplace(y, rep(1L, length(y)), matrix(1, length(y), 1),
    prior = list(type = "iid", n_units = length(y), obs_idx = seq_along(y)),
    family = "gaussian", phi = .2,
    control = list(progress = FALSE, diagnose_skew = FALSE,
      outer_design = list(points = points, weights = quadrature)))
  v <- .2 + exp(2 * u)
  n <- length(y)
  anchor <- tulpa:::.nl_scale_anchor(); rate <- -log(anchor[2]) / anchor[1]
  lp <- -.5 * ((n - 1) * log(v) + log(v + n * 2.5^2) +
    sum((y - mean(y))^2) / v + n * mean(y)^2 / (v + n * 2.5^2) + n * log(2 * pi)) +
    log(rate) + u - rate * exp(u)
  lz <- tulpa:::.tulpa_logsumexp(lp + log(quadrature))
  w <- exp(lp + log(quadrature) - lz)
  expect_identical(fit$integration, "user")
  expect_equal(fit$theta_grid, points)
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

test_that("per-side transport preserves contours and its inverse", {
  radii <- rbind(c(1, 4), c(.01, 8))
  map <- tulpa:::.nl_outer_side_map(radii)
  x <- cbind(seq(-20, 20, length.out = 101), seq(-30, 30, length.out = 101))
  expect_equal(map$forward(matrix(-2, 1, 2))[1, ], -radii[, 1], tolerance = 1e-12)
  expect_equal(map$forward(matrix(2, 1, 2))[1, ], radii[, 2], tolerance = 1e-12)
  expect_equal(map$inverse(map$forward(x)), x, tolerance = 1e-9)
  expect_true(all(apply(map$forward(x), 2, diff) > 0))
  expect_true(all(is.finite(map$log_derivative(x))))
})

test_that("outer Hermite and CDF quadrature covers analytic posterior reads", {
  prior <- lapply(c("local", "regional"), function(name)
    list(type = "hsgp", name = name, prior_sigma = c(1, .1), prior_range = c(1, .1)))
  rate <- -log(.1)
  range_rate <- -log(.1) / 2
  gh <- tulpa:::gauss_hermite_prob(5L)
  gl <- tulpa:::.nl_gauss_legendre_unit(5L)
  design <- as.matrix(expand.grid(gl$x, gl$x, gh$nodes, gh$nodes))
  points <- cbind(exp(design[, 3:4]), (-log1p(-design[, 1]) / rate)^2,
                  range_rate / (-log1p(-design[, 2])))
  colnames(points) <- c("b1.sigma2", "b1.lengthscale", "b2.sigma2", "b2.lengthscale")
  weights <- apply(as.matrix(expand.grid(gl$w, gl$w, gh$weights, gh$weights)), 1, prod)
  res <- list(theta_grid = points, weights = weights, axis_offsets = c(0L, 2L, 4L),
    outer_mode_u = rep(0, 4), outer_mode_cov_u = diag(4))
  with_fixed <- FALSE
  solve <- function(cells, retain) {
    sigma <- sqrt(cells[, 3]); ell <- cells[, 4]
    list(log_marginal = rowSums(dnorm(log(cells[, 1:2, drop = FALSE]), log = TRUE)) +
      log(rate / 2) + log(sigma) - rate * sigma + log(range_rate) - log(ell) - range_rate / ell,
      log_hyperprior_axes = colnames(cells),
      grid_modes = if (with_fixed) lapply(log(cells[, 1]) + .2 * log(cells[, 2]), function(x) c(x)),
      grid_hessians = if (with_fixed) rep(list(matrix(1 / .3)), nrow(cells)))
  }
  fit <- tulpa:::.nl_aghq_integrate(res, solve, prior, tulpa:::.nl_aghq_config(list()))
  posterior <- fit$outer_posterior
  probs <- c(.025, .5, .975)
  expected <- rbind(qnorm(probs), qnorm(probs),
    log((-log1p(-probs) / rate)^2), log(range_rate / (-log(probs))))
  got <- t(vapply(seq_len(4), function(j) tulpa:::.nl_wtd_quantile(
    log(posterior$points[, j]), posterior$weights, probs, outside = "clamp"), numeric(3)))
  error <- abs(sweep(got - expected, 1, c(1, 1, 2 * pi / sqrt(6), pi / sqrt(6)), "/"))
  expect_true(fit$outer_quadrature$converged)
  expect_lte(max(error), fit$outer_quadrature$error$max)
  expect_lte(abs(fit$log_evidence), fit$outer_quadrature$error$log_evidence + 1e-12)
  expect_lt(abs(fit$log_evidence), .005)
  expect_equal(as.numeric(crossprod(fit$weights, log(fit$theta_grid))),
    as.numeric(crossprod(posterior$weights, log(posterior$points))), tolerance = 1e-6)
  with_fixed <- TRUE
  fit <- tulpa:::.nl_aghq_integrate(res, solve, prior,
    tulpa:::.nl_aghq_config(list(quad_max_level = 4L)))
  expect_true(fit$outer_quadrature$converged)
  fixed <- tulpa:::.nested_fixed_moments(fit)
  fixed <- tulpa:::.nl_gauss_mixture_summary(fixed$mu, fixed$var, fixed$w)
  expect_lte(max(abs(fixed$quantiles - qnorm(probs, sd = sqrt(1.34))) / sqrt(1.34)),
    fit$outer_quadrature$error$max)
  posterior <- fit$outer_posterior
  expect_equal(tulpa:::.nl_outer_coupling_mass(posterior$partition, "column"), unname(fit$weights), tolerance = 1e-12)
  expect_equal(tulpa:::.nl_outer_coupling_mass(posterior$partition, "row"), posterior$weights, tolerance = 1e-12)
})

test_that("quadrature diagnostics retain a summary row with supplied and adaptive designs", {
  states <- list(
    list(method = "user", n = 2L),
    list(method = "aghq_log_spline", n = 25L, n_evaluated = 30L,
      error = list(max = .04, log_evidence = .003), tol = .05,
      evidence_tol = .005, converged = TRUE))
  for (state in states) {
    fit <- tulpa:::.finalize_fit(list(draws = matrix(seq_len(20L), ncol = 1L,
      dimnames = list(NULL, "beta")), outer_quadrature = state),
      backend = "nested_laplace")
    d <- diagnostics(fit)
    summary <- attr(d, "summary")
    expect_equal(nrow(summary), 1L)
    expect_identical(summary$outer_integration, state$method)
    expect_false(any(c("qmc_converged", "qmc_error", "qmc_tol") %in% names(summary)))
    if (identical(state$method, "user")) {
      expect_true(is.na(summary$outer_converged))
      expect_true(is.na(summary$outer_error))
      expect_output(print(d), "User quadrature")
    } else {
      expect_true(summary$outer_converged)
      expect_equal(summary$outer_error, .04)
      expect_equal(summary$outer_evidence_error, .003)
      expect_output(print(d), "Adaptive quadrature")
    }
  }
})

test_that("a multiscale HSGP fit retains its shared posterior at the public door", {
  skip_on_cran()
  set.seed(1371)
  d <- data.frame(lon = runif(12), lat = runif(12), x = rnorm(12),
                  y = rpois(12, 2), group = factor(rep(1:3, 4)))
  formulas <- list(y ~ x, y ~ x + (1 | group))
  for (i in seq_along(formulas)) {
    fit <- tulpa(formulas[[i]], data = d, family = "poisson",
      spatial = spatial_multiscale(~lon + lat, approx = "hsgp", m = 3),
      control = list(progress = FALSE, quad_max_level = 3L,
        quad_read_nodes = 2048L, quad_read_max_nodes = 2048L,
        quad_tol = 5, quad_evidence_tol = 1))
    inner <- if (!is.null(fit$nested)) fit$nested else if (!is.null(fit$joint_fit)) fit$joint_fit else fit
    expect_identical(inner$integration, "aghq")
    expect_identical(inner$outer_quadrature$read_max_nodes, 2048L)
    expect_true(all(inner$weights >= 0))
    expect_equal(tulpa:::.nl_outer_coupling_mass(inner$outer_posterior$partition, "column"),
                 unname(inner$weights), tolerance = 1e-12)
    expect_true(is.finite(inner$log_evidence))
    draws <- tulpa_hyper_draws(inner, n = 200L)
    expect_equal(dim(draws), c(200L, 3L + i))
    expect_true(all(draws > 0))
    expect_true(all(attr(draws, "within_cell") == "chord"))
    expect_output(print(inner), "Adaptive quadrature")
  }
})
