test_that("nested normal and bounded rules retain nodes and integrate moments", {
  for (normal in c(FALSE, TRUE)) {
    rules <- tulpa:::.nl_sparse_rules(normal)
    for (j in seq_along(rules)) {
      rule <- rules[[j]]
      expect_equal(sum(rule$w), 1, tolerance = 1e-14)
      expect_equal(sum(rule$x * rule$w), if (normal) 0 else .5, tolerance = 1e-14)
      if (j > 1L) {
        expect_true(all(rules[[j - 1L]]$x %in% rule$x))
        expect_equal(sum(rule$x^2 * rule$w), if (normal) 1 else 1 / 3,
                     tolerance = 1e-13)
      }
    }
  }
})

test_that("Smolyak differences integrate interaction polynomials in five dimensions", {
  rules <- lapply(c(FALSE, FALSE, TRUE, TRUE, TRUE), tulpa:::.nl_sparse_rules)
  indices <- as.matrix(expand.grid(rep(list(1:3), 5)))
  indices <- indices[rowSums(indices - 1L) <= 2L, , drop = FALSE]
  value <- 0
  for (i in seq_len(nrow(indices))) {
    grid <- tulpa:::.nl_sparse_tensor(rules, indices[i, ])
    x <- grid$points
    f <- 1 + x[, 1]^2 + x[, 3]^2 * x[, 4]^2 + x[, 5]^4
    value <- value + sum(grid$difference * f)
  }
  expect_equal(value, 1 + 1 / 3 + 1 + 3, tolerance = 1e-12)
})

test_that("sparse cache solves each nested node once across subspaces", {
  rules <- lapply(c(FALSE, TRUE, TRUE, FALSE, TRUE), tulpa:::.nl_sparse_rules)
  seen <- matrix(numeric(0), 0, 5)
  solve <- function(cells, retain) {
    seen <<- rbind(seen, cells)
    list(log_marginal = -rowSums(cells^2), grid_modes = as.list(cells[, 1]))
  }
  cache <- tulpa:::.nl_sparse_cache(rules, identity, solve)
  for (index in list(c(2, 2, 2, 2, 2), c(3, 2, 2, 2, 2),
                     c(2, 3, 2, 2, 2), c(3, 2, 2, 2, 2))) cache$tensor(index)
  expect_equal(nrow(unique(as.data.frame(seen))), nrow(seen))
  expect_equal(cache$state()$points, seen)
  expect_equal(cache$state()$native$log_marginal, -rowSums(seen^2))
})

test_that("sparse log interpolation preserves additive cubic fields", {
  rules <- lapply(c(FALSE, TRUE, TRUE, FALSE, TRUE), tulpa:::.nl_sparse_rules)
  indices <- as.matrix(expand.grid(rep(list(1:3), 5)))
  indices <- lapply(which(rowSums(indices - 1L) <= 2L), function(i) indices[i, ])
  f <- function(x) 2 + x[, 1]^3 - x[, 3]^2 + x[, 5]^3
  cache <- tulpa:::.nl_sparse_cache(rules, identity,
      function(cells, retain) list(log_marginal = f(cells)))
  for (index in indices) cache$tensor(index)
  query <- cbind(seq(.2, .8, length.out = 17), seq(-1, 1, length.out = 17),
    seq(-2, 2, length.out = 17), rep(.5, 17), seq(-1.5, 1.5, length.out = 17))
  got <- tulpa:::.nl_sparse_interpolate(tulpa:::.nl_sparse_combination(indices), cache,
    matrix(cache$state()$native$log_marginal), query)
  expect_equal(as.numeric(got), f(query), tolerance = 1e-11)
  expect_true(all(exp(got) > 0))
})

test_that("read interpolation caches probe rows and nested scramble prefixes exactly", {
  rules <- lapply(c(FALSE, FALSE), tulpa:::.nl_sparse_rules)
  cache <- tulpa:::.nl_sparse_cache(rules, identity,
    function(cells, retain) list(log_marginal = rowSums(cells)))
  grid <- cache$tensor(c(3L, 3L))
  smaller <- cache$tensor(c(2L, 2L))
  points <- cache$state()$points
  values <- cbind(points[, 1]^3 + points[, 2]^2, points[, 1] * points[, 2])
  design <- function(n) do.call(rbind, lapply(1:8, function(g)
    tulpa:::cpp_outer_read_design(n / 8L, 2L, 891L + g)))
  small <- design(128L); large <- design(256L)
  original <- tulpa:::cpp_outer_tensor_interpolate
  expected_small <- original(grid$nodes, values[grid$rows, ], small)
  expected_large <- original(grid$nodes, values[grid$rows, ], large)
  expected_other <- original(smaller$nodes, values[smaller$rows, ], large)
  seen <- integer()
  local_mocked_bindings(cpp_outer_tensor_interpolate = function(nodes, values, query, cubic = TRUE) {
    seen <<- c(seen, nrow(query))
    original(nodes, values, query, cubic)
  }, .package = "tulpa")
  predictions <- new.env(parent = emptyenv())
  combination <- list(list(index = c(3L, 3L), coefficient = 1))
  probe <- c(1L, 1L, 2L, 65L)
  got <- tulpa:::.nl_sparse_interpolate(combination, cache, values, small, predictions,
                                        rows = probe, groups = 8L)
  expect_identical(got, expected_small[probe, , drop = FALSE])
  expect_identical(seen, 3L)
  got <- tulpa:::.nl_sparse_interpolate(combination, cache, values, small, predictions, groups = 8L)
  expect_identical(got, expected_small)
  expect_identical(seen, c(3L, 125L))
  got <- tulpa:::.nl_sparse_interpolate(combination, cache, values, large, predictions, groups = 8L)
  expect_identical(got, expected_large)
  expect_identical(seen, c(3L, 125L, 128L))
  got <- tulpa:::.nl_sparse_interpolate(combination, cache, values, small, predictions, groups = 8L)
  expect_identical(got, expected_small)
  expect_identical(seen, c(3L, 125L, 128L))
  combination[[1L]]$index <- c(2L, 2L)
  got <- tulpa:::.nl_sparse_interpolate(combination, cache, values, large, predictions,
                                        rows = probe, groups = 8L)
  expect_identical(got, expected_other[probe, , drop = FALSE])
  expect_identical(seen, c(3L, 125L, 128L, 3L))
})

test_that("positive recombination recovers target mixture probabilities", {
  x <- seq(-3, 3, length.out = 31)
  features <- cbind(x, x^2, pnorm(.7, x, .5))
  mass <- dnorm(x, .3, .9); mass <- mass / sum(mass)
  fit <- tulpa:::.nl_sparse_positive_weights(features,
    as.numeric(crossprod(mass, features)), rep(1, length(x)))
  expect_true(all(fit$weights >= 0))
  expect_equal(sum(fit$weights), 1, tolerance = 1e-14)
  expect_equal(as.numeric(crossprod(fit$weights, features)),
    as.numeric(crossprod(mass, features)), tolerance = 1e-8)
  expect_lte(fit$residual, 1e-9)
})

test_that("nearest partitions preserve positive masses in five dimensions", {
  points <- as.matrix(expand.grid(rep(list(c(-1, 0, 1)), 5)))
  query <- rbind(points[1:5, ], rep(.2, 5))
  mass <- seq_len(nrow(query)); mass <- mass / sum(mass)
  P <- tulpa:::cpp_outer_nearest_partition(points, query, mass)
  expect_true(all(P@x >= 0))
  expect_equal(as.numeric(Matrix::rowSums(P)), mass, tolerance = 1e-14)
  expect_equal(as.numeric(Matrix::colSums(P))[1:5], mass[1:5], tolerance = 1e-14)
})

test_that("positive moment projection reports infeasible support with finite masses", {
  fit <- tulpa:::.nl_sparse_positive_weights(matrix(c(0, 1), ncol = 1L), 2, c(.5, .5))
  expect_true(all(is.finite(fit$weights) & fit$weights >= 0))
  expect_equal(sum(fit$weights), 1, tolerance = 1e-14)
  expect_gt(fit$residual, .5)
  expect_equal(sum(fit$weights * c(0, 1)), 1, tolerance = 1e-9)
})

test_that("read budgets follow the binding quantity and measured sampling rate", {
  expect_error(tulpa:::.nl_aghq_config(list(quad_read_max_nodes = 129L)),
               "divisible by eight")
  expect_error(tulpa:::.nl_aghq_config(list(quad_read_nodes = 1024L,
                                        quad_read_max_nodes = 512L)),
               "must not exceed")
  action <- function(...) tulpa:::.nl_sparse_read_action(...)$action
  expect_identical(tulpa:::.nl_sparse_read_action(65536, 524288,
    c(.01, .03), c(.1, .001), c(.05, .005))$goal, 2L)
  expect_identical(action(4096, 524288, c(.01, .1), c(.1, .001), c(.05, .02)),
                   "refine")
  expect_identical(action(65536, 524288, c(.005, .001), c(.1, .001), c(.05, .02)),
                   "grow")
  expect_identical(action(131072, 524288, c(.005, .001), c(.09, .001), c(.05, .02),
                         previous = list(nodes = 65536, error = c(.1, .001))), "refine")
  expect_identical(action(131072, 524288, c(.005, .001), c(.06, .001), c(.05, .02),
                         previous = list(nodes = 65536, error = c(.1, .001))), "grow")
  expect_identical(action(524288, 524288, c(.005, .001), c(.06, .001), c(.05, .02)),
                   "refine")
  expect_identical(action(65536, 524288, c(.045, .001), c(.02, .001), c(.05, .02)),
                   "refine")
  expect_identical(action(131072, 524288, c(.005, .001), c(Inf, .001), c(.05, .02),
                         previous = list(nodes = 65536, error = c(Inf, .001))), "refine")
  expect_identical(action(524288, 524288, c(.005, .001), c(.06, .001), c(.05, .02),
                         refit = TRUE), "refine")
  expect_identical(action(65536, 524288, c(.08, .001), c(.1, .001), c(.05, .02),
                         refit = TRUE), "refine")
  expect_identical(action(131072, 524288, c(.03, .001), c(.03, .001), c(.05, .02),
                         previous = list(nodes = 65536, error = c(.05, .001))), "grow")
  expect_identical(action(65536, 524288, c(.04, .001), c(.08, .001), c(.05, .02)),
                   "refine")
  expect_identical(action(131072, 524288, c(.005, .001), c(.09, .001), c(.05, .02),
                         previous = list(nodes = 65536, error = c(.1, .001)),
                         refit = TRUE), "refresh")
  expect_identical(action(196608, 196608, c(.001, .001), c(.075, .001), c(.05, .02),
                         previous = list(nodes = 131072, error = c(.1, .001)),
                         refit = TRUE), "refine")
})

test_that("adaptive quadrature resolves a five-axis analytic posterior and mixture", {
  skip_on_cran()
  prior <- lapply(c("local", "regional"), function(name)
    list(type = "hsgp", name = name, prior_sigma = c(1, .1), prior_range = c(1, .1)))
  prior[[3L]] <- list(type = "iid", name = "group", sigma_grid = exp(c(-2, 0, 2)))
  rate <- -log(.1)
  range_rate <- rate / 2
  gh <- tulpa:::gauss_hermite_prob(3L)
  gl <- tulpa:::.nl_gauss_legendre_unit(3L)
  design <- as.matrix(expand.grid(gl$x, gl$x, gh$nodes, gh$nodes, gh$nodes))
  points <- cbind(exp(design[, 3:4]), (-log1p(-design[, 1]) / rate)^2,
    range_rate / (-log1p(-design[, 2])), exp(design[, 5]))
  colnames(points) <- c("b1.sigma2", "b1.lengthscale", "b2.sigma2", "b2.lengthscale", "b3.sigma")
  weights <- Reduce(`*`, as.data.frame(expand.grid(gl$w, gl$w, gh$weights, gh$weights, gh$weights)))
  res <- list(theta_grid = points, weights = weights, axis_offsets = c(0L, 2L, 4L, 5L),
    outer_mode_u = rep(0, 5), outer_mode_cov_u = diag(5))
  count <- 0L
  probe_keys <- character()
  solve <- function(cells, retain) {
    count <<- count + nrow(cells)
    if (!retain) probe_keys <<- c(probe_keys, tulpa:::.nl_outer_point_key(cells))
    sigma <- sqrt(cells[, 3]); ell <- cells[, 4]
    list(log_marginal = rowSums(dnorm(log(cells[, c(1, 2, 5), drop = FALSE]), log = TRUE)) +
      log(rate / 2) + log(sigma) - rate * sigma + log(range_rate) - log(ell) - range_rate / ell,
      log_hyperprior_axes = colnames(cells),
      grid_modes = as.list(log(cells[, 1]) + .2 * log(cells[, 2]) + .3 * log(cells[, 5])),
      grid_hessians = rep(list(matrix(1 / .3)), nrow(cells)))
  }
  transport <- tulpa:::.nl_aghq_transform(res, prior, solve, 7)
  cloud <- points
  cloud[, 5] <- exp(1.2 + .4 * log(points[, 1]) - .3 * log(points[, 2]) + .5 * log(points[, 5]))
  adapted <- transport$adapt_read(cloud, weights)
  query <- matrix(c(.2, .7, .1, -.3, .5), 1)
  read <- transport$read(query)
  physical <- log(transport$physical(read$query))
  expected <- sweep(query[, 3:5, drop = FALSE] %*% t(adapted$L), 2L, adapted$center, "+")
  expect_equal(unname(physical[, c(1, 2, 5), drop = FALSE]), unname(expected), tolerance = 1e-12)
  jacobian <- vapply(seq_len(5L), function(j) {
    step <- matrix(0, 1, 5); step[j] <- 1e-5
    upper <- log(transport$physical(transport$read(query + step)$query))
    lower <- log(transport$physical(transport$read(query - step)$query))
    as.numeric(upper - lower) / 2e-5
  }, numeric(5))
  independent <- transport$log_density(read$query) +
    as.numeric(determinant(jacobian, logarithm = TRUE)$modulus) - sum(dt(query[, 3:5], 7, log = TRUE))
  expect_equal(read$correction, independent, tolerance = 1e-8)
  count <- 0L
  probe_keys <- character()
  fit <- tulpa:::.nl_aghq_integrate(res, solve, prior, tulpa:::.nl_aghq_config(list()))
  state <- fit$outer_quadrature
  expect_true(state$converged)
  expect_identical(state$construction, "adaptive_smolyak")
  expect_equal(state$n_evaluated, count)
  expect_identical(anyDuplicated(probe_keys), 0L)
  expect_lt(abs(fit$log_evidence), .005)
  probs <- c(.025, .5, .975)
  exact <- rbind(qnorm(probs), qnorm(probs), log((-log1p(-probs) / rate)^2),
    log(range_rate / (-log(probs))), qnorm(probs))
  scale <- c(1, 1, 2 * pi / sqrt(6), pi / sqrt(6), 1)
  for (i in seq_along(state$reads)) {
    read <- state$reads[[i]]
    hyper_error <- max(abs(sweep(t(read$hyper) - exact, 1L, scale, "/")))
    fixed_error <- max(abs(read$fixed - qnorm(probs, sd = sqrt(1.43))) / sqrt(1.43))
    expect_gte(read$error$max, max(hyper_error, fixed_error))
    expect_gte(read$error$log_evidence + 1e-12, abs(state$history[i, "log_evidence"]))
  }
  expect_true(all(fit$weights >= 0))
  final <- tail(state$reads, 1L)[[1L]]
  expect_lte(max(abs(final$fixed_mean - final$fixed_interpolated_mean)), 1e-6)
  expect_lte(max(abs(final$fixed_sd - final$fixed_interpolated_sd)), 1e-6)
  expect_equal(tulpa:::.nl_outer_coupling_mass(fit$outer_posterior$partition, "column"),
    unname(fit$weights), tolerance = 1e-12)

  local_mocked_bindings(.nl_outer_read_error = function(u, log_weight, replicate, proposed,
                                                       fixed = NULL, ...) {
    list(hyper_quantiles = matrix(.2, 3L, ncol(u)),
      fixed_quantiles = matrix(.2, 3L, ncol(fixed$mu)), log_evidence = .001,
      max = .2, complete = TRUE)
  }, .package = "tulpa")
  bounded <- tulpa:::.nl_aghq_integrate(res, solve, prior,
    tulpa:::.nl_aghq_config(list(quad_read_nodes = 128L, quad_read_max_nodes = 512L,
                              quad_max_level = 3L, quad_evidence_tol = 1)))
  state <- bounded$outer_quadrature
  expect_false(state$converged)
  expect_true(is.finite(state$error$max))
  expect_gte(state$error$max, .2)
  expect_lte(max(state$read_history[, "read_nodes"]), 512L)
  expect_gt(length(unique(state$read_history[, "points"])), 3L)
})
