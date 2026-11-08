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

test_that("adaptive quadrature resolves a five-axis analytic posterior and mixture", {
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
})
