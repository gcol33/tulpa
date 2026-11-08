test_that("common spline bases preserve sparse sums and clamped boundaries", {
  set.seed(902)
  for (d in c(2L, 4L, 5L)) {
    rules <- lapply(c(FALSE, FALSE, rep(TRUE, d - 2L)), tulpa:::.nl_sparse_rules)
    all <- as.matrix(expand.grid(rep(list(1:3), d)))
    indices <- lapply(which(rowSums(all - 1L) <= 2L), function(i) all[i, ])
    cache <- tulpa:::.nl_sparse_cache(rules, identity,
      function(cells, retain) list(log_marginal = rowSums(sin(cells))))
    for (index in indices) cache$tensor(index)
    points <- cache$state()$points
    values <- cbind(rowSums(sin(points)) + points[, 1] * points[, 2],
      rowSums(points^2), rep(1e6, nrow(points)))
    query <- matrix(runif(127L * d, -.5, 1.5), 127L, d)
    if (d > 2L) query[, 3:d] <- matrix(runif(127L * (d - 2L), -9, 9), 127L)
    combination <- tulpa:::.nl_sparse_combination(indices)
    expected <- matrix(0, nrow(query), ncol(values))
    for (term in combination) {
      grid <- cache$tensor(term$index)
      expected <- expected + term$coefficient * tulpa:::cpp_outer_tensor_interpolate(
        grid$nodes, values[grid$rows, , drop = FALSE], query)
    }
    previous <- NULL
    for (budget in c(16L, 65536L)) {
      compiled <- tulpa:::.nl_sparse_compile(combination, cache, values, budget, previous)
      got <- tulpa:::cpp_outer_sparse_evaluate(compiled$pointer, query)
      expect_lt(max(abs(got[, 1:2] - expected[, 1:2])), 1e-9)
      expect_identical(got[, 3], rep(1e6, nrow(query)))
      if (!is.null(previous)) expect_identical(compiled$pointer, previous$pointer)
      previous <- compiled
    }
    tulpa:::cpp_outer_sparse_clear(compiled$pointer)
    expect_error(tulpa:::cpp_outer_sparse_evaluate(compiled$pointer, query), "dimensions")
  }
})

test_that("common-basis compilation reproduces additive cubics and centered constants", {
  d <- 4L
  rules <- lapply(c(FALSE, FALSE, TRUE, TRUE), tulpa:::.nl_sparse_rules)
  all <- as.matrix(expand.grid(rep(list(1:3), d)))
  indices <- lapply(which(rowSums(all - 1L) <= 2L), function(i) all[i, ])
  cache <- tulpa:::.nl_sparse_cache(rules, identity,
    function(cells, retain) list(log_marginal = rowSums(cells)))
  for (index in indices) cache$tensor(index)
  points <- cache$state()$points
  values <- cbind(2 + rowSums(points^3), rep(1e6, nrow(points)))
  query <- cbind(seq(.1, .9, length.out = 61), seq(.9, .1, length.out = 61),
                 seq(-3, 3, length.out = 61), seq(2, -2, length.out = 61))
  compiled <- tulpa:::.nl_sparse_compile(tulpa:::.nl_sparse_combination(indices), cache,
                                         values, 65536L)
  got <- tulpa:::cpp_outer_sparse_evaluate(compiled$pointer, query)
  expect_equal(got[, 1L], 2 + rowSums(query^3), tolerance = 1e-10)
  expect_identical(got[, 2L], rep(1e6, nrow(query)))
  delta <- tulpa:::.nl_sparse_compile(tulpa:::.nl_sparse_combination(list(c(2, 2, 1, 1))),
                                     cache, values, 65536L, compiled)
  expect_identical(tulpa:::cpp_outer_sparse_evaluate(delta$pointer, query)[, 2L],
                   rep(0, nrow(query)))
  tulpa:::cpp_outer_sparse_clear(delta$pointer)
})
