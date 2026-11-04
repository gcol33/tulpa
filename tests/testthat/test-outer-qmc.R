.qmc_l_target <- function(U) {
  A <- sweep(U, 2L, c(-4, 0, 0, 0))
  B <- sweep(U, 2L, c(0, 0, -4, 0))
  a <- rowSums(dnorm(A, sd = .8, log = TRUE))
  b <- rowSums(dnorm(B, sd = .8, log = TRUE))
  m <- pmax(a, b)
  m + log(exp(a - m) + exp(b - m)) - log(2)
}

test_that("importance point counts are validated before fitting", {
  block <- list(type = "iid", n_units = 2L, obs_idx = 1:2)
  for (n in list(0, 1, NA_real_, Inf, "2", 2.5, .Machine$integer.max + 1)) {
    expect_error(tulpa_nested_laplace(c(0, 1), c(1L, 1L), matrix(1, 2L, 1L),
      prior = list(block), control = list(integration = "qmc", qmc_nodes = n)),
      "qmc_nodes.*integer")
  }
})

test_that("a point design retains its fixed hyperparameters", {
  for (tg in list(matrix(7, 1L, 1L, dimnames = list(NULL, "a")),
                  matrix(numeric(0), 1L, 0L))) {
    res <- list(theta_grid = tg, log_marginal = 0, weights = 1, integration = "grid")
    expect_identical(tulpa:::.nl_qmc_integrate(res, function(...) stop("unexpected solve"),
      rep("identity", ncol(tg)), n = 16L), res)
  }
  fit <- tulpa_nested_laplace(seq(-.5, .5, length.out = 6L), rep(1L, 6L),
    matrix(1, 6L, 1L), family = "gaussian", phi = .1,
    prior = list(type = "iid", n_units = 6L, obs_idx = 1:6, sigma_grid = 2),
    control = list(progress = FALSE, integration = "qmc", qmc_nodes = 16L,
                   auto_recenter = FALSE, axis_refine = "none"))
  expect_equal(as.numeric(fit$theta_grid), 2)
  expect_equal(as.numeric(fit$weights), 1)
})

test_that("importance quadrature resolves both arms of a continuous posterior", {
  skip_on_cran()
  U <- as.matrix(expand.grid(a = seq(-6, 2, by = 2), b = seq(-2, 2, by = 1),
                            c = seq(-6, 2, by = 2), d = seq(-2, 2, by = 1)))
  lm <- .qmc_l_target(U)
  res <- list(theta_grid = U, log_marginal = lm,
              weights = tulpa:::.nl_normalise_weights_safe(lm))
  out <- tulpa:::.nl_qmc_integrate(res, function(cells, retain)
    list(log_marginal = .qmc_l_target(cells),
         log_hyperprior_axes = colnames(U)), rep("identity", 4L), n = 32768L)
  expect_identical(out$integration, "qmc")
  expect_equal(nrow(out$theta_grid), 32768L)
  expect_equal(out$log_evidence, 0, tolerance = .015)
  mass <- vapply(out$outer_qmc$proposal, `[[`, numeric(1), "mass")
  expect_equal(mass * 32768L, round(mass * 32768L), tolerance = 1e-10)
  expect_equal(sum(mass), 1, tolerance = 1e-15)
  for (j in 1:4) {
    means <- if (j %in% c(1L, 3L)) c(-4, 0) else c(0, 0)
    exact <- vapply(c(.025, .5, .975), function(p)
      uniroot(function(x) mean(pnorm(x, means, .8)) - p, c(-10, 10))$root,
      numeric(1))
    got <- tulpa:::.nl_wtd_quantile(out$theta_grid[, j], out$weights,
                                   c(.025, .5, .975), outside = "clamp")
    expect_lt(max(abs(got - exact)), .06)
  }
})

test_that("importance quadrature retains fixed axes and stated bounds", {
  skip_on_cran()
  U <- cbind(a = seq(-2, 2, length.out = 31L), b = 7)
  res <- list(theta_grid = U, log_marginal = dnorm(U[, 1], log = TRUE),
              weights = rep(1 / nrow(U), nrow(U)))
  bounds <- rbind(c(-1, 7), c(1, 7))
  solved <- 0L
  out <- tulpa:::.nl_qmc_integrate(res, function(cells, retain) {
    solved <<- solved + nrow(cells)
    list(log_marginal = dnorm(cells[, 1L], log = TRUE), log_hyperprior_axes = "a")
  },
    rep("identity", 2L), n = 8192L, bounds = bounds)
  expect_true(all(out$theta_grid[, 1L] >= -1 & out$theta_grid[, 1L] <= 1))
  expect_true(all(out$theta_grid[, 2L] == 7))
  expect_equal(out$log_evidence, log(pnorm(1) - pnorm(-1)), tolerance = .02)
  expect_identical(out$outer_qmc$n_evaluated, solved)
})

test_that("chunked solves retain cell arrays and model metadata", {
  cells <- cbind(a = seq_len(5000L))
  solve <- function(x, retain) {
    v <- as.numeric(x[, 1L])
    list(log_marginal = v, modes = cbind(v, v^2), n_grid = nrow(x), n_x = 2L,
      model = list(family = "gaussian"), grid_hessians = lapply(v, function(z) matrix(z, 1L)))
  }
  out <- tulpa:::.nl_qmc_solve(cells, solve, TRUE)
  expect_equal(out$log_marginal, cells[, 1L])
  expect_equal(out$modes, solve(cells, TRUE)$modes)
  expect_identical(out$model, list(family = "gaussian"))
  expect_identical(out$n_x, 2L)
  expect_equal(out$n_grid, nrow(cells))
  expect_equal(vapply(out$grid_hessians, as.numeric, numeric(1)), cells[, 1L])
})

test_that("importance points retain a covariance design's density measure", {
  skip_on_cran()
  C <- as.matrix(expand.grid(L11 = seq(-3, 3, length.out = 7L),
                            L21 = seq(-2, 2, length.out = 7L),
                            L22 = seq(-5, 1, length.out = 7L)))
  design <- list(design = "cpc", p = 2L)
  G <- tulpa:::.hp_logchol_from_coords(C, design)
  mu <- c(0, .2, -2); sigma <- c(.8, .6, .9)
  density <- function(u) {
    z <- sweep(sweep(u, 2L, mu), 2L, sigma, "/")
    rowSums(dnorm(z, log = TRUE)) - sum(log(sigma))
  }
  lm <- density(C)
  res <- list(theta_grid = G, log_marginal = lm,
              weights = tulpa:::.nl_normalise_weights_safe(lm))
  out <- tulpa:::.nl_qmc_integrate(res, function(cells, retain)
    list(log_marginal = density(tulpa:::.hp_logchol_coords(cells, design)),
         log_hyperprior_axes = colnames(G)), rep("identity", 3L), n = 32768L)
  expect_equal(out$log_evidence, 0, tolerance = .015)
  expect_equal(out$outer_qmc$design_columns, colnames(G))
  got <- tulpa:::.hp_logchol_coords(out$theta_grid, design)
  for (j in 1:3) {
    q <- tulpa:::.nl_wtd_quantile(got[, j], out$weights, c(.025, .5, .975),
                                 outside = "clamp")
    expect_lt(max(abs(q - qnorm(c(.025, .5, .975), mu[j], sigma[j]))), .06)
  }
})
