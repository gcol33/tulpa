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
  for (tol in list(0, -1, NA_real_, Inf, "0.01", c(.01, .02))) {
    expect_error(tulpa_nested_laplace(c(0, 1), c(1L, 1L), matrix(1, 2L, 1L),
      prior = list(block), control = list(integration = "qmc", qmc_tol = tol)),
      "qmc_tol.*positive finite")
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
         log_hyperprior_axes = colnames(U)), rep("identity", 4L), n = 32768L,
    tol = .01)
  expect_identical(out$integration, "qmc")
  expect_lte(nrow(out$theta_grid), 32768L)
  expect_equal(out$log_evidence, 0, tolerance = .015)
  mass <- vapply(out$outer_qmc$proposal, `[[`, numeric(1), "mass")
  expect_equal(mass * out$outer_qmc$n, round(mass * out$outer_qmc$n), tolerance = 1e-10)
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

test_that("scrambled sequences extend exactly and preserve the caller's RNG", {
  set.seed(8L)
  rng <- .Random.seed
  full <- tulpa:::.nl_qmc_uniform(1024L, 4L, seed = 19L)
  parts <- rbind(tulpa:::.nl_qmc_uniform(512L, 4L, seed = 19L),
    tulpa:::.nl_qmc_uniform(512L, 4L, start = 512L, seed = 19L))
  expect_identical(parts, full)
  expect_identical(.Random.seed, rng)
  expect_false(identical(full, tulpa:::.nl_qmc_uniform(1024L, 4L, seed = 20L)))
})

test_that("Monte Carlo errors give nonfinite nodes zero mass in the original denominator", {
  u <- cbind(a = seq(-2, 2, length.out = 64L), b = sin(seq_len(64L)))
  lw <- rep(0, 64L)
  group <- rep(1:8, each = 8L)
  out <- list(grid_modes = lapply(seq_len(64L), function(i) u[i, ]),
    grid_hessians = rep(list(diag(c(1, 2))), 64L))
  bad <- 2:4
  lw[bad] <- c(NA_real_, Inf, -Inf)
  u[bad, ] <- NA_real_
  out$grid_modes[bad] <- rep(list(rep(NA_real_, 2L)), length(bad))
  out$grid_hessians[bad] <- rep(list(matrix(NA_real_, 2L, 2L)), length(bad))
  keep <- setdiff(seq_len(64L), bad)
  expected <- tulpa:::.nl_qmc_error(u[keep, ], lw[keep], group[keep],
    lapply(out, function(x) x[keep]), rep(8, 8L))
  got <- tulpa:::.nl_qmc_error(u, lw, group, out, rep(8, 8L))
  expect_equal(got, expected, tolerance = 1e-14)
  counts <- c(5, rep(8, 7L))
  expect_equal(got$log_evidence, qt(.975, 7) * sd(counts) / sqrt(8) / mean(counts),
               tolerance = 1e-14)
  expect_true(is.finite(got$max))
})

test_that("stratified proposal blocks extend each component's scrambled sequence", {
  proposal <- lapply(c(.35, .65), function(mass)
    list(mu = c(0, 1), scale = c(1, 2),
         mass = mass, df = 5, logdet = log(2)))
  proposal <- tulpa:::.nl_qmc_nodes(proposal, 512L)$proposal
  a <- tulpa:::.nl_qmc_nodes(proposal, 512L, seed = 4L)
  b <- tulpa:::.nl_qmc_nodes(proposal, 512L, start = 512L, seed = 4L)
  full <- tulpa:::.nl_qmc_nodes(proposal, 1024L, seed = 4L)
  sort_rows <- function(x) x[order(x[, 1], x[, 2]), , drop = FALSE]
  expect_identical(sort_rows(rbind(a$u, b$u)), sort_rows(full$u))
  expect_identical(sort_rows(cbind(rbind(a$u, b$u), c(a$log_q, b$log_q))),
                   sort_rows(cbind(full$u, full$log_q)))
})

test_that("an adaptive design stops at its error target and tighter targets grow it", {
  skip_on_cran()
  U <- as.matrix(expand.grid(a = seq(-4, 4, by = 1), b = seq(-4, 4, by = 1)))
  density <- function(u) rowSums(dnorm(u, log = TRUE))
  res <- list(theta_grid = U, log_marginal = density(U),
              weights = tulpa:::.nl_normalise_weights_safe(density(U)))
  solve <- function(cells, retain) list(log_marginal = density(cells),
                                        log_hyperprior_axes = colnames(U))
  loose <- tulpa:::.nl_qmc_integrate(res, solve, rep("identity", 2L), tol = .04)
  tight <- tulpa:::.nl_qmc_integrate(res, solve, rep("identity", 2L), tol = .004)
  expect_true(loose$outer_qmc$converged)
  expect_lt(loose$outer_qmc$n, tight$outer_qmc$n)
  expect_lte(loose$outer_qmc$error$max, .04)
  expect_equal(loose$log_evidence, 0, tolerance = .015)
  expect_equal(tight$log_evidence, 0, tolerance = .015)
})

test_that("a capped correlated design reports its remaining Monte Carlo error", {
  skip_on_cran()
  covariance <- diag(4L)
  covariance[1, 2] <- covariance[2, 1] <- .98
  covariance[3, 4] <- covariance[4, 3] <- -.98
  R <- chol(covariance)
  inverse <- backsolve(R, diag(4L))
  density <- function(u) rowSums(dnorm(u %*% inverse, log = TRUE)) - sum(log(diag(R)))
  U <- as.matrix(expand.grid(a = -3:3, b = -3:3, c = -3:3, d = -3:3))
  res <- list(theta_grid = U, log_marginal = density(U),
    weights = tulpa:::.nl_normalise_weights_safe(density(U)))
  out <- tulpa:::.nl_qmc_integrate(res, function(cells, retain)
    list(log_marginal = density(cells), log_hyperprior_axes = colnames(U)),
    rep("identity", 4L), n = 65536L)
  expect_false(out$outer_qmc$converged)
  expect_gt(out$outer_qmc$error$max, out$outer_qmc$tol)
  expect_equal(out$outer_qmc$n, 65536L)
  expect_equal(out$log_evidence, 0, tolerance = .015)
  for (j in 1:4) {
    q <- tulpa:::.nl_wtd_quantile(out$theta_grid[, j], out$weights,
      c(.025, .5, .975), outside = "clamp")
    expect_lt(max(abs(q - qnorm(c(.025, .5, .975)))), .06)
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
  out <- tulpa:::.nl_outer_solve(cells, solve, TRUE)
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

test_that("unattained importance precision reaches diagnostics and print", {
  qmc <- list(n = 4096L, error = list(max = .12), tol = .05, converged = FALSE)
  fit <- tulpa:::.finalize_fit(list(draws = matrix(seq_len(20L), ncol = 1L,
    dimnames = list(NULL, "beta")), outer_qmc = qmc), backend = "nested_laplace")
  for (object in list(fit, tulpa:::.finalize_fit(list(draws = fit$draws,
    joint_fit = fit), backend = "nested_laplace"))) {
    d <- diagnostics(object)
    expect_false(attr(d, "summary")$outer_converged)
    expect_equal(attr(d, "summary")$outer_error, .12)
    expect_equal(attr(d, "summary")$outer_tol, .05)
    expect_output(print(object), "target unattained")
    expect_output(print(d), "target unattained")
  }
})
