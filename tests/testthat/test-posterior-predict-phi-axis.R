# An INTEGRATED dispersion is part of the predictive distribution
# (gcol33/tulpa#825).
#
# `tulpa_nested_laplace_joint(phi_grid =)` lifts an arm's dispersion onto the
# outer grid and integrates it. `posterior_predict()` and `bayes_R2()` then
# sampled every replicate at one scalar, so an axis the fit had paid to
# integrate contributed no predictive uncertainty at all. Three claims:
#
#   1. THE CONVENTION -- a `phi_<arm>` grid column is in the R-level
#      convention (`phi` is the residual VARIANCE for gaussian), measured
#      against a fit whose residual SD is known, so handing a cell's value to
#      the sampler crosses no seam.
#   2. THE RESOLUTION -- `.tulpa_phi_draws()` gives each replicate the
#      dispersion of the cell it was drawn in, continuized within that cell
#      (gcol33/tulpa#823) rather than read as the grid node it sits on, and
#      falls back to the fit's scalar wherever there is no axis.
#   3. THE EFFECT -- the replicates' residual spread is the axis's, not one
#      cell's; and a fit with no axis is unchanged.
#
# The joint driver stores the per-cell linear predictor the mixture draws from
# since gcol33/tulpa#850, so claim 3 runs on a real fit rather than on a
# hand-built grid standing in for one.

.ppa_sim <- function(seed = 11L, n_s = 25L, N = 300L, sd_true = 0.45) {
  set.seed(seed)
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  sidx <- sample.int(n_s, N, replace = TRUE)
  w <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.5)))) * 0.6
  X <- cbind(1, rnorm(N))
  y <- as.numeric(X %*% c(0.4, -0.6)) + w[sidx] + rnorm(N, 0, sd_true)
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn),
       sidx = sidx, X = X, y = y, N = N, n_s = n_s, sd_true = sd_true)
}

.ppa_fit <- function(sim, phi_axis, phi0 = 0.2, control = list()) {
  arm <- list(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
              spatial_idx = as.integer(sim$sidx), re_idx = rep(0, sim$N),
              n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = phi0)
  blk <- list(type = "icar", n_spatial_units = sim$n_s,
              adj_row_ptr = sim$adj_row_ptr, adj_col_idx = sim$adj_col_idx,
              n_neighbors = sim$n_neighbors, sigma_grid = c(0.4, 0.6, 0.9))
  args <- list(responses = list(obs = arm), prior = blk,
               control = utils::modifyList(list(diagnose_k = FALSE), control))
  if (!is.null(phi_axis)) args$phi_grid <- list(obs = phi_axis)
  do.call(tulpa_nested_laplace_joint, args)
}


test_that("a phi_<arm> grid column is the residual VARIANCE", {
  skip_on_cran()
  sim <- .ppa_sim()
  fit <- .ppa_fit(sim, c(0.10, 0.16, 0.2025, 0.25, 0.35))

  proc <- tulpa:::.tulpa_response_process(fit, "test")
  expect_equal(proc$arm, "obs")
  expect_equal(proc$family, "gaussian")

  col <- paste0("phi_", proc$arm)
  expect_true(col %in% colnames(fit$theta_grid))
  # The axis posterior sits at the true residual VARIANCE. Under the kernel's
  # SD convention the mass would sit at sd_true instead, a factor of 2.2 away
  # from where it is, so this separates the two conventions rather than merely
  # being consistent with one.
  phi_hat <- sum(as.numeric(fit$weights) * fit$theta_grid[, col])
  expect_equal(phi_hat, sim$sd_true^2, tolerance = 0.15)
  expect_gt(abs(phi_hat - sim$sd_true), 10 * abs(phi_hat - sim$sd_true^2))
})


test_that(".tulpa_phi_draws resolves the cell's dispersion, or the scalar", {
  skip_on_cran()
  sim <- .ppa_sim()
  fit <- .ppa_fit(sim, c(0.10, 0.16, 0.2025, 0.25, 0.35), phi0 = 0.2)
  proc <- tulpa:::.tulpa_response_process(fit, "test")

  set.seed(3)
  S <- 800L
  cells <- sample.int(nrow(fit$theta_grid), S, TRUE, prob = fit$weights)
  p <- tulpa:::.tulpa_phi_draws(fit, proc, cells, S)

  expect_length(p, S)
  expect_true(all(p > 0))
  # Continuized, not the grid node: the node read has as many distinct values
  # as the axis has levels, the draw has one per replicate.
  nodes <- unique(fit$theta_grid[, paste0("phi_", proc$arm)])
  expect_gt(length(unique(p)), 10 * length(nodes))
  expect_equal(mean(p), sim$sd_true^2, tolerance = 0.2)

  # No cells (a fit whose eta did not come from the grid mixture) -> the
  # scalar, S times.
  expect_equal(tulpa:::.tulpa_phi_draws(fit, proc, NULL, 5L), rep(0.2, 5L))

  # A one-value column is a PIN -- part of the model, not an axis -- and takes
  # the same scalar path.
  pinned <- .ppa_fit(sim, NULL, phi0 = 0.2)
  pp <- tulpa:::.tulpa_response_process(pinned, "test")
  expect_equal(tulpa:::.tulpa_phi_draws(pinned, pp, c(1L, 1L, 2L), 3L),
               rep(0.2, 3L))
})


test_that("the dispersion axis reaches the replicates and the log-likelihood", {
  skip_on_cran()
  # A fit carrying both halves: the per-cell linear predictor a mixture draws
  # from, and an integrated dispersion axis. The joint driver stores the
  # predictor since gcol33/tulpa#850, so this runs on a REAL fit rather than on
  # a hand-built grid standing in for one.
  sim  <- .ppa_sim()
  axis <- c(0.10, 0.16, 0.2025, 0.25, 0.35)
  fit  <- .ppa_fit(sim, axis, phi0 = 0.2)

  expect_true(is.matrix(fit$fitted_eta))
  expect_equal(dim(fit$fitted_eta), c(nrow(fit$theta_grid), sim$N))
  expect_equal(dim(fit$fitted_eta_var), dim(fit$fitted_eta))
  expect_equal(length(fit$weights), nrow(fit$fitted_eta))
  expect_identical(tulpa:::.tulpa_linpred_source(fit), "grid_mixture")

  # The same fit with its dispersion PINNED: one grid column and the arm's own
  # scalar, and NOTHING else -- the same per-cell predictor, the same
  # within-cell variance, the same weights, so the same cells and the same eta
  # at one seed. The replicates' residual spread then follows the dispersion
  # alone.
  pin    <- 0.9
  pinned <- fit
  pinned$theta_grid[, "phi_obs"] <- pin
  pinned$responses$obs$phi       <- pin

  S <- 3000L
  set.seed(8); eta   <- tulpa:::.tulpa_eta_draws(fit, ndraws = S)
  set.seed(8); y_ax  <- posterior_predict(fit,    ndraws = S)
  set.seed(8); y_pin <- posterior_predict(pinned, ndraws = S)

  # Each replicate carries its own cell's residual variance, so the pooled
  # residual variance is the axis's posterior mean and not the pin's value.
  phi_bar <- sum(as.numeric(fit$weights) * fit$theta_grid[, "phi_obs"])
  expect_equal(var(as.numeric(y_ax)  - as.numeric(eta)), phi_bar,
               tolerance = 0.1)
  expect_equal(var(as.numeric(y_pin) - as.numeric(eta)), pin, tolerance = 0.1)

  # The same resolution reaches the pointwise log-likelihood WAIC / LOO read,
  # so the density scores each draw at the dispersion it was drawn under.
  ll_axis   <- tulpa:::.tulpa_eta_loglik(fit,    eta)
  ll_pinned <- tulpa:::.tulpa_eta_loglik(pinned, eta)
  expect_equal(dim(ll_axis), dim(eta))
  expect_false(isTRUE(all.equal(ll_axis, ll_pinned)))
  expect_true(all(is.finite(ll_axis)))

  # And the criteria / diagnostic doors read the arm's response and design
  # rather than the per-arm list a joint fit carries at `$y`.
  expect_equal(dim(tulpa:::.tulpa_pointwise_loglik(fit, ndraws = 40L)),
               c(40L, sim$N))
  expect_true(all(is.finite(bayes_R2(fit)$estimate)))
  expect_length(as.numeric(fitted(fit)), sim$N)
  expect_length(as.numeric(residuals(fit, type = "response")), sim$N)
})


test_that("control$fitted_var declines the within-cell spread, not the predictor", {
  skip_on_cran()
  # The per-row predictive variance is a real per-cell cost, so it is
  # declinable; the per-cell predictor is not the same quantity and stays
  # (gcol33/tulpa#850). A fit without it draws the across-cell spread only.
  sim <- .ppa_sim()
  fit <- .ppa_fit(sim, c(0.16, 0.2025, 0.25), control = list(fitted_var = FALSE))

  expect_true(is.matrix(fit$fitted_eta))
  expect_null(fit$fitted_eta_var)
  expect_identical(tulpa:::.tulpa_linpred_source(fit), "grid_mixture")
  expect_equal(dim(posterior_predict(fit, ndraws = 50L)), c(50L, sim$N))
})


test_that("a fit with no dispersion axis is scored at its own scalar", {
  skip_on_cran()
  # The whole no-axis population: `.tulpa_phi_draws()` returns S copies of the
  # fit's own `phi`, which is the scalar expression it replaced, so nothing
  # about a fit without an axis moves.
  set.seed(2)
  n <- 200L
  d <- data.frame(x = rnorm(n))
  d$y <- rnorm(n, 0.5 + 0.8 * d$x, 0.7)
  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
               phi = 0.49)

  proc <- tulpa:::.tulpa_response_process(fit, "test")
  expect_null(proc$arm)
  expect_equal(proc$family, "gaussian")
  expect_equal(unique(tulpa:::.tulpa_phi_draws(fit, proc, NULL, 12L)), 0.49)

  set.seed(7); a <- posterior_predict(fit, ndraws = 60L)
  set.seed(7); b <- posterior_predict(fit, ndraws = 60L)
  expect_identical(a, b)
  expect_true(all(is.finite(bayes_R2(fit)$estimate)))
  expect_true(all(is.finite(tulpa:::.tulpa_pointwise_loglik(fit))))
})


test_that("a multi-arm fit is refused by name, not by a missing family", {
  fake <- structure(
    list(backend = "nested_laplace",
         responses = list(a = list(family = "gaussian", phi = 1),
                          b = list(family = "poisson", phi = 1))),
    class = c("tulpa_nested_laplace_joint", "tulpa_fit"))
  expect_error(posterior_predict(fake), "2 arms")
  expect_error(posterior_predict(fake), "its own family and dispersion")

  bare <- structure(list(backend = "laplace"), class = "tulpa_fit")
  expect_error(posterior_predict(bare), "a single built-in family")
})
