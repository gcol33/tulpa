# Approximation-reliability diagnostics for deterministic (i.i.d.-draw)
# nested-Laplace fits: laplace_diagnostics() and the mcmc_diagnostics() dispatch.
#
# The reliability headline is a PSIS Pareto-k-hat (Vehtari et al. 2024) of the
# importance ratio log p_target - log q_proposal, scoring whether the
# deterministic approximation q is a faithful stand-in for the exact posterior.
# These tests are recovery tests, not shape tests: a target the Gaussian
# proposal covers must read k-hat < 0.5 (the approximation works), and a
# heavy-tailed target it cannot correct must read k-hat >= 0.7 (the diagnostic
# detects the failure). The structural / dispatch tests sit on top of those.

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

.rel_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# Two-arm joint fixture: binomial occupancy + gaussian positive sharing one
# ICAR field, the positive arm copying it with coefficient alpha.
.rel_sim <- function(N = 260, n_s = 24, sigma = 0.6, alpha_true = 1.0, seed = 7) {
  set.seed(seed)
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  rw    <- cumsum(rnorm(n_s, 0, sigma / sqrt(n_s)))
  phi_s <- rw - mean(rw)
  x <- rnorm(N)
  Xocc <- cbind(1, x)
  eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]
  occur   <- rbinom(N, 1, plogis(eta_occ))
  is_pos  <- occur == 1L
  Xpos    <- Xocc[is_pos, , drop = FALSE]
  spi_pos <- spatial_idx[is_pos]
  eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + alpha_true * phi_s[spi_pos]
  y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)
  list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx),
       Xocc = Xocc, occur = occur,
       Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos))
}

# Arms with the copy coefficient pinned (single alpha value): the proposal axis
# for alpha is degenerate, exercising the zero-variance-axis path.
.rel_arms <- function(sim, alpha_grid = 1.0) {
  list(
    occ = list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
               X = sim$Xocc, spatial_idx = sim$spatial_idx,
               re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
               family = "binomial", phi = 1.0),
    pos = list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
               X = sim$Xpos, spatial_idx = sim$spi_pos,
               re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
               sigma_re = 1.0, family = "gaussian", phi = 0.5,
               field_coef = list(name = "alpha", grid = alpha_grid))
  )
}

.rel_prior <- function(sim, sigma_grid = c(0.4, 0.9)) {
  adj <- .rel_chain_adj(sim$n_s)
  list(type = "icar", n_spatial_units = adj$n_spatial_units,
       adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
       n_neighbors = adj$n_neighbors, sigma_grid = sigma_grid)
}

# A minimal i.i.d. fit shell: a draws matrix tagged "iid" with grid weights /
# pareto fields, enough for the accessor without fitting anything.
.rel_iid_shell <- function(draws, weights = NULL, pareto_k = NA_real_,
                           is_ess = NA_real_) {
  structure(list(
    draws = draws, draws_kind = "iid",
    joint_fit = list(weights = weights, pareto_k = pareto_k,
                     pareto_k_is_ess = is_ess,
                     pareto_k_scope = "outer (hyperparameter) Gaussian proposal")
  ), class = c("tulpa_fit"))
}

# --------------------------------------------------------------------------- #
# (1) Recovery: the reliability k-hat is small when the Laplace proposal is     #
#     near-exact, large when the target is heavier than the proposal.          #
# --------------------------------------------------------------------------- #

test_that("reliability k-hat is small when the proposal covers the target (good case)", {
  set.seed(101)
  # The reliability headline reads the outer PSIS k-hat of
  # log p_target(theta) - log q_proposal(theta). A Gaussian target covered by a
  # wider Gaussian proposal has bounded importance weights: every moment finite,
  # so the GPD tail-shape k-hat is well below 0.5 and the IS efficiency is high.
  sg <- exp(seq(-3, 3, length.out = 61))
  lw <- stats::dnorm(log(sg), 0, 1.2, log = TRUE)        # proposal wider
  w  <- exp(lw - max(lw)); w <- w / sum(w)
  res <- list(theta_grid = matrix(sg, ncol = 1,
                                  dimnames = list(NULL, "sigma")),
              weights = w, prior = list(type = "icar"))
  refit_cover <- function(tm) {
    u <- log(tm[, "sigma"]); -0.5 * (u / 0.8)^2 - u      # target N(0, 0.8^2)
  }
  kd <- .joint_pareto_k(res, refit_cover, n_samples = 3000L)

  expect_true(is.finite(kd$pareto_k))
  expect_lt(kd$pareto_k, 0.5)                            # "good": k-hat < 0.5
  expect_gt(kd$is_ess, 0.5 * 3000)                       # high IS efficiency
  expect_equal(.tulpa_khat_band(kd$pareto_k), "good")
})

test_that("reliability k-hat is large when the target is heavier than the proposal (bad case)", {
  set.seed(203)
  # The grid proposal is concentrated (sd 0.5 on the log scale) but the target
  # is a wide Gaussian (sd 3): the proposal cannot cover the target's spread, so
  # the importance ratio rises steeply with the whitened radius and its right
  # tail is heavy. The GPD tail-shape k-hat sits in the unreliable band -- the
  # diagnostic detects the approximation failure.
  sg <- exp(seq(-3, 3, length.out = 61))
  lw <- stats::dnorm(log(sg), 0, 0.5, log = TRUE)        # narrow proposal
  w  <- exp(lw - max(lw)); w <- w / sum(w)
  res <- list(theta_grid = matrix(sg, ncol = 1,
                                  dimnames = list(NULL, "sigma")),
              weights = w, prior = list(type = "icar"))
  refit_heavy <- function(tm) {
    u <- log(tm[, "sigma"]); -0.5 * (u / 3)^2 - u         # wide target N(0, 3^2)
  }
  kd <- .joint_pareto_k(res, refit_heavy, n_samples = 4000L)

  expect_true(is.finite(kd$pareto_k))
  expect_gte(kd$pareto_k, 0.7)                           # "unreliable": >= 0.7
  expect_equal(.tulpa_khat_band(kd$pareto_k), "unreliable")
})

# --------------------------------------------------------------------------- #
# (2) Degenerate (pinned) axis: a zero-variance hyperparameter no longer        #
#     declines the whole fit; the k-hat is computed on the varying axes.       #
# --------------------------------------------------------------------------- #

test_that("a pinned (zero-variance) axis yields the same finite k-hat as the varying-axis fit", {
  set.seed(101)
  sg <- exp(seq(-3, 3, length.out = 41))
  refit_cover <- function(tm) { u <- log(tm[, "sigma"]); -0.5 * (u / 0.8)^2 - u }

  # Single varying axis.
  l1 <- stats::dnorm(log(sg), 0, 1.2, log = TRUE)
  res1 <- list(theta_grid = matrix(sg, ncol = 1, dimnames = list(NULL, "sigma")),
               weights = { e <- exp(l1 - max(l1)); e / sum(e) },
               prior = list(type = "icar"))
  set.seed(303); k1 <- .joint_pareto_k(res1, refit_cover, n_samples = 3000L)

  # Same axis plus an alpha axis PINNED at 0 (zero weighted variance).
  grid2 <- expand.grid(sigma = sg, alpha = 0)
  l2 <- stats::dnorm(log(grid2$sigma), 0, 1.2, log = TRUE)
  res2 <- list(theta_grid = as.matrix(grid2[, c("sigma", "alpha")]),
               weights = { e <- exp(l2 - max(l2)); e / sum(e) },
               prior = list(type = "icar"))
  colnames(res2$theta_grid) <- c("sigma", "alpha")
  set.seed(303); k2 <- .joint_pareto_k(res2, refit_cover, n_samples = 3000L)

  expect_true(is.finite(k2$pareto_k))                   # no longer declines
  expect_equal(k2$pareto_k, k1$pareto_k, tolerance = 1e-8)
})

# --------------------------------------------------------------------------- #
# (3) Shape / plumbing on a real small nested-Laplace fit.                     #
# --------------------------------------------------------------------------- #

test_that("laplace_diagnostics returns one finite row per parameter on a small joint fit", {
  skip_if_not_slow()
  sim   <- .rel_sim(seed = 41)
  fit   <- tulpa_nested_laplace_joint(
    responses = .rel_arms(sim, alpha_grid = 1.0), prior = .rel_prior(sim),
    control = list(diagnose_k = TRUE, k_samples = 150L, store_Q = TRUE))

  fit$draws      <- tulpa_posterior_draws(fit, idx = 1:4, n = 800)
  fit$draws_kind <- "iid"

  rel <- laplace_diagnostics(fit)
  expect_s3_class(rel, "laplace_diagnostics")
  expect_equal(nrow(rel), 4L)
  expect_setequal(names(rel),
                  c("parameter", "mean", "sd", "ess_bulk", "ess_tail", "rhat"))
  expect_true(all(is.finite(rel$mean)))
  expect_true(all(is.finite(rel$sd)))
  expect_true(all(is.finite(rel$rhat)))

  # i.i.d.-draw Monte-Carlo diagnostics: rhat ~ 1, ESS ~ n_draws.
  expect_true(all(abs(rel$rhat - 1) < 0.1))
  expect_true(all(rel$ess_bulk > 0.5 * 800))

  # The headline k-hat is finite (the pinned-alpha fit no longer declines) and
  # carries a band; grid quadrature reliability is attached.
  expect_true(is.finite(attr(rel, "pareto_k")))
  expect_true(attr(rel, "pareto_k_band") %in% c("good", "ok", "unreliable"))
  expect_true(is.finite(attr(rel, "ess_grid")))
  expect_gte(attr(rel, "max_weight"), 0)
})

# --------------------------------------------------------------------------- #
# (4) diagnostics() routing: chain behaviour preserved, i.i.d. routed here.    #
# --------------------------------------------------------------------------- #

test_that("diagnostics routes an i.i.d. fit to the reliability table", {
  set.seed(5)
  draws <- matrix(rnorm(1000 * 3), 1000, 3,
                  dimnames = list(NULL, c("a", "b", "c")))
  w   <- c(0.5, 0.3, 0.2)
  fit <- .rel_iid_shell(draws, weights = w, pareto_k = 0.42, is_ess = 850)

  md <- diagnostics(fit)
  expect_s3_class(md, "laplace_diagnostics")
  expect_equal(nrow(md), 3L)
  expect_equal(attr(md, "pareto_k"), 0.42)
  expect_equal(attr(md, "pareto_k_band"), "good")
  expect_equal(attr(md, "ess_grid"), 1 / sum(w^2))
  expect_true(all(is.finite(md$rhat)))
})

test_that("diagnostics keeps chain behaviour for an MCMC fit", {
  set.seed(6)
  draws <- matrix(rnorm(400 * 2), 400, 2, dimnames = list(NULL, c("a", "b")))
  fit <- structure(list(draws = draws, draws_kind = "chain"),
                   class = "tulpa_fit")
  md <- diagnostics(fit)
  expect_false(inherits(md, "laplace_diagnostics"))
  expect_true("rhat" %in% names(md))
  expect_equal(nrow(md), 2L)
})

test_that("diagnostics honours pars and reads a cached k-hat on an i.i.d. fit", {
  set.seed(8)
  draws <- matrix(rnorm(600 * 4), 600, 4,
                  dimnames = list(NULL, c("a", "b", "c", "d")))
  fit <- .rel_iid_shell(draws, weights = rep(0.25, 4),
                        pareto_k = 0.8, is_ess = 120)

  rel <- diagnostics(fit, pars = c("a", "c"))
  expect_equal(rel$parameter, c("a", "c"))
  expect_equal(attr(rel, "pareto_k"), 0.8)
  expect_equal(attr(rel, "pareto_k_band"), "unreliable")
  expect_equal(attr(rel, "ess_grid"), 4)                # uniform grid: ESS = K
})

test_that("the k-hat interpretation bands match the Vehtari et al. (2024) thresholds", {
  expect_equal(.tulpa_khat_band(0.3),  "good")
  expect_equal(.tulpa_khat_band(0.49), "good")
  expect_equal(.tulpa_khat_band(0.5),  "ok")
  expect_equal(.tulpa_khat_band(0.69), "ok")
  expect_equal(.tulpa_khat_band(0.7),  "unreliable")
  expect_equal(.tulpa_khat_band(1.2),  "unreliable")
  expect_true(is.na(.tulpa_khat_band(NA_real_)))
})
