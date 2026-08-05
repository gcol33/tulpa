# Inner-Laplace skewness diagnostic (gamma_3, gcol33/tulpa#272): whether the
# Gaussian (inner) Laplace approximation to the latent-field conditional
# posterior pi(x_i | theta, y) is itself a good fit -- the layer the outer
# Pareto-k-hat diagnostic (test-laplace_diagnostics.R) does NOT cover.
#
# These are recovery tests, not shape tests: a case where the Laplace
# approximation is analytically EXACT (gaussian family: the log-likelihood is
# exactly quadratic in eta, so the third derivative is identically zero) must
# read gamma_3 == 0; a case with a KNOWN, independently-computed exact
# posterior skewness (a numerically-integrated intercept-only rare-event
# binomial posterior) must have gamma_3 track that exact value, not just be
# "some large number". A likelihood the diagnostic cannot score (a coupled
# multi-process spec, e.g. zero-inflation) must read NaN, never a silently
# wrong 0 ("perfectly Gaussian").

# --------------------------------------------------------------------------- #
# (1) Structural: band / summary / combined-verdict helpers                   #
# --------------------------------------------------------------------------- #

test_that(".tulpa_gamma3_band bands |gamma_3| the same way for +/- signs", {
  expect_equal(.tulpa_gamma3_band(0), "good")
  expect_equal(.tulpa_gamma3_band(0.3), "good")
  expect_equal(.tulpa_gamma3_band(-0.3), "good")
  expect_equal(.tulpa_gamma3_band(0.7), "ok")
  expect_equal(.tulpa_gamma3_band(-0.7), "ok")
  expect_equal(.tulpa_gamma3_band(1.5), "unreliable")
  expect_equal(.tulpa_gamma3_band(-1.5), "unreliable")
  expect_true(is.na(.tulpa_gamma3_band(NA_real_)))
  expect_true(is.na(.tulpa_gamma3_band(NaN)))
})

test_that(".tulpa_inner_skew_summary aggregates finite values and counts NaN separately", {
  s <- .tulpa_inner_skew_summary(c(0.1, -0.2, NaN, 1.4), n_dropped = 3L)
  expect_equal(s$n_probed, 4L)
  expect_equal(s$n_scored, 3L)
  expect_equal(s$max_abs_gamma3, 1.4)
  expect_equal(s$band, "unreliable")
  expect_equal(s$n_dropped, 3L)
  expect_equal(s$share_unreliable, 1 / 3)

  # All-NaN: nothing scored, band is NA (not "good") -- "not computable" must
  # never collapse to a reliability verdict.
  s2 <- .tulpa_inner_skew_summary(c(NaN, NaN))
  expect_equal(s2$n_scored, 0L)
  expect_true(is.na(s2$band))

  expect_null(.tulpa_inner_skew_summary(NULL))
  expect_null(.tulpa_inner_skew_summary(numeric(0)))
})

test_that(".tulpa_combined_reliability reports which layer degrades, not a single conflated flag", {
  expect_match(.tulpa_combined_reliability("good", "good"), "^reliable")
  # High outer k-hat, healthy inner layer -- the #272 motivating case
  # (42/78 occu_cover species read as "broken" on outer k-hat alone when the
  # point estimates, governed by the inner layer, were fine).
  v1 <- .tulpa_combined_reliability("unreliable", "good")
  expect_match(v1, "^scoped: outer")
  v2 <- .tulpa_combined_reliability("good", "unreliable")
  expect_match(v2, "^scoped: inner")
  expect_match(.tulpa_combined_reliability("unreliable", "unreliable"), "^unreliable")
  expect_match(.tulpa_combined_reliability(NA_character_, NA_character_),
              "^not computed")
})

# --------------------------------------------------------------------------- #
# (2) Recovery: quiet where the Laplace approximation is analytically exact   #
# --------------------------------------------------------------------------- #

test_that("gamma_3 is exactly zero for a gaussian intercept (the log-lik is exactly quadratic)", {
  skip_on_cran()
  # d^3/deta^3 of a gaussian log-density in eta (identity link) is identically
  # zero, so the inner Laplace is EXACT and gamma_3 must be exactly 0, not
  # just "small" -- a real gap here (not noise) would mean the third-derivative
  # ladder for gaussian is wrong.
  set.seed(11)
  n <- 300L
  x <- rnorm(n)
  y <- 1 + 0.5 * x + rnorm(n, 0, 1)
  fit <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, n), X = cbind(1, x),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "gaussian", compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 stays near zero for a large-count poisson fit (CLT regime)", {
  skip_on_cran()
  set.seed(12)
  n <- 600L
  x <- rnorm(n)
  y <- rpois(n, exp(3 + 0.1 * x))   # mu ~ 20: large counts, CLT applies
  fit <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, n), X = cbind(1, x),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "poisson", compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_true(all(is.finite(fit$inner_skew)))
  expect_lt(max(abs(fit$inner_skew)), 0.05)
  expect_equal(.tulpa_gamma3_band(max(abs(fit$inner_skew))), "good")
})

# --------------------------------------------------------------------------- #
# (3) Recovery: fires where the Laplace approximation is known-poor, and      #
#     TRACKS the exact posterior skewness (not just "some large number")     #
# --------------------------------------------------------------------------- #

# Exact posterior skewness of an intercept-only binomial-logit model via
# direct numerical quadrature (no Laplace approximation at all) -- the
# independent ground truth gamma_3 is checked against. log-lik is
# S*eta - N*log1p(exp(eta)) for S successes out of N (all sharing one eta);
# the prior is the same weakly-informative N(0, sigma_beta^2) the C++ kernel
# uses (sigma_beta = 100, negligible curvature next to the likelihood here).
.exact_intercept_skew <- function(N, S, sigma_beta = 100) {
  log_post <- function(eta) S * eta - N * log1p(exp(eta)) - 0.5 * (eta / sigma_beta)^2
  mode <- stats::optimize(log_post, interval = c(-30, 30), maximum = TRUE)$maximum
  grid <- seq(mode - 15, mode + 15, length.out = 200000)
  dz <- grid[2] - grid[1]
  lp <- log_post(grid); lp <- lp - max(lp)
  w <- exp(lp); w <- w / (sum(w) * dz)
  mu  <- sum(grid * w) * dz
  sd  <- sqrt(sum((grid - mu)^2 * w) * dz)
  mu3 <- sum((grid - mu)^3 * w) * dz
  mu3 / sd^3
}

test_that("gamma_3 tracks the exact posterior skewness of a rare-event binomial intercept", {
  skip_on_cran()
  # Leading-order Edgeworth estimates systematically UNDER-estimate genuinely
  # large skewness (the series itself degrades as skewness grows) -- expect
  # gamma_3 / exact_skew in (0.8, 1.0], not equality, and confirm the
  # direction (fires large-magnitude) rather than an exact numeric match.
  cases <- list(c(N = 500, S = 230), c(N = 100, S = 3), c(N = 20, S = 2),
               c(N = 15, S = 1))
  for (cs in cases) {
    N <- cs[["N"]]; S <- cs[["S"]]
    y <- c(rep(1, S), rep(0, N - S))
    fit <- tulpa:::cpp_laplace_fit(
      y = as.numeric(y), n = rep(1L, N), X = matrix(1, N, 1),
      re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
      family = "binomial", compute_skew = TRUE, skew_idx = 1L
    )
    exact <- .exact_intercept_skew(N, S)
    ratio <- fit$inner_skew[1] / exact
    expect_gt(ratio, 0.8)
    expect_lte(ratio, 1.0)
  }
  # The most extreme case (N=15, one success) is a textbook Wald-CI failure:
  # exact skewness is comfortably past 1. gamma_3, the LEADING-ORDER estimate,
  # systematically under-shoots genuinely large skewness (confirmed by the
  # ratio check above), so it does not itself cross the "unreliable" cutoff
  # here -- but it must land off "good" (into "ok"), i.e. NOT read as a
  # healthy fit.
  extreme_exact <- .exact_intercept_skew(15, 1)
  expect_gt(abs(extreme_exact), 1.0)
  y <- c(1, rep(0, 14))
  fit <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, 15), X = matrix(1, 15, 1),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "binomial", compute_skew = TRUE, skew_idx = 1L
  )
  expect_false(identical(.tulpa_gamma3_band(fit$inner_skew[1]), "good"))

  # And the well-balanced N=500 case (roughly symmetric binomial) must read
  # "good" -- the diagnostic does not cry wolf on a healthy fit.
  y_bal <- c(rep(1, 230), rep(0, 270))
  fit_bal <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y_bal), n = rep(1L, 500), X = matrix(1, 500, 1),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "binomial", compute_skew = TRUE, skew_idx = 1L
  )
  expect_equal(.tulpa_gamma3_band(fit_bal$inner_skew[1]), "good")
})

# --------------------------------------------------------------------------- #
# (4) Decline behaviour: a coupled multi-process spec reports NaN, never a    #
#     silently-wrong 0 ("perfectly Gaussian")                                #
# --------------------------------------------------------------------------- #

test_that("gamma_3 declines to NaN (not 0) for a zero-inflated (multi-process) fit", {
  skip_on_cran()
  # build_spec_curvature3_fn declines whenever spec.n_processes != 1; ZI adds
  # a second process (the zero-inflation logit). Before the fix in
  # inner_laplace_skew.h this silently returned 0.0 ("no skew") instead of NaN
  # when the oracle was entirely absent -- this test pins that fix.
  set.seed(13)
  n <- 200L
  x <- rnorm(n); z <- rnorm(n)
  X <- cbind(1, x); X_zi <- cbind(1, z)
  mu <- exp(0.5 + 0.3 * x)
  zi_p <- plogis(-0.5 + 0.4 * z)
  y <- ifelse(rbinom(n, 1, zi_p) == 1, 0, rpois(n, mu))

  fit_zi <- tulpa:::cpp_laplace_fit_multi_re(
    y = as.numeric(y), n = rep(1L, n), X = X,
    re_idx_list = list(), re_ngroups = integer(0), re_sigma_list = list(),
    family = "poisson", X_zi = X_zi,
    compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_true(all(is.nan(fit_zi$inner_skew)))

  # The identical count-process model WITHOUT zero-inflation (n_processes == 1)
  # scores fine -- confirms the NaN above is the coupling gate, not a
  # dispatch failure.
  y_nozi <- rpois(n, mu)
  fit_nozi <- tulpa:::cpp_laplace_fit_multi_re(
    y = as.numeric(y_nozi), n = rep(1L, n), X = X,
    re_idx_list = list(), re_ngroups = integer(0), re_sigma_list = list(),
    family = "poisson",
    compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_true(all(is.finite(fit_nozi$inner_skew)))
})

# --------------------------------------------------------------------------- #
# (5) Front-door wiring: tulpa_nested_laplace() populates inner_skew for both #
#     single- and multi-block dispatch, and diagnostics() reports the         #
#     combined verdict                                                       #
# --------------------------------------------------------------------------- #

.isk_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s), function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

test_that("tulpa_nested_laplace() populates inner_skew by default (single- and multi-block)", {
  skip_on_cran()
  set.seed(14)
  n_s <- 12L; adj <- .isk_chain_adj(n_s)
  idx <- rep(seq_len(n_s), each = 10L); n <- length(idx)
  x <- rnorm(n)
  field <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
  prior <- list(type = "icar", n_spatial_units = n_s, spatial_idx = idx,
                adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                n_neighbors = adj$n_neighbors, tau_grid = c(0.5, 1, 2, 4, 8))

  fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                              family = "binomial")
  # Default probe scope is the p fixed-effects coefficients.
  expect_length(fit$inner_skew, 2L)
  expect_equal(fit$inner_skew_idx, c(1L, 2L))
  expect_true(all(is.finite(fit$inner_skew)))

  fit_multi <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x),
                                    prior = list(prior), family = "binomial")
  expect_length(fit_multi$inner_skew, 2L)
  # Single-block-wrapped-as-multi must agree with the direct single-block
  # dispatch (same model, same MAP cell, same kernel underneath).
  expect_equal(fit_multi$inner_skew, fit$inner_skew, tolerance = 1e-6)

  # control$diagnose_skew = FALSE turns the diagnostic off entirely.
  fit_off <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                                  family = "binomial",
                                  control = list(diagnose_skew = FALSE))
  expect_null(fit_off$inner_skew)

  # An explicit control$skew_idx overrides the default fixed-effects scope.
  fit_custom <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                                     family = "binomial",
                                     control = list(skew_idx = 1L))
  expect_length(fit_custom$inner_skew, 1L)
  expect_equal(fit_custom$inner_skew_idx, 1L)
})

test_that("diagnostics() reports the combined whole-fit verdict when inner_skew is present", {
  skip_on_cran()
  # A minimal i.i.d. fit shell (mirrors .rel_iid_shell in
  # test-laplace_diagnostics.R): enough structure for .tulpa_approx_diag_table
  # without fitting a model, isolating the print/attribute wiring from the
  # C++ kernel already covered above.
  draws <- matrix(rnorm(400), ncol = 2, dimnames = list(NULL, c("b0", "b1")))
  fit <- structure(list(
    draws = draws, draws_kind = "iid",
    joint_fit = list(weights = rep(1, 5) / 5, pareto_k = 0.3,
                     pareto_k_is_ess = 400, pareto_k_scope = "outer",
                     inner_skew = c(0.1, 1.4), inner_skew_idx = c(1L, 2L),
                     inner_skew_dropped = 0L)
  ), class = "tulpa_fit")

  tab <- .tulpa_approx_diag_table(fit)
  expect_equal(attr(tab, "inner_skew_band"), "unreliable")
  expect_equal(attr(tab, "pareto_k_band"), "good")
  expect_match(attr(tab, "reliability"), "^scoped: inner")
  expect_output(print(tab), "WHOLE-FIT")
  expect_output(print(tab), "whole-fit verdict")
})

# --------------------------------------------------------------------------- #
# (6) Joint front door: tulpa_nested_laplace_joint() single-block backends    #
#     (icar/bym2/car_proper), separable AND genuinely coupled arms           #
# --------------------------------------------------------------------------- #

.isk_biv_data <- function(seed = 11L, n_s = 10L) {
  set.seed(seed)
  adj <- .isk_chain_adj(n_s)
  spatial_idx <- seq_len(n_s)
  z_true <- rnorm(n_s, 0, 0.5)
  y0 <- 0.2 + z_true + rnorm(n_s, 0, 0.4)
  y1 <- -0.1 + 0.8 * z_true + rnorm(n_s, 0, 0.4)
  list(n_s = n_s, adj = adj, spatial_idx = spatial_idx, y0 = y0, y1 = y1)
}

.isk_biv_arm <- function(d, y_vec, coupled) {
  arm <- list(y = y_vec, n_trials = rep(1L, d$n_s),
             X = matrix(1, nrow = d$n_s, ncol = 1),
             spatial_idx = d$spatial_idx, family = "gaussian", phi = 1)
  if (coupled) {
    arm$coupled <- TRUE
    arm$cell_obs_map <- seq_len(d$n_s)
  }
  arm
}

.isk_biv_prior <- function(d) {
  list(type = "icar", n_spatial_units = d$adj$n_spatial_units,
       adj_row_ptr = d$adj$adj_row_ptr, adj_col_idx = d$adj$adj_col_idx,
       n_neighbors = d$adj$n_neighbors, sigma_grid = c(0.4, 0.8, 1.5))
}

test_that("tulpa_nested_laplace_joint() scores real gamma_3 for a separable 2-arm fit", {
  skip_on_cran()
  set.seed(21)
  N <- 260; n_s <- 24
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  phi_s <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  x <- rnorm(N)
  Xocc <- cbind(1, x)
  occur <- rbinom(N, 1, plogis(as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]))
  is_pos <- occur == 1L
  Xpos <- Xocc[is_pos, , drop = FALSE]
  spi_pos <- spatial_idx[is_pos]
  y_pos <- rnorm(sum(is_pos),
                 as.numeric(Xpos %*% c(0.2, -0.4)) + phi_s[spi_pos], 0.5)
  adj <- .isk_chain_adj(n_s)
  prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
               adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
               n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
  responses <- list(
    occ = list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
              spatial_idx = spatial_idx, re_idx = rep(0, N), n_re_groups = 0L,
              sigma_re = 1.0, family = "binomial", phi = 1.0),
    pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
              spatial_idx = spi_pos, re_idx = rep(0, length(y_pos)),
              n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.5,
              field_coef = list(name = "alpha", grid = c(0.5, 1.0, 1.5)))
  )

  fit <- tulpa_nested_laplace_joint(responses, prior,
                                    control = list(max_iter = 100L, tol = 1e-6,
                                                   diagnose_k = FALSE))
  # Default probe scope is every arm's fixed-effects coefficients (2 + 2).
  expect_length(fit$inner_skew, sum(fit$arm_layout$p))
  expect_true(all(is.finite(fit$inner_skew)))
  expect_equal(fit$inner_skew_idx, seq_len(sum(fit$arm_layout$p)))

  fit_off <- tulpa_nested_laplace_joint(responses, prior,
                                        control = list(max_iter = 100L, tol = 1e-6,
                                                       diagnose_k = FALSE,
                                                       diagnose_skew = FALSE))
  expect_null(fit_off$inner_skew)
})

test_that("tulpa_nested_laplace_joint() declines to NaN for a genuinely coupled fit", {
  skip_on_cran()
  # A coupled arm's per-obs oracle would score the WRONG (unused) likelihood
  # -- build_joint_curvature3_fns excludes it, so every probed index must
  # come back NaN, never a silently-wrong 0. Uses the test-only bivariate
  # gaussian CellCouplingSpec (registered in src/, mirrors
  # test-cell-coupling-cross-hess.R): both arms coupled == the occu_cover
  # shape (every arm coupled through the cell-coupling spec).
  cpp_register_test_bivariate_gaussian_coupling(lam00 = 2.0, lam11 = 1.5,
                                                lam01 = 0.7)
  skip_if_not(cpp_cell_coupling_registry_has("test_bivariate_gaussian"),
             "test_bivariate_gaussian coupling spec not registered")
  d <- .isk_biv_data()
  res <- tulpa_nested_laplace_joint(
    responses = list(a = .isk_biv_arm(d, d$y0, coupled = TRUE),
                     b = .isk_biv_arm(d, d$y1, coupled = TRUE)),
    prior     = .isk_biv_prior(d),
    cell_coupling = "test_bivariate_gaussian",
    control   = list(max_iter = 80L, tol = 1e-11, diagnose_k = FALSE)
  )
  expect_true(all(is.nan(res$inner_skew)))
  expect_length(res$inner_skew, sum(res$arm_layout$p))
})

# --------------------------------------------------------------------------- #
# (7) Joint front door: tulpa_nested_laplace_joint() MULTI-block path         #
#     (gcol33/tulpa#273 -- .nlj_multi_inner_skew_at_theta())                  #
# --------------------------------------------------------------------------- #

test_that("tulpa_nested_laplace_joint() wires gamma_3 through the multi-block dispatch", {
  skip_on_cran()
  # Two SEPARATE icar blocks, each contributing directly (no copy scaling) to
  # BOTH arms -- the simplest fit that routes through .joint_dispatch_multi()
  # rather than the single-block path the two tests above exercise.
  set.seed(37)
  n_s <- 8L
  adjA <- .isk_chain_adj(n_s); adjB <- .isk_chain_adj(n_s)
  N <- 80L
  iA <- sample.int(n_s, N, replace = TRUE)
  iB <- sample.int(n_s, N, replace = TRUE)
  pA <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.5))))
  pB <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  x <- rnorm(N); Xocc <- cbind(1, x)
  occ <- rbinom(N, 1, plogis(as.numeric(Xocc %*% c(-0.2, 0.4)) + pA[iA] + pB[iB]))
  is_pos <- occ == 1L
  Xpos <- Xocc[is_pos, , drop = FALSE]
  iAp <- iA[is_pos]; iBp <- iB[is_pos]
  y_pos <- rnorm(sum(is_pos),
                 as.numeric(Xpos %*% c(0.1, -0.3)) + pA[iAp] + pB[iBp], 0.5)

  responses <- list(
    occ = list(y = as.numeric(occ), n_trials = rep(1L, N), X = Xocc,
              spatial_idx = as.integer(iA), re_idx = rep(0, N),
              n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0),
    pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
              spatial_idx = as.integer(iAp), re_idx = rep(0, length(y_pos)),
              n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.5))
  prior <- list(
    list(type = "icar", n_spatial_units = n_s,
         adj_row_ptr = adjA$adj_row_ptr, adj_col_idx = adjA$adj_col_idx,
         n_neighbors = adjA$n_neighbors, sigma_grid = c(0.4, 0.9),
         spatial_idx = list(as.integer(iA), as.integer(iAp))),
    list(type = "icar", n_spatial_units = n_s,
         adj_row_ptr = adjB$adj_row_ptr, adj_col_idx = adjB$adj_col_idx,
         n_neighbors = adjB$n_neighbors, sigma_grid = c(0.3, 0.8),
         spatial_idx = list(as.integer(iB), as.integer(iBp))))

  fit <- tulpa_nested_laplace_joint(
      responses = responses, prior = prior,
      control = list(max_iter = 100L, tol = 1e-6, diagnose_k = FALSE))
  expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
  # Default probe scope is every arm's fixed-effects coefficients (2 + 2).
  expect_length(fit$inner_skew, sum(fit$arm_layout$p))
  expect_true(all(is.finite(fit$inner_skew)))
  expect_equal(fit$inner_skew_idx, seq_len(sum(fit$arm_layout$p)))

  fit_off <- tulpa_nested_laplace_joint(
      responses = responses, prior = prior,
      control = list(max_iter = 100L, tol = 1e-6, diagnose_k = FALSE,
                     diagnose_skew = FALSE))
  expect_null(fit_off$inner_skew)
})

# --------------------------------------------------------------------------- #
# (8) SPDE / GP bespoke Newton pair (gcol33/tulpa#273 item 3). cpp_laplace_fit_gp,
#     cpp_laplace_fit_spde and cpp_laplace_fit_spde_precomputed are standalone,
#     fixed-hyperparameter single fits with their OWN Newton implementation
#     (laplace_mode_gp / spde_run_single_fit) -- they are NOT reached by the
#     joint-multi driver's re-dispatch (the nested "nngp" / "spde" registry
#     entries integrate hyperparameters via the shared joint-multi machinery
#     instead, already covered above). Each has a dense branch
#     (laplace_newton_solve / run_spde_laplace) and a fully sparse
#     CHOLMOD-only branch (laplace_newton_solve_sparse) for n_x >=
#     SPARSE_THRESHOLD (200); both are exercised directly below via the
#     gaussian exact-zero invariant, which holds regardless of latent
#     structure (the third log-lik derivative is family-only).
# --------------------------------------------------------------------------- #

.isk_nngp_fixture <- function(n_spatial, nn_k = 10L, seed = 1) {
  set.seed(seed)
  coords <- cbind(runif(n_spatial), runif(n_spatial))
  order_idx <- order(coords[, 1], coords[, 2])
  coords_ord <- coords[order_idx, ]
  nn_idx <- matrix(0L, nrow = n_spatial, ncol = nn_k)
  nn_dist <- matrix(0, nrow = n_spatial, ncol = nn_k)
  for (i in 2:n_spatial) {
    dists <- sqrt((coords_ord[seq_len(i - 1), 1] - coords_ord[i, 1])^2 +
                  (coords_ord[seq_len(i - 1), 2] - coords_ord[i, 2])^2)
    n_cand <- min(length(dists), nn_k)
    ord <- order(dists)[seq_len(n_cand)]
    nn_idx[i, seq_len(n_cand)] <- ord
    nn_dist[i, seq_len(n_cand)] <- dists[ord]
  }
  list(coords = coords_ord, nn_idx = nn_idx, nn_dist = nn_dist,
       nn_order = as.integer(order_idx - 1L))
}

test_that("gamma_3 is exactly zero for a gaussian NNGP fit (dense Newton path)", {
  skip_on_cran()
  n_s <- 30L
  fx <- .isk_nngp_fixture(n_s, nn_k = 8L, seed = 21)
  w_true <- rnorm(n_s, 0, 0.3)
  y <- 1 + w_true + rnorm(n_s, 0, 1)

  fit <- tulpa:::cpp_laplace_fit_gp(
    y = as.numeric(y), n = rep(1L, n_s), X = matrix(1, n_s, 1),
    re_idx = rep(0, n_s), n_re_groups = 0L, sigma_re = 1.0,
    coords = fx$coords, nn_idx = fx$nn_idx, nn_dist = fx$nn_dist,
    nn_order = fx$nn_order, n_spatial = n_s, nn = 8L,
    sigma2_gp = 0.3, phi_gp = 0.4, cov_type = 0L,
    family = "gaussian",
    compute_skew = TRUE, skew_idx = as.integer(c(1, 2))
  )
  expect_true(fit$n_iter > 0)
  expect_lt(1L + n_s, 200L)  # this fixture must stay on the dense path
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 is exactly zero for a gaussian NNGP fit (sparse CHOLMOD path)", {
  skip_on_cran()
  n_s <- 210L
  fx <- .isk_nngp_fixture(n_s, nn_k = 10L, seed = 22)
  w_true <- rnorm(n_s, 0, 0.3)
  y <- 1 + w_true + rnorm(n_s, 0, 1)

  fit <- tulpa:::cpp_laplace_fit_gp(
    y = as.numeric(y), n = rep(1L, n_s), X = matrix(1, n_s, 1),
    re_idx = rep(0, n_s), n_re_groups = 0L, sigma_re = 1.0,
    coords = fx$coords, nn_idx = fx$nn_idx, nn_dist = fx$nn_dist,
    nn_order = fx$nn_order, n_spatial = n_s, nn = 10L,
    sigma2_gp = 0.3, phi_gp = 0.4, cov_type = 0L,
    family = "gaussian",
    compute_skew = TRUE, skew_idx = as.integer(c(1, 2))
  )
  expect_true(fit$n_iter > 0)
  expect_gte(1L + n_s, 200L)  # this fixture must force the sparse path
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 is exactly zero for a gaussian SPDE fit (dense Newton path)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  set.seed(23)
  n_obs <- 40L
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.4, 0.9), cutoff = 0.15)
  n_mesh <- mesh$n
  expect_lt(1L + n_mesh, 200L)  # this fixture must stay on the dense path

  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  w_true <- rnorm(n_mesh, 0, 0.3)
  y <- 1 + as.numeric(A %*% w_true) + rnorm(n_obs, 0, 1)

  range_true <- 0.3; sigma_true <- 0.4
  kappa <- sqrt(8) / range_true
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma_true)

  fit <- tulpa:::cpp_laplace_fit_spde(
    y = as.numeric(y), n_trials = rep(1L, n_obs), X = matrix(1, n_obs, 1),
    re_idx = rep(0, n_obs), n_re_groups = 0L, sigma_re = 1.0,
    A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = n_mesh, C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    kappa = kappa, tau_spde = tau_spde,
    family = "gaussian",
    compute_skew = TRUE, skew_idx = as.integer(c(1, 2))
  )
  expect_true(fit$n_iter > 0)
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 is exactly zero for a gaussian SPDE fit (sparse CHOLMOD path)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  set.seed(24)
  n_obs <- 150L
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.13, 0.3), cutoff = 0.05)
  n_mesh <- mesh$n
  expect_gte(1L + n_mesh, 200L)  # this fixture must force the sparse path

  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  w_true <- rnorm(n_mesh, 0, 0.2)
  y <- 1 + as.numeric(A %*% w_true) + rnorm(n_obs, 0, 1)

  range_true <- 0.2; sigma_true <- 0.3
  kappa <- sqrt(8) / range_true
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma_true)

  fit <- tulpa:::cpp_laplace_fit_spde(
    y = as.numeric(y), n_trials = rep(1L, n_obs), X = matrix(1, n_obs, 1),
    re_idx = rep(0, n_obs), n_re_groups = 0L, sigma_re = 1.0,
    A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = n_mesh, C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    kappa = kappa, tau_spde = tau_spde,
    family = "gaussian",
    compute_skew = TRUE, skew_idx = as.integer(c(1, 2))
  )
  expect_true(fit$n_iter > 0)
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 is exactly zero for a gaussian precomputed-rational SPDE fit", {
  skip_on_cran()
  set.seed(25)
  n <- 100L; h <- 1 / n; kappa <- 8
  C0 <- rep(h, n)
  G <- Matrix::bandSparse(n, n, c(-1, 0, 1),
                          list(rep(-1 / h, n - 1), rep(2 / h, n), rep(-1 / h, n - 1)))
  G <- as(G, "CsparseMatrix"); G[1, n] <- -1 / h; G[n, 1] <- -1 / h

  asm <- tulpa:::.spde_rational_assemble(C0, G, kappa = kappa, tau = 1,
                                         nu = 0.5, order = 4L, d = 2)
  R <- chol(as.matrix(asm$Q))
  x_true <- backsolve(R, rnorm(n))
  u_true <- as.numeric(asm$Pr %*% x_true)
  u_true <- u_true / sd(u_true)
  y <- 0.2 + u_true + rnorm(n, 0, 0.3)

  Qg  <- as(asm$Q, "generalMatrix")
  Prg <- as(asm$Pr, "CsparseMatrix")
  fit <- tulpa:::cpp_laplace_fit_spde_precomputed(
    y = as.numeric(y), n_trials = rep(1L, n), X = matrix(1, n, 1),
    re_idx = rep(0, n), n_re_groups = 0L, sigma_re = 1.0,
    n_obs = n, n_mesh = n,
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = Prg@x, Aeff_i = Prg@i, Aeff_p = Prg@p,
    family = "gaussian", phi = 0.09,
    compute_skew = TRUE, skew_idx = as.integer(c(1, 2))
  )
  expect_true(fit$n_iter > 0)
  expect_equal(fit$inner_skew, c(0, 0))
})
