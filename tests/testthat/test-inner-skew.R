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
  # ladder for gaussian is wrong. Asserted with expect_identical (tolerance 0):
  # this is the scalar K = 1 path, which the tensor generalization
  # (gcol33/tulpa#301) must leave bit for bit alone.
  set.seed(11)
  n <- 300L
  x <- rnorm(n)
  y <- 1 + 0.5 * x + rnorm(n, 0, 1)
  fit <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, n), X = cbind(1, x),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "gaussian", compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_identical(fit$inner_skew, c(0, 0))

  # The spec-driven single-process entry takes the same scalar oracle, and its
  # gaussian gamma_3 is exactly zero for the same reason.
  fit_spec <- tulpa:::cpp_laplace_fit_multi_re(
    y = as.numeric(y), n = rep(1L, n), X = cbind(1, x),
    re_idx_list = list(), re_ngroups = integer(0), re_sigma_list = list(),
    family = "gaussian", compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_identical(fit_spec$inner_skew, c(0, 0))
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
# (4) A multi-process likelihood is SCORED by the per-observation tensor       #
#     contraction (gcol33/tulpa#301), not declined                            #
# --------------------------------------------------------------------------- #

# gamma_3 as the derivation defines it, computed independently in R: the third
# derivative of a log posterior along x(t) = mode + t * Sigma e_i, divided by
# sigma_i^3. Nothing here touches the engine, so agreement pins the C++
# contraction against the formula rather than against itself.
.isk_num_hess <- function(f, p, h = 1e-4) {
  k <- length(p); H <- matrix(0, k, k)
  for (i in seq_len(k)) for (j in seq_len(k)) {
    ei <- rep(0, k); ei[i] <- h; ej <- rep(0, k); ej[j] <- h
    H[i, j] <- (f(p + ei + ej) - f(p + ei - ej) -
                f(p - ei + ej) + f(p - ei - ej)) / (4 * h^2)
  }
  (H + t(H)) / 2
}
.isk_along_curve_gamma3 <- function(f2, ctr) {
  S <- solve(-.isk_num_hess(f2, ctr))
  vapply(seq_along(ctr), function(i) {
    v <- S[, i]; s2 <- v[i]; s <- sqrt(s2)
    g <- function(t) f2(ctr + t * v)
    hh <- 0.15 * s / s2                 # an x_i displacement of 0.15 sigma
    (g(2 * hh) - 2 * g(hh) + 2 * g(-hh) - g(-2 * hh)) / (2 * hh^3) / s^3
  }, numeric(1))
}

test_that("gamma_3 scores a zero-inflated (multi-process) fit and tracks its exact skewness", {
  skip_on_cran()
  # A ZI mixture reads two linear predictors per observation, so there is no
  # per-eta third derivative. The per-observation tensor contraction
  # (src/curvature3_contract.h) supplies the cubic term instead of declining.
  set.seed(313)
  n <- 300L
  y <- ifelse(rbinom(n, 1, plogis(-0.6)) == 1, 0, rpois(n, exp(0.4)))
  fit_zi <- tulpa:::cpp_laplace_fit_multi_re(
    y = as.numeric(y), n = rep(1L, n), X = matrix(1, n, 1),
    re_idx_list = list(), re_ngroups = integer(0), re_sigma_list = list(),
    family = "poisson", X_zi = matrix(1, n, 1),
    compute_skew = TRUE, skew_idx = as.integer(1:2)
  )
  expect_true(all(is.finite(fit_zi$inner_skew)))
  expect_identical(fit_zi$inner_skew_declined, "")

  # The exact log posterior of this intercept-only ZI Poisson, with the priors
  # the kernel applies: N(0, 100^2) on the count block (tau = 1e-4) and
  # N(0, zi_prior_sd^2) with the default zi_prior_sd = 2.5 on the ZI block.
  zi_lp <- function(a, b) {
    mu <- exp(a); p <- plogis(b)
    ypos <- y[y > 0]
    sum(y == 0) * log(p + (1 - p) * exp(-mu)) +
      length(ypos) * log1p(-p) + sum(ypos) * a - length(ypos) * mu -
      sum(lgamma(ypos + 1)) - 0.5 * 1e-4 * a^2 - 0.5 * (1 / 2.5^2) * b^2
  }
  f2 <- function(p) zi_lp(p[1], p[2])
  ctr <- stats::optim(c(0, 0), function(v) -f2(v), method = "BFGS",
                      control = list(reltol = 1e-14))$par

  # (a) the contraction IS the cubic term of the formula, to finite-difference
  #     accuracy on both sides.
  expect_equal(fit_zi$inner_skew, .isk_along_curve_gamma3(f2, ctr),
               tolerance = 5e-3)

  # (b) and that term tracks the exact posterior: same sign, undershooting the
  #     exact marginal skewness as the leading-order expansion is documented to.
  q <- coupled_occ_quadrature(Vectorize(zi_lp), ctr, half = 10, n_grid = 1601L)
  exact <- c(q$a[["skew"]], q$b[["skew"]])
  expect_true(all(sign(fit_zi$inner_skew) == sign(exact)))
  expect_true(all(abs(fit_zi$inner_skew) < abs(exact)))
  # The count coordinate is barely skewed (|skew| ~ 0.11), which is the regime
  # the expansion is valid in; it must land close, not merely in the right
  # direction.
  expect_gt(fit_zi$inner_skew[1] / exact[1], 0.85)

  # The identical count-process model WITHOUT zero-inflation (n_processes == 1)
  # takes the scalar oracle -- confirms the tensor path is the multi-process
  # branch, not a change of dispatch for everything.
  fit_nozi <- tulpa:::cpp_laplace_fit_multi_re(
    y = as.numeric(rpois(n, exp(0.4))), n = rep(1L, n), X = matrix(1, n, 1),
    re_idx_list = list(), re_ngroups = integer(0), re_sigma_list = list(),
    family = "poisson",
    compute_skew = TRUE, skew_idx = 1L
  )
  expect_true(all(is.finite(fit_nozi$inner_skew)))
  expect_identical(fit_nozi$inner_skew_declined, "")
})

test_that("the inner-skew decline reason reaches the fit and the diagnostics layer", {
  # The R-side decline paths, exercised without a fit.
  d <- tulpa:::.inner_skew_decline(list(), "not_requested")
  expect_null(d$inner_skew)
  expect_identical(d$inner_skew_declined, "not_requested")
  expect_identical(d$inner_skew_arms_declined, integer(0))

  # The kernel's own reason is copied through verbatim, and "" means "scored".
  a <- tulpa:::.inner_skew_attach(list(), list(inner_skew = c(0.1, NaN),
                                               inner_skew_idx = c(1L, 2L),
                                               inner_skew_dropped = 4L,
                                               inner_skew_declined = "",
                                               inner_skew_arms_declined = c(2L, 3L)))
  expect_true(is.na(a$inner_skew_declined))
  expect_identical(a$inner_skew_arms_declined, c(2L, 3L))
  expect_identical(a$inner_skew_dropped, 4L)

  # Every reason reads back as a sentence; a structural one is flagged as such.
  for (r in c("not_requested", "no_probe_indices",
              "coupled_arm", "curvature3_unavailable", "no_finite_contribution",
              "no_oracle", "backend_unsupported", "solve_failed")) {
    expect_true(is.character(tulpa:::.inner_skew_decline_note(r)))
  }
  expect_null(tulpa:::.inner_skew_decline_note(NA_character_))
  # gcol33/tulpa#301 retired "coupled_likelihood": a multi-process spec is
  # scored by the per-observation tensor, so the reason has no producer left.
  expect_null(tulpa:::.inner_skew_decline_note("coupled_likelihood"))
  expect_false(tulpa:::.inner_skew_is_structural("coupled_likelihood"))
  expect_true(tulpa:::.inner_skew_is_structural("coupled_arm"))
  expect_false(tulpa:::.inner_skew_is_structural("not_requested"))
  expect_false(tulpa:::.inner_skew_is_structural(NA_character_))
  # The arms are named when the decline is partial or per-arm.
  expect_match(tulpa:::.inner_skew_decline_note("coupled_arm", c(1L, 2L)),
               "arms 1, 2", fixed = TRUE)
})

test_that("an unscorable inner layer is not reported as a disabled knob", {
  # The #296 motivating case: a fully coupled model printed as
  # `control$diagnose_skew = FALSE`, sending readers after a knob they had left
  # at its default TRUE. The verdict now says the layer is unscorable instead.
  structural <- tulpa:::.tulpa_combined_reliability(
    "unreliable", NA_character_, inner_declined = "coupled_arm")
  expect_match(structural, "not assessed")             # the #274 contract
  expect_match(structural, "unscorable for this model class", fixed = TRUE)

  off <- tulpa:::.tulpa_combined_reliability(
    "unreliable", NA_character_, inner_declined = "not_requested")
  expect_match(off, "not assessed")
  expect_false(grepl("unscorable", off))

  # And symmetrically for a family whose OUTER axis can never be scored.
  perm <- tulpa:::.tulpa_combined_reliability(
    NA_character_, "good", outer_declined = "unguessable_axis: rho_car")
  expect_match(perm, "unscorable for this family", fixed = TRUE)
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

test_that("gamma_3 is exactly zero for a coupled GAUSSIAN cell (K = 2 tensor, tolerance 0)", {
  skip_on_cran()
  # The K > 1 counterpart of the gaussian exact-zero invariant above. The
  # test-only bivariate gaussian CellCouplingSpec has a CONSTANT cross-arm
  # Hessian, so every difference quotient the tensor forms is exactly zero and
  # the contraction must return exactly 0 -- finite, not NaN (the cell IS
  # scorable) and not merely small. Before gcol33/tulpa#301 this fit came back
  # all-NaN because both arms were coupled and therefore excluded.
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
  expect_length(res$inner_skew, sum(res$arm_layout$p))
  expect_true(all(is.finite(res$inner_skew)))
  expect_identical(res$inner_skew, rep(0, sum(res$arm_layout$p)))
  expect_true(is.na(res$inner_skew_declined))
  expect_identical(res$inner_skew_arms_declined, integer(0))
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
# (8) Fixed-hyperparameter GP / SPDE single fits (gcol33/tulpa#273 item 3).
#     cpp_laplace_fit_gp, cpp_laplace_fit_spde and
#     cpp_laplace_fit_spde_precomputed are one-cell runs of the shared
#     joint-multi driver over their nested integrator's own LatentBlock
#     (gcol33/tulpa#277, #282), so they inherit the same gamma_3 pass the nested
#     entries take. The gaussian exact-zero invariant below is what pins that:
#     it holds regardless of latent structure or field size, because the third
#     log-lik derivative is family-only. Fixtures sit on both sides of
#     SPARSE_THRESHOLD (200) because n_x used to select the Newton container;
#     blocks_require_sparse() now decides it from the block, and both field
#     sizes stay covered.
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

test_that("gamma_3 is exactly zero for a gaussian NNGP fit (small field)", {
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
  expect_lt(1L + n_s, 200L)   # below SPARSE_THRESHOLD
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 is exactly zero for a gaussian NNGP fit (large field)", {
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
  expect_gte(1L + n_s, 200L)  # above SPARSE_THRESHOLD
  expect_equal(fit$inner_skew, c(0, 0))
})

test_that("gamma_3 is exactly zero for a gaussian SPDE fit (small mesh)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  set.seed(23)
  n_obs <- 40L
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.4, 0.9), cutoff = 0.15)
  n_mesh <- mesh$n
  expect_lt(1L + n_mesh, 200L)   # below SPARSE_THRESHOLD

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

test_that("gamma_3 is exactly zero for a gaussian SPDE fit (large mesh)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  set.seed(24)
  n_obs <- 150L
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.13, 0.3), cutoff = 0.05)
  n_mesh <- mesh$n
  expect_gte(1L + n_mesh, 200L)  # above SPARSE_THRESHOLD

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

# --------------------------------------------------------------------------- #
# (9) The exact-quadrature reference for a GENUINELY COUPLED likelihood        #
#     (gcol33/tulpa#300).                                                      #
#                                                                              #
# `.exact_intercept_skew()` above is the ground truth for the separable scalar #
# case: integrate the exact posterior on a grid, take its central moments, and #
# hold gamma_3 against them. Everything it covers is a single separable sum.   #
# A coupled cell has its per-obs sum replaced by a CellCouplingSpec term, and  #
# the cubic coefficient there is the contraction of the cell third-derivative  #
# tensor (gcol33/tulpa#301) -- a different computation reaching the same       #
# quantity, so it needs the same kind of arbiter.                              #
#                                                                              #
# The same construction is carried up to two dimensions here, over the         #
# engine's own coupled fixture (the two-arm occupancy mixture registered from  #
# src/ under "test_occupancy_mixture"): the conditional posterior of           #
# (beta_occ, beta_det) is integrated directly, with no Laplace approximation   #
# anywhere, so its marginal skewness is what a coupled cubic term has to       #
# reproduce.                                                                   #
#                                                                              #
# Three things have to hold for that number to be ground truth rather than a   #
# number: the quadrature has to reproduce the trusted scalar reference, the R  #
# density has to be the density the compiled spec evaluates, and the grid has  #
# to be converged. All three are asserted below.                               #
# --------------------------------------------------------------------------- #

test_that("the 2-D quadrature reproduces the scalar exact-skew reference on a product posterior", {
  skip_on_cran()
  # A posterior that separates, log post(a, b) = f(a) + g(b), has marginal-of-a
  # equal to f's own posterior. Summing b out of the two-dimensional grid must
  # therefore return what .exact_intercept_skew() returns for f -- the
  # machinery is checked against the reference it extends, on a case whose
  # answer is already known.
  sb <- 100
  f <- function(eta, N, S) S * eta - N * log1p(exp(eta)) - 0.5 * (eta / sb)^2
  prod_post <- function(a, b) f(a, 100, 3) + f(b, 200, 60)
  mode_a <- stats::optimize(f, c(-30, 30), N = 100, S = 3, maximum = TRUE)$maximum
  mode_b <- stats::optimize(f, c(-30, 30), N = 200, S = 60, maximum = TRUE)$maximum

  q <- coupled_occ_quadrature(prod_post, center = c(mode_a, mode_b),
                              half = 15, n_grid = 1501L)
  expect_equal(q$a[["skew"]], .exact_intercept_skew(100, 3), tolerance = 1e-4)
  expect_equal(q$b[["skew"]], .exact_intercept_skew(200, 60), tolerance = 1e-4)
})

test_that("the reference density is the density the compiled coupled spec evaluates", {
  skip_on_cran()
  # A quadrature of the wrong model is not a reference for anything. The R log
  # posterior is held against the compiled spec cell by cell, so what follows
  # integrates exactly what the inner Newton sees.
  coupled_occ_register()
  beta_prec <- 0.25
  d <- coupled_occ_data(seed = 311, n_cells = 100L, n_visits = 4L,
                        b_occ = 0.2, b_det = -0.5)
  lp <- coupled_occ_log_post(d, beta_prec)
  for (ab in list(c(0.0, 0.0), c(0.7, -1.1), c(-1.3, 0.4))) {
    cell_sum <- sum(vapply(seq_len(d$n_cells), function(cc) {
      rows <- ((cc - 1L) * d$n_visits + 1L):(cc * d$n_visits)
      cpp_cell_coupling_evaluate(
        "test_occupancy_mixture",
        eta = list(ab[1L], rep(ab[2L], d$n_visits)),
        y = list(0, d$y_det[rows]),
        family = c("binomial", "binomial"), phi = c(1, 1))$value
    }, numeric(1)))
    penalty <- 0.5 * beta_prec * (ab[1L]^2 + ab[2L]^2)
    expect_equal(lp(ab[1L], ab[2L]), cell_sum - penalty, tolerance = 1e-10)
  }
})

test_that("the coupled fixture has a converged, materially skewed exact posterior", {
  skip_on_cran()
  coupled_occ_register()
  beta_prec <- 0.25
  d <- coupled_occ_data(seed = 311, n_cells = 100L, n_visits = 4L,
                        b_occ = 0.2, b_det = -0.5)
  lp <- coupled_occ_log_post(d, beta_prec)
  ctr <- stats::optim(c(0, 0), function(v) -lp(v[1L], v[2L]),
                      method = "BFGS", control = list(reltol = 1e-14))$par

  coarse <- coupled_occ_quadrature(lp, ctr, half = 8,  n_grid = 901L)
  fine   <- coupled_occ_quadrature(lp, ctr, half = 16, n_grid = 1601L)
  # Widening the grid and halving the spacing does not move the answer: the
  # tails are inside the box and the peak is resolved.
  expect_equal(coarse$a, fine$a, tolerance = 1e-4)
  expect_equal(coarse$b, fine$b, tolerance = 1e-4)

  # Both coordinates are genuinely skewed, so a coupled cubic term checked
  # against this has something to be wrong about -- a fixture whose exact
  # posterior were Gaussian would certify nothing.
  expect_gt(fine$a[["skew"]], 0.4)
  expect_lt(fine$b[["skew"]], -0.08)
  expect_equal(.tulpa_gamma3_band(fine$a[["skew"]]), "ok")

  # The Laplace approximation the engine forms is measurably off here: the
  # exact posterior mean sits a tenth of a standard deviation up from the mode,
  # on the side the positive third moment puts it.
  expect_gt((fine$a[["mean"]] - ctr[1L]) / fine$a[["sd"]], 0.1)
})

test_that("the coupled cubic term reproduces the formula and tracks the exact skewness", {
  skip_on_cran()
  # The arbiter for gcol33/tulpa#301. Two things are checked, in that order,
  # because they answer different questions:
  #
  #  (a) does the cell tensor contraction compute the quantity the derivation
  #      defines? Held against the third derivative of the SAME exact log
  #      posterior along the SAME conditional-mean curve, computed in R with no
  #      engine involvement. Agreement here is exact up to finite differences.
  #  (b) does that quantity track the exact posterior? Held against the
  #      two-dimensional quadrature above. gamma_3 is a LEADING-ORDER estimate
  #      and undershoots as skewness grows (the scalar section (3) characterises
  #      the same behaviour), so the assertions are: right sign, undershoot in
  #      magnitude, and close agreement on the coordinate whose skewness is
  #      small enough for the expansion to be valid.
  coupled_occ_register()
  beta_prec <- 0.25
  d <- coupled_occ_data(seed = 311, n_cells = 100L, n_visits = 4L,
                        b_occ = 0.2, b_det = -0.5)
  lp <- coupled_occ_log_post(d, beta_prec)
  f2 <- function(p) lp(p[1], p[2])
  ctr <- stats::optim(c(0, 0), function(v) -f2(v), method = "BFGS",
                      control = list(reltol = 1e-14))$par

  fit <- tulpa_nested_laplace_joint(
    responses = coupled_occ_arms(d, beta_prec = 0.25),
    prior = coupled_occ_flat_prior(d),
    cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 300L, tol = 1e-12, diagnose_k = FALSE))
  expect_length(fit$inner_skew, sum(fit$arm_layout$p))
  expect_true(all(is.finite(fit$inner_skew)))
  expect_true(is.na(fit$inner_skew_declined))
  expect_identical(fit$inner_skew_arms_declined, integer(0))

  # (a)
  expect_equal(fit$inner_skew, .isk_along_curve_gamma3(f2, ctr),
               tolerance = 5e-3)

  # (b)
  q <- coupled_occ_quadrature(lp, ctr, half = 16, n_grid = 1601L)
  exact <- c(q$a[["skew"]], q$b[["skew"]])
  expect_true(all(sign(fit$inner_skew) == sign(exact)))
  expect_true(all(abs(fit$inner_skew) < abs(exact)))
  # The detection coordinate is only mildly skewed (|skew| ~ 0.13): the
  # expansion is valid there and must land close.
  expect_gt(fit$inner_skew[2] / exact[2], 0.8)
  # The occupancy coordinate is moderately skewed (~0.53) and the leading-order
  # term recovers a little over half of it. Pinned rather than smoothed over,
  # because it has a consequence a reader has to know: the exact skewness bands
  # "ok" while gamma_3 bands "good", so on a posterior this shape the band is
  # optimistic. gamma_3 is a lower bound on the skewness, not a two-sided
  # estimate of it.
  expect_gt(fit$inner_skew[1] / exact[1], 0.4)
  expect_lt(fit$inner_skew[1] / exact[1], 0.7)
  expect_identical(.tulpa_gamma3_band(exact[1]), "ok")
  expect_identical(.tulpa_gamma3_band(fit$inner_skew[1]), "good")
})
