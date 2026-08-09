# Posterior-draw and diagnostic accessors on a single-block nested-Laplace fit
# (gcol33/tulpa#347, #348, #349).
#
# The posterior of such a fit is the outer-grid Gaussian mixture retained as
# `$grid_modes` / `$grid_hessians` / `$weights`, so the equivalence gate is that
# the sampler and the reporting methods read ONE object: sampled moments must
# reproduce `.nested_fixed_moments()` and sampled quantiles must reproduce
# `confint()`'s mixture-CDF bounds. The tier-1 blocks build that mixture by hand
# so the arbiter is exact arithmetic rather than a fit; the tier-2 block runs the
# same gate on a real fit.

# A hand-built single-block nested-Laplace fit whose retained mixture is exactly
# the components passed in. `mu` is n_cell x p, `V` a list of p x p covariances,
# `w` the raw (unnormalized) grid weights.
nl_mixture_fit <- function(mu, V, w, drop = integer(0)) {
  p <- ncol(mu)
  H <- lapply(V, solve)
  H[drop] <- list(NULL)
  structure(
    list(
      modes = mu, weights = w,
      grid_modes = lapply(seq_len(nrow(mu)), function(k) mu[k, ]),
      grid_hessians = H,
      n_fixed = p, fixed_names = paste0("b", seq_len(p)),
      backend = "nested_laplace", draws_kind = "iid"
    ),
    class = c("tulpa_nested_laplace", "list", "tulpa_fit")
  )
}

nl_fixture <- function() {
  nl_mixture_fit(
    mu = rbind(c(-0.4, 1.2), c(0.1, 0.9), c(0.7, 1.5)),
    V = list(matrix(c(0.30, 0.05, 0.05, 0.20), 2, 2),
             matrix(c(0.12, -0.03, -0.03, 0.45), 2, 2),
             matrix(c(0.50, 0.00, 0.00, 0.10), 2, 2)),
    w = c(2, 5, 3)
  )
}

test_that("the mixture sampler reproduces the retained moments and bounds", {
  fit <- nl_fixture()
  mom <- tulpa:::.nested_fixed_moments(fit)

  set.seed(11)
  dr <- tulpa_posterior_draws(fit, n = 400000L)

  expect_equal(dim(dr), c(400000L, 2L))
  expect_identical(attr(dr, "draws_kind"), "iid")
  expect_identical(attr(dr, "scope"), "fixed")
  expect_equal(attr(dr, "retained_mass"), 1)
  expect_identical(colnames(dr), c("b1", "b2"))

  # Moments: the Monte-Carlo standard error of a mean is sd / sqrt(n), so a
  # tolerance of 0.02 marginal SDs is ~12 MC standard errors at n = 4e5.
  sd_m <- sqrt(diag(mom$cov))
  expect_lt(max(abs(colMeans(dr) - mom$mean) / sd_m), 0.02)
  expect_lt(max(abs(stats::cov(dr) - mom$cov) / tcrossprod(sd_m)), 0.02)

  # Bounds: the sampled quantiles must reproduce the mixture-CDF inversion the
  # fit reports, at more than one level.
  for (lev in c(0.95, 0.80)) {
    a <- (1 - lev) / 2
    ci <- confint(fit, level = lev)
    sq <- t(apply(dr, 2L, stats::quantile, c(a, 1 - a)))
    expect_lt(max(abs(sq - ci) / sd_m), 0.02)
  }
  expect_identical(attr(confint(fit), "interval_source"), "mixture_cdf")
})

test_that("a single-component grid samples that component's Gaussian", {
  fit <- nl_mixture_fit(mu = rbind(c(2, -1)),
                        V = list(matrix(c(0.4, 0.1, 0.1, 0.25), 2, 2)),
                        w = 1)
  set.seed(12)
  dr <- tulpa_posterior_draws(fit, n = 200000L)
  expect_equal(colMeans(dr), c(b1 = 2, b2 = -1), tolerance = 0.02)
  expect_equal(as.numeric(stats::cov(dr)), c(0.4, 0.1, 0.1, 0.25),
               tolerance = 0.02)
  expect_true(all(attr(dr, "cells") == 1L))
})

test_that("a grid that dropped a positive-weight cell samples the rest", {
  fit <- nl_mixture_fit(
    mu = rbind(c(-0.4, 1.2), c(0.1, 0.9), c(0.7, 1.5)),
    V = list(matrix(c(0.30, 0.05, 0.05, 0.20), 2, 2),
             matrix(c(0.12, -0.03, -0.03, 0.45), 2, 2),
             matrix(c(0.50, 0.00, 0.00, 0.10), 2, 2)),
    w = c(2, 5, 3), drop = 2L)

  mom <- tulpa:::.nested_fixed_moments(fit)
  expect_equal(mom$mass, 0.5)
  expect_identical(mom$keep, c(1L, 3L))

  set.seed(13)
  dr <- tulpa_posterior_draws(fit, n = 100000L)
  # The provenance travels with the draws, exactly as it does with confint().
  expect_equal(attr(dr, "retained_mass"), 0.5)
  expect_equal(attr(dr, "retained_mass"), attr(confint(fit), "retained_mass"))
  # No row came from the dropped cell, and the retained cells appear in their
  # renormalized proportion (2:3).
  expect_setequal(unique(attr(dr, "cells")), c(1L, 3L))
  expect_equal(mean(attr(dr, "cells") == 1L), 0.4, tolerance = 0.02)
  sd_m <- sqrt(diag(mom$cov))
  expect_lt(max(abs(colMeans(dr) - mom$mean) / sd_m), 0.03)
})

test_that("what a draw covers is stated, and asking past it says why", {
  fit <- nl_fixture()
  expect_equal(dim(tulpa_posterior_draws(fit, idx = 2L, n = 7L)), c(7L, 1L))
  expect_identical(colnames(tulpa_posterior_draws(fit, idx = 2L, n = 7L)), "b2")
  # The reason names the retained representation rather than the bare bound.
  expect_error(tulpa_posterior_draws(fit, idx = 3L),
               "fixed-effect block")
  expect_error(tulpa_posterior_draws(fit, idx = 3L),
               "releases the cell precision")
  expect_error(tulpa_posterior_draws(fit, n = 0), "positive integer")
})

test_that("a fit with no retained mixture, and a non-nested object, say why", {
  bare <- structure(list(backend = "nested_laplace"),
                    class = c("tulpa_nested_laplace", "list", "tulpa_fit"))
  expect_error(tulpa_posterior_draws(bare), "keep_grid_hessians")
  expect_error(tulpa_posterior_draws(structure(list(), class = "lm")),
               "nested-Laplace fits")
})

test_that("a non-PD retained component is named, not silently sampled", {
  fit <- nl_fixture()
  fit$grid_hessians[[2]] <- matrix(c(1, 2, 2, 1), 2, 2)   # indefinite
  expect_error(tulpa_posterior_draws(fit), "positive definite")
  expect_error(tulpa_posterior_draws(fit), "grid_hessians")
})

test_that(".nl_mixture_draw allocates rows by weight and tags their cells", {
  set.seed(14)
  out <- tulpa:::.nl_mixture_draw(
    w = c(0.2, 0.8), cell_id = c(4L, 9L), n = 20000L, p = 1L,
    draw_cell = function(i, n_i) matrix(i, n_i, 1L))
  expect_equal(dim(out), c(20000L, 1L))
  expect_setequal(unique(attr(out, "cells")), c(4L, 9L))
  expect_equal(mean(attr(out, "cells") == 4L), 0.2, tolerance = 0.02)
  # Every row carries the value its own component produced.
  expect_true(all(out[attr(out, "cells") == 4L, 1L] == 1))
  expect_true(all(out[attr(out, "cells") == 9L, 1L] == 2))
})

# --- gcol33/tulpa#349 -------------------------------------------------------

test_that("the draws accessors name the backend and what the fit carries", {
  fit <- nl_fixture()
  expect_message(res <- posterior_sample(fit), "nested_laplace")
  expect_null(res)
  expect_message(posterior_sample(fit), "carries no posterior draws")
  expect_message(posterior_sample(fit), "tulpa_posterior_draws")
  expect_message(posterior_sample(fit), "grid_modes")

  expect_message(arr <- tulpa_draws_array(fit), "tulpa_draws_array")
  expect_null(arr)
  expect_message(tulpa_draws_array(fit), "tulpa_posterior_draws")
})

test_that("the note describes each posterior representation it can meet", {
  note <- function(x) tulpa:::.tulpa_no_draws_note(x, "posterior_sample")

  expect_match(note(list(backend = "nested_laplace_joint", modes = matrix(0),
                         Q_csc_p_per_grid = list(1))),
               "full latent vector")
  expect_match(note(list(backend = "nested_laplace", modes = matrix(0, 1, 1),
                         weights = 1)),
               "keep_grid_hessians")
  expect_match(note(list(backend = "laplace", mode = 1, H_beta = matrix(1))),
               "Laplace mode")
  expect_match(note(list(backend = "eb", means = 1)), "posterior moments")
  expect_match(note(structure(list(), class = "tulpa_fit")),
               "no posterior representation")
})

test_that("the internal probes stay silent so their fallback is not narrated", {
  fit <- nl_fixture()
  expect_silent(tulpa:::.tulpa_draws_array(fit))
  expect_null(tulpa:::.tulpa_draws_array(fit))
  expect_silent(tulpa:::.fit_draws(fit))
})

# --- gcol33/tulpa#348 -------------------------------------------------------

# The reliability quantities live on the fit, so a draws-less fit can carry a
# full band. These are the fields the band readers consume.
nl_diag_fixture <- function() {
  fit <- nl_fixture()
  fit$pareto_k <- 1.016155
  fit$pareto_k_is_ess <- 42.5
  fit$pareto_k_scope <- "outer (hyperparameter) Gaussian proposal"
  fit$pareto_k_declined <- NA_character_
  fit$inner_skew <- c(0.21, 0.08)
  fit$inner_skew_idx <- c(1L, 2L)
  fit$inner_skew_dropped <- 0L
  fit$inner_pareto_k <- c(0.7041, 0.3012)
  fit$inner_pareto_k_rel_ess <- c(0.42, 0.51)
  fit$inner_pareto_k_is_ess <- c(80, 95)
  fit
}

test_that("diagnostics() reports the band on a fit with no draws", {
  fit <- nl_diag_fixture()
  d <- diagnostics(fit)

  expect_s3_class(d, "laplace_diagnostics")
  expect_equal(nrow(d), 0L)
  expect_identical(names(d), c("parameter", "mean", "sd", "ess_bulk",
                               "ess_tail", "rhat"))

  # The headline the guard used to withhold: a k-hat past the escalation
  # threshold is reported, banded, and folded into the verdict.
  expect_equal(attr(d, "pareto_k"), 1.016155)
  expect_identical(attr(d, "pareto_k_band"), "unreliable")
  expect_true(is.finite(attr(d, "inner_skew_max")))
  expect_true(is.finite(attr(d, "inner_pareto_k")))
  expect_true(is.character(attr(d, "reliability")))
  expect_true(is.finite(attr(d, "ess_grid")))

  s <- attr(d, "summary")
  expect_true(is.na(s$n_draws))
  expect_equal(s$pareto_k, 1.016155)
  expect_identical(s$reliability, attr(d, "reliability"))

  # The empty body records WHY and points at the accessor that fills it.
  expect_match(attr(d, "param_table_declined"), "no posterior draws")
  expect_match(attr(d, "param_table_declined"), "tulpa_posterior_draws")
})

test_that("the printed band survives an absent posterior sample", {
  out <- capture.output(print(diagnostics(nl_diag_fixture())))
  expect_true(any(grepl("pareto_k = 1.016", out, fixed = TRUE)))
  expect_true(any(grepl("unreliable", out, fixed = TRUE)))
  expect_true(any(grepl("whole-fit verdict", out, fixed = TRUE)))
  expect_true(any(grepl("per-parameter columns: none", out, fixed = TRUE)))
  expect_false(any(grepl("NA draws", out, fixed = TRUE)))
})

test_that("the same fit WITH draws reports the same band plus the body", {
  fit <- nl_diag_fixture()
  bare <- diagnostics(fit)
  set.seed(15)
  fit$draws <- tulpa_posterior_draws(fit, n = 500L)
  with_draws <- diagnostics(fit)

  expect_equal(nrow(with_draws), 2L)
  expect_equal(attr(with_draws, "pareto_k"), attr(bare, "pareto_k"))
  expect_identical(attr(with_draws, "reliability"), attr(bare, "reliability"))
  expect_equal(attr(with_draws, "summary")$n_draws, 500L)
  expect_null(attr(with_draws, "param_table_declined"))
})

test_that("a fit with neither draws nor a band still gets NULL, and says so", {
  # A plain Laplace fit has no outer grid to score, so both layers are empty and
  # a table would be a report about the absence of a report.
  fit <- structure(
    list(backend = "laplace", draws_kind = "iid",
         mode = c(0.2, 0.5), H_beta = diag(2)),
    class = c("tulpa_fit", "list"))
  expect_message(res <- diagnostics(fit), "no posterior draws")
  expect_message(diagnostics(fit), "nothing to report")
  expect_null(res)
})

test_that("a bad `pars` selector on a fit WITH draws is still an empty answer", {
  fit <- nl_diag_fixture()
  set.seed(16)
  fit$draws <- tulpa_posterior_draws(fit, n = 100L)
  expect_null(diagnostics(fit, pars = "not_a_parameter"))
})

# --- tier 2: the same gate on a real fit ------------------------------------

test_that("a fitted single-block nested-Laplace posterior samples its own grid", {
  skip_on_cran()

  set.seed(1)
  G <- 25L; m <- 8L; N <- G * m
  g <- rep(seq_len(G), each = m)
  u <- stats::rnorm(G, 0, 0.8)
  x <- stats::rnorm(N)
  y <- stats::rbinom(N, 1L, stats::plogis(-0.3 + 0.7 * x + u[g]))

  fit <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, N), X = cbind(int = 1, x = x),
    prior = list(list(type = "iid", obs_idx = g - 1L, n_units = G,
                      sigma_grid = exp(seq(log(0.2), log(2.5),
                                           length.out = 9)))),
    family = "binomial", phi = 1,
    control = list(keep_grid_hessians = TRUE, diagnose_k = FALSE,
                   diagnose_skew = FALSE))

  expect_null(fit[["draws"]])
  mom <- tulpa:::.nested_fixed_moments(fit)

  set.seed(17)
  dr <- tulpa_posterior_draws(fit, n = 200000L)
  expect_identical(colnames(dr), c("int", "x"))
  expect_identical(attr(dr, "scope"), "fixed")

  sd_m <- sqrt(diag(mom$cov))
  expect_lt(max(abs(colMeans(dr) - mom$mean) / sd_m), 0.02)
  expect_lt(max(abs(stats::cov(dr) - mom$cov) / tcrossprod(sd_m)), 0.03)

  ci <- confint(fit)
  sq <- t(apply(dr, 2L, stats::quantile, c(0.025, 0.975)))
  expect_lt(max(abs(sq - ci) / sd_m), 0.03)

  # And the accessors that had nothing to say about this fit now do.
  expect_message(posterior_sample(fit), "nested_laplace")
  d <- diagnostics(fit)
  expect_s3_class(d, "laplace_diagnostics")
  expect_true(is.finite(attr(d, "ess_grid")))
})
