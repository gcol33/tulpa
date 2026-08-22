# The Type-IV block's sum-to-zero margins as a low-rank mass term
# (gcol33/tulpa#597).
#
# test-lowrank-mass.R scores the storage and its algebra on group layouts of
# its own. This file is about the WIRING: that the engine hands the metric the
# margins the Type-IV precision actually carries, at the scaling the sampled
# parameterization implies, and that doing so buys the conditioning the
# diagonal cannot.
#
# The arbiter is the same one #585 used -- the numerical Hessian of the
# engine's own log-posterior over the interaction block (helper-st-iv.R).

# Condition number of a symmetric matrix.
st_iv_cond <- function(M) {
  e <- eigen(M, symmetric = TRUE, only.values = TRUE)$values
  max(e) / min(e)
}

# The block's exact precision at a position, plus the engine's own reads of it.
st_iv_margin_probe <- function(f, q_seed = 7) {
  probe <- st_iv_layout(f)
  set.seed(q_seed)
  q <- stats::rnorm(probe$n_params, sd = 0.3)
  q[probe$log_tau_st_idx + 1L] <- 0.4
  idx <- (probe$st_delta_start + 1L):probe$st_delta_end
  list(q = q, probe = probe, idx = idx,
       Q = -st_iv_num_hessian(f, q, idx),
       gmrf = st_iv_gmrf(f, q))
}

test_that("the engine reports the margins at the parameterization's scaling", {
  skip_on_cran()
  f <- st_iv_fixture()
  r <- st_iv_margin_probe(f)$gmrf
  expect_true(r$ok)
  expect_identical(r$n_spatial, as.integer(f$S))
  expect_identical(r$n_times, as.integer(f$T))
  # Centered: the penalty reads delta itself, so the precisions are the bare
  # s2z ones, each margin at its OWN length.
  expect_equal(r$lambda_row, 1 / (0.001 * f$T)^2)
  expect_equal(r$lambda_col, 1 / (0.001 * f$S)^2)

  # Non-centered: delta = z / sqrt(tau), so the penalty picks up the chain rule
  # twice and both margins scale by 1 / tau -- the same factor the assembled Q
  # carries, which is why the two are read off one result rather than derived
  # twice.
  fnc <- st_iv_fixture(st_parameterization = 1L)
  rnc <- st_iv_margin_probe(fnc)$gmrf
  expect_true(rnc$ok)
  expect_equal(rnc$lambda_row, (1 / exp(0.4)) / (0.001 * fnc$T)^2)
  expect_equal(rnc$lambda_col, (1 / exp(0.4)) / (0.001 * fnc$S)^2)
})

test_that("the margins are what a diagonal metric cannot rescale", {
  skip_on_cran()
  # A diagonal metric only rescales coordinates, so what any diagonal can reach
  # is bounded by cond(Q) after the best diagonal rescaling. Under inverse mass
  # v the leapfrog integrates diag(sqrt(v)) Q diag(sqrt(v)).
  for (cfg in list(list(temporal = "rw1"), list(temporal = "rw2"))) {
    f <- st_iv_fixture(temporal = cfg$temporal)
    pr <- st_iv_margin_probe(f)
    Q <- pr$Q
    v_marg <- pr$gmrf$inv_mass                      # what #585 installs
    scaled <- function(v) { d <- sqrt(v); st_iv_cond(outer(d, d) * Q) }

    cond_raw <- st_iv_cond(Q)
    cond_marg <- scaled(v_marg)
    cond_jac <- scaled(1 / diag(Q))
    expect_gt(cond_marg, 1e4)
    expect_gt(cond_jac, 1e4)
    # Jacobi is the diagonal rescaling that equalizes the diagonal, and it does
    # not move the conditioning at all: the stiffness is not on the diagonal.
    expect_equal(cond_jac / cond_raw, 1, tolerance = 1e-3)

    # The same diagonal PLUS the two margin directions. Both lambdas come from
    # S and T alone -- no position, no likelihood pass, no factorization.
    S <- f$S; T <- f$T
    R <- matrix(0, S, S * T); Cm <- matrix(0, T, S * T)
    for (s in seq_len(S)) R[s, ((s - 1) * T + 1):(s * T)] <- 1
    for (tt in seq_len(T)) Cm[tt, seq(tt, S * T, by = T)] <- 1
    M <- diag(1 / v_marg) + pr$gmrf$lambda_row * crossprod(R) +
      pr$gmrf$lambda_col * crossprod(Cm)
    Ei <- eigen(M, symmetric = TRUE)
    Mhalf_inv <- Ei$vectors %*% diag(1 / sqrt(Ei$values)) %*% t(Ei$vectors)
    cond_lr <- st_iv_cond(Mhalf_inv %*% Q %*% Mhalf_inv)

    expect_lt(cond_lr, min(cond_marg, cond_jac) / 1000)
    expect_lt(cond_lr, 100)
  }
})

test_that("the installed metric inverts the Type-IV precision-plus-margins", {
  skip_on_cran()
  # The engine's diagonal and lambdas, driven through the generic metric, have
  # to reproduce the dense M they describe. This is what ties the Type-IV
  # numbers to the storage test-lowrank-mass.R scores.
  f <- st_iv_fixture()
  r <- st_iv_margin_probe(f)$gmrf
  S <- f$S; T <- f$T
  term <- cpp_test_margin_mass_term(S, T, r$lambda_row, r$lambda_col,
                                    r$inv_mass)
  expect_true(term$ok)
  expect_identical(term$rank, as.integer(S + T))

  set.seed(3)
  p <- stats::rnorm(S * T)
  res <- cpp_test_lowrank_mass_apply(r$inv_mass, 0L, term$group_ptr,
                                     term$group_idx, term$lambda, p)
  expect_true(res$ok)

  R <- matrix(0, S, S * T); Cm <- matrix(0, T, S * T)
  for (s in seq_len(S)) R[s, ((s - 1) * T + 1):(s * T)] <- 1
  for (tt in seq_len(T)) Cm[tt, seq(tt, S * T, by = T)] <- 1
  M <- diag(1 / r$inv_mass) + r$lambda_row * crossprod(R) +
    r$lambda_col * crossprod(Cm)
  expect_equal(res$inv_mass_times_p, as.numeric(solve(M, p)),
               tolerance = 1e-8)
})

test_that("mass_matrix = 'gmrf_margin' parses and is distinct from 'gmrf'", {
  f <- st_iv_fixture()
  fit <- cpp_test_st_iv_nuts(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    mass_matrix = "gmrf_margin", n_iter = 4, n_warmup = 2, seed = 1)
  expect_identical(ncol(fit$draws), fit$n_params)
  expect_error(
    cpp_test_st_iv_nuts(
      f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
      mass_matrix = "gmrf_margins", n_iter = 4, n_warmup = 2, seed = 1),
    "mass_matrix must be one of")
})

test_that("a Type-IV NUTS fit under 'gmrf_margin' installs the margin term", {
  skip_if_not_slow()
  f <- st_iv_fixture(S1 = 4, S2 = 4, T = 6, family = "poisson")
  args <- list(f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx,
               f$S, f$T)

  fit <- do.call(cpp_test_st_iv_nuts,
                 c(args, list(mass_matrix = "gmrf_margin", n_iter = 400,
                              n_warmup = 200, seed = 3)))
  expect_true(fit$st_gmrf_applied)
  expect_true(fit$st_gmrf_margin_applied)
  expect_identical(fit$st_gmrf_declined, "")
  expect_true(all(is.finite(fit$draws)))

  # The reported diagonal is still the precision-informed one: the margin term
  # rides on top of it rather than replacing it, so a fit that drops the term
  # falls back to exactly the #585 metric.
  blk <- fit$inv_metric[(fit$st_delta_start + 1L):fit$st_delta_end]
  expect_true(all(blk >= 1e-3 & blk <= 1e3))

  # Asking for the plain precision-informed diagonal must not route through the
  # low-rank path, and asking for neither must not reach the override at all.
  fit_gmrf <- do.call(cpp_test_st_iv_nuts,
                      c(args, list(mass_matrix = "gmrf", n_iter = 4,
                                   n_warmup = 2, seed = 3)))
  expect_false(fit_gmrf$st_gmrf_margin_applied)
  fit_diag <- do.call(cpp_test_st_iv_nuts,
                      c(args, list(mass_matrix = "diag", n_iter = 4,
                                   n_warmup = 2, seed = 3)))
  expect_false(fit_diag$st_gmrf_applied)
  expect_false(fit_diag$st_gmrf_margin_applied)
})
