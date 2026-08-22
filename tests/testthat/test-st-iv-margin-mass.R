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

# One probe at a chosen log_tau, reported as the conditioning each metric buys.
st_iv_cond_at <- function(f, log_tau) {
  probe <- st_iv_layout(f)
  set.seed(7)
  q <- stats::rnorm(probe$n_params, sd = 0.3)
  q[probe$log_tau_st_idx + 1L] <- log_tau
  idx <- (probe$st_delta_start + 1L):probe$st_delta_end
  Q <- -st_iv_num_hessian(f, q, idx)
  g <- st_iv_gmrf(f, q)
  M <- st_iv_dense_mass(g, f$S, f$T)
  Rc <- chol(M)
  Li <- backsolve(Rc, diag(nrow(M)))
  e <- eigen(crossprod(Li, Q %*% Li), symmetric = TRUE, only.values = TRUE)$values
  max(e) / min(e)
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
  # RW1's kernel is the constants alone, so the two sums span it and there is
  # no third family (gcol33/tulpa#600).
  expect_equal(r$lambda_trend, 0)

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
  # jac_tol: how far Jacobi moves the conditioning. Under RW1 it does not move
  # it at all. Under RW2 the trend family puts real mass ON the diagonal, so
  # equalizing the diagonal is no longer a no-op -- it moves it 16%, against
  # the 1000x the low-rank term buys, which is the claim being made.
  for (cfg in list(list(temporal = "rw1", jac_tol = 1e-3),
                   list(temporal = "rw2", jac_tol = 0.25))) {
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
    expect_equal(cond_jac / cond_raw, 1, tolerance = cfg$jac_tol)

    # The same diagonal PLUS every direction the penalty carries. The lambdas
    # come from S and T alone -- no position, no likelihood pass, no
    # factorization.
    M <- st_iv_dense_mass(pr$gmrf, f$S, f$T)
    Ei <- eigen(M, symmetric = TRUE)
    Mhalf_inv <- Ei$vectors %*% diag(1 / sqrt(Ei$values)) %*% t(Ei$vectors)
    cond_lr <- st_iv_cond(Mhalf_inv %*% Q %*% Mhalf_inv)

    expect_lt(cond_lr, min(cond_marg, cond_jac) / 1000)
    expect_lt(cond_lr, 100)
  }
})

test_that("the RW2 kernel's ramp carries the penalty's own curvature", {
  skip_on_cran()
  # gcol33/tulpa#600. null(Q_s (x) Q_t) = null(Q_s) (x) R^T + R^S (x) null(Q_t),
  # and a non-cyclic RW2 marginal puts a linear ramp in null(Q_t). A
  # site-CONTRAST ramp -- c summing to zero, w_s = c_s v -- is therefore
  # annihilated by the Kronecker term AND by both sums (v sums to zero across
  # time, c across space), so whatever curvature it has is the trend family's
  # alone. Before that family existed it had none but the likelihood's.
  f <- st_iv_fixture(temporal = "rw2")
  S <- f$S; T <- f$T
  pr <- st_iv_margin_probe(f)
  v <- st_iv_trend(T)
  vv <- sum(v^2)

  expect_equal(pr$gmrf$lambda_trend, 1 / (0.001 * vv)^2)

  cc <- numeric(S); cc[1] <- 1; cc[2] <- -1        # sums to zero
  w <- as.numeric(t(outer(cc, v)))                 # s * T + t layout
  rayleigh <- as.numeric(crossprod(w, pr$Q %*% w) / crossprod(w))

  # v' delta_s = c_s v'v on this direction, so the penalty's Rayleigh quotient
  # along it is exactly lambda_trend * v'v.
  expect_equal(rayleigh, pr$gmrf$lambda_trend * vv, tolerance = 1e-3)
})

test_that("a cyclic RW2 kernel gets no trend family", {
  skip_on_cran()
  # A linear ramp is not periodic, so rw2_rank reports T - 1 for the cyclic
  # operator and its kernel is the constants alone -- the two sums span it.
  f <- st_iv_fixture(temporal = "rw2", temporal_cyclic = TRUE)
  expect_equal(st_iv_margin_probe(f)$gmrf$lambda_trend, 0)
})

test_that("the metric's conditioning saturates in tau", {
  skip_on_cran()
  # The regression guard #600 asks for. A metric that spans LESS than the
  # penalty leaves a residual on the directions it misses, and those directions
  # do not stiffen with tau while every other one does -- so cond(M^-1 Q) grows
  # LINEARLY in tau rather than saturating. Measured on this fixture before the
  # trend family existed: RW2 went 13.6 / 18.6 / 75.2 / 529 / 3900 over
  # log_tau 0 to 6, a factor of 7.4 per factor of 7.4 in tau, while RW1
  # saturated at 17.4. A single-position probe cannot see that, which is why
  # the assertion is on the RATIO across two positions.
  for (temporal in c("rw1", "rw2")) {
    f <- st_iv_fixture(temporal = temporal)
    c4 <- st_iv_cond_at(f, 4)
    c6 <- st_iv_cond_at(f, 6)
    expect_lt(c6, 100)
    expect_lt(c6 / c4, 1.25)
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
                                     term$group_idx, term$lambda, p,
                                     group_w = term$group_w)
  expect_true(res$ok)
  expect_equal(res$inv_mass_times_p,
               as.numeric(solve(st_iv_dense_mass(r, S, T), p)),
               tolerance = 1e-8)

  # RW2 adds S weighted groups covering the same coordinates the row groups
  # do, differing only in their weights -- which is why they need the weighted
  # storage rather than a second term.
  f2 <- st_iv_fixture(temporal = "rw2")
  r2 <- st_iv_margin_probe(f2)$gmrf
  t2 <- cpp_test_margin_mass_term(S, T, r2$lambda_row, r2$lambda_col,
                                  r2$inv_mass, 0L, r2$lambda_trend)
  expect_true(t2$ok)
  expect_identical(t2$rank, as.integer(S + T + S))
  expect_equal(tail(t2$group_w, S * T),
               rep(st_iv_trend(T), times = S))
  res2 <- cpp_test_lowrank_mass_apply(r2$inv_mass, 0L, t2$group_ptr,
                                      t2$group_idx, t2$lambda, p,
                                      group_w = t2$group_w)
  expect_true(res2$ok)
  expect_equal(res2$inv_mass_times_p,
               as.numeric(solve(st_iv_dense_mass(r2, S, T), p)),
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
