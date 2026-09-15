# The spatiotemporal interaction's AR1 time margin, and the GP interaction types
# the engine defines no density for.
#
# An AR1 margin used to be laid out (logit_rho_st) and given its prior, while
# the density never read rho: the Kronecker form had RW1 / RW2 arms only and the
# rank term mapped every other margin to rw2_rank. SEPARABLE / NONSEP_GP laid
# out two ranges that no density term read either (gcol33/tulpa#748).
#
# The arbiter is an independent DENSE computation in R: the structure matrix is
# built as a matrix (the AR1 precision as its tridiagonal, never through the
# engine's innovation factor), its rank and log pseudo-determinant are read off
# its eigenvalues, and the Gaussian log density is formed from them. Engine and
# reference are compared as DIFFERENCES between parameter points, so the
# constants neither side carries cancel.
#
# Fields are double-centred (every row and column sums to zero), so the soft
# sum-to-zero penalty is zero to rounding at every point and a 1e-8 comparison
# is not swamped by a 1e5-scale penalty. The penalty is still included in the
# reference.

st_d_fixture <- function(temporal = "ar1", st_type = "iv",
                         st_parameterization = 0, T = 5, family = "poisson") {
  f <- st_iv_fixture(S1 = 3, S2 = 3, T = T, family = family,
                     temporal = temporal,
                     st_parameterization = st_parameterization)
  f$st_type <- st_type
  f
}

st_d_args <- function(f) {
  list(y = f$y, X = f$X, s_idx = f$s_idx, t_idx = f$t_idx,
       adj_row_ptr = f$adj_row_ptr, adj_col_idx = f$adj_col_idx,
       S = f$S, T = f$T, family = f$family, temporal = f$temporal,
       temporal_cyclic = f$temporal_cyclic,
       st_parameterization = f$st_parameterization, st_type = f$st_type)
}

st_d_layout <- function(f) do.call(cpp_test_st_iv_layout, st_d_args(f))
st_d_prior  <- function(f, q) do.call(cpp_test_st_iv_log_prior,
                                      c(st_d_args(f), list(q = q)))
st_d_lp     <- function(f, q) do.call(cpp_test_st_iv_log_post,
                                      c(st_d_args(f), list(q = q)))

# ---------------------------------------------------------------------------
# Dense reference
# ---------------------------------------------------------------------------

ref_ar1_precision <- function(T, rho) {
  R <- diag(c(1, rep(1 + rho^2, T - 2), 1))
  R[cbind(1:(T - 1), 2:T)] <- -rho
  R[cbind(2:T, 1:(T - 1))] <- -rho
  R
}

ref_rw_precision <- function(T, order) crossprod(diff(diag(T), differences = order))

ref_time_precision <- function(temporal, T, rho) {
  switch(temporal,
         ar1 = ref_ar1_precision(T, rho),
         rw1 = ref_rw_precision(T, 1),
         rw2 = ref_rw_precision(T, 2))
}

ref_spatial_precision <- function(f) {
  S <- f$S
  Q <- matrix(0, S, S)
  for (s in seq_len(S)) {
    lo <- f$adj_row_ptr[s] + 1L
    hi <- f$adj_row_ptr[s + 1L]
    if (hi >= lo) for (k in lo:hi) Q[s, f$adj_col_idx[k]] <- -1
    Q[s, s] <- hi - lo + 1L
  }
  Q
}

ref_structure <- function(f, rho) {
  Qt <- ref_time_precision(f$temporal, f$T, rho)
  switch(f$st_type,
         ii = kronecker(diag(f$S), Qt),
         iv = kronecker(ref_spatial_precision(f), Qt))
}

ref_gauss <- function(x, P, tau) {
  ev <- eigen(P, symmetric = TRUE, only.values = TRUE)$values
  pos <- ev[ev > 1e-9 * max(ev)]
  0.5 * length(pos) * log(tau) + 0.5 * sum(log(pos)) -
    0.5 * tau * drop(crossprod(x, P %*% x))
}

ref_s2z_penalty <- function(delta, S, T) {
  M <- matrix(delta, S, T, byrow = TRUE)
  lam <- function(n) 1 / (1e-3 * n)^2
  -0.5 * (lam(T) * sum(rowSums(M)^2) + lam(S) * sum(colSums(M)^2))
}

# Prior on the sampled coordinates: PC prior on log tau (U = 1, alpha = 0.01,
# the fixture's anchors) and Uniform(-1, 1) on rho through logit((rho + 1) / 2).
ref_hyper <- function(log_tau, logit_rho) {
  rate <- -log(0.01) / 1
  sigma <- exp(-0.5 * log_tau)
  u <- stats::plogis(logit_rho)
  log(rate) - rate * sigma - log(2) - 0.5 * log_tau + log(u) + log(1 - u)
}

ref_prior <- function(f, z, log_tau, logit_rho) {
  tau <- exp(log_tau)
  rho <- 2 * stats::plogis(logit_rho) - 1
  P <- ref_structure(f, rho)
  nc <- f$st_parameterization == 1
  delta <- if (nc) z / sqrt(tau) else z
  ref_gauss(delta, P, tau) - (if (nc) 0.5 * length(z) * log(tau) else 0) +
    ref_s2z_penalty(delta, f$S, f$T) + ref_hyper(log_tau, logit_rho)
}

# An absolute comparison: a difference of log densities has no natural scale to
# be relative to, and one near zero would make a relative tolerance vacuous.
expect_abs_equal <- function(actual, expected, tol) {
  expect_true(is.finite(actual) && is.finite(expected))
  expect_lt(abs(actual - expected), tol)
}

double_centred <- function(S, T, seed) {
  set.seed(seed)
  M <- matrix(stats::rnorm(S * T), S, T)
  M <- sweep(M, 1, rowMeans(M))
  M <- sweep(M, 2, colMeans(M))
  as.numeric(t(M))                # s * T + t
}

# Two parameter points differing in the field, tau and rho at once.
st_d_points <- function(f) {
  lay <- st_d_layout(f)
  base <- numeric(lay$n_params)
  mk <- function(seed, log_tau, logit_rho) {
    q <- base
    q[(lay$st_delta_start + 1L):lay$st_delta_end] <- double_centred(f$S, f$T, seed)
    q[lay$log_tau_st_idx + 1L] <- log_tau
    q[lay$logit_rho_st_idx + 1L] <- logit_rho
    q
  }
  list(lay = lay, q1 = mk(11, 0.4, 1.1), q2 = mk(12, -0.3, -0.7))
}

expect_prior_matches_dense <- function(f) {
  pts <- st_d_points(f)
  lay <- pts$lay
  expect_gte(lay$logit_rho_st_idx, 0L)
  idx <- (lay$st_delta_start + 1L):lay$st_delta_end
  at <- function(q) ref_prior(f, q[idx], q[lay$log_tau_st_idx + 1L],
                              q[lay$logit_rho_st_idx + 1L])
  d_engine <- st_d_prior(f, pts$q2) - st_d_prior(f, pts$q1)
  d_ref <- at(pts$q2) - at(pts$q1)
  expect_abs_equal(d_engine, d_ref, 1e-8)
  invisible(c(engine = d_engine, ref = d_ref))
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

test_that("a correlation is laid out exactly where the density reads a time margin", {
  expect_gte(st_d_layout(st_d_fixture("ar1", "iv"))$logit_rho_st_idx, 0L)
  expect_gte(st_d_layout(st_d_fixture("ar1", "ii"))$logit_rho_st_idx, 0L)
  expect_identical(st_d_layout(st_d_fixture("ar1", "i"))$logit_rho_st_idx, -1L)
  expect_identical(st_d_layout(st_d_fixture("ar1", "iii"))$logit_rho_st_idx, -1L)
  expect_identical(st_d_layout(st_d_fixture("rw1", "iv"))$logit_rho_st_idx, -1L)
})

test_that("the GP interaction types are refused rather than laid out", {
  expect_error(st_d_layout(st_d_fixture("rw1", "separable")),
               "SEPARABLE spatiotemporal interaction has no density")
  expect_error(st_d_layout(st_d_fixture("rw1", "nonsep_gp")),
               "NONSEP_GP spatiotemporal interaction has no density")
})

test_that("a time margin with no density is refused where it would be read", {
  expect_error(st_d_layout(st_d_fixture("iid", "iv")), "temporal margin")
  expect_error(st_d_layout(st_d_fixture("iid", "ii")), "temporal margin")
  # Type III reads no temporal margin, so its temporal type is not consulted.
  expect_silent(st_d_layout(st_d_fixture("iid", "iii")))
})

# ---------------------------------------------------------------------------
# Density against the dense reference
# ---------------------------------------------------------------------------

test_that("Type IV with an AR1 margin is the Gaussian of Q_s (x) R(rho)", {
  expect_prior_matches_dense(st_d_fixture("ar1", "iv"))
})

test_that("Type II with an AR1 margin is the Gaussian of I_S (x) R(rho)", {
  expect_prior_matches_dense(st_d_fixture("ar1", "ii"))
})

test_that("non-centred Type IV with an AR1 margin carries the same prior on z", {
  expect_prior_matches_dense(st_d_fixture("ar1", "iv", st_parameterization = 1))
})

test_that("the reference reproduces the RW margins the engine already had", {
  f <- st_d_fixture("rw1", "iv")
  pts <- st_d_points(f)
  lay <- pts$lay
  idx <- (lay$st_delta_start + 1L):lay$st_delta_end
  at <- function(q) {
    tau <- exp(q[lay$log_tau_st_idx + 1L])
    rate <- -log(0.01)
    ref_gauss(q[idx], ref_structure(f, 0), tau) +
      ref_s2z_penalty(q[idx], f$S, f$T) +
      log(rate) - rate / sqrt(tau) - log(2) - 0.5 * log(tau)
  }
  q1 <- pts$q1; q2 <- pts$q2
  expect_abs_equal(st_d_prior(f, q2) - st_d_prior(f, q1), at(q2) - at(q1), 1e-8)
})

test_that("the log-posterior moves with logit_rho_st by the prior's own change", {
  for (nc in c(0, 1)) {
    f <- st_d_fixture("ar1", "iv", st_parameterization = nc)
    pts <- st_d_points(f)
    lay <- pts$lay
    q1 <- pts$q1
    q2 <- q1
    q2[lay$logit_rho_st_idx + 1L] <- -0.7
    idx <- (lay$st_delta_start + 1L):lay$st_delta_end
    lt <- q1[lay$log_tau_st_idx + 1L]
    d_lp <- st_d_lp(f, q2) - st_d_lp(f, q1)
    d_ref <- ref_prior(f, q2[idx], lt, -0.7) - ref_prior(f, q1[idx], lt, 1.1)
    # At a fixed field (and fixed tau under NC) eta does not move, so the whole
    # change is the prior's.
    expect_abs_equal(d_lp, d_ref, 1e-8)
    # And it is not the rho hyperprior alone: the field density reads rho.
    d_hyper <- ref_hyper(lt, -0.7) - ref_hyper(lt, 1.1)
    expect_gt(abs(d_lp - d_hyper), 1e-2)
  }
})

test_that("an HSGP-ST AR1 margin reads rho in its normalizer and its form", {
  EIG <- c(0.4, 1.1, 2.7)
  M <- length(EIG)
  T_st <- 6L
  lp <- function(delta, logit_rho) {
    cpp_test_st_hsgp_log_prior(delta = delta, eigenvalues = EIG, T = T_st,
                               log_tau_st = 0.3, log_sigma2_hsgp = -0.2,
                               log_lengthscale_hsgp = 0.1, temporal = "ar1",
                               temporal_cyclic = FALSE, logit_rho_st = logit_rho)
  }
  zero <- numeric(M * T_st)
  rho <- function(x) 2 * stats::plogis(x) - 1
  hyper <- function(x) log(stats::plogis(x)) + log(1 - stats::plogis(x))

  # At a zero field the only rho dependence is one 0.5 log|R(rho)| per basis
  # function plus the rho hyperprior.
  logdet <- function(x) as.numeric(determinant(ref_ar1_precision(T_st, rho(x)))$modulus)
  expect_abs_equal(lp(zero, -0.7) - lp(zero, 1.1),
                   0.5 * M * (logdet(-0.7) - logdet(1.1)) + hyper(-0.7) - hyper(1.1),
                   1e-10)

  # On one basis function the form is x' R(rho) x at that function's precision,
  # so the ratio of two zero-sum series' contributions is free of the spectral
  # density.
  x <- c(1.2, -0.4, 0.9, -1.5, 0.3, -0.5)
  y <- c(-0.8, 1.1, 0.2, 0.6, -1.9, 0.8)
  on_basis <- function(v, j) { d <- zero; d[(j - 1) * T_st + seq_len(T_st)] <- v; d }
  R <- ref_ar1_precision(T_st, rho(0.6))
  ratio <- (lp(on_basis(x, 2), 0.6) - lp(zero, 0.6)) /
           (lp(on_basis(y, 2), 0.6) - lp(zero, 0.6))
  expect_equal(ratio, drop(crossprod(x, R %*% x)) / drop(crossprod(y, R %*% y)),
               tolerance = 1e-8)
})

# ---------------------------------------------------------------------------
# The matrix form behind the precision-informed mass reads the same margin
# ---------------------------------------------------------------------------

test_that("the Type-IV mass override reproduces the log-posterior Hessian under AR1", {
  skip_on_cran()
  f <- st_iv_fixture(family = "poisson", temporal = "ar1")
  probe <- st_iv_layout(f)
  set.seed(7)
  q <- stats::rnorm(probe$n_params, sd = 0.3)
  q[probe$log_tau_st_idx + 1L] <- 0.4
  q[probe$logit_rho_st_idx + 1L] <- 0.9

  res <- st_iv_gmrf(f, q)
  expect_true(res$ok)
  expect_identical(res$reason, "")

  idx <- (probe$st_delta_start + 1L):probe$st_delta_end
  H <- st_iv_num_hessian(f, q, idx)
  expect_equal(res$inv_mass, diag(solve(-H)), tolerance = 1e-4)
})
