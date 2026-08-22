# Type-IV precision-informed diagonal mass (gcol33/tulpa#585).
#
# The arbiter throughout is the NUMERICAL HESSIAN of the engine's own
# log-posterior, restricted to the interaction block. diag(Q^-1) is defined as
# the marginal variance of the Gaussian that Hessian defines, so the assembly,
# the working-weight pass through eta_weights_fn, the sum-to-zero margins, the
# centered/non-centered scaling, the selected inversion and its permutation are
# all scored against something outside the assembly rather than against a
# second copy of it.
#
# That comparison is EXACT for both fixture families, not an approximation:
# the Type-IV prior is quadratic in the block, and exp(eta) (poisson) /
# exp(-2 log_sigma) (gaussian) are the exact second eta-derivatives, not
# expected-information stand-ins.

# The whole comparison, for one configuration.
expect_gmrf_matches_hessian <- function(f, q_seed = 7, tol = 1e-4) {
  probe <- st_iv_layout(f)
  n_params <- probe$n_params
  set.seed(q_seed)
  q <- stats::rnorm(n_params, sd = 0.3)
  q[probe$log_tau_st_idx + 1L] <- 0.4        # a tau away from 1 on both scales

  res <- st_iv_gmrf(f, q)
  expect_true(res$ok)
  expect_identical(res$reason, "")

  idx <- (probe$st_delta_start + 1L):probe$st_delta_end
  H <- st_iv_num_hessian(f, q, idx)
  ref <- diag(solve(-H))

  expect_equal(res$inv_mass, ref, tolerance = tol)
  invisible(res)
}

test_that("layout probing reports the interaction block", {
  f <- st_iv_fixture()
  probe <- st_iv_layout(f)
  expect_identical(probe$st_delta_end - probe$st_delta_start,
                   as.integer(f$S * f$T))
  # A wrong-length position is refused rather than read past.
  expect_error(st_iv_gmrf(f, rep(0, probe$n_params - 1L)), "q has")
})

test_that("diag(Q^-1) reproduces the log-posterior Hessian: poisson RW1", {
  skip_on_cran()
  expect_gmrf_matches_hessian(st_iv_fixture(family = "poisson", temporal = "rw1"))
})

test_that("diag(Q^-1) reproduces the log-posterior Hessian: poisson RW2", {
  skip_on_cran()
  expect_gmrf_matches_hessian(st_iv_fixture(family = "poisson", temporal = "rw2"))
})

test_that("diag(Q^-1) reproduces the log-posterior Hessian: gaussian RW1", {
  skip_on_cran()
  expect_gmrf_matches_hessian(st_iv_fixture(family = "gaussian", temporal = "rw1"))
})

test_that("the non-centered parameterization carries its own tau scaling", {
  skip_on_cran()
  # In z coordinates the prior is tau-free and BOTH the likelihood and the
  # sum-to-zero margins pick up 1/tau. Getting either scale wrong still
  # produces a finite, plausible-looking variance, so the Hessian is what says
  # the two parameterizations were not conflated.
  expect_gmrf_matches_hessian(
    st_iv_fixture(family = "poisson", st_parameterization = 1))
  expect_gmrf_matches_hessian(
    st_iv_fixture(family = "gaussian", st_parameterization = 1))
})

test_that("a disconnected spatial graph is handled at its true rank", {
  skip_on_cran()
  # Unit 5 (the grid centre) is cut loose: its Q_s degree is 0, so the whole
  # interaction over that unit is pinned by the sum-to-zero margins alone.
  f <- st_iv_fixture(family = "poisson", drop_edges_of = 5L)
  res <- expect_gmrf_matches_hessian(f)
  expect_identical(res$n_block, as.integer(f$S * f$T))
})

test_that("a spec with no IRLS callback declines, and says so", {
  f <- st_iv_fixture()
  probe <- st_iv_layout(f)
  q <- rep(0, probe$n_params)
  res <- st_iv_gmrf(f, q, with_eta_weights = FALSE)
  expect_false(res$ok)
  expect_identical(res$reason, "no_eta_weights_fn")
  expect_length(res$inv_mass, 0L)
})

test_that("mass_matrix names resolve, and an unknown one is refused", {
  f <- st_iv_fixture()
  # Reaching the sampler at all proves "gmrf" parses; a two-iteration run is
  # the cheapest way to ask.
  fit <- cpp_test_st_iv_nuts(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    mass_matrix = "gmrf", n_iter = 4, n_warmup = 2, seed = 1)
  expect_identical(ncol(fit$draws), fit$n_params)
  expect_error(
    cpp_test_st_iv_nuts(
      f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
      mass_matrix = "sparse_gmrf", n_iter = 4, n_warmup = 2, seed = 1),
    "mass_matrix must be one of")
})

test_that("a Type-IV NUTS fit under mass_matrix = 'gmrf' installs the override", {
  skip_if_not_slow()
  f <- st_iv_fixture(S1 = 4, S2 = 4, T = 6, family = "poisson")
  fit <- cpp_test_st_iv_nuts(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    mass_matrix = "gmrf", n_iter = 400, n_warmup = 200, seed = 3)
  expect_true(fit$st_gmrf_applied)
  expect_identical(fit$st_gmrf_declined, "")

  # The installed block is the override's own answer at some warmup position,
  # not the Welford variance, so it has to be positive and inside the shared
  # clamp band rather than merely finite.
  blk <- fit$inv_metric[(fit$st_delta_start + 1L):fit$st_delta_end]
  expect_true(all(is.finite(blk)))
  expect_true(all(blk >= 1e-3 & blk <= 1e3))

  # The plain diagonal is a valid metric for the same model, and asking for it
  # must not silently route through the override.
  fit_diag <- cpp_test_st_iv_nuts(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    mass_matrix = "diag", n_iter = 400, n_warmup = 200, seed = 3)
  expect_false(fit_diag$st_gmrf_applied)
})


test_that("the cyclic flag reaches the interaction's quadratic form", {
  # SpatiotemporalData::temporal_cyclic used to reach the rank in the
  # normalizer and not the quadratic form beside it (gcol33/tulpa#596), so a
  # cyclic RW2 interaction put rank_space * (T - 1) powers of tau in the
  # target against an operator of rank rank_space * (T - 2). The wrap edges
  # change the density at any position, so a flag that stops short of the
  # quadratic form shows here as two identical log-posteriors.
  f_acyc <- st_iv_fixture(family = "poisson", temporal = "rw2")
  f_cyc  <- st_iv_fixture(family = "poisson", temporal = "rw2",
                          temporal_cyclic = TRUE)
  probe <- st_iv_layout(f_acyc)
  set.seed(21)
  q <- stats::rnorm(probe$n_params, sd = 0.3)
  expect_false(isTRUE(all.equal(st_iv_lp(f_acyc, q), st_iv_lp(f_cyc, q))))

  f1_acyc <- st_iv_fixture(family = "poisson", temporal = "rw1")
  f1_cyc  <- st_iv_fixture(family = "poisson", temporal = "rw1",
                           temporal_cyclic = TRUE)
  expect_false(isTRUE(all.equal(st_iv_lp(f1_acyc, q), st_iv_lp(f1_cyc, q))))
})

test_that("diag(Q^-1) reproduces the log-posterior Hessian: cyclic RW1 and RW2", {
  skip_on_cran()
  # The assembly in st_type_iv_precision.h and the quadratic form in
  # tulpa_priors_st.h are two writings of one operator, and the numerical
  # Hessian reads the second. Scoring a cyclic fixture is what holds the
  # assembly's wrap rows to the density's wrap edges.
  expect_gmrf_matches_hessian(
    st_iv_fixture(family = "poisson", temporal = "rw1", temporal_cyclic = TRUE))
  expect_gmrf_matches_hessian(
    st_iv_fixture(family = "poisson", temporal = "rw2", temporal_cyclic = TRUE))
  expect_gmrf_matches_hessian(
    st_iv_fixture(family = "gaussian", temporal = "rw2", temporal_cyclic = TRUE))
})
