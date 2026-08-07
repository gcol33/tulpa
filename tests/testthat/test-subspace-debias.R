# Structural and correctness gates for the subspace debias (gcol33/tulpa#304).
# The statistical gate -- does correcting only the flagged directions recover
# the exact conditional marginal, and does the coupling closure have to run --
# is test-subspace-debias-recovery.R.

# ---- a small-group Bernoulli random-intercept fixture ----------------------
sd_fixture <- function(seed = 11L, G = 12L, per = 4L, b = c(-2.5, 0.8),
                       su = 0.7) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  u <- rnorm(G, 0, su)
  y <- rbinom(n, 1L, plogis(b[1] + b[2] * x + u[grp]))
  list(y = y, X = cbind(1, x), n_trials = rep(1L, n), su = su,
       re = list(list(idx = grp, n_groups = G, n_coefs = 1L, sigma = su)),
       rt = list(idx = grp, n_groups = G, n_coefs = 1L))
}

sd_laplace <- function(d, ...) {
  tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "binomial", ...)
}


test_that("the band table resolves gamma_3 and the inner k-hat per probed index", {
  d <- sd_fixture()
  f <- sd_laplace(d, compute_skew = TRUE, skew_idx = 1:2)
  b <- tulpa:::.subspace_bands(f)

  expect_identical(b$idx, 1:2)
  expect_true(all(is.finite(b$gamma3)))
  # The band is the WORSE of the two inner scores, so it can never be better
  # than what gamma_3 alone says.
  g_band <- vapply(b$gamma3, tulpa:::.tulpa_gamma3_band, character(1))
  expect_true(all(tulpa:::.subspace_band_rank(b$band) >=
                    tulpa:::.subspace_band_rank(g_band)))
  # This fixture is the known-skewed one, so the intercept is flagged.
  expect_gt(abs(b$gamma3[1]), tulpa:::.nl_diag("gamma3_ok"))
  expect_identical(b$band[1], "unreliable")
})


test_that("an index whose cubic term declines is banded by its inner k-hat", {
  # The #303 dependency the selector is built on: a coordinate gamma_3 cannot
  # score (a coupled likelihood ships no per-observation third derivative) must
  # still be banded, from the importance curve, rather than skipped -- otherwise
  # the selector's blind spot is exactly the model class that needs it most.
  d <- sd_fixture()
  f <- sd_laplace(d, compute_skew = TRUE, skew_idx = 1:2)
  f$inner_skew <- c(NaN, NaN)
  b <- tulpa:::.subspace_bands(f)

  expect_true(all(is.nan(b$gamma3)))
  expect_true(all(is.finite(b$pareto_k)))
  # The k-hat is banded only where the importance correction is material, so
  # every band present is one the k-hat carried on its own.
  material <- is.finite(b$rel_ess) &
    b$rel_ess < tulpa:::.nl_diag("inner_k_material_ess")
  expect_true(all(is.na(b$band[!material])))
  expect_identical(b$band[material],
                   vapply(b$pareto_k[material], tulpa:::.tulpa_khat_band,
                          character(1)))
  expect_true(any(material))
})


test_that("a fit with no inner-layer material bands nothing", {
  d <- sd_fixture()
  f <- sd_laplace(d)                      # compute_skew not requested
  b <- tulpa:::.subspace_bands(f)
  expect_identical(nrow(b), 0L)
  cfg <- tulpa:::.subspace_debias_config(TRUE)
  expect_length(tulpa:::.subspace_select(f, cfg)$idx, 0L)
})


test_that("the sampler reproduces the exact conditional when the target IS Gaussian", {
  # A gaussian response makes the joint log density exactly quadratic in the
  # latent field, so the Gaussian-conditional-mean surface carries exactly
  # N(0, Sigma_SS) and the Metropolis draws must reproduce it. This is the
  # sharpest available check that the surface, its scaling and the recorded
  # coordinates are the ones the header claims.
  set.seed(3)
  d <- sd_fixture()
  eta <- as.numeric(d$X %*% c(-0.4, 0.8))
  d$y <- eta + rnorm(length(eta), 0, 1)
  set.seed(7)
  f <- tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "gaussian",
                     phi = 1, debias = list(idx = 1:2, n_iter = 40000L,
                                            warmup = 3000L))
  expect_identical(as.integer(f$debias_idx), 1:2)
  expect_true(f$debias_accept > 0.1 && f$debias_accept < 0.6)
  emp <- stats::cov(f$debias_draws)
  # 40k correlated draws: a few percent of Monte-Carlo error on the scale, and
  # the mean sits at the mode.
  expect_equal(sqrt(diag(emp)), sqrt(diag(f$debias_sigma_ss)), tolerance = 0.06)
  expect_lt(max(abs(colMeans(f$debias_draws))),
            0.1 * min(sqrt(diag(f$debias_sigma_ss))))
})


test_that("sigma_ss is the inner Laplace's own marginal covariance of x_S", {
  d <- sd_fixture()
  f <- sd_laplace(d, return_joint_hessian = TRUE,
                  debias = list(idx = c(1L, 3L), n_iter = 200L, warmup = 100L))
  Sig <- solve(as.matrix(f$H_joint))
  expect_equal(f$debias_sigma_ss, Sig[c(1L, 3L), c(1L, 3L)],
               tolerance = 1e-6, ignore_attr = TRUE)
})


test_that("an empty or absent index set leaves the solve bit-for-bit unchanged", {
  d <- sd_fixture()
  set.seed(1); a <- sd_laplace(d)
  set.seed(1); b <- sd_laplace(d, debias = list(idx = integer(0)))
  expect_identical(a, b)

  # ... and the RNG stream is untouched, so anything drawn afterwards matches.
  set.seed(1); invisible(sd_laplace(d)); ra <- stats::rnorm(3)
  set.seed(1); invisible(sd_laplace(d, debias = list(idx = integer(0))))
  rb <- stats::rnorm(3)
  expect_identical(ra, rb)
})


test_that("the front door records S and leaves an empty S bit-for-bit identical", {
  d <- sd_fixture()
  f0 <- tulpa_re_cov_nested(d$y, d$n_trials, d$X, d$rt, family = "binomial",
                            control = list(seed = 1L))
  f1 <- tulpa_re_cov_nested(d$y, d$n_trials, d$X, d$rt, family = "binomial",
                            control = list(seed = 1L,
                                           subspace_debias = list(idx = integer(0))))
  expect_identical(f0$draws, f1$draws)
  expect_identical(f0$posterior, f1$posterior)
  expect_length(f1$subspace_debias$idx, 0L)

  f2 <- tulpa_re_cov_nested(d$y, d$n_trials, d$X, d$rt, family = "binomial",
                            control = list(seed = 1L, subspace_debias = TRUE))
  sd2 <- f2$subspace_debias
  expect_true(length(sd2$idx) > 0L)
  expect_identical(sd2$selected_by, "band")
  expect_identical(sd2$band_floor, tulpa:::.nl_diag("debias_select_band"))
  expect_true(all(sd2$idx %in% sd2$bands$idx))
  # Every selected index was banded at or above the floor it was selected at.
  fl <- tulpa:::.SUBSPACE_BAND_RANK[[sd2$band_floor]]
  rk <- tulpa:::.subspace_band_rank(sd2$bands$band[match(sd2$idx, sd2$bands$idx)])
  expect_true(all(rk >= fl))
  # The corrected coefficient's interval moved; the fit is not silently inert.
  expect_false(isTRUE(all.equal(confint(f0), confint(f2))))
})


test_that("the closure grows S by coupling strength and never duplicates a member", {
  d <- sd_fixture()
  f <- sd_laplace(d, return_joint_hessian = TRUE)
  H <- as.matrix(f$H_joint)
  n_x <- nrow(H)

  # An unreachable threshold adds nothing; a zero threshold takes everything
  # exactly once.
  expect_identical(tulpa:::.subspace_closure(H, 1:2, threshold = 10)$idx, 1:2)
  all_in <- tulpa:::.subspace_closure(H, 1:2, threshold = 0)$idx
  expect_identical(all_in, seq_len(n_x))
  expect_false(anyDuplicated(all_in) > 0L)

  # Lowering the threshold is monotone in the selected set.
  a <- tulpa:::.subspace_closure(H, 1:2, threshold = 0.2)$idx
  b <- tulpa:::.subspace_closure(H, 1:2, threshold = 0.1)$idx
  expect_true(all(a %in% b))

  # The cap keeps the strongest couplings.
  cap <- tulpa:::.subspace_closure(H, 1:2, threshold = 0, max_add = 3L)
  expect_length(cap$added, 3L)
  dd <- sqrt(abs(diag(H)))
  strength <- apply(abs(H[1:2, ] / outer(dd[1:2], dd)), 2L, max)
  strength[1:2] <- -Inf
  expect_setequal(cap$added, order(strength, decreasing = TRUE)[1:3])
})


test_that("the control knob validates its settings", {
  expect_null(tulpa:::.subspace_debias_config(NULL))
  expect_null(tulpa:::.subspace_debias_config(FALSE))
  expect_identical(tulpa:::.subspace_debias_config(TRUE)$band,
                   tulpa:::.nl_diag("debias_select_band"))
  expect_error(tulpa:::.subspace_debias_config(list(nonsense = 1)),
               "Unknown .*subspace_debias.* setting")
  expect_error(tulpa:::.subspace_debias_config(list(band = "great")),
               "must be one of")
  expect_error(tulpa:::.subspace_debias_config("on"),
               "must be TRUE, FALSE, or a list")
  d <- sd_fixture()
  expect_error(
    tulpa_re_cov_nested(d$y, d$n_trials, d$X, d$rt, family = "binomial",
                        control = list(subspace = TRUE)),
    "Unknown control knob")
})


test_that("the spatial Laplace path refuses the correction rather than ignoring it", {
  d <- sd_fixture()
  expect_error(
    tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "binomial",
                  spatial = list(type = "gp"), debias = list(idx = 1L)),
    "non-spatial")
  expect_error(
    tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "binomial",
                  spatial = list(type = "gp"), compute_skew = TRUE),
    "non-spatial")
})
