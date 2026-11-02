# The fractional-nu rSPDE latent carries its variance normalization
# (`.spde_assemble_at()`), so it sits on the field's scale and the inner Newton's
# absolute step test is met where the score is.

skip_on_cran()

.frac_fixture <- function(family = "poisson") {
  set.seed(4)
  n <- 200
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.1, 0.3), nu = 0.5)
  eta <- 1 + 0.5 * d$x + sin(3 * d$lon)
  d$y <- if (family == "poisson") rpois(n, exp(eta)) else eta + rnorm(n, 0, 0.5)
  list(d = d, sp = sp, X = model.matrix(~ x, d), n = n)
}

test_that("the fractional inner Newton converges on the rescaled latent", {
  skip_on_cran()
  fx <- .frac_fixture()
  asm <- tulpa:::.spde_assemble_at(fx$sp, 0.3, 0.5)
  Qg <- as(asm$Q, "generalMatrix")
  f <- tulpa:::cpp_laplace_fit_spde_precomputed(
    y = fx$d$y, n_trials = rep(1L, fx$n), X = fx$X, re_idx = rep(0, fx$n),
    n_re_groups = 0L, sigma_re = 1, n_obs = fx$n, n_mesh = length(asm$keep),
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = asm$A_eff@x, Aeff_i = asm$A_eff@i, Aeff_p = asm$A_eff@p,
    family = "poisson", phi = 1, max_iter = 100L, tol = 1e-6, n_threads = 1L)
  expect_true(f$converged)
  expect_lt(f$n_iter, 20L)
  expect_lt(f$score_max, 1e-8)
  # The field's marginal variance is sigma^2 = 0.25 whatever the latent scale.
  mv <- tulpa:::.spde_mean_marginal_var(asm$Q, asm$Pr, rep(1, length(asm$keep)))
  expect_equal(mv$value, 0.25, tolerance = 1e-8)
})

test_that("the det-lemma marginal is invariant to the latent's scale", {
  skip_on_cran()
  fx <- .frac_fixture()
  asm <- tulpa:::.spde_assemble_at(fx$sp, 0.3, 0.5)
  one <- tulpa:::.spde_nested_logmarginal_at(
    spatial = fx$sp, range = 0.3, sigma = 0.5, y = fx$d$y, X = fx$X,
    family = "poisson", phi = 1, n_trials = rep(1L, fx$n), order = 2L,
    max_iter = 100L, tol = 1e-6, n_threads = 1L, offset = NULL)
  Qg <- as(asm$Q, "generalMatrix")
  f <- tulpa:::cpp_laplace_fit_spde_precomputed(
    y = fx$d$y, n_trials = rep(1L, fx$n), X = fx$X, re_idx = rep(0, fx$n),
    n_re_groups = 0L, sigma_re = 1, n_obs = fx$n, n_mesh = length(asm$keep),
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = asm$A_eff@x, Aeff_i = asm$A_eff@i, Aeff_p = asm$A_eff@p,
    family = "poisson", phi = 1, max_iter = 100L, tol = 1e-6, n_threads = 1L)
  lm_at <- function(c) {
    tulpa:::cpp_spde_fractional_logmarginal(
      y = as.numeric(fx$d$y), X = fx$X, A_eff = asm$A_eff * c, Pl = asm$Pl * c,
      C0sub = as.numeric(fx$sp$C0_diag)[asm$keep], family = "poisson", phi = 1,
      beta_hat = f$mode[1:2], x_hat = f$mode[-(1:2)] / c,
      n_trials = rep(1L, fx$n))
  }
  expect_equal(lm_at(1), one$log_marginal, tolerance = 1e-8)
  expect_equal(lm_at(1 / asm$var_scale), lm_at(1), tolerance = 1e-6)
})
