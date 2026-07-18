# Regression tests for the 0.0.91 audit-fix batch (#228-239).

# --- #231: SPDE Matern (range, sigma) <-> (kappa, tau) is nu-general ---------

test_that("spde kappa/tau <-> range/sigma round-trips for every nu", {
  for (nu in c(0.5, 1.0, 1.5, 2.0, 2.5)) {
    for (range in c(0.3, 1.0, 4.0)) {
      for (sigma in c(0.2, 1.0, 3.0)) {
        kt <- tulpa:::.spde_kappa_tau(range, sigma, nu)
        rs <- tulpa:::.spde_range_sigma(kt$kappa, kt$tau_spde, nu)
        expect_equal(rs$range, range, tolerance = 1e-10,
                     info = sprintf("nu=%g range=%g", nu, range))
        expect_equal(rs$sigma, sigma, tolerance = 1e-10,
                     info = sprintf("nu=%g sigma=%g", nu, sigma))
      }
    }
  }
})

test_that("the nu=1 conversion is byte-identical to the old hardcoded formula", {
  # Guards against a regression on the default GP/SPDE path: at nu = 1 the
  # nu-general normalizer must reduce exactly to 1 / (sqrt(4*pi) * kappa * sigma).
  for (range in c(0.3, 1.0, 4.0)) for (sigma in c(0.2, 1.0, 3.0)) {
    kt <- tulpa:::.spde_kappa_tau(range, sigma, 1.0)
    kappa_old <- sqrt(8 * 1) / range
    tau_old   <- 1 / (sqrt(4 * pi) * kappa_old * sigma)
    expect_identical(kt$kappa, kappa_old)
    expect_identical(kt$tau_spde, tau_old)
  }
})

test_that("the nu-general marginal variance matches the closed form at nu=2", {
  # sigma^2 = 1 / (4*pi*nu * kappa^(2nu) * tau^2), so recovering sigma from
  # (kappa, tau, nu) must invert it. A nu=1 hardcode would be wrong here.
  nu <- 2; range <- 1.5; sigma <- 0.8
  kt <- tulpa:::.spde_kappa_tau(range, sigma, nu)
  var_closed <- 1 / (4 * pi * nu * kt$kappa^(2 * nu) * kt$tau_spde^2)
  expect_equal(sqrt(var_closed), sigma, tolerance = 1e-10)
})

# --- #232: select_main_params drops multi-index latent names ----------------

test_that("select_main_params strips 1- and 2-index latent names, keeps scalars", {
  nm <- c("(Intercept)", "beta_x", "sigma", "u[12]", "w[3]",
          "factor[1,1]", "factor[200,3]")
  keep <- select_main_params(nm)
  expect_setequal(keep, c("(Intercept)", "beta_x", "sigma"))
  expect_false(any(grepl("\\[", keep)))
})

# --- #233: sim diagnostics error rather than fabricate on unresolved obs -----

test_that(".resolve_obs errors when the response cannot be found", {
  fake <- structure(list(y = NULL, .internal = list(fit_args = list(y = NULL))),
                    class = "tulpa_fit")
  expect_error(tulpa:::.resolve_obs(fake), "observed response")
  # and resolves an explicit argument / the stored response
  expect_equal(tulpa:::.resolve_obs(fake, observed = c(1, 0, 1)), c(1, 0, 1))
  fake$y <- c(2, 3)
  expect_equal(tulpa:::.resolve_obs(fake), c(2, 3))
})

# --- #238: spatial_gp / spatial_svc reject unsupported cov at construction --

test_that("spatial_gp / spatial_svc validate cov and nu at construction", {
  expect_error(spatial_gp(~ x + y, cov = "gaussian"))
  expect_error(spatial_gp(~ x + y, cov = "spherical"))
  expect_error(spatial_gp(~ x + y, cov = "matern", nu = 0.75), "1.5, 2.5")
  expect_error(spatial_svc(~ x + y, cov = "gaussian"))
  # supported specs still construct
  expect_s3_class(spatial_gp(~ x + y, cov = "exponential"), "tulpa_gp")
  expect_s3_class(spatial_gp(~ x + y, cov = "matern", nu = 2.5), "tulpa_gp")
})

# --- #229: cyclic RW2 is honored on the joint multi-block kernel -------------

test_that("a cyclic RW2 joint block differs from its acyclic counterpart", {
  skip_on_cran()
  set.seed(229)
  Tt <- 6L
  N  <- 90L
  t_idx <- sample.int(Tt, N, replace = TRUE)
  f_true <- sin(2 * pi * seq_len(Tt) / Tt)          # periodic signal
  y <- rnorm(N, 0.3 + f_true[t_idx], 0.5)

  arm <- list(
    y = y, n_trials = rep(1L, N), X = matrix(1, N, 1),
    temporal_idx = t_idx,
    re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
    family = "gaussian", phi = 1.0
  )
  mk_prior <- function(cyclic) list(list(
    type = "rw2", temporal_idx = list(t_idx), n_times = Tt,
    cyclic = cyclic, sigma_grid = c(0.3, 0.6, 1.0)
  ))

  fit_cyc <- tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = mk_prior(TRUE))
  fit_acyc <- tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = mk_prior(FALSE))

  # Before the fix the joint kernel discarded `cyclic` for rw2, so these were
  # bit-identical. The wrap-edge coupling + rank now make them differ.
  expect_gt(max(abs(fit_cyc$log_marginal - fit_acyc$log_marginal)), 1e-6)
})
