# An HSGP field reached through each nested-Laplace door: the single-block
# kernel, the single-arm multi-block driver (gcol33/tulpa#945) and the joint
# multi-block driver (gcol33/tulpa#946). All three build the same DENSE_BASIS
# block from one factory, so on one declared grid they are one model.

.hsgp_door_data <- function(seed = 4L, n = 120L) {
  set.seed(seed)
  coords <- cbind(stats::runif(n), stats::runif(n))
  x <- stats::rnorm(n)
  y <- stats::rpois(n, exp(0.2 + 0.4 * x + 0.6 * sin(3 * coords[, 1])))
  basis <- cpp_hsgp_basis_2d(coords, 6L, 1.5)
  g <- expand.grid(s = c(0.3, 0.8), l = c(0.2, 0.5))
  list(n = n, y = y, X = cbind(1, x), basis = basis, sigma2 = g$s,
       lengthscale = g$l)
}

.hsgp_door_ctl <- list(axis_refine = "none", prune = FALSE, diagnose_k = FALSE)

.hsgp_registry_block <- function(d) {
  list(type = "hsgp", phi_basis = d$basis$phi_basis,
       lambda_eig = d$basis$lambda_eig,
       sigma2_grid = d$sigma2, lengthscale_grid = d$lengthscale)
}


test_that("the multi-block driver fits an hsgp block as the single-block kernel does", {
  skip_on_cran()
  d <- .hsgp_door_data()
  fit_with <- function(prior) {
    tulpa_nested_laplace(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                         family = "poisson", prior = prior,
                         control = .hsgp_door_ctl)
  }
  blk <- .hsgp_registry_block(d)
  single <- fit_with(blk)
  multi  <- fit_with(list(blk))
  expect_equal(multi$log_marginal, single$log_marginal, tolerance = 1e-6)
  expect_equal(unname(coef(multi)), unname(coef(single)), tolerance = 1e-6)
  # The per-row predictor and its variance are read off the basis at each
  # cell, so they agree with the single-block kernel's cell by cell.
  expect_equal(multi$fitted_eta, single$fitted_eta, tolerance = 1e-6)
  expect_equal(multi$fitted_eta_var, single$fitted_eta_var, tolerance = 1e-6)
})


test_that("the joint door evaluates an hsgp block at its declared grid", {
  skip_on_cran()
  # The joint driver handed the grid columns to a factory that read them as
  # log values, so a cell labelled (0.3, 0.2) solved the field at
  # (exp(0.3), exp(0.2)) (gcol33/tulpa#946).
  d <- .hsgp_door_data()
  reg <- tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
    prior = .hsgp_registry_block(d), control = .hsgp_door_ctl)
  jnt <- tulpa_nested_laplace_joint(
    responses = list(a = list(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                              family = "poisson")),
    prior = list(list(type = "hsgp", m_total = ncol(d$basis$phi_basis),
                      phi = list(d$basis$phi_basis), n_obs_per_arm = d$n,
                      eigenvalues = d$basis$lambda_eig,
                      sigma2_grid = d$sigma2,
                      lengthscale_grid = d$lengthscale)),
    control = .hsgp_door_ctl)
  expect_equal(unname(as.matrix(jnt$theta_grid)),
               unname(as.matrix(reg$theta_grid)))
  expect_equal(jnt$log_marginal, reg$log_marginal, tolerance = 1e-6)
  expect_equal(unname(jnt$modes[, 1:2]), unname(reg$modes[, 1:2]),
               tolerance = 1e-6)
})


test_that("an HSGP field + (1 | g) through tulpa() equals a direct multi-block call", {
  skip_on_cran()
  set.seed(9)
  n <- 100L
  dat <- data.frame(lon = stats::runif(n), lat = stats::runif(n),
                    x = stats::rnorm(n), g = rep(1:5, 20))
  dat$y <- stats::rpois(n, exp(0.3 + 0.5 * dat$x + 0.4 * sin(3 * dat$lon)))
  sp <- spatial_gp(~ lon + lat, approx = "hsgp")
  via <- suppressWarnings(tulpa(y ~ x + (1 | g), data = dat,
                                family = "poisson", spatial = sp,
                                mode = "nested_laplace"))
  re_blk <- list(type = "iid", obs_idx = as.integer(dat$g), n_units = 5L)
  direct <- suppressWarnings(tulpa_nested_laplace(
    y = dat$y, n_trials = rep(1L, n), X = stats::model.matrix(~ x, dat),
    family = "poisson",
    prior = list(tulpa:::.spatial_spec_to_nl_prior(
                   tulpa:::validate_hsgp(sp, dat)), re_blk)))
  expect_equal(via$backend, "nested_laplace")
  expect_equal(via$log_marginal, direct$log_marginal, tolerance = 1e-8)
  expect_equal(unname(coef(via)), unname(coef(direct)), tolerance = 1e-8)
})
