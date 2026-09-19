# SPDE path composability (#158f): a single iid random-intercept `(1 | g)` can
# ride alongside the Matern field, conditioned on sigma_re, through both
# fit_spde() and the tulpa() front door. beta_prior and latent() stay
# unsupported (kernel has no fixed-effect-prior arg), and random slopes / a
# fractional field with an RE term are rejected.

make_spde_re <- function(n = 250L, G = 12L, seed = 7L) {
  set.seed(seed)
  coords <- cbind(runif(n), runif(n))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.4),
                              cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                              prior_range = c(0.3, 0.5),
                              prior_sigma = c(0.6, 0.05))
  grp   <- rep(seq_len(G), length.out = n)
  b_g   <- rnorm(G, 0, 0.7)
  w     <- as.numeric(rnorm(spec$n_mesh, 0, 0.6)); w <- w - mean(w)
  x     <- rnorm(n)
  eta   <- 0.5 + 0.8 * x + as.numeric(spec$A %*% w) + b_g[grp]
  list(y = rpois(n, exp(eta)), X = cbind(1, x), x = x, spec = spec,
       grp = grp, G = G, coords = coords)
}

test_that("fit_spde() accepts a single random intercept and recovers beta", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  d <- make_spde_re()
  fit <- suppressWarnings(fit_spde(
    d$y, d$X, d$spec, family = "poisson",
    re_idx = d$grp, n_re_groups = d$G, sigma_re = 0.7,
    control = list(method = "ccd")))
  expect_false(is.null(fit$nested))
  expect_true(all(is.finite(fit$beta)))
  # Slope near the truth 0.8 (a real recovery bound, not a sanity check).
  expect_lt(abs(fit$beta[2] - 0.8), 0.25)
})

test_that("fit_spde() validates the RE index and rejects a fractional field", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  d <- make_spde_re(n = 80L, G = 5L)
  expect_error(
    fit_spde(d$y, d$X, d$spec, family = "poisson",
             re_idx = d$grp[-1], n_re_groups = d$G),
    "length\\(re_idx\\)"
  )
  expect_error(
    fit_spde(d$y, d$X, d$spec, family = "poisson",
             re_idx = rep(99L, length(d$y)), n_re_groups = d$G),
    "\\[1, n_re_groups"
  )
})

test_that("tulpa(mode = 'exact') routes an SPDE field to NUTS (Tier 1)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  d  <- make_spde_re(n = 150L, G = 5L)
  df <- data.frame(y = d$y, x = d$x)
  fit <- suppressWarnings(tulpa(
    y ~ x, data = df, family = "poisson", spatial = d$spec, mode = "exact",
    control = list(n_iter = 300L, n_warmup = 150L, seed = 5L)))
  expect_equal(fit$inference_tier, 1L)
  expect_identical(fit$draws_kind, "chain")
  expect_false(is.null(fit$draws))
  # The slope is recovered by the exact sampler. The column is read by the
  # RESOLVED fixed-effect name the fit carries -- the same one `param_names`,
  # `coef()` and `mode = "hmc"` give, and what
  # `.finalize_fit(fixed_names = colnames(X))` produces. Asking for `beta[2]`
  # errored on the subscript, so this comparison never ran (gcol33/tulpa#846).
  expect_true("x" %in% colnames(fit$draws))
  expect_lt(abs(mean(fit$draws[, "x"]) - 0.8), 0.3)
})

test_that("tulpa() routes (1 | g) + an SPDE field to the nested driver", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  # This used to assert backend "spde". fit_spde()'s grid has no RE-SD axis and
  # conditions on a scalar `sigma_re`, so on the path whose point is
  # integrating the hyperparameters the RE was the one variance component never
  # estimated; the field now goes to the generic nested driver as an `spde`
  # block beside the RE's own `iid` block (gcol33/tulpa#817). Recovery of the
  # slope is asserted unchanged, and the axis itself in
  # test-spde-re-integrated.R.
  d  <- make_spde_re()
  df <- data.frame(y = d$y, x = d$x, g = factor(d$grp))
  sp <- spatial_spde(coords = d$coords, nu = 1,
                     prior_range = c(0.3, 0.5), prior_sigma = c(0.6, 0.05))
  fit <- suppressWarnings(
    tulpa(y ~ x + (1 | g), data = df, family = "poisson",
          spatial = sp, sigma_re = 0.7))
  expect_identical(fit$backend, "nested_laplace")
  expect_lt(abs(coef(fit)[["x"]] - 0.8), 0.25)

  # beta_prior stays unsupported on either SPDE route.
  expect_error(
    tulpa(y ~ x + (1 | g), data = df, family = "poisson", spatial = sp,
          beta_prior = list(mean = 0, sd = 5)),
    "beta_prior"
  )
})
