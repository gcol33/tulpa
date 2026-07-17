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

test_that("tulpa() routes (1 | g) + an SPDE field to the spde backend", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  d  <- make_spde_re()
  df <- data.frame(y = d$y, x = d$x, g = factor(d$grp))
  sp <- spatial_spde(coords = d$coords, nu = 1,
                     prior_range = c(0.3, 0.5), prior_sigma = c(0.6, 0.05))
  fit <- suppressWarnings(
    tulpa(y ~ x + (1 | g), data = df, family = "poisson",
          spatial = sp, sigma_re = 0.7))
  expect_identical(fit$backend, "spde")
  expect_lt(abs(coef(fit)[["x"]] - 0.8), 0.25)

  # beta_prior stays unsupported on the SPDE path.
  expect_error(
    tulpa(y ~ x + (1 | g), data = df, family = "poisson", spatial = sp,
          beta_prior = list(mean = 0, sd = 5)),
    "beta_prior"
  )
})
