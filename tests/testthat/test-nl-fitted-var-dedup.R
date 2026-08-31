# test-nl-fitted-var-dedup.R
# Per-row predictive variance on the single-block nested-Laplace path.
#
# var(eta_i | theta_k) = a_i' H_k^{-1} a_i, where a_i is observation i's loading
# vector over the [beta | RE | blocks] latent layout. a_i is built from the row
# of X, the random-effect group, and the block-local index and weight the
# observation reads -- none of which move with the outer-grid cell. The only
# per-cell quantity is the block scalar d_fac_b(k), which multiplies every entry
# block b contributes at every row alike. Two observations agreeing on all of
# the above therefore carry the SAME loading vector at every cell and the same
# variance, and the driver solves one of them and hands the value to the rest.
#
# The fixture below has four distinct design rows -- (covariate, block unit)
# pairs -- repeated unequally and interleaved, so a class's members are never
# contiguous in the observation order. Two of the four share a covariate row and
# differ in the block unit; two share the block unit and differ in the covariate
# row, so a variance that ignored either half of the design would collapse a
# pair that must stay apart.

# Four design rows, indexed by `cls`:
#   1 = (x = -1, unit 1)   3 = (x =  0, unit 1)
#   2 = (x =  0, unit 2)   4 = (x =  1, unit 2)
# Classes 2 and 3 share the covariate row; classes 1 and 3 share the unit.
sim_fvdedup <- function(seed = 41L) {
  cls    <- c(1L, 3L, 2L, 4L, 1L, 2L, 3L, 1L, 4L, 2L, 1L, 3L)
  xv     <- c(-1, 0, 0, 1)[cls]
  unit   <- c(1L, 1L, 2L, 2L)[cls]
  N      <- length(cls)
  set.seed(seed)
  eta <- 0.4 - 0.7 * xv + c(0.5, -0.3)[unit]
  list(y = eta + stats::rnorm(N, sd = 0.3),
       X = cbind(1, xv), unit = unit, cls = cls, N = N, n_units = 2L)
}

# Unstructured Gaussian latent over the two units, integrated on a pinned sigma
# axis. auto_recenter = FALSE keeps the grid the one written here, so two fits
# of the same model are read at the same cells.
fit_fvdedup <- function(d, sigma_grid = c(0.5, 1.0, 2.0)) {
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$N), X = d$X,
    prior = list(list(type = "iid", obs_idx = d$unit,
                      n_units = d$n_units, sigma_grid = sigma_grid)),
    family = "gaussian", phi = 0.09,
    control = list(max_iter = 100L, tol = 1e-10, n_threads = 1L,
                   diagnose_k = FALSE, diagnose_skew = FALSE,
                   auto_recenter = FALSE)
  ))
}

test_that("fitted_eta_var is a finite positive [n_grid x N] matrix", {
  skip_on_cran()
  d  <- sim_fvdedup()
  f  <- fit_fvdedup(d)
  fv <- f$fitted_eta_var

  expect_s3_class(f, "tulpa_nested_laplace")
  expect_true(is.matrix(fv))
  expect_equal(dim(fv), c(nrow(f$theta_grid), d$N))
  expect_true(all(is.finite(fv)))
  expect_true(all(fv > 0))
})

test_that("observations with the same design row share fitted_eta_var exactly", {
  skip_on_cran()
  d  <- sim_fvdedup()
  f  <- fit_fvdedup(d)
  fv <- f$fitted_eta_var

  # Bit-for-bit, at every cell: the loading vectors are identical, so the
  # variances are one number, not two numbers that agree to a tolerance.
  for (c_id in sort(unique(d$cls))) {
    cols <- which(d$cls == c_id)
    ref  <- unname(fv[, cols[1]])
    for (j in cols[-1]) expect_identical(unname(fv[, j]), ref)
  }
})

test_that("fitted_eta_var separates rows differing in covariate or in block unit", {
  skip_on_cran()
  d  <- sim_fvdedup()
  f  <- fit_fvdedup(d)
  fv <- f$fitted_eta_var
  first <- vapply(1:4, function(c_id) which(d$cls == c_id)[1L], integer(1))

  # Same block unit, different covariate row.
  expect_true(all(abs(fv[, first[1]] - fv[, first[3]]) > 1e-9))
  # Same covariate row, different block unit. The two design cells carry
  # nearly the same curvature here, so what separates them is small; a key
  # that ignored the unit would merge the rows into one class and hand both
  # the same solved value, bit for bit.
  expect_false(identical(unname(fv[, first[2]]), unname(fv[, first[3]])))
  expect_true(all(abs(fv[, first[2]] - fv[, first[3]]) > 0))
})

test_that("fitted_eta_var follows the observation order the fit was given", {
  skip_on_cran()
  d    <- sim_fvdedup()
  f    <- fit_fvdedup(d)
  perm <- c(12L, 3L, 7L, 1L, 9L, 5L, 11L, 2L, 6L, 10L, 4L, 8L)

  dp        <- d
  dp$y      <- d$y[perm]
  dp$X      <- d$X[perm, , drop = FALSE]
  dp$unit   <- d$unit[perm]
  dp$cls    <- d$cls[perm]
  fp        <- fit_fvdedup(dp)

  # Same model, same pinned grid: only the row order moved.
  expect_equal(unname(fp$theta_grid), unname(f$theta_grid))
  # Column j of the permuted fit is column perm[j] of the original. A variance
  # attached to the wrong member of its class survives every within-fit check
  # above and fails here.
  expect_equal(unname(fp$fitted_eta_var),
               unname(f$fitted_eta_var[, perm, drop = FALSE]),
               tolerance = 1e-8)
})
