# The fractional rSPDE's variance normalization declines a singular Q
# (gcol33/tulpa#845).
#
# `.spde_mean_marginal_var()` estimates `tr(Pr Q^-1 Pr') / n` by probing
# against a Cholesky of the rational precision `Q`. The rational construction
# makes `cond(Q)` ~ 1e13 by design, and far outside the range a fit integrates
# it is not merely ill-conditioned but numerically SINGULAR -- `rcond(Q)` is
# 2.4e-17 at range 40 on a domain of extent 13.8. CHOLMOD reported that as a
# "not positive definite" warning and then recovered on its own terms, so a
# field scale was computed from a factorization that had not completed and the
# only sign of it was a raw library message with no coordinate attached.
#
# Where it came from, measured: every warning on the issue's fixture was raised
# under `.spde_mean_marginal_var()` (call stack captured, not inferred) from
# the MODE SEARCH at its own box corner -- `lower/upper = init +/- log(100)`,
# so `range = 100 * range_init ~ 275.6`, twenty times the coordinate extent.
# The CCD grid the fit then laid at the mode topped out at range 1.53 and
# carried zero weight above 100, which is why the coefficients do not move.

.svn_sim <- function(n = 120L, seed = 2L) {
  set.seed(seed)
  L <- cbind(lon = runif(n, 0, 10), lat = runif(n, 0, 10))
  w <- as.numeric(t(chol(0.64 * exp(-as.matrix(dist(L)) / 3) +
                           diag(1e-8, n))) %*% rnorm(n))
  d <- data.frame(L, x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x + w))
  d
}


test_that("the variance normalization reports a Q it could not factor", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  d <- .svn_sim()
  spec <- spatial_spde(~ lon + lat, data = d, nu = 1.5)

  fem <- tulpa:::.spde_fem_matrices(spec)
  C0 <- as.numeric(spec$C0_diag)
  keep <- which(C0 > 1e-12)
  Gk <- as(fem$G[keep, keep, drop = FALSE], "CsparseMatrix")
  C0k <- C0[keep]

  at <- function(range) {
    asm <- tulpa:::.spde_rational_assemble(
      C0 = C0k, G = Gk, kappa = sqrt(8 * 1.5) / range, tau = 1, nu = 1.5,
      order = 2L, d = 2)
    tulpa:::.spde_mean_marginal_var(asm$Q, asm$Pr, C0k)
  }

  # A coordinate the fit actually integrates: a value, and no complaint.
  ok <- expect_no_warning(at(1.0))
  expect_false(ok$singular)
  expect_true(is.finite(ok$value) && ok$value > 0)

  # The mode search's box corner, 100x the initial range: declined, and the
  # CHOLMOD warning is consumed here rather than reaching the caller.
  bad <- expect_no_warning(at(275.625))
  expect_true(bad$singular)
})


test_that("a coordinate with no field scale takes the -Inf zero-weight path", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  d <- .svn_sim()
  spec <- spatial_spde(~ lon + lat, data = d, nu = 1.5)
  args <- list(spatial = spec, y = d$y, X = cbind(1, d$x), family = "poisson",
               phi = 1, n_trials = rep(1L, nrow(d)), order = 2L,
               max_iter = 50L, tol = 1e-8, n_threads = 1L, offset = NULL)

  good <- do.call(tulpa:::.spde_nested_logmarginal_at,
                  c(args, list(range = 1.0, sigma = 0.6)))
  expect_true(is.finite(good$log_marginal))
  expect_false(isTRUE(good$asm$var_norm_singular))

  # Same convention a failed inner fit already takes: -Inf, so no grid can
  # weight it, by a stated decision rather than by CHOLMOD's own recovery.
  gone <- expect_no_warning(
    do.call(tulpa:::.spde_nested_logmarginal_at,
            c(args, list(range = 275.625, sigma = 0.03))))
  expect_identical(gone$log_marginal, -Inf)
  expect_identical(gone$n_iter, 0L)
  expect_false(gone$converged)
  expect_true(gone$asm$var_norm_singular)
})


test_that("a fractional fit runs clean and integrates no declined coordinate", {
  skip_on_cran()
  skip_if_not_installed("fmesher")
  d <- .svn_sim()
  nw <- 0L
  fit <- withCallingHandlers(
    tulpa(y ~ x, data = d, family = "poisson",
          spatial = spatial_spde(~ lon + lat, data = d, nu = 1.5)),
    warning = function(w) { nw <<- nw + 1L; invokeRestart("muffleWarning") })

  expect_identical(nw, 0L)
  expect_identical(fit$backend, "spde")
  # Unchanged by the decline: the coefficients are what they were while the
  # warnings were being raised, because no warned coordinate was ever in the
  # integration.
  expect_equal(unname(coef(fit)[1:2]), c(0.675, 0.557), tolerance = 0.01)

  # And the grid it integrates lives well inside the domain: the extent is
  # 13.8, the largest node 1.53.
  ext <- sqrt(sum(apply(as.matrix(d[, c("lon", "lat")]), 2,
                        function(z) diff(range(z)))^2))
  expect_lt(max(fit$nested$range_grid), ext)
  expect_true(all(is.finite(fit$log_marginal)))
})
