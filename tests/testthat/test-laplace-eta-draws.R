# Linear-predictor draws of a Laplace / EB fit (gcol33/tulpa#871).
#
# The fixed effects were drawn from N(coef, vcov) -- a MARGINAL spread that
# includes the intercept's aliasing with the random effects -- while the random
# effects were held at their mode, which dropped the negative correlation that
# cancels it. Every draw then shifted the whole linear predictor: p_waic came
# out ~2.5x the HMC value on the same model. The random effects are now drawn
# from their Gaussian conditional given each fixed-effect draw, under the joint
# Laplace precision `H_latent`, which paired with N(beta_hat, Schur^-1) is the
# joint Gaussian draw of [beta | b].

.eta_fixture <- function() {
  set.seed(1)
  n <- 200; G <- 10
  g <- factor(sample(letters[1:G], n, TRUE)); x <- rnorm(n)
  b0 <- rnorm(G, 0, 0.7)
  d <- data.frame(g, x)
  d$yp <- rpois(n, exp(0.3 + 0.8 * x + b0[g]))
  d
}

test_that("a laplace fit's eta draws have the joint Laplace covariance", {
  skip_on_cran()
  d <- .eta_fixture()
  l <- tulpa(yp ~ x + (1 | g), d, family = "poisson", mode = "laplace",
             sigma_re = 0.7)
  expect_true(inherits(l$H_latent, "Matrix"))
  expect_null(l$H_joint)
  expect_equal(dim(l$H_latent), c(12L, 12L))

  # Exact target: A H^-1 A' with A the [X | Z] design of eta.
  A <- cbind(l$model_matrix, as.matrix(Matrix::t(.tulpa_re_map(l))))
  V_eta <- A %*% solve(as.matrix(l$H_latent)) %*% t(A)
  eta_hat <- as.numeric(A %*% l$mode)

  set.seed(3)
  E <- .tulpa_eta_draws(l, ndraws = 8000L)
  expect_equal(dim(E), c(8000L, 200L))
  # Monte Carlo mean and variance of eta_i against the joint Gaussian.
  expect_lt(max(abs(colMeans(E) - eta_hat) / sqrt(diag(V_eta))), 0.1)
  expect_lt(max(abs(apply(E, 2, var) / diag(V_eta) - 1)), 0.12)

  # The marginal fixed-effect spread is still vcov(): only the RE conditional
  # moved.
  bv <- solve(as.matrix(l$H_latent))[1:2, 1:2]
  expect_equal(unname(vcov(l)), unname(bv), tolerance = 1e-6)
})

test_that("laplace WAIC no longer over-counts parameters against HMC's", {
  skip_on_cran()
  skip_if_not_installed("loo")
  d <- .eta_fixture()
  l <- tulpa(yp ~ x + (1 | g), d, family = "poisson", mode = "laplace",
             sigma_re = 0.7)
  w <- suppressWarnings(loo::waic(pointwise_loglik(l)))$estimates
  # Before the fix: p_waic 28.7, elpd -331.9. A manual joint (beta, b) Laplace
  # draw gives p_waic 8.95, elpd -304.0; HMC reads 11.3 / -313.6.
  expect_lt(w["p_waic", 1], 15)
  expect_gt(w["elpd_waic", 1], -315)
})

test_that("an EB fit carries H_latent and draws its random effects jointly", {
  skip_on_cran()
  skip_if_not_installed("loo")
  d <- .eta_fixture()
  e <- tulpa(yp ~ x + (1 | g), d, family = "poisson", mode = "eb")
  expect_true(inherits(e$H_latent, "Matrix"))
  expect_null(e$H_joint)
  w <- suppressWarnings(loo::loo(pointwise_loglik(e)))$estimates
  # Before the fix: p_loo 19.8.
  expect_lt(w["p_loo", 1], 15)
})

test_that("a fit without a joint precision keeps the random effects at the mode", {
  skip_on_cran()
  d <- .eta_fixture()
  l <- tulpa(yp ~ x + (1 | g), d, family = "poisson", mode = "laplace",
             sigma_re = 0.7)
  l$H_latent <- NULL
  set.seed(4)
  E <- .tulpa_eta_draws(l, ndraws = 50L)
  M <- .tulpa_re_map(l)
  b_hat <- l$mode[-(1:2)]
  beta <- E - matrix(as.numeric(Matrix::crossprod(M, b_hat)), 50, 200,
                     byrow = TRUE)
  # What remains is X beta exactly: rank 2 across the draws.
  expect_equal(qr(beta)$rank, 2L)
})
