.basis_scatter_reference <- function(A, B, w, g) {
  n <- nrow(A)
  Z <- cbind(1, seq_len(n) / n, diag(2)[rep_len(1:2, n), ], -0.7 * A, 1.3 * B)
  H <- crossprod(Z, w * Z)
  H[1:4, 1:4] <- 0
  gradient <- as.numeric(crossprod(Z, g))
  gradient[1:4] <- 0
  list(H = H, gradient = gradient)
}

test_that("batched basis products preserve signed observation curvature", {
  skip_on_cran()
  load_cpp_fixture("dense_basis_cross")
  set.seed(72)
  n <- 7L
  A <- matrix(rnorm(n * 3), n, 3)
  B <- matrix(rnorm(n * 2), n, 2)
  for (w in list(seq(0.2, 1, length.out = n), c(0.2, -0.5, 0.4, 1, -1.1, 0.7, 0))) {
    got <- cpp_test_dense_basis_scatter(A, B, w)
    target <- .basis_scatter_reference(A, B, w, 1 + 0.1 * (seq_len(n) - 1))
    expect_equal(got$H, target$H, tolerance = 1e-12)
    expect_equal(got$gradient, target$gradient, tolerance = 1e-12)
  }
})

test_that("beta basis products use the density's exact observed curvature", {
  skip_on_cran()
  load_cpp_fixture("dense_basis_cross")
  set.seed(73)
  n <- 7L
  A <- matrix(rnorm(n * 3), n, 3); B <- matrix(rnorm(n * 2), n, 2)
  y <- c(rep(0.99, 6), 0.001); eta <- rep(2, n); phi <- 5
  mu <- plogis(eta); u <- mu * (1 - mu)
  score_mu <- phi * (qlogis(y) - digamma(mu * phi) + digamma((1 - mu) * phi))
  hess_mu <- -phi^2 * (trigamma(mu * phi) + trigamma((1 - mu) * phi))
  w <- -(hess_mu * u^2 + score_mu * u * (1 - 2 * mu))
  expect_true(any(w < 0))
  target <- .basis_scatter_reference(A, B, w, score_mu * u)
  got <- cpp_test_dense_basis_beta(A, B, y, eta, phi)
  expect_equal(got$H, target$H, tolerance = 1e-10)
  expect_equal(got$gradient, target$gradient, tolerance = 1e-10)
})

test_that("two HSGP blocks reproduce the exact Gaussian basis posterior", {
  skip_on_cran()
  set.seed(91)
  n <- 25L
  xy <- cbind(runif(n), runif(n))
  X <- cbind(1, rnorm(n))
  y <- rnorm(n)
  bases <- lapply(c(4L, 5L), function(m)
    tulpa:::cpp_hsgp_basis_2d(xy, m, 1.5))
  sig2 <- c(0.6, 0.9); ell <- c(0.2, 0.7)
  blocks <- lapply(seq_along(bases), function(b) list(type = "hsgp",
    m_total = ncol(bases[[b]]$phi_basis), n_obs_per_arm = n,
    phi = list(bases[[b]]$phi_basis), eigenvalues = bases[[b]]$lambda_eig,
    cov = "matern", nu = 1.5, sigma2_grid = sig2[b],
    lengthscale_grid = ell[b]))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, X = X, family = "gaussian", phi = 0.4)),
    prior = blocks,
    control = list(progress = FALSE, axis_refine = "none", force_sparse = TRUE)))
  A <- do.call(cbind, lapply(seq_along(bases), function(b) {
    w <- bases[[b]]$lambda_eig
    s <- 6 * pi * 3^1.5 / ell[b]^3 * (3 / ell[b]^2 + w)^(-2.5)
    sweep(bases[[b]]$phi_basis, 2, sqrt(sig2[b] * s), `*`)
  }))
  Z <- cbind(X, A)
  V <- solve(crossprod(Z) / 0.4 + diag(c(rep(1 / 2.5^2, 2), rep(1, ncol(A)))))
  mu <- as.numeric(V %*% crossprod(Z, y) / 0.4)
  eta <- posterior_linpred(fit, summary = TRUE)
  expect_equal(eta$estimate, as.numeric(Z %*% mu), tolerance = 1e-6)
  expect_equal(eta$sd^2, rowSums((Z %*% V) * Z), tolerance = 1e-6)
})
