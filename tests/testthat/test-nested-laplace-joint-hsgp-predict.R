# An `hsgp` block given by its coordinates (a basis builder) on the joint door,
# predicted at new coordinates.
#
# At one outer-grid cell and a gaussian response with its variance pinned, the
# latent posterior is exactly Gaussian, so the prediction at a new row has a
# closed form: with the design A (fixed effects, then the basis times
# sqrt(S(lambda)) times the varying-coefficient covariate), prior precision
# P = diag(tau_beta, ..., 1, ..., 1) and residual variance phi,
#   x | y ~ N(m, H^-1),  H = A'A / phi + P,  m = H^-1 A'y / phi,
# and eta_new = a' x ~ N(a' m, a' H^-1 a). The basis below is written out in R
# from the Laplacian eigenfunctions on [-L, L], independently of the engine's.

.jp_basis <- function(xy, m, domain) {
  M <- m * m
  phi <- matrix(0, nrow(xy), M)
  lam <- numeric(M)
  xs <- xy[, 1] - domain[1]; ys <- xy[, 2] - domain[2]
  L1 <- domain[3]; L2 <- domain[4]
  for (j1 in seq_len(m)) for (j2 in seq_len(m)) {
    j <- (j1 - 1L) * m + j2
    lam[j] <- (pi * j1 / (2 * L1))^2 + (pi * j2 / (2 * L2))^2
    phi[, j] <- sin(pi * j1 * (xs + L1) / (2 * L1)) / sqrt(L1) *
      sin(pi * j2 * (ys + L2) / (2 * L2)) / sqrt(L2)
  }
  list(phi = phi, lam = lam)
}

.jp_fixture <- function(seed = 5L, n = 80L) {
  set.seed(seed)
  xy <- cbind(lon = runif(n), lat = runif(n))
  x <- rnorm(n)
  f <- 0.6 * sin(3 * xy[, 1]) * cos(2 * xy[, 2])
  phi_var <- 0.3
  y <- 0.2 + (0.5 + f) * x + rnorm(n, 0, sqrt(phi_var))
  xy_new <- cbind(lon = c(0.1, 0.5, 0.93, 0.3), lat = c(0.2, 0.55, 0.4, 0.9))
  list(xy = xy, X = cbind(intercept = 1, x = x), y = y, phi = phi_var,
       xy_new = xy_new, X_new = cbind(1, c(-1, 0.5, 2, 1.2)))
}

.jp_fit <- function(fx, sigma2 = 0.4, lengthscale = 0.3, m = 4L) {
  tulpa_nested_laplace_joint(
    responses = list(a = list(y = fx$y, X = fx$X, family = "gaussian",
                              phi = fx$phi)),
    prior = list(list(type = "hsgp", coords = list(fx$xy), m = m, c = 1.5,
                      svc_column = 2L, sigma2_grid = sigma2,
                      lengthscale_grid = lengthscale)),
    control = list(axis_refine = "none", tol = 1e-10, max_iter = 100L,
                   diagnose_k = FALSE))
}

test_that("a joint hsgp block built from coordinates keeps its builder", {
  skip_on_cran()
  fx <- .jp_fixture()
  fit <- .jp_fit(fx)
  bs <- fit$blocks[[1]]$basis
  expect_equal(bs$m, 4L)
  ext <- apply(fx$xy, 2, range)
  expect_equal(bs$domain, c(colMeans(ext), 1.5 * (ext[2, ] - ext[1, ]) / 2),
               ignore_attr = TRUE)
  expect_identical(fit$basis_block_cov$idx, c(1:2, 3:18))
  expect_equal(dim(fit$basis_block_cov$cov[[1]]), c(18L, 18L))
  # The leading block is the fixed-effect block the summaries read.
  expect_equal(solve(fit$grid_hessians[[1]]),
               fit$basis_block_cov$cov[[1]][1:2, 1:2], tolerance = 1e-8)
})

test_that("the joint hsgp-svc prediction at new coordinates is the exact Gaussian one", {
  skip_on_cran()
  fx <- .jp_fixture()
  sigma2 <- 0.4; ell <- 0.3; m <- 4L
  fit <- .jp_fit(fx, sigma2, ell, m)
  pr <- predict(fit, newdata = list(X = fx$X_new, coords = fx$xy_new),
                se.fit = TRUE, level = 0.9)

  domain <- fit$blocks[[1]]$basis$domain
  b_tr <- .jp_basis(fx$xy, m, domain)
  b_new <- .jp_basis(fx$xy_new, m, domain)
  sS <- sqrt(sigma2 * 2 * pi * ell^2 * exp(-0.5 * ell^2 * b_tr$lam))
  A <- cbind(fx$X, sweep(b_tr$phi * fx$X[, 2], 2, sS, `*`))
  a <- cbind(fx$X_new, sweep(b_new$phi * fx$X_new[, 2], 2, sS, `*`))
  tau_beta <- 1 / tulpa:::cpp_test_default_sigma_beta()^2
  H <- crossprod(A) / fx$phi + diag(c(tau_beta, tau_beta, rep(1, m * m)))
  mean_x <- solve(H, crossprod(A, fx$y) / fx$phi)
  mu <- as.numeric(a %*% mean_x)
  se <- sqrt(rowSums((a %*% solve(H)) * a))

  expect_equal(pr$fit, mu, tolerance = 1e-6)
  expect_equal(pr$se.fit, se, tolerance = 1e-6)
  expect_equal(pr$upper, mu + qnorm(0.95) * se, tolerance = 1e-5)
  expect_equal(pr$lower, mu - qnorm(0.95) * se, tolerance = 1e-5)
})

test_that("a joint prediction at the fitted rows reproduces the fitted predictor", {
  skip_on_cran()
  fx <- .jp_fixture()
  fit <- .jp_fit(fx)
  eta <- predict(fit, newdata = list(X = fx$X, coords = fx$xy))
  mode <- fit$modes[1, ]
  b <- tulpa:::cpp_hsgp_basis_2d_domain(fx$xy, 4L, fit$blocks[[1]]$basis$domain)
  sS <- tulpa:::cpp_hsgp_sqrt_spectral(b$lambda_eig, 0.4, 0.3)
  expect_equal(eta, as.numeric(fx$X %*% mode[1:2] +
                                 fx$X[, 2] * (b$phi_basis %*% (sS * mode[3:18]))),
               tolerance = 1e-10)
})

test_that("a joint prediction refuses a design it cannot read", {
  skip_on_cran()
  fx <- .jp_fixture()
  fit <- .jp_fit(fx)
  expect_error(predict(fit), "needs `newdata`")
  expect_error(predict(fit, newdata = data.frame(x = 1)), "list with `X`")
  expect_error(predict(fit, newdata = list(X = fx$X_new)), "must carry `coords`")
  expect_error(predict(fit, newdata = list(X = fx$X_new[, 1, drop = FALSE],
                                           coords = fx$xy_new)), "fitted with 2")
})

test_that("a builder block given with its basis as well is refused", {
  skip_on_cran()
  fx <- .jp_fixture()
  expect_error(
    tulpa_nested_laplace_joint(
      responses = list(a = list(y = fx$y, X = fx$X, family = "gaussian",
                                phi = fx$phi)),
      prior = list(list(type = "hsgp", coords = list(fx$xy), m = 4L,
                        phi = list(matrix(0, 80, 16)), sigma2_grid = 0.4,
                        lengthscale_grid = 0.3))),
    "not both")
})

test_that("a two-arm builder block shares one domain and predicts either arm", {
  skip_on_cran()
  fx <- .jp_fixture()
  set.seed(9)
  xy2 <- cbind(lon = runif(50, 0.2, 1.4), lat = runif(50))
  X2 <- cbind(1, rnorm(50))
  y2 <- rpois(50, exp(0.3 + 0.4 * X2[, 2]))
  fit <- tulpa_nested_laplace_joint(
    responses = list(a = list(y = fx$y, X = fx$X, family = "gaussian",
                              phi = fx$phi),
                     b = list(y = y2, X = X2, family = "poisson")),
    prior = list(list(type = "hsgp", coords = list(fx$xy, xy2), m = 3L,
                      svc_column = 2L, sigma2_grid = 0.4,
                      lengthscale_grid = 0.3)),
    control = list(axis_refine = "none", diagnose_k = FALSE))
  ext <- apply(rbind(fx$xy, xy2), 2, range)
  expect_equal(fit$blocks[[1]]$basis$domain,
               c(colMeans(ext), 1.5 * (ext[2, ] - ext[1, ]) / 2),
               ignore_attr = TRUE)
  expect_error(predict(fit, newdata = list(X = X2[1:3, ], coords = xy2[1:3, ])),
               "name the one")
  pr <- predict(fit, newdata = list(X = X2[1:3, ], coords = xy2[1:3, ]),
                arm = "b", type = "response", se.fit = TRUE)
  expect_equal(nrow(pr), 3L)
  expect_true(all(pr$lower < pr$fit & pr$fit < pr$upper & pr$fit > 0))
})
