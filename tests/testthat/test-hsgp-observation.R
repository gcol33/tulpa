.obs_fixture <- function(family = "gaussian", reduced = TRUE, count = 2L, store = TRUE) {
  set.seed(182L)
  n <- 9L
  xy <- cbind(runif(n), runif(n))
  X <- cbind(1, rnorm(n))
  y <- if (family == "gaussian") .4 + .3 * X[, 2] + rnorm(n, sd = .4)
       else rpois(n, exp(.4 + .3 * X[, 2]))
  blocks <- lapply(seq_len(count), function(b) list(type = "hsgp", coords = list(xy),
    m = 5L, c = 1.8, sigma2_grid = c(.3, .7)[b], lengthscale_grid = c(.15, .4)[b],
    observation_space = reduced))
  fit <- tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, X = X, family = family, phi = .16)),
    prior = blocks, control = list(progress = FALSE, tol = 1e-12,
      axis_refine = "none", diagnose_k = FALSE, store_Q = store))
  list(fit = fit, xy = xy, X = X, y = y)
}

test_that("observation coordinates preserve Gaussian evidence and prediction distributions", {
  skip_on_cran()
  small <- .obs_fixture()
  full <- .obs_fixture(reduced = FALSE)
  expect_equal(ncol(small$fit$modes), 11L)
  expect_equal(ncol(full$fit$modes), 52L)
  arms <- list(list(X = small$X, n_re_groups = 0L))
  expect_equal(tulpa:::.joint_multi_layout(arms, small$fit$blocks)$n_x,
               ncol(small$fit$modes))
  expect_equal(tulpa:::.joint_multi_layout(arms, full$fit$blocks)$n_x,
               ncol(full$fit$modes))
  nd <- list(X = cbind(1, c(-1, .2, 1)), coords = cbind(c(.2, .5, .8), c(.7, .4, .2)))
  pr <- predict(small$fit, newdata = nd, se.fit = TRUE)
  reference <- predict(full$fit, newdata = nd, se.fit = TRUE)
  expect_equal(pr, reference, tolerance = 1e-8)
  expect_equal(small$fit$log_marginal, full$fit$log_marginal, tolerance = 1e-8)
  bs <- lapply(1:2, function(j) {
    b <- small$fit$blocks[[j]]
    spectral <- tulpa:::cpp_hsgp_sqrt_spectral(b$eigenvalues,
      c(.3, .7)[j], c(.15, .4)[j], tulpa:::.hsgp_kernel_code(b$kernel))
    phi_new <- tulpa:::cpp_hsgp_basis_2d_domain(nd$coords, 5L,
      b$basis$domain)$phi_basis
    list(train = sweep(b$phi[[1]], 2, spectral, "*"),
         new = sweep(phi_new, 2, spectral, "*"))
  })
  V <- .16 * diag(9) + 2.5^2 * tcrossprod(small$X) +
    tcrossprod(bs[[1]]$train) + tcrossprod(bs[[2]]$train)
  exact <- -.5 * (9 * log(2 * pi) + as.numeric(determinant(V, logarithm = TRUE)$modulus) +
    sum(small$y * solve(V, small$y)))
  expect_equal(small$fit$log_marginal - small$fit$log_hyperprior, exact, tolerance = 1e-8)
  cross <- 2.5^2 * tcrossprod(nd$X, small$X) +
    tcrossprod(bs[[1]]$new, bs[[1]]$train) +
    tcrossprod(bs[[2]]$new, bs[[2]]$train)
  prior <- 2.5^2 * tcrossprod(nd$X) +
    tcrossprod(bs[[1]]$new) + tcrossprod(bs[[2]]$new)
  expect_equal(pr$fit, as.numeric(cross %*% solve(V, small$y)), tolerance = 1e-8)
  expect_equal(pr$se.fit, sqrt(diag(prior - cross %*% solve(V, t(cross)))),
               tolerance = 1e-8)
})

test_that("a single HSGP field preserves its basis-space prediction distribution", {
  skip_on_cran()
  for (family in c("gaussian", "poisson")) {
    small <- .obs_fixture(family, count = 1L)
    full <- .obs_fixture(family, FALSE, count = 1L)
    nd <- list(X = cbind(1, c(-1, .2, 1)), coords = cbind(c(.2, .5, .8), c(.7, .4, .2)))
    expect_equal(ncol(small$fit$modes), 11L)
    expect_equal(small$fit$log_marginal, full$fit$log_marginal, tolerance = 1e-8)
    expect_equal(predict(small$fit, newdata = nd, se.fit = TRUE),
                 predict(full$fit, newdata = nd, se.fit = TRUE), tolerance = 1e-8)
  }
})

test_that("summed HSGP predictions use the retained joint covariance block", {
  skip_on_cran()
  small <- .obs_fixture(store = FALSE)$fit
  full <- .obs_fixture(reduced = FALSE, store = FALSE)$fit
  nd <- list(X = cbind(1, c(-1, .2, 1)), coords = cbind(c(.2, .5, .8), c(.7, .4, .2)))
  expect_null(small$Q_csc_x_per_grid)
  expect_identical(small$basis_block_cov$idx, 1:11)
  expect_equal(predict(small, newdata = nd, se.fit = TRUE),
               predict(full, newdata = nd, se.fit = TRUE), tolerance = 1e-8)
})

test_that("Poisson observation coordinates preserve each component and their cross covariance", {
  skip_on_cran()
  small <- .obs_fixture("poisson")
  full <- .obs_fixture("poisson", FALSE)
  expect_equal(small$fit$log_marginal, full$fit$log_marginal, tolerance = 1e-8)
  s <- small$fit; f <- full$fit
  nd <- list(X = cbind(1, c(-1, .2, 1)), coords = cbind(c(.2, .5, .8), c(.7, .4, .2)))
  expect_equal(predict(s, newdata = nd, se.fit = TRUE),
               predict(f, newdata = nd, se.fit = TRUE), tolerance = 1e-8)
  C <- matrix(0, 6L, ncol(s$modes)); B <- matrix(0, 6L, ncol(f$modes))
  residual <- matrix(0, 6L, 6L)
  training <- prediction <- spectral <- vector("list", 2L)
  for (b in 1:2) {
    p <- s$blocks[[b]]
    phi <- tulpa:::cpp_hsgp_basis_2d_domain(nd$coords, 5L, p$basis$domain)$phi_basis
    spectral[[b]] <- tulpa:::cpp_hsgp_sqrt_spectral(p$eigenvalues,
      c(.3, .7)[b], c(.15, .4)[b], tulpa:::.hsgp_kernel_code(p$kernel))
    training[[b]] <- p$phi[[1]]
    prediction[[b]] <- phi
    rows <- (b - 1L) * 3L + 1:3
    B[rows, 2L + (b - 1L) * 25L + 1:25] <- sweep(phi, 2L, spectral[[b]], "*")
  }
  projection <- tulpa:::cpp_hsgp_sum_projection(training, prediction, spectral)
  C[, 2L + 1:9] <- do.call(rbind, lapply(projection, `[[`, "basis"))
  residual <- tcrossprod(do.call(rbind, lapply(projection, `[[`, "residual_basis")))
  expect_equal(as.numeric(C %*% s$modes[1, ]), as.numeric(B %*% f$modes[1, ]), tolerance = 1e-8)
  covariance <- function(fit) {
    n <- ncol(fit$modes)
    Q <- Matrix::sparseMatrix(p = fit$Q_csc_p_per_grid[[1]],
      i = fit$Q_csc_i_per_grid[[1]], x = fit$Q_csc_x_per_grid[[1]],
      dims = c(n, n), symmetric = TRUE, index1 = FALSE)
    as.matrix(solve(Q))
  }
  Cs <- covariance(s)
  Cf <- covariance(f)
  expect_equal(C %*% Cs %*% t(C) + residual, B %*% Cf %*% t(B), tolerance = 1e-8)
})

test_that("orthogonal prediction residuals reproduce the full finite-basis covariance", {
  skip_on_cran()
  set.seed(2L)
  train <- matrix(rnorm(7 * 23), 7, 23)
  pred <- matrix(rnorm(5 * 23), 5, 23)
  spectral <- exp(seq(-3, 2, length.out = 23))
  p <- tulpa:::cpp_hsgp_observation_projection(train, pred, spectral)
  expect_equal(tcrossprod(p$basis) + tcrossprod(p$residual_basis),
               tcrossprod(sweep(pred, 2, spectral, "*")), tolerance = 1e-12)
  train[7, ] <- train[1, ]
  p <- tulpa:::cpp_hsgp_observation_projection(train, pred, spectral)
  expect_equal(tcrossprod(p$basis) + tcrossprod(p$residual_basis),
               tcrossprod(sweep(pred, 2, spectral, "*")), tolerance = 1e-12)
})

test_that("space-time HSGP field means use the cell's observation loading", {
  skip_on_cran()
  set.seed(43)
  n <- 12L
  xy <- cbind(runif(n), runif(n))
  X <- matrix(1, n, 1)
  y <- rnorm(n)
  tt <- rep(1:3, 4)
  fit <- suppressWarnings(fit_st_nested(y, X, NULL, NULL, tt, 3L,
    spatial_type = "hsgp", temporal_type = "ar1", family = "gaussian", phi = .4,
    coords = xy, hsgp_m = 5L, hsgp_c = 1.8,
    control = list(n_grid_spatial = 2L, n_grid_temporal = 2L,
                   n_grid_rho = 2L)))
  expect_equal(length(fit$field_cols$spatial), n)
  eta <- as.numeric(coef(fit)[1]) + fit$spatial_effects + fit$temporal_effects[tt]
  expect_equal(eta, as.numeric(crossprod(fit$weights, fit$fitted_eta)), tolerance = 1e-8)
})

test_that("a shared varying HSGP field retains its training rows across response arms", {
  skip_on_cran()
  set.seed(91)
  xy <- cbind(runif(6), runif(6))
  coords <- list(xy, xy[1:5, ])
  X <- list(cbind(1, rnorm(6)), cbind(1, rnorm(5)))
  X[[1]][1:2, 2] <- 0
  responses <- lapply(1:2, function(k)
    list(y = rnorm(nrow(X[[k]])), X = X[[k]], family = "gaussian", phi = .3))
  names(responses) <- c("a", "b")
  fit <- function(reduced) tulpa_nested_laplace_joint(responses,
    prior = list(list(type = "hsgp", coords = coords, m = 5L, c = 1.8,
      svc_column = 2L, sigma2_grid = .4, lengthscale_grid = .2,
      observation_space = reduced)), control = list(progress = FALSE, tol = 1e-12,
      axis_refine = "none", diagnose_k = FALSE))
  a <- fit(TRUE); b <- fit(FALSE)
  nd <- list(X = cbind(1, c(-1, 2)), coords = xy[c(2, 5), ] + .01)
  expect_equal(ncol(a$modes), 15L)
  expect_equal(a$log_marginal, b$log_marginal, tolerance = 1e-8)
  expect_equal(predict(a, newdata = nd, arm = "a", se.fit = TRUE),
               predict(b, newdata = nd, arm = "a", se.fit = TRUE), tolerance = 1e-8)
})
