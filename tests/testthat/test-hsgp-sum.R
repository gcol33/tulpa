.sum_hsgp_fixture <- function(reduced = TRUE, family = "gaussian") {
  set.seed(123L)
  xy <- cbind(runif(7), runif(7))
  xy[7, ] <- xy[1, ]
  coords <- list(xy[1:5, ], xy[c(2, 4, 6, 7), ])
  X <- lapply(coords, function(z) cbind(1, rnorm(nrow(z))))
  X[[1]][1:2, 2] <- 0
  responses <- lapply(seq_along(X), function(k) list(
    X = X[[k]], y = if (family == "gaussian") rnorm(nrow(X[[k]]))
        else rpois(nrow(X[[k]]), exp(.3 + .2 * X[[k]][, 2])),
    family = family, phi = .3))
  names(responses) <- c("a", "b")
  hsgp <- lapply(1:2, function(b) list(type = "hsgp", coords = coords,
    m = c(3L, 4L)[b], c = 1.8, svc_column = c(2L, 1L)[b],
    sigma2_grid = c(.4, .7)[b], lengthscale_grid = c(.2, .5)[b],
    observation_space = reduced))
  prior <- list(hsgp[[1]], list(type = "iid", n_units = 2L,
    obs_idx = list(c(1L, 1L, 2L, 2L, 2L), c(1L, 2L, 2L, 1L)), sigma_grid = .4), hsgp[[2]])
  fit <- tulpa_nested_laplace_joint(responses, prior = prior,
    control = list(progress = FALSE, axis_refine = "none", diagnose_k = FALSE,
                   store_Q = TRUE, tol = 1e-12))
  list(fit = fit, responses = responses, coords = coords)
}

test_that("summed HSGP coordinates retain mixed-block and varying-arm predictions", {
  skip_on_cran()
  for (family in c("gaussian", "poisson")) {
    small <- .sum_hsgp_fixture(family = family)
    full <- .sum_hsgp_fixture(FALSE, family)
    s <- small$fit; f <- full$fit
    expect_equal(ncol(s$modes), 4L + 9L + 2L)
    expect_equal(s$arm_layout$block_start, c(4L, 13L, 4L))
    expect_equal(s$arm_layout$block_size, c(9L, 2L, 9L))
    expect_equal(s$log_marginal, f$log_marginal, tolerance = 1e-8)
    for (arm in c("a", "b")) {
      nd <- list(X = cbind(1, c(-.3, 1.2)),
                 coords = rbind(c(.4, .5), c(.6, .4)),
                 index = list(b2 = c(1L, 2L)))
      a <- predict(s, newdata = nd, arm = arm, se.fit = TRUE)
      b <- predict(f, newdata = nd, arm = arm, se.fit = TRUE)
      expect_equal(a, b, tolerance = 1e-8)
      expect_lt(max(abs(as.matrix(a[, c("lower", "upper")] -
                                  b[, c("lower", "upper")]) / b$se.fit)), 1e-8)
    }
  }
})

test_that("component residuals preserve their full joint Gaussian covariance", {
  skip_on_cran()
  set.seed(321L)
  train <- list(matrix(rnorm(5 * 13), 5, 13), matrix(rnorm(5 * 7), 5, 7))
  pred <- lapply(c(13L, 7L), function(m) matrix(rnorm(3L * m), 3L, m))
  spectral <- lapply(c(13L, 7L), function(m) exp(seq(-3, 1, length.out = m)))
  for (duplicate in c(FALSE, TRUE)) {
    if (duplicate) train <- lapply(train, function(a) { a[5, ] <- a[1, ]; a })
    p <- tulpa:::cpp_hsgp_sum_projection(train, pred, spectral)
    projected <- do.call(rbind, lapply(p, `[[`, "basis"))
    residual <- do.call(rbind, lapply(p, `[[`, "residual_basis"))
    B <- matrix(0, 6L, 20L)
    B[1:3, 1:13] <- sweep(pred[[1]], 2, spectral[[1]], "*")
    B[4:6, 14:20] <- sweep(pred[[2]], 2, spectral[[2]], "*")
    expect_equal(tcrossprod(projected) + tcrossprod(residual), tcrossprod(B),
                 tolerance = 1e-12)
    expect_gt(max(abs(tcrossprod(residual)[1:3, 4:6])), .01)
    A <- do.call(cbind, Map(function(a, s) sweep(a, 2, s, "*"), train, spectral))
    y <- seq(-1, 1, length.out = 5)
    V <- solve(diag(20) + crossprod(A) / .3)
    m <- as.numeric(V %*% crossprod(A, y) / .3)
    training_projection <- Reduce(`+`, lapply(
      tulpa:::cpp_hsgp_sum_projection(train, train, spectral), `[[`, "basis"))
    v <- solve(diag(5) + crossprod(training_projection) / .3)
    mu <- as.numeric(v %*% crossprod(training_projection, y) / .3)
    expect_equal(as.numeric(projected %*% mu), as.numeric(B %*% m), tolerance = 1e-8)
    expect_equal(projected %*% v %*% t(projected) + tcrossprod(residual),
                 B %*% V %*% t(B), tolerance = 1e-8)
  }
})

test_that("registry field reads locate an interleaved random effect beside a shared field", {
  skip_on_cran()
  set.seed(412L)
  xy <- cbind(runif(8), runif(8))
  bases <- lapply(c(3L, 4L), function(m) tulpa:::cpp_hsgp_basis_2d(xy, m, 1.8))
  X <- cbind(1, rnorm(8))
  y <- rnorm(8)
  fit <- function(reduced) {
    hsgp <- lapply(1:2, function(b) list(type = "hsgp",
      phi_basis = bases[[b]]$phi_basis, lambda_eig = bases[[b]]$lambda_eig,
      observation_space = reduced, sigma2_grid = c(.4, .7)[b],
      lengthscale_grid = c(.2, .5)[b]))
    tulpa_nested_laplace(y, rep(1L, 8), X, family = "gaussian", phi = .3,
      prior = list(hsgp[[1]], list(type = "iid", n_units = 2L,
        obs_idx = rep(1:2, 4), sigma_grid = .4), hsgp[[2]]),
      control = list(progress = FALSE, axis_refine = "none", diagnose_k = FALSE,
                     keep_grid_hessians = TRUE, tol = 1e-12))
  }
  a <- fit(TRUE); b <- fit(FALSE)
  expect_equal(ncol(a$modes), 12L)
  expect_equal(a$log_marginal, b$log_marginal, tolerance = 1e-8)
  expect_equal(a$fitted_eta, b$fitted_eta, tolerance = 1e-8)
  expect_equal(a$fitted_eta_var, b$fitted_eta_var, tolerance = 1e-8)
  expect_equal(a$grid_re_mean, b$grid_re_mean, tolerance = 1e-8)
  expect_equal(a$grid_re_var, b$grid_re_var, tolerance = 1e-8)
  expect_equal(tulpa:::.nl_block_cols(a$block_latent_offsets, 12L, 2L, 3L), 11:12)
})

test_that("component prediction quantiles integrate the shared posterior over outer cells", {
  skip_on_cran()
  set.seed(943L)
  xy <- cbind(runif(7), runif(7))
  X <- cbind(1, rnorm(7))
  y <- rnorm(7)
  nd <- cbind(c(.3, .5, .7), c(.6, .4, .3))
  fit <- function(reduced, n_outer = 1L) tulpa_nested_laplace_joint(
    list(a = list(y = y, X = X, family = "gaussian", phi = .3)),
    prior = lapply(c(3L, 4L), function(m) list(type = "hsgp", coords = list(xy),
      m = m, c = 1.8, sigma2_grid = c(.3, .7), lengthscale_grid = c(.15, .4),
      observation_space = reduced)), control = list(progress = FALSE,
        axis_refine = "none", diagnose_k = FALSE, store_Q = TRUE, tol = 1e-12,
        n_threads_outer = n_outer))
  a <- fit(TRUE); b <- fit(FALSE)
  parallel <- fit(TRUE, 2L)
  expect_equal(parallel$fitted_eta, a$fitted_eta, tolerance = 1e-12)
  expect_equal(parallel$log_marginal, a$log_marginal, tolerance = 1e-12)
  expect_equal(a$log_evidence, b$log_evidence, tolerance = 1e-8)
  summary <- function(fit, shared) {
    mu <- vv <- matrix(0, nrow(fit$modes), 6)
    for (k in seq_len(nrow(fit$modes))) {
      train <- pred <- spectra <- vector("list", 2L)
      for (j in 1:2) {
        p <- fit$blocks[[j]]
        train[[j]] <- p$phi[[1L]]
        pred[[j]] <- tulpa:::cpp_hsgp_basis_2d_domain(nd, p$basis$m, p$basis$domain)$phi_basis
        spectra[[j]] <- tulpa:::cpp_hsgp_sqrt_spectral(p$eigenvalues,
          fit$theta_grid[k, paste0("b", j, ".sigma2")],
          fit$theta_grid[k, paste0("b", j, ".lengthscale")],
          tulpa:::.hsgp_kernel_code(p$kernel))
      }
      C <- matrix(0, 6, ncol(fit$modes))
      residual <- rep(0, 6)
      if (shared) {
        p <- tulpa:::cpp_hsgp_sum_projection(train, pred, spectra)
        C[, 2L + 1:7] <- do.call(rbind, lapply(p, `[[`, "basis"))
        residual <- unlist(lapply(p, function(p) rowSums(p$residual_basis^2)))
      } else for (j in 1:2) {
        cols <- fit$arm_layout$block_start[j] + seq_len(fit$arm_layout$block_size[j])
        C[(j - 1L) * 3L + 1:3, cols] <- sweep(pred[[j]], 2, spectra[[j]], "*")
      }
      Q <- Matrix::sparseMatrix(p = fit$Q_csc_p_per_grid[[k]],
        i = fit$Q_csc_i_per_grid[[k]], x = fit$Q_csc_x_per_grid[[k]],
        dims = rep(ncol(fit$modes), 2), symmetric = TRUE, index1 = FALSE)
      V <- as.matrix(solve(Q))
      mu[k, ] <- as.numeric(C %*% fit$modes[k, ])
      vv[k, ] <- rowSums((C %*% V) * C) + residual
    }
    tulpa:::.nl_gauss_mixture_summary(mu, vv, fit$weights, probs = c(.025, .5, .975))
  }
  aa <- summary(a, TRUE); bb <- summary(b, FALSE)
  expect_equal(aa$mean, bb$mean, tolerance = 1e-8)
  expect_equal(aa$sd, bb$sd, tolerance = 1e-8)
  expect_lt(max(abs(aa$quantiles - bb$quantiles) / bb$sd), 1e-8)
})

test_that("field prediction draws use the summed correlated residual", {
  skip_on_cran()
  set.seed(818L)
  xy <- cbind(lon = runif(7), lat = runif(7))
  X <- cbind(intercept = 1, x = rnorm(7))
  y <- rnorm(7)
  bases <- lapply(c(3L, 4L), function(m) tulpa:::cpp_hsgp_basis_2d(xy, m, 1.8))
  prior <- lapply(1:2, function(j) list(type = "hsgp", name = c("local", "regional")[j],
    phi_basis = bases[[j]]$phi_basis, lambda_eig = bases[[j]]$lambda_eig,
    sigma2_grid = c(.4, .7)[j], lengthscale_grid = c(.2, .5)[j]))
  a <- tulpa_nested_laplace(y, rep(1L, 7), X, prior, family = "gaussian", phi = .3,
    control = list(progress = FALSE, axis_refine = "none", diagnose_k = FALSE, tol = 1e-12))
  a$spatial <- list(type = "multiscale", approx = "hsgp", coords_matrix = xy,
    coord_vars = c("lon", "lat"), scale_basis = stats::setNames(
      lapply(c(3L, 4L), function(m) list(m = m, c = 1.8)), c("local", "regional")))
  src <- tulpa:::.field_source_grid(a)
  nd <- data.frame(lon = c(.3, .5, .7), lat = c(.6, .4, .3))
  inp <- lapply(src$layout$fields, function(comp)
    tulpa:::.field_hsgp_inputs(a, src, nd, comp, 1L))
  p <- tulpa:::cpp_hsgp_sum_projection(lapply(inp, `[[`, "training"),
    lapply(inp, `[[`, "prediction"), lapply(inp, function(p) p$spectral[[1L]]))
  P <- Reduce(`+`, lapply(p, `[[`, "basis"))
  R <- Reduce(`+`, lapply(p, `[[`, "residual_basis"))
  L <- a$modes[rep(1L, 4L), , drop = FALSE]
  set.seed(913L)
  got <- tulpa:::.field_hsgp_sum_at(a, src, nd, L, rep(1L, 4L), TRUE, src$layout$fields)
  set.seed(913L)
  expected <- L[, 3:9] %*% t(P) + matrix(rnorm(4L * ncol(R)), 4L) %*% t(R)
  expect_equal(got, expected, tolerance = 1e-12)
  point <- tulpa:::.field_eta_rows(a, src, list(X = X, offset = 0), NULL,
    a$modes, 1L, FALSE, des = NULL)
  expect_equal(as.numeric(point), as.numeric(a$fitted_eta), tolerance = 1e-8)
})

test_that("a summed HSGP checkpoint retains the fitted predictor", {
  skip_on_cran()
  set.seed(921L)
  xy <- cbind(runif(7), runif(7))
  X <- cbind(1, rnorm(7))
  y <- rnorm(7)
  path <- tempfile(fileext = ".ckpt")
  on.exit(unlink(path), add = TRUE)
  fit <- function(checkpoint) tulpa_nested_laplace_joint(
    list(a = list(y = y, X = X, family = "gaussian", phi = .3)),
    prior = lapply(c(3L, 4L), function(m) list(type = "hsgp", coords = list(xy),
      m = m, c = 1.8, sigma2_grid = c(.3, .7), lengthscale_grid = .3)),
    control = list(progress = FALSE, axis_refine = "none", diagnose_k = FALSE,
                   checkpoint = checkpoint, tol = 1e-12))
  reference <- fit(NULL)
  fresh <- fit(list(path = path, resume = FALSE))
  resumed <- fit(list(path = path, resume = TRUE))
  expect_identical(resumed$fitted_eta, fresh$fitted_eta)
  expect_equal(resumed$fitted_eta, reference$fitted_eta, tolerance = 1e-12)
  bytes <- readBin(path, "raw", n = file.size(path))
  writeBin(bytes[seq_len(as.integer(length(bytes) * .6))], path)
  partial <- fit(list(path = path, resume = TRUE))
  expect_equal(partial$fitted_eta, reference$fitted_eta, tolerance = 1e-12)
})

test_that("summed HSGP loadings validate their observation rows", {
  skip_on_cran()
  blocks <- lapply(c(9L, 16L), function(m) list(type = "hsgp",
    phi_basis = matrix(1, 6, m), lambda_eig = seq_len(m), kernel = 1L))
  expect_error(tulpa:::cpp_nested_laplace_multi(
    y = rep(0, 7), n = rep(1L, 7), X = matrix(1, 7, 1), re_idx = numeric(0),
    n_re_groups = 0L, sigma_re = 1, blocks_spec = blocks,
    theta_grid = matrix(c(.3, .2, .7, .5), 1, 4), axis_offsets = c(0L, 2L, 4L),
    family = "gaussian"), "basis rows")
})
