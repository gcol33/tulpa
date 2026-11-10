test_that("collapsed areal targets and reconstructed fields match dense Gaussian integration", {
  S <- 6L; group <- rep(1:S, each = 3L); N <- length(group)
  X <- cbind(1, seq(-1, 1, length.out = N)); y <- sin(seq_len(N))
  W <- matrix(0, S, S)
  for (edge in list(c(1, 3), c(3, 5), c(2, 4))) W[edge[1], edge[2]] <- W[edge[2], edge[1]] <- 1
  components <- list(c(1, 3, 5), c(2, 4), 6)
  weights <- c(1, 2, 1, 2, 1, 3)
  for (type in c('icar', 'bym2')) for (restricted in c(FALSE, TRUE)) {
    sp <- list(type = type, parameterization = 'collapsed', spatial_idx = group,
      n_spatial_units = S, adj_row_ptr = c(0L, 1L, 2L, 4L, 5L, 6L, 6L),
      adj_col_idx = c(2L, 3L, 0L, 4L, 1L, 2L), n_neighbors = as.integer(rowSums(W)),
      scale_factor = 1.7, node_prec = weights, component_centering = TRUE)
    P <- diag(S)
    multiplier <- if (type == 'bym2') weights else rep(1, S)
    Q <- diag(multiplier) %*% (diag(rowSums(W)) - W)
    for (cc in components) {
      coefficient <- if (length(cc) > 1) 1 else multiplier[cc]
      Q[cc, cc] <- Q[cc, cc] + coefficient / length(cc)
      if (length(cc) > 1) P[cc, cc] <- P[cc, cc] - 1 / length(cc)
    }
    projection <- diag(N)
    if (restricted) {
      basis <- qr.Q(qr(X))
      sp$restriction_basis <- basis
      projection <- projection - tcrossprod(basis)
    }
    point <- if (type == 'icar') c(.1, -.2, .4) else c(.1, -.2, .3, -.7)
    at <- function(q, draws = 0L) tulpa:::cpp_test_collapsed_spatial(y, X, sp, q, draws)
    r <- at(point, 20000L); zero <- at(numeric(length(point)))
    residual <- y - X %*% point[1:2]
    if (type == 'icar') {
      precision <- exp(point[3]) * Q
      B <- projection %*% P[group, ]
      expected_prior <- c(-point[1:2] / 2.5^2, -.5 + .5 * (-log(.01) / 3) * exp(-.5 * point[3]))
    } else {
      precision <- matrix(0, 2 * S, 2 * S)
      precision[1:S, 1:S] <- Q; precision[S + (1:S), S + (1:S)] <- diag(S)
      rho <- plogis(point[4]); sigma <- exp(point[3])
      B <- projection %*% (sigma * cbind(1.7 * sqrt(rho) * P[group, ], sqrt(1 - rho) * diag(S)[group, ]))
      expected_prior <- c(-point[1:2] / 2.5^2, 1 - (-log(.01) / 3) * sigma, 1 - 2 * rho)
    }
    V <- diag(N) + B %*% solve(precision, t(B))
    marginal <- -.5 * sum(residual * solve(V, residual)) - .5 * as.numeric(determinant(V)$modulus)
    H <- precision + crossprod(B)
    mean <- drop(solve(H, crossprod(B, residual))); covariance <- solve(H)
    mode <- c(r$mode_phi, r$mode_theta)
    draws <- cbind(r$phi, r$theta)
    expect_equal(r$lp - r$prior, marginal, tolerance = 2e-9)
    expect_equal(mode, mean, tolerance = 2e-8)
    expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
    expect_equal(r$prior_gradient, expected_prior, tolerance = 1e-10)
    expect_equal(r$prior_gradient, r$numerical_prior, tolerance = 2e-5)
    expect_true(r$verified)
    expect_true(all(abs(colMeans(draws) - mean) < 6 * sqrt(diag(covariance) / nrow(draws))))
    covariance_se <- sqrt((covariance^2 + outer(diag(covariance), diag(covariance))) / (nrow(draws) - 1))
    expect_true(all(abs(cov(draws) - covariance) < 6 * covariance_se))
    rate <- -log(.01) / 3
    prior_change <- -.5 * sum(point[1:2]^2) / 2.5^2 +
      if (type == 'icar') -.5 * point[3] - rate * (exp(-.5 * point[3]) - 1) else
        point[3] - rate * (exp(point[3]) - 1) + log(plogis(point[4])) +
          log(plogis(-point[4])) + 2 * log(2)
    expect_equal(r$prior - zero$prior, prior_change, tolerance = 1e-10)
  }
})
