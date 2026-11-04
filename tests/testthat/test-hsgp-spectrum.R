test_that("the 2-D frequency rectangle retains the stated spectral variance", {
  skip_on_cran_budget()
  for (kernel in names(tulpa:::.HSGP_KERNELS)) {
    for (S in list(c(1, 1), c(1, 0.6))) {
      a <- tulpa:::.hsgp_m_coefficient(S, kernel)
      omega <- pi * a * max(S) / (2 * S)
      nu <- switch(kernel, matern32 = 1.5, matern52 = 2.5)
      density <- function(x, y) {
        r2 <- x^2 + y^2
        if (is.null(nu)) return(exp(-r2 / 2) / (2 * pi))
        nu * (2 * nu)^nu / pi * (2 * nu + r2)^(-nu - 1)
      }
      mass <- 4 * integrate(function(x) vapply(x, function(xx)
        integrate(function(y) density(xx, y), 0, omega[2L],
                  rel.tol = 1e-9)$value, numeric(1)), 0, omega[1L],
        rel.tol = 1e-8)$value
      expect_gte(mass, 0.99 - 1e-8)
      expect_equal(1 - mass, tulpa:::.hsgp_spectral_tail(a, S, kernel),
                    tolerance = 1e-8)
    }
  }
  a <- tulpa:::.hsgp_m_coefficient(c(1, 1), "matern32")
  expect_gt(a, tulpa:::.HSGP_KERNELS$matern32$m_coef)
  expect_equal(a, 4.537001685179831, tolerance = 1e-7)
})

test_that("the 2-D boundary and basis preserve the rough-field covariance", {
  skip_on_cran()
  set.seed(82)
  xy <- rbind(c(0, 0), c(1, 1), matrix(runif(160), 80, 2))
  D <- as.matrix(dist(xy))
  for (ell in c(0.08, 0.12)) {
    sz <- tulpa:::.hsgp_basis_size(c(0.5, 0.5), ell, kernel = "matern32")
    expect_lte(tulpa:::.hsgp_boundary_error(sz$c, c(0.5, 0.5), ell,
                                           "matern32"), 0.01 + 1e-8)
    b <- tulpa:::cpp_hsgp_basis_2d(xy, sz$m, sz$c)
    s <- 6 * pi * 3^1.5 / ell^3 * (3 / ell^2 + b$lambda_eig)^(-2.5)
    Kb <- tcrossprod(sweep(b$phi_basis, 2, sqrt(s), `*`))
    r <- sqrt(3) * D / ell
    K <- (1 + r) * exp(-r)
    expect_lt(max(abs(Kb - K)), 0.02)
    obs <- seq_len(60L); out <- 61:82
    y <- as.numeric(t(chol(K[obs, obs] + diag(0.2, length(obs)))) %*%
                      rnorm(length(obs)))
    posterior <- function(C) {
      cross <- C[out, obs]
      V <- solve(C[obs, obs] + diag(0.2, length(obs)))
      list(mean = as.numeric(cross %*% V %*% y),
           sd = sqrt(diag(C)[out] - rowSums((cross %*% V) * cross)))
    }
    exact <- posterior(K); approx <- posterior(Kb)
    expect_lt(max(abs(approx$mean - exact$mean) / exact$sd), 0.1)
    expect_lt(max(abs(approx$sd - exact$sd) / exact$sd), 0.1)
  }
})

test_that("the 2-D covariance variation meets the paper's target at a corner", {
  skip_on_cran_budget()
  S <- 0.5
  q <- seq(-S, S, length.out = 129L)
  w <- (2 * S / 128 / 3) * c(1, rep(c(4, 2), 63), 4, 1)
  W <- outer(w, w)
  for (ell in c(0.08, 0.12)) {
    sz <- tulpa:::.hsgp_basis_size(c(S, S), ell, kernel = "matern32")
    L <- sz$c * S
    j <- seq_len(sz$m)
    eig <- (pi * j / (2 * L))^2
    spec <- 6 * pi * 3^1.5 / ell^3 *
      (3 / ell^2 + outer(eig, eig, `+`))^(-2.5)
    phi <- sin(outer(q + L, j) * pi / (2 * L)) / sqrt(L)
    anchor <- sin((S + L) * j * pi / (2 * L)) / sqrt(L)
    approx <- phi %*% (spec * outer(anchor, anchor)) %*% t(phi)
    r <- sqrt(3 * outer((q - S)^2, (q - S)^2, `+`)) / ell
    exact <- (1 + r) * exp(-r)
    variation <- sum(abs(exact - approx) * W) / sum(exact * W)
    expect_lt(variation, 0.01)
  }
})

test_that("known-hyperparameter Gaussian fields retain held-out interval coverage", {
  skip_if_not_slow()
  coverage <- vapply(seq_len(20L), function(seed) {
    set.seed(1100L + seed)
    xy <- rbind(c(0, 0), c(1, 1), matrix(runif(236), 118, 2))
    obs <- seq_len(80L); out <- 81:120
    ell <- 0.12; sigma2 <- 0.64; noise <- 0.2
    r <- sqrt(3) * as.matrix(dist(xy)) / ell
    K <- sigma2 * (1 + r) * exp(-r)
    eta <- rnorm(1, sd = 2.5) +
      as.numeric(t(chol(K + diag(1e-10, nrow(K)))) %*% rnorm(nrow(K)))
    y <- eta[obs] + rnorm(length(obs), sd = sqrt(noise))
    sz <- tulpa:::.hsgp_basis_size(c(0.5, 0.5), ell, kernel = "matern32")
    fit <- suppressWarnings(tulpa_nested_laplace_joint(
      responses = list(a = list(y = y, X = matrix(1, length(obs), 1),
                                family = "gaussian", phi = noise)),
      prior = list(list(type = "hsgp", coords = list(xy[obs, ]),
                        m = sz$m, c = sz$c, cov = "matern", nu = 1.5,
                        sigma2_grid = sigma2, lengthscale_grid = ell)),
      control = list(progress = FALSE, axis_refine = "none", diagnose_k = FALSE)))
    p <- predict(fit, newdata = list(X = matrix(1, length(out), 1),
                                    coords = xy[out, ]), se.fit = TRUE)
    mean(p$lower <= eta[out] & eta[out] <= p$upper)
  }, numeric(1))
  expect_gte(mean(coverage), 0.9)
})
