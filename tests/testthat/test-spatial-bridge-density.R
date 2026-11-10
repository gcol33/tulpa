spatial_bridge_basis <- function(coords, m = 2L, c = 1.5) {
  center <- apply(coords, 2, function(x) mean(range(x)))
  L <- c * apply(coords, 2, function(x) diff(range(x)) / 2)
  L[L == 0] <- .1
  pairs <- as.matrix(expand.grid(j2 = seq_len(m), j1 = seq_len(m)))[, 2:1]
  phi <- vapply(seq_len(nrow(pairs)), function(j) {
    sin(pi * pairs[j, 1] * (coords[, 1] - center[1] + L[1]) / (2 * L[1])) *
      sin(pi * pairs[j, 2] * (coords[, 2] - center[2] + L[2]) / (2 * L[2])) /
      sqrt(prod(L))
  }, numeric(nrow(coords)))
  eigen <- (pi * pairs[, 1] / (2 * L[1]))^2 + (pi * pairs[, 2] / (2 * L[2]))^2
  list(phi = phi, eigen = eigen)
}

test_that("proper CAR priors and spatial projections match dense precision references", {
  set.seed(191)
  S <- 4L; N <- 24L
  group <- rep(seq_len(S), each = N / S)
  X <- cbind(1, rnorm(N)); y <- rnorm(N)
  W <- matrix(0, S, S)
  for (s in 1:3) W[s, s + 1L] <- W[s + 1L, s] <- 1
  degree <- rowSums(W)
  sp <- list(type = 'car_proper', n_spatial_units = S, spatial_idx = group,
    adj_row_ptr = c(0L, 1L, 3L, 5L, 6L), adj_col_idx = c(1L, 0L, 2L, 1L, 3L, 2L),
    n_neighbors = as.integer(degree), rho_lower = -.8, rho_upper = .9,
    rho_prior_a = 2, rho_prior_b = 3, sigma_prior_U = 3, sigma_prior_alpha = .01,
    adj_eigenvalues = eigen(W / sqrt(outer(degree, degree)), symmetric = TRUE)$values)
  for (center in c(FALSE, TRUE)) for (restricted in c(FALSE, TRUE)) {
    sp$center <- center
    Q <- qr.Q(qr(X))
    if (restricted) sp$restriction_basis <- Q else sp$restriction_basis <- NULL
    evaluate <- function(shift) tulpa:::cpp_test_field_coordinates(y, X,
      spatial_spec = sp, shift = shift)
    zero <- evaluate(0); r <- evaluate(.4)
    reference <- function(q) {
      u <- plogis(q[4]); rho <- -.8 + 1.7 * u; tau <- exp(q[3])
      H <- tau * (diag(degree) - rho * W)
      w <- q[5:8]; f <- if (center) w - mean(w) else w
      eta <- drop(X %*% q[1:2]) + f[group]
      if (restricted) eta <- drop(X %*% q[1:2]) +
        drop((diag(N) - tcrossprod(Q)) %*% f[group])
      rate <- -log(.01) / 3
      prior <- -.5 * sum(q[1:2]^2) / 2.5^2 - .5 * q[3] - rate * exp(-.5 * q[3]) +
        2 * log(u) + 3 * log1p(-u) + .5 * as.numeric(determinant(H)$modulus) -
        .5 * sum(w * drop(H %*% w))
      list(prior = prior, eta = eta)
    }
    ref <- reference(r$x); ref0 <- reference(zero$x)
    expect_equal(r$eta[, 1], ref$eta, tolerance = 1e-10)
    expect_equal(r$prior_lp - zero$prior_lp, ref$prior - ref0$prior, tolerance = 1e-10)
    expect_equal(r$spatial_log_normalizer,
      .5 * sum(log(degree)) - S * .5 * log(2 * pi) - lbeta(2, 3), tolerance = 1e-12)
    expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
    expect_true(r$verified)
  }
})

test_that("HSGP and multiscale HSGP densities share the dense spectral Gaussian model", {
  set.seed(192)
  N <- 20L
  coords <- cbind(runif(N), runif(N)); X <- cbind(1, rnorm(N)); y <- rnorm(N)
  basis <- spatial_bridge_basis(coords)
  normal <- function(mean = 0, sd = 1) list(range_prior =
    list(type = 'lognormal', meanlog = mean, sdlog = sd))
  cases <- list(
    hsgp = c(list(type = 'hsgp', coords = coords, m = 2L, c = 1.5, kernel = 0L), normal()),
    multiscale_hsgp = list(type = 'multiscale_hsgp', coords = coords, m = 2L, c = 1.5,
      local_range = normal(-1, .5), regional_range = normal(.7, .5)))
  for (type in names(cases)) for (restricted in c(FALSE, TRUE)) {
    sp <- cases[[type]]
    Q <- qr.Q(qr(X))
    if (restricted) sp$restriction_basis <- Q
    evaluate <- function(shift) tulpa:::cpp_test_field_coordinates(y, X,
      spatial_spec = sp, shift = shift)
    zero <- evaluate(0); r <- evaluate(.4)
    reference <- function(q) {
      prior <- -.5 * sum(q[1:2]^2) / 2.5^2
      eta <- drop(X %*% q[1:2]); field <- rep(0, N)
      offsets <- if (type == 'hsgp') 3L else c(3L, 9L)
      for (k in seq_along(offsets)) {
        off <- offsets[k]; logvar <- q[off]; logrange <- q[off + 1L]
        beta <- q[seq.int(off + 2L, length.out = 4L)]
        ell <- exp(logrange)
        spectrum <- exp(logvar) * 2 * pi * ell^2 * exp(-.5 * ell^2 * basis$eigen)
        field <- field + drop(basis$phi %*% (sqrt(spectrum) * beta))
        meanlog <- if (type == 'hsgp') 0 else c(-1, .7)[k]
        sdlog <- if (type == 'hsgp') 1 else .5
        prior <- prior + .5 * logvar - (-log(.01)) * exp(.5 * logvar) -
          .5 * ((logrange - meanlog) / sdlog)^2 - .5 * sum(beta^2)
      }
      if (restricted) field <- drop((diag(N) - tcrossprod(Q)) %*% field)
      list(prior = prior, eta = eta + field)
    }
    ref <- reference(r$x); ref0 <- reference(zero$x)
    expect_equal(r$eta[, 1], ref$eta, tolerance = 1e-10, info = type)
    expect_equal(r$prior_lp - zero$prior_lp, ref$prior - ref0$prior, tolerance = 1e-10)
    expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
    expect_true(r$verified)
    if (restricted) expect_equal(drop(crossprod(Q, r$eta[, 1] - X %*% r$x[1:2])),
      rep(0, ncol(Q)), tolerance = 1e-10)
  }
})

test_that("restriction bases are validated before sampler evaluation", {
  y <- 1:4; X <- matrix(1, 4, 1)
  sp <- list(type = 'hsgp', coords = cbind(1:4, c(0, 1, 0, 1)), m = 2L, c = 1.5, kernel = 0L,
    range_prior = list(type = 'lognormal', meanlog = 0, sdlog = 1),
    restriction_basis = matrix(1, 4, 1))
  expect_error(tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = sp), 'orthonormal')
  sp$restriction_basis <- matrix(.5, 3, 1)
  expect_error(tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = sp), 'dimensions')
})

test_that("NNGP covariance arms and repeated-location restriction match dense conditionals", {
  set.seed(193)
  S <- 5L; N <- 20L
  coords <- cbind(runif(S), runif(S))
  d <- data.frame(lon = coords[rep(1:S, each = 4L), 1],
    lat = coords[rep(1:S, each = 4L), 2])
  X <- cbind(1, rnorm(N)); y <- rnorm(N); Q <- qr.Q(qr(X))
  for (cov in c('exponential', 'matern32', 'matern52', 'gaussian', 'spherical')) {
    sp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(
      spatial_gp(~ lon + lat, nn = 4L, cov = if (grepl('matern', cov)) 'matern' else 'exponential',
        nu = if (cov == 'matern52') 2.5 else 1.5), d))
    sp$cov_type <- c(exponential = 0L, matern32 = 1L, matern52 = 4L,
      gaussian = 2L, spherical = 3L)[[cov]]
    sp$gp_parameterization <- 0L
    sp$parameterization <- 'centered'
    sp$range_prior <- list(type = 'uniform', lower = .001, upper = 100)
    sp$restriction_basis <- Q
    r <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = sp, shift = .25)
    zero <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = sp, shift = 0)
    locations <- sp$coords
    D <- as.matrix(dist(locations))
    kernel <- function(distance, sigma2, phi) {
      z <- distance / phi
      switch(cov, exponential = sigma2 * exp(-z),
        matern32 = sigma2 * (1 + sqrt(3) * z) * exp(-sqrt(3) * z),
        matern52 = sigma2 * (1 + sqrt(5) * z + 5 * z^2 / 3) * exp(-sqrt(5) * z),
        gaussian = sigma2 * exp(-z^2),
        spherical = sigma2 * ifelse(z < 1, 1 - 1.5 * z + .5 * z^3, 0))
    }
    reference <- function(q) {
      variance <- exp(q[3]); phi <- exp(q[4]); w <- q[5:9]
      K <- kernel(D, variance, phi)
      log_density <- 0
      for (j in seq_len(S)) {
        node <- sp$nn_order[j] + 1L
        previous <- if (j > 1) sp$nn_order[seq_len(j - 1L)] + 1L else integer()
        if (length(previous)) {
          C <- K[previous, previous, drop = FALSE] + diag(1e-8, length(previous))
          a <- solve(C, K[previous, node])
          mu <- sum(a * w[previous]); v <- max(1e-10, variance - sum(a * K[previous, node]))
        } else { mu <- 0; v <- variance + 1e-8 }
        log_density <- log_density + dnorm(w[node], mu, sqrt(v), log = TRUE)
      }
      prior <- -.5 * sum(q[1:2]^2) / 2.5^2 + .5 * q[3] -
        (-log(.01)) * exp(.5 * q[3]) + q[4] + log_density
      field <- w[sp$obs_to_loc + 1L]
      list(prior = prior, eta = drop(X %*% q[1:2] + (diag(N) - tcrossprod(Q)) %*% field))
    }
    ref <- reference(r$x); ref0 <- reference(zero$x)
    expect_equal(r$eta[, 1], ref$eta, tolerance = 1e-9, info = cov)
    expect_equal(r$prior_lp - zero$prior_lp, ref$prior - ref0$prior, tolerance = 1e-7, info = cov)
    expect_equal(r$gradient, r$numerical, tolerance = 2e-5, info = cov)
    expect_true(r$verified)
  }
})
