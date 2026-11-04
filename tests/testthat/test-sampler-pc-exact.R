pc_exact_log_sd <- function(x, U, alpha) {
  rate <- -log(alpha) / U
  log(rate) + x - rate * exp(x)
}

test_that("flat hyperpriors keep only stated areal scale priors", {
  sp <- list(type = "icar", adjacency = matrix(c(0, 1, 1, 0), 2),
             spatial_idx = 1:2)
  block <- .spatial_spec_to_nl_prior(sp)
  expect_null(.hp_axis_prior("tau", block, hyperprior = "flat")$fn)
  sp$prior_sigma <- c(0.8, 0.1)
  block <- .spatial_spec_to_nl_prior(sp)
  expect_true(is.function(.hp_axis_prior("tau", block, hyperprior = "flat")$fn))
})

pc_exact_mvn <- function(y, V) {
  R <- chol(V)
  z <- forwardsolve(t(R), y)
  -length(y) * log(2 * pi) / 2 - sum(log(diag(R))) - sum(z^2) / 2
}

test_that("PC Gaussian-field scale updates agree with exact conditional quadrature", {
  load_cpp_fixture("pc_scale")
  for (rank in c(0L, 1L, 4L, 15L)) {
    quad <- if (rank == 0L) 0 else 0.03 * rank
    x <- seq(-24, 4, length.out = 28001)
    lp <- pc_exact_log_sd(x, 0.8, 0.1) - rank * x - 0.5 * quad * exp(-2 * x)
    w <- exp(lp - max(lp)); w <- w / sum(w)
    mu <- sum(w * x); sd <- sqrt(sum(w * (x - mu)^2))
    set.seed(60 + rank)
    draws <- cpp_test_pc_scale(60000L, quad, rank, 0.8, 0.1)[-seq_len(1000L)]
    expect_lt(abs(mean(draws) - mu) / sd, 0.03)
    expect_lt(abs(stats::sd(draws) / sd - 1), 0.03)
  }
})

test_that("BYM2 mixing weights follow the continuous exact conditional", {
  load_cpp_fixture("pc_scale")
  field <- c(-0.7, 0.1, 0.6); iid <- c(0.4, -0.5, 0.8)
  for (informative in c(FALSE, TRUE)) {
    omega <- if (informative) c(3, 5, 8) else numeric(3)
    residual <- if (informative) c(-2, 1, 5) else numeric(3)
    lp <- function(r) {
      u <- outer(sqrt(r + 1e-10), field) + outer(sqrt(1 - r + 1e-10), iid)
      as.numeric(u %*% residual - (u^2) %*% (omega / 2)) + dbeta(r, 0.8, 1.3, log = TRUE)
    }
    Z <- integrate(function(r) exp(lp(r)), 0, 1, rel.tol = 1e-10)$value
    mu <- integrate(function(r) r * exp(lp(r)), 0, 1, rel.tol = 1e-10)$value / Z
    v <- integrate(function(r) (r - mu)^2 * exp(lp(r)), 0, 1, rel.tol = 1e-10)$value / Z
    set.seed(74)
    draw <- cpp_test_bym2_rho(60000L, field, iid, omega, residual, 0.8, 1.3)[-1:-1000]
    expect_lt(abs(mean(draw) - mu) / sqrt(v), 0.03)
    expect_lt(abs(sd(draw) / sqrt(v) - 1), 0.03)
    expect_gt(length(unique(draw)), 50000L)
  }
})

test_that("BYM2 PC scales match exact conditionals through the weak-data limit", {
  load_cpp_fixture("pc_scale")
  field <- c(1, -1); iid <- c(0.2, 0.3)
  v <- sqrt(0.4 + 1e-10) * field + sqrt(0.6 + 1e-10) * iid
  cases <- list(c(0, 0), c(1e-20, 0), c(1, 1), c(1, 3))
  for (case in cases) {
    omega <- rep(case[1], 2); residual <- case[2] * c(3, -2)
    A <- sum(residual * v) - (-log(0.1) / 0.8); B <- sum(omega * v^2)
    density <- function(x) exp(A * x - B * x^2 / 2)
    Z <- integrate(density, 0, Inf, rel.tol = 1e-10)$value
    mu <- integrate(function(x) x * density(x), 0, Inf, rel.tol = 1e-10)$value / Z
    variance <- integrate(function(x) (x - mu)^2 * density(x), 0, Inf,
                          rel.tol = 1e-10)$value / Z
    set.seed(78)
    draw <- cpp_test_bym2_sigma(60000L, field, iid, omega, residual, 0.8, 0.1)
    expect_true(all(is.finite(draw) & draw > 0))
    expect_lt(abs(mean(draw) - mu) / sqrt(variance), 0.03)
    expect_lt(abs(sd(draw) / sqrt(variance) - 1), 0.03)
  }
})

pc_areal_fixture <- function(type) {
  J <- 4L
  W <- matrix(0, J, J)
  W[cbind(1:3, 2:4)] <- W[cbind(2:4, 1:3)] <- 1
  unit <- rep(1:J, each = 6L)
  y <- rep(c(-0.65, 0.3, 0.5, -0.15), each = 6L) +
    rep(c(-0.4, -0.2, 0, 0.1, 0.15, 0.35), J)
  X <- matrix(1, length(y), 1)
  cols <- lapply(1:J, function(i) which(W[i, ] != 0) - 1L)
  sp <- list(type = type, spatial_idx = as.integer(unit), n_spatial_units = J,
             adj_row_ptr = as.integer(c(0, cumsum(lengths(cols)))),
             adj_col_idx = as.integer(unlist(cols)),
             n_neighbors = as.integer(rowSums(W)),
             sigma_prior_U = 0.8, sigma_prior_alpha = 0.1)
  Q <- diag(rowSums(W)) - W
  C <- diag(J) - 1 / J
  Cicar <- C %*% solve(Q + 1 / J) %*% C
  if (type == "car_proper") {
    sp$adj_eigenvalues <- eigen(W / sqrt(outer(rowSums(W), rowSums(W))),
                                symmetric = TRUE)$values
    sp$rho_lower <- 0
    sp$rho_upper <- 0.95
  }
  sp$scale_factor <- 1
  A <- outer(unit, 1:J, "==") * 1
  list(y = y, X = X, W = W, Q = Q, Cicar = Cicar, A = A, sp = sp)
}

test_that("areal sampler entries reject an infinite PC scale anchor", {
  f <- pc_areal_fixture("icar")
  f$sp$sigma_prior_U <- Inf
  expect_error(tulpa_sample_glmm(f$y, NULL, f$X, "gaussian", "hmc",
    spatial_spec = f$sp, control = list(n_iter = 2L, warmup = 1L)),
    "PC prior anchors")
  expect_error(cpp_pg_binomial_gibbs_spatial(1L, 2L, matrix(1, 1, 1),
    0L, 0L, 1L, 2L, list(2L, 1L), c(1L, 1L),
    prior_sigma_spatial_U = Inf, n_iter = 2L, n_warmup = 1L),
    "prior_sigma_spatial_U")
})

pc_areal_exact <- function(f, h = 0.06) {
  ls <- seq(-13, 3, by = h)
  type <- f$sp$type
  u <- if (type == "icar") 0.5 else (seq_len(160L) - 0.5) / 160L
  B <- 0.16 * diag(length(f$y)) + tcrossprod(f$X)
  lp <- vapply(u, function(r) vapply(ls, function(x) {
    K <- switch(type, icar = f$Cicar,
      bym2 = r * f$Cicar + (1 - r) * diag(4),
      car_proper = solve(diag(rowSums(f$W)) - 0.95 * r * f$W))
    pc_exact_mvn(f$y, B + exp(2 * x) * f$A %*% K %*% t(f$A)) +
      pc_exact_log_sd(x, f$sp$sigma_prior_U, f$sp$sigma_prior_alpha)
  }, 0.0), numeric(length(ls)))
  w <- exp(lp - max(lp)); w <- rowSums(w) / sum(w)
  mu <- sum(w * ls)
  c(mean = mu, sd = sqrt(sum(w * (ls - mu)^2)))
}

test_that("areal sampler scales agree with exact Gaussian quadrature", {
  skip_if_not_slow()
  for (type in c("icar", "car_proper", "bym2")) {
    f <- pc_areal_fixture(type)
    exact <- pc_areal_exact(f)
    fine <- pc_areal_exact(f, h = 0.03)
    expect_lt(max(abs(exact - fine)), 1e-5)
    d <- data.frame(y = f$y, unit = factor(f$sp$spatial_idx))
    spatial_spec <- list(type = type, adjacency = f$W,
                         prior_sigma = c(0.8, 0.1), scale_factor = 1,
                         rho_bounds = c(lower = 0, upper = 0.95))
    fit <- tulpa(y ~ 1 + spatial(unit), data = d, family = "gaussian", mode = "hmc",
      phi = 0.16, beta_prior = prior_normal(0, 1), spatial = spatial_spec,
      control = list(n_iter = 4000L, warmup = 2000L, n_chains = 4L,
                     seed = 71L))
    x <- if (type == "bym2") fit$draws[, "log_sigma_spatial"] else
      -0.5 * fit$draws[, "log_tau_spatial"]
    expect_lt(abs(mean(x) - fine[["mean"]]) / fine[["sd"]], 0.12,
              label = type)
    message(sprintf("PC %s mean %.8f exact %.8f error_sd %.6f", type,
                    mean(x), fine[["mean"]],
                    (mean(x) - fine[["mean"]]) / fine[["sd"]]))
  }
})

# A two-node intrinsic field has one independent coordinate. The binomial
# likelihood and the normal intercept prior are integrated on a rectangle;
# integrating the PC scale out analytically leaves a one-dimensional integral.
pc_temporal_exact <- function(cyclic = FALSE, h = 0.025) {
  a <- seq(-3, 3, by = h)
  b <- seq(-6, 6, by = h)
  mult <- if (cyclic) 8 else 4
  log_marginal <- vapply(a, function(x) {
    if (x == 0) x <- h / 4
    z <- integrate(function(ls) {
      s <- exp(ls)
      exp(dnorm(x, 0, s / sqrt(mult), log = TRUE) +
            pc_exact_log_sd(ls, 1, 0.01))
    }, -22, 3, rel.tol = 1e-9)$value
    log(z)
  }, 0.0)
  lp <- outer(a, b, function(x, beta)
    dbinom(9, 20, plogis(beta + x), log = TRUE) +
    dbinom(13, 20, plogis(beta - x), log = TRUE)) +
      outer(log_marginal, dnorm(b, 0, 1, log = TRUE), "+")
  w <- exp(lp - max(lp)); w <- w / sum(w)
  c(mean_a = sum(rowSums(w) * a), sd_a = sqrt(sum(rowSums(w) * a^2) -
      sum(rowSums(w) * a)^2), mean_beta = sum(colSums(w) * b))
}

test_that("temporal PG intrinsic fields agree with exact binomial quadrature", {
  skip_if_not_slow()
  for (cyclic in c(FALSE, TRUE)) {
    exact <- pc_temporal_exact(cyclic)
    fine <- pc_temporal_exact(cyclic, h = 0.0125)
    expect_lt(max(abs(exact - fine)), 0.003)
    d <- data.frame(t = 1:2, y = c(9L, 13L), n = 20L)
    temporal <- if (cyclic) temporal_multiscale("t", trend = "none", seasonal = 2L)
      else temporal_multiscale("t", trend = "rw1")
    set.seed(81)
    fit <- tulpa(cbind(y, n-y) ~ 1, data = d, family = "binomial",
      temporal = temporal, mode = "gibbs", beta_prior = prior_normal(0, 1),
      control = list(n_iter = 50000L, warmup = 10000L))
    field <- draws_block(fit, if (cyclic) "seasonal" else "trend")[, 1]
    expect_lt(abs(mean(field) - fine[["mean_a"]]) / fine[["sd_a"]], 0.08)
    expect_lt(abs(sd(field) / fine[["sd_a"]] - 1), 0.08)
    message(sprintf("PC temporal cyclic=%s mean_error_sd %.6f sd_ratio %.6f",
                    cyclic, (mean(field) - fine[["mean_a"]]) / fine[["sd_a"]],
                    sd(field) / fine[["sd_a"]]))
  }
})

pc_temporal_proper_exact <- function(ar1, nq = 96L) {
  # Golub-Welsch nodes for expectation under a standard normal.
  H <- matrix(0, nq, nq)
  H[cbind(seq_len(nq - 1L), 2:nq)] <- sqrt(seq_len(nq - 1L))
  H <- H + t(H)
  ee <- eigen(H, symmetric = TRUE)
  z <- ee$values; weight <- ee$vectors[1, ]^2
  z1 <- rep(z, each = nq); z2 <- rep(z, nq)
  weights <- as.numeric(outer(weight, weight))
  ls <- seq(-14, 2, by = 0.05)
  rhos <- if (ar1) (seq_len(80L) - 0.5) / 40 - 1 else 0
  rho_weight <- if (ar1) dbeta((rhos + 1) / 2, 2, 2) else 1
  likelihood <- vapply(ls, function(s) {
    sigma2 <- exp(2 * s)
    sum(vapply(seq_along(rhos), function(i) {
      r <- rhos[i]
      plus <- sqrt((2 + sigma2 / (1 - r)) / 2) * z1
      minus <- sqrt(sigma2 / (2 * (1 + r))) * z2
      sum(weights * dbinom(9, 20, plogis(plus + minus)) *
            dbinom(13, 20, plogis(plus - minus))) * rho_weight[i]
    }, 0.0))
  }, 0.0)
  w <- likelihood * exp(pc_exact_log_sd(ls, 1, 0.01)); w <- w / sum(w)
  mu <- sum(w * ls)
  c(mean = mu, sd = sqrt(sum(w * (ls - mu)^2)))
}

test_that("temporal PG proper scales agree with exact binomial quadrature", {
  skip_if_not_slow()
  for (ar1 in c(FALSE, TRUE)) {
    exact <- pc_temporal_proper_exact(ar1)
    fine <- pc_temporal_proper_exact(ar1, nq = 128L)
    expect_lt(max(abs(exact - fine)), 0.005)
    d <- data.frame(t = 1:2, y = c(9L, 13L), n = 20L)
    set.seed(90)
    fit <- tulpa(cbind(y, n-y) ~ 1, data = d, family = "binomial",
      temporal = temporal_multiscale("t", trend = "none",
        short_term = if (ar1) "ar1" else "iid"), mode = "gibbs",
      beta_prior = prior_normal(0, 1),
      control = list(n_iter = 80000L, warmup = 10000L))
    x <- fit$draws[, "log_sigma2_short"] / 2
    expect_lt(abs(mean(x) - fine[["mean"]]) / fine[["sd"]], 0.1)
    expect_lt(abs(sd(x) / fine[["sd"]] - 1), 0.1)
    message(sprintf("PC temporal ar1=%s scale_mean_error_sd %.6f scale_sd_ratio %.6f",
                    ar1, (mean(x) - fine[["mean"]]) / fine[["sd"]],
                    sd(x) / fine[["sd"]]))
  }
})

test_that("a singleton PG AR1 field has its stationary prior variance", {
  skip_if_not_slow()
  set.seed(91)
  fit <- tulpa:::cpp_pg_binomial_gibbs_temporal(0L, 0L, matrix(1, 1, 1),
    1L, 0L, 1L, 1L, 0L, 0L, 1L, n_iter = 100000L, n_warmup = 10000L,
    prior_beta_sd = 1, verbose = FALSE)
  z <- fit$short_term[, 1] * sqrt(1 - fit$rho_short^2) / fit$sigma_short
  expect_lt(abs(mean(z)), 0.05)
  expect_lt(abs(sd(z) - 1), 0.05)
  expect_lt(abs(mean(fit$sigma_short) * (-log(0.01)) - 1), 0.05)
})

test_that("GP prediction uses the fitted covariance nugget", {
  xy <- rbind(c(0, 0), c(1, 0))
  target <- matrix(c(1e-5, 0), 1)
  field <- matrix(c(0.3, -0.2), 1)
  s2 <- 1e-5; range <- 2
  C <- s2 * exp(-as.matrix(dist(xy)) / range) + diag(1e-8, 2)
  c <- s2 * exp(-sqrt(rowSums((xy - matrix(target, 2, 2, byrow = TRUE))^2)) / range)
  B <- solve(C, c)
  v <- s2 + 1e-8 - sum(c * B)
  for (z in c(0, 1)) {
    got <- tulpa:::cpp_gp_field_rows(target, xy, field, 0L, s2, range,
                                     2L, 0L, matrix(z, 1, 1))
    expect_equal(as.numeric(got), sum(B * field) + z * sqrt(v), tolerance = 1e-12)
  }
})
