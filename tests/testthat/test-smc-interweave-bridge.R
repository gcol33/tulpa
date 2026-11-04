test_that("NNGP interweaving preserves each transformed bridge conditional", {
  skip_if_not_slow()
  load_cpp_fixture("smc_normalizer")
  d <- data.frame(x = c(0, 0.4, 1), y = c(0.1, 0.8, 0))
  sp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(spatial_gp(~x+y, nn = 2L), d))
  ord <- sp$nn_order + 1L
  D <- as.matrix(dist(sp$coords))[ord, ord]
  initial_z <- (0.5 * (1:3) - 0.7)[ord]
  field <- as.numeric(t(chol(exp(-D) + diag(1e-8, 3))) %*% initial_z)
  ls <- seq(-16, 8, by = 0.08); lr <- seq(-10, 16, by = 0.08)
  rate_s <- -log(sp$sigma2_prior_alpha) / sp$sigma2_prior_U
  rate_r <- -log(sp$phi_prior_alpha) * sp$phi_prior_U
  terms <- lapply(lr, function(r) vapply(ls, function(s) {
    L <- t(chol(exp(s) * exp(-D / exp(r)) + diag(1e-8, 3)))
    z <- numeric(3); z[ord] <- forwardsolve(L, field)
    v <- numeric(3); v[ord] <- forwardsolve(L, rep(1, 3))
    q <- -0.5 * sum((c(s, r, z) - 0.2)^2) - 0.35 * (s - z[1])^2
    p <- log(rate_s) + s / 2 - log(2) - rate_s * exp(s / 2) +
      log(rate_r) - r - rate_r * exp(-r) - sum(z^2) / 2
    c(q = q, p = p, logdet = sum(log(diag(L))),
      Aq = 1 + sum(v^2) + 0.7 * v[1]^2,
      Bq = 0.2 + sum(v * (z - 0.2)) - 0.7 * v[1] * (s - z[1]),
      Ap = 1 + sum(v^2), Bp = sum(v * z))
  }, numeric(7)))
  for (level in c(FALSE, TRUE)) for (power in c(0, 0.4, 1)) {
    lp <- vapply(terms, function(t)
      (1 - power) * t["q", ] + power * t["p", ] - t["logdet", ], numeric(length(ls)))
    if (level) {
      lp <- lp + vapply(terms, function(t) {
        A <- (1 - power) * t["Aq", ] + power * t["Ap", ]
        B <- (1 - power) * t["Bq", ] + power * t["Bp", ]
        B^2 / (2 * A) - log(A) / 2
      }, numeric(length(ls)))
    }
    w <- exp(lp - max(lp)); w <- w / sum(w)
    mu <- c(sum(rowSums(w) * ls), sum(colSums(w) * lr))
    sd <- sqrt(c(sum(rowSums(w) * (ls - mu[1])^2),
                  sum(colSums(w) * (lr - mu[2])^2)))
    draw <- cpp_test_nngp_interweave(25000L, power, sp, level)[-seq_len(1000L), ]
    expect_lt(max(abs(colMeans(draw[, 1:2]) - mu) / sd), 0.07)
    for (i in seq(1, nrow(draw), length.out = 20)) {
      L <- t(chol(exp(draw[i, 1]) * exp(-D / exp(draw[i, 2])) + diag(1e-8, 3)))
      signal <- as.numeric(L %*% draw[i, ord + 2L]) + if (level) draw[i, 6] else 0
      expect_equal(signal, field, tolerance = 1e-9)
    }
  }
})

test_that("physical NNGP field HMC preserves the exact bridge conditional", {
  skip_if_not_slow()
  load_cpp_fixture("smc_normalizer")
  d <- data.frame(x = c(0, 0.4, 1), y = c(0.1, 0.8, 0))
  sp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(spatial_gp(~x+y, nn = 2L), d))
  ord <- sp$nn_order + 1L
  D <- as.matrix(dist(sp$coords))
  K <- exp(0.4) * exp(-D / exp(-0.6)) + diag(1e-8, 3)
  T <- diag(4)
  T[ord + 1L, ord + 1L] <- solve(t(chol(K[ord, ord])))
  v <- T[2L, ]
  Qq <- crossprod(T) + 0.7 * tcrossprod(v)
  bq <- 0.2 * colSums(T) + 0.7 * 0.4 * v
  H <- cbind(1, outer(sp$obs_to_loc + 1L, 1:3, "==") * 1)
  Qp <- diag(4); Qp[-1L, -1L] <- solve(K)
  Qp <- Qp + crossprod(H) / 0.25
  bp <- as.numeric(crossprod(H, c(1, -0.4, 0.7)) / 0.25)
  for (power in c(0, 0.4, 1)) {
    V <- solve((1 - power) * Qq + power * Qp)
    mu <- as.numeric(V %*% ((1 - power) * bq + power * bp))
    got <- cpp_test_nngp_field_block(40000L, power, sp)
    draw <- got$draws[-seq_len(2000L), ]
    expect_lt(got$gradient_error, 1e-5)
    expect_equal(got$log_variance, 0.4)
    expect_equal(got$log_range, -0.6)
    expect_lt(max(abs(colMeans(draw) - mu) / sqrt(diag(V))), 0.08)
    expect_lt(max(abs(cov(draw) - V) / sqrt(outer(diag(V), diag(V)))), 0.07)
  }
})
