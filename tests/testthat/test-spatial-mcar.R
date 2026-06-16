# Inline correlated areal fields (separable MCAR) via a single bar:
# spatial(graph, ~ 1 + x | cell) (gcol33/tulpa#89). The per-cell coefficient
# vector (u_c, s_c) shares a cross-covariance Sigma; covariance Sigma (x) Q^-1.

.mcar_grid_adj <- function(nx, ny) {
  n <- nx * ny
  W <- matrix(0L, n, n)
  id <- function(i, j) (j - 1L) * nx + i
  for (i in seq_len(nx)) for (j in seq_len(ny)) {
    if (i < nx) { a <- id(i, j); b <- id(i + 1L, j); W[a, b] <- W[b, a] <- 1L }
    if (j < ny) { a <- id(i, j); b <- id(i, j + 1L); W[a, b] <- W[b, a] <- 1L }
  }
  W
}

# Draw correlated fields (u, s) with cov Sigma (x) Q^-1 via the LMC factorization
# u = L11 z1, s = L21 z1 + L22 z2 with z1, z2 iid smooth CAR (rho = 0.99, proper
# so invertible, near-intrinsic), L = chol(Sigma) lower. Centre each field.
.mcar_sim <- function(adj, Sigma) {
  n <- nrow(adj)
  Qp <- diag(rowSums(adj)) - 0.99 * adj
  U  <- chol(Qp)
  z1 <- backsolve(U, stats::rnorm(n)); z1 <- z1 - mean(z1)
  z2 <- backsolve(U, stats::rnorm(n)); z2 <- z2 - mean(z2)
  L  <- t(chol(Sigma))
  u <- L[1, 1] * z1
  s <- L[2, 1] * z1 + L[2, 2] * z2
  list(u = u - mean(u), s = s - mean(s))
}

test_that("spatial(~ ... | cell) builds one coupled MCAR block", {
  adj <- .mcar_grid_adj(4L, 4L)
  f <- spatial(graph = adj, formula = ~ 1 + x | cell)
  expect_true(f$correlated)
  d <- data.frame(cell = rep(seq_len(16L), each = 2L), x = rnorm(32L))
  blocks <- tulpa:::.spatial_field_blocks(f, d)
  expect_length(blocks, 1L)
  expect_identical(blocks[[1]]$type, "mcar")
  expect_identical(blocks[[1]]$n_fields, 2L)
  expect_length(blocks[[1]]$field_weight, 2L)
  expect_setequal(blocks[[1]]$field_names, c("cell.Intercept", "cell.x"))

  # Correlated + proper is out of scope.
  expect_error(spatial(graph = adj, formula = ~ 1 | cell, proper = TRUE),
               "not in scope")
})

test_that("MCAR recovers the correlated fields and Sigma vs simulated truth", {
  skip_on_cran()

  nx <- ny <- 10L
  adj <- .mcar_grid_adj(nx, ny)
  n_s <- nx * ny
  sig1 <- 1.0; sig2 <- 0.8; rho_true <- 0.6
  Sigma <- matrix(c(sig1^2, rho_true * sig1 * sig2,
                    rho_true * sig1 * sig2, sig2^2), 2, 2)

  cor_u <- cor_s <- rho_med <- numeric(0)
  rho_cover <- logical(0)
  for (seed in 1:3) {
    set.seed(seed)
    fld <- .mcar_sim(adj, Sigma)
    n_per <- 40L
    cell <- rep(seq_len(n_s), each = n_per)
    N <- length(cell)
    x <- rnorm(N)
    eta <- 0.3 + fld$u[cell] + x * fld$s[cell]
    y <- rnorm(N, eta, 0.4)
    d <- data.frame(y = y, x = x, cell = cell)

    fit <- suppressWarnings(tulpa(
      y ~ spatial(graph = adj, formula = ~ 1 + x | cell),
      data = d, family = "gaussian", mode = "laplace",
      control = list(n_draws = 200L)))

    expect_s3_class(fit, "tulpa_spatial_field_fit")
    expect_true(isTRUE(fit$correlated))
    # Strong, well-identified data: the outer Sigma grid is the mode-centred CCD
    # over the log-Cholesky coordinates (1 + 2*3 + 2^3 = 15 nodes for p = 2), not
    # the fixed log-Cholesky tensor -- the documented follow-up. The outer
    # Pareto-k is then scored (was NA when the fit declined the CCD).
    expect_identical(fit$integration, "ccd")
    expect_false(is.na(fit$pareto_k))
    u_hat <- fit$spatial_fields[["cell.Intercept"]]$mean
    s_hat <- fit$spatial_fields[["cell.x"]]$mean
    cor_u <- c(cor_u, abs(cor(u_hat, fld$u)))
    cor_s <- c(cor_s, abs(cor(s_hat, fld$s)))

    rq <- fit$mcar_summary[["rho_12"]]$q
    rho_med <- c(rho_med, rq[2L])
    rho_cover <- c(rho_cover, rq[1L] <= rho_true && rho_true <= rq[3L])
  }

  # Both fields recover every seed.
  expect_true(all(cor_u > 0.7))
  expect_true(all(cor_s > 0.7))
  # rho recovers a clearly positive cross-correlation (true 0.6), and the
  # marginalized CI covers the truth in the majority of seeds.
  expect_gt(median(rho_med), 0.3)
  expect_gte(sum(rho_cover), 2L)
})

test_that("MCAR mode-centred CCD scales to p = 3 (general p > 2)", {
  skip_on_cran()
  set.seed(3)
  nx <- ny <- 8L
  adj <- .mcar_grid_adj(nx, ny)
  n_s <- nx * ny
  # Lower Cholesky factor of the 3-field cross-covariance; Sigma = L L'.
  L3 <- matrix(c(1.0,  0.0, 0.0,
                 0.4,  0.8, 0.0,
                 -0.2, 0.3, 0.7), 3, 3, byrow = TRUE)
  Sigma3 <- L3 %*% t(L3)
  R3 <- cov2cor(Sigma3)

  # Three correlated near-intrinsic CAR fields via the LMC factorization.
  Qp <- diag(rowSums(adj)) - 0.99 * adj
  U  <- chol(Qp)
  Z  <- sapply(1:3, function(k) { z <- backsolve(U, rnorm(n_s)); z - mean(z) })
  Fl <- Z %*% t(L3)
  Fl <- apply(Fl, 2L, function(col) col - mean(col))

  cell <- rep(seq_len(n_s), each = 30L)
  N <- length(cell)
  x1 <- rnorm(N); x2 <- rnorm(N)
  eta <- 0.2 + Fl[cell, 1] + x1 * Fl[cell, 2] + x2 * Fl[cell, 3]
  y <- rnorm(N, eta, 0.4)

  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 + x1 + x2 | cell),
    data = data.frame(y = y, x1 = x1, x2 = x2, cell = cell),
    family = "gaussian", mode = "laplace", control = list(n_draws = 200L)))

  expect_s3_class(fit, "tulpa_spatial_field_fit")
  # CCD over the 6 log-Cholesky axes -- a polynomial node count where the fixed
  # tensor would be thousands of cells.
  expect_identical(fit$integration, "ccd")
  expect_identical(ncol(fit$theta_grid), 6L)
  # All three cross-correlations recover with a CI covering the truth.
  for (ij in list(c(1L, 2L), c(1L, 3L), c(2L, 3L))) {
    nm <- sprintf("rho_%d%d", ij[1L], ij[2L])
    q  <- fit$mcar_summary[[nm]]$q
    rho_true <- R3[ij[1L], ij[2L]]
    expect_true(q[1L] <= rho_true && rho_true <= q[3L],
                info = sprintf("%s CI (%.3f, %.3f) should cover %.3f",
                               nm, q[1L], q[3L], rho_true))
  }
})

test_that("MCAR fit print reports the Sigma cross-covariance (rho)", {
  skip_on_cran()
  set.seed(11)
  adj <- .mcar_grid_adj(8L, 8L)
  n_s <- 64L
  Sigma <- matrix(c(1, 0.5, 0.5, 0.7), 2, 2)
  fld <- .mcar_sim(adj, Sigma)
  cell <- rep(seq_len(n_s), each = 25L)
  x <- rnorm(length(cell))
  y <- rnorm(length(cell), 0.2 + fld$u[cell] + x * fld$s[cell], 0.4)
  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 + x | cell),
    data = data.frame(y = y, x = x, cell = cell), family = "gaussian",
    mode = "laplace", control = list(n_draws = 200L)))
  expect_false(is.null(fit$mcar_summary[["rho_12"]]))
  out <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(out, "MCAR")
  expect_match(out, "rho_12")
})
