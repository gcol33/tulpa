# Per-row predictive variance of the linear predictor on the joint-driver
# single-arm entries (nngp / hsgp / spde / the spatiotemporal ones).
#
# var(eta_r | theta_k, y) = a_r' Sigma_k a_r, with a_r the row's loading vector
# and Sigma_k the inverse of the precision the cell's Laplace approximation is
# taken of. Every fit these entries produce used to carry `fitted_eta` alone, so
# the grid mixture behind WAIC / LOO / posterior_predict() held the across-cell
# spread and nothing within a cell.
#
# The arbiter is the cell's own stored precision (`store_Q = TRUE`) inverted in
# R against a loading matrix written from the model's layout, not read from the
# engine. On the sparse driver a large intrinsic field's sum-to-zero pin is left
# off the stored precision and folded in by Woodbury; TULPA_S2Z_DENSIFY_MAX = 0
# forces that fold on a field the default would densify, so the two storages of
# the same pin have to report the same variance.

.jev_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(nbr)))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(lengths(nbr)),
       n_spatial_units = n_s)
}

.jev_nn <- function(coords, k) {
  n <- nrow(coords)
  ord <- order(coords[, 1], coords[, 2])
  co <- coords[ord, , drop = FALSE]
  nn_idx <- matrix(0L, n, k)
  nn_dist <- matrix(0, n, k)
  for (i in 2:n) {
    d <- sqrt((co[1:(i - 1), 1] - co[i, 1])^2 + (co[1:(i - 1), 2] - co[i, 2])^2)
    nc <- min(length(d), k)
    o <- order(d)[1:nc]
    nn_idx[i, seq_len(nc)] <- o
    nn_dist[i, seq_len(nc)] <- d[o]
  }
  list(coords = co, nn_idx = nn_idx, nn_dist = nn_dist,
       nn_order = as.integer(ord - 1L), nn = k)
}

# diag(A Q_k^{-1} A') for every stored cell precision.
.jev_reference <- function(fit, A) {
  n_x <- fit$Q_csc_n
  t(vapply(seq_along(fit$Q_csc_p_per_grid), function(k) {
    L <- Matrix::sparseMatrix(i = fit$Q_csc_i_per_grid[[k]],
                              p = fit$Q_csc_p_per_grid[[k]],
                              x = fit$Q_csc_x_per_grid[[k]],
                              dims = c(n_x, n_x), index1 = FALSE)
    Q <- L + Matrix::t(L) - Matrix::Diagonal(n_x, Matrix::diag(L))
    V <- as.matrix(Matrix::solve(Q, t(A)))
    rowSums(A * t(V))
  }, numeric(nrow(A))))
}

.jev_st_icar <- function(n_s, n_t, N, seed) {
  set.seed(seed)
  adj <- .jev_chain_adj(n_s)
  s_idx <- sample.int(n_s, N, replace = TRUE)
  t_idx <- sample.int(n_t, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  eta <- 0.3 + 0.5 * X[, 2] + sin(s_idx / 3) + cos(t_idx / 2)
  args <- c(list(
    y = as.numeric(rpois(N, exp(eta))), n = rep(1L, N), X = X,
    re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
    spatial_idx = as.integer(s_idx),
    temporal_idx = as.integer(t_idx), n_times = n_t,
    tau_spatial_grid = c(1, 3), temporal_type = "rw1",
    tau_temporal_grid = c(2, 4), rho_temporal_grid = NULL, cyclic = FALSE,
    family = "poisson", phi = 1, max_iter = 100L, tol = 1e-10, n_threads = 1L),
    adj)
  A <- cbind(X,
             outer(s_idx, seq_len(n_s), `==`) * 1,
             outer(t_idx, seq_len(n_t), `==`) * 1)
  list(args = args, A = A)
}

test_that("st_icar reports the variance its dense precision implies", {
  skip_on_cran()
  fx <- .jev_st_icar(n_s = 12L, n_t = 6L, N = 150L, seed = 7L)
  fit <- do.call(cpp_nested_laplace_st_icar,
                 modifyList(fx$args, list(store_Q = TRUE)))
  V <- fit$fitted_eta_var
  expect_true(is.matrix(V))
  expect_equal(dim(V), c(length(fit$log_marginal), nrow(fx$A)))
  expect_true(all(is.finite(V)) && all(V > 0))
  expect_equal(V, .jev_reference(fit, fx$A), tolerance = 1e-8)
})

test_that("a folded sum-to-zero pin gives the variance of the stored one", {
  skip_on_cran()
  fx <- .jev_st_icar(n_s = 60L, n_t = 10L, N = 400L, seed = 13L)
  args <- modifyList(fx$args, list(store_Q = TRUE, force_sparse = TRUE))

  withr::local_envvar(TULPA_S2Z_DENSIFY_MAX = NA)
  stored <- do.call(cpp_nested_laplace_st_icar, args)
  # The intercept and the two intrinsic fields' levels are aliased in eta and
  # separated only by the fixed-effect ridge, so this precision's condition
  # number is about 1e8 and a relative error near eps * 1e8 = 2e-8 is what any
  # factorization of it carries.
  expect_equal(stored$fitted_eta_var, .jev_reference(stored, fx$A),
               tolerance = 1e-6)

  withr::local_envvar(TULPA_S2Z_DENSIFY_MAX = "0")
  folded <- do.call(cpp_nested_laplace_st_icar, args)
  # The fold is what this run exercises: its stored precision no longer holds
  # the pin, so inverting it alone gives a different answer.
  expect_false(isTRUE(all.equal(folded$Q_csc_x_per_grid,
                                stored$Q_csc_x_per_grid)))
  expect_equal(folded$fitted_eta_var, stored$fitted_eta_var, tolerance = 1e-10)
})

.jev_nngp <- function(seed = 5L) {
  set.seed(seed)
  # A regular lattice at short ranges keeps the precision's condition number
  # near 200. Near-coincident locations, or a range long against the spacing,
  # take it past 1e11, where independent dense and sparse inverses in R already
  # disagree with each other in the sixth digit and can referee nothing finer.
  n_g <- 40L
  co <- as.matrix(expand.grid(x = (1:8) / 8, y = (1:5) / 5))
  nb <- .jev_nn(co, 8L)
  N <- 2L * n_g
  s_idx <- rep(seq_len(n_g), 2L)
  X <- matrix(1, N, 1)
  args <- list(
    y = as.numeric(rbinom(N, 1L, 0.5)), n = rep(1L, N), X = X,
    re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
    spatial_idx = as.integer(s_idx), coords = nb$coords,
    nn_idx = nb$nn_idx, nn_dist = nb$nn_dist, nn_order = nb$nn_order,
    n_spatial = n_g, nn = nb$nn,
    sigma2_grid = c(0.5, 1, 0.5, 1), phi_gp_grid = c(0.2, 0.2, 0.3, 0.3),
    cov_type = 0L, family = "binomial", phi = 1,
    max_iter = 100L, tol = 1e-10, n_threads = 1L)
  list(args = args, A = cbind(X, outer(s_idx, seq_len(n_g), `==`) * 1))
}

test_that("nngp reports the variance its precision implies, one solve per location", {
  skip_on_cran()
  fx <- .jev_nngp()
  fit <- do.call(cpp_nested_laplace_nngp,
                 modifyList(fx$args, list(store_Q = TRUE)))
  V <- fit$fitted_eta_var
  expect_equal(dim(V), c(4L, nrow(fx$A)))
  expect_true(all(is.finite(V)) && all(V > 0))
  expect_equal(V, .jev_reference(fit, fx$A), tolerance = 1e-10)
  # The two replicates of every location share a loading vector, and so share
  # the value exactly.
  n_g <- fx$args$n_spatial
  expect_identical(V[, seq_len(n_g)], V[, n_g + seq_len(n_g)])
})

test_that("hsgp carries a per-row variance that follows the basis row", {
  skip_on_cran()
  set.seed(9L)
  N <- 60L; M <- 10L
  phi_basis <- matrix(rnorm(N * M), N, M)
  phi_basis[31:60, ] <- phi_basis[1:30, ]
  X <- cbind(1, rep(rnorm(30), 2L))
  fit <- cpp_nested_laplace_hsgp(
    y = as.numeric(rbinom(N, 1L, 0.5)), n = rep(1L, N), X = X,
    re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
    phi_basis = phi_basis,
    lambda_eig = sort(abs(rnorm(M)) + 0.1, decreasing = TRUE),
    sigma2_grid = c(0.5, 1), lengthscale_grid = c(0.5, 1),
    family = "binomial", phi = 1, max_iter = 100L, tol = 1e-10,
    n_threads = 1L)
  V <- fit$fitted_eta_var
  expect_equal(dim(V), c(2L, N))
  expect_true(all(is.finite(V)) && all(V > 0))
  expect_identical(V[, 1:30], V[, 31:60])
  # The variance moves with the cell's spectral scaling.
  expect_false(isTRUE(all.equal(V[1, ], V[2, ])))
})

test_that("a resumed joint-driver grid reproduces fitted_eta_var", {
  skip_on_cran()
  fx <- .jev_nngp(seed = 17L)
  path <- tempfile(fileext = ".ckpt")
  on.exit(unlink(path), add = TRUE)
  args <- modifyList(fx$args, list(checkpoint_path = path))
  first <- do.call(cpp_nested_laplace_nngp, args)
  resumed <- do.call(cpp_nested_laplace_nngp, args)
  expect_identical(resumed$fitted_eta_var, first$fitted_eta_var)
  expect_identical(resumed$log_marginal, first$log_marginal)
})
