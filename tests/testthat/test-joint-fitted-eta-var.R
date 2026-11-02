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
# off the stored precision and folded in by Woodbury; TULPA_S2Z_DENSIFY_MAX
# selects the storage (0 forces the fold at any size, a large value stores the
# pin at any size), so the two storages of the same pin have to report the same
# variance.

skip_on_cran()

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

# diag(A Q_k^{-1} A') for every stored cell precision. `fold` adds back a pin
# the sparse driver keeps off the stored precision whatever its size (an RW
# chain's, S2ZStorage::Fold): `idx` its latent coordinates, `coef` the per-cell
# coefficient tau_k / n of its 11'.
# Var(A x) under the stored precision Q at each cell, conditioned by kriging on
# the constraint rows `constr` (index vectors, every weight 1): the fields'
# levels are hard constraints on the inner solve (gcol33/tulpa#1053).
.jev_reference <- function(fit, A, fold = NULL, constr = NULL) {
  n_x <- fit$Q_csc_n
  t(vapply(seq_along(fit$Q_csc_p_per_grid), function(k) {
    L <- Matrix::sparseMatrix(i = fit$Q_csc_i_per_grid[[k]],
                              p = fit$Q_csc_p_per_grid[[k]],
                              x = fit$Q_csc_x_per_grid[[k]],
                              dims = c(n_x, n_x), index1 = FALSE)
    Q <- L + Matrix::t(L) - Matrix::Diagonal(n_x, Matrix::diag(L))
    if (!is.null(fold)) {
      Q <- as.matrix(Q)
      Q[fold$idx, fold$idx] <- Q[fold$idx, fold$idx] + fold$coef[k]
    }
    V <- as.matrix(Matrix::solve(Q, t(A)))
    v <- rowSums(A * t(V))
    if (length(constr)) {
      Ac <- t(vapply(constr, function(i) { a <- numeric(n_x); a[i] <- 1; a },
                     numeric(n_x)))
      W <- as.matrix(Matrix::solve(Q, t(Ac)))
      AW <- A %*% W
      v <- v - rowSums((AW %*% solve(Ac %*% W)) * AW)
    }
    v
  }, numeric(nrow(A))))
}

test_that("st_icar reports the variance its dense precision implies", {
  skip_on_cran()
  fx <- .st_icar_fixture(n_s = 12L, n_t = 6L, N = 150L, seed = 7L)
  fit <- do.call(cpp_nested_laplace_st_icar,
                 modifyList(fx$args, list(store_Q = TRUE)))
  V <- fit$fitted_eta_var
  expect_true(is.matrix(V))
  expect_equal(dim(V), c(length(fit$log_marginal), nrow(fx$A)))
  expect_true(all(is.finite(V)) && all(V > 0))
  expect_equal(V, .jev_reference(fit, fx$A, constr = fx$constr), tolerance = 1e-8)
})

test_that("a folded sum-to-zero pin gives the variance of the stored one", {
  skip_on_cran()
  fx <- .st_icar_fixture(n_s = 60L, n_t = 10L, N = 400L, seed = 13L)
  args <- modifyList(fx$args, list(store_Q = TRUE, force_sparse = TRUE))

  withr::local_envvar(TULPA_S2Z_DENSIFY_MAX = "100000")
  stored <- do.call(cpp_nested_laplace_st_icar, args)
  # Even densified, the RW1 chain's pin stays off the stored precision (an RW
  # block always folds), so the reference adds it back; the variance is then
  # the inverse of the full precision to 1e-15.
  expect_equal(stored$fitted_eta_var,
               .jev_reference(stored, fx$A, fold = fx$rw_fold,
                              constr = fx$constr),
               tolerance = 1e-6)

  withr::local_envvar(TULPA_S2Z_DENSIFY_MAX = "0")
  folded <- do.call(cpp_nested_laplace_st_icar, args)
  # The fold is what this run exercises: its stored precision no longer holds
  # the pin, so inverting it alone gives a different answer.
  expect_false(isTRUE(all.equal(folded$Q_csc_x_per_grid,
                                stored$Q_csc_x_per_grid)))
  expect_equal(folded$fitted_eta_var, stored$fitted_eta_var, tolerance = 1e-10)
})

test_that("two folded intrinsic pins read the exact constrained variance and block", {
  skip_on_cran()
  n_s <- 60L; n_t <- 10L
  fx <- .st_icar_fixture(n_s = n_s, n_t = n_t, N = 400L, seed = 13L)
  args <- modifyList(fx$args, list(store_Q = TRUE, force_sparse = TRUE))
  withr::local_envvar(TULPA_S2Z_DENSIFY_MAX = "0")
  fit <- do.call(cpp_nested_laplace_st_icar, args)

  # With both pins folded, the stored precision holds the opposite shift of the
  # two fields (+c on every ICAR node, -c on every RW1 node, eta unchanged)
  # through the uniform Laplace ridge alone, so its smallest eigenvalue is that
  # ridge while the pinned precision's condition number is in the thousands.
  p <- ncol(fx$args$X)
  icar <- p + seq_len(n_s)
  rw <- p + n_s + seq_len(n_t)
  n_x <- fit$Q_csc_n
  for (k in seq_along(fit$Q_csc_p_per_grid)) {
    L <- Matrix::sparseMatrix(i = fit$Q_csc_i_per_grid[[k]],
                              p = fit$Q_csc_p_per_grid[[k]],
                              x = fit$Q_csc_x_per_grid[[k]],
                              dims = c(n_x, n_x), index1 = FALSE)
    H <- as.matrix(L + Matrix::t(L) - Matrix::Diagonal(n_x, Matrix::diag(L)))
    shift <- numeric(n_x); shift[icar] <- 1; shift[rw] <- -1
    shift <- shift / sqrt(sum(shift^2))
    expect_lt(drop(crossprod(shift, H %*% shift)), 1e-9)

    # The fields' levels are hard constraints of the inner solve
    # (gcol33/tulpa#1053), so the covariance is the stored precision's
    # restricted to sum-to-zero fields, N (N' H N)^{-1} N' with N a basis of
    # the constraint's null space; a pin on a level the constraint fixes adds
    # nothing to it.
    U <- matrix(0, n_x, 2L); U[icar, 1L] <- 1; U[rw, 2L] <- 1
    N <- qr.Q(qr(U), complete = TRUE)[, -(1:2)]
    V_exact <- N %*% solve(crossprod(N, H %*% N), t(N))
    exact <- rowSums((fx$A %*% V_exact) * fx$A)
    expect_equal(fit$fitted_eta_var[k, ], exact, tolerance = 1e-12)

    # The fixed-effect block of the same constrained covariance.
    blk <- cpp_joint_inner_vcov_blocks(
      list(fit$Q_csc_p_per_grid[[k]]), list(fit$Q_csc_i_per_grid[[k]]),
      list(fit$Q_csc_x_per_grid[[k]]), n_x, seq_len(p), p,
      list(as.integer(icar), as.integer(rw)),
      field_marginal = FALSE, n_threads = 1L)[[1L]]
    expect_equal(blk, V_exact[seq_len(p), seq_len(p)], tolerance = 1e-12)
  }
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
