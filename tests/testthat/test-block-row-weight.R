# A block's per-row design weight (`svc_weight` -> LatentBlock::row_weight) is
# read on every contribution kind, not only on the one-cell-per-row areal kind.
# An SPDE block reaches its latent through the barycentric projector A
# (INDEXED_MULTI: several mesh nodes per observation); before gcol33/tulpa#463
# both eta assembly and the scatter dropped the weight there, so a block
# registered as a varying coefficient fitted as a plain field with nothing on
# the returned object to say so.
#
# The arbiters are the two ends of the weight: a constant 1 must reproduce the
# unweighted fit exactly (the weight enters at one layer, so multiplying by 1.0
# is a no-op), and a non-constant weight must move the fit.

# One-dimensional linear FEM on m equally spaced nodes: lumped mass diagonal and
# the tridiagonal stiffness matrix, both at element length h.
.rw_fem_1d <- function(m, h = 1.0) {
    c0 <- rep(h, m); c0[1] <- h / 2; c0[m] <- h / 2
    i <- c(seq_len(m), seq_len(m - 1), 2:m)
    j <- c(seq_len(m), 2:m, seq_len(m - 1))
    x <- c(c(1, rep(2, m - 2), 1) / h, rep(-1 / h, m - 1), rep(-1 / h, m - 1))
    list(c0_diag = c0, G1 = Matrix::sparseMatrix(i = i, j = j, x = x,
                                                 dims = c(m, m)))
}

# Barycentric projector: observation r sits at position s_r on the mesh and
# loads the two nodes of its element.
.rw_projector <- function(s, m, h = 1.0) {
    N <- length(s)
    e <- pmin(pmax(floor(s / h) + 1L, 1L), m - 1L)
    t <- s / h - (e - 1L)
    Matrix::sparseMatrix(
        i = c(seq_len(N), seq_len(N)),
        j = c(e, e + 1L),
        x = c(1 - t, t),
        dims = c(N, m)
    )
}

.rw_spde_block <- function(A, fem, m, N, svc_weight = NULL) {
    A <- methods::as(A, "CsparseMatrix")
    G <- methods::as(fem$G1, "CsparseMatrix")
    out <- list(
        type    = "spde",
        n_mesh  = as.integer(m),
        n_obs   = as.integer(N),
        A_x     = as.numeric(A@x),
        A_i     = as.integer(A@i),
        A_p     = as.integer(A@p),
        C0_diag = as.numeric(fem$c0_diag),
        G1_x    = as.numeric(G@x),
        G1_i    = as.integer(G@i),
        G1_p    = as.integer(G@p),
        nu      = 1.0
    )
    if (!is.null(svc_weight)) out$svc_weight <- as.numeric(svc_weight)
    out
}

.rw_fit <- function(block, y, X, theta_grid) {
    tulpa:::cpp_nested_laplace_multi(
        y = y, n = rep(1L, length(y)), X = X,
        re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
        blocks_spec = list(block),
        theta_grid = theta_grid, axis_offsets = c(0L, 2L),
        family = "gaussian", phi = 1.0,
        max_iter = 60L, tol = 1e-9, n_threads = 1L
    )
}

.rw_fixture <- function(seed = 4631L, m = 12L, N = 60L) {
    set.seed(seed)
    fem <- .rw_fem_1d(m)
    s   <- runif(N, 0, m - 1)
    A   <- .rw_projector(s, m)
    X   <- cbind(1, rnorm(N))
    w   <- 0.5 + s / (m - 1)             # non-constant, strictly positive
    y   <- as.numeric(X %*% c(0.3, -0.4) + w * sin(s) + rnorm(N, sd = 0.3))
    list(fem = fem, A = A, X = X, y = y, w = w, m = m, N = N,
         theta_grid = cbind(c(1.0, 2.0), c(0.8, 1.2)))   # (range, sigma)
}

test_that("an SPDE block's svc_weight reaches eta and the mode", {
    f <- .rw_fixture()

    plain <- .rw_fit(.rw_spde_block(f$A, f$fem, f$m, f$N), f$y, f$X, f$theta_grid)
    ones  <- .rw_fit(.rw_spde_block(f$A, f$fem, f$m, f$N, rep(1, f$N)),
                     f$y, f$X, f$theta_grid)
    wtd   <- .rw_fit(.rw_spde_block(f$A, f$fem, f$m, f$N, f$w),
                     f$y, f$X, f$theta_grid)

    expect_true(all(is.finite(plain$log_marginal)))
    expect_true(all(is.finite(wtd$log_marginal)))

    # A weight of exactly 1 enters at one layer and cannot perturb anything.
    expect_identical(as.numeric(ones$log_marginal),
                     as.numeric(plain$log_marginal))
    expect_identical(as.numeric(ones$modes), as.numeric(plain$modes))

    # A non-constant weight is a different model: it must move both the mode
    # and the marginal. Before the fix these were identical to `plain`.
    expect_false(isTRUE(all.equal(as.numeric(wtd$log_marginal),
                                  as.numeric(plain$log_marginal))))
    expect_gt(max(abs(as.numeric(wtd$modes) - as.numeric(plain$modes))), 1e-6)
})

test_that("a weighted SPDE block matches the field scaled by hand", {
    # eta_i = X_i beta + w_i (A u)_i is the same model as one whose projector
    # rows are pre-scaled by w, so the two fits agree to the bit: this pins
    # WHERE the weight enters, not merely that it does.
    f <- .rw_fixture(seed = 4632L)
    A_scaled <- Matrix::Diagonal(x = f$w) %*% f$A

    wtd <- .rw_fit(.rw_spde_block(f$A, f$fem, f$m, f$N, f$w),
                   f$y, f$X, f$theta_grid)
    pre <- .rw_fit(.rw_spde_block(A_scaled, f$fem, f$m, f$N),
                   f$y, f$X, f$theta_grid)

    expect_equal(as.numeric(wtd$log_marginal), as.numeric(pre$log_marginal),
                 tolerance = 1e-10)
    expect_equal(as.numeric(wtd$modes), as.numeric(pre$modes),
                 tolerance = 1e-8)
})

test_that("svc_weight of the wrong length is refused", {
    f <- .rw_fixture(seed = 4633L)
    expect_error(
        .rw_fit(.rw_spde_block(f$A, f$fem, f$m, f$N, rep(1, f$N - 1L)),
                f$y, f$X, f$theta_grid),
        "svc_weight has length"
    )
})
