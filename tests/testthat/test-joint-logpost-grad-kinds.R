# The finite-difference gate on the joint nested-Laplace gradient, across every
# block contribution kind (gcol33/tulpa#420).
#
# cpp_test_joint_logpost_grad's DENSE assembly resolves a block only through
# `idx`, so it sees INDEXED_SINGLE and nothing else. It used to skip the rest
# silently: hand it an hsgp block and it returned a logpost and a grad for the
# model with that block REMOVED, and a central difference of that logpost
# reproduced that grad, so the gate reported agreement on a model no fit solves.
# Every fit carrying such a block runs the SPARSE kernel instead
# (blocks_require_sparse), and that kernel had no finite-difference cross-check
# reachable from R at all.
#
# The dense path now refuses a block it cannot resolve, and `sparse = TRUE`
# gates the scatter those fits actually run. These cover one block per kind:
# DENSE_BASIS (hsgp), INDEXED_MULTI (mcar), BILINEAR_FACTOR (lf), against the
# INDEXED_SINGLE (icar) the gate already had.

.gk_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

.gk_arms <- function(N = c(24L, 28L), p = 2L, seed = 11L) {
    set.seed(seed)
    arms <- lapply(N, function(n) {
        X <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
        list(y = rnorm(n), n_trials = rep(1L, n), X = X,
             re_idx = rep(0, n), n_re_groups = 0L, sigma_re = 1.0,
             family = "gaussian", phi = 1.0)
    })
    lapply(seq_along(arms), function(k)
        .normalise_joint_arm_multi(arms[[k]], k))
}

# Build the cpp-side block spec and its axis grid from a front-door spec, so the
# gate is fed exactly what the dispatcher would feed the kernel.
.gk_block <- function(spec_R, arms_norm) {
    pb <- .joint_block_axis_grid(spec_R, FALSE, numeric(0), 1L)
    list(bs = .joint_block_spec_for_cpp(pb$prepared, n_arms = length(arms_norm),
                                        block_index = 1L, arms = arms_norm),
         grid = pb$grid)
}

.gk_eval <- function(arms_norm, blk, x, sparse) {
    cpp_test_joint_logpost_grad(
        arms_list = arms_norm,
        copy_arms = integer(0), copy_blocks = integer(0),
        blocks_spec = list(blk$bs),
        theta_grid = blk$grid[1L, , drop = FALSE],
        axis_offsets = as.integer(c(0L, ncol(blk$grid))),
        x = x, k_grid = 0L, sparse = sparse)
}

# Central difference of the gate's own logpost against the gate's own grad.
.gk_fd_check <- function(arms_norm, blk, n_x, sparse, seed = 99L, tol = 1e-4) {
    set.seed(seed)
    x0 <- rnorm(n_x, sd = 0.35)
    base <- .gk_eval(arms_norm, blk, x0, sparse)
    expect_equal(base$n_x, n_x)
    h <- 1e-5
    g_fd <- vapply(seq_len(n_x), function(j) {
        xp <- x0; xp[j] <- xp[j] + h
        xm <- x0; xm[j] <- xm[j] - h
        (.gk_eval(arms_norm, blk, xp, sparse)$logpost -
         .gk_eval(arms_norm, blk, xm, sparse)$logpost) / (2 * h)
    }, numeric(1))
    expect_lt(max(abs(base$grad - g_fd)), tol)
    list(grad = base$grad, logpost = base$logpost, x = x0)
}

# --------------------------------------------------------------------------
# Fixtures, one per contribution kind
# --------------------------------------------------------------------------

.gk_icar <- function(arms_norm, n_s = 8L) {
    adj <- .gk_chain_adj(n_s)
    idx <- lapply(arms_norm, function(a) sample.int(n_s, length(a$y), TRUE))
    list(spec = c(list(type = "icar", spatial_idx = idx,
                       sigma_grid = c(0.6, 0.9)), adj),
         n_latent = n_s)
}

.gk_hsgp <- function(arms_norm, m_total = 6L) {
    # DENSE_BASIS: every observation reads every basis column.
    set.seed(5L)
    phi <- lapply(arms_norm, function(a) {
        s <- runif(length(a$y))
        outer(s, seq_len(m_total), function(u, m) sin(m * pi * u))
    })
    list(spec = list(type = "hsgp", m_total = m_total, phi = phi,
                     n_obs_per_arm = vapply(arms_norm, function(a) length(a$y),
                                            integer(1)),
                     eigenvalues = (seq_len(m_total) * pi)^2,
                     sigma2_grid = c(0.5, 1.0),
                     lengthscale_grid = c(0.4, 0.8)),
         n_latent = m_total)
}

.gk_mcar <- function(arms_norm, n_s = 6L, n_fields = 2L) {
    # INDEXED_MULTI: each observation reads one cell of each of n_fields fields.
    adj <- .gk_chain_adj(n_s)
    idx <- lapply(arms_norm, function(a) sample.int(n_s, length(a$y), TRUE))
    fw <- lapply(seq_len(n_fields), function(f)
        lapply(arms_norm, function(a) {
            if (f == 1L) rep(1, length(a$y)) else as.numeric(a$X[, 2L])
        }))
    list(spec = c(list(type = "mcar", spatial_idx = idx, n_fields = n_fields,
                       field_weight = fw), adj),
         n_latent = n_fields * n_s)
}

.gk_lf <- function(arms_norm, n_latent = 7L) {
    # BILINEAR_FACTOR: each observation reads one factor slot and one loading.
    idx <- lapply(arms_norm, function(a) sample.int(n_latent, length(a$y), TRUE))
    list(spec = list(type = "lf", n_latent = n_latent, obs_idx = idx),
         n_latent = n_latent + length(arms_norm))
}

# --------------------------------------------------------------------------

test_that("the dense gate refuses a block its scatter cannot resolve", {
    arms_norm <- .gk_arms()
    for (nm in c("hsgp", "mcar", "lf")) {
        fx <- switch(nm, hsgp = .gk_hsgp(arms_norm), mcar = .gk_mcar(arms_norm),
                     lf = .gk_lf(arms_norm))
        blk <- .gk_block(fx$spec, arms_norm)
        n_x <- sum(vapply(arms_norm, function(a) ncol(a$X), integer(1))) +
               fx$n_latent
        expect_error(.gk_eval(arms_norm, blk, rnorm(n_x), sparse = FALSE),
                     "not INDEXED_SINGLE", info = nm)
    }
})

test_that("the sparse gate matches central FD on every contribution kind", {
    arms_norm <- .gk_arms()
    p_tot <- sum(vapply(arms_norm, function(a) ncol(a$X), integer(1)))
    set.seed(3L)
    for (nm in c("icar", "hsgp", "mcar", "lf")) {
        fx <- switch(nm,
                     icar = .gk_icar(arms_norm), hsgp = .gk_hsgp(arms_norm),
                     mcar = .gk_mcar(arms_norm), lf   = .gk_lf(arms_norm))
        blk <- .gk_block(fx$spec, arms_norm)
        r <- .gk_fd_check(arms_norm, blk, p_tot + fx$n_latent, sparse = TRUE)
        # The block genuinely contributes: its latent coordinates carry gradient.
        latent_idx <- (p_tot + 1L):(p_tot + fx$n_latent)
        expect_gt(max(abs(r$grad[latent_idx])), 1e-8, label = nm)
    }
})

test_that("dense and sparse agree on a pure INDEXED_SINGLE spec", {
    # The one place the two scatters can be pinned to each other. Without it a
    # divergence between them would show only as a fit-level difference.
    arms_norm <- .gk_arms(seed = 31L)
    p_tot <- sum(vapply(arms_norm, function(a) ncol(a$X), integer(1)))
    set.seed(3L)
    fx <- .gk_icar(arms_norm)
    blk <- .gk_block(fx$spec, arms_norm)
    n_x <- p_tot + fx$n_latent
    set.seed(41L)
    x0 <- rnorm(n_x, sd = 0.35)
    d <- .gk_eval(arms_norm, blk, x0, sparse = FALSE)
    s <- .gk_eval(arms_norm, blk, x0, sparse = TRUE)
    expect_equal(s$logpost, d$logpost, tolerance = 1e-10)
    expect_equal(s$grad, d$grad, tolerance = 1e-10)
})
