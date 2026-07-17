# Generic posterior sampler for the grid-integrated joint nested-Laplace
# backend -- the inla.posterior.sample analogue.
#
# Given a joint fit and a latent index set, draw from the outer-grid MIXTURE
#
#   x ~ sum_k w_k * N(m_k, V_k)
#
# where k ranges over the outer hyperparameter grid, w_k are the normalized
# grid weights, m_k is the per-cell inner mode, and V_k is the constrained
# inner-Laplace covariance at cell k (the inverse of the per-grid sparse
# precision Q block, with the ICAR / BYM2 sum-to-zero constraint imposed via
# conditioning by kriging, Rue & Held 2005). Sampling the mixture -- rather
# than collapsing the grid to a single moment-matched Gaussian -- is the
# correct primitive for marginalizing nonlinear derived quantities (change in
# occupancy, expected-cover products) per the "Marginalize Derived Quantities"
# rule, and is written once here rather than re-derived in every consumer.

#' Posterior draws from a joint nested-Laplace fit
#'
#' @description
#' Draw from the outer-grid mixture posterior of a joint nested-Laplace fit
#' ([tulpa_nested_laplace_joint()]) -- the engine analogue of
#' `inla.posterior.sample()`. Each draw picks an outer-grid cell
#' `k ~ Categorical(weights)` and then samples the inner latent vector from the
#' constrained Gaussian `N(m_k, V_k)` at that cell, where `m_k` is the cell's
#' inner mode and `V_k` is the inner-Laplace covariance with the ICAR / BYM2
#' sum-to-zero field constraint imposed (conditioning by kriging). The draws
#' are i.i.d. samples from `sum_k weights_k * N(m_k, V_k)`.
#'
#' Sampling the mixture is the faithful primitive for marginalizing nonlinear
#' derived quantities (e.g. `plogis(eta_2) - plogis(eta_1)`, expected-cover
#' products `p * mu`): compute the derived quantity per draw, then summarize.
#' Collapsing the grid to a single moment-matched Gaussian biases skewed or
#' multimodal-over-grid posteriors.
#'
#' @param fit A `tulpa_nested_laplace_joint` fit (single-block or multi-block).
#'   The fit must have been produced with `control$store_Q = TRUE` so the
#'   per-grid sparse precision `Q_csc_*_per_grid` is available.
#' @param idx Optional integer vector of 1-based latent indices to return.
#'   `NULL` (default) returns the full latent vector. The latent vector stacks
#'   per-arm fixed effects, per-arm random effects, then the latent field(s);
#'   use `fit$arm_layout` (`beta_start`, `re_start`, `phi_start` / `theta_start`
#'   / `field_starts`, all 0-based) to map a sub-block to indices.
#' @param n Number of posterior draws (default 1000).
#' @param ... Unused; for S3 compatibility.
#'
#' @return A numeric matrix `[n x length(idx)]` of latent draws, one row per
#'   draw, columns named `x<idx>`. Carries `attr(., "draws_kind") = "iid"`
#'   (consistent with the draws-provenance gate) and
#'   `attr(., "cells")`, the outer-grid cell index each row was drawn from.
#'
#' @details
#' The constrained draw at cell `k` uses the sparse Cholesky of the stored
#' precision `Q_k` and the conditioning-by-kriging correction
#' \deqn{z_c = z - Q_k^{-1} A^\top (A Q_k^{-1} A^\top)^{-1} A z,}
#' where `z ~ N(0, Q_k^{-1})` and `A` stacks the field sum-to-zero rows (one for
#' an ICAR / CAR field; two -- structured and unstructured -- for a BYM2 field;
#' one per spatial block in a multi-block fit). The returned draw is
#' `m_k + z_c`, restricted to `idx`. Because `m_k` already satisfies the
#' constraint (the inner solve centres the field), the mean is left unchanged
#' and only the covariance is constrained, so the per-cell marginal matches the
#' inner-Laplace constrained covariance exactly.
#'
#' Cells with zero outer-grid weight (e.g. pruned cells) or no stored `Q` are
#' dropped and the remaining weights renormalized. A degenerate single-cell
#' grid (quadrature ESS 1) returns draws from that cell's `N(m_1, V_1)`.
#'
#' @seealso [tulpa_nested_laplace_joint()], [posterior_sample()]
#' @export
tulpa_posterior_draws <- function(fit, idx = NULL, n = 1000, ...) {
    UseMethod("tulpa_posterior_draws")
}

#' @export
tulpa_posterior_draws.default <- function(fit, idx = NULL, n = 1000, ...) {
    stop("tulpa_posterior_draws() is implemented for joint nested-Laplace fits ",
         "(class 'tulpa_nested_laplace_joint'). Got: ",
         paste(class(fit), collapse = ", "), ".", call. = FALSE)
}

#' @export
tulpa_posterior_draws.tulpa_nested_laplace_joint <- function(fit, idx = NULL,
                                                             n = 1000, ...) {
    n <- as.integer(n)
    if (length(n) != 1L || is.na(n) || n < 1L) {
        stop("`n` must be a single positive integer.", call. = FALSE)
    }

    Qp  <- fit$Q_csc_p_per_grid
    Qi  <- fit$Q_csc_i_per_grid
    Qx  <- fit$Q_csc_x_per_grid
    n_x <- fit$Q_csc_n
    if (is.null(Qp) || is.null(Qi) || is.null(Qx) || is.null(n_x)) {
        stop("tulpa_posterior_draws(): the fit carries no per-grid precision ",
             "(`Q_csc_*_per_grid`). Refit with `control$store_Q = TRUE` to ",
             "enable posterior sampling.", call. = FALSE)
    }
    n_x <- as.integer(n_x)

    modes <- fit$modes
    if (is.null(modes) || !is.matrix(modes) || ncol(modes) != n_x) {
        stop("tulpa_posterior_draws(): `fit$modes` is missing or its column ",
             "count does not match `Q_csc_n` (", n_x, ").", call. = FALSE)
    }

    # idx: default to the full latent vector.
    if (is.null(idx)) {
        idx <- seq_len(n_x)
    } else {
        idx <- as.integer(idx)
        if (anyNA(idx) || any(idx < 1L) || any(idx > n_x)) {
            stop("`idx` must be 1-based indices into the latent vector ",
                 "(1..", n_x, ").", call. = FALSE)
        }
    }
    p_out <- length(idx)

    # Outer-grid weights -> valid cells (finite positive weight with stored Q).
    w <- fit$weights
    n_grid <- length(Qp)
    if (is.null(w) || length(w) != n_grid) {
        stop("tulpa_posterior_draws(): `fit$weights` length (",
             length(w), ") does not match the grid size (", n_grid, ").",
             call. = FALSE)
    }
    has_Q <- vapply(seq_len(n_grid), function(k) {
        !is.null(Qp[[k]]) && length(Qx[[k]]) > 0L
    }, logical(1))
    valid <- is.finite(w) & w > 0 & has_Q
    if (!any(valid)) {
        stop("tulpa_posterior_draws(): no outer-grid cell has positive weight ",
             "and a stored precision; nothing to sample.", call. = FALSE)
    }
    if (any(is.finite(w) & w > 0 & !has_Q)) {
        warning("tulpa_posterior_draws(): some positive-weight cells carry no ",
                "stored precision and were dropped from the mixture.",
                call. = FALSE)
    }
    cells <- which(valid)
    w_valid <- w[cells]
    w_valid <- w_valid / sum(w_valid)

    # Field sum-to-zero constraint over the FULL latent vector. Shared across
    # cells (the constraint structure is grid-constant); only Q changes per cell.
    A_cols <- .joint_constraint_cols(fit$arm_layout, n_x)
    k_constr <- length(A_cols)
    A_t <- if (k_constr > 0L) {
        ii <- unlist(A_cols)
        jj <- rep(seq_len(k_constr), vapply(A_cols, length, integer(1)))
        Matrix::sparseMatrix(i = ii, j = jj, x = 1, dims = c(n_x, k_constr))
    } else NULL

    # Allocate draws across cells: multinomial(n, w_valid). Each cell then
    # draws its share i.i.d. from the constrained N(m_k, V_k).
    counts <- as.integer(stats::rmultinom(1L, size = n, prob = w_valid))

    out <- matrix(0.0, n, p_out)
    row_cells <- integer(n)
    pos <- 0L
    for (kk in seq_along(cells)) {
        n_k <- counts[kk]
        if (n_k == 0L) next
        k <- cells[kk]
        Z <- .joint_draw_cell(Qp[[k]], Qi[[k]], Qx[[k]], n_x, A_t, n_k)
        # Z is n_x x n_k draws of the zero-mean constrained Gaussian; add the
        # cell mode and subset to idx.
        m_k <- modes[k, ]
        block <- Z[idx, , drop = FALSE] + m_k[idx]      # p_out x n_k (recycled)
        out[(pos + 1L):(pos + n_k), ] <- t(block)
        row_cells[(pos + 1L):(pos + n_k)] <- k
        pos <- pos + n_k
    }

    colnames(out) <- paste0("x", idx)
    attr(out, "draws_kind") <- "iid"
    attr(out, "cells") <- row_cells
    out
}

# Field sum-to-zero constraint columns for a joint latent layout. Returns a
# list of 1-based index vectors, one per sum-to-zero constraint, over the
# full n_x latent vector. Handles BOTH layout shapes:
#   * multi-block (`.joint_multi_layout`): `field_starts` + `block_start` /
#     `block_size` + `field_block_types`; a BYM2 block contributes two
#     constraints (structured phi + unstructured theta), every other spatial
#     block one.
#   * single-block (`.joint_layout`): `phi_start` [, `theta_start`] without
#     block sizes; phi spans [phi_start, theta_start) (or to n_x when there is
#     no theta), theta spans [theta_start, n_x).
# A non-spatial fit returns an empty list (unconstrained draw).
.joint_constraint_cols <- function(layout, n_x) {
    cols <- list()
    if (is.null(layout)) return(cols)

    fs <- layout$field_starts
    if (!is.null(fs) && !is.null(layout$block_start) &&
        !is.null(layout$block_size)) {
        types  <- layout$field_block_types %||% rep("icar", length(fs))
        bstart <- layout$block_start
        bsize  <- layout$block_size
        for (i in seq_along(fs)) {
            s0   <- fs[i]
            type <- tolower(types[i])
            b    <- match(s0, bstart)
            sz   <- if (is.na(b)) NA_integer_ else bsize[b]
            if (is.na(sz)) next
            if (type == "bym2") {
                # Only the structured (ICAR) component phi is rank-deficient and
                # sum-to-zero constrained (matching its inner-solve centerer).
                # The unstructured theta has a proper N(0,1) prior and no
                # centerer, so constraining it removes a genuine degree of
                # freedom and under-disperses the field draws.
                nu <- as.integer(sz / 2L)
                cols[[length(cols) + 1L]] <- s0 + seq_len(nu)
            } else {
                cols[[length(cols) + 1L]] <- s0 + seq_len(as.integer(sz))
            }
        }
        return(cols)
    }

    # Single-block: derive field extents from phi_start / theta_start. Only the
    # structured phi is constrained; theta_start marks the BYM2 unstructured
    # component, which has a proper N(0,1) prior and is NOT sum-to-zero (see the
    # multi-block branch above).
    if (!is.null(layout$phi_start)) {
        n_phi <- (layout$theta_start %||% n_x) - layout$phi_start
        if (n_phi > 0L) cols[[length(cols) + 1L]] <- layout$phi_start +
            seq_len(n_phi)
    }
    cols
}

# Draw `n_k` zero-mean samples of the constrained Gaussian N(0, V_k) for one
# outer-grid cell, returned as an n_x x n_k matrix. `Qp` / `Qi` / `Qx` are the
# stored lower-triangle CSC triplets (0-based) of the cell precision Q_k; `A_t`
# is the shared n_x x k_constr constraint matrix (or NULL for an unconstrained
# block).
#
# Unconstrained part: z ~ N(0, Q^{-1}) via the sparse Cholesky factor
#   P Q P' = L L'  =>  z = P' L'^{-1} u,  u ~ N(0, I).
# Constraint (conditioning by kriging, Rue & Held 2005):
#   z_c = z - W (A Q^{-1} A')^{-1} (A z),  W = Q^{-1} A'.
# The resulting covariance is Q^{-1} - Q^{-1} A' (A Q^{-1} A')^{-1} A Q^{-1},
# exactly the constrained inner-Laplace covariance.
.joint_draw_cell <- function(Qp, Qi, Qx, n_x, A_t, n_k) {
    Qk_lt <- Matrix::sparseMatrix(
        i = as.integer(Qi) + 1L,
        p = as.integer(Qp),
        x = as.numeric(Qx),
        dims = c(n_x, n_x),
        symmetric = FALSE,
        index1 = TRUE
    )
    Qk <- Matrix::forceSymmetric(Qk_lt, uplo = "L")

    Ch <- tryCatch(Matrix::Cholesky(Qk, perm = TRUE, LDL = FALSE),
                   error = function(e) NULL)
    if (is.null(Ch)) {
        # The (intercept, mean-field) direction of an improper field can leave
        # Q_k numerically singular; lift it with a tiny ridge. The field
        # constraint below projects that direction out, so the ridge's effect
        # on the constrained draw is negligible.
        jit <- 1e-8 * (mean(Matrix::diag(Qk)) + 1e-8)
        Qk <- Qk + Matrix::Diagonal(n_x, jit)
        Ch <- Matrix::Cholesky(Qk, perm = TRUE, LDL = FALSE)
    }

    U <- matrix(stats::rnorm(n_x * n_k), n_x, n_k)
    # z = P' L'^{-1} U
    z <- as.matrix(Matrix::solve(Ch, Matrix::solve(Ch, U, system = "Lt"),
                                 system = "Pt"))

    if (!is.null(A_t)) {
        W  <- as.matrix(Matrix::solve(Ch, A_t, system = "A"))  # Q^{-1} A'
        M  <- as.matrix(Matrix::crossprod(A_t, W))             # A Q^{-1} A'
        Az <- as.matrix(Matrix::crossprod(A_t, z))             # A z
        z  <- z - W %*% solve(M, Az)
    }
    z
}

# Law-of-total-covariance moments of the joint mixture restricted to `idx`:
#   mean  = sum_k w_k m_k[idx]
#   Sigma = sum_k w_k ( V_k[idx, idx] + (m_k[idx] - mean)(m_k[idx] - mean)' )
# This is the analytic target the mixture draws reproduce to MC error; it also
# serves as the moment-matched-Gaussian fallback. Returns NULL when the fit
# carries no stored precision.
.joint_mixture_moments <- function(fit, idx = NULL) {
    Qp  <- fit$Q_csc_p_per_grid
    Qi  <- fit$Q_csc_i_per_grid
    Qx  <- fit$Q_csc_x_per_grid
    n_x <- fit$Q_csc_n
    if (is.null(Qp) || is.null(Qi) || is.null(Qx) || is.null(n_x)) return(NULL)
    n_x <- as.integer(n_x)
    modes <- fit$modes
    if (is.null(idx)) idx <- seq_len(n_x) else idx <- as.integer(idx)
    p_out <- length(idx)

    w <- fit$weights
    n_grid <- length(Qp)
    has_Q <- vapply(seq_len(n_grid), function(k)
        !is.null(Qp[[k]]) && length(Qx[[k]]) > 0L, logical(1))
    valid <- is.finite(w) & w > 0 & has_Q
    cells <- which(valid)
    w_valid <- w[cells] / sum(w[cells])

    A_cols <- .joint_constraint_cols(fit$arm_layout, n_x)
    k_constr <- length(A_cols)
    A_t <- if (k_constr > 0L) {
        ii <- unlist(A_cols)
        jj <- rep(seq_len(k_constr), vapply(A_cols, length, integer(1)))
        Matrix::sparseMatrix(i = ii, j = jj, x = 1, dims = c(n_x, k_constr))
    } else NULL

    # Selector for idx columns of Q^{-1}.
    E <- Matrix::sparseMatrix(i = idx, j = seq_len(p_out), x = 1,
                              dims = c(n_x, p_out))

    mbar <- as.numeric(crossprod(w_valid, modes[cells, idx, drop = FALSE]))
    Sigma <- matrix(0.0, p_out, p_out)
    for (kk in seq_along(cells)) {
        k <- cells[kk]
        V_block <- .joint_inner_vcov_block_one(Qp[[k]], Qi[[k]], Qx[[k]], n_x,
                                               idx, E, A_t)
        if (is.null(V_block)) next
        dk <- modes[k, idx] - mbar
        Sigma <- Sigma + w_valid[kk] * (V_block + tcrossprod(dk))
    }
    Sigma <- (Sigma + t(Sigma)) / 2
    list(mean = mbar, Sigma = Sigma)
}

# Constrained inner-Laplace covariance of the `idx` block at one cell:
#   V = Q^{-1}[idx, idx] - (Q^{-1} A')[idx]' (A Q^{-1} A')^{-1} (A Q^{-1})[, idx]
# `E` selects the idx columns; `A_t` is the shared constraint matrix (or NULL).
.joint_inner_vcov_block_one <- function(Qp, Qi, Qx, n_x, idx, E, A_t) {
    Qk_lt <- Matrix::sparseMatrix(
        i = as.integer(Qi) + 1L, p = as.integer(Qp), x = as.numeric(Qx),
        dims = c(n_x, n_x), symmetric = FALSE, index1 = TRUE
    )
    Qk <- Matrix::forceSymmetric(Qk_lt, uplo = "L")
    V <- tryCatch(Matrix::solve(Qk, E), error = function(e) NULL)
    if (is.null(V)) return(NULL)
    V_block <- as.matrix(V[idx, , drop = FALSE])
    if (!is.null(A_t)) {
        W <- tryCatch(Matrix::solve(Qk, A_t), error = function(e) NULL)
        if (!is.null(W)) {
            AV <- as.matrix(Matrix::crossprod(A_t, V))
            M  <- as.matrix(Matrix::crossprod(A_t, W))
            V_block <- V_block - crossprod(AV, solve(M, AV))
        }
    }
    V_block
}
