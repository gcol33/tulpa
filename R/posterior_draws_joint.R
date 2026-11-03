# The joint nested-Laplace arm of `tulpa_posterior_draws()` -- the front door,
# the shared mixture allocator and the single-block arm live in
# R/posterior_draws.R.
#
# Given a joint fit and a latent index set, draw from the outer-grid MIXTURE
#
#   x ~ sum_k w_k * N(m_k, V_k)
#
# where k ranges over the outer hyperparameter grid, w_k are the normalized
# grid weights, m_k is the per-cell inner mode, and V_k is the covariance of
# the cell's inner Gaussian: the stored precision plus the sum-to-zero pins it
# leaves off, conditioned on the hard rows the inner solve carried
# (`.inner_constraint()`). Sampling the mixture -- rather
# than collapsing the grid to a single moment-matched Gaussian -- is the
# correct primitive for marginalizing nonlinear derived quantities (change in
# occupancy, expected-cover products) per the "Marginalize Derived Quantities"
# rule, and is written once here rather than re-derived in every consumer.

#' Posterior draws from a joint nested-Laplace fit
#'
#' @description
#' The [tulpa_posterior_draws()] method for a joint nested-Laplace fit
#' ([tulpa_nested_laplace_joint()]). Each draw picks an outer-grid cell
#' `k ~ Categorical(weights)` and then samples the inner latent vector from the
#' inner Gaussian `N(m_k, V_k)` at that cell, where `m_k` is the cell's inner
#' mode and `V_k` the covariance the cell's log-marginal was read off. The
#' draws are i.i.d. samples from `sum_k weights_k * N(m_k, V_k)`.
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
#'   (consistent with the draws-provenance gate), `attr(., "cells")` -- the
#'   outer-grid cell index each row was drawn from -- `attr(., "theta")`, the
#'   hyperparameter half of the same rows continuized within each row's own cell
#'   ([tulpa_hyper_draws()]) -- and `attr(., "scope") = "latent"`.
#'
#' @details
#' The inner Gaussian at cell `k` has precision
#' \eqn{M_k = Q_k + S^\top D_k S}, the stored precision `Q_k` plus the
#' sum-to-zero pins `S` the sparse inner solve folds at solve time (an
#' intrinsic field's augmented prior, precision `D_k`), conditioned on the hard
#' rows `A x = 0` the inner solve carried (an intrinsic field's level row where
#' a fixed effect absorbs its level, a Knorr-Held interaction's null space):
#' \eqn{V_k = B (B^\top M_k B)^{-1} B^\top}, `B` a basis of the null space of
#' `A`. A pinned field with no level row keeps its level, at the pin's
#' precision. The returned draw is `m_k` plus a draw of `N(0, V_k)`,
#' restricted to `idx`; `m_k` already satisfies the hard rows.
#'
#' Cells with zero outer-grid weight (e.g. pruned cells) or no stored `Q` are
#' dropped and the remaining weights renormalized. A degenerate single-cell
#' grid (quadrature ESS 1) returns draws from that cell's `N(m_1, V_1)`.
#'
#' @seealso [tulpa_posterior_draws()], [tulpa_nested_laplace_joint()],
#'   [posterior_sample()]
#' @export
tulpa_posterior_draws.tulpa_nested_laplace_joint <- function(fit, idx = NULL,
                                                             n = 1000, ...) {
    n <- .nl_draws_check_n(n)

    Qp  <- fit$Q_csc_p_per_grid
    Qi  <- fit$Q_csc_i_per_grid
    Qx  <- fit$Q_csc_x_per_grid
    n_x <- fit$Q_csc_n
    if (is.null(Qp) || is.null(Qi) || is.null(Qx) || is.null(n_x)) {
        stop("tulpa_posterior_draws(): the fit carries no per-grid precision ",
             "(`Q_csc_*_per_grid`), so the latent vector cannot be sampled. ",
             "Refit with `control$store_Q = TRUE` to enable it. The retained ",
             "fixed-effect mixture is still samplable: ",
             "`tulpa:::tulpa_posterior_draws.tulpa_nested_laplace(fit)`.",
             call. = FALSE)
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

    cons <- .inner_constraint(fit)

    # Each cell draws its share i.i.d. from the constrained N(m_k, V_k); the
    # allocation across cells is the shared `.nl_mixture_draw()`.
    out <- .nl_mixture_draw(
        w = w_valid, cell_id = cells, n = n, p = p_out,
        draw_cell = function(kk, n_k) {
            k <- cells[kk]
            Z <- .joint_draw_cell(Qp[[k]], Qi[[k]], Qx[[k]], n_x,
                                  .inner_constraint_cell(cons, k), n_k)
            # Z is n_x x n_k draws of the zero-mean constrained Gaussian; add
            # the cell mode and subset to idx.
            m_k <- modes[k, ]
            t(Z[idx, , drop = FALSE] + m_k[idx])   # p_out x n_k, recycled
        })

    colnames(out) <- paste0("x", idx)
    attr(out, "draws_kind") <- "iid"
    attr(out, "scope") <- "latent"
    .nl_attach_hyper_draws(out, fit)
}

# Draw `n_k` zero-mean samples of cell k's inner Gaussian N(0, V_k), returned
# as an n_x x n_k matrix. `Qp` / `Qi` / `Qx` are the stored lower-triangle CSC
# triplets (0-based) of the cell precision Q_k; `constr` the cell's constraint,
# `.inner_constraint_cell()`.
#
# V = B (B' M B)^{-1} B' with M = Q + S' D S (the soft pins S, precision D)
# and B a basis of null(A) (the hard rows A). The draw is the engine's
# (cpp_constrained_cell_draw): it factors Q with one anchor per row on its
# diagonal, the factor every covariance read of the cell uses
# (InvBlockConstraint), and maps the factor's draws onto N(0, V). Q is never
# inverted along a row, so it need carry no curvature on a direction only a
# row identifies -- a pin the sparse joint Newton folds at solve time, a
# Knorr-Held interaction's null space.
.joint_draw_cell <- function(Qp, Qi, Qx, n_x, constr, n_k) {
    U <- matrix(stats::rnorm(n_x * n_k), n_x, n_k)
    cpp_constrained_cell_draw(as.integer(Qp), as.integer(Qi), as.numeric(Qx),
                              as.integer(n_x), .constraint_rows_for_cpp(constr$hard),
                              U,
                              soft_cols_list = if (!is.null(constr$soft_D))
                                  .constraint_rows_for_cpp(constr$soft),
                              soft_D = constr$soft_D)
}

# Law-of-total-covariance moments of the joint mixture restricted to `idx`:
#   mean  = sum_k w_k m_k[idx]
#   Sigma = sum_k w_k ( V_k[idx, idx] + (m_k[idx] - mean)(m_k[idx] - mean)' )
# with V_k the covariance of cell k's inner Gaussian, read by the engine's
# extraction (cpp_joint_inner_vcov_blocks) under the same constraint the draws
# take. This is the analytic target the mixture draws reproduce to MC error.
# Returns NULL when the fit carries no stored precision.
.joint_mixture_moments <- function(fit, idx = NULL) {
    Qp  <- fit$Q_csc_p_per_grid
    Qi  <- fit$Q_csc_i_per_grid
    Qx  <- fit$Q_csc_x_per_grid
    n_x <- fit$Q_csc_n
    if (is.null(Qp) || is.null(Qi) || is.null(Qx) || is.null(n_x)) return(NULL)
    n_x <- as.integer(n_x)
    modes <- fit$modes
    if (is.null(idx)) idx <- seq_len(n_x) else idx <- as.integer(idx)

    w <- fit$weights
    n_grid <- length(Qp)
    has_Q <- vapply(seq_len(n_grid), function(k)
        !is.null(Qp[[k]]) && length(Qx[[k]]) > 0L, logical(1))
    valid <- is.finite(w) & w > 0 & has_Q
    cells <- which(valid)
    w_valid <- w[cells] / sum(w[cells])

    ccpp <- .inner_constraint_cpp(.inner_constraint(fit, cells))
    blocks <- cpp_joint_inner_vcov_blocks(
        Q_p_per_grid = Qp[cells], Q_i_per_grid = Qi[cells],
        Q_x_per_grid = Qx[cells], n_x = n_x, idx = idx,
        n_dense = length(idx), A_cols_list = ccpp$A_cols_list,
        field_marginal = FALSE, n_threads = 1L,
        soft_cols_list = ccpp$soft_cols_list,
        soft_D_per_grid = ccpp$soft_D_per_grid)

    mbar <- as.numeric(crossprod(w_valid, modes[cells, idx, drop = FALSE]))
    Sigma <- matrix(0.0, length(idx), length(idx))
    for (kk in seq_along(cells)) {
        V_block <- blocks[[kk]]
        if (is.null(V_block)) next
        dk <- modes[cells[kk], idx] - mbar
        Sigma <- Sigma + w_valid[kk] * (V_block + tcrossprod(dk))
    }
    Sigma <- (Sigma + t(Sigma)) / 2
    list(mean = mbar, Sigma = Sigma)
}
