# helper-joint-mixture-moments.R
#
# The analytic reference for a joint nested-Laplace fit's latent mixture, read
# off the stored per-cell precisions. Shared by test-posterior-draws-joint.R,
# test-inner-constraint-read.R and test-nested-laplace-joint-fixed-moments.R.

# Law-of-total-covariance moments of the joint mixture restricted to `idx`:
#   mean  = sum_k w_k m_k[idx]
#   Sigma = sum_k w_k ( V_k[idx, idx] + (m_k[idx] - mean)(m_k[idx] - mean)' )
# with V_k the covariance of cell k's inner Gaussian, read by the engine's
# extraction (cpp_joint_inner_vcov_blocks) under the same constraint the draws
# take: the analytic target the mixture draws reproduce to MC error, and the
# fixed-effect mixture vcov() reads.
# Returns NULL when the fit carries no stored precision.
joint_mixture_moments <- function(fit, idx = NULL) {
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

    ccpp <- tulpa:::.inner_constraint_cpp(tulpa:::.inner_constraint(fit, cells))
    blocks <- tulpa:::cpp_joint_inner_vcov_blocks(
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
