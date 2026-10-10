# gcol33/tulpa#952: which positive-weight cells of a saved occu_cover fit carry
# a stored precision Q that is not positive definite, and what the fit recorded
# about them. Usage: Rscript inspect_fit.R <fit.rds> [lib]
args <- commandArgs(TRUE)
if (length(args) >= 2L && nzchar(args[2])) .libPaths(c(args[2], .libPaths()))
suppressPackageStartupMessages(library(Matrix))
f <- readRDS(args[1])
j <- if (!is.null(f$fit$joint_fit)) f$fit$joint_fit else f$joint_fit
n_x <- as.integer(j$Q_csc_n)
w <- j$weights
cat("cells", length(w), " n_x", n_x, " hyperprior", j$hyperprior %||% NA, "\n")
`%||%` <- function(x, y) if (is.null(x)) y else x
lay <- j$arm_layout
A_cols <- tulpa:::.joint_constraint_cols(lay, n_x)
pin <- if (length(A_cols)) {
    ii <- unlist(A_cols)
    jj <- rep(seq_along(A_cols), lengths(A_cols))
    Matrix::sparseMatrix(i = ii, j = jj, x = 1, dims = c(n_x, length(A_cols)))
} else NULL
pos <- which(is.finite(w) & w > 0)
cat("positive-weight cells:", length(pos), "\n")
rows <- list()
t0 <- Sys.time()
for (k in pos) {
    Q <- Matrix::forceSymmetric(Matrix::sparseMatrix(
        i = as.integer(j$Q_csc_i_per_grid[[k]]) + 1L,
        p = as.integer(j$Q_csc_p_per_grid[[k]]),
        x = as.numeric(j$Q_csc_x_per_grid[[k]]), dims = c(n_x, n_x),
        index1 = TRUE), uplo = "L")
    ok_raw <- !inherits(try(Matrix::Cholesky(Q, perm = TRUE, LDL = FALSE),
                            silent = TRUE), "try-error")
    jit <- 1e-8 * (mean(Matrix::diag(Q)) + 1e-8)
    ok_jit <- !inherits(try(Matrix::Cholesky(Q + Matrix::Diagonal(n_x, jit),
                                             perm = TRUE, LDL = FALSE),
                            silent = TRUE), "try-error")
    ok_pin <- if (is.null(pin)) NA else
        !inherits(try(Matrix::Cholesky(Q + Matrix::tcrossprod(pin),
                                       perm = TRUE, LDL = FALSE),
                      silent = TRUE), "try-error")
    msg <- if (!ok_jit) {
        e <- try(Matrix::Cholesky(Q + Matrix::Diagonal(n_x, jit), perm = FALSE,
                                  LDL = FALSE), silent = TRUE)
        conditionMessage(attr(e, "condition"))
    } else ""
    rows[[length(rows) + 1L]] <- data.frame(
        cell = k, weight = signif(w[k], 4),
        hyper = paste(sprintf("%s=%.4g", colnames(j$theta_grid), j$theta_grid[k, ]),
                      collapse = " "),
        converged = j$converged[k] %||% NA, score_max = signif(j$score_max[k], 3),
        pd_cond = j$pd_conditioned[k] %||% NA,
        s2z_fb = j$s2z_log_det_fallback[k] %||% NA,
        s2z_indef = j$s2z_field_indefinite[k] %||% NA,
        chol_raw = ok_raw, chol_jit = ok_jit, chol_pinned = ok_pin, err = msg)
}
out <- do.call(rbind, rows)
out <- out[order(-out$weight), ]
if (length(args) >= 3L) utils::write.csv(out, args[3], row.names = FALSE)
print(utils::head(out[, setdiff(names(out), "err")], 40), row.names = FALSE, right = FALSE)
cat("
cells failing the ridged Cholesky:", sum(!out$chol_jit), " weight", sum(out$weight[!out$chol_jit]), "
")
print(out[!out$chol_jit, ], row.names = FALSE, right = FALSE)
cat("\nlatent layout:\n"); str(lay[c("beta_start", "re_start", "field_starts",
                                     "block_start", "block_size",
                                     "field_block_types")])
