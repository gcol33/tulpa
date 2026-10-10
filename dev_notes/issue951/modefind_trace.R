# Trace the placement mode-find of the pilot + prune issue951 fixture: every
# finite-difference stencil's centre, gradient and Hessian diagonal.
args <- commandArgs(TRUE)
if (length(args) >= 1L && nzchar(args[1])) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(tulpa))
env <- new.env(); env$test_that <- function(...) invisible(NULL)
sys.source(file.path("tests", "testthat", "test-joint-placement-block-scale.R"),
           envir = env, toplevel.env = env, keep.source = FALSE)
fx <- env$.bs_fixture()
ns <- asNamespace("tulpa")
orig_st <- get(".joint_ccd_fd_stencil", ns)
wrap_st <- function(u, eval1, h, ...) {
    st <- orig_st(u, eval1, h, ...)
    cat(sprintf("stencil at u = (%s)\n  f0 %s grad (%s)\n  diagH (%s)\n",
                paste(sprintf("%.3f", u), collapse = ", "),
                format(st$f0 %||% NA),
                paste(sprintf("%.3g", st$grad), collapse = ", "),
                paste(sprintf("%.3g", diag(st$hess)), collapse = ", ")))
    st
}
`%||%` <- function(x, y) if (is.null(x)) y else x
orig_pm <- get(".nl_placement_mode", ns)
wrap_pm <- function(tg, w, tags, eval_logpost, set_warm = NULL, axis_nodes = NULL) {
    cat("placement mode: cells", nrow(tg), "start cell",
        paste(sprintf("%s=%.4g", colnames(tg), tg[which.max(w), ]), collapse = " "), "\n")
    orig_pm(tg, w, tags, eval_logpost, set_warm, axis_nodes)
}
orig_mf <- get(".joint_ccd_modefind", ns)
wrap_mf <- function(u0, eval1, lower, upper, h, ...) {
    e2 <- function(U) {
        f <- eval1(U)
        if (nrow(U) == 7L) cat("  line search f:", paste(sprintf("%.3f", as.numeric(f)), collapse = " "), "
")
        f
    }
    cat("  box lower (", paste(sprintf("%.2f", lower), collapse = ", "), ") upper (",
        paste(sprintf("%.2f", upper), collapse = ", "), ") h (", paste(sprintf("%.3f", h), collapse=", "), ")
")
    orig_mf(u0, e2, lower, upper, h, ...)
}
unlockBinding(".joint_ccd_modefind", ns); assign(".joint_ccd_modefind", wrap_mf, ns)
unlockBinding(".joint_ccd_fd_stencil", ns); assign(".joint_ccd_fd_stencil", wrap_st, ns)
unlockBinding(".nl_placement_mode", ns); assign(".nl_placement_mode", wrap_pm, ns)
fit <- tulpa_nested_laplace_joint(
    fx$responses, fx$prior, copy = fx$copy,
    control = list(diagnose_k = FALSE, integration = "grid_adaptive",
                   recenter_pilot = isTRUE(as.logical(args[2] %||% "TRUE")),
                   prune = TRUE))
