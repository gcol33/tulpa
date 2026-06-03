#' Profile the inner Laplace solve by phase
#'
#' Times the sparse joint Laplace solver one phase at a time -- scatter (the
#' Hessian and gradient assembly), factorize (numeric Cholesky), eta, line
#' search, and the rest -- and returns the breakdown as a data frame. The
#' accumulator aggregates across the parallel outer-grid worker threads, so the
#' reported times cover the whole fit rather than only the calling thread.
#'
#' Use it to settle where a per-cell solve spends its time, e.g. whether a slow
#' joint \code{occu_cover()} fit is bound by the assembly scatter or the
#' Cholesky factorize:
#'
#' \preformatted{
#'   p <- tulpa_profile(
#'     tulpa_nested_laplace_joint(..., control = list(integration = "ccd"))
#'   )
#'   print(p)            # rows ordered by time; scatter vs factorize at top
#'   fit <- attr(p, "value")
#' }
#'
#' @param expr An expression that runs a fit (for example a call to
#'   \code{tulpa_nested_laplace_joint()}). Evaluated once, after the profile
#'   counters are reset.
#' @param sort Logical; order rows by descending time. Default \code{TRUE}.
#'
#' @return A data frame with one row per phase and columns \code{phase},
#'   \code{seconds}, \code{calls}, \code{ms_per_call} (mean wall time per phase
#'   call), and \code{share} (fraction of total timed seconds). The fit result
#'   is attached as the \code{"value"} attribute.
#'
#' @export
tulpa_profile <- function(expr, sort = TRUE) {
    cpp_profile_reset()
    value <- expr  # lazy arg: forced here, after the reset
    prof  <- cpp_profile_read()

    us    <- as.numeric(prof$us)
    calls <- as.integer(prof$calls)
    sec   <- us / 1e6
    total <- sum(sec)

    df <- data.frame(
        phase       = as.character(prof$names),
        seconds     = sec,
        calls       = calls,
        ms_per_call = ifelse(calls > 0, (us / 1e3) / calls, 0),
        share       = if (total > 0) sec / total else 0,
        stringsAsFactors = FALSE
    )
    if (isTRUE(sort)) df <- df[order(-df$seconds), , drop = FALSE]
    rownames(df) <- NULL
    attr(df, "value") <- value
    df
}
