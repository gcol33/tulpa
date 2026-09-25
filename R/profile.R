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
#' @section Which fits are timed:
#' Only the SPARSE path of the joint nested-Laplace inner solver carries phase
#' timers: \code{tulpa_nested_laplace_joint()} (and the joint drivers built on
#' it) when its inner solve runs sparse, which
#' \code{control = list(force_sparse = TRUE)} selects outright. The
#' single-response solvers behind \code{tulpa_laplace()},
#' \code{tulpa_nested_laplace()} and \code{tulpa()}, the dense joint path and
#' the samplers are not instrumented; profiling one of them records nothing,
#' and \code{tulpa_profile()} then warns rather than returning an all-zero
#' table as if it were a measurement.
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
#' @examples
#' \donttest{
#' # A binomial response over a 30-unit ICAR chain, joint solver forced sparse.
#' set.seed(1)
#' n_s <- 30L; N <- 150L
#' s <- sample.int(n_s, N, replace = TRUE)
#' x <- rnorm(N)
#' y <- rbinom(N, 1, plogis(0.3 * x + sin(s / 5)))
#' nb <- lapply(seq_len(n_s), function(i) setdiff(c(i - 1L, i + 1L), c(0L, n_s + 1L)))
#' prior <- list(type = "icar", n_spatial_units = n_s,
#'               adj_row_ptr = as.integer(c(0L, cumsum(lengths(nb)))),
#'               adj_col_idx = as.integer(unlist(nb)) - 1L,
#'               n_neighbors = lengths(nb), sigma_grid = c(0.5, 1))
#' arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
#'             spatial_idx = s, family = "binomial")
#' tulpa_profile(tulpa_nested_laplace_joint(
#'   responses = list(occ = arm), prior = prior,
#'   control = list(force_sparse = TRUE, progress = FALSE)))
#' }
#' @export
tulpa_profile <- function(expr, sort = TRUE) {
    cpp_profile_reset()
    value <- expr  # lazy arg: forced here, after the reset
    prof  <- cpp_profile_read()

    us    <- as.numeric(prof$us)
    calls <- as.integer(prof$calls)
    sec   <- us / 1e6
    total <- sum(sec)

    # An all-zero table reads as "every phase took no time"; what it means is
    # that the expression never reached an instrumented solver (#887).
    if (all(calls == 0L)) {
        warning("tulpa_profile(): no instrumented phase was reached, so ",
                "nothing was timed. Only the sparse path of the joint ",
                "nested-Laplace solver is instrumented ",
                "(tulpa_nested_laplace_joint(), e.g. with ",
                "control = list(force_sparse = TRUE)); see ?tulpa_profile.",
                call. = FALSE)
    }

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
