#' Profile a fit by solver phase
#'
#' Times a fit one solver phase at a time -- the inner Newton solve's eta,
#' scatter (the Hessian and gradient assembly), factorize (Cholesky), line
#' search and final pass, the nested-Laplace outer grid's cell solves, and the
#' NUTS sampler's iterations and gradient evaluations -- and returns the
#' breakdown as a data frame. The accumulator aggregates across the parallel
#' outer-grid worker threads and across concurrently run chains, so the
#' reported times cover the whole fit rather than only the calling thread
#' (and can therefore exceed its wall time).
#'
#' Use it to settle where a fit spends its time, e.g. whether a slow joint
#' \code{occu_cover()} fit is bound by the assembly scatter or the Cholesky
#' factorize:
#'
#' \preformatted{
#'   p <- tulpa_profile(
#'     tulpa_nested_laplace_joint(..., control = list(integration = "ccd"))
#'   )
#'   print(p)            # rows ordered by time; scatter vs factorize at top
#'   fit <- attr(p, "value")
#' }
#'
#' Timing is off outside \code{tulpa_profile()}: an ordinary fit pays one
#' flag read per phase scope and nothing else.
#'
#' @section Which fits are timed:
#' \itemize{
#'   \item The single-response Laplace solve behind \code{tulpa_laplace()},
#'     \code{tulpa(mode = "laplace")} and every per-cell inner solve of
#'     \code{tulpa_nested_laplace()} / \code{tulpa(mode = "nested_laplace")}:
#'     phases \code{eta}, \code{scatter}, \code{factorize},
#'     \code{line_search}, \code{log_det}, \code{log_lik_prior},
#'     \code{hessian_extract} and \code{inner_diagnostics}.
#'   \item The joint solver behind \code{tulpa_nested_laplace_joint()}, on both
#'     its dense and its sparse path (the sparse one adds
#'     \code{pattern_build} and \code{prep}).
#'   \item The nested-Laplace outer grid, single-response and joint:
#'     \code{outer_grid_cell}, one call per solved cell.
#'   \item The samplers: \code{nuts_warmup} and \code{nuts_sampling}, one call
#'     per NUTS iteration per chain, and \code{gradient}, one call per
#'     log-density gradient evaluation (NUTS leapfrog steps, static HMC, the
#'     step-size search and the other samplers that evaluate the gradient
#'     through the engine).
#' }
#' An expression that reaches none of these -- a Gibbs or quadrature backend
#' that runs neither an engine Newton solve nor an engine gradient, or no fit
#' at all -- records nothing, and \code{tulpa_profile()} then warns rather than returning an
#' all-zero table as if it were a measurement.
#'
#' @section Enclosing phases:
#' \code{outer_grid_cell}, \code{nuts_warmup} and \code{nuts_sampling}
#' ENCLOSE the leaf phases timed inside them (a cell's inner Newton phases, an
#' iteration's gradient evaluations), so their seconds overlap the leaves'
#' rather than adding to them. Their \code{share} is \code{NA}; the shares of
#' the leaf phases sum to one.
#'
#' @param expr An expression that runs a fit (for example a call to
#'   \code{tulpa_laplace()} or \code{tulpa()}). Evaluated once, after the
#'   profile counters are reset and timing is switched on.
#' @param sort Logical; order rows by descending time. Default \code{TRUE}.
#'
#' @return A data frame with one row per phase and columns \code{phase},
#'   \code{seconds}, \code{calls}, \code{ms_per_call} (mean wall time per phase
#'   call), and \code{share} (fraction of the total timed seconds of the leaf
#'   phases; \code{NA} for an enclosing phase). The fit result is attached as
#'   the \code{"value"} attribute.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 200L; X <- cbind(1, rnorm(n))
#' y <- rbinom(n, 1, plogis(X %*% c(0, 0.5)))
#' tulpa_profile(tulpa_laplace(y, rep(1L, n), X, family = "binomial"))
#' }
#' @export
tulpa_profile <- function(expr, sort = TRUE) {
    cpp_profile_reset()
    was_on <- cpp_profile_enable(TRUE)
    # Restored however `expr` exits, so an error inside the fit does not leave
    # every later fit in the session paying for the clock reads.
    on.exit(cpp_profile_enable(was_on), add = TRUE)
    value <- expr  # lazy arg: forced here, after the reset
    prof  <- cpp_profile_read()

    us    <- as.numeric(prof$us)
    calls <- as.integer(prof$calls)
    leaf  <- !as.logical(prof$enclosing)
    sec   <- us / 1e6
    # An enclosing phase overlaps the leaves timed inside it, so the total the
    # shares divide by is the leaves' alone.
    total <- sum(sec[leaf])

    # An all-zero table reads as "every phase took no time"; what it means is
    # that the expression never reached an instrumented solver (#887).
    if (all(calls == 0L)) {
        warning("tulpa_profile(): no instrumented phase was reached, so ",
                "nothing was timed. The Laplace, nested-Laplace and joint ",
                "solvers and the NUTS sampler are instrumented; see ",
                "?tulpa_profile for which fits are timed.",
                call. = FALSE)
    }

    df <- data.frame(
        phase       = as.character(prof$names),
        seconds     = sec,
        calls       = calls,
        ms_per_call = ifelse(calls > 0, (us / 1e3) / calls, 0),
        share       = ifelse(leaf, if (total > 0) sec / total else 0,
                             NA_real_),
        stringsAsFactors = FALSE
    )
    if (isTRUE(sort)) df <- df[order(-df$seconds), , drop = FALSE]
    rownames(df) <- NULL
    attr(df, "value") <- value
    df
}
