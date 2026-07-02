#' Select the symplectic integrator for HMC and NUTS
#'
#' Get or set the symplectic splitting integrator that the exact-MCMC tier
#' (HMC / NUTS) uses to build its trajectory proposals. The integrator is
#' backed by the SIMP library, which supplies leapfrog and the higher-order
#' Yoshida members from one triple-jump composition.
#'
#' The default, `"leapfrog"`, reproduces tulpa's historical trajectory step
#' exactly. `"yoshida4"` is an order-4 scheme that samples reliably and takes
#' three gradient evaluations per step.
#'
#' `"yoshida6"` and `"yoshida8"` are also available but experimental for NUTS.
#' High-order composition integrators have a sharp step-size stability
#' threshold, which the dual-averaging step-size adaptation targeting a fixed
#' acceptance rate tends to push against. In practice `"yoshida6"` needs a high
#' `adapt_delta` (0.95 or more) to sample well, and `"yoshida8"` often fails to
#' adapt at all. They remain useful for fixed-step-size integration through the
#' SIMP package directly. For sampling, prefer `"leapfrog"` or `"yoshida4"`.
#'
#' The choice is process-global (like the gradient mode): set it once before
#' fitting. It is read on the main thread at the start of sampling.
#'
#' @param name Integrator name: `"leapfrog"` (default), `"yoshida4"`,
#'   `"yoshida6"`, or `"yoshida8"`. Omit to query the current selection
#'   without changing it.
#'
#' @return If `name` is omitted, the current integrator name. If `name` is
#'   given, the previous name is returned invisibly.
#'
#' @examples
#' tulpa_integrator()            # current integrator
#' old <- tulpa_integrator("yoshida4")
#' tulpa_integrator(old)         # restore
#'
#' @export
tulpa_integrator <- function(name) {
  if (missing(name) || is.null(name)) {
    return(tulpa_get_integrator_cpp())
  }
  if (!is.character(name) || length(name) != 1L) {
    stop("`name` must be a single integrator name", call. = FALSE)
  }
  invisible(tulpa_set_integrator_cpp(name))  # C++ validates; errors on unknown
}
