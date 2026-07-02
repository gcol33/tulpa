#' Select the symplectic integrator for HMC and NUTS
#'
#' Get or set the symplectic splitting integrator that the exact-MCMC tier
#' (HMC / NUTS) uses to build its trajectory proposals. The integrator is
#' backed by the SIMP library, which supplies leapfrog and the higher-order
#' Yoshida members from one triple-jump composition.
#'
#' The default, `"leapfrog"`, reproduces tulpa's historical trajectory step
#' exactly.
#'
#' `"minerror2"` is a two-stage order-two scheme whose coefficient is tuned to
#' cancel the leading energy error on a Gaussian target. Since mass adaptation
#' drives a posterior toward an isotropic Gaussian, it conserves energy well
#' near the adapted optimum and adapts to larger step sizes, at two gradient
#' evaluations per step. It is the recommended choice when leapfrog's step size
#' is limited by energy error on a near-Gaussian posterior.
#'
#' `"yoshida4"` is an order-4 scheme that also samples reliably (three gradient
#' evaluations per step). `"yoshida6"` and `"yoshida8"` are available but
#' experimental for NUTS: high-order composition integrators have a sharp
#' step-size stability threshold that the dual-averaging adaptation pushes
#' against, so `"yoshida6"` needs a high `adapt_delta` (0.95 or more) and
#' `"yoshida8"` often fails to adapt. For sampling, prefer `"leapfrog"`,
#' `"minerror2"`, or `"yoshida4"`.
#'
#' The choice is process-global (like the gradient mode): set it once before
#' fitting. It is read on the main thread at the start of sampling.
#'
#' @param name Integrator name: `"leapfrog"` (default), `"minerror2"`,
#'   `"yoshida4"`, `"yoshida6"`, or `"yoshida8"`. Omit to query the current
#'   selection without changing it.
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
