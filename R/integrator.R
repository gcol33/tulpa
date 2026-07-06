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
#' `"adaptive2"` and `"adaptive3"` are step-adapted versions of the two- and
#' three-stage schemes. Rather than fixing the coefficient in advance, each NUTS
#' chain resolves it at the end of warmup for its own operating point: the
#' coefficient that minimizes the worst-case energy error over the band of
#' dimensionless steps the chain actually takes, read off from the adapted mass
#' matrix and the local posterior curvature. Where `"minerror2"` is optimal only
#' in the small-step limit, `"adaptive2"` tracks the chain's realised step band;
#' `"adaptive3"` spends a third gradient per step to hold a small error over a
#' wider band. Both run a fixed placeholder of the same stage family during
#' warmup, so the dual-averaged step size carries over. Step-adaptation applies
#' to NUTS (the default sampler); the fixed-trajectory HMC path uses the
#' placeholder.
#'
#' `"mts"` is a multiple-time-stepping (RESPA) integrator. It splits the force
#' into a stiff but cheap prior part -- the Gaussian latent structure -- and a
#' smooth but expensive likelihood part. Each NUTS trajectory step takes
#' `mts_substeps` inner leapfrog substeps against the prior force while
#' evaluating the likelihood gradient only once, so the outer step can be larger
#' without the stiff prior forcing it small. It pays one full gradient per step
#' (as leapfrog does) plus `mts_substeps` cheap prior gradients, and helps most
#' when the latent field is stiff relative to a comparatively flat likelihood.
#' Like the other schemes it applies to NUTS.
#'
#' The choice is process-global (like the gradient mode): set it once before
#' fitting. It is read on the main thread at the start of sampling.
#'
#' @param name Integrator name: `"leapfrog"` (default), `"minerror2"`,
#'   `"adaptive2"`, `"adaptive3"`, `"mts"`, `"yoshida4"`, `"yoshida6"`, or
#'   `"yoshida8"`. Omit to query the current selection without changing it.
#' @param mts_substeps Number of inner prior-force substeps per trajectory step
#'   for the `"mts"` integrator (default 4). Ignored by the other integrators.
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
tulpa_integrator <- function(name, mts_substeps = 4L) {
  if (missing(name) || is.null(name)) {
    return(tulpa_get_integrator_cpp())
  }
  if (!is.character(name) || length(name) != 1L) {
    stop("`name` must be a single integrator name", call. = FALSE)
  }
  if (!is.numeric(mts_substeps) || length(mts_substeps) != 1L ||
      mts_substeps < 1L) {
    stop("`mts_substeps` must be a single integer >= 1", call. = FALSE)
  }
  # C++ validates the name; errors on unknown.
  invisible(tulpa_set_integrator_cpp(name, as.integer(mts_substeps)))
}
