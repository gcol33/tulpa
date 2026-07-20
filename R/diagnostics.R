# =============================================================================
# diagnostics.R -- the posterior-diagnostic front door.
#
# A fit's `$draws` may be an autocorrelated MCMC chain, an i.i.d. sample from a
# deterministic approximation (nested Laplace, VI, SMC), or absent entirely
# (a mode + covariance point summary). Each carries a different reliability
# question, and asking the wrong one produces a clean-looking answer that means
# nothing: split-Rhat on i.i.d. draws sits at ~1 and ESS ~ n_draws by
# construction, which reads as a convergence pass while saying nothing about
# approximation bias.
#
# `diagnostics()` reads the draws provenance off the fit and routes to the
# diagnostic that applies. The routing table is a registry keyed by provenance
# kind, so a new engine class is one entry here plus its table builder -- the
# same shape as `.tulpa_diag_measures` (convergence.R) and `BACKEND_REGISTRY`.
# =============================================================================

# Provenance kind -> the diagnostic that answers for it.
#
# `fn`      table builder, called as fn(fit, pars, measures, probs)
# `honours` argument names the builder acts on; the rest are accepted and
#           ignored, so one front-door signature serves every kind
# `what`    the reliability question the kind's table answers, for messages
.tulpa_diag_registry <- list(
  chain = list(
    fn      = function(fit, pars, measures, probs)
                .tulpa_chain_diag_table(fit, pars, measures, probs),
    honours = c("pars", "measures", "probs"),
    what    = "chain mixing (Rhat / ESS / MCSE)"
  ),
  iid = list(
    fn      = function(fit, pars, measures, probs)
                .tulpa_approx_diag_table(fit, pars),
    honours = "pars",
    what    = "approximation reliability (PSIS k-hat / grid quadrature ESS)"
  ),
  point = list(
    fn      = function(fit, pars, measures, probs) {
                message(.tulpa_non_chain_msg(fit))
                NULL
              },
    honours = character(0),
    what    = "nothing -- a point summary carries no posterior sample"
  )
)

#' Posterior diagnostics for a fitted model
#'
#' @description
#' The diagnostic front door for any tulpa fit. What a fit's posterior draws
#' can be asked depends on how they were produced, so this reads the draws
#' provenance and returns the diagnostic that applies:
#'
#' \describe{
#'   \item{MCMC chain draws}{improved Rhat (the maximum of rank-normalized
#'     split-Rhat and folded split-Rhat), bulk / tail / mean / sd / quantile
#'     effective sample size, and Monte Carlo standard errors, following
#'     Vehtari et al. (2021). Split-Rhat is defined for a single chain, so a
#'     result is produced for any number of chains.}
#'   \item{i.i.d. approximation draws}{the approximation-reliability table --
#'     the PSIS tail-shape `pareto_k` scored against the exact inner-Laplace
#'     marginal, the outer-grid quadrature effective sample size, and a
#'     per-parameter posterior summary. See [laplace_diagnostics()] for the
#'     full description of this table and its attributes.}
#'   \item{point summaries}{no sample to diagnose; returns `NULL` with a
#'     message naming the backend.}
#' }
#'
#' Provenance is read from `$draws_kind` (stamped by `tulpa_dispatch`), falling
#' back to the backend registry's `emits` property and then to an inner
#' `$joint_fit`. A fit that predates the tag is treated as a chain, so an older
#' fit is never silently refused.
#'
#' @param fit A `tulpa_fit` (or subclass) carrying posterior `$draws`. Multiple
#'   chains are recognised from a 3D `[iter, chain, param]` draws array, a
#'   `$chain_id` row map, or an `$n_chains` count over chain-major rows.
#' @param pars Optional character vector of parameter names to restrict to.
#' @param measures Character vector selecting which diagnostics to compute, in
#'   output-column order. Available: `"rhat"`, `"rhat_bulk"`, `"rhat_fold"`,
#'   `"ess_bulk"`, `"ess_tail"`, `"ess_mean"`, `"ess_sd"`, `"mcse_mean"`,
#'   `"mcse_sd"`, `"ess_quantile"`, `"mcse_quantile"`. Defaults to the core set
#'   `c("rhat", "ess_bulk", "ess_tail")`. Applies to chain fits; the
#'   approximation-reliability table has a fixed set of columns.
#' @param probs Numeric probabilities for the quantile-based measures
#'   (`"ess_quantile"`, `"mcse_quantile"`); each expands to one column named
#'   e.g. `ess_q5`, `ess_q95`. Default `c(0.05, 0.95)`. Chain fits only.
#' @return For a chain fit, a data frame with a `parameter` column followed by
#'   one column per requested measure; entries are `NA` for parameters that are
#'   constant or have too few draws. For an i.i.d. approximation fit, the
#'   `laplace_diagnostics` table (see that function for its attributes). For a
#'   point fit, `NULL`.
#' @references
#' Vehtari, Gelman, Simpson, Carpenter & Burkner (2021). Rank-normalization,
#'   folding, and localization: an improved Rhat for assessing convergence of
#'   MCMC. \emph{Bayesian Analysis} 16(2):667-718.
#'
#' Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed importance
#'   sampling. \emph{JMLR} 25(72):1-58.
#' @section Extending:
#' `diagnostics()` is an S3 generic so a model package can answer for its own
#' fit class (`diagnostics.ratiod_fit()`, say) rather than shadowing this
#' export with a same-named function. The default method does the provenance
#' routing described above and is what a method should delegate to once it has
#' assembled a draws array.
#'
#' @param ... Passed to the method.
#' @seealso [laplace_diagnostics()] for the approximation-reliability table in
#'   full, [tulpa_draws_array()], [plot_rhat()], [plot_ess()],
#'   [diagnostic_summary()], [check_diagnostics()]
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(60))
#' df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
#'
#' # chain fit -> Rhat / ESS
#' hmc <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
#'              control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
#'                             seed = 1L))
#' diagnostics(hmc)
#'
#' # deterministic fit -> PSIS approximation reliability
#' smc <- tulpa(y ~ x, data = df, family = "poisson", mode = "smc")
#' diagnostics(smc)
#' }
#' @export
diagnostics <- function(fit, ...) {
  UseMethod("diagnostics")
}

#' @rdname diagnostics
#' @export
diagnostics.default <- function(fit, pars = NULL,
                                measures = c("rhat", "ess_bulk", "ess_tail"),
                                probs = c(0.05, 0.95), ...) {
  measures <- match.arg(measures, names(.tulpa_diag_measures), several.ok = TRUE)

  kind <- .tulpa_draws_kind(fit)
  # An untagged fit predates the `emits` property; chain diagnostics are the
  # conservative reading (they are computable on any draws, and a genuine
  # chain is the case that would be wrongly refused).
  if (is.na(kind)) kind <- "chain"

  entry <- .tulpa_diag_registry[[kind]]
  if (is.null(entry)) {
    stop(sprintf(
      "Unknown draws provenance '%s'. Known kinds: %s.",
      kind, paste(names(.tulpa_diag_registry), collapse = ", ")), call. = FALSE)
  }

  entry$fn(fit, pars = pars, measures = measures, probs = probs)
}
