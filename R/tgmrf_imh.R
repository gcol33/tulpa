#' Fit a custom tgmrf latent block
#'
#' @description
#' One front door for inference over a user-defined [tgmrf()] latent block's
#' hyperparameter vector `theta`. `mode` selects the inference target and
#' correction:
#'
#' * `"imh"` (default) -- Tier-1 pseudo-marginal MCMC against an unbiased
#' importance estimate of `p(y | theta)`, with jointly resampled latent draws.
#' * `"nuts"` -- NUTS on the Laplace-approximate theta marginal, followed by
#' PSIS importance correction to the joint posterior. Tier 2: finite weighted
#' approximation. The correction is applied only when its Pareto-k passes
#' the sample-size-dependent reliability threshold.
#' * `"vi"` -- Tier-2 structured: a single-path Pathfinder Gaussian fit of the
#' Laplace-approximate theta marginal.
#' * `"nuts_joint"` -- Tier-1 exact MCMC sampling the FULL joint
#' `(beta, z, theta)` against its true density (requires a
#' C++-backend block, see [tgmrf_cpp()]).
#'
#' The inference method is an argument, not a parallel verb.
#'
#' @param y,n_trials,X Response, binomial trial counts (or `NULL`), and the
#' fixed-effect design matrix.
#' @param block A [tgmrf()] latent block.
#' @param family,phi Observation family and its dispersion.
#' @param re_idx,n_re_groups,sigma_re Optional scalar random-intercept structure.
#' @param mode Inference engine: `"imh"`, `"nuts"`, `"vi"`, or `"nuts_joint"`.
#' @param control A list of numerical / tuning knobs for the chosen `mode`
#' (e.g. `n_iter`, `warmup`, `thin`, `scale` for `"imh"`; `epsilon`,
#' `max_depth`, `target_accept` for the NUTS modes; `n_draws`, `max_lbfgs`,
#' `lbfgs_tol` for `"vi"`; plus the shared `pilot_axis_points`, `max_iter`,
#' `tol`, `n_threads`, `verbose`). An unknown knob for the chosen `mode`
#' errors rather than being silently ignored.
#' @param ... Individual `control` knobs may also be passed directly by name;
#' they are merged into `control` (a named argument here wins over the same
#' name inside `control`).
#' @return A `tulpa_fit` object; `$backend` and `$mode` record
#' the engine used. Every mode reports the fixed effects: `$draws` holds the
#' `beta` columns first, any retained random effects and latent `z`, then `theta`, so
#' `coef()`, `summary()` and `posterior_sample()` read them as on any other
#' fit. `"imh"` returns the selected IS particle retained by pseudo-marginal
#' MH; `"nuts_joint"` samples the full joint density. Corrected `"nuts"`
#' resamples whole joint rows with PSIS weights; `$correction_applied`,
#' `$correction_declined`, `$pareto_k`, `$pareto_k_is_ess` and `$target`
#' describe the result. With `control$is_correction = FALSE`, too few draws,
#' or a k above the gate, NUTS returns its uncorrected Laplace theta chain
#' with Gaussian conditional beta draws, `inference_mode = "structured"`,
#' `inference_tier = 2L`. Corrected NUTS also remains Tier 2 and carries
#' `correction = "psis"`; its resampled rows have `draws_kind = "iid"` and
#' their dependence through the source chain is recorded in `$source_index`
#' and `$source_theta_draws`. `"vi"` returns a Gaussian theta approximation
#' and conditional Gaussian beta draws. `$draws_beta`, `$draws_z` (when
#' available) and `$draws_theta` split the joint rows by block.
#'
#' The inner IS budget is calibrated at the outer Laplace mode, using
#' repeated independent likelihood estimates. The package setting
#' `.TGMRF_IS` sets the target log-estimate SD (1.2), initial budget (16),
#' calibration batches (16), ceiling (65536), and t proposal df (7).
#' `$is_particles` and `$log_estimate_sd` report the chosen budget and measured
#' SD. Each estimate costs that many exact joint-density evaluations plus
#' one inner mode/precision solve. `$inner_solves` counts adapter evaluations
#' after the pilot; `$is_likelihood_evaluations` also includes calibration.
#' @references
#' Andrieu and Roberts (2009), The pseudo-marginal approach for efficient
#' Monte Carlo computations, Annals of Statistics 37:697-725,
#' \doi{10.1214/07-AOS574}. Each MH proposal pays for an unbiased inner
#' estimate and retains that estimate on rejection.
#'
#' Doucet, Pitt, Deligiannidis and Kohn (2015), Efficient implementation of
#' Markov chain Monte Carlo when using an unbiased likelihood estimator,
#' Biometrika 102:295-313, \doi{10.1093/biomet/asu075}. Repeated pilot
#' estimates calibrate the particle budget against log-estimate variability.
#'
#' Vehtari, Simpson, Gelman, Yao and Gabry (2024), Pareto smoothed importance
#' sampling, JMLR 25(72):1-58, <https://www.jmlr.org/papers/v25/19-556.html>.
#' NUTS correction pays one IS estimate per retained theta and a tail fit.
#'
#' Gomez-Rubio and Rue (2018), Markov chain Monte Carlo with the integrated
#' nested Laplace approximation, Statistics and Computing 28:1033-1051,
#' \doi{10.1007/s11222-017-9778-y}.
#' MCMC with an INLA marginal retains its deterministic approximation error.
#'
#' Rue, Martino and Chopin (2009), Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations,
#' JRSS B 71:319-392, \doi{10.1111/j.1467-9868.2008.00700.x}.
#' The inner mode and sparse precision solve build the proposal at each theta.
#'
#' @seealso [tgmrf()] for the block, [tgmrf_cpp()] for the compiled-block form.
#' @examples
#' \donttest{
#' # An iid random intercept written as a tgmrf block over 15 groups; the
#' # inference method is an argument, not a parallel verb. See the
#' # custom-process article at https://gillescolling.com/tulpa/articles/tgmrf.html.
#' set.seed(3)
#' J <- 15L; g <- rep(seq_len(J), each = 6L); n <- length(g); x <- rnorm(n)
#' y <- rpois(n, exp(0.3 + 0.5 * x + rnorm(J, 0, 0.6)[g]))
#' blk <- tgmrf(
#'   Q = function(theta) Matrix::Diagonal(J, exp(-2 * theta[1])),
#'   prior = function(theta) stats::dnorm(theta[1], 0, 1, log = TRUE),
#'   init = c(log_sigma = 0), obs_idx = g)
#' fit <- tulpa_tgmrf(y, rep(1L, n), cbind(1, x), block = blk,
#'                    family = "poisson", mode = "imh",
#'                    control = list(n_iter = 300L, warmup = 150L))
#' mean(exp(fit$draws[, "log_sigma"]))   # posterior mean group SD
#' }
#' @export
tulpa_tgmrf <- function(y, n_trials, X, block,
                        family = "binomial", phi = 1.0,
                        re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                        mode = c("imh", "nuts", "vi", "nuts_joint"),
                        control = list(), ...) {
  mode <- match.arg(mode)
  if (!is.list(control)) stop("`control` must be a list.", call. = FALSE)
  impl <- switch(mode,
                 imh        = .tgmrf_fit_imh,
                 nuts       = .tgmrf_fit_nuts,
                 vi         = .tgmrf_fit_vi,
                 nuts_joint = .tgmrf_fit_nuts_joint)
  base  <- list(y = y, n_trials = n_trials, X = X, block = block,
                family = family, phi = phi, re_idx = re_idx,
                n_re_groups = n_re_groups, sigma_re = sigma_re)
  knobs <- utils::modifyList(control, list(...))
  # The knobs carry the per-mode numerical settings; reject names the chosen
  # mode's fitter does not accept so a typo is not silently dropped.
  allowed <- setdiff(names(formals(impl)), names(base))
  bad <- setdiff(names(knobs), allowed)
  if (length(bad)) {
    stop(sprintf("Unknown control knob(s) for mode = '%s': %s.\n  Allowed: %s.",
                 mode, paste(bad, collapse = ", "),
                 paste(allowed, collapse = ", ")), call. = FALSE)
  }
  fit <- do.call(impl, c(base, knobs))
  fit$mode <- mode
  fit
}

#' Pseudo-marginal independence MH for a tgmrf block
#'
#' The outer t proposal uses the Laplace mode and curvature. Each proposal
#' runs one joint inner solve and integrates the exact likelihood and latent
#' priors using a calibrated number of importance particles. The nonnegative
#' likelihood estimate is unbiased. MH retains this estimate and its selected
#' joint latent particle on rejection, giving the true joint posterior as
#' the stationary marginal (Andrieu and Roberts 2009).
#'
#' @noRd
.tgmrf_fit_imh <- function(y, n_trials, X, block,
                            family = "binomial",
                            phi = 1.0,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            n_iter = 2000L, warmup = n_iter %/% 2L,
                            thin = 1L, scale = 1.0,
                            pilot_axis_points = 5L,
                            max_iter = 50L, tol = 1e-7,
                            n_threads = 1L,
                            verbose = FALSE) {

  if (!inherits(block, "tgmrf")) {
    stop("`block` must be a tgmrf object.", call. = FALSE)
  }
  # Validate the run length before computing the pilot.
  .check_run_length(n_iter, warmup, "tulpa_tgmrf[mode = 'imh']")

  d <- block$theta_dim
  N <- length(y)
  obs_idx <- .tgmrf_obs_idx(block, N)

  # --- 1. Pilot grid fit ----------------------------------------------------
  pl <- .tgmrf_pilot(
    y = y, n_trials = n_trials, X = X, block = block, obs_idx = obs_idx,
    pilot_axis_points = pilot_axis_points,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    max_iter = max_iter, tol = tol, n_threads = n_threads
  )
  pilot <- pl$fit

  # --- 2. Outer mode and curvature ------------------------------------------
  # One inner Laplace solve per theta (shared builder, see
  # .tgmrf_make_log_marginal); no bounds wall, the IMH target is the
  # unbounded posterior over theta. The proposal is centred on the mode the
  # outer mode-find reaches from the pilot argmax and scaled by the inverse
  # curvature there.
  .lm <- .tgmrf_make_log_marginal(
    y = y, n_trials = n_trials, X = X, block = block, obs_idx = obs_idx,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    max_iter = max_iter, tol = tol, n_threads = n_threads
  )
  # Eager structural check at the feasible pilot argmax (non-swallowing): a
  # failure here is a bug, not numerical infeasibility, so it surfaces.
  if (!is.finite(.lm$raw(pl$theta_init))) {
    stop("Pilot log_marginal at the grid argmax is not finite -- ",
         "tighten `bounds` or check the user `Q()`/`prior()`.", call. = FALSE)
  }
  om <- .tgmrf_outer_mode(pl, .lm, block)
  theta_mode <- om$mode
  H_theta <- .spd_inverse(om$cov)
  dimnames(H_theta) <- dimnames(om$cov)

  estimator <- .tgmrf_make_is(.lm, theta_mode, y, n_trials, X, block, obs_idx,
                              re_idx, n_re_groups, sigma_re, family, phi)
  mh <- .imh_laplace_chain(
    evaluate = estimator$evaluate,
    mode    = theta_mode,
    hessian = H_theta,
    n_iter  = n_iter,
    warmup  = warmup,
    thin    = thin,
    scale   = scale,
    verbose = verbose
  )

  draws <- mh$draws
  colnames(draws) <- block$theta_names

  fit <- list(
    draws          = draws,
    means          = colMeans(draws),
    sds            = apply(draws, 2L, stats::sd),
    n_params       = d,
    n_samples      = nrow(draws),
    mean_accept    = mh$mean_accept,
    accept_prob    = mh$accept_prob,
    log_prob       = mh$log_prob,
    pilot          = pilot,
    mode_theta     = theta_mode,
    hessian_theta  = H_theta,
    mode_status    = om$status,
    inference_mode = "exact",
    inference_tier = 1L,
    backend        = "tgmrf_imh",
    target         = "joint posterior p(beta, re, z, theta | y)",
    correction     = "pseudo_marginal",
    selection_reason = "Pseudo-marginal MH on the true joint posterior, using unbiased IS likelihood estimates",
    is_particles   = estimator$n_particles,
    log_estimate_sd = estimator$log_estimate_sd,
    is_target_log_sd = estimator$target_log_sd,
    is_likelihood_evaluations = estimator$likelihood_evaluations(),
    inner_solves   = .lm$inner_solves()
  )
  joint <- do.call(rbind, lapply(mh$states, `[[`, "draw"))
  fit <- .tgmrf_with_joint(fit, draws, joint, X, n_re_groups)
  .finalize_fit(fit, draws_kind = "chain",
               n_fixed = fit$n_fixed, fixed_names = fit$fixed_names,
               param_names = fit$param_names,
               data = list(y = y, n_trials = n_trials, model_matrix = X,
                          family = family, phi = phi))
}
