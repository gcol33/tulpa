# ============================================================================
# Empirical Bayes over random-effect covariances.
#
# The estimator is the mode of the same outer objective the nested-Laplace
# Sigma integrator maximizes before it places its integration nodes:
#
#     theta_hat = argmax_theta [ log p_Laplace(y | Sigma(theta)) + log pi(theta) ]
#
# so EB and tulpa_re_cov_nested() are not two estimators that agree by
# construction -- they are one objective, .re_cov_theta_fit(), read at two
# different points. EB reports the plug-in at theta_hat; the nested integrator
# carries on and marginalizes around it.
#
# What that costs is the whole reason both exist: conditioning on theta_hat
# drops the hyperparameter uncertainty, so fixed-effect intervals from an EB fit
# are narrower than the marginalized ones -- most visibly with few groups, where
# the variance-component marginal is skewed and its mode sits below its median.
# EB is therefore registered as a Tier-2 ("structured") backend and is opt-in by
# name: auto never selects it.
# ============================================================================


#' Empirical-Bayes random-effect covariances
#'
#' @description
#' Estimate one or more random-effect covariances `Sigma` by maximizing the
#' Laplace marginal likelihood over them (plus the hyperprior), then report the
#' fixed effects conditional on the maximizer. This is the plug-in ("ML-II" /
#' empirical-Bayes) counterpart of [tulpa_re_cov_nested()], which integrates over
#' `Sigma` instead of fixing it at the maximizer.
#'
#' @details
#' Blocks, coordinates and the default hyperprior are exactly those of
#' [tulpa_re_cov_nested()] -- a correlated `(1 + x | g)` term is a full
#' `Sigma = L L'` in log-Cholesky coordinates, an uncorrelated `(1 + x || g)`
#' term is diagonal in log-SD coordinates, and a scalar `(1 | g)` term is the
#' degenerate one-coefficient block. Both functions call the same outer
#' objective and the same optimizer, so `tulpa_eb()$theta_hat` and
#' `tulpa_re_cov_nested()$theta_hat` are the same estimate on the same data.
#'
#' The reported fixed-effect covariance is the conditional one at `theta_hat`
#' (`solve(H_beta)`). It does not include the hyperparameter uncertainty that
#' [tulpa_re_cov_nested()] integrates over, so EB intervals are narrower --
#' increasingly so as the number of groups falls. Use the nested integrator when
#' the variance components themselves, or calibrated fixed-effect intervals, are
#' the target; use EB when the point estimate is, or as a fast starting fit.
#'
#' @param y,n_trials,X,family,phi Passed to [tulpa_laplace()] for the inner
#'   solve. `n_trials = NULL` defaults to 1 (binary / single-trial).
#' @param re_terms Either a single random-effect term or a list of them; see
#'   [tulpa_re_cov_nested()] for the per-term fields.
#' @param prior_sigma,eta Hyperparameters of the default PC + LKJ prior (see
#'   [re_cov_pc_lkj_prior()]). Ignored when `log_prior_theta` is supplied. The
#'   prior is part of the maximized objective, so it regularizes the estimate:
#'   with few groups it is what keeps a block off the `sigma = 0` boundary.
#' @param log_prior_theta Optional `function(theta)` returning a scalar log
#'   prior density on the full stacked parameter vector, replacing the default.
#'   Supply `function(theta) 0` for an unpenalized maximum-marginal-likelihood
#'   estimate (which can collapse to `sigma = 0` on small designs).
#' @param beta_prior Optional Gaussian prior on the fixed effects, threaded into
#'   every inner [tulpa_laplace()] solve (`list(mean, sd)`).
#' @param offset Optional observation-level offset on the linear predictor
#'   (length `length(y)`), e.g. `log(exposure)` for a rate model. Not supported
#'   with `n_quad > 1`, which errors rather than dropping it.
#' @param estimate_phi Estimate the family's dispersion alongside the
#'   random-effect covariances, instead of conditioning on `phi`. When `TRUE`
#'   the supplied `phi` is the starting value and `fit$phi` is the estimate,
#'   with `fit$phi_estimated` distinguishing the two cases. `log(phi)` joins the
#'   maximization as one further coordinate, carrying the exact derivative of
#'   the Laplace log-marginal with respect to it, so the cost is one more
#'   coordinate for BFGS and not a second optimization.
#'
#'   The dispersion enters unpenalized -- the hyperprior covers the covariance
#'   coordinates only -- so this is the ML-II estimate of `phi`, not a MAP under
#'   an undeclared prior.
#'
#'   Available for `neg_binomial_2`, `gaussian` and `gamma`, and refused
#'   elsewhere rather than approximated: `poisson` and `binomial` have no free
#'   dispersion at all, and for the remaining families the derivative is not
#'   registered (see `R/family_dispersion.R` for why `beta` in particular is
#'   held back). Needs `n_quad = 1`.
#' @param n_quad Quadrature order for the inner marginal. `1` (default) uses the
#'   joint-field Laplace inner solve. `> 1` refines it with `n_quad`-point
#'   adaptive Gauss-Hermite quadrature, which requires a single shared grouping
#'   factor; see [tulpa_re_cov_nested()].
#' @param marginal Report fixed-effect intervals that carry the hyperparameter
#'   uncertainty, instead of the intervals conditional on `theta_hat`. The
#'   posterior for `theta` is taken as Gaussian around the maximizer with
#'   covariance `solve(H_theta)`, the inner mode is linearized in `theta`, and
#'   the law of total variance adds `J solve(H_theta) J'` to the conditional
#'   covariance, where `J = d mode / d theta`. Both `H_theta` and `J` come from
#'   one central-difference stencil over the outer objective, costing
#'   `1 + 2k^2` further inner solves for `k` hyperparameter coordinates (`k` is
#'   `1` for a scalar `(1 | g)` block and `3` for a correlated `(1 + x | g)`
#'   one). Default `FALSE`. Widens intervals; never narrows them. When the
#'   variance components themselves are the target, or the correction's two
#'   approximations look strained (a strongly skewed variance-component
#'   marginal), integrate with [tulpa_re_cov_nested()] instead.
#' @param control A named list of numerical knobs: `max_iter`, `tol`,
#'   `n_threads` (inner-solve controls, see [tulpa_laplace()]), and
#'   `outer_maxit` (iteration budget for the maximization over `Sigma`, default
#'   500; applies to the Nelder-Mead simplex used from two parameters up, since
#'   the one-parameter case is bracketed by Brent). Exhausting the budget warns.
#'   `outer_reltol` sets that maximization's convergence tolerance (default
#'   `1e-10` for the gradient-driven methods, `1e-8` for the simplex, which
#'   cannot resolve as finely); it is converted to L-BFGS-B's `factr` on the
#'   bounded path, so one request means the same thing whichever method runs.
#'   `sigma_init` supplies the starting random-effect SD -- a scalar, or one
#'   per coefficient across all blocks -- replacing the method-of-moments guess
#'   taken from a pilot fit at `Sigma = I`. Worth setting when the true scale is
#'   far from 1, where that pilot starts the search on a flat stretch, and when
#'   a run should be reproducible from its inputs rather than from a pilot fit.
#'   It is diagonal: it sets each coefficient's scale and leaves any correlation
#'   to be fitted.
#'   Two further knobs tune `marginal = TRUE` and are inert without it:
#'   `marginal_step` (the stencil step in `theta` space, default `1e-3`) and
#'   `marginal_richardson` (default `FALSE`; evaluate the stencil at `step` and
#'   `step / 2` and extrapolate, turning the `O(step^2)` truncation error into
#'   `O(step^4)` at twice the solves -- worth it only when the inner solver's
#'   own noise sits well below the truncation error, i.e. a tight `tol`).
#'
#' @return A `tulpa_fit` with:
#'   - `mode`, `H_beta`: the fixed-effect (and, on the `n_quad = 1` path, random-
#'     effect) mode and the fixed-effect precision at `theta_hat`, driving
#'     `coef`/`confint`/`vcov`/`summary`.
#'   - `map`: the `Sigma` / `sigma` / `rho` summary at `theta_hat` (a single list
#'     for one block, a named list of them for several).
#'   - `Sigma`: the estimated covariance (a matrix for one block, a named list
#'     of matrices for several).
#'   - `theta_hat`, `log_marginal`, `layout`, `n_blocks`, `n_coefs`.
#'   - `converged`: whether the inner Newton solve at `theta_hat` converged.
#'   - `outer_convergence`: `optim`'s code for the maximization over `Sigma`
#'     (`0` on success). A non-zero value also warns.
#'   - With `marginal = TRUE` and a correction that formed: `cov_marginal` (the
#'     widened fixed-effect covariance, which `vcov` / `summary` / `confint`
#'     then report), `cov_conditional` (the `solve(H_beta)` they would otherwise
#'     have reported, kept so the two are comparable on one fit), `H_theta` and
#'     `theta_cov` (the outer Hessian at `theta_hat` and its inverse, named by
#'     `theta_names`), and the `marginal_step` / `marginal_richardson` actually
#'     used. All absent when the correction was not requested or could not be
#'     formed, so `is.null(fit$cov_marginal)` tests whether the reported
#'     intervals are marginal. A requested correction that fails warns and
#'     leaves the conditional covariance in place.
#'
#' @seealso [tulpa_re_cov_nested()] to integrate over `Sigma` rather than fix it;
#'   [tulpa_laplace()] for the inner solve.
#'
#' @references
#' Casella (1985). An introduction to empirical Bayes data analysis.
#' \emph{The American Statistician} 39(2):83-87.
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations.
#' \emph{JRSS-B} 71(2):319-392.
#' @examples
#' \donttest{
#' set.seed(1)
#' G <- 30L; per <- 10L; n <- G * per
#' grp <- rep(seq_len(G), each = per); x <- rnorm(n)
#' b <- rnorm(G, 0, 0.8)
#' y <- rpois(n, exp(0.3 + 0.5 * x + b[grp]))
#' re_term <- list(idx = grp, n_groups = G, n_coefs = 1L)
#' fit <- tulpa_eb(y, NULL, cbind(1, x), re_term, family = "poisson")
#' fit$map$sigma          # empirical-Bayes RE standard deviation
#' }
#' @export
tulpa_eb <- function(y, n_trials = NULL, X, re_terms,
                     family = "binomial", phi = 1.0,
                     prior_sigma = c(3, 0.05), eta = 2,
                     log_prior_theta = NULL,
                     beta_prior = NULL, offset = NULL, n_quad = 1L,
                     marginal = FALSE,
                     estimate_phi = FALSE,
                     control = list()) {
  tulpa_check_control(control, .CONTROL_KEYS$eb, "tulpa_eb")
  max_iter    <- as.integer(control$max_iter %||% 100L)
  tol         <- control$tol %||% 1e-8
  n_threads   <- as.integer(control$n_threads %||% 1L)
  outer_maxit <- as.integer(control$outer_maxit %||% 500L)
  outer_reltol <- control$outer_reltol
  if (!is.null(outer_reltol) &&
      (!is.numeric(outer_reltol) || length(outer_reltol) != 1L ||
       !is.finite(outer_reltol) || outer_reltol <= 0)) {
    stop("`control$outer_reltol` must be a single positive number.",
         call. = FALSE)
  }
  marginal_step       <- control$marginal_step %||% 1e-3
  marginal_richardson <- isTRUE(control$marginal_richardson)
  if (!is.logical(marginal) || length(marginal) != 1L || is.na(marginal)) {
    stop("`marginal` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(marginal_step) || length(marginal_step) != 1L ||
      !is.finite(marginal_step) || marginal_step <= 0) {
    stop("`control$marginal_step` must be a single positive number.",
         call. = FALSE)
  }
  n_quad <- as.integer(n_quad)
  if (n_quad < 1L) stop("`n_quad` must be >= 1.", call. = FALSE)

  # need_scale = FALSE: the numerical outer Hessian exists to place integration
  # nodes. EB places none, and asking Nelder-Mead for it costs O(k^2) extra
  # objective evaluations -- each one a full inner Laplace solve.
  core <- .re_cov_theta_fit(
    y = y, n_trials = n_trials, X = X, re_terms = re_terms,
    family = family, phi = phi,
    prior_sigma = prior_sigma, eta = eta, log_prior_theta = log_prior_theta,
    beta_prior = beta_prior, n_quad = n_quad,
    max_iter = max_iter, tol = tol, n_threads = n_threads,
    caller = "tulpa_eb", need_scale = FALSE, outer_maxit = outer_maxit,
    offset = offset, estimate_phi = estimate_phi,
    outer_reltol = outer_reltol, sigma_init = control$sigma_init)

  layout    <- core$layout
  theta_hat <- core$theta_hat
  p_fix     <- core$p_fix
  # The dispersion actually fitted: the estimate when it was free, otherwise
  # the value conditioned on. Everything downstream reads this one name, so a
  # fit cannot report one dispersion and have been computed at another.
  phi_fit   <- core$phi_hat %||% phi

  # Re-solve at theta_hat through the SAME closure the optimizer drove, so the
  # reported fit cannot come from a differently-configured inner solve.
  L_hat <- .re_cov_theta_to_L_list(theta_hat, layout)
  fit_hat <- core$inner_fit(L_hat, phi_fit)
  if (is.null(fit_hat) || is.null(fit_hat$mode) ||
      length(fit_hat$log_marginal) != 1L || !is.finite(fit_hat$log_marginal)) {
    stop("tulpa_eb(): the inner Laplace solve failed at the maximizing Sigma. ",
         "Check the data, the hyperprior, and the family.", call. = FALSE)
  }

  beta_names <- colnames(core$X) %||% paste0("beta", seq_len(p_fix))
  beta_mean  <- stats::setNames(fit_hat$mode[seq_len(p_fix)], beta_names)

  map <- .re_cov_map_summary(theta_hat, layout)
  Sigma <- if (length(layout) == 1L) map$Sigma else
    lapply(map, `[[`, "Sigma")

  # Read before clearing.
  log_marg <- fit_hat$log_marginal

  # Hyperparameter-uncertainty correction. Off by default: it costs 1 + 2k^2
  # further inner solves and replaces an exact conditional covariance with an
  # approximate marginal one, so it is a deliberate choice rather than a
  # silent upgrade. A failed correction leaves the conditional covariance in
  # place and says so -- reporting the conditional as if it were marginal is
  # the one outcome worth a warning.
  corr <- NULL
  if (isTRUE(marginal)) {
    corr <- .eb_marginal_correction(
      core = core, theta_hat = theta_hat, p_fix = p_fix,
      H_beta = fit_hat$H_beta,
      step = marginal_step, richardson = marginal_richardson)
    if (is.null(corr)) {
      warning("tulpa_eb(): the marginal correction could not be formed (a ",
              "stencil solve failed, or the outer Hessian was not positive ",
              "definite -- a variance component on the boundary is the usual ",
              "cause). Reporting the conditional covariance at theta_hat, so ",
              "intervals exclude hyperparameter uncertainty.", call. = FALSE)
    }
  }
  # Carried only when the correction succeeded, so `names(fit)` on a plain EB
  # fit is unchanged and `is.null(fit$cov_marginal)` is an honest test of
  # whether the reported intervals are marginal. `vcov()` and the summary table
  # prefer `cov_marginal` over `H_beta`, so attaching it is what switches them;
  # `cov_conditional` rides along so the two are comparable on one fit.
  corr_fields <- if (is.null(corr)) list() else list(
    cov_marginal    = corr$cov_marginal,
    cov_conditional = corr$cov_conditional,
    theta_cov       = corr$theta_cov,
    theta_names     = corr$theta_names,
    H_theta         = corr$H_theta,
    marginal_step   = corr$step,
    marginal_richardson = corr$richardson
  )

  # `c()` concatenates rather than overwrites, so a name appearing in both
  # fit_hat and the list below lands TWICE: `names(fit)` shows it duplicated and
  # `fit$name` silently answers with whichever came first. So clear every name
  # this function is about to set. That includes the inner solve's provenance
  # stamps -- fit_hat is itself a finalized tulpa_laplace fit, and since
  # .finalize_fit() only fills absent fields, a surviving `backend = "laplace"`
  # would label this fit a Laplace one.
  # `phi` is cleared alongside the provenance stamps: fit_hat carries the value
  # its inner solve ran at, and the list below sets the reported one. Leaving
  # both would duplicate the name and answer with whichever came first.
  fit_hat[c("backend", "draws_kind", "inference_mode", "inference_tier",
            "selection_reason", "log_marginal", "phi")] <- NULL

  .finalize_fit(c(fit_hat, list(
    map          = map,
    Sigma        = Sigma,
    beta         = beta_mean,
    means        = beta_mean,
    param_names  = beta_names,
    process_info = list(list(name = "fixed_effects", p = p_fix,
                             coef_names = beta_names)),
    n_params     = p_fix,
    N            = length(y),
    theta_hat    = theta_hat,
    log_marginal = log_marg,
    # The dispersion the fit was computed at, always present so downstream code
    # need not know whether it was estimated. `phi_estimated` is what
    # distinguishes an estimate from the value the caller conditioned on --
    # reporting `phi` alone would make the two indistinguishable.
    phi          = phi_fit,
    phi_estimated = isTRUE(core$estimate_phi),
    # optim's code for the outer maximization, for a programmatic check; a
    # non-zero value has already warned.
    outer_convergence = as.integer(core$opt$convergence),
    layout       = layout,
    n_blocks     = length(layout),
    n_coefs      = vapply(layout, `[[`, integer(1), "nc")
  ), corr_fields), backend = "eb", n_fixed = p_fix, fixed_names = beta_names)
}
