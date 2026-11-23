# ============================================================================
# Empirical Bayes over random-effect covariances.
#
# The estimator is the mode of the log posterior the nested-Laplace Sigma
# integrator integrates:
#
#     theta_hat = argmax_theta [ log p_Laplace(y | Sigma(theta)) + log pi(theta) ]
#
# found by `.eb_theta_fit()`. EB reports the plug-in at theta_hat;
# tulpa_re_cov_nested() integrates the same log posterior on the joint door and
# reads this mode as its own `theta_hat`.
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
#' `tulpa_re_cov_nested()$theta_hat` are the same estimate on the same data --
#' which requires `hyperprior` to default the same way on both: `"proper"`, the
#' PC + LKJ prior (see [re_cov_pc_lkj_prior()]) at the anchor every scale axis
#' of the engine defaults to, which at small G keeps a block off the
#' `sigma = 0` boundary. `hyperprior = "flat"` maximizes the marginal likelihood
#' alone.
#'
#' Under the proper hyperprior the estimate is a posterior mode, and a mode
#' moves under a change of coordinates: `theta_hat` is the mode in the
#' log-Cholesky coordinates above, and `map` is the covariance read off at that
#' point, not the mode of the posterior on `(sigma, rho)`. Under
#' `hyperprior = "flat"` the maximizer of the marginal likelihood is the same
#' covariance in any coordinates.
#'
#' The marginal likelihood is marginal over the fixed effects too: the inner
#' Laplace approximation integrates `beta` out together with the random
#' effects (under `beta_prior`, or the engine default `N(0, 2.5^2)`)
#' rather than profiling it. `hyperprior = "flat"` is therefore a RESTRICTED,
#' REML-type estimate, not maximum likelihood: for a gaussian response it is
#' REML, and for a non-gaussian one it agrees with
#' `glmmTMB(..., REML = TRUE)`, not with the ML fit of `lme4::glmer()` or
#' `glmmTMB(..., REML = FALSE)`. On a binomial random intercept with 40 groups
#' of 6 the three read RE SDs of 0.7931 (`tulpa_eb`), 0.7931 (glmmTMB REML) and
#' 0.7594 (ML); the fixed effects at `theta_hat` match the REML fit's as well.
#' The restriction is what removes ML's downward bias in the variance
#' components when `beta` is estimated from the same data, and it carries over
#' to `n_quad > 1` and to `estimate_phi = TRUE`. Validating against `glmer()`
#' should expect exactly that gap.
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
#' @param phi2 Optional second dispersion, threaded into every inner
#'   [tulpa_laplace()] solve: the Student-t degrees of freedom (`family = "t"`,
#'   default 4 when `NULL`) or the Tweedie variance power (`family = "tweedie"`,
#'   required -- a defaulted power would be a statistical decision the caller
#'   never made). A `phi2` supplied for any other family errors rather than being
#'   ignored. It is conditioned on, never estimated: `estimate_phi` covers `phi`
#'   alone.
#' @param re_terms Either a single random-effect term or a list of them; see
#'   [tulpa_re_cov_nested()] for the per-term fields.
#' @param prior_sigma,eta Hyperparameters of the PC + LKJ prior used when
#'   `hyperprior = "proper"` (see [re_cov_pc_lkj_prior()]); `NULL` (the
#'   defaults) is `c(3, 0.01)` and 2. Ignored when
#'   `hyperprior = "flat"` or `log_prior_theta` is supplied. When active, the
#'   prior is part of the maximized objective, so it regularizes the estimate:
#'   with few groups it is what keeps a block off the `sigma = 0` boundary.
#' @param hyperprior `"proper"` (default) or `"flat"`. `"flat"` maximizes with
#'   `log_prior_theta` the zero function -- an unpenalized restricted
#'   (REML-type) estimate, `beta` integrated out rather than profiled (see
#'   Details), which can reach the `sigma = 0` boundary on small
#'   designs (see the `"lower end of the search bracket"` warning). `"proper"`
#'   builds the PC + LKJ prior from `prior_sigma` / `eta`, regularizing the
#'   estimate away from that boundary. Ignored when `log_prior_theta` is
#'   supplied. Must match `hyperprior` on the paired [tulpa_re_cov_nested()]
#'   call for the two to share `theta_hat`.
#' @param log_prior_theta Optional `function(theta)` returning a scalar log
#'   prior density on the full stacked parameter vector, overriding
#'   `hyperprior` entirely. Default `NULL`, which defers to `hyperprior`.
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
#'   coordinate for the outer mode-find and not a second optimization.
#'
#'   The dispersion enters unpenalized -- the hyperprior covers the covariance
#'   coordinates only -- so this is the ML-II estimate of `phi`, not a MAP under
#'   an undeclared prior. Like the covariances it is read off the likelihood
#'   marginal over `beta`, so it is the restricted (REML-type) estimate: for a
#'   gaussian response, the REML residual variance.
#'
#'   Available for every family carrying a dispersion, which is every front-door
#'   family except `poisson`, `binomial` and `truncated_poisson` -- those have no
#'   free dispersion at all, so estimating one is a category error rather than a
#'   missing feature, and it is refused. Needs `n_quad = 1`.
#'
#'   Alongside `X_zi` both mixture kinds are covered. A hurdle (a zero-truncated
#'   base) has zero branch `log(pi)`, which carries no dispersion, so the base
#'   family's registered derivative is already the mixture's. Genuine zero
#'   inflation has zero branch `log(pi + (1 - pi) P(Y = 0))`, which depends on
#'   `phi` through `P(Y = 0)` and couples it to both linear predictors; that
#'   branch is supplied for `neg_binomial_2`, the only untruncated mixture family
#'   here with a free dispersion. Other untruncated bases are refused rather than
#'   handed the base derivative under a model it does not describe.
#' @param X_zi Optional zero-inflation design matrix (`length(y)` rows), making
#'   the model a two-process mixture: each observation is a structural zero with
#'   probability `plogis(X_zi beta_zi)` and otherwise follows `family`. Paired
#'   with a zero-truncated family it is the hurdle model. The random effects
#'   enter the count predictor only, and the maximization is over the same
#'   covariance coordinates -- the mixture changes the inner solve, not the
#'   outer objective's parameters. The ZI coefficients are reported alongside
#'   the count ones in `coef()` / `vcov()`, so the fixed block is
#'   `ncol(X) + ncol(X_zi)` wide. Needs `n_quad = 1`: the adaptive
#'   Gauss-Hermite inner marginal runs through a single-predictor oracle.
#' @param zi_prior_sd Prior SD on `beta_zi`, keeping the logit identified where
#'   a level carries no zeros (the likelihood alone would send it to `-Inf`).
#'   Ignored when `X_zi` is `NULL`.
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
#'   `outer_maxit` (the round budget of the maximization over `Sigma`, default
#'   500). The maximization is the outer mode-find the nested-Laplace doors
#'   place their grids with: damped Newton rounds on the analytic gradient and
#'   outer Hessian where the family has them, on finite differences of the
#'   objective where it does not, every log scale boxed to `[1e-4, 1e3]`.
#'   Exhausting the budget warns. `outer_reltol` sets its convergence tolerance
#'   (default `1e-10`), relative to the log posterior: the search stops when
#'   the gain its next Newton step predicts falls below it, after taking that
#'   step.
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
#'   - `outer_convergence`: the code of the maximization over `Sigma`, `0` on
#'     convergence and `1` where the `outer_maxit` round budget stopped it
#'     (which also warns).
#'   - `singular`: `TRUE` when the estimated covariance is singular, a boundary
#'     fit in the sense of `lme4::isSingular()`: a standard deviation collapsed
#'     to the lower end of its bracket, or a correlated block whose Cholesky
#'     diagonal did, so that one coefficient's effect is a linear function of
#'     the others (its SD zero or its multiple correlation one). The marginal
#'     likelihood then has its supremum on the boundary of the covariance and
#'     the search stops at the bracket, which also warns.
#'   - `boundary`: one row per log-scale coordinate at either end of its
#'     bracket (`block`, `coef`, `end` `"lower"` / `"upper"`, and `sigma`, that
#'     coefficient's standard deviation); no rows when none is.
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
                     family = "binomial", phi = 1.0, phi2 = NULL,
                     prior_sigma = NULL, eta = NULL,
                     hyperprior = c("proper", "flat"),
                     log_prior_theta = NULL,
                     beta_prior = NULL, offset = NULL, n_quad = 1L,
                     marginal = FALSE,
                     estimate_phi = FALSE,
                     X_zi = NULL, zi_prior_sd = 2.5,
                     control = list()) {
  tulpa_check_control(control, .CONTROL_KEYS$eb, "tulpa_eb")
  .device_session <- .device_start(control)
  on.exit(cpp_device_end(.device_session), add = TRUE)
  # A family typo reached the inner solve and came back as 39 warnings and "the
  # inner Laplace solve failed" (gcol33/tulpa#886); name it, as
  # tulpa_re_cov_nested() does.
  family <- .canonical_family(family)
  .family_or_stop(family)
  hyperprior <- .hp_choice(match.arg(hyperprior))
  log_prior_theta <- .re_cov_resolve_hyperprior(hyperprior, log_prior_theta)
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
  # `marginal` is in .CONTROL_KEYS$eb so tulpa() can forward it to the argument
  # below, which is where it lives here. Accepting it in `control` too would
  # take it and do nothing with it. Tested by name rather than with `$`, which
  # partial-matches on lists and would read `marginal_step` as `marginal`.
  if ("marginal" %in% names(control)) {
    stop("`marginal` is an argument of tulpa_eb(), not a control knob: call ",
         "tulpa_eb(..., marginal = TRUE). It is accepted in `control` by ",
         "tulpa() only, which forwards it here.", call. = FALSE)
  }
  if (!is.logical(marginal) || length(marginal) != 1L || is.na(marginal)) {
    stop("`marginal` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(marginal_step) || length(marginal_step) != 1L ||
      !is.finite(marginal_step) || marginal_step <= 0) {
    stop("`control$marginal_step` must be a single positive number.",
         call. = FALSE)
  }
  n_quad <- .check_n_quad(n_quad)

  core <- .eb_theta_fit(
    y = y, n_trials = n_trials, X = X, re_terms = re_terms,
    family = family, phi = phi, phi2 = phi2,
    prior_sigma = prior_sigma, eta = eta, log_prior_theta = log_prior_theta,
    beta_prior = beta_prior, n_quad = n_quad,
    max_iter = max_iter, tol = tol, n_threads = n_threads,
    caller = "tulpa_eb", outer_maxit = outer_maxit,
    offset = offset, estimate_phi = estimate_phi,
    outer_reltol = outer_reltol, sigma_init = control$sigma_init,
    X_zi = X_zi, zi_prior_sd = zi_prior_sd)

  layout    <- core$layout
  theta_hat <- core$theta_hat
  # The whole fixed prefix of the mode. Under zero inflation that is
  # [beta | beta_zi], and both blocks are reported: the mixture's logit
  # coefficients are fixed effects of the model, not nuisance scaling.
  p_fix     <- core$p_fixed
  # The dispersion actually fitted: the estimate when it was free, otherwise
  # the value conditioned on. Everything downstream reads this one name, so a
  # fit cannot report one dispersion and have been computed at another.
  phi_fit   <- core$phi_hat %||% phi

  # Re-solve at theta_hat through the SAME closure the optimizer drove, so the
  # reported fit cannot come from a differently-configured inner solve. It keeps
  # the joint latent precision (`H_latent`), so posterior_predict() and the
  # WAIC/LOO criteria draw the random effects jointly with the fixed effects
  # rather than holding them at the mode (gcol33/tulpa#871). The raw kernel copy
  # is dropped: `H_latent` is the one a caller reads. The AGHQ inner marginal
  # (n_quad > 1) integrates each group out and has no joint field to keep.
  L_hat <- .re_cov_theta_to_L_list(theta_hat, layout)
  fit_hat <- core$inner_fit(L_hat, phi_fit, joint_hessian = (n_quad == 1L))
  if (!is.null(fit_hat)) fit_hat$H_joint <- NULL
  if (is.null(fit_hat) || is.null(fit_hat$mode) ||
      length(fit_hat$log_marginal) != 1L || !is.finite(fit_hat$log_marginal)) {
    stop("tulpa_eb(): the inner Laplace solve failed at the maximizing Sigma. ",
         "Check the data, the hyperprior, and the family.", call. = FALSE)
  }

  beta_names <- c(colnames(core$X) %||% paste0("beta", seq_len(core$p_fix)),
                  if (core$p_zi > 0L)
                    colnames(core$X_zi) %||% paste0("zi_", seq_len(core$p_zi)))
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
    # The correction propagates the curvature of the FULL outer objective, so an
    # estimated dispersion rides along as the log-phi coordinate: `theta_hat_full`
    # is [covariance coordinates..., log phi_hat] when phi was estimated and the
    # covariance half alone otherwise. Passing it lets the phi column of the mode
    # Jacobian carry dispersion uncertainty into the fixed-effect intervals.
    corr <- .eb_marginal_correction(
      core = core, theta_hat = core$theta_hat_full, p_fix = p_fix,
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
  # `phi_estimated` goes with it for exactly that reason: the inner solve
  # CONDITIONED on a phi and is stamped FALSE, so leaving its copy in would
  # answer FALSE on the one door that does estimate one (gcol33/tulpa#849).
  fit_hat[c("backend", "draws_kind", "inference_mode", "inference_tier",
            "selection_reason", "log_marginal", "phi", "phi_estimated")] <- NULL

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
    # The outer maximization's code, for a programmatic check; a
    # non-zero value has already warned.
    outer_convergence = as.integer(core$opt$convergence),
    singular     = core$singular,
    boundary     = core$boundary,
    layout       = layout,
    n_blocks     = length(layout),
    n_coefs      = vapply(layout, `[[`, integer(1), "nc")
  ), corr_fields), backend = "eb", n_fixed = p_fix, fixed_names = beta_names)
}

# The empirical-Bayes objective: everything up to and including the mode of
# g(theta) = log_marginal(Sigma(theta)) + log_prior_theta(theta) over the
# stacked covariance coordinates (and log phi with `estimate_phi`).
#
# It returns the live `inner_fit` closure rather than a value, so tulpa_eb()
# re-solves at theta_hat through the SAME closure the mode-find drove and no
# second inner-solve configuration can drift from it; the marginal correction
# (`.eb_marginal_correction()`) differentiates the same objective through
# `exact_grad_at`. tulpa_re_cov_nested() reads its mode as `theta_hat`.
.eb_theta_fit <- function(y, n_trials, X, re_terms,
                          family, phi, phi2 = NULL,
                          prior_sigma, eta, log_prior_theta,
                          beta_prior, n_quad,
                          max_iter, tol, n_threads,
                          caller,
                          outer_maxit = 500L,
                          offset = NULL,
                          estimate_phi = FALSE,
                          outer_reltol = NULL,
                          sigma_init = NULL,
                          X_zi = NULL, zi_prior_sd = 2.5,
                          weights = NULL,
                          warn_boundary = TRUE) {
  re_terms <- .as_re_terms_list(re_terms)
  if (!is.matrix(X)) X <- as.matrix(X)
  .require_fixed_effects(X, caller)
  vd <- .validate_glm_design(y, X, n_trials, caller)
  n_trials <- vd$n_trials
  # Zero inflation adds a SECOND linear predictor, so the inner solve's mode is
  # [beta | beta_zi | RE] rather than [beta | RE]. `p_fix` stays the count
  # block -- the AGHQ profile below optimizes over it alone -- and `p_fixed` is
  # the whole fixed prefix anything reading the mode has to skip.
  if (!is.null(X_zi)) {
    X_zi <- as.matrix(X_zi)
    if (nrow(X_zi) != length(y)) {
      stop(caller, "(): `X_zi` has ", nrow(X_zi), " rows but y has length ",
           length(y), ".", call. = FALSE)
    }
  }
  p_zi <- if (is.null(X_zi)) 0L else ncol(X_zi)
  has_zi <- p_zi > 0L
  layout <- .re_cov_block_layout(re_terms, length(y))
  # Without random-effect terms the outer objective has no covariance
  # coordinate; it still has one when the dispersion is estimated, the log phi
  # of a fixed-effect model, which is then the whole outer vector.
  if (length(layout) == 0L && !isTRUE(estimate_phi)) {
    stop(caller, "(): no random-effect terms and no dispersion to estimate. ",
         "The outer objective is over the random-effect covariances (and, ",
         "with `estimate_phi = TRUE`, the dispersion), so there is nothing to ",
         "fit; use tulpa_laplace() for a fixed-effect-only model at a given ",
         "phi.", call. = FALSE)
  }
  k <- sum(vapply(layout, `[[`, integer(1), "k"))
  if (is.null(log_prior_theta)) {
    log_prior_theta <- .re_cov_joint_prior(layout, prior_sigma, eta)
  }

  # The second dispersion reaches the inner solve rather than being defaulted
  # there. Left unthreaded, `t` fits at the compiled kernel's built-in degrees of
  # freedom whatever the caller asked for, and `tweedie` fails inside the inner
  # solve's tryCatch as a generic convergence failure. Both are resolved here:
  # a phi2 supplied for a family that takes none errors (matching
  # tulpa_laplace()), and tweedie's variance power is required by name rather
  # than defaulted, since a silently chosen power is a statistical decision the
  # caller never made.
  phi2 <- .phi2_or_stop(family, phi2)
  if (identical(.family_base(family), "tweedie")) .tweedie_power(phi2)

  # The dispersion rides as one extra coordinate, log phi, appended after the
  # covariance coordinates. Refused rather than approximated wherever the exact
  # gradient for it is unavailable: the whole point of estimating phi here is
  # that the outer objective is differentiated exactly, and a derivative-free
  # search over a k+1 dimensional simplex is a different (and much worse)
  # method wearing the same argument.
  if (isTRUE(estimate_phi)) {
    if (is.null(.family_dphi(family))) {
      stop(caller, "(): `estimate_phi = TRUE` is not available for family '",
           family, "'. The dispersion derivative of the Laplace log-marginal ",
           "is registered for: ",
           paste(.dispersion_families(), collapse = ", "),
           ". Families without a free dispersion (poisson, binomial, ",
           "truncated_poisson) have nothing to estimate.", call. = FALSE)
    }
    if (n_quad > 1L) {
      stop(caller, "(): `estimate_phi = TRUE` needs the joint-field Laplace ",
           "inner solve (`n_quad = 1`). The adaptive Gauss-Hermite path runs ",
           "through a compiled per-group oracle that the dispersion gradient ",
           "does not reach.", call. = FALSE)
    }
    if (!is.finite(phi) || phi <= 0) {
      stop(caller, "(): `phi` is the starting value for the dispersion when ",
           "`estimate_phi = TRUE`, so it must be finite and positive.",
           call. = FALSE)
    }
  }
  # Index of the dispersion coordinate within the stacked theta, or NA.
  phi_idx <- if (isTRUE(estimate_phi)) k + 1L else NA_integer_
  # Split a stacked theta into its covariance half and the dispersion it
  # implies. One definition, so the objective, the gradient and the reported
  # estimate cannot disagree about which coordinate is which.
  theta_split <- function(theta) {
    if (is.na(phi_idx)) return(list(re = theta, phi = phi))
    list(re = theta[seq_len(k)], phi = exp(theta[[phi_idx]]))
  }

  # AGHQ refinement (n_quad > 1) replaces the joint-field Laplace inner marginal
  # with the per-group adaptive Gauss-Hermite marginal (the same debias as
  # tulpa_re_aghq, applied INSIDE the Sigma integration), profiling the fixed
  # effects out by an inner optimization + a fixed-effect Laplace term. The
  # per-group integral only factorizes over ONE shared grouping factor, so AGHQ
  # is available only then; crossed RE terms keep the joint-field Laplace inner
  # solve (n_quad = 1).
  idx1 <- if (length(layout)) layout[[1L]]$idx
  ng1  <- if (length(layout)) layout[[1L]]$n_groups
  single_factor <- !is.null(idx1) && all(vapply(layout, function(b)
    identical(b$idx, idx1) && identical(b$n_groups, ng1), logical(1)))
  if (n_quad > 1L && !single_factor) {
    stop("`n_quad > 1` (adaptive Gauss-Hermite refinement of the inner solve) ",
         "requires a single shared grouping factor; this model has crossed RE ",
         "terms. Use `n_quad = 1` (the joint-field Laplace inner solve).",
         call. = FALSE)
  }
  use_core <- single_factor && n_quad > 1L

  # The compiled per-group oracle behind the AGHQ inner marginal carries one
  # linear predictor, so it cannot express the zero-inflation mixture. Refused
  # rather than run without the mixture, which would fit a different model.
  if (use_core && has_zi) {
    stop(caller, "(): `n_quad > 1` (the adaptive Gauss-Hermite inner marginal) ",
         "does not carry a zero-inflation process -- its compiled per-group ",
         "oracle has a single linear predictor. Use `n_quad = 1` (the ",
         "joint-field Laplace inner solve), which does.", call. = FALSE)
  }

  # cpp_glmm_oracle_make() takes no offset, so the AGHQ inner marginal cannot
  # carry one. Silently dropping it would fit a different model; say so instead.
  if (use_core && !is.null(offset) && any(offset != 0)) {
    stop(caller, "(): `n_quad > 1` (the adaptive Gauss-Hermite inner marginal) ",
         "does not support an offset -- the compiled per-group oracle carries ",
         "no offset term. Use `n_quad = 1` (the joint-field Laplace inner ",
         "solve), which does.", call. = FALSE)
  }
  # Observation weights scale each row's log-likelihood inside the joint-field
  # inner solve (tulpa_laplace() validates them). The per-group oracle has no
  # weight channel, so the same refusal as the offset.
  if (use_core && !is.null(weights) && any(weights != 1)) {
    stop(caller, "(): `n_quad > 1` (the adaptive Gauss-Hermite inner marginal) ",
         "does not support observation weights -- the compiled per-group ",
         "oracle carries no weight term. Use `n_quad = 1` (the joint-field ",
         "Laplace inner solve), which does.", call. = FALSE)
  }

  # Inner solve: Laplace log-marginal at the supplied per-block covariances.
  # Failures at extreme grid edges (non-finite / non-convergent) return -Inf so
  # the cell gets zero weight rather than aborting the integration.
  # `phi_` defaults to the fixed dispersion, so every caller that conditions on
  # it is unchanged; the dispersion-estimating path passes exp(log phi) from the
  # theta vector instead.
  inner_logmarg <- function(L_list, phi_ = phi) {
    val <- tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi_, phi2 = phi2, return_hessian = FALSE,
        beta_prior = beta_prior, offset = offset, weights = weights,
        X_zi = X_zi, zi_prior_sd = zi_prior_sd,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      )$log_marginal,
      error = function(e) -Inf
    )
    if (length(val) != 1L || !is.finite(val)) -Inf else val
  }

  # Inner solve that also keeps the joint posterior precision, which the exact
  # gradient differentiates through. Separate from inner_logmarg because the
  # extra factor costs memory the objective-only path has no use for.
  # `return_hessian = FALSE`: the exact gradient reads H_joint and the mode, not
  # the marginal fixed-effect block, and this runs at every trial theta the
  # optimizer visits -- including extremes where the Schur legitimately declines
  # and would warn about a fit nobody reports.
  inner_fit_grad <- function(L_list, phi_ = phi) {
    tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi_, phi2 = phi2, return_hessian = FALSE,
        return_joint_hessian = TRUE,
        beta_prior = beta_prior, offset = offset, weights = weights,
        X_zi = X_zi, zi_prior_sd = zi_prior_sd,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      ),
      error = function(e) NULL
    )
  }

  # Full inner solve at the integration nodes: the Laplace log-marginal plus the
  # fixed-effect mode and its MARGINAL covariance (solve(H_beta)). Paid O(n_grid)
  # times, not per outer step.
  #
  # `re_cov = TRUE` additionally asks for the per-(term, group) marginal
  # covariance blocks Cov(b_g | y, Sigma). The nested driver requests them at its
  # integration nodes so the per-group posterior can be marginalized over the
  # Sigma grid with its within-node curvature (ranef()); it costs one triangular
  # solve per random-effect coefficient against the factor the log-determinant
  # already built, so it is off wherever nobody reads the blocks (tulpa_eb()'s
  # re-solve at theta_hat, which reports the mode).
  #
  # `compute_skew` / `debias` are the two inner-layer extras the subspace debias
  # needs: the first turns the node solve into the reliability
  # probe the selector reads, the second hands it the selected index set so the
  # flagged coordinates come back as Metropolis draws instead of a Gaussian.
  # Both default off, so every existing caller solves exactly what it did.
  inner_fit <- function(L_list, phi_ = phi, re_cov = FALSE,
                        compute_skew = FALSE, skew_idx = NULL, debias = NULL,
                        joint_hessian = FALSE) {
    tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi_, phi2 = phi2, return_hessian = TRUE,
        return_re_cov = isTRUE(re_cov),
        return_joint_hessian = isTRUE(joint_hessian),
        beta_prior = beta_prior, offset = offset, weights = weights,
        X_zi = X_zi, zi_prior_sd = zi_prior_sd,
        max_iter = max_iter, tol = tol, n_threads = n_threads,
        compute_skew = isTRUE(compute_skew), skew_idx = skew_idx,
        debias = debias
      ),
      error = function(e) NULL
    )
  }

  # inner_logmarg at many covariances: `L_lists` is a list of per-block factor
  # lists, the value one log-marginal per element, -Inf where inner_logmarg
  # would give -Inf. One compiled call marshals the model once and solves every
  # point against it, starting each Newton iteration from `x_init` (the latent
  # mode at the proposal centre, or NULL for a cold start), across
  # `n_threads_outer` threads. This is the target an importance batch over the
  # covariances evaluates; inner_logmarg stays the one-point objective the
  # optimizer calls. The model arguments are the ones the inner solves above
  # validate on their first call through tulpa_laplace(), so the batch reads
  # them as validated.
  inner_logmarg_batch <- function(L_lists, x_init = NULL, n_threads_outer = 1L,
                                  max_iter_ = max_iter) {
    if (!length(L_lists)) return(numeric(0))
    model_args <- .laplace_multi_re_model_args(
      y = .numeric_response(family, y), n_trials = n_trials, X = X,
      re_list = .re_cov_build_re_list(L_lists[[1L]], layout), family = family,
      phi = phi, phi2 = phi2, weights = weights, offset = offset,
      bp = .normalize_beta_prior(beta_prior, ncol(X)), X_zi = X_zi,
      zi_prior_sd = zi_prior_sd)
    batch <- lapply(L_lists, function(L_list)
      .laplace_multi_re_sigma_list(.re_cov_build_re_list(L_list, layout)))
    do.call(cpp_laplace_log_marginal_multi_re_batch, c(model_args, list(
      re_sigma_batch = batch, max_iter = as.integer(max_iter_), tol = tol,
      n_threads_outer = as.integer(n_threads_outer),
      x_init = if (is.null(x_init)) NULL else as.numeric(x_init))))
  }

  # --- pilot init: method-of-moments per block from a Sigma = I fit ----------
  p_fix   <- ncol(X)
  p_fixed <- p_fix + p_zi
  L0_list <- lapply(layout, function(bl) diag(bl$nc))
  pilot <- tryCatch(
    tulpa_laplace(
      y = y, n_trials = n_trials, X = X,
      re_list = .re_cov_build_re_list(L0_list, layout),
      family = family, phi = phi, phi2 = phi2, return_hessian = FALSE,
      beta_prior = beta_prior, offset = offset, weights = weights,
      X_zi = X_zi, zi_prior_sd = zi_prior_sd,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    ),
    error = function(e) NULL
  )
  L_init_list <- L0_list
  if (!is.null(pilot) && !is.null(pilot$mode)) {
    # Skip the WHOLE fixed prefix: with zero inflation the mode carries
    # beta_zi between the count coefficients and the random effects, and
    # reading it as random-effect values would seed the search from garbage.
    re_vals <- pilot$mode[-seq_len(p_fixed)]
    pos <- 0L
    for (m in seq_along(layout)) {
      bl  <- layout[[m]]
      len <- bl$n_groups * bl$nc
      if (pos + len <= length(re_vals) && bl$n_groups > bl$nc) {
        U <- matrix(re_vals[pos + seq_len(len)], ncol = bl$nc, byrow = TRUE)
        S <- stats::cov(U)
        if (all(is.finite(S))) {
          if (!bl$full) S <- diag(diag(S), bl$nc)     # diagonal block: drop covs
          diag(S) <- pmax(diag(S), 1e-3)
          ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
          if (min(ev) > 1e-8) {
            L_init_list[[m]] <- if (bl$full) t(chol(S)) else {
              Lm <- matrix(0, bl$nc, bl$nc); diag(Lm) <- sqrt(diag(S)); Lm
            }
          }
        }
      }
      pos <- pos + len
    }
  }
  # A caller-supplied starting SD replaces the pilot's method-of-moments guess.
  # Worth having because the pilot fits at Sigma = I, which on a design whose
  # true scale is far from 1 starts the search in a flat region -- and because
  # it makes a run reproducible from its inputs rather than from a pilot fit.
  # Diagonal only: it sets each coefficient's scale and leaves any correlation
  # to be fitted, since a starting correlation is far harder to guess than a
  # starting scale.
  if (!is.null(sigma_init)) {
    s0 <- as.numeric(sigma_init)
    if (any(!is.finite(s0)) || any(s0 <= 0)) {
      stop(caller, "(): `control$sigma_init` must be finite and positive.",
           call. = FALSE)
    }
    nc_all <- vapply(layout, `[[`, integer(1), "nc")
    if (length(s0) == 1L) s0 <- rep(s0, sum(nc_all))
    if (length(s0) != sum(nc_all)) {
      stop(sprintf(paste0(
        "%s(): `control$sigma_init` must have length 1 or %d (one per ",
        "random-effect coefficient across all blocks); got %d."),
        caller, sum(nc_all), length(s0)), call. = FALSE)
    }
    pos <- 0L
    for (m in seq_along(layout)) {
      nc <- layout[[m]]$nc
      Lm <- matrix(0, nc, nc)
      diag(Lm) <- s0[pos + seq_len(nc)]
      L_init_list[[m]] <- Lm
      pos <- pos + nc
    }
  }

  theta0 <- .re_cov_L_list_to_theta(L_init_list, layout)
  # The supplied `phi` becomes the starting value rather than the fixed value.
  if (isTRUE(estimate_phi)) theta0 <- c(theta0, log(phi))

  # --- AGHQ inner solve (single shared grouping factor, n_quad > 1) ----------
  # Replace the joint-field Laplace inner_logmarg / inner_fit with the per-group
  # adaptive Gauss-Hermite marginal from the shared compiled engine: at a fixed
  # Sigma, profile the fixed effects beta out by an inner optimization of
  # sum_g log M_g(beta, Sigma) and add the fixed-effect Laplace correction
  # (0.5 p log 2pi - 0.5 logdet H_beta + log prior(beta_hat)), so log_marginal
  # integrates beta exactly as the joint-field path does -- but with each
  # per-group integral debiased by n_quad-point quadrature. beta is integrated
  # (not just profiled), so the reported fixed-effect posterior is the marginal
  # (ML-II) one, not the joint-mode (PQL) estimate.
  if (use_core) {
    nc_terms <- vapply(layout, function(b) b$nc, integer(1))
    full_vec <- vapply(layout, function(b) isTRUE(b$full), logical(1))
    Zc  <- do.call(cbind, lapply(layout, function(b) b$Z))   # n x sum(nc), stacked
    orc <- cpp_glmm_oracle_make(family, phi, as.numeric(y), as.numeric(n_trials),
                                as.matrix(X), as.matrix(Zc), as.integer(idx1), ng1)
    bp  <- .normalize_beta_prior(beta_prior, p_fix)
    beta_warm <- if (!is.null(pilot) && !is.null(pilot$mode))
      pilot$mode[seq_len(p_fix)] else rep(0, p_fix)

    # The fixed-effect profile at each covariance point, compiled
    # (`cpp_aghq_profile_batch()`, src/aghq_re_profile.h): Newton steps on the
    # analytic beta-gradient of sum_g log M_g + log prior(beta), the curvature a
    # central difference of that gradient, and the Laplace-integrated value
    # F(beta_hat) + (p / 2) log 2 pi - (1 / 2) log |H|. One point is a node or
    # an optimizer step; a batch is the outer proposal's draws, spread over
    # `n_threads_outer` threads, each point starting from `beta_warm` so its
    # value does not depend on the thread that solved it.
    profile <- function(sc_rows, n_threads_outer = 1L) {
      cpp_aghq_profile_batch(
        eta_pts = sc_rows, oracle = orc, nc = nc_terms, full = full_vec,
        n_quad = as.integer(n_quad), beta_init = beta_warm,
        prior_mean = bp$mean, prior_sd = bp$sd,
        n_threads = as.integer(n_threads_outer))
    }
    core_solve <- function(L_list) {
      sc <- .re_cov_L_list_to_theta(L_list, layout)
      r  <- profile(matrix(sc, nrow = 1L))
      if (!is.finite(r$log_marginal[1L])) return(NULL)
      list(log_marginal = r$log_marginal[1L], mode = r$beta[1L, ],
           H_beta = matrix(r$H[1L, , ], p_fix, p_fix))
    }
    # Same signature as the joint-field pair above: the outer objective always
    # passes the dispersion, so both inner solves have to take it. `phi_` is
    # deliberately unused here -- `estimate_phi = TRUE` is refused for
    # `n_quad > 1` further up, so on this path phi never leaves its starting
    # value and is already baked into the compiled oracle by
    # cpp_glmm_oracle_make(). It is NOT a value that needs threading through:
    # rebuilding the oracle per solve would copy X, Z and y for nothing.
    inner_logmarg <- function(L_list, phi_ = phi) {
      r <- core_solve(L_list); if (is.null(r)) -Inf else r$log_marginal
    }
    # Same signature as the joint-field batch above. There is no latent mode to
    # warm start from (`x_init` is the joint-field solve's), so every point
    # starts from `beta_warm`, and the compiled profile runs its own Newton
    # budget, so `max_iter_` is unused.
    inner_logmarg_batch <- function(L_lists, x_init = NULL, n_threads_outer = 1L,
                                    max_iter_ = max_iter) {
      if (!length(L_lists)) return(numeric(0))
      sc_rows <- do.call(rbind, lapply(L_lists, .re_cov_L_list_to_theta,
                                       layout = layout))
      profile(sc_rows, n_threads_outer)$log_marginal
    }
    # `re_cov` is accepted and ignored: this inner solve integrates each group's
    # random effects out by quadrature rather than conditioning at their mode, so
    # it produces no per-group mean or covariance to hand back. The driver reads
    # `re_conditional` (FALSE here) and says so rather than reporting an empty
    # random-effect table.
    # The inner-layer extras the joint-field solve carries are refused here
    # rather than ignored: this solve integrates each group out, so there is no
    # conditional latent field to score a cubic term on, no joint precision to
    # read a coupling from, and nothing for a subspace sampler to move. The
    # front door refuses `control$subspace_debias` at `n_quad > 1` for the same
    # reason; this is the guard for anything reaching the closure directly.
    inner_fit <- function(L_list, phi_ = phi, re_cov = FALSE,
                          compute_skew = FALSE, skew_idx = NULL, debias = NULL,
                          joint_hessian = FALSE) {
      if (isTRUE(compute_skew) || !is.null(debias) || isTRUE(joint_hessian)) {
        stop("the adaptive Gauss-Hermite inner marginal (n_quad > 1) exposes ",
             "no conditional latent field, so it carries neither the ",
             "inner-Laplace diagnostics nor the subspace debias. Use ",
             "n_quad = 1 (the joint-field Laplace inner solve).", call. = FALSE)
      }
      core_solve(L_list)
    }
  }

  # --- mode of g(theta) = log_marginal(Sigma(theta)) + log_prior ------------
  # The hyperprior is over the COVARIANCE coordinates; the dispersion enters
  # unpenalized, which is what makes this the ML-II estimate of phi rather than
  # a MAP under an undeclared prior. So the prior only ever sees theta's
  # covariance half.
  negg <- function(theta) {
    sp <- theta_split(theta)
    L_list <- .re_cov_theta_to_L_list(sp$re, layout)
    -(inner_logmarg(L_list, sp$phi) + log_prior_theta(sp$re))
  }

  # Analytic outer gradient. Available when the inner solve is the joint-field
  # Laplace (the AGHQ inner marginal at n_quad > 1 is a different objective, and
  # its Fisher-identity gradient is not this one) and the family has an exact
  # curvature derivative. `use_exact_grad` is resolved once here rather than per
  # outer step so a family without one falls back cleanly instead of alternating
  # between two objectives.
  use_exact_grad <- !use_core &&
    isTRUE(tryCatch(cpp_family_has_curvature_derivative(family),
                    error = function(e) FALSE)) &&
    (!has_zi || .laplace_exact_supports_zi(family))

  # Estimating the dispersion is defined by the exact gradient carrying a
  # log-phi coordinate. Without that gradient the fallbacks would search the
  # extra dimension derivative-free -- a different and much weaker method under
  # the same argument -- so refuse instead of silently substituting it.
  if (isTRUE(estimate_phi) && !use_exact_grad) {
    stop(caller, "(): `estimate_phi = TRUE` needs the exact outer gradient, ",
         "which is unavailable for family '", family, "' on this path. ",
         "The dispersion would otherwise be searched derivative-free, which ",
         "is a different method than the one this argument selects.",
         call. = FALSE)
  }
  # The gradient's dispersion coordinate covers a mixture through two sources:
  # the base family's registered phi derivatives on the rows the model leaves
  # additively separable, and the mixture engine on a genuine mixture's y = 0
  # rows, where log(pi + (1 - pi) P(Y = 0, phi)) carries phi through P(Y = 0) and
  # couples it to both predictors. A hurdle needs only the first -- its zero
  # branch is log(pi), phi-free. Refused where the second is unregistered, so a
  # family never maximizes along the base derivative under a model it does not
  # describe.
  if (isTRUE(estimate_phi) && has_zi &&
      !isTRUE(tryCatch(cpp_family_has_zi_phi_deriv(family),
                       error = function(e) FALSE))) {
    stop(caller, "(): `estimate_phi = TRUE` is not available alongside a ",
         "zero-inflation process on the family '", family,
         "'. The mixture's zero branch depends on phi through P(Y = 0), and ",
         "no dispersion derivative for that branch is registered here, so ",
         "maximizing would optimize a different objective than the one ",
         "reported. Condition on `phi`, profile it by refitting over a grid, ",
         "or use a family whose mixture dispersion IS registered ",
         "(neg_binomial_2, or any zero-truncated base as a hurdle).",
         call. = FALSE)
  }

  # The hyperprior is a closed form in theta with no inner solve behind it, so
  # its gradient is a central difference costing 2k prior evaluations -- no
  # Laplace solves. Only the expensive half of the gradient needs to be exact.
  # Differentiated over the covariance coordinates only, then padded with a zero
  # for the unpenalized dispersion so it lines up with the stacked gradient.
  d_log_prior_theta <- function(theta_re, h = 1e-6) {
    g <- vapply(seq_along(theta_re), function(j) {
      tp <- theta_re; tp[j] <- tp[j] + h
      tm <- theta_re; tm[j] <- tm[j] - h
      (log_prior_theta(tp) - log_prior_theta(tm)) / (2 * h)
    }, numeric(1))
    if (is.na(phi_idx)) g else c(g, 0)
  }

  # Curvature of the same hyperprior, over the covariance coordinates only. The
  # closed-form outer Hessian is exact for the Laplace marginal but the prior is
  # user-supplied with no closed form, so its d2/dtheta2 is a second central
  # difference -- costing only prior evaluations, no inner Laplace solves. h is
  # larger than the gradient's because a second difference divides by h^2.
  d2_log_prior_theta <- function(theta_re, h = 1e-4) {
    kk <- length(theta_re)
    Hp <- matrix(0, kk, kk)
    lp0 <- log_prior_theta(theta_re)
    for (j in seq_len(kk)) {
      tp <- theta_re; tp[j] <- tp[j] + h
      tm <- theta_re; tm[j] <- tm[j] - h
      Hp[j, j] <- (log_prior_theta(tp) - 2 * lp0 + log_prior_theta(tm)) / h^2
    }
    if (kk > 1L) for (j in seq_len(kk - 1L)) for (l in (j + 1L):kk) {
      tpp <- theta_re; tpp[j] <- tpp[j] + h; tpp[l] <- tpp[l] + h
      tpm <- theta_re; tpm[j] <- tpm[j] + h; tpm[l] <- tpm[l] - h
      tmp <- theta_re; tmp[j] <- tmp[j] - h; tmp[l] <- tmp[l] + h
      tmm <- theta_re; tmm[j] <- tmm[j] - h; tmm[l] <- tmm[l] - h
      Hp[j, l] <- Hp[l, j] <-
        (log_prior_theta(tpp) - log_prior_theta(tpm) -
         log_prior_theta(tmp) + log_prior_theta(tmm)) / (4 * h^2)
    }
    Hp
  }

  # Value and gradient of the FULL outer objective, log_marginal + log_prior, at
  # one theta. Everything downstream -- the optimizer, the marginal correction's
  # curvature -- reads this, so there is one definition of what is being
  # differentiated and the prior term cannot be attached twice or dropped once.
  # Cached because the mode-find asks for value, gradient and curvature at the
  # same theta and the inner solve is the entire cost.
  grad_cache <- new.env(parent = emptyenv())
  eval_grad_at <- function(theta, want_jacobian = FALSE, want_hessian = FALSE) {
    want_list <- want_jacobian || want_hessian
    key <- paste(c(signif(theta, 12), if (want_jacobian) "J", if (want_hessian) "H"),
                 collapse = ",")
    if (!is.null(grad_cache[[key]])) return(grad_cache[[key]])
    sp <- theta_split(theta)
    L_list <- .re_cov_theta_to_L_list(sp$re, layout)
    fit <- inner_fit_grad(L_list, sp$phi)
    val <- if (is.null(fit) || !is.finite(fit$log_marginal %||% NA_real_)) NULL else {
      # The settled-mode gate refuses the GRADIENT, not the value: an inner mode
      # short of stationarity moves the log marginal only at second order. A
      # line search evaluates the value alone at its trial points, so the refusal
      # is kept with the evaluation and signalled where a gradient is read.
      refused <- NULL
      r <- withCallingHandlers(.laplace_exact_re_grad(
        fit = fit, y = y, X = X, n_trials = n_trials, offset = offset,
        weights = weights,
        re_list = .re_cov_build_re_list(L_list, layout),
        layout = layout, L_list = L_list, family = family,
        # The same phi2 the inner solves ran at: the gradient has to describe
        # the model the inner solve fitted, not a differently-parameterized one.
        # NA_real_ is the "no second dispersion" spelling the family kernels
        # take, so an absent phi2 reaches them as it always did.
        phi = sp$phi, phi2 = phi2 %||% NA_real_, want_jacobian = want_jacobian,
        want_hessian = want_hessian, X_zi = X_zi,
        # Appends the log-phi coordinate, so the returned gradient is stacked in
        # the same order as theta.
        want_phi = !is.na(phi_idx)
      ), tulpa_unsettled_mode = function(w) {
        refused <<- w
        invokeRestart("muffleWarning")
      })
      if (is.null(r)) {
        if (is.null(refused)) NULL
        else list(f = fit$log_marginal + log_prior_theta(sp$re), g = NULL,
                  refused = refused)
      } else {
        # The prior shifts the objective but not the mode, so it enters the
        # gradient and leaves J alone. H below is the Laplace marginal's exact
        # curvature over the covariance coordinates; the prior's own curvature is
        # added where the negative outer Hessian is formed (exact_grad_at).
        g <- (if (want_list) r$grad else r) + d_log_prior_theta(sp$re)
        list(f = fit$log_marginal + log_prior_theta(sp$re), g = g,
             J = if (want_list) r$J else NULL,
             H = if (want_hessian) r$H else NULL)
      }
    }
    grad_cache[[key]] <- val
    val
  }

  negg_exact <- function(theta) {
    r <- eval_grad_at(theta)
    if (is.null(r) || !is.finite(r$f)) return(.Machine$double.xmax)
    -r$f
  }
  negg_gr <- function(theta) {
    r <- eval_grad_at(theta)
    if (is.null(r)) return(rep(0, length(theta)))
    if (is.null(r$g)) {
      warning(r$refused)
      return(rep(0, length(theta)))
    }
    -r$g
  }

  n_theta <- length(theta0)

  # The dispersion coordinate is bracketed, for the same reason sigma is. As phi
  # grows the objective FLATTENS rather than turning over -- a negative binomial
  # becomes a Poisson, a gamma becomes tight around its mean -- so an oversized
  # early step lands in a region where the gradient is ~0 and an unbounded
  # search reports convergence at a value with no support in the data: on a
  # 480-row negative binomial whose profile peaks cleanly at phi = 3, unbounded
  # BFGS stopped at 3.5e15.
  phi_lo <- log(1e-6)
  phi_hi <- log(1e6)

  # No `log_marginal` here on purpose: backing it out of opt$value would be a
  # second, differently-derived copy of a number both callers already read off an
  # actual fit -- and one that goes NaN whenever the optimizer's last step hit
  # the failure sentinel.
  # `exact_grad_at` is the analytic outer gradient and, on request, the exact
  # mode Jacobian at a given theta -- NULL when this path has no exact gradient
  # (the AGHQ inner marginal, or a family whose working weight has no closed-form
  # eta-derivative). tulpa_eb(marginal = TRUE) uses it to replace the
  # finite-difference stencil; a NULL sends it back to the stencil.
  exact_grad_at <- if (!use_exact_grad) NULL else
    function(theta, want_jacobian = FALSE, want_hessian = FALSE) {
      r <- eval_grad_at(theta, want_jacobian = want_jacobian,
                        want_hessian = want_hessian)
      if (is.null(r)) return(NULL)
      if (is.null(r$g)) {
        warning(r$refused)
        return(NULL)
      }
      if (isTRUE(want_hessian)) {
        # The negative outer Hessian the marginal correction consumes:
        # -(d2 log_marginal + d2 log_prior). The marginal half is closed-form
        # (r$H); the prior half is the cheap second difference above. NULL r$H
        # (a family without a second curvature derivative) hands the Hessian back
        # to the differencing stencil.
        if (is.null(r$H)) return(NULL)
        sp <- theta_split(theta)
        Hp <- tryCatch(d2_log_prior_theta(sp$re), error = function(e) NULL)
        if (is.null(Hp)) return(NULL)
        # The dispersion enters the objective unpenalized, so the prior adds a
        # zero row/column for log phi -- pad it to match the Laplace Hessian's
        # phi border (r$H is (k+1)x(k+1) when phi is estimated).
        if (!is.na(phi_idx) && nrow(r$H) == nrow(Hp) + 1L) {
          Hp <- rbind(cbind(Hp, rep(0, nrow(Hp))), rep(0, ncol(Hp) + 1L))
        }
        if (!all(dim(Hp) == dim(r$H))) return(NULL)
        Ht <- -(r$H + Hp)
        return(list(grad = r$g, J = r$J, H = (Ht + t(Ht)) / 2))
      }
      if (isTRUE(want_jacobian)) list(grad = r$g, J = r$J) else r$g
    }

  # --- the mode: the shared outer mode-find ---------------------------------
  # `.nl_mode_curvature()`, the search the joint door's placement runs, here on
  # this objective, in these coordinates, at the relative tolerance `reltol` on
  # the log posterior. Each round's local quadratic is the analytic gradient and
  # outer Hessian where the family has both (`exact_grad_at`), the analytic
  # gradient differenced where it has no second curvature derivative, and a
  # finite-difference stencil of the objective where it has no exact gradient
  # (the AGHQ inner marginal, or such a family). Every log scale -- a diagonal
  # block's log SD, a full block's log L_ii -- is boxed to the random-effect SD
  # bracket and log phi to its own; a strictly-lower log-Cholesky entry keeps
  # the whole line.
  reltol <- outer_reltol %||% 1e-10
  # Each log-scale coordinate with the block and the coefficient it scales.
  log_scale_idx <- integer(0); log_scale_blk <- integer(0)
  log_scale_coef <- integer(0); pos <- 0L
  for (m in seq_along(layout)) {
    bl <- layout[[m]]
    at <- if (bl$full) pos + diag(.re_logchol_indices(bl$nc)) else pos + seq_len(bl$k)
    log_scale_idx  <- c(log_scale_idx, at)
    log_scale_blk  <- c(log_scale_blk, rep(m, length(at)))
    log_scale_coef <- c(log_scale_coef, seq_along(at))
    pos <- pos + bl$k
  }
  sd_lo <- log(.re_cov_search("sd_range")[1L])
  sd_hi <- log(.re_cov_search("sd_range")[2L])
  lower <- rep(-Inf, n_theta); upper <- rep(Inf, n_theta)
  lower[log_scale_idx] <- sd_lo; upper[log_scale_idx] <- sd_hi
  if (!is.na(phi_idx)) { lower[phi_idx] <- phi_lo; upper[phi_idx] <- phi_hi }
  box <- list(lower = lower, upper = upper, span = upper - lower,
              trust = rep(.re_cov_search("trust"), n_theta))
  value_at <- if (use_exact_grad) function(th) -negg_exact(th) else
    function(th) -negg(th)
  eval1 <- function(U) {
    v <- apply(as.matrix(U), 1L, value_at)
    v[!is.finite(v) | v <= -.Machine$double.xmax] <- -Inf
    v
  }
  fd_step <- rep(1e-3, n_theta)
  fd_stencil <- function(u) .joint_ccd_fd_stencil(u, eval1, fd_step)
  exact_stencil <- function(u) {
    r <- exact_grad_at(u, want_hessian = TRUE)
    if (!is.null(r)) return(list(grad = r$grad, hess = -r$H))
    g0 <- exact_grad_at(u)
    if (is.null(g0)) return(NULL)
    H <- vapply(seq_len(n_theta), function(j) {
      up <- u; dn <- u
      up[j] <- up[j] + fd_step[j]; dn[j] <- dn[j] - fd_step[j]
      gu <- exact_grad_at(up); gd <- exact_grad_at(dn)
      if (is.null(gu) || is.null(gd)) return(rep(NA_real_, n_theta))
      (gu - gd) / (2 * fd_step[j])
    }, numeric(n_theta))
    list(grad = g0, hess = 0.5 * (H + t(H)))
  }
  find_mode <- function(stencil) .nl_mode_curvature(
    pmin(pmax(theta0, lower), upper), eval1, box, stencil = stencil,
    reltol = reltol, max_rounds = as.integer(outer_maxit),
    max_halve = .re_cov_search("max_halve"))
  # One warning per fit rather than one per trial theta if the exact gradient is
  # declined because an inner mode did not settle (.laplace_mode_settled). A
  # search that walked on a declined gradient is discarded and run again on the
  # objective's own differences.
  unsettled <- .new_unsettled_state()
  mc <- .with_unsettled_report({
    m <- if (use_exact_grad) find_mode(exact_stencil)
    if (is.null(m) || !is.null(m$declined) || unsettled$n > 0L) {
      m <- find_mode(fd_stencil)
    }
    m
  }, caller, unsettled)
  if (!is.null(mc$declined)) {
    stop(caller, "(): the outer mode-find over the random-effect ",
         "covariance(s) found no mode with a usable curvature from its start. ",
         "Check the data, the hyperprior and `control$sigma_init`.",
         call. = FALSE)
  }
  theta_hat <- mc$par
  # `opt` keeps optim's record shape: 0 for a converged search, 1 for one the
  # round cap (`outer_maxit`) stopped.
  opt <- list(par = theta_hat, value = -mc$value,
              convergence = if (identical(mc$status, "not_converged")) 1L else 0L,
              counts = c(rounds = mc$rounds))

  # A search the round cap stopped returns wherever it stood, not a maximizer,
  # and for EB that IS the estimate.
  if (!identical(opt$convergence, 0L)) {
    warning(caller, "(): the outer optimization over the random-effect ",
            "covariance(s) did not converge in ", outer_maxit, " rounds. ",
            "The reported estimate is where it stopped. Consider a longer ",
            "run, a different `re_prior` / `prior_sigma`, or more data per ",
            "group.", call. = FALSE)
  }

  # A coordinate pinned at its bracket is a finding, not an estimate. A
  # dispersion at the upper end means the extra-Poisson variation the parameter
  # exists to carry is not there; a variance component at the lower end is the
  # classic empirical-Bayes collapse to sigma = 0. Reported rather than
  # returned silently, since the number itself looks like any other fit.
  if (!is.na(phi_idx) && is.finite(theta_hat[[phi_idx]])) {
    lp <- theta_hat[[phi_idx]]
    if (lp <= phi_lo + 1e-6 || lp >= phi_hi - 1e-6) {
      warning(sprintf(paste0(
        "%s(): the estimated dispersion hit the %s end of its search bracket ",
        "(phi = %.3g). %s Treat it as a boundary hit, not a fitted value."),
        caller, if (lp <= phi_lo + 1e-6) "lower" else "upper", exp(lp),
        if (lp >= phi_hi - 1e-6) {
          paste("The data show no dispersion beyond what the mean-variance",
                "relation and the random effects already explain; a family",
                "without a free dispersion may be the better model.")
        } else {
          paste("The dispersion is collapsing, which usually means the",
                "response is far more variable than the family allows.")
        }), call. = FALSE)
    }
  }
  # A log scale at the lower end is a SINGULAR covariance, the boundary fit
  # glmer reports: a diagonal block's SD, or a full block's leading SD, has
  # collapsed to zero, and a full block's later Cholesky diagonal L_ii at the
  # floor leaves coefficient i's effect a linear function of the coefficients
  # before it -- its SD zero, or its multiple correlation with them one. The
  # outer objective's supremum then lies on the boundary of the covariance
  # cone, and the box edge is where the search reports it. A log scale at the
  # upper end is capped rather than fitted.
  #
  # `warn_boundary = FALSE` records the boundary without warning. A caller that
  # integrates over the covariance reports a posterior that is valid whether or
  # not this one point sits on the boundary, so the hit describes `theta_hat`
  # alone and is recorded with it.
  L_hat_b <- .re_cov_theta_to_L_list(theta_split(theta_hat)$re, layout)
  boundary <- data.frame(block = character(0), coef = integer(0),
                         end = character(0), sigma = numeric(0),
                         stringsAsFactors = FALSE)
  for (a in seq_along(log_scale_idx)) {
    j <- log_scale_idx[a]
    if (!is.finite(theta_hat[j])) next
    at_lo <- theta_hat[j] <= sd_lo + 1e-6
    at_hi <- theta_hat[j] >= sd_hi - 1e-6
    if (!(at_lo || at_hi)) next
    m <- log_scale_blk[a]; i <- log_scale_coef[a]; bl <- layout[[m]]
    lab <- .re_cov_block_label(bl, m)
    L <- L_hat_b[[m]]
    sd_i <- sqrt(sum(L[i, ]^2))
    boundary[nrow(boundary) + 1L, ] <- list(lab, i, if (at_lo) "lower" else "upper",
                                            sd_i)
    what <- if (bl$nc > 1L) sprintf("coefficient %d of block '%s'", i, lab) else
      sprintf("block '%s'", lab)
    if (!isTRUE(warn_boundary)) next
    if (at_lo && bl$full && i > 1L) {
      mult_cor <- sqrt(max(0, 1 - (L[i, i] / sd_i)^2))
      warning(sprintf(paste0(
        "%s(): the random-effect covariance of block '%s' is singular (a ",
        "boundary fit): the Cholesky diagonal of coefficient %d hit the lower ",
        "end of the search bracket, so its effect is a linear function of the ",
        "coefficients before it. Its standard deviation is %.3g and its ",
        "multiple correlation with them %.6f. The supremum of the marginal ",
        "likelihood lies on the boundary of the covariance; treat this as a ",
        "boundary estimate, not a fitted value."),
        caller, lab, i, sd_i, mult_cor), call. = FALSE)
      next
    }
    detail <- if (at_lo) {
      paste("The variance component has collapsed: the data carry no",
            "detectable between-group variation, or the hyperprior is not",
            "holding it off the boundary.")
    } else {
      paste("The between-group variation exceeds the bracket, so the value",
            "is capped rather than fitted.")
    }
    warning(sprintf(paste0(
      "%s(): the random-effect standard deviation of %s hit the %s end of the ",
      "search bracket (sigma = %.3g). %s Treat it as a boundary hit, not a ",
      "fitted value."),
      caller, what, if (at_lo) "lower" else "upper", sd_i, detail),
      call. = FALSE)
  }

  # The dispersion is reported separately from theta_hat's covariance half, and
  # `theta_hat` is trimmed to the covariance coordinates so every downstream
  # consumer (.re_cov_theta_to_L_list, the map summary, the marginal correction)
  # keeps receiving the vector length it was written for. `phi_hat` is the
  # estimate; `phi` stays the value the caller supplied, which is the start.
  sp_hat <- theta_split(theta_hat)
  list(layout = layout, k = k, n_trials = n_trials, X = X, p_fix = p_fix,
       # The count block and the whole fixed prefix. They differ only under
       # zero inflation, where the mode is [beta | beta_zi | RE]; callers that
       # slice the mode or name the fixed effects want `p_fixed`, while the
       # AGHQ profile optimizes over the count block alone.
       X_zi = X_zi, p_zi = p_zi, p_fixed = p_fixed,
       log_prior_theta = log_prior_theta,
       inner_logmarg = inner_logmarg, inner_fit = inner_fit,
       inner_logmarg_batch = inner_logmarg_batch,
       # Does `inner_fit` condition on the random effects (returning their mode
       # in the latent tail of `$mode`, and their covariance blocks on request)?
       # TRUE for the joint-field Laplace inner solve, FALSE for the AGHQ one,
       # which integrates them out per group. The per-group posterior is
       # reportable exactly when this is TRUE.
       re_conditional = !use_core,
       exact_grad_at = exact_grad_at,
       theta0 = theta0, theta_hat = sp_hat$re, opt = opt,
       boundary = boundary, singular = any(boundary$end == "lower"),
       theta_hat_full = theta_hat,
       phi_hat = if (is.na(phi_idx)) NULL else sp_hat$phi,
       estimate_phi = isTRUE(estimate_phi))
}
