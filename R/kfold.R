# kfold.R
# ------------------------------------------------------------------------------
# Refit-based K-fold cross-validation: the exact (refit) counterpart to the
# importance-sampling PSIS-LOO in tulpa_criteria(). When the outer Pareto k-hat
# gate flags PSIS-LOO as unreliable, K-fold recomputes the held-out predictive
# density by actually refitting on each training partition. The model is refit
# through its stored call (tulpa()'s `$call`), so only fixed-effect / GLMM fits
# are supported -- subsetting the observations would break a spatial / temporal
# field's structure, so those are rejected loudly.
# ------------------------------------------------------------------------------

#' K-fold cross-validation for a tulpa fit
#'
#' @description
#' Splits the data into `K` folds, refits the model on each `K - 1` training
#' partition (via the fit's stored `tulpa()` call), and accumulates the held-out
#' fold's pointwise log predictive density
#' \eqn{\log \frac{1}{S}\sum_s p(y_i \mid \eta_i^{(s)})} over the training
#' posterior draws. The summed `elpd_kfold` is directly comparable to the
#' `elpd_loo` from [tulpa_criteria()] -- the exact refit counterpart to the
#' PSIS-LOO approximation, for when the Pareto k-hat gate flags LOO as
#' unreliable.
#'
#' Fixed-effect / GLMM fits only: subsetting the observations breaks a spatial or
#' temporal field, so those fits are rejected (use PSIS-LOO via
#' [tulpa_criteria()]). A held-out row of a random-effect group the training
#' rows saw is scored at that group's effect, and one of a group they did not
#' see integrates a new effect over the term's covariance, as
#' [posterior_predict()] does at new data.
#'
#' @param object A `tulpa_fit` fitted through [tulpa()] (must carry `$call`).
#' @param data The data frame the model was fit to.
#' @param K Number of folds (default 10).
#' @param folds Optional integer vector of fold ids, length `nrow(data)`; a
#'   random balanced partition is drawn when `NULL`.
#' @param n_trials Optional binomial denominators (length `nrow(data)`);
#'   defaults to the trials stored on the fit, else 1 (Bernoulli). Each fold's
#'   refit receives the training rows' trials, and the held-out density is
#'   scored at the test rows' trials.
#' @param seed Optional seed for the random partition.
#'
#' @return A list with `elpd_kfold` (summed held-out elpd), `se_elpd_kfold` (its
#'   standard error), `pointwise` (per-observation held-out elpd), `folds`, and
#'   `K`.
#'
#' @seealso [tulpa_criteria()] for PSIS-LOO / WAIC on a single fit.
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = rnorm(120))
#' d$y <- rpois(120, exp(0.4 + 0.6 * d$x))
#' fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
#' cv  <- tulpa_kfold(fit, data = d, K = 5, seed = 1)
#' cv$elpd_kfold
#' }
#' @export
tulpa_kfold <- function(object, data, K = 10L, folds = NULL,
                        n_trials = NULL, seed = NULL) {
  su <- .cv_refit_setup(object, data, n_trials, "Refit-based K-fold")
  cl <- su$cl; data <- su$data; n <- su$n
  fam <- su$fam; phi <- su$phi; y <- su$y; nt <- su$nt

  .seed_scoped(seed)
  if (is.null(folds)) {
    folds <- sample(rep(seq_len(K), length.out = n))
  } else {
    if (length(folds) != n) {
      stop("`folds` must have length nrow(data) (", n, ").", call. = FALSE)
    }
    folds <- as.integer(folds)
  }
  fold_ids <- sort(unique(folds))
  K <- length(fold_ids)

  refit_env <- parent.frame()
  inject_trials <- .family_reads_trials(fam)
  pointwise <- rep(NA_real_, n)

  for (k in fold_ids) {
    test  <- which(folds == k)
    train <- which(folds != k)
    fit_k <- .cv_refit(cl, data, train, nt, inject_trials, refit_env,
                       paste0("K-fold refit on fold ", k),
                       obs_w = object$obs_weights)
    pointwise[test] <- .cv_heldout_lpd(fit_k, data, test, y, nt, fam, phi,
                                       paste0("Fold ", k),
                                       phi2 = object$phi2)
  }

  elpd <- sum(pointwise)
  se   <- sqrt(n * stats::var(pointwise))
  list(elpd_kfold = elpd, se_elpd_kfold = se,
       pointwise = pointwise, folds = folds, K = K)
}

# The refit contract both resampling verbs share. A refit-based estimator needs
# a fit it can rebuild (a recorded call and formula) whose observations are
# exchangeable, so a spatial or temporal field -- whose structure a row subset
# would break -- is refused up front. `what` names the calling verb in that
# message; everything else is identical between them.
#
# Returns the pieces the caller works from: the stored call, the coerced data
# and its row count, the family and dispersion, the response evaluated on
# `data`, and the resolved trial counts.
.cv_refit_setup <- function(object, data, n_trials, what) {
  if (!inherits(object, "tulpa_fit")) {
    stop("`object` must be a tulpa_fit.", call. = FALSE)
  }
  if (!is.null(object$spatial) || !is.null(object$temporal) ||
      !is.null(object$temporal_field)) {
    stop(what, " is not supported for spatial / temporal-field fits: ",
         "subsetting the observations breaks the field structure. Use ",
         "tulpa_criteria() (PSIS-LOO) for those.", call. = FALSE)
  }
  if (is.null(object$call)) {
    stop("`object` carries no `$call` to refit from; fit through tulpa().",
         call. = FALSE)
  }
  if (is.null(object$formula)) {
    stop("`object` carries no `$formula`; cannot recover the response.",
         call. = FALSE)
  }

  data <- as.data.frame(data)
  n    <- nrow(data)
  y    <- eval(object$formula[[2L]], envir = data)
  if (length(y) != n) {
    stop("The response evaluated on `data` has length ", length(y),
         " but `data` has ", n, " rows.", call. = FALSE)
  }

  list(cl = object$call, data = data, n = n,
       fam = object$family, phi = object$phi %||% 1.0, y = y,
       nt = .cv_trials(object, n_trials, n))
}

# Resolve the full-data trial counts for a refit-CV verb: the caller's
# `n_trials`, else the trials stored on the fit, else 1 (Bernoulli).
.cv_trials <- function(object, n_trials, n) {
  nt <- if (!is.null(n_trials)) as.integer(n_trials)
        else if (!is.null(object$n_trials)) as.integer(object$n_trials)
        else rep(1L, n)
  if (length(nt) == 1L) nt <- rep(nt, n)
  if (length(nt) != n) {
    stop("`n_trials` must have length nrow(data) (", n, "); got ",
         length(nt), ".", call. = FALSE)
  }
  nt
}

# Refit the stored tulpa() call on a row subset of `data`. The training rows'
# trial counts and observation weights are injected as literals so a stored
# `n_trials = <expr>` / `weights = <expr>` argument (which would evaluate to
# the full-length vector) cannot misalign against the subset data.
#
# `inject_trials` is decided from the FAMILY (`.family_reads_trials()`), not
# from whether the fit carries a `$n_trials` field: every fit carries one since
# gcol33/tulpa#781, so reading the field injected an `n_trials =` argument into
# a poisson / gaussian refit, which tulpa() refuses (gcol33/tulpa#837). For a
# binomial family the resolved counts default to 1 where the call supplied
# none, so injecting them unconditionally restates what the call already meant.
# A cbind(successes, failures) response never reaches here -- it evaluates to a
# two-column matrix, which `.cv_refit_setup()`'s length check refuses.
.cv_refit <- function(cl, data, train, nt, inject_trials, env, label,
                      obs_w = NULL) {
  ccl <- cl
  ccl$data <- data[train, , drop = FALSE]
  if (inject_trials) ccl$n_trials <- nt[train]
  if (!is.null(obs_w)) ccl$weights <- obs_w[train]
  tryCatch(eval(ccl, envir = env),
           error = function(e) {
             stop(label, " failed: ", conditionMessage(e), ". The stored ",
                  "call's arguments must be resolvable in the calling scope.",
                  call. = FALSE)
           })
}

# Exact held-out log predictive density log 1/S sum_s p(y_i | eta_i^(s)) at
# the rows `test`, over the training refit's posterior draws of the linear
# predictor there (`.tulpa_eta_draws()`): the fixed effects and, for a
# random-effect term, the effect of a group the training rows saw drawn with
# them, and the effect of a group they did not see drawn from the term's
# covariance. The draws are pinned, so a fold scores the same on every call.
.cv_heldout_lpd <- function(fit_k, data, test, y, nt, fam, phi, label,
                            phi2 = NULL) {
  eta <- tryCatch(
    .tulpa_eta_draws(fit_k, newdata = data[test, , drop = FALSE],
                     synth_seed = .DIAGNOSTIC_SEED$criteria),
    error = function(e) stop(label, ": ", conditionMessage(e), call. = FALSE))
  logit_zi <- attr(eta, "logit_zi")

  yte <- y[test]; ntte <- nt[test]
  ll <- vapply(seq_len(nrow(eta)), function(s) {
    .response_loglik(eta[s, ], if (!is.null(logit_zi)) logit_zi[s, ], yte, fam,
                     n_trials = ntte, phi = phi, phi2 = phi2)
  }, numeric(length(test)))                    # [n_test x S]
  if (is.null(dim(ll))) ll <- matrix(ll, nrow = length(test))

  apply(ll, 1L, logmeanexp)
}

#' Selective refit of high-Pareto-k observations (reloo)
#'
#' @description
#' PSIS-LOO with exact refits where the importance sampling is unreliable:
#' observations whose Pareto k-hat exceeds `k_threshold` are re-scored by
#' refitting the model without that observation (through the fit's stored
#' [tulpa()] call, as in [tulpa_kfold()]) and evaluating the exact held-out
#' log predictive density. All other observations keep their PSIS-LOO value,
#' so the cost is one refit per flagged observation rather than per fold.
#'
#' The same restrictions as [tulpa_kfold()] apply: fixed-effect / GLMM fits
#' only (subsetting breaks a spatial / temporal field), and a held-out row is
#' scored at its group's effect as there.
#'
#' @param object A `tulpa_fit` fitted through [tulpa()] (must carry `$call`).
#' @param data The data frame the model was fit to.
#' @param k_threshold Pareto k-hat above which an observation is refit
#'   (default 0.7, the standard PSIS reliability gate).
#' @param n_trials Optional binomial denominators (length `nrow(data)`);
#'   defaults to the trials stored on the fit, else 1.
#' @param ndraws Number of posterior draws used for the PSIS-LOO baseline
#'   (defaults to all stored draws, or 400 on the draw-free Laplace tier).
#'
#' @return A list with `elpd_loo` (corrected), `se_elpd_loo`, `looic`,
#'   `pointwise` (per-observation elpd, exact at the refit observations),
#'   `reloo_idx` (indices refit), `pareto_k` (the original k-hat values), and
#'   `k_threshold`.
#'
#' @seealso [tulpa_kfold()] for the full refit-CV; [tulpa_criteria()] for
#'   PSIS-LOO / WAIC without refits.
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = rnorm(120))
#' d$y <- rpois(120, exp(0.4 + 0.6 * d$x))
#' fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
#' rl  <- tulpa_reloo(fit, data = d)
#' rl$elpd_loo
#' }
#' @export
tulpa_reloo <- function(object, data, k_threshold = .nl_diag("k_usable"),
                        n_trials = NULL, ndraws = NULL) {
  su <- .cv_refit_setup(object, data, n_trials, "reloo")
  cl <- su$cl; data <- su$data; n <- su$n
  fam <- su$fam; phi <- su$phi; y <- su$y; nt <- su$nt

  # PSIS-LOO baseline from the fit's own pointwise log-likelihood.
  ll  <- .tulpa_pointwise_loglik(object, ndraws = ndraws,
                                 caller = "tulpa_reloo()")
  cr  <- tulpa_criteria(ll, criteria = "loo", pointwise = TRUE)
  pointwise <- cr$pointwise$elpd_loo
  pareto_k  <- cr$pointwise$pareto_k

  flagged <- which(is.finite(pareto_k) & pareto_k > k_threshold)
  refit_env <- parent.frame()
  inject_trials <- .family_reads_trials(fam)

  for (i in flagged) {
    fit_i <- .cv_refit(cl, data, setdiff(seq_len(n), i), nt, inject_trials,
                       refit_env, paste0("reloo refit without observation ", i),
                       obs_w = object$obs_weights)
    pointwise[i] <- .cv_heldout_lpd(fit_i, data, i, y, nt, fam, phi,
                                    paste0("Observation ", i),
                                    phi2 = object$phi2)
  }

  elpd <- sum(pointwise)
  se   <- sqrt(n * stats::var(pointwise))
  list(elpd_loo = elpd, se_elpd_loo = se, looic = -2 * elpd,
       pointwise = pointwise, reloo_idx = flagged, pareto_k = pareto_k,
       k_threshold = k_threshold)
}
