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
#' [tulpa_criteria()]). Held-out random-effect groups contribute at their prior
#' mean (population-level held-out prediction), matching [predict()].
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
  if (!inherits(object, "tulpa_fit")) {
    stop("`object` must be a tulpa_fit.", call. = FALSE)
  }
  if (!is.null(object$spatial) || !is.null(object$temporal) ||
      !is.null(object$temporal_field)) {
    stop("Refit-based K-fold is not supported for spatial / temporal-field ",
         "fits: subsetting the observations breaks the field structure. Use ",
         "tulpa_criteria() (PSIS-LOO) for those.", call. = FALSE)
  }
  cl <- object$call
  if (is.null(cl)) {
    stop("`object` carries no `$call` to refit from; fit through tulpa().",
         call. = FALSE)
  }
  if (is.null(object$formula)) {
    stop("`object` carries no `$formula`; cannot recover the response.",
         call. = FALSE)
  }

  data <- as.data.frame(data)
  n    <- nrow(data)
  fam  <- object$family
  phi  <- object$phi %||% 1.0
  y    <- eval(object$formula[[2L]], envir = data)
  if (length(y) != n) {
    stop("The response evaluated on `data` has length ", length(y),
         " but `data` has ", n, " rows.", call. = FALSE)
  }
  nt <- .cv_trials(object, n_trials, n)

  if (!is.null(seed)) set.seed(as.integer(seed))
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
  inject_trials <- !is.null(cl$n_trials) || !is.null(object$n_trials)
  pointwise <- rep(NA_real_, n)

  for (k in fold_ids) {
    test  <- which(folds == k)
    train <- which(folds != k)
    fit_k <- .cv_refit(cl, data, train, nt, inject_trials, refit_env,
                       paste0("K-fold refit on fold ", k),
                       obs_w = object$weights)
    pointwise[test] <- .cv_heldout_lpd(fit_k, data, test, y, nt, fam, phi,
                                       paste0("Fold ", k))
  }

  elpd <- sum(pointwise)
  se   <- sqrt(n * stats::var(pointwise))
  list(elpd_kfold = elpd, se_elpd_kfold = se,
       pointwise = pointwise, folds = folds, K = K)
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
# the rows `test`, under a training refit's fixed-effect posterior. Held-out
# random-effect groups contribute at their prior mean (population level).
.cv_heldout_lpd <- function(fit_k, data, test, y, nt, fam, phi, label) {
  beta <- stats::coef(fit_k)
  Xte  <- .tulpa_fixed_design(fit_k, data[test, , drop = FALSE])
  miss <- setdiff(names(beta), colnames(Xte))
  if (length(miss)) {
    stop(label, ": held-out data cannot reproduce fixed-effect column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  Xte <- Xte[, names(beta), drop = FALSE]

  # Posterior draws of the fixed effects on the training refit; fall back to
  # the point estimate (a single-draw plug-in) when a fit carries no draws.
  B <- .kfold_fixed_draws(fit_k, names(beta))
  eta <- Xte %*% t(B)                          # [n_test x S]

  yte <- y[test]; ntte <- nt[test]
  ll <- vapply(seq_len(ncol(eta)), function(s) {
    family_loglik(eta[, s], yte, fam, n_trials = ntte, phi = phi)
  }, numeric(length(test)))                    # [n_test x S]
  if (is.null(dim(ll))) ll <- matrix(ll, nrow = length(test))

  apply(ll, 1L, logmeanexp)
}

# Fixed-effect posterior draws (`[S x p]`, columns ordered by `bnm`) from a fit,
# via the provenance-agnostic accessor; a point fit yields a single-row plug-in.
.kfold_fixed_draws <- function(fit, bnm) {
  dr <- tryCatch(posterior_sample(fit), error = function(e) NULL)
  if (!is.null(dr) && is.matrix(dr) && all(bnm %in% colnames(dr))) {
    return(dr[, bnm, drop = FALSE])
  }
  matrix(stats::coef(fit)[bnm], nrow = 1L,
         dimnames = list(NULL, bnm))
}

# Pointwise log-likelihood [S x n_obs] for a front-door fit, from the per-draw
# linear predictor (.tulpa_eta_draws) and the stored y / n_trials / phi. The
# input tulpa_criteria() wants when the backend did not store $draws$log_lik.
#' @keywords internal
.tulpa_pointwise_loglik <- function(object, ndraws = NULL) {
  y <- object$y
  if (is.null(y)) {
    stop("The fit stores no `$y`; cannot build the pointwise log-likelihood.",
         call. = FALSE)
  }
  eta <- .tulpa_eta_draws(object, ndraws = ndraws)
  nt  <- object$n_trials
  phi <- object$phi %||% 1.0
  ll  <- matrix(NA_real_, nrow(eta), ncol(eta))
  for (s in seq_len(nrow(eta))) {
    ll[s, ] <- family_loglik(eta[s, ], y, object$family,
                             n_trials = nt, phi = phi)
  }
  ll
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
#' only (subsetting breaks a spatial / temporal field), and held-out
#' random-effect groups contribute at their prior mean.
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
tulpa_reloo <- function(object, data, k_threshold = 0.7,
                        n_trials = NULL, ndraws = NULL) {
  if (!inherits(object, "tulpa_fit")) {
    stop("`object` must be a tulpa_fit.", call. = FALSE)
  }
  if (!is.null(object$spatial) || !is.null(object$temporal) ||
      !is.null(object$temporal_field)) {
    stop("reloo is not supported for spatial / temporal-field fits: ",
         "subsetting the observations breaks the field structure.",
         call. = FALSE)
  }
  cl <- object$call
  if (is.null(cl)) {
    stop("`object` carries no `$call` to refit from; fit through tulpa().",
         call. = FALSE)
  }
  if (is.null(object$formula)) {
    stop("`object` carries no `$formula`; cannot recover the response.",
         call. = FALSE)
  }

  data <- as.data.frame(data)
  n    <- nrow(data)
  fam  <- object$family
  phi  <- object$phi %||% 1.0
  y    <- eval(object$formula[[2L]], envir = data)
  if (length(y) != n) {
    stop("The response evaluated on `data` has length ", length(y),
         " but `data` has ", n, " rows.", call. = FALSE)
  }
  nt <- .cv_trials(object, n_trials, n)

  # PSIS-LOO baseline from the fit's own pointwise log-likelihood.
  ll  <- .tulpa_pointwise_loglik(object, ndraws = ndraws)
  cr  <- tulpa_criteria(ll, criteria = "loo", pointwise = TRUE)
  pointwise <- cr$pointwise$elpd_loo
  pareto_k  <- cr$pointwise$pareto_k

  flagged <- which(is.finite(pareto_k) & pareto_k > k_threshold)
  refit_env <- parent.frame()
  inject_trials <- !is.null(cl$n_trials) || !is.null(object$n_trials)

  for (i in flagged) {
    fit_i <- .cv_refit(cl, data, setdiff(seq_len(n), i), nt, inject_trials,
                       refit_env, paste0("reloo refit without observation ", i),
                       obs_w = object$weights)
    pointwise[i] <- .cv_heldout_lpd(fit_i, data, i, y, nt, fam, phi,
                                    paste0("Observation ", i))
  }

  elpd <- sum(pointwise)
  se   <- sqrt(n * stats::var(pointwise))
  list(elpd_loo = elpd, se_elpd_loo = se, looic = -2 * elpd,
       pointwise = pointwise, reloo_idx = flagged, pareto_k = pareto_k,
       k_threshold = k_threshold)
}
