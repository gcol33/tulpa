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
#' @param n_trials Optional held-out binomial denominators (length `nrow(data)`);
#'   defaults to 1 (Bernoulli / single-trial).
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
  nt <- if (is.null(n_trials)) rep(1L, n) else as.integer(n_trials)

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
  pointwise <- rep(NA_real_, n)

  for (k in fold_ids) {
    test  <- which(folds == k)
    train <- which(folds != k)

    ccl <- cl
    ccl$data <- data[train, , drop = FALSE]
    fit_k <- tryCatch(eval(ccl, envir = refit_env),
                      error = function(e) {
                        stop("K-fold refit on fold ", k, " failed: ",
                             conditionMessage(e), ". The stored call's ",
                             "arguments must be resolvable in the calling ",
                             "scope.", call. = FALSE)
                      })

    beta <- stats::coef(fit_k)
    Xte  <- .tulpa_fixed_design(fit_k, data[test, , drop = FALSE])
    miss <- setdiff(names(beta), colnames(Xte))
    if (length(miss)) {
      stop("Fold ", k, ": held-out data cannot reproduce fixed-effect ",
           "column(s): ", paste(miss, collapse = ", "), call. = FALSE)
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

    pointwise[test] <- apply(ll, 1L, logmeanexp)
  }

  elpd <- sum(pointwise)
  se   <- sqrt(n * stats::var(pointwise))
  list(elpd_kfold = elpd, se_elpd_kfold = se,
       pointwise = pointwise, folds = folds, K = K)
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
