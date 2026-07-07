# bayes_r2.R
# ------------------------------------------------------------------------------
# Bayesian R-squared (Gelman, Goodrich, Gabry & Vehtari 2019): per posterior
# draw, the variance of the fitted response means over that variance plus the
# model-based residual variance. The residual variance is the family's response
# variance Var(y | eta) averaged over observations (sigma^2 for gaussian,
# mean of mu_i for poisson, ...), so the definition covers every builtin
# family, and the per-draw values give a full posterior for R^2.
# ------------------------------------------------------------------------------

#' Bayesian R-squared
#'
#' @description
#' Per posterior draw `s`, `R2_s = Var_i(mu_si) / (Var_i(mu_si) + Var_res_s)`,
#' where `mu_si` are the response-scale fitted means and `Var_res_s` is the
#' family's residual variance averaged over observations (Gelman et al. 2019).
#' The linear predictor is rebuilt per draw exactly as in
#' [posterior_predict()] (fixed effects + random effects + offset at the
#' training data).
#'
#' @param object A `tulpa_fit` object from [tulpa()].
#' @param ... Passed to methods.
#' @return With `summary = TRUE` (default) a one-row data frame with
#'   `estimate` (posterior median), `std.error`, and the `probs` quantiles;
#'   with `summary = FALSE` the vector of per-draw R^2 values.
#' @references
#' Gelman, Goodrich, Gabry & Vehtari (2019). R-squared for Bayesian
#' regression models. \emph{The American Statistician} 73(3):307-309.
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = rnorm(200))
#' d$y <- rnorm(200, 2 * d$x, 1)
#' fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi = 1)
#' bayes_R2(fit)
#' }
#' @export
bayes_R2 <- function(object, ...) {
  UseMethod("bayes_R2")
}

#' @param ndraws Number of posterior draws to use. Defaults to all stored
#'   draws, or 400 on the draw-free Laplace tier.
#' @param summary Summarize the per-draw values (default `TRUE`).
#' @param probs Quantiles reported by the summary (default 2.5% / 97.5%).
#' @param seed Optional integer seed (RNG state is restored on exit), used by
#'   the Gaussian fixed-effect sampling on draw-free fits.
#' @rdname bayes_R2
#' @export
bayes_R2.tulpa_fit <- function(object, ndraws = NULL, summary = TRUE,
                               probs = c(0.025, 0.975), seed = NULL, ...) {
  if (!is.character(object$family) || length(object$family) != 1L) {
    stop("bayes_R2() supports fits with a builtin character family.",
         call. = FALSE)
  }
  if (!is.null(seed)) {
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
      get(".Random.seed", envir = .GlobalEnv)
    } else NULL
    set.seed(seed)
    on.exit({
      if (is.null(old_seed)) rm(".Random.seed", envir = .GlobalEnv)
      else assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }, add = TRUE)
  }

  eta <- .tulpa_eta_draws(object, ndraws = ndraws)
  n_trials <- object$n_trials
  phi <- object$phi %||% 1.0
  fam <- object$family

  r2 <- vapply(seq_len(nrow(eta)), function(s) {
    e <- eta[s, ]
    mu <- family_response_mean(e, fam, n_trials = n_trials)
    var_fit <- stats::var(mu)
    var_res <- mean(family_variance(e, fam, n_trials = n_trials, phi = phi))
    var_fit / (var_fit + var_res)
  }, numeric(1))

  if (!summary) return(r2)
  q <- stats::quantile(r2, probs)
  out <- data.frame(estimate = stats::median(r2), std.error = stats::sd(r2))
  for (i in seq_along(probs)) out[[names(q)[i]]] <- q[[i]]
  rownames(out) <- "R2"
  out
}
