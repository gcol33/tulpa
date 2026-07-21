# ============================================================================
# Random-effect covariance summary.
#
# `ranef()` reports the per-level deviations; this reports the covariance they
# are drawn from -- the quantity a user reads to answer "how much between-group
# variation is there", and the one lme4 spells `VarCorr`.
#
# Where the number comes from depends on what the backend did with Sigma, and
# the three cases are NOT interchangeable, so each is labelled rather than
# flattened into one column:
#
#   estimated    the backend maximized or integrated over Sigma and this is the
#                result (empirical Bayes, the RE-covariance integrators).
#   sampled      Sigma was a sampled parameter; the summary is a posterior mean
#                over draws.
#   conditioned  Sigma was fixed by the caller and the fit conditions on it, so
#                this is an input echoed back, not an estimate.
#
# Reporting a conditioned value without saying so would present `sigma_re = 1`
# -- the default when a user says nothing -- as though the data had produced it.
# ============================================================================


#' Random-effect variances and correlations
#'
#' @description
#' Summarize the random-effect covariance of a fit: the standard deviation of
#' each random-effect coefficient, the correlations between them within a term,
#' and the covariance matrix itself.
#'
#' Whether the covariance was estimated, sampled, or merely conditioned on is
#' reported alongside it, because the three are different claims. A fit from
#' `mode = "laplace"` conditions on `sigma_re`, so its values are the ones
#' supplied; `mode = "eb"` estimates them; a sampler tier integrates them.
#'
#' @param x A `tulpa_fit`.
#' @param sigma Ignored, for compatibility with the `VarCorr` generic.
#' @param ... Ignored.
#'
#' @return A data frame with one row per random-effect coefficient: `term`,
#'   `coef`, `sd`, and `source` (one of `"estimated"`, `"sampled"`,
#'   `"conditioned"`). Correlated terms additionally carry the covariance
#'   matrices in the `"cov"` attribute, one per term, each with a `"correlation"`
#'   attribute. Returns an empty data frame when the fit has no random effects.
#'
#' @seealso [ranef()] for the per-level deviations, [tulpa_eb()] to estimate the
#'   covariance rather than condition on it.
#' @examples
#' \donttest{
#' set.seed(1)
#' G <- 30L; per <- 10L; n <- G * per
#' grp <- rep(seq_len(G), each = per); x <- rnorm(n)
#' b <- rnorm(G, 0, 0.8)
#' d <- data.frame(y = rpois(n, exp(0.3 + 0.5 * x + b[grp])), x = x,
#'                 g = factor(grp))
#' fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb")
#' VarCorr(fit)
#' }
#' @export
VarCorr <- function(x, sigma = 1, ...) UseMethod("VarCorr")


# Per-term covariance matrices a fit carries directly, from the backends that
# determine Sigma themselves. `$Sigma` is a bare matrix for a single block and a
# named list for several.
#' @keywords internal
.varcorr_from_sigma <- function(object) {
  S <- object$Sigma
  if (is.null(S)) return(NULL)
  if (is.matrix(S)) list(S) else if (is.list(S)) S else NULL
}


# Posterior-mean covariance from a sampler fit. `log_sigma_re[...]` columns hold
# the log SDs; averaging exp() of the draws is the posterior mean SD, which is
# not exp(mean(log sigma)) -- the difference is the whole reason to average on
# the SD scale rather than the log one.
#' @keywords internal
.varcorr_from_draws <- function(object, layout) {
  dm <- tryCatch(.re_draws_mat(object), error = function(e) NULL)
  if (is.null(dm) || is.null(colnames(dm))) return(NULL)
  cols <- grep("^log_sigma_re", colnames(dm))
  if (!length(cols)) return(NULL)
  sds <- colMeans(exp(dm[, cols, drop = FALSE]))

  nc_all <- vapply(layout, function(rt) as.integer(rt$n_coefs %||% 1L),
                   integer(1))
  if (length(sds) != sum(nc_all)) return(NULL)
  pos <- 0L
  out <- vector("list", length(layout))
  for (m in seq_along(layout)) {
    s <- sds[pos + seq_len(nc_all[m])]
    pos <- pos + nc_all[m]
    out[[m]] <- diag(s^2, nrow = length(s))
  }
  out
}


# The SDs a conditioning fit was given. Recovered from the call rather than
# guessed: a fit that conditioned on the default reports that default, which is
# the case most worth labelling.
#' @keywords internal
.varcorr_from_conditioned <- function(object, layout) {
  s <- object$sigma_re
  if (is.null(s)) {
    cl <- object$call
    if (!is.null(cl) && !is.null(cl$sigma_re)) {
      s <- tryCatch(eval(cl$sigma_re, parent.frame()), error = function(e) NULL)
    }
  }
  nc_all <- vapply(layout, function(rt) as.integer(rt$n_coefs %||% 1L),
                   integer(1))
  if (is.null(s)) s <- 1
  s <- as.numeric(s)
  # One value per TERM is the front door's contract; recycle it across a term's
  # coefficients.
  if (length(s) == 1L) s <- rep(s, length(layout))
  if (length(s) != length(layout)) return(NULL)
  lapply(seq_along(layout), function(m) diag(s[m]^2, nrow = nc_all[m]))
}


#' @rdname VarCorr
#' @export
VarCorr.tulpa_fit <- function(x, sigma = 1, ...) {
  layout <- x$re_layout
  if (is.null(layout) || length(layout) == 0L) {
    return(data.frame(term = character(0), coef = character(0),
                      sd = numeric(0), source = character(0),
                      stringsAsFactors = FALSE))
  }

  # Order matters: a fit that estimated Sigma also has a conditioning fallback
  # available, and reporting the fallback would understate what it did.
  cov_list <- .varcorr_from_sigma(x)
  src <- "estimated"
  if (is.null(cov_list)) {
    cov_list <- .varcorr_from_draws(x, layout)
    src <- "sampled"
  }
  if (is.null(cov_list)) {
    cov_list <- .varcorr_from_conditioned(x, layout)
    src <- "conditioned"
  }
  if (is.null(cov_list) || length(cov_list) != length(layout)) {
    return(data.frame(term = character(0), coef = character(0),
                      sd = numeric(0), source = character(0),
                      stringsAsFactors = FALSE))
  }

  rows <- list()
  covs <- vector("list", length(layout))
  for (m in seq_along(layout)) {
    rt <- layout[[m]]
    Sm <- as.matrix(cov_list[[m]])
    cls <- rt$coef_labels %||% "(Intercept)"
    if (length(cls) != nrow(Sm)) cls <- paste0("coef", seq_len(nrow(Sm)))
    sd_m <- sqrt(pmax(diag(Sm), 0))
    # Correlation is only defined where every SD is positive; a collapsed
    # component leaves it NA rather than dividing by zero.
    R <- if (all(sd_m > 0)) Sm / tcrossprod(sd_m) else
      matrix(NA_real_, nrow(Sm), ncol(Sm))
    dimnames(Sm) <- list(cls, cls)
    dimnames(R) <- list(cls, cls)
    attr(Sm, "correlation") <- R
    covs[[m]] <- Sm
    rows[[m]] <- data.frame(
      term = rt$group_var %||% paste0("term", m),
      coef = cls, sd = sd_m, source = src,
      row.names = NULL, stringsAsFactors = FALSE)
  }
  names(covs) <- vapply(seq_along(layout), function(m)
    layout[[m]]$group_var %||% paste0("term", m), character(1))

  out <- do.call(rbind, rows)
  attr(out, "cov") <- covs
  out
}
