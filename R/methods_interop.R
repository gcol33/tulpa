# ============================================================================
# lme4- / posterior-facing accessors.
#
# `fixef` and `as_draws` are generics other packages already own: lme4 and nlme
# both export `fixef`, posterior exports `as_draws` and its shape-specific
# variants. tulpa declares its own so the accessors work with nothing else
# attached, and .onLoad() (R/zzz.R) additionally registers the tulpa_fit methods
# against whichever of those packages is installed. Declaring a generic without
# that registration is what makes a package mask lme4::fixef when both are
# attached -- dispatch then depends on attach order.
#
# The registration direction matters: tulpa's methods are registered ON the
# other packages' generics, so `lme4::fixef(fit)` resolves without lme4, nlme or
# posterior appearing in Imports. Nothing here loads them.
# ============================================================================


#' Fixed-effect coefficients (lme4-compatible)
#'
#' @description
#' The fixed-effect point estimates, equivalent to [coef()] on a `tulpa_fit`.
#' Provided under the `fixef` name for code written against the `lme4` / `nlme`
#' interface: `lme4::fixef(fit)` and `nlme::fixef(fit)` dispatch here too when
#' either package is installed, so a `tulpa_fit` can be dropped into an
#' `lme4`-shaped workflow.
#'
#' @details
#' Note the deliberate difference from `lme4::coef.merMod`, which returns
#' per-group sums of the fixed and random effects. On a `tulpa_fit`, [coef()]
#' returns the fixed effects alone and `fixef()` is its synonym; the random
#' effects are [ranef()].
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Named numeric vector of fixed-effect estimates.
#' @seealso [coef.tulpa_fit()], [ranef()]
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(80))
#' df$y <- rpois(80, exp(0.5 + 0.3 * df$x))
#' fit <- tulpa(y ~ x, data = df, family = "poisson")
#' fixef(fit)
#' }
#' @export
fixef <- function(object, ...) UseMethod("fixef")

#' @rdname fixef
#' @export
fixef.tulpa_fit <- function(object, ...) stats::coef(object, ...)


#' Posterior draws in the posterior package's format
#'
#' @description
#' Convert a fit's posterior draws to a `posterior` draws object. `as_draws()`
#' returns the `draws_array` shape; `as_draws_array()`, `as_draws_matrix()`,
#' `as_draws_df()` and `as_draws_rvars()` return theirs. When the `posterior`
#' package is installed these are also registered against its generics, so
#' `posterior::as_draws(fit)` works.
#'
#' @details
#' Fits differ in whether they carry draws at all. Sampler and nested-Laplace
#' fits do, and convert directly. A Gaussian-approximation fit (`mode =
#' "laplace"`, `mode = "eb"`) carries a mode and a precision instead, and
#' converting it means *drawing from the approximation* -- which is a modelling
#' decision, not a format change, because every downstream `posterior` summary
#' would then treat a normal approximation as a posterior sample. So it is
#' opt-in: pass `n_draws` to synthesize that many draws from
#' `N(coef(object), vcov(object))`, or get an error naming the alternative.
#' Synthesized draws form a single chain and cover the fixed effects only.
#'
#' @param x A `tulpa_fit` object.
#' @param n_draws Number of draws to synthesize from the Gaussian approximation
#'   for a fit that carries none. `NULL` (default) errors on such a fit rather
#'   than silently approximating. Ignored, with a warning, when the fit already
#'   carries draws.
#' @param seed Optional integer seed for the synthesis. The RNG state is
#'   restored afterwards.
#' @param ... Passed to the corresponding `posterior` converter.
#' @return A `posterior` draws object of the requested shape.
#' @seealso [tulpa_draws_array()] for the base R array without the dependency,
#'   [posterior_sample()] for the raw matrix.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(80))
#' df$y <- rpois(80, exp(0.5 + 0.3 * df$x))
#' fit <- tulpa(y ~ x, data = df, family = "poisson")
#' if (requireNamespace("posterior", quietly = TRUE)) {
#'   posterior::summarise_draws(as_draws(fit))
#' }
#' }
#' @export
as_draws <- function(x, ...) UseMethod("as_draws")

#' @rdname as_draws
#' @export
as_draws.tulpa_fit <- function(x, n_draws = NULL, seed = NULL, ...) {
  as_draws_array.tulpa_fit(x, n_draws = n_draws, seed = seed, ...)
}

#' @rdname as_draws
#' @export
as_draws_array <- function(x, ...) UseMethod("as_draws_array")

#' @rdname as_draws
#' @export
as_draws_array.tulpa_fit <- function(x, n_draws = NULL, seed = NULL, ...) {
  .as_draws_convert(x, "as_draws_array", n_draws, seed, ...)
}

#' @rdname as_draws
#' @export
as_draws_matrix <- function(x, ...) UseMethod("as_draws_matrix")

#' @rdname as_draws
#' @export
as_draws_matrix.tulpa_fit <- function(x, n_draws = NULL, seed = NULL, ...) {
  .as_draws_convert(x, "as_draws_matrix", n_draws, seed, ...)
}

#' @rdname as_draws
#' @export
as_draws_df <- function(x, ...) UseMethod("as_draws_df")

#' @rdname as_draws
#' @export
as_draws_df.tulpa_fit <- function(x, n_draws = NULL, seed = NULL, ...) {
  .as_draws_convert(x, "as_draws_df", n_draws, seed, ...)
}

#' @rdname as_draws
#' @export
as_draws_rvars <- function(x, ...) UseMethod("as_draws_rvars")

#' @rdname as_draws
#' @export
as_draws_rvars.tulpa_fit <- function(x, n_draws = NULL, seed = NULL, ...) {
  .as_draws_convert(x, "as_draws_rvars", n_draws, seed, ...)
}


# Shared conversion body. One place decides what "the draws" are, so every
# shape-specific method agrees about provenance -- the alternative is five
# methods that could each answer the Gaussian-approximation question differently.
.as_draws_convert <- function(x, shape, n_draws, seed, ...) {
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop("`posterior` is required for ", shape, "(); install it from CRAN, or ",
         "use tulpa_draws_array() for the base R [iteration, chain, parameter] ",
         "array.", call. = FALSE)
  }
  arr <- tulpa_draws_array(x)
  if (!is.null(arr)) {
    if (!is.null(n_draws)) {
      warning("`n_draws` is ignored: this fit carries its own posterior draws. ",
              "It only applies to a Gaussian-approximation fit, which has none.",
              call. = FALSE)
    }
  } else {
    arr <- .synth_gaussian_draws(x, n_draws, seed, shape)
  }
  do.call(getExportedValue("posterior", shape), c(list(arr), list(...)))
}


# Draw from the fixed-effect Gaussian approximation N(coef, vcov). Chol-based, so
# a non-PD vcov fails loudly here rather than producing draws from a silently
# repaired matrix.
.synth_gaussian_draws <- function(x, n_draws, seed, shape) {
  if (is.null(n_draws)) {
    stop("This fit carries no posterior draws (backend '", x$backend %||% "?",
         "' reports a mode and a precision, not a sample), so ", shape,
         "() has nothing to convert. Pass `n_draws` to draw that many samples ",
         "from the fixed-effect Gaussian approximation N(coef, vcov) instead -- ",
         "note that those are draws from the approximation, not from the ",
         "posterior -- or refit with a sampler (mode = \"exact\") or the nested ",
         "integrator for genuine draws.", call. = FALSE)
  }
  n_draws <- as.integer(n_draws)
  if (length(n_draws) != 1L || is.na(n_draws) || n_draws < 1L) {
    stop("`n_draws` must be a single positive integer.", call. = FALSE)
  }
  mu <- stats::coef(x)
  V  <- stats::vcov(x)
  if (is.null(mu) || is.null(V)) {
    stop("This fit exposes neither draws nor a coef/vcov pair, so it cannot be ",
         "converted to draws.", call. = FALSE)
  }
  p <- length(mu)
  R <- tryCatch(chol(V), error = function(e)
    stop("The fixed-effect covariance is not positive definite, so the ",
         "Gaussian approximation cannot be sampled: ", conditionMessage(e),
         call. = FALSE))
  z <- .with_preserved_seed({
    if (!is.null(seed)) set.seed(seed)
    matrix(stats::rnorm(n_draws * p), n_draws, p)
  })
  m <- sweep(z %*% R, 2L, mu, `+`)
  colnames(m) <- names(mu)
  array(m, dim = c(n_draws, 1L, p),
        dimnames = list(NULL, NULL, names(mu)))
}
