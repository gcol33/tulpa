# smoother.R
# ------------------------------------------------------------------------------
# Covariate smoothers: `s(x)` in a tulpa() formula puts an RW2 (default) or RW1
# GMRF over the binned covariate -- the temporal-field construction with the
# bin index as "time", so the smoothness hyperparameter integrates through the
# same nested-Laplace temporal kernels (single block alone, the joint
# multi-block stack alongside a spatial field or other blocks). The covariate
# is binned onto k equally spaced nodes (or its sorted unique values when there
# are fewer than k), which keeps the walk's equal-spacing assumption.
# ------------------------------------------------------------------------------

# Build a nested-Laplace temporal-shaped block (plus metadata) from one
# unevaluated `s(...)` formula call. Recognized arguments:
#   s(x)                       RW2 over <= 30 nodes (the default)
#   s(x, k = 20)               node budget
#   s(x, structure = "rw1")    first-difference walk instead of RW2
#' @keywords internal
.smooth_block_from_call <- function(cl, data, env) {
  if (length(cl) < 2L) {
    stop("s(...) needs a covariate, e.g. s(elevation).", call. = FALSE)
  }
  args  <- as.list(cl)[-1]
  nm    <- names(args) %||% rep("", length(args))
  x_pos <- which(nm == "")[1]
  if (is.na(x_pos)) {
    stop("s(...) needs the covariate as its first (unnamed) argument.",
         call. = FALSE)
  }
  var_expr <- args[[x_pos]]
  var <- paste(deparse(var_expr), collapse = "")

  x <- eval(var_expr, envir = data, enclos = env)
  if (!is.numeric(x)) {
    stop("s(", var, "): the covariate must be numeric.", call. = FALSE)
  }
  if (anyNA(x) || any(!is.finite(x))) {
    stop("s(", var, "): the covariate contains NA / non-finite values.",
         call. = FALSE)
  }

  k <- as.integer(eval(args$k %||% 30L, envir = env))
  if (is.na(k) || k < 3L) {
    stop("s(", var, "): `k` must be an integer >= 3.", call. = FALSE)
  }
  type <- match.arg(as.character(eval(args$structure %||% "rw2", envir = env)),
                    c("rw2", "rw1"))

  ux <- sort(unique(x))
  if (length(ux) < 3L) {
    stop("s(", var, "): needs at least 3 distinct covariate values.",
         call. = FALSE)
  }
  if (length(ux) <= k) {
    nodes <- ux
    idx   <- match(x, ux)
    k_eff <- length(ux)
  } else {
    breaks <- seq(min(x), max(x), length.out = k + 1L)
    idx    <- findInterval(x, breaks, rightmost.closed = TRUE,
                           all.inside = TRUE)
    nodes  <- (breaks[-1L] + breaks[-(k + 1L)]) / 2
    k_eff  <- k
  }

  block <- list(
    type         = type,
    temporal_idx = as.integer(idx),
    n_times      = as.integer(k_eff),
    n_groups     = 1L
  )
  if (type == "rw1") block$cyclic <- FALSE

  list(block = block,
       meta  = list(var = var, nodes = nodes, k = as.integer(k_eff),
                    type = type, idx = as.integer(idx)))
}

#' Extract fitted covariate smooths
#'
#' @description
#' The grid-marginalized posterior mean of each `s(x)` term's latent values,
#' one estimate per node (bin midpoint or unique covariate value). The latent
#' block values are read from the fit's per-grid modes, weighted by the
#' hyperparameter grid weights.
#'
#' @param object A `tulpa_fit` from [tulpa()] with `s(...)` term(s) in the
#'   formula.
#' @param term Which smoother: index or covariate name. Default 1.
#' @return A data frame with columns `x` (node location) and `estimate` (the
#'   posterior-mean smooth at that node), with the covariate name as an
#'   attribute `"var"`.
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = runif(300, -2, 2))
#' d$y <- rpois(300, exp(0.3 + sin(2 * d$x)))
#' fit <- tulpa(y ~ s(x), data = d, family = "poisson")
#' sm <- smooth_effects(fit)
#' plot(sm$x, sm$estimate, type = "l")
#' }
#' @export
smooth_effects <- function(object, term = 1L) {
  sm <- object$smooth_terms
  if (is.null(sm) || !length(sm)) {
    stop("This fit carries no s(...) smoother terms.", call. = FALSE)
  }
  if (is.character(term)) {
    j <- match(term, vapply(sm, function(s) s$var, character(1)))
    if (is.na(term <- j)) {
      stop("No smoother on covariate '", term, "'.", call. = FALSE)
    }
  }
  term <- as.integer(term)
  if (term < 1L || term > length(sm)) {
    stop("`term` must be in 1..", length(sm), ".", call. = FALSE)
  }

  modes <- object$modes
  w     <- object$weights
  if (!is.matrix(modes) || is.null(w)) {
    stop("The fit carries no per-grid modes to read the smooth from.",
         call. = FALSE)
  }
  # Smoother blocks are appended LAST in the latent stack (beta, RE, blocks...),
  # so term j's values are indexed from the tail of the joint mode vector.
  # A latent(...) block would sit after them, making tail indexing wrong.
  if ((object$n_latent_blocks %||% 0L) > 0L) {
    stop("smooth_effects() cannot locate the smoother blocks when latent(...) ",
         "blocks are present; read fit$modes directly.", call. = FALSE)
  }
  end <- ncol(modes)
  cols <- NULL
  for (j in rev(seq_along(sm))) {
    kj <- sm[[j]]$k
    if (j == term) cols <- (end - kj + 1L):end
    end <- end - kj
  }

  wn  <- w / sum(w)
  est <- as.numeric(crossprod(wn, modes[, cols, drop = FALSE]))
  out <- data.frame(x = sm[[term]]$nodes, estimate = est)
  attr(out, "var") <- sm[[term]]$var
  out
}
