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
  breaks <- NULL
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

  # The smoother's role tag tells it apart from a temporal field of the same
  # walk type (`.nl_block_roles()`).
  block <- list(
    type         = type,
    temporal_idx = as.integer(idx),
    n_times      = as.integer(k_eff),
    n_groups     = 1L,
    role         = "smooth"
  )
  if (type == "rw1") block$cyclic <- FALSE

  list(block = block,
       meta  = list(var = var, nodes = nodes, k = as.integer(k_eff),
                    type = type, idx = as.integer(idx), breaks = breaks))
}

# The node of the smoother `meta` (an entry of a fit's `smooth_terms`) each row
# of `newdata` falls on, by the rule the fit binned its covariate with: the
# bin of the equally spaced breaks, or the matching value where the nodes are
# the covariate's distinct values. A value beyond the fitted range, or between
# two distinct fitted values, is refused: the smooth carries a value at its
# nodes only.
#' @keywords internal
.smooth_node_at <- function(meta, newdata) {
  x <- tryCatch(eval(str2lang(meta$var), envir = newdata,
                     enclos = globalenv()),
                error = function(e) {
                  stop("newdata cannot evaluate the covariate of s(", meta$var,
                       "): ", conditionMessage(e), call. = FALSE)
                })
  x <- as.numeric(x)
  if (length(x) != nrow(newdata) || anyNA(x)) {
    stop("s(", meta$var, ") needs a finite covariate value on every row of ",
         "newdata.", call. = FALSE)
  }
  br <- meta$breaks
  if (is.null(br)) {
    tol <- 1e-8 * max(1, diff(range(meta$nodes)))
    idx <- vapply(x, function(v) {
      j <- which(abs(meta$nodes - v) <= tol)
      if (length(j)) j[1L] else NA_integer_
    }, integer(1))
    if (anyNA(idx)) {
      stop("s(", meta$var, ") was fitted at the covariate's ", length(meta$nodes),
           " distinct values, and ", sum(is.na(idx)), " newdata value(s) are ",
           "none of them; the smooth carries a value at those values only.",
           call. = FALSE)
    }
    return(as.integer(idx))
  }
  tol <- 1e-8 * max(1, diff(range(br)))
  out <- x < br[1L] - tol | x > br[length(br)] + tol
  if (any(out)) {
    stop(sum(out), " newdata value(s) of ", meta$var, " lie outside the range ",
         "s(", meta$var, ") was fitted over [", signif(br[1L], 6), ", ",
         signif(br[length(br)], 6), "], where the smooth carries no value.",
         call. = FALSE)
  }
  as.integer(findInterval(x, br, rightmost.closed = TRUE, all.inside = TRUE))
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
#' @seealso [field_effects()] for a spatial field, [ranef()] for random
#'   effects.
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
  # A smoother's block is found by its role, the smoothers in the order of
  # `smooth_terms`, at the columns the solve reported for it (a single block
  # is the tail of [fixed effects, random effects, block]).
  blocks <- .field_block_list(object$prior)
  ks <- which(.nl_block_roles(blocks) == "smooth")
  if (length(ks) != length(sm)) {
    stop("The fit's latent blocks do not carry its ", length(sm),
         " smoother(s).", call. = FALSE)
  }
  cols <- if (length(blocks) == 1L) {
    seq.int(ncol(modes) - sm[[term]]$k + 1L, ncol(modes))
  } else {
    .nl_block_cols(object$block_latent_offsets, ncol(modes), ks[term],
                   length(blocks))
  }
  if (length(cols) != sm[[term]]$k) {
    stop("The fit does not locate the latent block of s(", sm[[term]]$var,
         ").", call. = FALSE)
  }

  on  <- is.finite(w) & w > 0
  wn  <- w[on] / sum(w[on])
  est <- as.numeric(crossprod(wn, modes[on, cols, drop = FALSE]))
  out <- data.frame(x = sm[[term]]$nodes, estimate = est)
  attr(out, "var") <- sm[[term]]$var
  out
}


# The smoother half of a fit's display, read from the metadata `tulpa()` attaches
# for smooth_effects(). Silent on a fit with no s(...) terms, so
# `print.summary.tulpa_fit()` can call it unconditionally and any fit carrying smoothers
# lists them.
#' @keywords internal
.print_smooth_section <- function(x) {
  sm <- x$smooth_terms
  if (!length(sm)) return(invisible(NULL))
  cat("\nSmooth terms:\n")
  print(data.frame(
    term  = vapply(sm, function(s) paste0("s(", s$var, ")"), character(1)),
    basis = vapply(sm, function(s) toupper(s$type), character(1)),
    nodes = vapply(sm, function(s) as.integer(s$k), integer(1)),
    row.names = NULL))
  cat("\nFitted values: smooth_effects(fit), plot(fit, type = \"smooth\")\n")
  invisible(NULL)
}


# One panel per s(...) term, from the node-by-node posterior smooth_effects()
# returns. Behind `plot(fit, type = "smooth")`; errors on a fit with no smoothers
# rather than drawing an empty frame.
#' @keywords internal
.plot_smooths <- function(x, term = NULL, ...) {
  sm <- x$smooth_terms
  if (!length(sm)) {
    stop("This fit carries no s(...) smoother terms to plot.", call. = FALSE)
  }
  which_terms <- if (is.null(term)) seq_along(sm) else term
  if (length(which_terms) > 1L) {
    op <- graphics::par(mfrow = grDevices::n2mfrow(length(which_terms)))
    on.exit(graphics::par(op), add = TRUE)
  }
  for (j in which_terms) {
    e <- smooth_effects(x, j)
    graphics::plot(e$x, e$estimate, type = "l",
                   xlab = attr(e, "var"),
                   ylab = paste0("s(", attr(e, "var"), ")"), ...)
  }
  invisible(x)
}
