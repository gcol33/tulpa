# Posterior sampling from a nested-Laplace outer-grid mixture -- the front door
# and the piece both backends share.
#
# Every nested tier defines the same object: a Gaussian mixture over the outer
# hyperparameter grid,
#
#   x ~ sum_k w_k N(m_k, V_k),
#
# and a draw from it is two steps -- pick a cell by weight, then draw that cell's
# Gaussian. What differs between backends is only the second step and what a
# component covers: the joint backend keeps each cell's full sparse precision, so
# its component is the whole latent vector; the single-block backend releases
# that precision once the fixed-effect block is extracted, so its component is
# that block. The allocation, the row bookkeeping and the provenance attributes
# are written once here and driven by a per-backend cell sampler.

#' Posterior draws from a nested-Laplace fit
#'
#' @description
#' Draw from the outer-grid mixture posterior of a nested-Laplace fit -- the
#' engine analogue of `inla.posterior.sample()`. Each draw picks an outer-grid
#' cell `k ~ Categorical(weights)` and then samples that cell's inner Gaussian,
#' so the draws are i.i.d. samples from `sum_k w_k N(m_k, V_k)`.
#'
#' Sampling the mixture is the faithful primitive for marginalizing nonlinear
#' derived quantities (e.g. `plogis(eta_2) - plogis(eta_1)`, expected-cover
#' products `p * mu`): compute the derived quantity per draw, then summarize.
#' Collapsing the grid to a single moment-matched Gaussian biases skewed or
#' multimodal-over-grid posteriors.
#'
#' # What a draw covers
#'
#' It depends on which representation the backend retained, and the returned
#' matrix says so in its `scope` attribute.
#'
#' \describe{
#'   \item{`tulpa_nested_laplace_joint`}{the FULL latent vector -- per-arm fixed
#'     effects, per-arm random effects, then the latent field(s) -- because the
#'     joint fit retains each cell's sparse precision over that vector
#'     (`control$store_Q = TRUE`). `scope` is `"latent"`.}
#'   \item{`tulpa_nested_laplace` (single-block)}{the FIXED-EFFECT block. This
#'     backend inverts each cell's precision into the marginal fixed-effect
#'     block and releases the precision itself, so the latent field is not part
#'     of the retained per-cell Gaussian and cannot be sampled from the fit.
#'     `scope` is `"fixed"`.}
#' }
#'
#' @param fit A nested-Laplace fit ([tulpa_nested_laplace()] or
#'   [tulpa_nested_laplace_joint()]).
#' @param idx Optional integer vector of 1-based indices into whatever a draw
#'   covers for this backend (see above); `NULL` (default) returns all of them.
#' @param n Number of posterior draws (default 1000).
#' @param ... Unused; for S3 compatibility.
#'
#' @return A numeric matrix `[n x length(idx)]`, one row per draw. Carries
#'   `attr(., "draws_kind") = "iid"` (consistent with the draws-provenance
#'   gate), `attr(., "cells")` -- the outer-grid cell index each row was drawn
#'   from -- and `attr(., "scope")`, which of the two representations above the
#'   columns are.
#'
#' @seealso [tulpa_nested_laplace()], [tulpa_nested_laplace_joint()],
#'   [posterior_sample()]
#' @export
tulpa_posterior_draws <- function(fit, idx = NULL, n = 1000, ...) {
    UseMethod("tulpa_posterior_draws")
}

#' @export
tulpa_posterior_draws.default <- function(fit, idx = NULL, n = 1000, ...) {
    stop("tulpa_posterior_draws() is implemented for nested-Laplace fits ",
         "(classes 'tulpa_nested_laplace' and 'tulpa_nested_laplace_joint'). ",
         "Got: ", paste(class(fit), collapse = ", "), ".", call. = FALSE)
}

#' @export
tulpa_posterior_draws.tulpa_nested_laplace <- function(fit, idx = NULL,
                                                       n = 1000, ...) {
    n <- .nl_draws_check_n(n)

    # The posterior IS the retained mixture: `.nested_fixed_moments()` is the one
    # marginalizer every nested-tier read goes through, so sampling reads exactly
    # the components `summary()` / `confint()` report from and the two answers
    # cannot drift apart (gcol33/tulpa#347).
    mom <- .nested_fixed_moments(fit)
    if (is.null(mom)) {
        stop("tulpa_posterior_draws(): this fit retains no per-cell ",
             "fixed-effect mixture, so there is nothing to sample. The ",
             "components come from `$grid_modes` / `$grid_hessians` / ",
             "`$weights`; refit with `control$keep_grid_hessians = TRUE` ",
             "(the default on the `tulpa()` front door) to retain them.",
             call. = FALSE)
    }

    p <- ncol(mom$mu)
    if (is.null(idx)) {
        idx <- seq_len(p)
    } else {
        idx <- as.integer(idx)
        if (anyNA(idx) || any(idx < 1L) || any(idx > p)) {
            stop("`idx` must be 1-based indices into the fixed-effect block ",
                 "(1..", p, "). A single-block nested-Laplace fit retains the ",
                 "marginal fixed-effect block of each grid cell and releases ",
                 "the cell precision itself, so the latent field is not part ",
                 "of what this backend can sample.", call. = FALSE)
        }
    }

    H <- fit$grid_hessians
    keep <- mom$keep
    covs <- lapply(keep, function(g) {
        V <- .nested_cell_fixed_cov(H[[g]], p)[idx, idx, drop = FALSE]
        .nl_draw_chol(V)
    })
    bad <- vapply(covs, is.null, logical(1))
    if (any(bad)) {
        stop("tulpa_posterior_draws(): the retained fixed-effect covariance of ",
             sum(bad), " of ", length(covs), " mixture components is not ",
             "positive definite, so those cells cannot be sampled. The cells ",
             "are `$grid_hessians[[k]]` at k = ",
             paste(keep[bad], collapse = ", "), ".", call. = FALSE)
    }

    mu <- mom$mu[, idx, drop = FALSE]
    out <- .nl_mixture_draw(
        w = mom$w, cell_id = keep, n = n, p = length(idx),
        draw_cell = function(i, n_i) {
            Z <- matrix(stats::rnorm(n_i * length(idx)), n_i, length(idx))
            Z %*% covs[[i]] + rep(mu[i, ], each = n_i)
        })

    nm <- fit$fixed_names %||% fit$param_names
    colnames(out) <- if (!is.null(nm) && length(nm) >= max(idx)) {
        nm[idx]
    } else paste0("beta", idx)
    attr(out, "draws_kind") <- "iid"
    attr(out, "scope") <- "fixed"
    # Same provenance the interval read carries (gcol33/tulpa#342): a grid that
    # dropped a positive-weight cell is sampled conditional on the cells that
    # remain, and `retained_mass` below 1 is how a reader tells that apart from a
    # complete grid.
    attr(out, "retained_mass") <- mom$mass
    out
}

# Shared draw allocator for an outer-grid Gaussian mixture. `w` are the
# pre-normalized component weights, `cell_id[i]` the fit's own grid-cell index of
# component i (recorded per row so a caller can group draws by cell), `p` the
# column count, and `draw_cell(i, n_i)` an n_i x p block of draws from component
# i. Rows are allocated across components by one multinomial rather than one
# categorical per draw, so a component's block is sampled in a single vectorized
# call.
.nl_mixture_draw <- function(w, cell_id, n, p, draw_cell) {
    counts <- as.integer(stats::rmultinom(1L, size = n, prob = w))
    out <- matrix(0.0, n, p)
    row_cells <- integer(n)
    pos <- 0L
    for (i in seq_along(w)) {
        n_i <- counts[i]
        if (n_i == 0L) next
        rows <- (pos + 1L):(pos + n_i)
        out[rows, ] <- draw_cell(i, n_i)
        row_cells[rows] <- cell_id[i]
        pos <- pos + n_i
    }
    attr(out, "cells") <- row_cells
    out
}

# Upper-triangular Cholesky factor R of a dense component covariance (so
# `Z %*% R` has covariance V for standard normal rows), or NULL when V is not
# usable -- the caller turns that into a message naming the offending cells
# rather than emitting silently wrong draws.
.nl_draw_chol <- function(V) {
    if (!all(is.finite(V))) return(NULL)
    tryCatch(chol(V), error = function(e) NULL)
}

# Shared `n` validation for the posterior-draw methods.
.nl_draws_check_n <- function(n) {
    n <- as.integer(n)
    if (length(n) != 1L || is.na(n) || n < 1L) {
        stop("`n` must be a single positive integer.", call. = FALSE)
    }
    n
}
