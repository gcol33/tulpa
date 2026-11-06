#' Spatiotemporal interaction specification
#'
#' @description
#' A Knorr-Held (2000) space-time interaction for [tulpa()]: one latent value
#' \eqn{\delta_{st}} per area and time, added to the linear predictor of every
#' observation at that area and time. Pass the spec as
#' `tulpa(..., spatiotemporal = spatiotemporal(type = "IV"))`; it rides the
#' nested-Laplace path beside the spatial and temporal main effects, with the
#' interaction precision \eqn{\tau} integrated on the outer grid.
#'
#' @param spatial The areal spatial spec whose graph structures the
#'   interaction ([spatial_car()], [spatial_bym2()] or
#'   `list(type = "icar", adjacency = W)`). `NULL` (default) takes the fit's
#'   own `spatial =` spec. The interaction uses the graph's intrinsic CAR
#'   structure \eqn{R_s = D - W} whatever the main effect's family.
#' @param temporal The temporal spec structuring the interaction in time,
#'   [temporal_rw1()] or [temporal_rw2()] (cyclic allowed). `NULL` (default)
#'   takes the fit's own `temporal =` spec.
#' @param type Interaction type:
#'   - `"I"` (or `"iid"`): unstructured, \eqn{R = I_T \otimes I_S}.
#'   - `"II"`: each area its own temporal walk, \eqn{R = R_t \otimes I_S}.
#'   - `"III"`: each time its own spatial ICAR, \eqn{R = I_T \otimes R_s}.
#'   - `"IV"` (or `"separable"`): \eqn{R = R_t \otimes R_s}.
#' @param tau_grid Optional outer-grid nodes for the interaction precision
#'   \eqn{\tau}; the default is the engine's intrinsic-field precision axis.
#' @param shared Not used.
#'
#' @return A `tulpa_spatiotemporal` spec.
#'
#' @details
#' The interaction's prior is \eqn{\delta \sim N(0, (\tau R)^-)} conditioned
#' on \eqn{A \delta = 0}, where the rows of \eqn{A} span the null space of
#' \eqn{R}: for type II a walk's null space per area (its level, and its
#' linear trend under an acyclic RW2), for type III a sum over each connected
#' component of the graph at every time, for type IV both. These are hard
#' linear constraints: the inner Newton solves on the constraint set and the
#' Laplace approximation integrates over the \eqn{ST - r} dimensional
#' subspace, with the log-density normalised by the generalized determinant
#' \eqn{|R|_+}. The interaction therefore carries no area level and no time
#' level (and, under RW2, no area-specific trend) that the main effects
#' already model.
#'
#' The cells are laid out area-fastest: cell \eqn{(s, t)} is coordinate
#' \eqn{(t - 1) S + s}. A time whose rows all have an `NA` response is still a
#' node of the walk, so the interaction forecasts it.
#' [spatiotemporal_effects()] reads the posterior back per cell.
#'
#' @references
#' Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
#' in disease risk. Statistics in Medicine, 19(17-18), 2555-2567.
#'
#' Goicoa, T., Adin, A., Ugarte, M. D., & Hodges, J. S. (2018). In
#' spatio-temporal disease mapping models, identifiability constraints affect
#' PQL and INLA results. Stochastic Environmental Research and Risk Assessment,
#' 32, 749-770.
#'
#' @seealso [spatiotemporal_effects()], [spatial_car()], [temporal_rw1()]
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' S <- 8; n_t <- 6
#' W <- matrix(0, S, S)
#' for (i in 1:(S - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
#' d <- expand.grid(region = 1:S, year = 1:n_t)
#' d$y <- rpois(nrow(d), exp(1 + 0.3 * sin(d$region) + 0.1 * d$year))
#' fit <- tulpa(y ~ 1 + spatial(region), data = d, family = "poisson",
#'              spatial = list(type = "icar", adjacency = W),
#'              temporal = temporal_rw1("year"),
#'              spatiotemporal = spatiotemporal(type = "IV"),
#'              mode = "nested_laplace", control = list(progress = FALSE))
#' head(spatiotemporal_effects(fit, format = "summary"))
#' }
#' @export
spatiotemporal <- function(spatial = NULL,
                           temporal = NULL,
                           type = c("IV", "I", "II", "III", "iid", "separable"),
                           tau_grid = NULL,
                           shared = NULL) {
  type <- match.arg(type)
  type <- switch(type, iid = "I", separable = "IV", type)
  if (!is.null(tau_grid) &&
      (!is.numeric(tau_grid) || !length(tau_grid) ||
       any(!is.finite(tau_grid)) || any(tau_grid <= 0))) {
    stop("`tau_grid` must be a vector of positive finite precisions.",
         call. = FALSE)
  }
  structure(list(spatial = spatial, temporal = temporal, type = type,
                 tau_grid = if (!is.null(tau_grid)) as.numeric(tau_grid)),
            class = "tulpa_spatiotemporal")
}

#' @export
print.tulpa_spatiotemporal <- function(x, ...) {
  cat("tulpa spatiotemporal interaction: Knorr-Held type ", x$type, "\n",
      sep = "")
  invisible(x)
}

# Structure of an intrinsic walk over `n_times` nodes: crossprod of the
# first (RW1) or second (RW2) differences, closed into a ring when cyclic.
.st_walk_structure <- function(n_times, type, cyclic) {
  if (n_times < 2L) {
    stop("A space-time interaction needs at least two time points.",
         call. = FALSE)
  }
  I <- diag(n_times)
  D <- if (cyclic) {
    D1 <- I - I[c(seq.int(2L, n_times), 1L), , drop = FALSE]
    if (type == "rw1") D1 else D1 %*% D1
  } else {
    diff(I, differences = if (type == "rw1") 1L else 2L)
  }
  Matrix::Matrix(crossprod(D), sparse = TRUE)
}

# Basis of the walk structure's null space: the constant, and the linear ramp
# for an acyclic RW2 (on a ring a ramp is not periodic).
.st_walk_null <- function(n_times, type, cyclic) {
  if (type == "rw2" && !cyclic) {
    cbind(1, seq_len(n_times) - (n_times + 1) / 2)
  } else {
    matrix(1, n_times, 1L)
  }
}

# log of the product of the nonzero eigenvalues of a PSD matrix whose null
# space has dimension `k`.
.st_log_pdet <- function(R, k) {
  ev <- sort(eigen(as.matrix(R), symmetric = TRUE, only.values = TRUE)$values,
             decreasing = TRUE)
  m <- length(ev) - k
  if (m <= 0L) return(0)
  sum(log(ev[seq_len(m)]))
}

# The interaction's structure R, its generalized log-determinant and the rows
# spanning null(R), for `type` over S areas (graph `adjacency`) and `n_times`
# times. Cells are area-fastest: cell (s, t) is (t - 1) * S + s, so a
# Kronecker product reads R_t (x) R_s.
#
# null(R_t (x) R_s) = null(R_t) (x) R^S  +  R^T (x) null(R_s): per area the
# walk's null vectors over time (type II), per time the indicator of each
# connected component (type III), both for type IV. Type IV's two families
# overlap in null(R_t) (x) null(R_s), so an independent subset is kept: the
# column-pivoted QR of the rows' Gram matrix picks it, and its size is checked
# against S T - rank(R).
#
# The matrix handed to the kernel is R itself: the driver solves on A x = 0
# (hard_constraints.h), so R need carry no curvature along null(R). An A'A term
# would, but its rows span whole time slices and whole series and fill the
# factor of the inner precision.
.st_interaction_structure <- function(type, adjacency, n_times, walk = "rw1",
                                      cyclic = FALSE) {
  W <- as.matrix(adjacency)
  diag(W) <- 0
  S <- nrow(W)
  n_t <- as.integer(n_times)
  n <- S * n_t
  Rs <- Matrix::Matrix(diag(rowSums(W), S) - W, sparse = TRUE)
  Rt <- .st_walk_structure(n_t, walk, cyclic)
  Is <- Matrix::Diagonal(S)
  It <- Matrix::Diagonal(n_t)
  comps <- .graph_components(W)
  Nt <- .st_walk_null(n_t, walk, cyclic)
  k_t <- ncol(Nt)
  n_c <- length(comps)
  cell <- function(s, t) (t - 1L) * S + s
  rows_area <- unlist(lapply(seq_len(S), function(s) lapply(seq_len(k_t),
    function(j) list(cols = cell(s, seq_len(n_t)), w = as.numeric(Nt[, j])))),
    recursive = FALSE)
  rows_time <- unlist(lapply(seq_len(n_t), function(t) lapply(comps,
    function(cc) list(cols = cell(as.integer(cc), t), w = rep(1, length(cc))))),
    recursive = FALSE)
  R <- switch(type,
    I   = Matrix::kronecker(It, Is),
    II  = Matrix::kronecker(Rt, Is),
    III = Matrix::kronecker(It, Rs),
    IV  = Matrix::kronecker(Rt, Rs))
  rows <- switch(type, I = list(), II = rows_area, III = rows_time,
                 IV = c(rows_area, rows_time))
  rank_R <- switch(type, I = n, II = S * (n_t - k_t), III = (S - n_c) * n_t,
                   IV = (S - n_c) * (n_t - k_t))
  A <- NULL
  if (length(rows)) {
    A <- Matrix::sparseMatrix(
      i = rep(seq_along(rows), lengths(lapply(rows, `[[`, "cols"))),
      j = unlist(lapply(rows, `[[`, "cols")),
      x = unlist(lapply(rows, `[[`, "w")), dims = c(length(rows), n))
  }
  if (length(rows) > n - rank_R) {
    q <- qr(as.matrix(Matrix::tcrossprod(A)), tol = 1e-9)
    keep <- sort(q$pivot[seq_len(q$rank)])
    rows <- rows[keep]
    A <- A[keep, , drop = FALSE]
  }
  if (length(rows) != n - rank_R) {
    stop(sprintf(paste0(
      "Internal: the type %s interaction's constraint rows span %d ",
      "directions, but its structure has a %d-dimensional null space."),
      type, length(rows), n - rank_R), call. = FALSE)
  }
  log_pdet_s <- sum(vapply(comps, function(cc) {
    m <- length(cc)
    if (m < 2L) return(0)
    Lc <- as.matrix(Rs[cc, cc]) + 1 / m
    as.numeric(determinant(Lc, logarithm = TRUE)$modulus)
  }, numeric(1)))
  log_pdet_t <- .st_log_pdet(Rt, k_t)
  log_pdet <- switch(type,
    I   = 0,
    II  = S * log_pdet_t,
    III = n_t * log_pdet_s,
    IV  = (S - n_c) * log_pdet_t + (n_t - k_t) * log_pdet_s)
  S_csc <- methods::as(methods::as(Matrix::forceSymmetric(R),
                                   "generalMatrix"), "CsparseMatrix")
  list(S_p = S_csc@p, S_i = S_csc@i, S_x = S_csc@x, R = R,
       log_pdet = log_pdet, constr = rows, n_latent = n, n_spatial = S,
       n_times = n_t, rank = rank_R)
}

# The nested-Laplace block of a spatiotemporal() spec against `data`: the
# areas from the spatial spec's unit column, the times from the temporal spec,
# and the structure above. `spatial` / `temporal` are the fit's own specs, used
# where the interaction spec leaves its own NULL; `spatial_var` is the formula's
# spatial(col) term.
.st_interaction_block <- function(st, spatial, temporal, data, spatial_var) {
  sp <- st$spatial %||% spatial
  tm <- st$temporal %||% temporal
  if (is.null(sp) || is.null(sp$adjacency)) {
    stop("spatiotemporal() needs an areal graph: give it an areal spatial spec ",
         "(spatial_car() / spatial_bym2() / list(type = 'icar', adjacency = W)) ",
         "or pass one as tulpa(spatial = ).", call. = FALSE)
  }
  if (is.null(tm)) {
    stop("spatiotemporal() needs a temporal walk: give it temporal_rw1() / ",
         "temporal_rw2() or pass one as tulpa(temporal = ).", call. = FALSE)
  }
  walk <- tolower(tm$type %||% "")
  if (!walk %in% c("rw1", "rw2")) {
    stop(sprintf(paste0(
      "spatiotemporal() structures time by an intrinsic walk (rw1 / rw2); a ",
      "'%s' temporal spec has a parameter the interaction's fixed structure ",
      "does not carry."), walk), call. = FALSE)
  }
  if (!is.null(tm$group_var)) {
    stop("spatiotemporal() takes an ungrouped temporal spec; the areas are its ",
         "groups already.", call. = FALSE)
  }
  adj <- as.matrix(.validate_adjacency_arg(
    sp$adjacency, "spatiotemporal()$spatial$adjacency"))
  S <- nrow(adj)
  unit_col <- sp$group_var %||% spatial_var
  if (is.null(unit_col) || !unit_col %in% names(data)) {
    stop("spatiotemporal(): no area column. Give the spatial spec a ",
         "`group_var`, or add a spatial(col) term to the formula.",
         call. = FALSE)
  }
  s_idx <- .resolve_spatial_idx(data[[unit_col]], S, adj, unit_col)
  tv <- validate_temporal(tm, data)
  str <- .st_interaction_structure(st$type, adj, tv$n_times, walk,
                                   isTRUE(tm$cyclic))
  blk <- list(type = "st_interaction",
              obs_idx = as.integer((tv$time_index - 1L) * S + s_idx),
              n_latent = str$n_latent, S_p = str$S_p, S_i = str$S_i,
              S_x = str$S_x, log_pdet = str$log_pdet, constr = str$constr,
              n_spatial = S, n_times = tv$n_times, kh_type = st$type,
              time_levels = tv$time_levels,
              unit_levels = rownames(adj) %||% as.character(seq_len(S)))
  if (!is.null(st$tau_grid)) blk$tau_grid <- st$tau_grid
  blk
}


#' Extract spatiotemporal effects from fitted model
#'
#' @description
#' Extract posterior distributions of spatiotemporal interaction effects
#' from a fitted tulpa model.
#'
#' @param object A nested-Laplace fit of [tulpa()] carrying a
#'   [spatiotemporal()] interaction, or any fit storing a `$spatiotemporal`
#'   Knorr-Held block (`n_spatial`, `n_times`) with its draws in
#'   `$.internal$spatiotemporal_draws` (a model package's own fitter), which
#'   the default method reads.
#' @param format Output format: `"array"` (default, S x T x draws), `"long"`
#'   (data frame with s, t, draw, value columns), or `"summary"` (posterior summaries).
#' @param probs Quantiles to compute if `format = "summary"`.
#' @param ... Ignored
#'
#' @return For `format = "array"`, an S x T x draws numeric array carrying
#'   `n_spatial`, `n_times` and `n_draws` attributes; for `"long"`, a data
#'   frame with columns `s`, `t`, `draw` and `value`; for `"summary"`, a
#'   `tulpa_st_summary` data frame with columns `s`, `t`, `mean`, one column
#'   per entry of `probs` and `sd`, which has `print()` and `plot()` methods.
#'
#' @examples
#' # A Knorr-Held interaction block is attached by a model package's own
#' # fitter (see spatiotemporal() for what the engine fits). The accessor reads
#' # its layout and an n_draws x (S * T) draw matrix stored s-fastest: column
#' # (t - 1) * S + s holds region s at time t. Here both are simulated for
#' # S = 3 regions over T = 4 times.
#' set.seed(1)
#' S <- 3L; n_t <- 4L
#' st_mean <- outer(c(-0.5, 0, 0.5), seq(-0.3, 0.3, length.out = n_t))
#' draws <- matrix(rnorm(100 * S * n_t, mean = rep(c(st_mean), each = 100),
#'                       sd = 0.1), nrow = 100)
#' fit <- structure(
#'   list(spatiotemporal = list(n_spatial = S, n_times = n_t),
#'        .internal = list(spatiotemporal_draws = draws)),
#'   class = "tulpa_fit")
#'
#' st_effects <- spatiotemporal_effects(fit, format = "summary")
#' head(st_effects)
#' dim(spatiotemporal_effects(fit, format = "array"))
#'
#' @export
spatiotemporal_effects <- function(object,
                                   format = c("array", "long", "summary"),
                                   probs = c(0.025, 0.5, 0.975),
                                   ...) {
  UseMethod("spatiotemporal_effects")
}


#' @rdname spatiotemporal_effects
#' @export
spatiotemporal_effects.default <- function(object,
                                           format = c("array", "long", "summary"),
                                           probs = c(0.025, 0.5, 0.975),
                                           ...) {

  format <- match.arg(format)

  # Check if model has spatiotemporal effects
  if (!is.list(object) || is.null(object$spatiotemporal)) {
    stop("Model carries no Knorr-Held interaction. Fit one with ",
         "tulpa(..., spatiotemporal = spatiotemporal(type = ...)); a model ",
         "package's own fitter may attach a `$spatiotemporal` block instead.",
         call. = FALSE)
  }

  st_info <- object$spatiotemporal
  S <- st_info$n_spatial
  n_t <- st_info$n_times

  # Get interaction draws
  st_draws <- object$.internal$spatiotemporal_draws

  if (is.null(st_draws)) {
    stop("Spatiotemporal draws not found in model output", call. = FALSE)
  }

  n_draws <- dim(st_draws)[1]

  if (format == "array") {
    # Reshape to S x T x draws
    result <- array(NA_real_, dim = c(S, n_t, n_draws))
    for (d in seq_len(n_draws)) {
      result[, , d] <- matrix(st_draws[d, ], nrow = S, ncol = n_t, byrow = FALSE)
    }

    attr(result, "n_spatial") <- S
    attr(result, "n_times") <- n_t
    attr(result, "n_draws") <- n_draws
    return(result)

  } else if (format == "long") {
    # Create long-format data frame. Each draw's row is stored s-fastest
    # (matrix(nrow = S, ncol = T) column-major, as in the array format), so
    # enumerate s before t to keep the labels aligned when S != T.
    result <- expand.grid(
      draw = seq_len(n_draws),
      s = seq_len(S),
      t = seq_len(n_t)
    )
    result$value <- as.vector(st_draws)
    result <- result[, c("s", "t", "draw", "value")]
    return(result)

  } else {
    # Compute summary statistics
    st_mat <- matrix(NA_real_, nrow = S * n_t, ncol = 3 + length(probs))
    colnames(st_mat) <- c("s", "t", "mean", paste0("q", probs * 100))

    for (i in seq_len(S)) {
      for (j in seq_len(n_t)) {
        # Draws are stored s-fastest (column-major S x T), matching the array /
        # long formats; index the (s = i, t = j) column accordingly so labels
        # stay aligned when S != T.
        idx <- (j - 1) * S + i
        st_mat[idx, "s"] <- i
        st_mat[idx, "t"] <- j
        st_mat[idx, "mean"] <- mean(st_draws[, idx])

        qs <- quantile(st_draws[, idx], probs = probs)
        for (k in seq_along(probs)) {
          st_mat[idx, 3 + k] <- qs[k]
        }
      }
    }

    result <- as.data.frame(st_mat)
    result$sd <- apply(st_draws, 2, sd)

    attr(result, "n_spatial") <- S
    attr(result, "n_times") <- n_t
    attr(result, "n_draws") <- n_draws
    class(result) <- c("tulpa_st_summary", "data.frame")
    return(result)
  }
}

#' @rdname spatiotemporal_effects
#' @param ndraws Number of joint posterior draws for `format = "array"` or
#'   `"long"` on a fit carrying a [spatiotemporal()] interaction.
#' @param seed Optional integer seed for those draws (RNG state is restored on
#'   exit).
#' @details
#' On a fit carrying a [spatiotemporal()] interaction the summary is the exact
#' Gaussian mixture over the outer grid of each cell's posterior (mean, SD and
#' the `probs` quantiles), the read [latent_effects()] makes for a `latent()`
#' block. The `"array"` and `"long"` formats are joint draws of all S x T
#' cells: a grid cell drawn by its weight, then the latent vector from that
#' cell's Gaussian conditioned on the model's constraints, so a derived
#' quantity over cells (a sum over areas, a change between two times) carries
#' the cells' correlation.
#' @export
spatiotemporal_effects.tulpa_fit <- function(object,
                                             format = c("array", "long", "summary"),
                                             probs = c(0.025, 0.5, 0.975),
                                             ndraws = 1000L, seed = NULL, ...) {
  sizes <- object$grid_spatiotemporal_sizes
  if (!length(sizes)) return(NextMethod())
  format <- match.arg(format)
  b <- as.integer(names(sizes)[1L])
  blk <- object$blocks[[b]]
  S <- as.integer(blk$n_spatial)
  n_t <- as.integer(blk$n_times)
  cols <- seq_len(sizes[1L])
  w <- as.numeric(object$weights)

  if (format == "summary") {
    mu <- object$grid_spatiotemporal_mean[, cols, drop = FALSE]
    V <- object$grid_spatiotemporal_var
    V <- if (is.matrix(V)) V[, cols, drop = FALSE]
    keep <- is.finite(w) & w > 0 & is.finite(rowSums(mu))
    if (!is.null(V)) keep <- keep & is.finite(rowSums(V))
    mx <- .nl_gauss_mixture_summary(mu[keep, , drop = FALSE],
                                    if (!is.null(V)) pmax(V[keep, , drop = FALSE], 0),
                                    w[keep] / sum(w[keep]), probs = probs)
    out <- data.frame(s = rep(seq_len(S), n_t), t = rep(seq_len(n_t), each = S),
                      mean = mx$mean)
    for (k in seq_along(probs)) out[[paste0("q", probs[k] * 100)]] <- mx$quantiles[, k]
    out$sd <- mx$sd
    attr(out, "n_spatial") <- S
    attr(out, "n_times") <- n_t
    class(out) <- c("tulpa_st_summary", "data.frame")
    return(out)
  }

  .seed_scoped(seed)
  draws <- .st_interaction_draws(object, b, as.integer(ndraws))
  spatiotemporal_effects.default(
    list(spatiotemporal = list(n_spatial = S, n_times = n_t),
         .internal = list(spatiotemporal_draws = draws)),
    format = format, probs = probs)
}

# Joint posterior draws of block `b` (an st_interaction block) of a nested
# fit: `ndraws` rows, each a grid cell drawn by weight and the latent vector
# from that cell's inner Gaussian (`.inner_constraint()`), by
# .joint_draw_cell().
.st_interaction_draws <- function(fit, b, ndraws) {
  Qp <- fit$Q_csc_p_per_grid
  Qi <- fit$Q_csc_i_per_grid
  Qx <- fit$Q_csc_x_per_grid
  n_x <- as.integer(fit$Q_csc_n)
  if (is.null(Qp) || !length(n_x)) {
    stop("spatiotemporal_effects(): the fit retains no per-cell precision to ",
         "draw the interaction from.", call. = FALSE)
  }
  off <- fit$block_latent_offsets
  cols <- .nl_block_cols(off, n_x, b, length(off) - 1L)
  cons <- .inner_constraint(fit)
  w <- as.numeric(fit$weights)
  ok <- which(is.finite(w) & w > 0 &
                vapply(seq_along(Qp), function(k) length(Qp[[k]]) == n_x + 1L,
                       logical(1)))
  if (!length(ok)) {
    stop("spatiotemporal_effects(): no outer-grid cell carries a precision ",
         "and a positive weight.", call. = FALSE)
  }
  out <- .nl_mixture_draw(w[ok] / sum(w[ok]), ok, ndraws, length(cols),
    function(i, n_i) {
      k <- ok[i]
      z <- .joint_draw_cell(Qp[[k]], Qi[[k]], Qx[[k]], n_x,
                            .inner_constraint_cell(cons, k), n_i)
      t(fit$modes[k, cols] + z[cols, , drop = FALSE])
    })
  attr(out, "cells") <- NULL
  out
}

#' @rdname spatiotemporal_effects
#' @param x A `tulpa_st_summary` table.
#' @export
print.tulpa_st_summary <- function(x, ...) {
  .print_effect_table(x, sprintf(
    "Spatiotemporal interaction posterior summary: %d locations x %d times",
    attr(x, "n_spatial"), attr(x, "n_times")), ...)
}


#' Plot method for spatiotemporal effects
#'
#' @param x Spatiotemporal effects object
#' @param type Plot type: `"heatmap"` (default), `"time_series"`, or `"spatial_map"`
#' @param ... Additional arguments passed to plotting functions
#'
#' @return A `ggplot` object when ggplot2 is installed; otherwise `NULL`
#'   invisibly, after drawing a base-graphics plot. Called for the side effect
#'   of visualizing the spatiotemporal interaction effects.
#'
#' @importFrom graphics image matplot barplot abline
#' @importFrom grDevices hcl.colors
#'
#' @export
plot.tulpa_st_summary <- function(x, type = "heatmap", ...) {

  S <- attr(x, "n_spatial")
  n_t <- attr(x, "n_times")

  if (type == "heatmap") {
    # Create matrix of means
    mean_mat <- matrix(x$mean, nrow = S, ncol = n_t, byrow = FALSE)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(
        s = rep(seq_len(S), n_t),
        t = rep(seq_len(n_t), each = S),
        value = as.vector(mean_mat)
      )

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$t, y = .data$s, fill = .data$value)) +
        ggplot2::geom_tile() +
        ggplot2::scale_fill_gradient2(
          low = "blue", mid = "white", high = "red",
          midpoint = 0
        ) +
        ggplot2::labs(
          title = "Spatiotemporal Interaction Effects",
          x = "Time",
          y = "Space",
          fill = "Effect"
        ) +
        theme_tulpa()

      return(p)
    }

    # Base R fallback
    image(seq_len(n_t), seq_len(S), t(mean_mat),
          xlab = "Time", ylab = "Space",
          main = "Spatiotemporal Interaction Effects",
          col = hcl.colors(100, "RdBu", rev = TRUE),
          ...)

  } else if (type == "time_series") {
    # Plot time series for each spatial unit
    mean_mat <- matrix(x$mean, nrow = S, ncol = n_t, byrow = FALSE)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(
        s = factor(rep(seq_len(S), n_t)),
        t = rep(seq_len(n_t), each = S),
        value = as.vector(mean_mat)
      )

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$t, y = .data$value, color = .data$s, group = .data$s)) +
        ggplot2::geom_line(alpha = 0.6) +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        ggplot2::labs(
          title = "Spatiotemporal Effects by Location",
          x = "Time",
          y = "Interaction Effect",
          color = "Location"
        ) +
        theme_tulpa()

      return(p)
    }

    # Base R fallback
    matplot(seq_len(n_t), t(mean_mat), type = "l", lty = 1,
            xlab = "Time", ylab = "Interaction Effect",
            main = "Spatiotemporal Effects by Location", ...)
    abline(h = 0, lty = 2, col = "gray50")

  } else if (type == "spatial_map") {
    # The summary is indexed by spatial unit (s = 1..S) x time (t = 1..T) with
    # no geographic coordinates, so "spatial_map" is the time-averaged effect
    # profile over the spatial units, not a coordinate map.
    mean_mat <- matrix(x$mean, nrow = S, ncol = n_t, byrow = FALSE)
    spatial_mean <- rowMeans(mean_mat)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(s = seq_len(S), value = spatial_mean)
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$s, y = .data$value)) +
        ggplot2::geom_col(fill = "steelblue") +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed",
                            color = "gray50") +
        ggplot2::labs(
          title = "Time-averaged spatial effect",
          x = "Spatial unit", y = "Mean interaction effect"
        ) +
        theme_tulpa()
      return(p)
    }

    # Base R fallback
    barplot(spatial_mean, names.arg = seq_len(S),
            xlab = "Spatial unit", ylab = "Mean interaction effect",
            main = "Time-averaged spatial effect", ...)
    abline(h = 0, lty = 2, col = "gray50")

  } else {
    stop(sprintf(paste0(
      "Unknown plot `type` = '%s'. Use 'heatmap', 'time_series', or ",
      "'spatial_map'."), type), call. = FALSE)
  }

  invisible(NULL)
}


#' Non-separable spatiotemporal GP
#'
#' @description
#' Specify a non-separable Gaussian Process for spatiotemporal effects.
#' Unlike separable models where the covariance factors as \eqn{C_s \otimes C_t},
#' non-separable models allow for direct space-time interaction in the covariance.
#'
#' No tulpa backend fits a joint space-time covariance, so this constructor
#' errors. A separable additive model, a spatial GP plus a temporal field, is
#' fitted by `tulpa(spatial = spatial_gp(...), temporal = )` (NNGP or HSGP)
#' and by [fit_st_nested()] (`spatial_type = "hsgp"` or `"nngp"`).
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2.
#' @param time_var Name of the time variable in data.
#' @param cov_space Spatial covariance: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param cov_time Temporal covariance: `"exponential"` (default), `"matern"`,
#'   or `"gaussian"`.
#' @param nonsep_type Non-separability type:
#'   - `"product"`: \eqn{C_{st} = C_s \cdot C_t} (separable, for reference)
#'   - `"sum"`: \eqn{C_{st} = C_s + C_t}
#'   - `"gneiting"`: Gneiting (2002) non-separable class
#'   - `"cressie_huang"`: Cressie-Huang (1999) non-separable class
#' @param nn Number of nearest neighbors for NNGP approximation. Default 15.
#' @param shared Logical; if TRUE (default), effect enters both processes.
#'
#' @return Nothing: the call always signals an error.
#'
#' @details
#' The non-separable covariance functions allow for more flexible space-time
#' dependence:
#'
#' **Gneiting class:**
#' \deqn{C(h, u) = \frac{\sigma^2}{(a|u|^{2\alpha} + 1)^{\tau}} \exp\left(-\frac{c\|h\|^{2\gamma}}{(a|u|^{2\alpha} + 1)^{\beta\gamma}}\right)}
#'
#' where h is spatial lag, u is temporal lag, and parameters control the
#' space-time interaction.
#'
#' **Cressie-Huang class:**
#' Constructed via Fourier transform methods to ensure positive definiteness.
#'
#' @references
#' Gneiting, T. (2002). Nonseparable, stationary covariance functions for
#' space-time data. Journal of the American Statistical Association, 97(458), 590-600.
#'
#' Cressie, N., & Huang, H. C. (1999). Classes of nonseparable, spatio-temporal
#' stationary covariance functions. Journal of the American Statistical
#' Association, 94(448), 1330-1340.
#'
#' @examples
#' # No backend fits a joint space-time covariance, so the constructor refuses:
#' try(spatiotemporal_gp(~ lon + lat, "year", nonsep_type = "gneiting"))
#' @export
spatiotemporal_gp <- function(coords,
                              time_var,
                              cov_space = c("exponential", "matern", "gaussian", "spherical"),
                              cov_time = c("exponential", "matern", "gaussian"),
                              nonsep_type = c("product", "sum", "gneiting", "cressie_huang"),
                              nn = 15,
                              shared = NULL) {
  stop("Non-separable spatiotemporal GP fields are not fitted by any tulpa ",
       "backend: no solver carries a joint space-time covariance. The additive ",
       "model, a spatial GP plus a temporal field, is fitted by ",
       "tulpa(spatial = spatial_gp(...), temporal = ) (NNGP or HSGP) and by ",
       "fit_st_nested(spatial_type = 'hsgp' or 'nngp', ...).", call. = FALSE)
}
