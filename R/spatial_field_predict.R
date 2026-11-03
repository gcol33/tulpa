# spatial_field_predict.R
# ------------------------------------------------------------------------------
# The linear predictor of a fit carrying a continuous spatial field (an SPDE
# Matern field, a GP / NNGP field, a Hilbert-space GP field, or a multi-scale
# field, the sum of two NNGP or two HSGP components) at any set of locations:
# the training rows or the rows of `newdata`.
#
# The fit's posterior over its latent vector x = [fixed effects, random
# effects, field] is the outer-grid mixture sum_k w_k N(m_k, Q_k^-1), one
# inner-Laplace Gaussian per hyperparameter cell. The field at a location is a
# function of the field at the fitted support and of the cell's
# hyperparameters:
#
#   * SPDE: w(s) = A(s) w_mesh, the finite-element projector at s, so the
#     field anywhere is a linear map of the mesh-node field.
#   * HSGP: f(s) = Phi(s) diag(sqrt(S_k)) z, the Laplacian basis at s on the
#     domain the fit was built on, with the spectral scales of cell k.
#   * NNGP: w(s) | w ~ N(alpha' w_nb, sigma2 - c' C^-1 c), the NNGP predictive
#     given the field at the nearest fitted locations, under cell k's
#     covariance; new locations are conditionally independent given w.
#
# One draw of eta at the locations picks a cell by its weight, draws x from
# that cell's Gaussian and maps it; the draws are therefore draws from the
# posterior predictive of eta, carrying the within-cell uncertainty of the
# coefficients and the field, the field's conditional (kriging) variance away
# from the data, and the spread over the hyperparameter cells. The point
# prediction is the exact posterior mean, the weighted mean of the same map at
# the cell modes. A multi-scale field is mapped component by component, each
# with its own block's hyperparameters, and the components summed.
# predict(), posterior_linpred() and posterior_predict() all read these two
# functions.
#
# A cell's precision is not kept on the fit: it is as large as the latent
# vector's sparse graph and a grid can carry hundreds of cells. The cells a
# set of draws lands in are solved again, with the fit's own inputs, by the
# kernel that solved them during the fit (`.field_cell_solve()`), and the
# re-solved modes are checked against the stored ones.
# ------------------------------------------------------------------------------

# The kind of map a fit's continuous field takes to a location ("spde",
# "nngp", "hsgp"), NULL for any other field. A multi-scale field takes the map
# of its components' approximation.
#' @keywords internal
.field_kind <- function(sp) {
  type <- sp$type %||% ""
  if (identical(type, .FRONTDOOR_MULTISCALE)) {
    return(if (identical(sp$approx %||% "nngp", "hsgp")) "hsgp" else "nngp")
  }
  switch(type, spde = "spde", gp = , nngp = "nngp", hsgp = "hsgp", NULL)
}

# The continuous field a fit carries and the outer cells its latent posterior
# is mixed over: `kind` ("spde", "nngp", "hsgp"), `modes` (cells x n_x),
# `weights`, `theta` (cells x axes, named), `solve` (the inner-solve record a
# cell is solved again from; NULL where the fit keeps none) and `layout`
# (`.field_layout()`). NULL for a fit without such a field, or one whose cells
# were not retained.
#' @keywords internal
.field_source <- function(object) {
  sp <- object$spatial
  kind <- .field_kind(sp)
  if (is.null(kind) || !is.null(sp$projector) || isTRUE(sp$rsr)) return(NULL)
  cells <- .field_cells(object, kind)
  if (is.null(cells)) return(NULL)
  cells$kind <- kind
  cells$layout <- .field_layout(object, kind, cells)
  if (is.null(cells$layout)) return(NULL)
  cells
}

# Where the cells sit on each door: a nested-Laplace grid on the fit itself, an
# SPDE grid in `$nested`, and a fit at fixed hyperparameters as one cell.
#' @keywords internal
.field_cells <- function(object, kind) {
  modes <- object$modes
  if (is.matrix(modes) && length(object$weights) == nrow(modes)) {
    return(list(modes = modes, weights = as.numeric(object$weights),
                theta = .nl_theta_matrix(object), solve = object$field_solve))
  }
  if (!identical(kind, "spde")) return(NULL)
  nst <- object$nested
  if (is.list(nst) && is.matrix(nst$modes) &&
      length(nst$weights) == nrow(nst$modes)) {
    return(list(modes = nst$modes, weights = as.numeric(nst$weights),
                theta = nst$theta_grid, solve = object$field_solve))
  }
  if (is.numeric(object$mode) && is.numeric(object$range) &&
      is.numeric(object$sigma)) {
    return(list(modes = matrix(object$mode, nrow = 1L), weights = 1,
                theta = cbind(range = object$range, sigma = object$sigma),
                solve = object$field_solve))
  }
  NULL
}

# The latent columns of each component: `beta` (the fixed effects), `fields`
# (one entry per field component: its `cols`, the component at its fitted
# support, its prior `block`, and the `prefix` its hyperparameter axes carry on
# the outer grid; a multi-scale field has two), and `re` (the formula random
# effects, with
# `scale(k)` the factor turning cell k's coordinates into effects: the cell's
# random-effect SD on a multi-block grid, whose iid block is non-centred, and 1
# where the coordinates are the effects; `sd_per_cell` marks the first, where a
# new group's effect is drawn at the row's cell SD). NULL when the columns
# cannot be located.
#' @keywords internal
.field_layout <- function(object, kind, cells) {
  n_x <- ncol(cells$modes)
  p <- as.integer(object$n_fixed %||% ncol(object$model_matrix))
  sp <- object$spatial
  width <- switch(kind,
    spde = as.integer(sp$n_mesh),
    nngp = as.integer(sp$n_unique %||% nrow(sp$unique_coords)),
    hsgp = as.integer(sp[["m"]])^2L)
  prior <- object$prior
  if (is.list(prior) && is.null(prior$type) && length(prior)) {
    roles <- .nl_block_roles(prior)
    fields <- lapply(which(roles == "spatial"), function(k) {
      list(cols = .nl_block_cols(object$block_latent_offsets, n_x, k,
                                 length(roles)),
           block = prior[[k]], prefix = paste0("b", k, "."))
    })
    rl <- .nl_re_field_load(prior, object$block_latent_offsets, n_x,
                            cells$theta)
    re <- if (!is.null(rl)) {
      list(cols = rl$cols[[1L]], sd_per_cell = TRUE,
           scale = function(k) as.numeric(rl$weights(k)[[1L]]))
    }
  } else {
    fields <- if (width <= n_x - p) {
      list(list(cols = seq.int(n_x - width + 1L, n_x), block = prior,
                prefix = ""))
    }
    n_re <- n_x - width - p
    re <- if (n_re > 0L) {
      list(cols = p + seq_len(n_re), sd_per_cell = FALSE, scale = function(k) 1)
    }
  }
  if (!length(fields) ||
      any(vapply(fields, function(f) length(f$cols) != width, logical(1)))) {
    return(NULL)
  }
  for (j in seq_along(fields)) fields[[j]]$cols <- as.integer(fields[[j]]$cols)
  list(beta = seq_len(p), fields = fields, re = re)
}

# A field hyperparameter of every cell, by its bare axis name and the prefix
# its component's block carries on the outer grid (`""` on a single-block
# grid).
#' @keywords internal
.field_theta <- function(theta, axis, prefix = "") {
  j <- which(colnames(theta) == paste0(prefix, axis))
  if (length(j) != 1L) {
    stop("The fit's outer grid carries no single `", axis, "` axis for its ",
         "spatial field.", call. = FALSE)
  }
  as.numeric(theta[, j])
}

# The coordinates of the rows to predict at, on the scale the field was fitted
# on: the training coordinates when `newdata` is NULL, else the field's own
# coordinate columns of `newdata`, standardized as the training ones were.
#' @keywords internal
.field_coords_at <- function(object, newdata) {
  sp <- object$spatial
  cm <- sp$coords_matrix
  if (is.null(newdata)) return(.coords_plain(cm))
  cv <- sp$coord_vars
  miss <- setdiff(cv, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s) the spatial field was ",
         "fitted on: ", paste(miss, collapse = ", "), call. = FALSE)
  }
  nc <- as.matrix(newdata[, cv, drop = FALSE])
  if (isTRUE(sp$scale_coords)) {
    ctr <- attr(cm, "scaled:center")
    scl <- attr(cm, "scaled:scale")
    if (!is.null(ctr) && !is.null(scl)) {
      nc <- sweep(sweep(nc, 2, ctr, "-"), 2, scl, "/")
    }
  }
  .coords_plain(nc)
}

# The field component `comp` (an entry of the layout's `fields`) as R x n rows
# at the prediction rows, for its rows `Fr` (R x width, the component at its
# fitted support) drawn in cells `cells` (indices into the fit's grid).
# `draw = TRUE` adds the NNGP conditional deviation; `FALSE` gives the
# conditional mean, which for the SPDE and HSGP maps is the map itself.
#' @keywords internal
.field_at <- function(object, src, newdata, Fr, cells, draw, comp) {
  sp <- object$spatial
  switch(src$kind,
    spde = {
      A <- if (is.null(newdata)) sp$A else .spde_A_at(object, newdata)
      as.matrix(Fr %*% Matrix::t(A))
    },
    hsgp = {
      ct <- .coords_2col(.coords_plain(sp$coords_matrix), "hsgp() prediction")
      cn <- .coords_2col(.field_coords_at(object, newdata), "hsgp() prediction")
      cc <- as.numeric(if (identical(sp$type, .FRONTDOOR_MULTISCALE))
                         sp$c_boundary else sp[["c"]])
      dom <- cpp_hsgp_domain_2d(ct, cc)
      out <- abs(cn[, 1L] - dom[1L]) > dom[3L] | abs(cn[, 2L] - dom[2L]) > dom[4L]
      if (any(out)) {
        stop(sum(out), " of ", nrow(cn), " prediction coordinate(s) lie ",
             "outside the HSGP domain (the fitted extent widened by c = ", cc,
             "), where the basis approximates no Gaussian process. Refit with ",
             "a larger `c` in spatial_gp(approx = \"hsgp\"), or use the NNGP ",
             "field.", call. = FALSE)
      }
      basis <- cpp_hsgp_basis_2d_domain(cn, as.integer(sp[["m"]]), dom)
      s2 <- .field_theta(src$theta, "sigma2", comp$prefix)
      ls <- .field_theta(src$theta, "lengthscale", comp$prefix)
      u <- unique(cells)
      sq <- do.call(rbind, lapply(u, function(k) as.numeric(
        cpp_hsgp_sqrt_spectral(basis$lambda_eig, s2[k], ls[k]))))
      (Fr * sq[match(cells, u), , drop = FALSE]) %*% t(basis$phi_basis)
    },
    nngp = {
      if (is.null(newdata)) {
        otl <- sp$obs_to_loc
        if (is.null(otl)) {
          stop("The fit records no map from its observations to the field's ",
               "locations ($spatial$obs_to_loc).", call. = FALSE)
        }
        return(Fr[, otl, drop = FALSE])
      }
      blk <- comp$block
      nn  <- as.integer(blk$nn %||% sp$nn %||% 10L)
      cty <- as.integer(blk$cov_type %||% gp_cov_type(sp))
      nc <- .field_coords_at(object, newdata)
      u <- unique(cells)
      noise <- if (draw) matrix(stats::rnorm(nrow(Fr) * nrow(nc)), nrow(Fr))
               else matrix(0, nrow(Fr), nrow(nc))
      cpp_gp_field_rows(nc, .coords_plain(sp$unique_coords), Fr,
                        as.integer(match(cells, u) - 1L),
                        .field_theta(src$theta, "sigma2", comp$prefix)[u],
                        .field_theta(src$theta, "phi_gp", comp$prefix)[u],
                        nn, cty, noise)
    })
}

# The linear predictor (R x n) at the prediction rows for latent rows `L`
# (R x n_x) drawn in cells `cells`: the fixed effects and the offset of design
# `D`, the field (`.field_at()`, summed over its components), and the fitted
# groups' random effects of the terms `sel` keeps on term design `des`. The
# random-effect coefficient rows travel as the `"re"` attribute.
#' @keywords internal
.field_eta_rows <- function(object, src, D, newdata, L, cells, draw, des,
                            sel = NULL) {
  lay <- src$layout
  beta <- L[, lay$beta, drop = FALSE]
  nm <- object$fixed_names
  X <- D$X
  if (!is.null(nm) && length(nm) == ncol(beta) && all(nm %in% colnames(X))) {
    X <- X[, nm, drop = FALSE]
  } else if (ncol(X) != ncol(beta)) {
    stop("The prediction design does not match the fit's fixed effects.",
         call. = FALSE)
  }
  eta <- beta %*% t(X)
  if (any(D$offset != 0)) eta <- sweep(eta, 2, rep_len(D$offset, ncol(eta)), "+")
  for (comp in lay$fields) {
    eta <- eta + .field_at(object, src, newdata, L[, comp$cols, drop = FALSE],
                           cells, draw, comp)
  }
  B <- NULL
  if (!is.null(lay$re)) {
    nr <- length(lay$re$cols)
    sc <- vapply(cells, function(k) rep_len(lay$re$scale(k), nr), numeric(nr))
    B <- L[, lay$re$cols, drop = FALSE] * t(matrix(sc, nr))
    if (!is.null(sel) && any(sel)) {
      re_eta <- .tulpa_re_eta(object, des, B, sel)
      if (!is.null(re_eta)) eta <- eta + re_eta
    }
  }
  dimnames(eta) <- NULL
  attr(eta, "re") <- B
  eta
}

# The posterior mean of the linear predictor at the prediction rows: the
# weighted mean of `.field_eta_rows()` at every cell's mode, a group the fit
# did not see at its prior mean, zero.
#' @keywords internal
.field_point_eta <- function(object, src, newdata, sel = NULL,
                             accessor = "predict") {
  w <- src$weights
  on <- is.finite(w) & w > 0 & is.finite(rowSums(src$modes))
  if (!any(on)) {
    .accessor_unavailable(accessor, object, "an outer-grid cell carrying weight")
  }
  cells <- which(on)
  D <- .tulpa_designs(object, newdata, accessor)
  des <- if (is.null(newdata)) object$re_design
         else .tulpa_re_design_at(object, newdata)
  E <- .field_eta_rows(object, src, D, newdata,
                       src$modes[cells, , drop = FALSE], cells, draw = FALSE,
                       des = des, sel = sel)
  as.numeric(colSums(w[cells] * E) / sum(w[cells]))
}

# `S` draws of the latent vector from the outer-grid mixture: the rows are
# allocated over the cells by their weights (`.nl_mixture_cells()`), the cells
# drawn in are solved again for their precision, and each row is drawn from
# its cell's Gaussian N(m_k, Q_k^-1), conditioned on the cell's sum-to-zero
# groups. Returns the S x n_x rows with the cell of each row as `"cells"`.
#' @keywords internal
.field_latent_draws <- function(object, src, S) {
  w <- src$weights
  on <- is.finite(w) & w > 0 & is.finite(rowSums(src$modes))
  al <- .nl_mixture_cells(w[on] / sum(w[on]), which(on), S)
  need <- which(on)[al$counts > 0L]
  sol <- .field_cell_solve(object, src, need)
  n_x <- ncol(src$modes)
  dev <- max(abs(sol$modes - src$modes[need, , drop = FALSE]))
  if (!is.finite(dev) || dev > 1e-3 * (1 + max(abs(src$modes[need, ])))) {
    stop("Solving the fit's outer cells again did not reproduce their stored ",
         "modes (largest difference ", signif(dev, 3), "), so the precision ",
         "it returned is not the fit's.", call. = FALSE)
  }
  L <- matrix(0, S, n_x)
  row_cells <- integer(S)
  pos <- 0L
  for (j in seq_along(need)) {
    n_j <- al$counts[match(need[j], which(on))]
    rows <- pos + seq_len(n_j)
    Z <- .joint_draw_cell(sol$Q_p[[j]], sol$Q_i[[j]], sol$Q_x[[j]], n_x,
                          sol$constraint_cols, n_j)
    L[rows, ] <- t(Z + src$modes[need[j], ])
    row_cells[rows] <- need[j]
    pos <- pos + n_j
  }
  attr(L, "cells") <- row_cells
  L
}

# The precision (lower-triangle CSC, one list entry per cell) and mode of the
# fit's cells `cells`, solved again with the inputs the fit was solved with by
# the kernel that solved it: the SPDE grid kernel, the registry kernel of a
# single block, or the multi-block driver. `constraint_cols` are the
# sum-to-zero groups the precision is conditioned on.
#' @keywords internal
.field_cell_solve <- function(object, src, cells) {
  rec <- src$solve
  if (is.null(rec)) {
    stop("This fit keeps no record of its inner solve, so its latent ",
         "posterior cannot be drawn at new locations (a fractional-nu SPDE ",
         "field, or a fit through a model-supplied likelihood). The point ",
         "prediction, predict(se.fit = FALSE), is available.", call. = FALSE)
  }
  theta <- src$theta[cells, , drop = FALSE]
  res <- switch(rec$door,
    spde = .spde_grid_solve(
      object$spatial, y = as.numeric(object$y),
      n_trials = as.integer(object$n_trials), X = object$model_matrix,
      family = object$family, phi = object$phi, offset = object$offset,
      solve = rec$args, range_vec = as.numeric(theta[, "range"]),
      sigma_vec = as.numeric(theta[, "sigma"]), store_Q = TRUE,
      extra = list(compute_fitted_var = FALSE)),
    registry = ,
    multi = {
      a <- rec$args
      a$y <- as.numeric(object$y)
      a$X <- object$model_matrix
      a$store_Q <- TRUE
      a$prune_tol <- 0
      a$compute_fitted_var <- FALSE
      if (identical(rec$door, "multi")) {
        .nl_dispatch_multi(a, object$prior, theta_grid_override = theta)
      } else {
        blk <- .nl_registry_write_theta(list(object$prior), theta,
                                        colnames(theta))[[1L]]
        .nl_dispatch(rec$type, a, blk, object$prior)
      }
    })
  if (!is.list(res$Q_csc_x_per_grid) ||
      length(res$Q_csc_x_per_grid) != length(cells)) {
    stop("The inner solve returned no precision for the fit's cells.",
         call. = FALSE)
  }
  list(Q_p = res$Q_csc_p_per_grid, Q_i = res$Q_csc_i_per_grid,
       Q_x = res$Q_csc_x_per_grid,
       modes = matrix(res$modes, nrow = length(cells)),
       constraint_cols = res$constraint_cols)
}

# Linear-predictor draws (S x n) at the prediction rows of a continuous-field
# fit, with the random effects of the terms `sel` keeps: a fitted group's from
# the same latent draw, a group the fit did not see from its covariance at the
# row's cell. The cell of each row travels as `"cells"`.
#' @keywords internal
.field_eta_draws <- function(object, src, newdata, S, sel = NULL) {
  L <- .field_latent_draws(object, src, S)
  cells <- attr(L, "cells")
  D <- .tulpa_designs(object, newdata, "posterior_predict")
  des <- if (is.null(newdata)) object$re_design
         else .tulpa_re_design_at(object, newdata)
  eta <- .field_eta_rows(object, src, D, newdata, L, cells, draw = TRUE,
                         des = des, sel = sel)
  attr(eta, "re") <- NULL
  if (!is.null(newdata) && !is.null(sel) && any(sel) && length(des)) {
    sig <- if (isTRUE(src$layout$re$sd_per_cell) && length(des) == 1L) {
      list(lapply(cells, function(k) matrix(src$layout$re$scale(k)[1L]^2)))
    }
    eta <- eta + .re_new_group_eta(object, des, NULL, S, sel, sig = sig)
  }
  attr(eta, "cells") <- cells
  eta
}
