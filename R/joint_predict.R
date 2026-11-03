# Prediction for a joint nested-Laplace fit (tulpa_nested_laplace_joint()) at
# new rows of one response arm.
#
# Within a cell k of the outer grid the latent posterior is the Gaussian at the
# cell's mode m_k with covariance C_k, and a new row's linear predictor is a
# linear map of the latent vector, eta = o + a_k' x: its offset, its design row
# on the arm's fixed effects, its random intercept, and every block's
# coordinates at the row's unit (or basis row) times the block's amplitude on
# the arm at the cell -- a copy block's alpha x sigma, the arm's field_coef, a
# non-centred iid block's sigma, a varying coefficient's covariate. So eta is
# N(o + a_k' m_k, a_k' C_k a_k) in cell k, and the prediction is the mixture of
# those Gaussians over the cells, weighted by the outer-grid weights.
#
# The loadings a_k are not rebuilt here: the new rows are handed to the kernel
# as a pseudo-arm of the fit's own inputs, and cpp_joint_arm_loadings() walks
# each row the way the kernel assembles eta (visit_joint_arm_row()). C_k is the
# cell's constrained inverse precision (`store_Q`, kept by default on a fit
# with a latent term); a fit without it reads the per-cell fixed-effect and
# basis-builder block it keeps, and covers only rows that load on nothing else.

.joint_predict <- function(object, newdata, arm, type, se.fit, level) {
  responses <- object$responses
  arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
  k <- .joint_predict_arm(arm, arm_names)
  nd <- .joint_predict_newdata(newdata, as.matrix(responses[[k]]$X),
                               arm_names[k])
  n_new <- nrow(nd$X)

  inp <- .joint_eta_inputs(object)
  inp <- .joint_eta_inputs_at_rows(inp, object, k, nd, arm_names[k])
  args <- .joint_eta_kernel_args(inp)

  w <- as.numeric(object$weights)
  modes <- object$modes
  on <- which(is.finite(w) & w > 0 & is.finite(rowSums(modes)))
  ld <- cpp_joint_arm_loadings(
    arms_list = args$arms_list, copy_arms = args$copy_arms,
    copy_blocks = args$copy_blocks, blocks_spec = args$blocks_spec,
    theta_grid = args$theta_grid, axis_offsets = args$axis_offsets,
    arm = k - 1L, cells = as.integer(on - 1L))
  n_x <- ncol(modes)
  if (!identical(as.integer(ld$n_x), as.integer(n_x))) {
    stop("The kernel's latent layout at the new rows (", ld$n_x, " ",
         "coordinates) is not the fit's (", n_x, ").", call. = FALSE)
  }
  new_group <- .joint_predict_new_groups(object, k, nd)
  cov_src <- if (se.fit) .joint_predict_cov_source(object, n_x)

  mu <- matrix(NA_real_, nrow(modes), n_new)
  vv <- matrix(NA_real_, nrow(modes), n_new)
  for (c in seq_along(on)) {
    j <- on[c]
    L <- ld$loadings[[c]]
    A <- Matrix::sparseMatrix(i = L$i, j = L$j, x = L$x,
                              dims = c(n_new, n_x))
    mu[j, ] <- ld$offset + as.numeric(A %*% modes[j, ])
    if (se.fit) {
      vv[j, ] <- .joint_predict_cell_var(object, cov_src, j, A) +
        new_group$var
    }
  }

  family <- responses[[k]]$family %||% "gaussian"
  phi <- responses[[k]]$phi %||% 1.0
  to_resp <- function(e) family_response_mean(e, family, phi = phi)
  if (!se.fit) {
    eta <- as.numeric(crossprod(w[on] / sum(w[on]), mu[on, , drop = FALSE]))
    return(if (type == "response") to_resp(eta) else eta)
  }
  a <- (1 - level) / 2
  ms <- .nl_gauss_mixture_summary(mu, vv, w, probs = c(a, 1 - a))
  lo <- ms$quantiles[, 1L]
  hi <- ms$quantiles[, 2L]
  fit <- ms$mean
  if (type == "response") {
    r_lo <- to_resp(lo); r_hi <- to_resp(hi)
    lo <- pmin(r_lo, r_hi); hi <- pmax(r_lo, r_hi)
    fit <- to_resp(fit)
  }
  data.frame(fit = fit, se.fit = ms$sd, lower = lo, upper = hi)
}

.joint_predict_arm <- function(arm, arm_names) {
  if (is.null(arm)) {
    if (length(arm_names) == 1L) return(1L)
    stop("A joint fit has ", length(arm_names), " response arms; name the one ",
         "to predict with `arm` (", paste(arm_names, collapse = ", "), ").",
         call. = FALSE)
  }
  k <- if (is.character(arm)) match(arm, arm_names) else as.integer(arm)
  if (length(k) != 1L || is.na(k) || k < 1L || k > length(arm_names)) {
    stop("`arm` must name one of the fit's arms (",
         paste(arm_names, collapse = ", "), ") or index it.", call. = FALSE)
  }
  k
}

# `newdata` for a joint fit: `X` (the arm's design at the new rows), and as the
# arm's latent terms need them `index` (each row's unit per block), `weight`
# (a block's per-row design weight), `coords` (for a basis builder), `basis`
# (a block given by its basis), `projector` (a projected block's rows),
# `re_idx` (each row's random-intercept group) and `offset`.
.joint_predict_newdata <- function(newdata, X_fit, arm_name) {
  if (!is.list(newdata) || is.data.frame(newdata) || is.null(newdata$X)) {
    stop("For a joint fit, `newdata` is a list with `X`, the arm's design at ",
         "the new rows (the columns of arm '", arm_name, "'s X, in its order), ",
         "and, for the latent terms reaching the arm, `index` (each row's unit ",
         "per block), `coords` (for a basis builder) and `re_idx` (each row's ",
         "random-intercept group).", call. = FALSE)
  }
  X <- as.matrix(newdata$X)
  if (ncol(X) != ncol(X_fit)) {
    stop("`newdata$X` has ", ncol(X), " columns; arm '", arm_name, "' was ",
         "fitted with ", ncol(X_fit), ".", call. = FALSE)
  }
  storage.mode(X) <- "double"
  n <- nrow(X)
  coords <- newdata$coords
  if (!is.null(coords)) {
    coords <- as.matrix(coords)
    if (ncol(coords) != 2L || nrow(coords) != n || !all(is.finite(coords))) {
      stop("`newdata$coords` must be a finite ", n, " x 2 matrix, one ",
           "row per row of `newdata$X`.", call. = FALSE)
    }
    storage.mode(coords) <- "double"
  }
  offset <- newdata$offset
  if (!is.null(offset)) {
    offset <- as.numeric(offset)
    if (length(offset) != n || !all(is.finite(offset))) {
      stop("`newdata$offset` must be ", n, " finite values.", call. = FALSE)
    }
  }
  re_idx <- newdata$re_idx
  if (!is.null(re_idx)) {
    re_idx <- as.integer(re_idx)
    if (length(re_idx) != n) {
      stop("`newdata$re_idx` must have one entry per row (", n, ").",
           call. = FALSE)
    }
  }
  list(X = X, n = n, coords = coords, offset = offset, re_idx = re_idx,
       index = newdata$index, weight = newdata$weight, basis = newdata$basis,
       projector = newdata$projector)
}

# Is there a latent term beyond the fixed effects in a joint call: a block in
# `prior`, or random intercepts on an arm?
#' @keywords internal
.joint_carries_latent <- function(prior, responses) {
  if (length(.field_block_list(prior))) return(TRUE)
  any(vapply(responses, function(a) {
    isTRUE(as.integer(a$n_re_groups %||% 0L) > 0L)
  }, logical(1)))
}

# ---- the kernel's inputs at new rows -------------------------------------------

# The inputs the fit's grid was solved with, rebuilt from what the fit keeps:
# the normalised arms, the blocks (the multi-block door's prepared blocks, or
# the single-block door's prior), the copy map and the outer grid's cells.
#' @keywords internal
.joint_eta_inputs <- function(object) {
  lay <- object$eta_layout
  if (is.null(lay)) {
    stop("This joint fit records no kernel layout ($eta_layout), so its ",
         "predictor cannot be rebuilt at new rows; refit it with this version ",
         "of tulpa.", call. = FALSE)
  }
  responses <- object$responses
  tg <- as.matrix(object$theta_grid)
  if (identical(lay$door, "multi")) {
    arms <- lapply(seq_along(responses), function(k)
      .normalise_joint_arm_multi(responses[[k]], k))
    return(list(door = "multi", arms = arms, blocks = object$blocks,
                cp = lay$cp, axis_offsets = lay$axis_offsets, theta = tg))
  }
  arms <- lapply(seq_along(responses), function(k)
    .normalise_joint_arm(responses[[k]], k))
  list(door = "single", arms = arms, prior = object$prior, type = lay$type,
       cp = lay$cp, theta = tg)
}

# The kernel arguments of `inp` (`.joint_eta_inputs()`), packed by the code the
# fit's own kernel call packs them with.
#' @keywords internal
.joint_eta_kernel_args <- function(inp) {
  if (identical(inp$door, "multi")) {
    B <- length(inp$blocks)
    return(list(
      arms_list = inp$arms,
      copy_arms = as.integer(inp$cp$copy_arms_zero),
      copy_blocks = as.integer(inp$cp$copy_blocks_zero),
      blocks_spec = .joint_blocks_spec_for_cpp(inp$blocks, length(inp$arms),
                                               arms = inp$arms),
      theta_grid = .joint_multi_cpp_grid(inp$theta, inp$axis_offsets, B,
                                         inp$cp),
      axis_offsets = as.integer(inp$axis_offsets)))
  }
  grids <- lapply(seq_len(ncol(inp$theta)), function(j) inp$theta[, j])
  names(grids) <- colnames(inp$theta)
  ki <- .joint_single_kernel_inputs(inp$type, inp$arms, inp$prior, inp$cp,
                                    grids)
  list(arms_list = inp$arms, copy_arms = as.integer(inp$cp$copy_arm_zero),
       copy_blocks = as.integer(ki$copy_block),
       blocks_spec = list(ki$block_spec), theta_grid = ki$theta_grid,
       axis_offsets = as.integer(ki$axis_offsets))
}

# `inp` with arm `k` carrying the new rows `nd` in place of its own: the arm's
# design, offset and random-intercept groups, and each block's per-arm entries
# for it. The arm's other per-row fields (response, trials, bounds) take its
# first row's values, which the predictor does not read.
#' @keywords internal
.joint_eta_inputs_at_rows <- function(inp, object, k, nd, arm_name) {
  a <- inp$arms[[k]]
  N <- length(a$y)
  n <- nd$n
  take <- rep(1L, n)
  for (nm in names(a)) {
    v <- a[[nm]]
    if (is.matrix(v) && nrow(v) == N) {
      a[[nm]] <- v[take, , drop = FALSE]
    } else if (is.atomic(v) && !is.matrix(v) && length(v) == N && N > 1L) {
      a[[nm]] <- v[take]
    }
  }
  a$X <- nd$X
  a$predict_rows <- TRUE
  if (!is.null(nd$offset) || length(a$offset)) {
    a$offset <- nd$offset %||% rep(0, n)
  }
  if (isTRUE((object$arm_layout$n_re[k] %||% 0L) > 0L)) {
    if (is.null(nd$re_idx)) {
      stop("Arm '", arm_name, "' carries random intercepts; give each new ",
           "row's group in `newdata$re_idx` (NA for a group the fit did not ",
           "see).", call. = FALSE)
    }
    r <- nd$re_idx
    r[is.na(r)] <- 0L
    a$re_idx <- as.numeric(r)
  }
  off <- .joint_arm_field_off(object, k)
  if (identical(inp$door, "single")) {
    if (length(a$spatial_idx)) {
      a$spatial_idx <- .joint_rows_index(nd, 1L, NULL, n, off, arm_name,
                                         sprintf("block 1 (%s)", inp$type))
    }
  } else {
    n_arms <- length(inp$arms)
    inp$blocks <- lapply(seq_along(inp$blocks), function(b)
      .joint_block_at_rows(inp$blocks[[b]], b, k, n_arms, N, nd, off,
                           arm_name))
  }
  inp$arms[[k]] <- a
  inp
}

# Does arm `k` carry no field (`field_coef = 0`)? Its rows then load on no
# block whatever unit they name.
#' @keywords internal
.joint_arm_field_off <- function(object, k) {
  fc <- object$responses[[k]]$field_coef
  is.numeric(fc) && length(fc) == 1L && fc == 0
}

# Each new row's unit in block `b` (`newdata$index`, by position, by `b<b>` or
# by the block's name). An arm that carries no field takes unit 1, which it
# does not load on.
#' @keywords internal
.joint_rows_index <- function(nd, b, name, n, off, arm_name, label) {
  ix <- nd$index
  v <- if (is.list(ix)) {
    nm <- names(ix)
    if (!is.null(nm) && paste0("b", b) %in% nm) ix[[paste0("b", b)]]
    else if (!is.null(nm) && !is.null(name) && name %in% nm) ix[[name]]
    else if (is.null(nm) && length(ix) >= b) ix[[b]]
  } else if (!is.null(ix) && b == 1L) ix
  if (is.null(v)) {
    if (off) return(rep(1L, n))
    stop("Arm '", arm_name, "' is reached by ", label, "; give each new ",
         "row's unit in `newdata$index` (a list with one integer vector per ",
         "block, by position, `b", b, "` or the block's name).", call. = FALSE)
  }
  v <- as.integer(v)
  if (length(v) != n || anyNA(v)) {
    stop("`newdata$index` for ", label, " must hold ", n, " units, one per ",
         "row, none missing.", call. = FALSE)
  }
  v
}

# A per-row entry of block `b` given in `newdata` (`field` one of `weight`,
# `basis`, `projector`), by position, `b<b>` or the block's name.
#' @keywords internal
.joint_rows_entry <- function(nd, field, b, name) {
  ix <- nd[[field]]
  if (!is.list(ix)) return(NULL)
  nm <- names(ix)
  if (!is.null(nm) && paste0("b", b) %in% nm) return(ix[[paste0("b", b)]])
  if (!is.null(nm) && !is.null(name) && name %in% nm) return(ix[[name]])
  if (is.null(nm) && length(ix) >= b) return(ix[[b]])
  NULL
}

# Block `p` (position `b`, prepared as the fit's kernel read it) with its arm-`k`
# entries at the new rows: the unit index (`spatial_idx` / `temporal_idx` /
# `obs_idx`), a varying coefficient's `svc_weight`, a correlated block's
# per-field `field_weight`, a projected block's `projector`, and a basis --
# rebuilt from the new coordinates on the fitted domain for a basis builder,
# given in `newdata$basis` otherwise. A per-arm entry sized to the arm's
# fitted rows that none of these covers is refused, naming it: read at the
# fitted rows, it would describe other rows.
#' @keywords internal
.joint_block_at_rows <- function(p, b, k, n_arms, N, nd, off, arm_name) {
  n <- nd$n
  label <- sprintf("block %d (%s%s)", b, p$type %||% "?",
                   if (!is.null(p$name)) paste0(" '", p$name, "'") else "")
  per_arm <- function(v) if (is.list(v)) v else rep(list(v), n_arms)
  done <- character(0)
  for (f in c("spatial_idx", "temporal_idx", "obs_idx")) {
    if (is.null(p[[f]])) next
    v <- per_arm(p[[f]])
    v[[k]] <- .joint_rows_index(nd, b, p$name, n, off, arm_name, label)
    p[[f]] <- v
    done <- c(done, f)
  }
  need <- function(field, what) {
    v <- .joint_rows_entry(nd, field, b, p$name)
    if (is.null(v)) {
      stop("Arm '", arm_name, "' is reached by ", label, ", which carries ",
           what, "; give it for the new rows in `newdata$", field, "` (by ",
           "position, `b", b, "` or the block's name).", call. = FALSE)
    }
    v
  }
  if (!is.null(p$svc_weight)) {
    v <- per_arm(p$svc_weight)
    v[[k]] <- if (off) rep(0, n) else as.numeric(need("weight", "a per-row weight"))
    p$svc_weight <- v
    done <- c(done, "svc_weight")
  }
  if (!is.null(p$field_weight)) {
    nw <- if (off) NULL else need("weight", "one per-row weight per field")
    if (!off && (!is.list(nw) || length(nw) != length(p$field_weight))) {
      stop("`newdata$weight` for ", label, " must be a list of ",
           length(p$field_weight), " per-row weights, one per field.",
           call. = FALSE)
    }
    p$field_weight <- lapply(seq_along(p$field_weight), function(a) {
      v <- per_arm(p$field_weight[[a]])
      v[[k]] <- if (off) rep(0, n) else as.numeric(nw[[a]])
      v
    })
    done <- c(done, "field_weight")
  }
  if (!is.null(p$projector)) {
    v <- per_arm(p$projector)
    v[[k]] <- if (off) Matrix::Matrix(0, n, ncol(as.matrix(v[[k]])))
              else need("projector", "a projector onto its units")
    p$projector <- v
    done <- c(done, "projector")
  }
  if (identical(tolower(p$type %||% ""), "hsgp") && is.list(p$phi)) {
    M <- ncol(as.matrix(p$phi[[k]]))
    reached <- p$basis$reached %||% rep(TRUE, n_arms)
    p$phi[[k]] <- if (off || !isTRUE(reached[k])) {
      matrix(0, n, M)
    } else if (!is.null(p$basis)) {
      if (is.null(nd$coords)) {
        stop("Arm '", arm_name, "' is reached by ", label, ", a field with a ",
             "basis builder, so `newdata` must carry `coords` (an n x 2 ",
             "matrix of the new rows' coordinates).", call. = FALSE)
      }
      cpp_hsgp_basis_2d_domain(nd$coords, p$basis$m, p$basis$domain)$phi_basis
    } else {
      as.matrix(need("basis", "a basis evaluated at the rows"))
    }
    done <- c(done, "phi")
  }
  if (!is.null(p$n_obs_per_arm)) {
    p$n_obs_per_arm[k] <- n
    done <- c(done, "n_obs_per_arm")
  }
  rows_k <- function(v) {
    if (is.matrix(v) || methods::is(v, "Matrix")) nrow(v) else length(v)
  }
  for (f in setdiff(names(p), done)) {
    v <- p[[f]]
    if (is.list(v) && length(v) == n_arms && N != n &&
        identical(rows_k(v[[k]]), as.integer(N))) {
      stop("Arm '", arm_name, "' is reached by ", label, ", whose `", f,
           "` is laid over the arm's fitted rows; predict() has no entry for ",
           "it at new rows.", call. = FALSE)
    }
  }
  p
}

# ---- uncertainty ----------------------------------------------------------------

# The rows of arm `k` whose random-intercept group the fit did not see: their
# effect is drawn from N(0, sigma_re^2), adding sigma_re^2 to the predictor's
# variance in every cell.
#' @keywords internal
.joint_predict_new_groups <- function(object, k, nd) {
  n_re <- object$arm_layout$n_re[k] %||% 0L
  if (!isTRUE(n_re > 0L) || is.null(nd$re_idx)) return(list(var = 0))
  r <- nd$re_idx
  new <- is.na(r) | r < 1L
  if (any(!new & r > n_re)) {
    stop("`newdata$re_idx` names groups beyond the arm's ", n_re, "; a group ",
         "the fit did not see is NA.", call. = FALSE)
  }
  s <- as.numeric(object$responses[[k]]$sigma_re %||% 1)
  list(var = ifelse(new, s^2, 0))
}

# Where each cell's latent covariance comes from: the stored precision under
# the inner Gaussian's constraint (`.inner_constraint()`, any loading), else the
# per-cell block over the fixed effects and basis-builder coefficients the fit
# keeps.
#' @keywords internal
.joint_predict_cov_source <- function(object, n_x) {
  if (is.list(object$Q_csc_x_per_grid) &&
      length(object$Q_csc_x_per_grid) == nrow(object$modes)) {
    return(list(kind = "Q"))
  }
  rec <- object$basis_block_cov
  list(kind = "block",
       idx = rec$idx %||% seq_len(object$n_fixed %||%
                                    sum(object$arm_layout$p)))
}

# The variance of each new row's predictor in cell `j`, diag(A C_j A'), C_j
# the cell's covariance over the coordinates A loads, read off its stored
# precision under the cell's constraint (`cpp_joint_inner_vcov_blocks()`).
#' @keywords internal
.joint_predict_cell_var <- function(object, src, j, A) {
  if (identical(src$kind, "Q")) {
    used <- sort(unique(Matrix::summary(A)$j))
    if (!length(used)) return(rep(0, nrow(A)))
    ccpp <- .inner_constraint_cpp(.inner_constraint(object, j))
    C <- cpp_joint_inner_vcov_blocks(
      Q_p_per_grid = object$Q_csc_p_per_grid[j],
      Q_i_per_grid = object$Q_csc_i_per_grid[j],
      Q_x_per_grid = object$Q_csc_x_per_grid[j],
      n_x = as.integer(ncol(A)), idx = as.integer(used),
      n_dense = length(used), A_cols_list = ccpp$A_cols_list,
      field_marginal = FALSE, n_threads = 1L,
      soft_cols_list = ccpp$soft_cols_list,
      soft_D_per_grid = ccpp$soft_D_per_grid)[[1L]]
    if (is.null(C) || !all(is.finite(C))) {
      stop("Cell ", j, " of the outer grid carries weight but its precision ",
           "could not be factored, so the prediction has no uncertainty to ",
           "report.", call. = FALSE)
    }
    As <- as.matrix(A[, used, drop = FALSE])
    return(pmax(rowSums((As %*% C) * As), 0))
  }
  used <- unique(Matrix::summary(A)$j)
  if (!all(used %in% src$idx)) {
    stop("The new rows load on latent coordinates beyond the fixed effects ",
         "and basis-builder blocks, whose covariance this fit did not keep; ",
         "refit with `control$store_Q = TRUE` (the default for a fit with a ",
         "latent term) for the prediction's uncertainty.", call. = FALSE)
  }
  Ck <- .joint_predict_cell_cov(object, j, length(src$idx))
  if (is.null(Ck)) {
    stop("Cell ", j, " of the outer grid carries weight but no latent ",
         "covariance block, so the prediction has no uncertainty to ",
         "report (", object$basis_block_cov_declined %||%
           object$grid_fixed_declined %||% "not retained", ").",
         call. = FALSE)
  }
  As <- as.matrix(A[, src$idx, drop = FALSE])
  pmax(rowSums((As %*% Ck) * As), 0)
}

# The per-cell latent covariance over `basis_block_cov$idx`, or the fixed-effect
# block a fit with no builder block retains; NULL when neither is there.
.joint_predict_cell_cov <- function(object, j, q) {
  rec <- object$basis_block_cov
  if (!is.null(rec)) {
    Ck <- rec$cov[[j]]
    return(if (is.matrix(Ck) && nrow(Ck) == q) Ck else NULL)
  }
  H <- object$grid_hessians
  if (!is.list(H) || length(H) < j || !is.matrix(H[[j]])) return(NULL)
  Ck <- tryCatch(.spd_inverse(H[[j]]), error = function(e) NULL)
  if (is.null(Ck) || nrow(Ck) != q) return(NULL)
  Ck
}
