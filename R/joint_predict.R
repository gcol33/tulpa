# Prediction for a joint nested-Laplace fit (tulpa_nested_laplace_joint()) at a
# new design for one response arm.
#
# Within a cell k of the outer grid the latent posterior is the Gaussian at the
# cell's mode m_k with covariance C_k, so a new row's linear predictor
# eta = a' x is N(a' m_k, a' C_k a) there, with `a` the row's loading on the
# latent: the arm's design row on its fixed effects and, for every block with a
# basis builder that reaches the arm, the basis evaluated at the row's
# coordinates on the fitted domain, times sqrt(S(lambda_j)) at the cell's
# (sigma2, lengthscale), times the varying-coefficient covariate when the block
# carries `svc_column`. The prediction is the mixture of those Gaussians over
# the cells, weighted by the outer-grid weights. C_k is the per-cell block over
# the fixed effects and the builder blocks' coefficients
# (`basis_block_cov`, extracted in the cell's own solve); a fit with no builder
# block reads the fixed-effect block from `grid_hessians`.

.joint_predict <- function(object, newdata, arm, type, se.fit, level) {
  responses <- object$responses
  arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
  k <- .joint_predict_arm(arm, arm_names)
  resp <- responses[[k]]
  X_fit <- as.matrix(resp$X)
  nd <- .joint_predict_newdata(newdata, X_fit, arm_names[k])
  X_new <- nd$X
  n_new <- nrow(X_new)

  lay <- object$arm_layout
  beta_idx <- lay$beta_start[k] + seq_len(ncol(X_fit))
  blocks <- object$blocks %||% .field_block_list(object$prior) %||% list()
  builder <- which(vapply(blocks, function(p) {
    !is.null(p$basis) && isTRUE(p$basis$reached[k])
  }, logical(1)))
  .joint_predict_unread(object, blocks, builder, k, arm_names[k])
  if (length(builder) && is.null(nd$coords)) {
    stop("This arm is reached by a field with a basis builder, so `newdata` ",
         "must carry `coords` (an n x 2 matrix of the new rows' coordinates).",
         call. = FALSE)
  }

  # The latent indices the loading touches and their position in the per-cell
  # covariance block.
  cov_rec <- object$basis_block_cov
  idx <- cov_rec$idx %||% seq_len(object$n_fixed %||% max(beta_idx))
  pos_beta <- match(beta_idx, idx)
  basis_new <- lapply(builder, function(b) {
    bs <- blocks[[b]]$basis
    out <- cpp_hsgp_basis_2d_domain(nd$coords, bs$m, bs$domain)
    w <- if (is.null(blocks[[b]]$svc_column)) rep(1, n_new)
         else X_new[, as.integer(blocks[[b]]$svc_column)]
    list(phi = out$phi_basis * w, lambda = out$lambda_eig,
         pos = match(lay$block_start[b] + seq_len(lay$block_size[b]), idx),
         axes = paste0("b", b, ".", c("sigma2", "lengthscale")))
  })
  if (anyNA(pos_beta) || any(vapply(basis_new, function(x) anyNA(x$pos),
                                    logical(1)))) {
    stop("The fit's per-cell covariance does not cover this arm's latent ",
         "coordinates.", call. = FALSE)
  }

  modes <- object$modes
  tg <- object$theta_grid
  w <- as.numeric(object$weights)
  n_cell <- nrow(modes)
  mu <- matrix(NA_real_, n_cell, n_new)
  vv <- matrix(NA_real_, n_cell, n_new)
  for (j in seq_len(n_cell)) {
    if (!is.finite(w[j]) || w[j] <= 0) next
    A <- matrix(0, n_new, length(idx))
    A[, pos_beta] <- X_new
    for (bn in basis_new) {
      s <- cpp_hsgp_sqrt_spectral(bn$lambda, tg[j, bn$axes[1L]],
                                  tg[j, bn$axes[2L]])
      A[, bn$pos] <- sweep(bn$phi, 2L, s, `*`)
    }
    mu[j, ] <- as.numeric(A %*% modes[j, idx])
    if (se.fit) {
      Ck <- .joint_predict_cell_cov(object, j, length(idx))
      if (is.null(Ck)) {
        stop("Cell ", j, " of the outer grid carries weight but no latent ",
             "covariance block, so the prediction has no uncertainty to ",
             "report (", object$basis_block_cov_declined %||%
               object$grid_fixed_declined %||% "not retained", ").",
             call. = FALSE)
      }
      vv[j, ] <- pmax(rowSums((A %*% Ck) * A), 0)
    }
  }

  family <- resp$family %||% "gaussian"
  phi <- resp$phi %||% 1.0
  to_resp <- function(e) family_response_mean(e, family, phi = phi)
  if (!se.fit) {
    keep <- is.finite(w) & w > 0
    eta <- as.numeric(crossprod(w[keep] / sum(w[keep]), mu[keep, , drop = FALSE]))
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

.joint_predict_newdata <- function(newdata, X_fit, arm_name) {
  if (!is.list(newdata) || is.data.frame(newdata) || is.null(newdata$X)) {
    stop("For a joint fit, `newdata` is a list with `X`, the arm's design at ",
         "the new rows (the columns of arm '", arm_name, "'s X, in its order), ",
         "and `coords` where a field with a basis builder reaches the arm.",
         call. = FALSE)
  }
  X <- as.matrix(newdata$X)
  if (ncol(X) != ncol(X_fit)) {
    stop("`newdata$X` has ", ncol(X), " columns; arm '", arm_name, "' was ",
         "fitted with ", ncol(X_fit), ".", call. = FALSE)
  }
  storage.mode(X) <- "double"
  coords <- newdata$coords
  if (!is.null(coords)) {
    coords <- as.matrix(coords)
    if (ncol(coords) != 2L || nrow(coords) != nrow(X) ||
        !all(is.finite(coords))) {
      stop("`newdata$coords` must be a finite ", nrow(X), " x 2 matrix, one ",
           "row per row of `newdata$X`.", call. = FALSE)
    }
    storage.mode(coords) <- "double"
  }
  list(X = X, coords = coords)
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

# Refuse a prediction for arm `k` that a latent term reaches without a map to
# the new rows: a block with no basis builder (its unit at a new row is not
# given; a builder block whose coordinates leave the arm out carries none of
# it), unless the arm carries no field (`field_coef = 0`), and the arm's
# own random intercepts. Held at zero, they would leave a predictor of another
# model.
.joint_predict_unread <- function(object, blocks, builder, k, arm_name) {
  fc <- object$responses[[k]]$field_coef
  off <- is.numeric(fc) && length(fc) == 1L && fc == 0
  unreached <- which(vapply(blocks, function(p) {
    !is.null(p$basis) && !isTRUE(p$basis$reached[k])
  }, logical(1)))
  held <- if (!off) setdiff(seq_along(blocks), c(builder, unreached))
          else integer(0)
  lab <- vapply(held, function(b) sprintf("block %d (%s%s)", b,
                                          blocks[[b]]$type %||% "?",
                                          if (!is.null(blocks[[b]]$name))
                                            paste0(" '", blocks[[b]]$name, "'")
                                          else ""), character(1))
  if (isTRUE((object$arm_layout$n_re[k] %||% 0L) > 0L)) {
    lab <- c(lab, "the arm's random intercepts (`re_idx`)")
  }
  if (length(lab)) {
    stop("Arm '", arm_name, "' is reached by ", paste(lab, collapse = ", "),
         ", which predict() on a tulpa_nested_laplace_joint() fit reads at new ",
         "rows only through a basis builder (an `hsgp` block given by its ",
         "coordinates); a prediction without them would be another model's.",
         call. = FALSE)
  }
  invisible(NULL)
}
