# posterior_predict.R
# ------------------------------------------------------------------------------
# Posterior-predictive replicates for tulpa_fit objects: draw the in-sample
# linear predictor (every component the fit estimated), push it through the
# family's sampling function (.FAMILY_OPS$sample), and return a draws x n_obs
# matrix. simulate.tulpa_fit() is the base-R alias;
# pp_check() falls back to this when the fit carries no stored y_rep.
# ------------------------------------------------------------------------------

# Sparse map from the random-effect coefficient vector (group-major,
# coef-within-group -- the layout .tulpa_param_layout names and the draws tail
# uses) to the per-observation RE contribution: eta_re = b' M with M (n_re x
# n_obs). Built from a term design: the $re_design rows tulpa() attaches (the
# training rows), or `.tulpa_re_design_at()` at new data, where a row whose
# group the fit did not see has no fitted coefficient and gets no entry. NULL
# when the fit carries no formula RE terms or no design.
#' @keywords internal
.tulpa_re_map <- function(object, des = object$re_design) {
  if (is.null(des) || !length(des)) return(NULL)
  if (identical(des, object$re_design) && is.null(object$model_matrix)) {
    return(NULL)
  }
  n_obs <- length(des[[1L]]$group_idx)

  i_idx <- integer(0); j_idx <- integer(0); x_val <- numeric(0)
  offset <- 0L
  for (rt in des) {
    nc <- rt$n_coefs %||% 1L
    Z  <- .re_term_Z(rt, n_obs)
    if (is.null(Z) || ncol(Z) != nc) return(NULL)
    seen <- which(!is.na(rt$group_idx))
    for (cc in seq_len(nc)) {
      i_idx <- c(i_idx, offset + (rt$group_idx[seen] - 1L) * nc + cc)
      j_idx <- c(j_idx, seen)
      x_val <- c(x_val, Z[seen, cc])
    }
    offset <- offset + (rt$n_groups %||% max(rt$group_idx, na.rm = TRUE)) * nc
  }
  Matrix::sparseMatrix(i = i_idx, j = j_idx, x = x_val,
                       dims = c(offset, n_obs))
}

# A term's per-observation design (n_obs x n_coefs): the intercept column and
# the slope columns, in the order its coefficients are laid out.
#' @keywords internal
.re_term_Z <- function(rt, n_obs) {
  cbind(if (isTRUE(rt$has_intercept)) rep(1, n_obs), rt$slope_matrix)
}

# The rows of the stacked RE coefficient vector (`.tulpa_re_map()` layout) that
# belong to each term, as a list of integer vectors.
#' @keywords internal
.re_term_rows <- function(des) {
  pos <- 0L
  lapply(des, function(rt) {
    k <- (rt$n_groups %||% max(rt$group_idx, na.rm = TRUE)) * (rt$n_coefs %||% 1L)
    r <- pos + seq_len(k)
    pos <<- pos + k
    r
  })
}

# The random-effect term designs at `newdata`, rebuilt from the fit's formula
# with `.re_term_design()` against the groups and slope factor levels the fit
# recorded: a group label the fit saw gets its fitted index, any other gets NA
# (a new group). NULL when the fit has no formula random effects.
#' @keywords internal
.tulpa_re_design_at <- function(object, newdata) {
  des <- object$re_design
  if (is.null(des) || !length(des)) return(NULL)
  if (is.null(object$formula)) {
    stop("predict(): the fit carries random-effect terms but no model formula ",
         "to rebuild their design at `newdata` from.", call. = FALSE)
  }
  specs <- tulpa_parse_formula(object$formula)$random_effects
  if (length(specs) != length(des) || any(vapply(des, function(rt)
        is.null(rt$levels), logical(1)))) {
    stop("predict(): the fit does not record the groups of its random-effect ",
         "terms, so `newdata` cannot be matched to them; refit it with this ",
         "version of tulpa, or use re_form = NA for the population prediction.",
         call. = FALSE)
  }
  env <- environment(object$formula) %||% parent.frame()
  lapply(seq_along(des), function(m) {
    rt <- des[[m]]
    miss <- setdiff(specs[[m]]$group_vars, names(newdata))
    if (length(miss)) {
      stop(sprintf(paste0(
        "`newdata` has no column %s for the random-effect term (... | %s). ",
        "Add it to predict for those groups, or pass re_form = NA for the ",
        "population prediction."),
        paste(shQuote(miss), collapse = ", "), rt$group_var), call. = FALSE)
    }
    td <- .re_term_design(specs[[m]], newdata, env, levels = rt$levels,
                          xlevels = rt$slope_xlevels %||% list())
    if (!identical(td$slope_names, rt$slope_names %||% character(0))) {
      stop(sprintf(paste0(
        "predict(): `newdata` gives the random-slope columns %s for the term ",
        "(... | %s); the fit has %s."),
        paste(td$slope_names, collapse = ", "), rt$group_var,
        paste(rt$slope_names, collapse = ", ")), call. = FALSE)
    }
    c(td[c("group_idx", "group_label", "slope_matrix", "slope_names")],
      rt[c("group_var", "has_intercept", "n_groups", "n_coefs", "levels")])
  })
}

# Which formula random-effect terms a prediction conditions on, from `re_form`
# (lme4's `re.form`): NULL all of them, NA or `~ 0` none (the population
# level), a one-sided formula of bar terms those terms. Returns a logical vector
# over the fit's terms, or NULL when the fit has none.
#' @keywords internal
.re_form_select <- function(object, re_form = NULL) {
  des <- object$re_design
  if (is.null(des) || !length(des)) return(NULL)
  n <- length(des)
  if (is.null(re_form)) return(rep(TRUE, n))
  if (length(re_form) == 1L && !inherits(re_form, "formula") && is.na(re_form)) {
    return(rep(FALSE, n))
  }
  if (!inherits(re_form, "formula")) {
    stop("`re_form` must be NULL (every random-effect term), NA or ~0 ",
         "(none: the population level), or a one-sided formula of ",
         "random-effect terms.", call. = FALSE)
  }
  bars <- findbars(re_form[[length(re_form)]])
  if (!length(bars)) return(rep(FALSE, n))
  want <- unlist(lapply(bars, function(b)
    vapply(parse_bar_term(b), `[[`, character(1), "group_var")))
  have <- vapply(des, function(rt) rt$group_var %||% NA_character_, character(1))
  unknown <- setdiff(want, have)
  if (length(unknown)) {
    stop(sprintf("`re_form` names the grouping %s, which the fit has no term for (it has %s).",
                 paste(shQuote(unknown), collapse = ", "),
                 paste(shQuote(have), collapse = ", ")), call. = FALSE)
  }
  have %in% want
}

# The random-effect part of the linear predictor (rows x n_obs) on a term
# design `des`: `re` the coefficient rows (rows x n_re) on the fitted groups,
# the terms `sel` leaves out held at zero. A row whose group the fit did not see
# gets zero here; its effect is drawn by `.re_new_group_eta()`.
#' @keywords internal
.tulpa_re_eta <- function(object, des, re, sel = NULL) {
  M <- .tulpa_re_map(object, des)
  if (is.null(M) || is.null(re)) return(NULL)
  if (!is.null(sel) && !all(sel)) {
    rows <- unlist(.re_term_rows(des)[!sel])
    re[, rows] <- 0
  }
  as.matrix(re %*% M)
}

# The random-effect covariance of each term at each of `S` rows, as a list over
# terms of a list of S matrices. Where the fit carries the covariance per draw
# the rows read it at the draw they belong to (`keep`): a Gibbs fit's sampled
# Sigma, a nested fit's integration node, a sampler's `log_sigma_re` / `L_re`
# columns. Any other fit carries one covariance -- an estimate, or the value it
# conditioned on -- and every row reads that one ([VarCorr()]'s).
#' @keywords internal
.re_sigma_rows <- function(object, keep, S) {
  des <- object$re_design
  nc <- vapply(des, function(rt) as.integer(rt$n_coefs %||% 1L), integer(1))
  M <- length(des)
  per_row <- NULL

  sd <- object[["Sigma_draws"]]
  if (is.list(sd) && length(sd)) {
    rows <- keep %||% seq_len(length(sd))
    per_row <- lapply(sd[rows], function(s) if (is.matrix(s)) list(s) else s)
  } else if (!is.null(object$draw_nodes) && is.matrix(object$theta_grid) &&
             is.list(object$layout)) {
    picks <- object$draw_nodes[keep %||% seq_along(object$draw_nodes)]
    node_sig <- lapply(seq_len(nrow(object$theta_grid)), function(k)
      lapply(.re_cov_theta_to_L_list(object$theta_grid[k, ], object$layout),
             function(L) L %*% t(L)))
    per_row <- node_sig[picks]
  } else if (is.list(object$model_inputs)) {
    per_row <- .sampler_re_sigma_rows(object, keep, nc)
  }
  if (!is.null(per_row) && length(per_row) == S &&
      all(vapply(per_row, length, integer(1)) == M)) {
    return(lapply(seq_len(M), function(m) lapply(per_row, `[[`, m)))
  }
  vc <- .varcorr_cov_list(object)$cov_list
  if (is.null(vc) || length(vc) != M) {
    stop("posterior_predict(): a new group's effect needs the random-effect ",
         "covariance, which this fit does not report.", call. = FALSE)
  }
  lapply(vc, function(Sm) rep(list(as.matrix(Sm)), S))
}

# Per-draw covariances of a ModelData sampler fit's terms, from its
# `log_sigma_re[...]` and (correlated terms) `L_re[...]` columns, in the layout
# `.re_noncentered_effects()` reads. NULL when a column the map needs is absent.
#' @keywords internal
.sampler_re_sigma_rows <- function(object, keep, nc) {
  dm <- tryCatch(.re_draws_mat(object), error = function(e) NULL)
  if (is.null(dm) || is.null(colnames(dm))) return(NULL)
  if (!is.null(keep)) dm <- dm[keep, , drop = FALSE]
  nm <- colnames(dm)
  M <- length(nc)
  slopes <- any(nc > 1L)
  sig_cols <- lapply(seq_len(M), function(m) {
    if (slopes) sprintf("log_sigma_re[t%d.c%d]", m, seq_len(nc[m]))
    else if (M > 1L) sprintf("log_sigma_re[t%d]", m) else "log_sigma_re"
  })
  if (!all(unlist(sig_cols) %in% nm)) return(NULL)
  L_terms <- lapply(seq_len(M), function(m) {
    cols <- grep(sprintf("^L_re\\[t%d\\.", m), nm)
    if (nc[m] > 1L && length(cols)) .re_chol_from_raw(dm[, cols, drop = FALSE], nc[m])
  })
  lapply(seq_len(nrow(dm)), function(s) lapply(seq_len(M), function(m) {
    sg <- exp(dm[s, sig_cols[[m]]])
    R <- if (is.null(L_terms[[m]])) diag(nc[m]) else {
      L <- matrix(L_terms[[m]][s, , ], nc[m], nc[m]); L %*% t(L)
    }
    diag(sg, nrow = nc[m]) %*% R %*% diag(sg, nrow = nc[m])
  }))
}

# The effect of each group the fit did not see, drawn per row from that row's
# covariance (`sig`, by default `.re_sigma_rows()`: a list over terms of the S
# per-row matrices), on the linear predictor (S x n_obs) of the term design
# `des`. Rows of one new group share its draw. Zero where every group was
# fitted, and for the terms `sel` leaves out.
#' @keywords internal
.re_new_group_eta <- function(object, des, keep, S, sel = NULL, sig = NULL) {
  n_obs <- length(des[[1L]]$group_idx)
  out <- matrix(0, S, n_obs)
  fresh <- vapply(seq_along(des), function(m)
    (is.null(sel) || sel[m]) && anyNA(des[[m]]$group_idx), logical(1))
  if (!any(fresh)) return(out)
  if (is.null(sig)) sig <- .re_sigma_rows(object, keep, S)
  for (m in which(fresh)) {
    rt <- des[[m]]
    nc <- rt$n_coefs %||% 1L
    Z  <- .re_term_Z(rt, n_obs)
    rows <- which(is.na(rt$group_idx))
    lab  <- factor(rt$group_label[rows])
    G    <- nlevels(lab)
    L <- NULL
    for (s in seq_len(S)) {
      if (is.null(L) || !identical(sig[[m]][[s]], sig[[m]][[s - 1L]])) {
        L <- .re_chol_spd(sig[[m]][[s]])                    # lower, L L' = Sigma
      }
      b <- matrix(stats::rnorm(G * nc), G, nc) %*% t(L)    # G x nc
      out[s, rows] <- out[s, rows] +
        rowSums(Z[rows, , drop = FALSE] * b[as.integer(lab), , drop = FALSE])
    }
  }
  out
}

# Posterior-mean estimate of the RE coefficient vector (length n_re), in the
# group-major / coef-within-group layout of `.tulpa_re_map()`: the mean of the
# fit's own RE draws; on an RE-covariance integrator fit the weighted mean of
# the per-node conditional means `re_nodes` (the exact mean of the mixture
# `ranef()` reports); else the tail of the Laplace `$mode` or the `$means`
# vector past the fixed block. NULL when none carries the RE block.
#' @keywords internal
.tulpa_re_point <- function(object, n_re) {
  rd <- .re_coef_draws(object)
  if (!is.null(rd) && ncol(rd) == n_re) return(as.numeric(colMeans(rd)))
  rn <- object$re_nodes
  if (is.matrix(rn) && ncol(rn) == n_re && !is.null(object$weights) &&
      length(object$weights) == nrow(rn)) {
    w <- as.numeric(object$weights)
    ok <- is.finite(w) & w > 0 & is.finite(rowSums(rn))
    if (any(ok)) {
      return(as.numeric(colSums(w[ok] * rn[ok, , drop = FALSE]) / sum(w[ok])))
    }
  }
  nf <- object$n_fixed %||% 0L
  for (v in list(object$mode, object$means)) {
    if (is.numeric(v) && length(v) >= nf + n_re) {
      return(as.numeric(v)[(nf + 1L):(nf + n_re)])
    }
  }
  NULL
}

# Random-effect draws (S x n_re) for a Laplace / EB fit, from the joint
# Gaussian of the latent vector [beta | b] at the mode, conditional on the
# fixed-effect draws `beta` (S x p) it is paired with. NULL when the fit
# carries no joint precision (`H_latent`) laid out as [fixed, n_re].
#
# The fixed effects are drawn from N(coef, vcov) upstream and the random
# effects used to be held at their mode. That kept the marginal fixed-effect
# spread -- which includes the intercept's aliasing with the random effects --
# and dropped the negative posterior correlation that cancels it, so every draw
# shifted the whole linear predictor: p_waic / p_loo inflated about 2.5x and a
# Laplace fit ranked ~17 elpd below the same model fitted by HMC
# (gcol33/tulpa#871). Under precision H with blocks B = H[beta, b] and
# D = H[b, b], b | beta ~ N(b_hat - D^-1 B' (beta - beta_hat), D^-1), so this
# draw paired with N(beta_hat, Schur(H)^-1) IS the joint draw, and paired with
# a corrected marginal (EB's `cov_marginal`) it keeps that correction on beta.
#' @keywords internal
.laplace_re_conditional_draws <- function(object, beta, n_re) {
  .gaussian_re_conditional_draws(object$H_latent, object$mode,
                                 object$n_fixed %||% 0L, beta, n_re)
}

# The conditional draw of `.laplace_re_conditional_draws()` for one Gaussian
# latent posterior N(mode, H^-1) over [fixed (nf) | random effects (n_re)].
#' @keywords internal
.gaussian_re_conditional_draws <- function(H, mode, nf, beta, n_re) {
  if (is.null(H) || !is.numeric(mode) || nf < 1L || n_re < 1L ||
      !is.matrix(beta) || ncol(beta) != nf ||
      nrow(H) != nf + n_re || length(mode) != nrow(H)) {
    return(NULL)
  }
  i_f <- seq_len(nf)
  i_r <- nf + seq_len(n_re)
  D <- Matrix::forceSymmetric(H[i_r, i_r, drop = FALSE])
  ch <- tryCatch(Matrix::Cholesky(D, perm = TRUE, LDL = FALSE),
                 error = function(e) NULL)
  if (is.null(ch)) return(NULL)
  S <- nrow(beta)
  dbeta <- sweep(beta, 2L, mode[i_f], "-")                  # S x nf
  shift <- Matrix::solve(ch, Matrix::crossprod(H[i_f, i_r, drop = FALSE],
                                               t(dbeta)))   # n_re x S
  # D = P' L L' P, so P' L'^-1 z has covariance D^-1.
  z <- matrix(stats::rnorm(n_re * S), n_re, S)
  noise <- Matrix::solve(ch, Matrix::solve(ch, z, system = "Lt"),
                         system = "Pt")
  b <- mode[i_r] - as.matrix(shift) + as.matrix(noise)
  t(b)
}

# Random-effect draws (S x n_re) on a node mixture: a nested fit over the
# random-effect covariance (`tulpa_re_cov_nested()`) draws each fixed-effect row
# from one integration node (`draw_nodes`, aligned with the rows of `$draws`),
# and that node's inner Laplace solve is a joint Gaussian over [beta | b]
# (`latent_nodes[[k]]`: its mode and observed-curvature precision). Each row's
# random effects are drawn from the Gaussian conditional given its beta at the
# node it came from, so the pair is a draw from that node's joint posterior and
# the rows together sample the mixture over Sigma. `keep` indexes the stored
# rows `beta` was taken from. NULL when the fit carries no node mixture, or a
# picked node carries no precision.
#' @keywords internal
.node_re_conditional_draws <- function(object, beta, keep, n_re) {
  nodes <- object$latent_nodes
  picks <- object$draw_nodes
  if (!is.list(nodes) || is.null(picks) || is.null(keep)) return(NULL)
  picks <- picks[keep]
  if (length(picks) != nrow(beta) || anyNA(picks)) return(NULL)
  out <- matrix(NA_real_, nrow(beta), n_re)
  for (k in unique(picks)) {
    rows <- which(picks == k)
    nd <- nodes[[k]]
    if (!is.list(nd)) return(NULL)
    b <- .gaussian_re_conditional_draws(nd$H, nd$mode, length(nd$mode) - n_re,
                                        beta[rows, , drop = FALSE], n_re)
    if (is.null(b)) return(NULL)
    out[rows, ] <- b
  }
  out
}

# Random-effect coefficient draws (S x n_re, group-major / coef-within-group)
# paired row for row with the fixed-effect draws `beta` that
# `.fixed_coef_draws()` returned alongside `keep`. The one resolution order
# posterior_predict() and tulpa_simulate(theta = fit) share: the fit's own RE
# draws (rows `keep`), else the conditional draw at each row's integration node
# on a node-mixture fit, else a Laplace / EB conditional draw given `beta`, else
# the point value repeated on every row. NULL when the fit carries none of the
# four, so a caller decides what a missing RE block means rather than reading
# it as zero (gcol33/tulpa#891).
#' @keywords internal
.tulpa_re_coef_draws <- function(object, beta, keep, n_re) {
  rd <- .re_coef_draws(object)
  if (!is.null(rd) && ncol(rd) == n_re) {
    if (!is.null(keep)) rd <- rd[keep, , drop = FALSE]
    return(rd)
  }
  rd <- .node_re_conditional_draws(object, beta, keep, n_re)
  if (!is.null(rd)) return(rd)
  rd <- .laplace_re_conditional_draws(object, beta, n_re)
  if (!is.null(rd)) return(rd)
  b <- .tulpa_re_point(object, n_re)
  if (is.null(b)) return(NULL)
  matrix(b, nrow(beta), n_re, byrow = TRUE)
}

# Where a fit's in-sample linear predictor comes from.
#
# `"sampler_model"`: a ModelData sampler fit. Each draw row is the full
# parameter vector of the model recorded in `$model_inputs`, so eta is read
# back through the engine's own assembly, every latent component included.
#
# `"grid_mixture"`: a nested-Laplace fit carrying the per-cell `fitted_eta`
# (and `fitted_eta_var`) its inner solves assembled. What the grid defines for
# eta_i is the mixture `sum_k w_k N(fitted_eta[k, i], fitted_eta_var[k, i])`,
# which already holds the fields, the random effects and the offset.
#
# The design mapping an intrinsic/proper-CAR or BYM2 areal field's own raw
# latent (`n_field` coordinates) to its contribution: `unit_map` (n_units x
# n_field) to the per-unit field value, and `Z_field = Z_ind unit_map` (n_obs x
# n_field) to the per-observation eta contribution -- the same map each
# `.marginal_H_beta_*` areal helper builds for its Schur correction
# (`R/marginal_se_spatial.R`), reused here (not duplicated) because it is
# exactly the map a conditional-Laplace fit's mode assembled eta through.
# BYM2's `d_fac` scaling (`sigma * sqrt(rho) * scale_factor` for the
# structured component, `sigma * sqrt(1 - rho)` for the unstructured one) is
# folded in at the fixed `sigma = 1`, `rho = 0.5` the conditional kernel
# hardcodes. NULL for any other spatial type (gp / hsgp / spde), which are not
# reached by this design.
#' @keywords internal
.areal_field_design <- function(spatial, n_obs) {
  type <- tolower(spatial$type %||% "")
  if (!type %in% c("icar", "car", "car_proper", "bym2")) return(NULL)
  n_units <- nrow(as.matrix(spatial$adjacency))
  I_u <- Matrix::Diagonal(n_units)
  unit_map <- if (type == "bym2") {
    scale_factor <- spatial$scale_factor %||% 1.0
    d_phi   <- sqrt(0.5 + 1e-10) * scale_factor
    d_theta <- sqrt(1 - 0.5 + 1e-10)
    cbind(d_phi * I_u, d_theta * I_u)
  } else {
    I_u
  }
  Z_ind <- .field_design_Z(spatial$spatial_idx, n_units, n_obs)
  list(n_field = ncol(unit_map), unit_map = unit_map,
       Z_field = Z_ind %*% unit_map)
}

# The raw latent of a conditional-Laplace areal field (icar / car /
# car_proper / bym2) and the design that maps it (`.areal_field_design()`).
# The kernel lays its mode out as `[beta (p), re (n_re), field (n_field)]`
# (`dispatch_laplace_spatial()`), so the field is located from the recorded
# sizes of the blocks before it -- the design's columns and the random-effect
# map's rows -- and the read is refused (NULL) when those sizes do not add up
# to the mode's length, rather than taking whatever sits at its tail.
#' @keywords internal
.laplace_areal_field <- function(object, n_obs) {
  mode <- object$mode
  if (is.null(object$spatial) || !is.numeric(mode)) return(NULL)
  fd <- .areal_field_design(object$spatial, n_obs)
  if (is.null(fd)) return(NULL)
  p <- if (is.matrix(object$model_matrix)) ncol(object$model_matrix)
       else object$n_fixed %||% 0L
  M <- .tulpa_re_map(object)
  lead <- p + if (is.null(M)) 0L else nrow(M)
  if (length(mode) != lead + fd$n_field) return(NULL)
  list(u = as.numeric(mode[lead + seq_len(fd$n_field)]), design = fd)
}

# Per-unit draws (S x n_units) of a Polya-Gamma Gibbs areal field, read off the
# columns `.pg_as_chain()` (`R/fit_gibbs.R`) stores in `$draws` alongside the
# fixed effects, so each row is the field value every observation of that unit
# receives. An icar / car field is `phi_spatial[k]` itself, and a restricted
# (rsr) field's `phi_spatial[k]` is the projected field the kernel adds to eta.
# A bym2 field's `phi_spatial[k]` / `theta_spatial[k]` are the kernel's
# unit-scale components, combined per draw into `sigma * (sqrt(rho) *
# scale_factor * phi + sqrt(1 - rho) * theta)` at that draw's
# `log_sigma_spatial` / `logit_rho_bym2`, as the kernel builds its eta
# (gcol33/tulpa#964). `keep` subsets rows. NULL for a non-areal field, and on a
# ModelData sampler fit, whose `phi_spatial[k]` columns are the raw coordinates
# its density centres and scales on the way into eta.
#' @keywords internal
.gibbs_field_unit_draws <- function(object, keep) {
  spatial <- object$spatial
  if (is.null(spatial) || is.list(object$model_inputs)) return(NULL)
  type <- tolower(spatial$type %||% "")
  dn <- colnames(object$draws)
  if (!type %in% c("icar", "car", "rsr", "bym2") || is.null(dn)) return(NULL)

  phi_cols <- grep("^phi_spatial\\[", dn)
  if (!length(phi_cols)) return(NULL)
  U <- object$draws[keep, phi_cols, drop = FALSE]

  if (type == "bym2") {
    th_cols <- grep("^theta_spatial\\[", dn)
    if (length(th_cols) != length(phi_cols) ||
        !all(c("log_sigma_spatial", "logit_rho_bym2") %in% dn)) {
      return(NULL)
    }
    sigma <- exp(object$draws[keep, "log_sigma_spatial"])
    rho <- stats::plogis(object$draws[keep, "logit_rho_bym2"])
    scale_factor <- spatial$scale_factor %||% 1.0
    U <- U * (sigma * sqrt(rho) * scale_factor) +
      object$draws[keep, th_cols, drop = FALSE] * (sigma * sqrt(1 - rho))
  }
  U
}

# The continuous fields of a Polya-Gamma Gibbs chain, one entry per sampled
# NNGP field: the column stem of its values at the fitted locations, the
# suffix of its `log_sigma2_<suffix>` / `log_phi_<suffix>` columns, and the
# neighbour count it was sampled under. A gp / nngp / gp_rsr field is `gp_w`;
# a multiscale field is the sum of its local and regional scales. NULL for any
# other field.
#' @keywords internal
.gibbs_gp_fields <- function(spatial) {
  type <- tolower(spatial$type %||% "")
  nn_of <- function(info) {
    as.integer(if (is.matrix(info$nn_idx)) ncol(info$nn_idx) else spatial$nn %||% 10L)
  }
  if (type %in% c("gp", "nngp", "gp_rsr")) {
    return(list(list(stem = "gp_w", suffix = "gp",
                     nn = nn_of(spatial$neighbor_info))))
  }
  if (type %in% c("multiscale", "multiscale_gp")) {
    return(list(
      list(stem = "gp_local", suffix = "gp_local",
           nn = nn_of(spatial$neighbor_info_local)),
      list(stem = "gp_regional", suffix = "gp_regional",
           nn = nn_of(spatial$neighbor_info_regional))))
  }
  NULL
}

# Per-draw eta contribution (S x n_obs) of every field a Polya-Gamma Gibbs
# chain samples, at the training rows, for the stored draw rows `keep` (the
# rows `.fixed_coef_draws()` took of the same chain). Each term is the field
# the kernel adds to eta:
#
#   * areal (icar / car / rsr / bym2): the per-unit draws of
#     `.gibbs_field_unit_draws()` gathered through `spatial_idx`;
#   * continuous (gp / nngp / gp_rsr / multiscale): `gp_w[i]`, or
#     `gp_local[i] + gp_regional[i]`, at observation i's location (the
#     samplers fit one observation per location, in coordinate order);
#   * temporal (temporal_multiscale): `trend[t] + seasonal[(t - 1) %% period +
#     1] + short_term[t]` at the row's time index t.
#
# At new rows the same fields are mapped by the one field draw path
# (`.field_source()`, `R/spatial_field_predict.R`). NULL when the fit is not a
# Gibbs chain or samples no field.
#' @keywords internal
.gibbs_field_eta_contrib <- function(object, keep, n_obs) {
  dr <- object$draws
  if (is.list(object$model_inputs) || !is.matrix(dr) || is.null(colnames(dr))) {
    return(NULL)
  }
  dn <- colnames(dr)
  cols_of <- function(stem) grep(paste0("^", stem, "\\["), dn)
  out <- NULL
  add <- function(C) out <<- if (is.null(out)) C else out + C
  sp <- object$spatial
  if (!is.null(sp)) {
    U <- .gibbs_field_unit_draws(object, keep)
    if (!is.null(U)) {
      Z <- .field_design_Z(sp$spatial_idx, ncol(U), n_obs)
      add(as.matrix(U %*% Matrix::t(Z)))
    }
    for (g in .gibbs_gp_fields(sp)) {
      cw <- cols_of(g$stem)
      if (!length(cw)) next
      W <- dr[keep, cw, drop = FALSE]
      add(W[, as.integer(sp$obs_to_loc %||% seq_len(n_obs)), drop = FALSE])
    }
  }
  tp <- object$temporal
  if (!is.null(tp) && !is.null(tp$time_index)) {
    t_idx <- as.integer(tp$time_index)
    period <- as.integer(tp$seasonal %||% 0L)
    idx <- list(trend = t_idx, short_term = t_idx,
                seasonal = if (period > 0L) (t_idx - 1L) %% period + 1L)
    for (stem in names(idx)) {
      cs <- cols_of(stem)
      if (!length(cs) || is.null(idx[[stem]])) next
      add(dr[keep, cs, drop = FALSE][, idx[[stem]], drop = FALSE])
    }
  }
  if (!is.null(out)) dimnames(out) <- NULL
  out
}

# The one arm of a joint nested-Laplace fit, or NULL.
#
# A joint fit carries a response process, a design and a response PER ARM, so
# each of those is resolvable exactly when the fit has one arm -- and then that
# arm's own fields ARE the fit's. One predicate, so the family, the dispersion,
# the response and the design are read off the same place rather than each
# accessor deciding for itself what a one-arm fit is. `arm` is the name the arm
# was given, which is what locates a `phi_<arm>` dispersion axis on the outer
# grid (gcol33/tulpa#850).
#' @keywords internal
.tulpa_single_arm <- function(object) {
  arms <- object$responses
  if (!is.list(arms) || length(arms) != 1L || !is.list(arms[[1L]])) return(NULL)
  nm <- names(arms)[1L]
  list(spec = arms[[1L]],
       arm  = if (is.character(nm) && nzchar(nm)) nm else NULL)
}

# The response process a fit samples from: the family, the trial counts and the
# dispersion that turn a linear predictor into a distribution over y, plus the
# response itself. A `tulpa()` fit carries them flat; a joint fit carries them
# on its arm (`.tulpa_single_arm()`).
#
# Errors through `.accessor_unavailable()` where there is no single process,
# naming how many arms there are rather than only that a family is missing.
#' @keywords internal
.tulpa_response_process <- function(object, accessor) {
  arms <- object$responses
  one <- .tulpa_single_arm(object)
  # A tulpa() fit routed through the one-arm joint door carries the process
  # flat AND its arm, whose name locates an integrated dispersion axis.
  if (is.character(object$family) && length(object$family) == 1L) {
    return(list(family = object$family, n_trials = object$n_trials,
                phi = object$phi, phi2 = object$phi2, y = object$y,
                arm = one$arm, bounds = object$response_bounds))
  }
  if (!is.null(one) && is.character(one$spec$family) &&
      length(one$spec$family) == 1L) {
    a <- one$spec
    return(list(family = a$family, n_trials = a$n_trials,
                phi = a$phi, phi2 = a$phi2, y = a$y, arm = one$arm,
                bounds = .joint_arm_bounds(a)))
  }
  what <- if (is.list(arms) && length(arms) > 1L) {
    sprintf(paste0("a single response process to sample from: it has %d arms, ",
                   "each with its own family and dispersion"), length(arms))
  } else {
    "a single built-in family to sample the response from"
  }
  .accessor_unavailable(accessor, object, what)
}


# The dispersion at each of `S` replicates.
#
# Where the fit INTEGRATED the dispersion -- a `phi_<arm>` column of the outer
# grid, from `tulpa_nested_laplace_joint(phi_grid =)` -- each replicate takes
# the value of the cell it was drawn in, continuized within that cell by
# `tulpa_hyper_draws()` (gcol33/tulpa#823) rather than read as the grid node it
# sits on. Sampling every replicate at one scalar instead dropped an axis the
# fit had already paid to integrate (gcol33/tulpa#825).
#
# `cells` is the outer cell each replicate came from, which only the grid
# mixture has; everything else carries a single dispersion and is returned as
# `S` copies of it, so the caller indexes `phi[s]` on one code path.
#
# The axis is in the R-level convention -- `phi` is the residual VARIANCE for
# gaussian and lognormal at every R-level door, converted to the kernel's SD
# inside `.phi_to_kernel()` -- which is the convention the scalar it replaces
# is in, so no seam is crossed here. Measured against a gaussian fit with
# residual SD 0.45: the axis posterior concentrates at 0.1975 against a truth
# of 0.45^2 = 0.2025 (`test-posterior-predict-phi-axis.R`).
#' @keywords internal
.tulpa_phi_draws <- function(object, proc, cells, S) {
  # `.validate_family_phi()` admits only a positive finite scalar at every
  # door, so the no-axis answer is S copies of one number and the callers'
  # `phi[s]` is the scalar expression it replaces, value for value.
  scalar <- rep(as.numeric(proc$phi %||% 1.0)[1L], S)
  if (is.null(cells) || is.null(proc$arm)) return(scalar)
  tg <- object$theta_grid
  col <- paste0("phi_", proc$arm)
  if (!is.matrix(tg) || !(col %in% colnames(tg))) return(scalar)
  if (length(unique(tg[, col])) < 2L) return(scalar)   # pinned: part of the model
  hd <- tryCatch(tulpa_hyper_draws(object, cells = cells),
                 error = function(e) NULL)
  if (is.null(hd) || !(col %in% colnames(hd)) || nrow(hd) != S) {
    return(as.numeric(tg[cells, col]))
  }
  as.numeric(hd[, col])
}


# `"field_mixture"`: a grid fit (a nested-Laplace grid, the SPDE grid, a fit
# at fixed hyperparameters as one cell) carrying a latent field with its cells
# and the record its cells are solved again from, and no per-cell
# `fitted_eta` (`.field_source()`). Its linear predictor is the field's latent
# mixture mapped to the training rows (`.field_eta_draws()`).
#
# `"coefficients"`: everything else. The fit carries its fixed effects (and at
# most the formula random effects and an SPDE field) rather than its linear
# predictor, so eta is assembled from those.
#' @keywords internal
.tulpa_linpred_source <- function(object) {
  if (is.list(object$model_inputs) && is.matrix(object$draws) &&
      nrow(object$draws) > 0L) {
    return("sampler_model")
  }
  if (isTRUE(object$subspace_debias$outer$applied) &&
      identical(object$subspace_debias$method, "exact") &&
      !is.null(.field_source_grid(object))) return("field_mixture")
  if (is.matrix(object$fitted_eta) && !is.null(object$weights) &&
      length(object$weights) == nrow(object$fitted_eta)) {
    return("grid_mixture")
  }
  src <- .field_source_grid(object)
  if (!is.null(src) && !is.null(src$solve)) return("field_mixture")
  "coefficients"
}

# Fixed-effect coefficient draws (S x p), zero-inflation coefficients included,
# columns named by the fit's fixed block. The fit's own draws when it carries
# any, else a Gaussian draw at coef() / vcov(). `keep` indexes the stored draw
# rows taken (NULL for a Gaussian draw), so the random-effect draws can be read
# off the same rows.
#' @keywords internal
.fixed_coef_draws <- function(object, ndraws = NULL) {
  fd   <- .fixed_draws_mat(object)
  keep <- NULL
  if (!is.null(fd)) {
    S <- nrow(fd)
    keep <- if (!is.null(ndraws) && ndraws < S) sample.int(S, ndraws)
            else seq_len(S)
    beta <- fd[keep, , drop = FALSE]
  } else {
    S  <- as.integer(ndraws %||% 400L)
    mu <- coef(object)
    V  <- vcov(object)
    if (anyNA(V)) {
      stop("posterior_predict(): the fixed-effect covariance is unavailable ",
           "(vcov() returned NA).", call. = FALSE)
    }
    L <- tryCatch(chol(V), error = function(e) {
      chol(V + diag(1e-10 * max(diag(V), 1), nrow(V)))
    })
    beta <- matrix(rep(mu, each = S), S) +
      matrix(stats::rnorm(S * length(mu)), S) %*% L
    colnames(beta) <- names(mu)
  }
  nm <- object$fixed_names
  if (!is.null(nm) && length(nm) == ncol(beta)) colnames(beta) <- nm
  list(beta = beta, keep = keep)
}

# Linear-predictor posterior draws (S x n_obs).
#
# At the training design (newdata = NULL) the draws carry every component the
# fit estimated, read from the source `.tulpa_linpred_source()` names: the
# engine's own eta at each sampler draw, the per-cell grid mixture on a
# nested-Laplace fit, the latent mixture of a grid fit's field, and otherwise
# the fixed effects plus the offset, the formula random effects and a
# posterior-mean field. A fit carrying a latent field the one field draw path
# maps (`.field_source()`: a continuous, areal or temporal field on any
# backend) is drawn through it at `newdata` and at the training rows when
# `re_form` keeps a subset of the terms (`.field_eta_draws()`): the field at
# the rows given the fitted field, per cell (a grid's hyperparameter cell, a
# sampler's stored draw), with the fitted groups' effects from the same latent
# draw and a group the fit did not see drawn from the term's covariance. Every
# other fit reads the coefficients source: fixed effects from
# `.fixed_coef_draws()` and the offset, at `newdata` a varying coefficient's
# field at the newdata coordinates / times, and the formula random effects (a
# fitted group's effect from the same row as the fixed effects, a new group's
# from `.re_new_group_eta()`). `re_form` selects the random-effect terms
# (`.re_form_select()`). `latent = FALSE` reads the coefficients source with
# the random effects and no latent field.
#
# On a zero-inflated fit the structural-zero logit is drawn from the same rows
# and returned as the `"logit_zi"` attribute (S x n_obs), so the two predictors
# of one replicate come from one posterior draw. The logit is `X_zi beta_zi`,
# with no offset and no random effect, which is how the engine assembles it.
#
# `synth_seed` pins every randomized step RNG-neutrally: read-only callers (the
# WAIC/LOO criteria layer) get identical draws on every call and leave the
# session stream untouched; predictive callers leave it NULL for fresh draws.
#' @keywords internal
.tulpa_eta_draws <- function(object, newdata = NULL, ndraws = NULL,
                             synth_seed = NULL, re_form = NULL,
                             latent = TRUE) {
  if (!is.null(synth_seed)) {
    .preserve_seed_in_frame()
    set.seed(as.integer(synth_seed))
  }
  sel <- .re_form_select(object, re_form)
  src <- NULL
  source <- if (is.null(newdata) && latent && (is.null(sel) || all(sel))) {
    .tulpa_linpred_source(object)
  } else if (latent && !is.null(src <- .field_source(object))) {
    "field_mixture"
  } else "coefficients"
  if (identical(source, "field_mixture")) {
    if (is.null(src)) src <- .field_source(object)
    if (!is.null(object$zi_model_matrix) && !length(src$layout$zi)) {
      stop("posterior_predict(): a grid fit with a latent field carries no ",
           "zero-inflation predictor to draw.", call. = FALSE)
    }
    S <- as.integer(ndraws %||% if (isTRUE(src$point)) nrow(src$modes) else 400L)
    return(.field_eta_draws(object, src, newdata, S, sel))
  }
  if (identical(source, "sampler_model")) {
    return(.tulpa_eta_draws_sampler(object, ndraws))
  }
  if (identical(source, "grid_mixture")) {
    if (!is.null(object$zi_model_matrix)) {
      stop("posterior_predict(): a nested-Laplace grid carries no ",
           "zero-inflation predictor to draw.", call. = FALSE)
    }
    return(.tulpa_eta_draws_grid(object, ndraws))
  }

  D  <- .tulpa_designs(object, newdata, "posterior_predict")
  cd <- .fixed_coef_draws(object, ndraws)
  beta <- cd$beta
  keep <- cd$keep

  zi_nm    <- colnames(D$X_zi)
  count_nm <- setdiff(colnames(beta), zi_nm)
  if (!is.null(colnames(beta)) && all(count_nm %in% colnames(D$X)) &&
      all(zi_nm %in% colnames(beta))) {
    eta <- beta[, count_nm, drop = FALSE] %*% t(D$X[, count_nm, drop = FALSE])
  } else if (is.null(zi_nm) && ncol(D$X) == ncol(beta)) {
    eta <- beta %*% t(D$X)
  } else {
    stop("posterior_predict(): design columns do not match the fixed-effect ",
         "draws.", call. = FALSE)
  }
  if (any(D$offset != 0)) {
    eta <- sweep(eta, 2, rep_len(D$offset, ncol(eta)), "+")
  }

  # A varying coefficient is a field at the newdata coordinates / times, drawn
  # from the same stored rows as the fixed effects.
  if (!is.null(newdata)) {
    vc <- .varying_coef_eta_contrib(object, newdata, rows = keep)
    if (!is.null(vc)) eta <- eta + vc
  }

  # The formula random effects `re_form` keeps: a fitted group's effect drawn
  # with the fixed effects of its row, a group the fit did not see drawn from
  # the term's covariance.
  des <- if (is.null(newdata)) object$re_design
         else .tulpa_re_design_at(object, newdata)
  rd <- NULL
  if (!is.null(sel) && any(sel)) {
    M <- .tulpa_re_map(object, des)
    if (!is.null(M) && length(M@x)) {
      rd <- .tulpa_re_coef_draws(object, beta, keep, nrow(M))
      if (is.null(rd) && !is.null(newdata)) {
        .re_estimate_unavailable(object, "posterior_predict")
      }
      if (is.null(rd)) {
        message("posterior_predict(): no random-effect draws or point ",
                "values available; replicates are population-level.")
      }
    }
  }
  if (is.null(newdata) && latent) {
    eta <- .tulpa_add_latent_eta(object, eta, rd, keep, point = FALSE,
                                 sel = sel)
  } else if (is.null(newdata)) {
    re_eta <- .tulpa_re_eta(object, des, rd, sel)
    if (!is.null(re_eta)) eta <- eta + re_eta
  } else if (!is.null(sel) && any(sel)) {
    re_eta <- .tulpa_re_eta(object, des, rd, sel)
    if (!is.null(re_eta)) eta <- eta + re_eta
    eta <- eta + .re_new_group_eta(object, des, keep, nrow(eta), sel)
  }

  if (!is.null(zi_nm)) {
    attr(eta, "logit_zi") <- beta[, zi_nm, drop = FALSE] %*%
      t(D$X_zi[, zi_nm, drop = FALSE])
  }
  eta
}

# Add the random-effect and latent-field contributions to in-sample linear
# predictor rows `eta` (rows x n_obs) on the "coefficients" source. `re` is the
# RE coefficient matrix paired row for row with `eta` (NULL for none), mapped
# through `.tulpa_re_map()`, the terms `sel` leaves out held at zero. `keep`
# indexes the stored draw rows a sampled Polya-Gamma field is read at. With `point = TRUE`, `eta` is the one row of the
# posterior-mean predictor and a sampled field enters at its posterior mean over
# every stored draw; every other component is already a point value. The
# predictor is linear in each component, so the point row is exactly the mean
# of the rows the draw path assembles.
#' @keywords internal
.tulpa_add_latent_eta <- function(object, eta, re, keep, point = FALSE,
                                  sel = NULL) {
  n_obs <- ncol(eta)
  if (!is.null(re)) {
    re_eta <- .tulpa_re_eta(object, object$re_design, re, sel)
    if (!is.null(re_eta)) eta <- eta + re_eta
  }
  if (identical(object$spatial$type, "spde") &&
      !is.null(object$spatial_effects) && !is.null(object$spatial$A)) {
    eta <- sweep(eta, 2,
                 as.numeric(object$spatial$A %*% object$spatial_effects), "+")
  } else if (!is.null(object$spatial) && !is.null(object$mode)) {
    # Conditional-Laplace areal fit (icar / car / car_proper / bym2): the
    # field's raw latent is the block of the fit's own mode after the fixed
    # and random effects (`[beta, re, field]`, matching every
    # `.marginal_H_beta_*` areal helper's own layout,
    # `R/marginal_se_spatial.R`), so it is read off exactly as it entered
    # eta at the mode -- a posterior-mean field, like the SPDE branch above
    # (gcol33/tulpa#795).
    lf <- .laplace_areal_field(object, n_obs)
    if (!is.null(lf)) {
      eta <- sweep(eta, 2, as.numeric(lf$design$Z_field %*% lf$u), "+")
    }
  } else if (is.matrix(object$draws) &&
             (!is.null(object$spatial) || !is.null(object$temporal))) {
    # Polya-Gamma Gibbs fit: the field is actually SAMPLED (its columns sit
    # alongside the fixed effects in `$draws`), so every draw gets its own
    # field value instead of a posterior mean (gcol33/tulpa#795).
    rows <- if (point || is.null(keep)) seq_len(nrow(object$draws)) else keep
    fc <- .gibbs_field_eta_contrib(object, rows, n_obs)
    if (!is.null(fc)) {
      eta <- if (point) sweep(eta, 2, colMeans(fc), "+") else eta + fc
    }
  } else if (!is.null(object$field_eta_contrib)) {
    # Inline spatial() / temporal() joint field fit: the field's own
    # weighted-mode posterior mean, read back per observation at fit time
    # (`.bar_field_fit_core()`, `R/spatial_field.R`), since the joint
    # multi-block driver behind these fits carries no single-cell
    # `fitted_eta` for `.tulpa_linpred_source()` to read (gcol33/tulpa#795).
    eta <- sweep(eta, 2, object$field_eta_contrib, "+")
  }
  eta
}

# Every ModelData sampler stores a draw in one form (`src/sampler_field_store.h`):
# each non-centered GP / SVC / multiscale-GP / temporal-GP / multiscale-trend /
# multiscale short-term block's slice holds the reconstructed field rather than
# the sampled coordinates. Re-running such a draw through
# `initialize_generic_state()` at the spec's own parameterization would apply
# the non-centered forward transform a SECOND time, to a slice that is no
# longer the coordinates (gcol33/tulpa#822). Reading the stored draws back as
# centered undoes exactly that: the forward transform is the identity on an
# already-centered field, matching what the store wrote.
#' @keywords internal
.stored_draw_field_specs <- function(mi) {
  spatial_spec  <- mi$spatial_spec
  svc_spec      <- mi$svc_spec
  temporal_spec <- mi$temporal_spec
  tvc_spec <- mi$tvc_spec
  if (identical(tvc_spec$structure, "gp")) tvc_spec$parameterization <- "centered"
  if (!is.null(spatial_spec$gp_parameterization) &&
      spatial_spec$gp_parameterization == 1L) {
    spatial_spec$gp_parameterization <- 0L
  }
  if (!is.null(spatial_spec$msgp_parameterization) &&
      spatial_spec$msgp_parameterization == 1L) {
    spatial_spec$msgp_parameterization <- 0L
  }
  if (!is.null(svc_spec$svc_parameterization) &&
      svc_spec$svc_parameterization == 1L) {
    svc_spec$svc_parameterization <- 0L
  }
  if (identical(temporal_spec$type, "gp") ||
      identical(temporal_spec$type, "multiscale")) {
    temporal_spec$parameterization <- "centered"
  }
  list(spatial_spec = spatial_spec, svc_spec = svc_spec,
       temporal_spec = temporal_spec, tvc_spec = tvc_spec)
}

# The engine's linear predictor (rows of `D` x observations) for draw rows `D`
# laid out as the fit's `$draws`, assembled from the inputs the sampler
# received (`$model_inputs`) with the stored-draw field specs, so a field is
# read back exactly as its density put it into eta. `offset` is the
# observation offset to add (NULL for none).
#' @keywords internal
.sampler_engine_eta <- function(object, D, offset) {
  args <- .sampler_model_args(object)
  args$offset_nullable <- offset
  do.call(cpp_tulpa_glmm_eta_draws, c(list(draws = D), args))
}

# The model a ModelData sampler fit was run on, as the argument list every
# compiled reader of its `$draws` takes (`cpp_tulpa_glmm_eta_draws()`,
# `cpp_varying_coef_draws()`): the inputs the sampler received, with the
# stored-draw field specs, so each reader rebuilds the one layout the draws were
# written on.
#' @keywords internal
.sampler_model_args <- function(object) {
  mi <- object$model_inputs
  fs <- .stored_draw_field_specs(mi)
  list(y = mi$y, n_trials = mi$n_trials, X = mi$X,
       family = mi$family, phi = mi$phi, sigma_beta = mi$sigma_beta,
       offset_nullable = mi$offset, re_spec = mi$re_spec,
       spatial_spec = fs$spatial_spec, temporal_spec = fs$temporal_spec,
       sigma_re_scale = mi$sigma_re_scale, phi2 = mi$phi2,
       svc_spec = fs$svc_spec, tvc_spec = fs$tvc_spec, zi_spec = mi$zi_spec)
}

# Engine eta at each sampler draw (the "sampler_model" source). The
# zero-inflation coefficients sit wherever the engine's parameter layout puts
# them, so their span is asked of the layout built from the same inputs rather
# than assumed to follow the count coefficients.
#' @keywords internal
.tulpa_eta_draws_sampler <- function(object, ndraws = NULL) {
  D <- object$draws
  S <- nrow(D)
  if (!is.null(ndraws) && ndraws < S) D <- D[sample.int(S, ndraws), , drop = FALSE]
  mi <- object$model_inputs
  eta <- .sampler_engine_eta(object, D, offset = mi$offset)
  if (!is.null(mi$zi_spec)) {
    zi_cols <- .layout_span_cols(.tulpa_sampler_layout(object)$beta_zi)
    attr(eta, "logit_zi") <- D[, zi_cols, drop = FALSE] %*% t(mi$zi_spec$X)
  }
  eta
}

# Draws from the per-cell eta mixture of a nested-Laplace fit (the
# "grid_mixture" source). A row picks a cell by its outer weight and draws each
# observation from that cell's Gaussian, so every row carries one hyperparameter
# value and each column is the grid's own marginal for eta_i. Within a cell the
# observations are drawn independently: the cell's joint eta covariance is not
# retained, only its diagonal. A fit run with `control$fitted_var = FALSE`
# carries no diagonal either, and its draws hold the across-cell spread only.
#' @keywords internal
.tulpa_eta_draws_grid <- function(object, ndraws = NULL) {
  M <- object$fitted_eta
  w <- as.numeric(object$weights)
  w[!is.finite(w)] <- 0
  if (sum(w) <= 0) {
    stop("posterior_predict(): the outer grid carries no weight on any cell.",
         call. = FALSE)
  }
  S <- as.integer(ndraws %||% 400L)
  cells <- sample.int(nrow(M), S, replace = TRUE, prob = w)
  eta <- M[cells, , drop = FALSE]
  V <- object$fitted_eta_var
  if (is.matrix(V) && identical(dim(V), dim(M))) {
    eta <- eta + sqrt(pmax(V[cells, , drop = FALSE], 0)) *
      matrix(stats::rnorm(length(eta)), nrow(eta))
  }
  dimnames(eta) <- NULL
  # The cell each row was drawn in travels with the rows, so a caller needing
  # the hyperparameter value that row was drawn UNDER -- the dispersion, in
  # `.tulpa_phi_draws()` -- reads it off the same draw rather than re-sampling
  # the mixture and getting a different one (gcol33/tulpa#825).
  attr(eta, "cells") <- cells
  eta
}

# The point in-sample linear predictor a fit's fitted values and residuals are
# read at, `list(eta, logit_zi)` over the training rows.
#
# `"population"`: the fixed effects and the offset at coef(), every random
# effect and latent field held at zero (`.tulpa_point_linpred()`).
#
# `"conditional"`: the posterior mean of the in-sample linear predictor with
# every component the fit estimated, from the same source the replicates of
# posterior_predict() / simulate() are drawn from (`.tulpa_linpred_source()`):
# the mean over the stored draws of the engine's own eta on a sampler fit, the
# weight-averaged per-cell `fitted_eta` on a nested-Laplace grid, the
# weight-averaged map of the cell modes on a continuous-field mixture
# (`.field_point_eta()`), and otherwise
# coef() plus the posterior-mean random effects (`.tulpa_re_point()`) and field
# (`.tulpa_add_latent_eta(point = TRUE)`). Residuals read against this
# predictor are the ones the replicates are checked against, so a structure the
# model's field or group effects absorbed is absent from them.
#' @keywords internal
.tulpa_fitted_linpred <- function(object, effects = c("conditional", "population"),
                                  accessor) {
  effects <- match.arg(effects)
  if (identical(effects, "population")) {
    lp <- .tulpa_point_linpred(object, NULL, accessor)
    return(list(eta = lp$eta, logit_zi = lp$logit_zi))
  }
  source <- .tulpa_linpred_source(object)
  if (identical(source, "sampler_model")) {
    eta <- .tulpa_eta_draws_sampler(object)
    lz <- attr(eta, "logit_zi")
    return(list(eta = as.numeric(colMeans(eta)),
                logit_zi = if (!is.null(lz)) as.numeric(colMeans(lz))))
  }
  if (identical(source, "grid_mixture")) {
    if (!is.null(object$zi_model_matrix)) {
      .accessor_unavailable(accessor, object,
                            "a zero-inflation predictor on the nested-Laplace grid")
    }
    # A cell the screen dropped carries zero weight and no solved eta (NA), so
    # the mean runs over the cells that carry weight.
    w <- as.numeric(object$weights)
    w[!is.finite(w)] <- 0
    on <- w > 0
    if (!any(on)) {
      .accessor_unavailable(accessor, object, "an outer-grid cell carrying weight")
    }
    M <- object$fitted_eta[on, , drop = FALSE]
    return(list(eta = as.numeric(colSums(w[on] * M) / sum(w[on])),
                logit_zi = NULL))
  }
  if (identical(source, "field_mixture")) {
    return(list(eta = .field_point_eta(object, .field_source(object), NULL,
                                       sel = .re_form_select(object, NULL),
                                       accessor = accessor),
                logit_zi = NULL))
  }
  lp <- .tulpa_point_linpred(object, NULL, accessor)
  eta <- .tulpa_add_latent_eta(object, matrix(lp$eta, nrow = 1L), NULL,
                               keep = NULL, point = TRUE)
  re <- .tulpa_re_point_eta(object, object$re_design, accessor = accessor)
  if (!is.null(re)) eta <- eta + re
  list(eta = as.numeric(eta), logit_zi = lp$logit_zi)
}

# The posterior-mean random-effect part of the linear predictor on a term design
# `des` (the training rows, or `.tulpa_re_design_at()` at new data): each
# fitted group at the posterior mean of its effect (`.tulpa_re_point()`), a
# group the fit did not see at its prior mean, zero, and the terms `sel` leaves
# out at zero. The one point read fitted(), residuals() and predict() share.
# NULL when the fit has no formula random effects; a fit that has them but
# carries no estimate of them is refused through `accessor`.
#' @keywords internal
.tulpa_re_point_eta <- function(object, des, sel = NULL, accessor) {
  M <- .tulpa_re_map(object, des)
  if (is.null(M)) return(NULL)
  if (!length(M@x)) return(rep(0, ncol(M)))   # every row a new group
  b <- .tulpa_re_point(object, nrow(M))
  if (is.null(b)) .re_estimate_unavailable(object, accessor)
  as.numeric(.tulpa_re_eta(object, des, matrix(b, nrow = 1L), sel))
}

# Refuse a conditional read on a fit that carries no estimate of its group
# effects, with the fit's own reason where it states one.
#' @keywords internal
.re_estimate_unavailable <- function(object, accessor) {
  why <- object$ranef_unavailable
  .accessor_unavailable(accessor, object, paste0(
    "a posterior estimate of its random effects for the conditional ",
    "predictor", if (is.character(why)) paste0(" (", why, ")"),
    "; re_form = NA gives the population level"))
}

#' Posterior predictive replicates
#'
#' @description
#' Draw replicated responses from the posterior predictive distribution: the
#' in-sample linear predictor is drawn with every component the fit estimated
#' -- fixed effects, formula random effects, the offset, and any spatial or
#' temporal field -- and pushed through the family's sampling distribution. The
#' same draws give the pointwise log-likelihood behind
#' `compare_models(criterion = "waic")` / `"loo"`.
#'
#' Where the draws come from follows what the fit carries:
#' \itemize{
#'   \item A ModelData sampler fit (`mode = "hmc"` and its siblings) evaluates
#'     the engine's own linear predictor at each draw, so a field is drawn
#'     jointly with everything else.
#'   \item A nested-Laplace fit draws from its outer grid: a replicate picks a
#'     cell by its weight and draws each observation from that cell's Gaussian
#'     for the linear predictor (`fitted_eta`, `fitted_eta_var`). The cell's
#'     joint covariance across observations is not retained, so observations
#'     within a replicate are independent given the cell. A fit run with
#'     `control$fitted_var = FALSE` carries no `fitted_eta_var`, and its
#'     replicates hold the across-cell spread only.
#'   \item Any other fit carries its coefficients rather than its linear
#'     predictor. Fits with posterior draws use them (fixed and random effects
#'     jointly per draw). A Laplace or EB fit (`mode = "laplace"` / `"eb"`)
#'     samples the fixed effects from `N(coef(fit), vcov(fit))` and each
#'     draw's random effects from their Gaussian conditional given those fixed
#'     effects under the joint Laplace precision of `[beta | b]` at the mode,
#'     so the two carry their posterior correlation (on a conditional Laplace
#'     fit this is the joint Gaussian draw; an EB fit is conditional on the
#'     estimated variance components). A conditional-Laplace fit with an
#'     NNGP, HSGP, SPDE or areal field draws the coefficients, the group
#'     effects and the field jointly from the Laplace Gaussian of its one
#'     cell (`.field_eta_draws()`), the cell solved again by the kernel of
#'     its block.
#' }
#' At `newdata` each replicate carries the formula random effects, as
#' [predict.tulpa_fit()] does: a row of a group the fit saw takes that group's
#' effect, drawn jointly with the fixed effects of the replicate, and a row of a
#' group it did not see takes a new effect drawn from the term's covariance --
#' at the replicate's own draw of it where the fit samples or integrates the
#' covariance (`re_cov = "gibbs"`, `"nested"`, the samplers), at its estimate or
#' conditioned value otherwise. Rows of one new group share its effect. `re_form
#' = NA` drops them (the population level). A spatially or temporally varying
#' coefficient enters at the `newdata` coordinates / times, drawn per replicate
#' from its conditional given that posterior draw.
#'
#' A zero-inflated fit (`ziformula`) draws the structural-zero logit from the
#' same posterior draw as the count predictor; each replicate observation is a
#' structural zero with probability `plogis(X_zi beta_zi)` and otherwise a draw
#' from the family (a zero-truncated family gives the hurdle model).
#'
#' @param object A `tulpa_fit` object from [tulpa()].
#' @param ... Passed to methods.
#' @return A `ndraws x n_obs` numeric matrix of replicated responses.
#' @seealso [pp_check()], which uses these replicates; [simulate.tulpa_fit()].
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(y = rpois(100, 4), x = rnorm(100))
#' fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
#' yrep <- posterior_predict(fit, ndraws = 100)
#' dim(yrep)  # 100 x 100
#' }
#' @export
posterior_predict <- function(object, ...) {
  UseMethod("posterior_predict")
}

#' @param newdata Optional data frame of covariates to predict at, with the
#'   grouping columns of the random-effect terms `re_form` keeps. `NULL`
#'   (default) replicates at the training data with every fitted component
#'   included.
#' @param ndraws Number of posterior draws to use. Defaults to all stored
#'   draws, or 400 on the draw-free Laplace tier.
#' @param n_trials Binomial / beta-binomial trial counts for the replicates.
#'   Defaults to the training trials when `newdata` is `NULL`, else 1.
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @param re_form Which formula random-effect terms to condition on, as lme4's
#'   `re.form`: `NULL` (default) every term, `NA` or `~0` none (the population
#'   level), or a one-sided formula naming a subset, e.g. `~ (1 | site)`.
#' @rdname posterior_predict
#' @export
posterior_predict.tulpa_fit <- function(object, newdata = NULL, ndraws = NULL,
                                        n_trials = NULL, seed = NULL,
                                        re_form = NULL, ...) {
  proc <- .tulpa_response_process(object, "posterior_predict")
  .seed_scoped(seed)

  eta <- .tulpa_eta_draws(object, newdata = newdata, ndraws = ndraws,
                          re_form = re_form)
  logit_zi <- attr(eta, "logit_zi")
  if (is.null(n_trials)) {
    n_trials <- if (is.null(newdata)) proc$n_trials else NULL
  }
  # One dispersion per replicate: the cell's own where the fit integrated a
  # dispersion axis, the fit's scalar repeated otherwise.
  phi <- .tulpa_phi_draws(object, proc, attr(eta, "cells"), nrow(eta))
  bounds <- if (is.null(newdata)) proc$bounds
            else .response_bounds_at(object, newdata)

  yrep <- matrix(NA_real_, nrow(eta), ncol(eta))
  for (s in seq_len(nrow(eta))) {
    yrep[s, ] <- .response_sample(eta[s, ],
                                  if (!is.null(logit_zi)) logit_zi[s, ],
                                  proc$family, n_trials = n_trials,
                                  phi = phi[s], phi2 = proc$phi2,
                                  bounds = bounds)
  }
  yrep
}

#' Posterior of the linear predictor
#'
#' @description
#' The posterior of the linear predictor `eta` at the training data with every
#' component the fit estimated: fixed effects, the offset, formula random
#' effects, spatial and temporal fields and `latent()` blocks. It is the
#' quantity [posterior_predict()] pushes through the family, read here on the
#' link scale; [fitted.tulpa_fit()] is its posterior mean at the training data
#' and [predict.tulpa_fit()] summarizes it at new data.
#'
#' On a nested-Laplace fit the posterior of each `eta_i` is the grid mixture
#' `sum_k w_k N(fitted_eta[k, i], fitted_eta_var[k, i])`, and `summary = TRUE`
#' reports its exact mean, SD and quantiles. A fit run with
#' `control$fitted_var = FALSE` carries no within-cell variance, and its summary
#' reports the mixture mean alone (`source = "mode"`, `sd` and bounds `NA`).
#' Every other fit summarizes its draws.
#'
#' At `newdata`, a latent field is drawn at the new rows from its conditional
#' distribution given the fitted field (within each hyperparameter cell and
#' mixed over the cells, or given each stored draw of a sampler): a continuous
#' field at the new locations, an areal or temporal field at the row's fitted
#' unit or time (see [predict.tulpa_fit()]); these are the draws
#' [predict.tulpa_fit()] summarizes, so the two agree to Monte Carlo error.
#'
#' @param object A `tulpa_fit` object.
#' @param ... Passed to methods.
#' @return With `summary = FALSE`, an `ndraws x n_obs` matrix of draws. With
#'   `summary = TRUE`, a data frame with one row per observation: `estimate`,
#'   `sd`, `conf.low`, `conf.high` (at `probs`) and `source` (`"mixture"` for
#'   the exact grid mixture, `"mode"` for its mean alone, `"sampled"` for a
#'   draw summary).
#' @seealso [posterior_predict()], [predict.tulpa_fit()].
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(y = rpois(100, 4), x = rnorm(100), g = gl(10, 10))
#' fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "laplace")
#' head(posterior_linpred(fit, summary = TRUE))
#' }
#' @export
posterior_linpred <- function(object, ...) {
  UseMethod("posterior_linpred")
}

#' @param newdata Optional data frame of covariates, read as in
#'   [posterior_predict()]. `NULL` (default) reads the training data.
#' @param ndraws Number of draws (`summary = FALSE`, or a draw-based summary).
#' @param summary Logical; return a per-observation summary instead of draws.
#' @param probs The two probabilities of the reported interval.
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @param re_form Which formula random-effect terms to condition on, as in
#'   [posterior_predict()].
#' @rdname posterior_linpred
#' @export
posterior_linpred.tulpa_fit <- function(object, newdata = NULL, ndraws = NULL,
                                        summary = FALSE,
                                        probs = c(0.025, 0.975),
                                        seed = NULL, re_form = NULL, ...) {
  .seed_scoped(seed)
  if (isTRUE(summary) && is.null(newdata) && is.null(re_form) &&
      identical(.tulpa_linpred_source(object), "grid_mixture")) {
    M <- object$fitted_eta
    V <- object$fitted_eta_var
    if (!is.matrix(V) || !identical(dim(V), dim(M))) V <- NULL
    w <- as.numeric(object$weights)
    keep <- is.finite(w) & w > 0 & is.finite(rowSums(M))
    if (!is.null(V)) keep <- keep & is.finite(rowSums(V))
    mx <- .nl_gauss_mixture_summary(
      M[keep, , drop = FALSE], if (!is.null(V)) pmax(V[keep, , drop = FALSE], 0),
      w[keep] / sum(w[keep]), probs = probs)
    return(data.frame(estimate = mx$mean, sd = mx$sd,
                      conf.low = mx$quantiles[, 1L],
                      conf.high = mx$quantiles[, 2L],
                      source = if (is.null(V)) "mode" else "mixture",
                      row.names = NULL,
                      stringsAsFactors = FALSE))
  }
  eta <- .tulpa_eta_draws(object, newdata = newdata, ndraws = ndraws,
                          re_form = re_form)
  attr(eta, "cells") <- NULL
  attr(eta, "logit_zi") <- NULL
  if (!isTRUE(summary)) return(eta)
  data.frame(
    estimate  = colMeans(eta),
    sd        = apply(eta, 2L, stats::sd),
    conf.low  = apply(eta, 2L, stats::quantile, probs = probs[1L], names = FALSE),
    conf.high = apply(eta, 2L, stats::quantile, probs = probs[2L], names = FALSE),
    source    = "sampled", row.names = NULL, stringsAsFactors = FALSE)
}

#' Simulate responses from a fitted tulpa model
#'
#' @description
#' Base-R alias for [posterior_predict()]: each simulation is one posterior
#' predictive replicate at the training data. A categorical fit
#' ([tulpa_multinomial()], [tulpa_ordinal()]) simulates factor columns carrying
#' the response levels.
#'
#' @param object A `tulpa_fit` object.
#' @param nsim Number of simulated datasets (default 1).
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @param ... Ignored.
#' @return A data frame with `nsim` columns (`sim_1`, ...), one row per
#'   observation, following the [stats::simulate()] convention: its `"seed"`
#'   attribute is the value of `seed` (with the RNG kind as attribute `"kind"`)
#'   when one was given, else the state of `.Random.seed` before simulation.
#' @export
simulate.tulpa_fit <- function(object, nsim = 1, seed = NULL, ...) {
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    stats::runif(1)
  }
  rng_state <- if (is.null(seed)) get(".Random.seed", envir = .GlobalEnv)
               else structure(seed, kind = as.list(RNGkind()))

  yrep <- posterior_predict(object, ndraws = nsim, seed = seed)
  levels_rep  <- attr(yrep, "levels")
  ordered_rep <- isTRUE(attr(yrep, "ordered"))
  if (nrow(yrep) > nsim) yrep <- yrep[seq_len(nsim), , drop = FALSE]
  out <- as.data.frame(t(yrep))
  if (!is.null(levels_rep)) {
    out[] <- lapply(out, function(code) {
      factor(levels_rep[code], levels = levels_rep, ordered = ordered_rep)
    })
  }
  names(out) <- paste0("sim_", seq_len(ncol(out)))
  rownames(out) <- NULL
  attr(out, "seed") <- rng_state
  out
}
