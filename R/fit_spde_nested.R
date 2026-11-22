# The nested-Laplace integration of fit_spde() over (range, sigma).
#
# An integer-nu field is an `spde` block of the joint door: one arm carrying the
# response, the fixed effects and the conditioned random intercept, and the
# field's precision assembled per cell by the C++ block factory. The outer grid
# is the joint door's (placement from the outer mode, refinement, cheap screen,
# outer Pareto-k), and its fit is the record `$nested` holds.
#
# A fractional-nu field is the covariance-based rational SPDE, whose
# precision and observation map are assembled per cell in R
# (`.spde_nested_logmarginal_at()`); no compiled block carries that operator, so
# its cells reach the generic callback driver `tulpa_hyper_grid()` instead.

# The SPDE spec as a joint-door `spde` block of one arm. The PC anchors the
# spec carries -- stated or defaulted -- place the outer grid either way
# (`default_range_prior` / `default_sigma_prior`); under `"flat"` only a stated
# anchor is a density on its axis, as in `.spde_hyperprior_record()`.
.spde_joint_block <- function(sp, n_obs, hyperprior = "proper") {
  flat <- identical(.hp_choice(hyperprior), "flat")
  stated <- sp$prior_stated %||% c(range = TRUE, sigma = TRUE)
  n <- .nl_grid_par("spde_direct", "n")
  span <- .nl_grid_par("spde_direct", "span")
  ladder <- exp(seq(-log(span), log(span), length.out = n))
  cells <- expand.grid(range = sp$prior_range[1L] * ladder,
                       sigma = sp$prior_sigma[1L] * ladder)
  list(
    type                = "spde",
    role                = "spatial",
    n_mesh              = as.integer(sp$n_mesh),
    A_x                 = list(as.numeric(sp$A_x)),
    A_i                 = list(as.integer(sp$A_i)),
    A_p                 = list(as.integer(sp$A_p)),
    n_obs_per_arm       = as.integer(n_obs),
    C0_diag             = as.numeric(sp$C0_diag),
    G1_x                = as.numeric(sp$G1_x),
    G1_i                = as.integer(sp$G1_i),
    G1_p                = as.integer(sp$G1_p),
    nu                  = as.numeric(sp$nu),
    range_grid          = auto_grid(cells$range),
    sigma_grid          = auto_grid(cells$sigma),
    prior_range        = if (!flat || stated[["range"]]) sp$prior_range,
    prior_sigma         = if (!flat || stated[["sigma"]]) sp$prior_sigma,
    default_range_prior = sp$prior_range,
    default_sigma_prior = sp$prior_sigma
  )
}

# The integer-nu nested fit: the joint door over one arm and the field's
# block. `re` is the conditioned random intercept (`re_idx`, `n_re_groups`,
# `sigma_re`) or NULL; its effects sit between the fixed effects and the mesh
# nodes in every cell's latent vector, the order a fit_spde() mode has.
.spde_nested_joint <- function(sp, y, X, family, phi, n_trials, offset,
                               beta_prior, re, hyperprior, control) {
  arm <- list(y = y, X = X, family = family, phi = phi, n_trials = n_trials,
              offset = offset)
  if (!is.null(beta_prior)) {
    arm$beta_prior_mean <- beta_prior$mean
    arm$beta_prior_prec <- 1 / beta_prior$sd^2
  }
  if (!is.null(re)) arm[names(re)] <- re
  tulpa_nested_laplace_joint(
    responses = list(y = arm),
    prior = list(.spde_joint_block(sp, length(y), hyperprior)),
    hyperprior = hyperprior, control = control)
}

# The fractional-nu nested fit: `tulpa_hyper_grid()` over (range, sigma), each
# cell the rational operator's Laplace marginal. The axes span the PC anchors
# times / divided by the `spde_direct` span in log; the CCD is centred at the mode its own
# mode-find reaches inside that box, and `control$integration = "grid"` takes
# the tensor of the axes instead.
.spde_nested_fractional <- function(sp, y, X, family, phi, n_trials, offset,
                                    beta_prior, hyperprior, control) {
  block <- .spde_joint_block(sp, length(y), hyperprior)
  anchors <- c(range = sp$prior_range[1L], sigma = sp$prior_sigma[1L])
  span <- .nl_grid_par("spde_direct", "span")
  specs <- lapply(names(anchors), function(a) {
    hp <- .hp_axis_prior(a, block, hyperprior = hyperprior)
    hyper_axis_spec(a, grid = anchors[[a]] * exp(seq(-log(span), log(span),
                                                     length.out = 9L)),
                    log_prior = hp$fn, log_scale = TRUE, bounds = c(0, Inf))
  })
  order <- .spde_rational_order(sp)
  max_iter <- as.integer(control$max_iter %||% 100L)
  tol <- control$tol %||% 1e-6
  n_threads <- as.integer(control$n_threads %||% 1L)
  inner <- function(h) {
    f <- .spde_nested_logmarginal_at(
      spatial = sp, range = h[["range"]], sigma = h[["sigma"]], y = y, X = X,
      family = family, phi = phi, n_trials = n_trials, order = order,
      max_iter = max_iter, tol = tol, n_threads = n_threads, offset = offset,
      beta_prior = beta_prior)
    list(log_marginal = f$log_marginal)
  }
  res <- tulpa_hyper_grid(
    specs, inner, combine = "none",
    design = if (identical(control$integration, "grid")) "grid" else "ccd",
    ccd_start = anchors)
  res$prior <- list(block)
  res
}

# A fit_spde() result from its nested record `nested` (a joint-door fit, or a
# `tulpa_hyper_grid()` result on the fractional path). The fit reports the cell
# of highest posterior density as its mode: that cell's inner mode, from the
# record where the record kept it, else solved again by `refit(range, sigma)`.
# The record keeps the outer grid the accessors read (`.nl_outer_record()`) and
# gains the (range, sigma) summaries and per-node field variances fit_spde()
# fits report.
.spde_fit_from_nested <- function(nested, sp, spatial, n_fixed, refit = NULL) {
  tg <- nested$theta_grid
  bare <- .hyper_axis_bare(colnames(tg))
  rg <- as.numeric(tg[, match("range", bare)])
  sg <- as.numeric(tg[, match("sigma", bare)])
  w <- nested$weights
  lp <- nested$log_marginal
  best <- which.max(ifelse(is.finite(lp), lp, -Inf))
  nested$range_grid <- rg
  nested$sigma_grid <- sg
  nested$range_mean <- sum(w * rg)
  nested$sigma_mean <- sum(w * sg)
  nested$best_idx <- best
  nested$range_best <- rg[best]
  nested$sigma_best <- sg[best]

  M <- nested$modes
  if (is.matrix(M) && nrow(M) == length(lp)) {
    mode <- as.numeric(M[best, ])
    n_x <- length(mode)
    at <- list(mode = mode, beta = mode[seq_len(n_fixed)],
               spatial_effects = mode[seq.int(n_x - as.integer(sp$n_mesh) + 1L, n_x)],
               log_det_Q = nested$log_det_Q[best])
    nested <- .spde_grid_field_var(nested, sp$n_mesh)
  } else {
    at <- refit(rg[best], sg[best])
  }

  lhp <- nested$log_hyperprior %||% nested$log_prior %||% 0
  # A cell the cheap screen dropped is never solved; convergence is a statement
  # about the cells that were.
  dropped <- as.logical(nested$prune_mask %||% logical(length(lp)))
  solved_converged <- as.logical(nested$converged %||% TRUE)
  if (length(solved_converged) == length(lp)) {
    solved_converged <- solved_converged[!dropped]
  }
  declined <- nested$log_hyperprior_declined
  if (length(declined)) names(declined) <- .hyper_axis_bare(names(declined))
  out <- list(
    mode            = at$mode,
    beta            = at$beta,
    spatial_effects = at$spatial_effects,
    log_det_Q       = at$log_det_Q,
    log_marginal    = lp - lhp,
    log_evidence    = nested$log_evidence %||% NA_real_,
    log_evidence_declined = nested$log_evidence_declined,
    log_hyperprior_declined = if (length(declined)) declined,
    converged       = all(solved_converged),
    spatial         = spatial,
    range           = rg[best],
    sigma           = sg[best]
  )
  out$prune_mask <- nested$prune_mask
  k <- grep("^pareto_k", names(nested), value = TRUE)
  out[k] <- nested[k]
  out$nested <- nested
  out
}

# The marginal variance of every mesh node within each cell of a nested
# record: the diagonal of the cell's inverse precision on unit loadings, one
# Takahashi pass per cell (`cpp_loading_marginal_var()`), conditioned on the
# record's constraint rows. With the cells' modes these are the Gaussian
# components of the grid mixture `field_effects()` summarizes. The mesh nodes
# are the last `n_mesh` latent coordinates.
.spde_grid_field_var <- function(nested, n_mesh) {
  M <- nested$modes
  Q_p <- nested$Q_csc_p_per_grid
  n_x <- ncol(M)
  if (!is.list(Q_p) || length(Q_p) != nrow(M) ||
      !identical(as.integer(nested$Q_csc_n), n_x)) {
    return(nested)
  }
  mesh_cols <- seq.int(n_x - as.integer(n_mesh) + 1L, n_x)
  p_mesh <- length(mesh_cols)
  nested$grid_field_var <- cpp_loading_marginal_var(
    Q_p_per_grid = Q_p, Q_i_per_grid = nested$Q_csc_i_per_grid,
    Q_x_per_grid = nested$Q_csc_x_per_grid, n_x = n_x,
    load_idx = as.integer(mesh_cols), load_ptr = seq.int(0L, p_mesh),
    load_w = matrix(1, nrow(M), p_mesh),
    A_cols_list = .constraint_rows_for_cpp(nested$constraint_cols))
  nested$grid_field_var_cols <- mesh_cols
  nested
}
