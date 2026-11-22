# The adaptive Gauss-Hermite inner marginal of a joint fit (`control$n_quad`).
#
# The joint driver's inner solve is a Laplace approximation of
# p(y | theta) = int p(y | beta, b) p(b | theta) db dbeta at the joint mode of
# [beta | b]. For a model whose only latent blocks are random effects over ONE
# grouping factor, the integral over b factorizes over the groups, and each
# group's integral can be taken by n_quad-point adaptive Gauss-Hermite
# quadrature instead (the debias `tulpa_re_aghq()` applies; Laplace is
# n_quad = 1). The fixed effects are then integrated by a Laplace step over
# F(beta) = sum_g log M_g(beta, theta) + log p(beta), which is the compiled
# profile `cpp_aghq_profile_batch()` runs against the shared single-arm GLMM
# oracle (`cpp_glmm_oracle_make()`, src/glmm_oracle.h).
#
# The cell solve below takes the place of the compiled Laplace kernel behind
# the driver's one `call_kernel` closure, so the placement, the screen-free
# grid, its refinement, the CCD and the outer Pareto-k all read the quadrature
# marginal through the same door. Each cell's covariance coordinates go to the
# profile in the log-Cholesky packing every random-effect covariance shares
# (`.re_logchol_to_L()`): log sigma for an `iid` block, the block's
# `L<i><j>` columns in `.mcar_logchol_names()` order for an `miid` one.

# The families the GLMM oracle carries (`glmm_family_from_string()`).
.JOINT_AGHQ_FAMILIES <- c("binomial", "poisson", "gaussian", "neg_binomial_2")

# Why a fit at n_quad > 1 reports no per-group random effects.
.AGHQ_RANEF_UNAVAILABLE <- paste0(
  "the adaptive Gauss-Hermite inner marginal (n_quad > 1) integrates each ",
  "group's random effects out by quadrature instead of conditioning at ",
  "their mode, so this fit carries no per-group posterior. Refit with ",
  "n_quad = 1 (the joint-field Laplace inner solve) or with ",
  "control$re_cov = \"gibbs\", both of which report random effects.")

# The quadrature plan of a joint fit, or NULL at n_quad = 1. Refused, naming
# the feature, for a fit the per-group factorization does not describe: more
# than one arm, an arm-level random intercept, a block that is not a random
# effect over the shared grouping factor, a copy, a dispersion axis, weights,
# a model-supplied likelihood or a family the oracle does not carry.
.joint_aghq_plan <- function(n_quad, arms, blocks_spec, prepared, axis_names,
                             axis_offsets, cp, cell_coupling, phi_axes,
                             laplace_kernel) {
  n_quad <- .check_n_quad(n_quad)
  if (n_quad == 1L) return(NULL)
  refuse <- function(why) {
    stop("`control$n_quad > 1` (the adaptive Gauss-Hermite inner marginal) ",
         "needs a single arm whose latent blocks are random effects over one ",
         "grouping factor; this fit has ", why, ".", call. = FALSE)
  }
  if (length(arms) != 1L) refuse(sprintf("%d arms", length(arms)))
  a <- arms[[1L]]
  fam <- .canonical_family(a$family)
  if (!is.null(a$likelihood)) refuse("a model-supplied likelihood")
  if (!fam %in% .JOINT_AGHQ_FAMILIES) refuse(sprintf("family '%s'", fam))
  if (a$n_re_groups > 0L) refuse("an arm-level random intercept")
  if (!is.null(a$weights) && any(a$weights != 1)) refuse("observation weights")
  if (!identical(cell_coupling, "separable") || isTRUE(a$coupled)) {
    refuse("a cell-coupled likelihood")
  }
  if (isTRUE(cp$has_copy)) refuse("a copied block")
  if (any(lengths(phi_axes %||% list()) > 1L)) refuse("a dispersion axis")
  if (!length(prepared)) refuse("no latent block")
  N <- length(a$y)
  idx <- NULL; ng <- NULL
  Z <- list(); nc <- integer(0); full <- logical(0); cols <- list()
  for (b in seq_along(prepared)) {
    spec <- blocks_spec[[b]]
    type <- tolower(spec$type %||% "")
    if (!type %in% c("iid", "miid")) refuse(sprintf("a '%s' block", type))
    gi <- as.integer(spec$obs_idx[[1L]])
    gn <- as.integer(if (type == "iid") spec$n_units else spec$n_groups)
    if (is.null(idx)) { idx <- gi; ng <- gn }
    if (!identical(gi, idx) || !identical(gn, ng)) {
      refuse("random effects over more than one grouping factor")
    }
    span <- seq.int(axis_offsets[b] + 1L, length.out = axis_offsets[b + 1L] -
                      axis_offsets[b])
    bare <- .axis_bare(axis_names[span])
    if (type == "iid") {
      j <- match(TRUE, bare %in% .JOINT_FIELD_SD_AXES)
      if (is.na(j) || length(span) != 1L) refuse("an iid block without one scale axis")
      w <- if (is.null(spec$svc_weight)) rep(1, N) else spec$svc_weight[[1L]]
      Z[[b]] <- matrix(as.numeric(w), N, 1L)
      nc <- c(nc, 1L); full <- c(full, FALSE)
      cols[[b]] <- list(col = span[j], kind = bare[j])
    } else {
      p <- as.integer(spec$n_fields)
      at <- match(.mcar_logchol_names(p), bare)
      if (anyNA(at) || length(span) != length(at)) {
        refuse("an miid block whose axes are not its log-Cholesky coordinates")
      }
      Z[[b]] <- vapply(spec$field_weight, function(f) as.numeric(f[[1L]]),
                       numeric(N))
      dim(Z[[b]]) <- c(N, p)
      nc <- c(nc, p); full <- c(full, p > 1L)
      cols[[b]] <- list(col = span[at], kind = "logchol")
    }
  }
  bp_prec <- a$beta_prior_prec
  p_fix <- ncol(a$X)
  prior <- if (length(bp_prec)) list(
    mean = rep_len(as.numeric(a$beta_prior_mean %||% 0), p_fix),
    sd = 1 / sqrt(rep_len(as.numeric(bp_prec), p_fix)))
  off <- as.numeric(a$offset %||% numeric(0))
  orc <- cpp_glmm_oracle_make(
    fam, a$phi, as.numeric(a$y), as.numeric(a$n_trials), as.matrix(a$X),
    do.call(cbind, Z), idx, ng,
    offset = if (length(off) && any(off != 0)) off else numeric(0))
  # Every profile starts from the fixed effects of the Laplace solve at
  # Sigma = I (unit scales, zero log-Cholesky coordinates), so a cell's value
  # does not depend on which cell or thread solved it.
  pilot <- matrix(0, 1L, length(axis_names),
                  dimnames = list(NULL, as.character(axis_names)))
  for (cb in cols) if (cb$kind != "logchol") pilot[1L, cb$col] <- 1
  warm <- tryCatch(.joint_with_quiet_opts(laplace_kernel(pilot))$modes[
                     1L, seq_len(p_fix)],
                   error = function(e) NULL)
  list(n_quad = n_quad, oracle = orc, nc = nc, full = full, cols = cols,
       p_fix = p_fix, prior = prior,
       beta_init = if (length(warm) == p_fix && all(is.finite(warm))) warm
                   else rep(0, p_fix))
}

# The profile's covariance coordinates of each row of `theta_mat`.
.joint_aghq_coords <- function(plan, theta_mat) {
  tm <- as.matrix(theta_mat)
  do.call(cbind, lapply(plan$cols, function(cb) {
    v <- tm[, cb$col, drop = FALSE]
    switch(cb$kind,
           sigma = log(v),
           tau = -0.5 * log(v),
           logchol = v)
  }))
}

# The quadrature counterpart of `.joint_multi_call_factory()`: the same
# signature, returning the fields the driver reads off a cell solve. Each
# cell's fixed-effect block is the inverse of the profile curvature; the
# random-effect coordinates of `modes` are NA, since the groups were
# integrated out rather than conditioned on. A cell whose profile failed
# carries -Inf and is unconverged. The Laplace-only requests (precision, skew
# probe, debias, cheap screen) have no meaning here: the caller turns them off.
.joint_aghq_call_factory <- function(plan, n_x, n_threads) {
  force(plan); force(n_x); force(n_threads)
  function(theta_mat, x_init = NULL, store_Q = FALSE, phi_grid_per_arm = NULL,
           n_threads_outer = 1L, tile_ids = NULL, tile_pilot_cells = NULL,
           prune_tol = 0.0, x_init_per_cell = NULL, max_iter_ = NULL,
           tol_ = NULL, inner_refresh_ = NULL, compute_skew = FALSE,
           skew_idx = NULL, fixed_block = FALSE, debias = NULL, cila = NULL,
           screen_log_offset = NULL, screen_only = FALSE) {
    eta <- .joint_aghq_coords(plan, theta_mat)
    S <- nrow(eta)
    r <- cpp_aghq_profile_batch(
      eta_pts = eta, oracle = plan$oracle, nc = plan$nc, full = plan$full,
      n_quad = plan$n_quad, beta_init = plan$beta_init,
      prior_mean = plan$prior$mean, prior_sd = plan$prior$sd,
      n_threads = as.integer(max(n_threads_outer, n_threads)))
    ok <- is.finite(r$log_marginal)
    modes <- matrix(NA_real_, S, n_x)
    modes[, seq_len(plan$p_fix)] <- r$beta
    out <- list(log_marginal = as.numeric(r$log_marginal), modes = modes,
                converged = ok, n_iter = rep(NA_integer_, S))
    if (isTRUE(fixed_block)) {
      p <- plan$p_fix
      out$cov_block_per_grid <- lapply(seq_len(S), function(s) {
        if (!ok[s]) return(NULL)
        tryCatch(.spd_inverse(matrix(r$H[s, , ], p, p)),
                 error = function(e) NULL)
      })
    }
    out
  }
}
