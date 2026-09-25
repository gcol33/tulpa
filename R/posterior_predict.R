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
# n_obs). Built from the $re_design rows tulpa() attaches. NULL when the fit
# carries no formula RE terms or no design.
#' @keywords internal
.tulpa_re_map <- function(object) {
  des <- object$re_design
  if (is.null(des) || !length(des)) return(NULL)
  X <- object$model_matrix
  if (is.null(X)) return(NULL)
  n_obs <- nrow(X)

  i_idx <- integer(0); j_idx <- integer(0); x_val <- numeric(0)
  offset <- 0L
  for (rt in des) {
    nc <- rt$n_coefs %||% 1L
    Z  <- cbind(
      if (isTRUE(rt$has_intercept)) rep(1, n_obs),
      rt$slope_matrix
    )
    if (is.null(Z) || ncol(Z) != nc) return(NULL)
    for (cc in seq_len(nc)) {
      i_idx <- c(i_idx, offset + (rt$group_idx - 1L) * nc + cc)
      j_idx <- c(j_idx, seq_len(n_obs))
      x_val <- c(x_val, Z[, cc])
    }
    offset <- offset + (rt$n_groups %||% max(rt$group_idx)) * nc
  }
  Matrix::sparseMatrix(i = i_idx, j = j_idx, x = x_val,
                       dims = c(offset, n_obs))
}

# Point estimate of the RE coefficient vector (length n_re) for fits without
# RE draws: the tail of the Laplace $mode or the $means vector past the fixed
# block. NULL when neither carries the RE block.
#' @keywords internal
.tulpa_re_point <- function(object, n_re) {
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
  H <- object$H_latent
  mode <- object$mode
  nf <- object$n_fixed %||% 0L
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

# Random-effect coefficient draws (S x n_re, group-major / coef-within-group)
# paired row for row with the fixed-effect draws `beta` that
# `.fixed_coef_draws()` returned alongside `keep`. The one resolution order
# posterior_predict() and tulpa_simulate(theta = fit) share: the fit's own RE
# draws (rows `keep`), else a Laplace / EB conditional draw given `beta`, else
# the point value repeated on every row. NULL when the fit carries none of the
# three, so a caller decides what a missing RE block means rather than reading
# it as zero (gcol33/tulpa#891).
#' @keywords internal
.tulpa_re_coef_draws <- function(object, beta, keep, n_re) {
  rd <- .re_coef_draws(object)
  if (!is.null(rd) && ncol(rd) == n_re) {
    if (!is.null(keep)) rd <- rd[keep, , drop = FALSE]
    return(rd)
  }
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
# The (n_field, Z_field) design mapping an intrinsic/proper-CAR or BYM2 areal
# field's own raw latent to its eta contribution -- the same map each
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
  if (type == "bym2") {
    scale_factor <- spatial$scale_factor %||% 1.0
    d_phi   <- sqrt(0.5 + 1e-10) * scale_factor
    d_theta <- sqrt(1 - 0.5 + 1e-10)
    Z_ind   <- .field_design_Z(spatial$spatial_idx, n_units, n_obs)
    return(list(n_field = 2L * n_units, Z_field = cbind(d_phi * Z_ind, d_theta * Z_ind)))
  }
  list(n_field = n_units,
       Z_field = .field_design_Z(spatial$spatial_idx, n_units, n_obs))
}

# Per-draw eta contribution of a Polya-Gamma Gibbs field, read off the
# `phi_spatial[k]` (icar / car) or `phi_spatial[k]` + `theta_spatial[k]`
# (bym2) columns `.pg_as_chain()` (`R/fit_gibbs.R`) already stores in
# `$draws` alongside the fixed effects -- the kernel's own scaled field
# state, so no `d_fac` is reapplied. `keep` subsets rows to match whatever
# subsample `.fixed_coef_draws()` took of the same chain. NULL for any
# spatial type the Gibbs sampler does not carry a field column for (gp /
# multiscale_gp are sampled but not covered here; the caller adds nothing).
#' @keywords internal
.gibbs_field_eta_contrib <- function(object, keep, n_obs) {
  spatial <- object$spatial
  if (is.null(spatial)) return(NULL)
  type <- tolower(spatial$type %||% "")
  dn <- colnames(object$draws)
  if (!type %in% c("icar", "car", "bym2") || is.null(dn)) return(NULL)

  phi_cols <- grep("^phi_spatial\\[", dn)
  if (!length(phi_cols)) return(NULL)
  n_units <- length(phi_cols)
  Z <- .field_design_Z(spatial$spatial_idx, n_units, n_obs)
  U <- object$draws[keep, phi_cols, drop = FALSE]

  if (type == "bym2") {
    th_cols <- grep("^theta_spatial\\[", dn)
    if (length(th_cols) != n_units) return(NULL)
    U <- U + object$draws[keep, th_cols, drop = FALSE]
  }
  # Coerce back to a base matrix: Matrix is an Imports, not attached for a
  # caller of the package, and `eta` downstream (and every plain colMeans() /
  # arithmetic a user runs on a posterior_predict() draws matrix) expects one.
  as.matrix(U %*% Matrix::t(Z))
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
  if (is.character(object$family) && length(object$family) == 1L) {
    return(list(family = object$family, n_trials = object$n_trials,
                phi = object$phi, phi2 = object$phi2, y = object$y, arm = NULL))
  }
  one <- .tulpa_single_arm(object)
  if (!is.null(one) && is.character(one$spec$family) &&
      length(one$spec$family) == 1L) {
    a <- one$spec
    return(list(family = a$family, n_trials = a$n_trials,
                phi = a$phi, phi2 = a$phi2, y = a$y, arm = one$arm))
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


# `"coefficients"`: everything else. The fit carries its fixed effects (and at
# most the formula random effects and an SPDE field) rather than its linear
# predictor, so eta is assembled from those.
#' @keywords internal
.tulpa_linpred_source <- function(object) {
  if (is.list(object$model_inputs) && is.matrix(object$draws) &&
      nrow(object$draws) > 0L) {
    return("sampler_model")
  }
  if (is.matrix(object$fitted_eta) && !is.null(object$weights) &&
      length(object$weights) == nrow(object$fitted_eta)) {
    return("grid_mixture")
  }
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
# nested-Laplace fit, and otherwise the fixed effects plus the offset, the
# formula random effects and a posterior-mean SPDE field. Fixed effects on that
# last source come from `.fixed_coef_draws()`. At `newdata` the prediction is
# population-level (fixed effects and the offset only), matching predict().
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
                             synth_seed = NULL) {
  if (!is.null(synth_seed)) {
    .preserve_seed_in_frame()
    set.seed(as.integer(synth_seed))
  }
  source <- if (is.null(newdata)) .tulpa_linpred_source(object) else "coefficients"
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

  if (is.null(newdata)) {
    M <- .tulpa_re_map(object)
    if (!is.null(M)) {
      rd <- .tulpa_re_coef_draws(object, beta, keep, nrow(M))
      if (!is.null(rd)) {
        eta <- eta + as.matrix(rd %*% M)
      } else {
        message("posterior_predict(): no random-effect draws or point ",
                "values available; replicates are population-level.")
      }
    }

    if (identical(object$spatial$type, "spde") &&
        !is.null(object$spatial_effects) && !is.null(object$spatial$A)) {
      eta <- sweep(eta, 2,
                   as.numeric(object$spatial$A %*% object$spatial_effects), "+")
    } else if (!is.null(object$spatial) && !is.null(object$mode)) {
      # Conditional-Laplace areal fit (icar / car / car_proper / bym2): the
      # field's raw latent is the TAIL of the fit's own mode (`[beta, re,
      # field]`, matching every `.marginal_H_beta_*` areal helper's own
      # layout, `R/marginal_se_spatial.R`), so it is read off exactly as it
      # entered eta at the mode -- a posterior-mean field, like the SPDE
      # branch above (gcol33/tulpa#795).
      fd <- .areal_field_design(object$spatial, nrow(D$X))
      if (!is.null(fd) && length(object$mode) >= fd$n_field) {
        u <- object$mode[(length(object$mode) - fd$n_field + 1L):length(object$mode)]
        eta <- sweep(eta, 2, as.numeric(fd$Z_field %*% u), "+")
      }
    } else if (is.matrix(object$draws) && !is.null(object$spatial)) {
      # Polya-Gamma Gibbs fit: the field is actually SAMPLED (a
      # `phi_spatial[k]` / `theta_spatial[k]` column per unit alongside the
      # fixed effects in `$draws`), so every draw gets its own field value
      # instead of a posterior mean (gcol33/tulpa#795).
      fc <- .gibbs_field_eta_contrib(object, keep %||% seq_len(nrow(object$draws)),
                                     nrow(D$X))
      if (!is.null(fc)) eta <- eta + fc
    } else if (!is.null(object$field_eta_contrib)) {
      # Inline spatial() / temporal() joint field fit: the field's own
      # weighted-mode posterior mean, read back per observation at fit time
      # (`.bar_field_fit_core()`, `R/spatial_field.R`), since the joint
      # multi-block driver behind these fits carries no single-cell
      # `fitted_eta` for `.tulpa_linpred_source()` to read (gcol33/tulpa#795).
      eta <- sweep(eta, 2, object$field_eta_contrib, "+")
    }
  }

  if (!is.null(zi_nm)) {
    attr(eta, "logit_zi") <- beta[, zi_nm, drop = FALSE] %*%
      t(D$X_zi[, zi_nm, drop = FALSE])
  }
  eta
}

# The NUTS store (`src/hmc_nuts_chain_iter_store.h`) overwrites each
# non-centered GP / SVC / multiscale-GP block's stored slice with the
# reconstructed field `w` before the draw is written out (`q` itself stays
# `z` for sampling) -- no other ModelData kernel does this, so a "hmc" fit's
# stored draws already hold `w` where every other backend's still hold `z`.
# Re-running such a draw through `initialize_generic_state()` at the spec's
# own `gp_parameterization` / `svc_parameterization` / `msgp_parameterization`
# therefore applies the non-centered forward transform a SECOND time, to a
# slice that is no longer `z` (gcol33/tulpa#822). Reading the stored draws
# back as centered undoes exactly that: the forward transform is the
# identity on an already-centered field, matching what the store wrote.
#' @keywords internal
.stored_draw_field_specs <- function(object, mi) {
  spatial_spec <- mi$spatial_spec
  svc_spec     <- mi$svc_spec
  if (!identical(object$backend, "hmc")) {
    return(list(spatial_spec = spatial_spec, svc_spec = svc_spec))
  }
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
  list(spatial_spec = spatial_spec, svc_spec = svc_spec)
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
  fs <- .stored_draw_field_specs(object, mi)
  eta <- cpp_tulpa_glmm_eta_draws(
    draws = D, y = mi$y, n_trials = mi$n_trials, X = mi$X,
    family = mi$family, phi = mi$phi, sigma_beta = mi$sigma_beta,
    offset_nullable = mi$offset, re_spec = mi$re_spec,
    spatial_spec = fs$spatial_spec, temporal_spec = mi$temporal_spec,
    sigma_re_scale = mi$sigma_re_scale, phi2 = mi$phi2,
    svc_spec = fs$svc_spec, tvc_spec = mi$tvc_spec, zi_spec = mi$zi_spec)
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
#'     estimated variance components). A conditional-Laplace spatial fit holds
#'     its random effects and areal field at their posterior mode, which
#'     over-disperses the linear predictor wherever the intercept is aliased
#'     with them; an SPDE field enters at its posterior mean.
#' }
#' At `newdata` the prediction is population level (random effects at zero),
#' matching [predict.tulpa_fit()].
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

#' @param newdata Optional data frame of covariates to predict at. Population
#'   level (fixed effects only); `NULL` (default) replicates at the training
#'   data with every fitted component included.
#' @param ndraws Number of posterior draws to use. Defaults to all stored
#'   draws, or 400 on the draw-free Laplace tier.
#' @param n_trials Binomial / beta-binomial trial counts for the replicates.
#'   Defaults to the training trials when `newdata` is `NULL`, else 1.
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @rdname posterior_predict
#' @export
posterior_predict.tulpa_fit <- function(object, newdata = NULL, ndraws = NULL,
                                        n_trials = NULL, seed = NULL, ...) {
  proc <- .tulpa_response_process(object, "posterior_predict")
  .seed_scoped(seed)

  eta <- .tulpa_eta_draws(object, newdata = newdata, ndraws = ndraws)
  logit_zi <- attr(eta, "logit_zi")
  if (is.null(n_trials)) {
    n_trials <- if (is.null(newdata)) proc$n_trials else NULL
  }
  # One dispersion per replicate: the cell's own where the fit integrated a
  # dispersion axis, the fit's scalar repeated otherwise.
  phi <- .tulpa_phi_draws(object, proc, attr(eta, "cells"), nrow(eta))

  yrep <- matrix(NA_real_, nrow(eta), ncol(eta))
  for (s in seq_len(nrow(eta))) {
    yrep[s, ] <- .response_sample(eta[s, ],
                                  if (!is.null(logit_zi)) logit_zi[s, ],
                                  proc$family, n_trials = n_trials,
                                  phi = phi[s], phi2 = proc$phi2)
  }
  yrep
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
