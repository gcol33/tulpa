#' Sample an SPDE GLM via NUTS, optionally jointly over Matern hypers
#'
#' @description
#' Bayesian counterpart to [fit_spde()] / [laplace_spde_at()]. Routes the
#' SPDE-augmented GLM through tulpa's full NUTS backend via the generic
#' `LikelihoodSpec` interface. Two modes:
#'
#' * **Legacy / fixed-hyper** (`joint = FALSE`, the default): conditions on
#'   the supplied `(range, sigma)`. Samples the latent block
#'   `(beta, w_mesh, log_phi)` jointly. Useful as the inner step of an outer
#'   CCD / nested-Laplace integration over hypers.
#' * **Joint hypers** (`joint = TRUE`): samples `(beta, z_mesh, log_kappa,
#'   log_tau, log_phi)` jointly via the non-centered transform
#'   `w = L(theta)^{-T} z`. A PC prior (Fuglstad et al. 2019 JASA) on
#'   `(range, sigma)` is taken from `prior_range`, `prior_sigma`. Returns
#'   additional `w_draws`, `range_draws`, `sigma_draws`, `kappa_draws`,
#'   `tau_draws` columns/vectors on top of the raw parameter draws.
#'
#' @param y Response vector. Family-specific:
#'   * `gaussian`: any real
#'   * `poisson`, `neg_binomial_2`: non-negative integers
#'   * `binomial`: non-negative integers in `[0, n_trials]`
#'   * `gamma`: strictly positive reals
#'   * `beta`: strictly in `(0, 1)`
#' @param X Fixed-effects design matrix.
#' @param spatial A `tulpa_spatial` object from [spatial_spde()] /
#'   [spatial_spde_custom()] -- supplies the FEM matrices (C0, G1) and
#'   projection (A) plus the smoothness `nu`.
#' @param family One of `"gaussian"`, `"poisson"`, `"binomial"`,
#'   `"gamma"`, `"neg_binomial_2"`, `"beta"`.
#' @param n_trials Integer vector for `family = "binomial"` (else ignored).
#' @param joint Logical. `FALSE` (default) conditions on fixed
#'   `(range, sigma)`. `TRUE` activates joint sampling of
#'   `(log_kappa, log_tau)` with the PC prior from `prior_range`, `prior_sigma`.
#' @param range,sigma Fixed-hyper mode only. Matern range and marginal SD
#'   on the field. Default to the SPDE prior medians
#'   (`spatial$prior_range[1]`, `spatial$prior_sigma[1]`).
#' @param prior_range,prior_sigma Joint mode only. PC prior anchors as
#'   `c(value, alpha)` pairs:
#'     * `prior_range = c(r0, a_r)` encodes `P(range < r0) = a_r`
#'     * `prior_sigma = c(s0, a_s)` encodes `P(sigma > s0) = a_s`
#'   Default to the spec's anchors (`spatial$prior_range`,
#'   `spatial$prior_sigma`) when those are length-2 PC anchors.
#' @param log_kappa_init,log_tau_init Joint mode only. Initial values for
#'   the hyper slots. Default to the value implied by the PC anchor's
#'   `(r0, s0)` pair via `kappa = sqrt(8 nu) / r0`,
#'   `tau = 1 / (sqrt(4 pi) * kappa * s0)`.
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on each coefficient with SD `sd` (default
#'   `list(mean = 0, sd = 10)`).
#' @param log_phi_prior_sd Prior SD on `log(phi)`. Role of `phi` is
#'   family-specific:
#'   * `gaussian`: `phi` is the residual SD (sampled jointly)
#'   * `gamma`: `phi` is the Gamma shape (sampled jointly)
#'   * `neg_binomial_2`: `phi` is the NB size `r` (sampled jointly)
#'   * `beta`: `phi` is the Beta precision (sampled jointly)
#'   * `poisson`, `binomial`: `log_phi` is held tight and ignored downstream.
#' @param log_phi_init Starting value for `log(phi)`.
#' @param control A named list of numerical / sampler knobs (statistical
#'   arguments stay in the signature): `n_iter` (default 2000), `n_warmup`
#'   (default 1000), `max_treedepth` (default 10), `adapt_delta` (default 0.8),
#'   `seed` (`NULL` draws from the session RNG), `verbose` (default FALSE),
#'   `noncenter` (fixed-hyper only, default TRUE: sample the mesh field in a
#'   non-centered `v = L^{-T} z` parameterisation -- the same target density,
#'   a priori isotropised; ignored when `joint = TRUE`), and `mass_matrix`
#'   (NUTS metric: `"auto"` (default) picks a dense metric at `<= 200`
#'   parameters and diagonal above, capturing the fixed-effect / field
#'   cross-curvature that corrects the intercept marginal; `"diag"`, `"dense"`,
#'   `"block_diag"` force the choice).
#'
#' @return A list with `draws` (matrix `n_samples x n_params`), `means`,
#'   `phi_summary` (where applicable), `accept_prob`, `divergent`,
#'   `treedepth`, `epsilon`, `joint_hypers`, plus the supplied `spatial`
#'   spec. In `joint = TRUE` mode additionally:
#'   `w_draws` (transformed `z -> w` per draw), `range_draws`,
#'   `sigma_draws`, `kappa_draws`, `tau_draws`, and `range_summary`,
#'   `sigma_summary` (mean/median/5%/95% quantiles).
#'
#' @seealso [fit_spde()] for the Laplace counterpart and the nested-Laplace
#'   path over (range, sigma).
#'
#' @examples
#' \dontrun{
#' # SPDE-field NUTS. `spatial` is an SPDE spec built from a mesh, e.g.
#' # spatial_spde(~ x + y, df). See fit_spde() for the Laplace analogue.
#' fit <- tulpa_nuts_spde(y, X, spatial = spatial_spde(~ x + y, df))
#' }
#' @export
tulpa_nuts_spde <- function(y, X, spatial,
                            family           = c("gaussian", "poisson", "binomial",
                                                 "gamma", "neg_binomial_2", "beta"),
                            n_trials         = NULL,
                            joint            = FALSE,
                            range            = NULL,
                            sigma            = NULL,
                            prior_range      = NULL,
                            prior_sigma      = NULL,
                            log_kappa_init   = NULL,
                            log_tau_init     = NULL,
                            beta_prior       = list(mean = 0, sd = 10),
                            log_phi_prior_sd = 3,
                            log_phi_init     = 0,
                            control          = list()) {

  .check_control(control, .CONTROL_KEYS$nuts_spde, "tulpa_nuts_spde")
  family      <- match.arg(family)
  sigma_beta  <- .beta_prior_ridge_sd(beta_prior, default_sd = 10)
  # Perf / sampler knobs live in control (statistical args stay in the
  # signature).
  noncenter     <- isTRUE(control$noncenter %||% TRUE)
  mass_matrix   <- match.arg(control$mass_matrix %||% "auto",
                             c("auto", "diag", "dense", "block_diag"))
  n_iter        <- as.integer(control$n_iter %||% 2000L)
  n_warmup      <- as.integer(control$n_warmup %||% 1000L)
  max_treedepth <- as.integer(control$max_treedepth %||% 10L)
  adapt_delta   <- control$adapt_delta %||% 0.8
  seed          <- control$seed
  verbose       <- isTRUE(control$verbose)
  if (!inherits(spatial, "tulpa_spatial") || !identical(spatial$type, "spde")) {
    stop("`spatial` must be an SPDE tulpa_spatial object ",
         "(see `spatial_spde()` / `spatial_spde_custom()`).",
         call. = FALSE)
  }
  # Fractional nu uses the operator-based rational field (BRASIL roots).
  # The fixed-hyper sampler (joint = FALSE) precomputes the rational precision
  # Q = Pl' C^{-1} Pl in R (.spde_assemble_at) and samples the auxiliary weights
  # x ~ N(0, Q^{-1}); the field is u = Pr x. Joint-over-hypers fractional NUTS
  # stays gated -- the rational roots are not differentiable in kappa, so the
  # non-centered z -> w transform has no fractional analogue.
  frac_fixed_hyper <- FALSE
  if (.spde_nu_is_fractional(spatial$nu)) {
    if (isTRUE(joint)) {
      stop("Joint-hyper NUTS for a fractional-nu SPDE is not supported: ",
           "the rational roots are not differentiable in ",
           "kappa. Use joint = FALSE (fixed range/sigma) for fractional NUTS, ",
           "the nested-Laplace path over (range, sigma) for full hyper ",
           "integration, or an integer nu for joint NUTS.", call. = FALSE)
    }
    frac_fixed_hyper <- TRUE
  }

  y <- as.numeric(y)
  N <- length(y)
  X <- as.matrix(X)
  if (nrow(X) != N) {
    stop("nrow(X) must equal length(y).", call. = FALSE)
  }

  joint <- isTRUE(joint)

  if (!joint) {
    # Legacy fixed-hyper mode. (range, sigma) are scalars; PC anchors and
    # log_kappa/log_tau inits are not in the parameter vector.
    if (is.null(range)) range <- spatial$prior_range[1]
    if (is.null(sigma)) sigma <- spatial$prior_sigma[1]
    if (!is.numeric(range) || length(range) != 1L || !is.finite(range) || range <= 0) {
      stop("`range` must be a positive scalar.", call. = FALSE)
    }
    if (!is.numeric(sigma) || length(sigma) != 1L || !is.finite(sigma) || sigma <= 0) {
      stop("`sigma` must be a positive scalar.", call. = FALSE)
    }
    prior_range_0     <- -1.0
    prior_range_alpha <- -1.0
    prior_sigma_0     <- -1.0
    prior_sigma_alpha <- -1.0
    log_kappa_init_v  <- 0.0
    log_tau_init_v    <- 0.0
  } else {
    # Joint mode: build PC anchors from prior_range / prior_sigma or pull
    # from the spec. Each must be a length-2 c(value, alpha) pair.
    if (is.null(prior_range)) prior_range <- spatial$prior_range
    if (is.null(prior_sigma)) prior_sigma <- spatial$prior_sigma
    if (!is.numeric(prior_range) || length(prior_range) != 2L ||
        any(!is.finite(prior_range)) || prior_range[1] <= 0 ||
        prior_range[2] <= 0 || prior_range[2] >= 1) {
      stop("`prior_range` must be a length-2 numeric c(value, alpha) with ",
           "value > 0 and alpha in (0, 1).", call. = FALSE)
    }
    if (!is.numeric(prior_sigma) || length(prior_sigma) != 2L ||
        any(!is.finite(prior_sigma)) || prior_sigma[1] <= 0 ||
        prior_sigma[2] <= 0 || prior_sigma[2] >= 1) {
      stop("`prior_sigma` must be a length-2 numeric c(value, alpha) with ",
           "value > 0 and alpha in (0, 1).", call. = FALSE)
    }
    prior_range_0     <- prior_range[1]
    prior_range_alpha <- prior_range[2]
    prior_sigma_0     <- prior_sigma[1]
    prior_sigma_alpha <- prior_sigma[2]

    nu_used <- spatial$nu
    # Init at the PC prior median for each marginal, not at the tail
    # anchor (which deliberately sits deep in the prior's tail by
    # construction -- starting NUTS there triggers wild step-size adaptation).
    #
    # Marginal PC priors:
    #   pi(range) = (lambda_r/2) range^{-3/2} exp(-lambda_r * range^{-1/2})
    #   pi(sigma) = lambda_s exp(-lambda_s * sigma)
    # with lambda_r = -log(alpha_r) * sqrt(r0), lambda_s = -log(alpha_s)/s0.
    # u = range^{-1/2} ~ Exp(lambda_r), so median(u) = log(2)/lambda_r and
    # median(range) = (lambda_r / log(2))^2. median(sigma) = log(2)/lambda_s.
    if (is.null(log_kappa_init) || is.null(log_tau_init)) {
      lambda_r <- -log(prior_range_alpha) * sqrt(prior_range_0)
      lambda_s <- -log(prior_sigma_alpha) / prior_sigma_0
      range_med <- (lambda_r / log(2))^2
      sigma_med <- log(2) / lambda_s
      kappa_med <- sqrt(8 * nu_used) / range_med
      tau_med   <- 1.0 / (sqrt(4 * pi) * kappa_med * sigma_med)
      if (is.null(log_kappa_init)) log_kappa_init <- log(kappa_med)
      if (is.null(log_tau_init))   log_tau_init   <- log(tau_med)
    }
    log_kappa_init_v <- as.numeric(log_kappa_init)
    log_tau_init_v   <- as.numeric(log_tau_init)

    # range / sigma get passed-through as numeric placeholders so the Rcpp
    # signature stays positional; they are ignored when joint_hypers=TRUE.
    range <- if (is.null(range)) prior_range_0 else range
    sigma <- if (is.null(sigma)) prior_sigma_0 else sigma
  }

  kappa    <- sqrt(8 * spatial$nu) / range
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma)
  # Operator order alpha = nu + d/2 (d = 2). For integer nu this is an
  # int; for fractional nu the rational expansion handles non-integer alpha
  # downstream, and the int passed here only labels the closest integer
  # spde construction for the legacy fixed-hyper path.
  alpha    <- as.integer(round(spatial$nu)) + 1L
  rat      <- rational_spde_coefficients(spatial$nu)

  if (is.null(n_trials)) n_trials <- rep(1L, N)
  n_trials <- as.integer(n_trials)

  # Fractional fixed-hyper: assemble the rational precision Q and rational
  # observation map A_eff = A Pr at (range, sigma) on the non-orphan submesh, and
  # pass them precomputed (the legacy poles/weights QBuilder cannot reproduce the
  # BRASIL field). The sampler then draws the auxiliary weights x (named "w") of
  # length n_sub; the field u = Pr x is reconstructed below. Integer / joint runs
  # keep the FEM construction.
  A_x_use <- spatial$A_x; A_i_use <- spatial$A_i; A_p_use <- spatial$A_p
  n_mesh_use <- spatial$n_mesh
  C0_use <- spatial$C0_diag
  G1_x_use <- spatial$G1_x; G1_i_use <- spatial$G1_i; G1_p_use <- spatial$G1_p
  Q_pre_x <- NULL; Q_pre_i <- NULL; Q_pre_p <- NULL
  frac_asm <- NULL
  if (frac_fixed_hyper) {
    order    <- spatial$rational_order %||% 2L
    frac_asm <- .spde_assemble_at(spatial, range, sigma, order = order)
    n_sub    <- length(frac_asm$keep)
    Aeff     <- as(frac_asm$A_eff, "CsparseMatrix")
    # Full both-triangle storage: .spde_assemble_at returns Q as a symmetric
    # dsCMatrix (one stored triangle), but the C++ SPDE prior loop sums over
    # stored CSC entries and assumes full symmetric storage (it matches the
    # SpdeQBuilder convention). Coerce to generalMatrix so both triangles are
    # passed, as the Laplace path already does.
    Qg       <- as(frac_asm$Q, "generalMatrix")
    fem      <- .spde_fem_matrices(spatial)
    Gk       <- as(fem$G[frac_asm$keep, frac_asm$keep, drop = FALSE], "CsparseMatrix")
    A_x_use <- Aeff@x; A_i_use <- Aeff@i; A_p_use <- Aeff@p
    n_mesh_use <- n_sub
    C0_use  <- as.numeric(spatial$C0_diag)[frac_asm$keep]
    G1_x_use <- Gk@x; G1_i_use <- Gk@i; G1_p_use <- Gk@p
    Q_pre_x <- Qg@x; Q_pre_i <- Qg@i; Q_pre_p <- Qg@p
  }

  # Resolve the metric (after n_mesh_use is final). "auto" -> dense below the
  # DENSE_MAX_PARAMS = 200 ceiling (captures beta/field cross-curvature),
  # diagonal above it. n_params = p + field block + log_phi (+2 hyper slots
  # when joint).
  mass_code <- switch(mass_matrix, diag = 0L, dense = 1L, block_diag = 2L,
                      auto = -1L)
  if (mass_code == -1L) {
    n_params  <- ncol(X) + n_mesh_use + 1L + if (joint) 2L else 0L
    mass_code <- if (n_params <= 200L) 1L else 0L
  }

  res <- cpp_tulpa_fit_spde_nuts(
    y_r              = y,
    n_trials_r       = n_trials,
    X_r              = X,
    A_x              = A_x_use,
    A_i              = A_i_use,
    A_p              = A_p_use,
    n_obs            = N,
    n_mesh           = n_mesh_use,
    C0_diag          = C0_use,
    G1_x             = G1_x_use,
    G1_i             = G1_i_use,
    G1_p             = G1_p_use,
    kappa            = kappa,
    tau_spde         = tau_spde,
    family           = family,
    alpha            = alpha,
    nu               = as.numeric(spatial$nu),
    sigma_beta       = sigma_beta,
    log_phi_prior_sd = log_phi_prior_sd,
    log_phi_init     = log_phi_init,
    n_iter           = as.integer(n_iter),
    n_warmup         = as.integer(n_warmup),
    max_treedepth    = as.integer(max_treedepth),
    adapt_delta      = adapt_delta,
    seed             = as.integer(seed %||% sample.int(.Machine$integer.max, 1L)),
    verbose          = isTRUE(verbose),
    # Rational poles/weights feed the legacy integer-mesh joint path only; the
    # fractional fixed-hyper path passes Q precomputed (Q_precomp_*) instead.
    rational_poles_nullable   = if (!rat$is_integer && !frac_fixed_hyper) rat$poles   else NULL,
    rational_weights_nullable = if (!rat$is_integer && !frac_fixed_hyper) rat$weights else NULL,
    joint_hypers      = joint,
    prior_range_0     = prior_range_0,
    prior_range_alpha = prior_range_alpha,
    prior_sigma_0     = prior_sigma_0,
    prior_sigma_alpha = prior_sigma_alpha,
    log_kappa_init    = log_kappa_init_v,
    log_tau_init      = log_tau_init_v,
    Q_precomp_x       = Q_pre_x,
    Q_precomp_i       = Q_pre_i,
    Q_precomp_p       = Q_pre_p,
    # Non-centering is a fixed-hyper option; the joint path is intrinsically
    # non-centered, so this flag only takes effect when joint = FALSE.
    noncenter         = isTRUE(noncenter) && !joint,
    mass_matrix       = mass_code
  )

  res$range   <- range
  res$sigma   <- sigma
  res$spatial <- spatial
  res$joint   <- joint

  # Fractional fixed-hyper: the auxiliary weights x on the submesh are either
  # the sampled block (centered) or the transformed field v = L^{-T} z exposed
  # as `w_draws` (non-centered). Reconstruct the field u = Pr x and
  # scatter to the full mesh (orphan nodes carry zero field), so `field_draws`
  # is comparable to the integer path's field draws.
  if (frac_fixed_hyper) {
    if (!is.null(res$w_draws)) {
      x_draws <- res$w_draws
    } else {
      w_cols  <- res$p + seq_len(n_mesh_use)
      x_draws <- res$draws[, w_cols, drop = FALSE]
    }
    Pr      <- frac_asm$Pr
    field_sub <- as.matrix(Pr %*% t(x_draws))            # n_sub x n_sample
    field_full <- matrix(0, nrow(res$draws), spatial$n_mesh)
    field_full[, frac_asm$keep] <- t(field_sub)
    colnames(field_full) <- paste0("u[", seq_len(spatial$n_mesh), "]")
    res$field_draws <- field_full
    res$keep        <- frac_asm$keep
    res$rational    <- TRUE
  } else if (!is.null(res$w_draws) && !joint) {
    # Integer fixed-hyper non-centered: the transformed field v = L^{-T} z is
    # the mesh field directly. Expose it as field_draws for parity with the
    # fractional path (the raw `draws` block is z).
    field_full <- res$w_draws
    colnames(field_full) <- paste0("u[", seq_len(spatial$n_mesh), "]")
    res$field_draws <- field_full
  }

  # phi summary on the natural scale, for families that sample it. The
  # meaning of phi is family-specific: residual SD (gaussian), shape
  # (gamma), size r (neg_binomial_2), precision (beta).
  if (family %in% c("gaussian", "gamma", "neg_binomial_2", "beta")) {
    phi_draws <- exp(res$draws[, "log_phi"])
    res$phi_summary <- c(
      mean   = mean(phi_draws),
      median = stats::median(phi_draws),
      q05    = unname(stats::quantile(phi_draws, 0.05)),
      q95    = unname(stats::quantile(phi_draws, 0.95))
    )
  }

  if (joint) {
    res$range_summary <- c(
      mean   = mean(res$range_draws),
      median = stats::median(res$range_draws),
      q05    = unname(stats::quantile(res$range_draws, 0.05)),
      q95    = unname(stats::quantile(res$range_draws, 0.95))
    )
    res$sigma_summary <- c(
      mean   = mean(res$sigma_draws),
      median = stats::median(res$sigma_draws),
      q05    = unname(stats::quantile(res$sigma_draws, 0.05)),
      q95    = unname(stats::quantile(res$sigma_draws, 0.95))
    )
  }

  res
}
