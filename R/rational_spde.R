#' Rational Approximation Coefficients for a Fractional SPDE
#'
#' Returns the coefficient descriptor for an SPDE Matern field of operator order
#' `alpha = nu + 1` (in 2D). For integer `nu` the construction is exact and no
#' rational approximation is needed. For fractional `nu` it returns the
#' rational-SPDE roots from the BRASIL best-rational approximation (see Details).
#'
#' @param nu Matern smoothness parameter. A non-negative number; may be
#'   fractional (e.g. 0.5, 1.5, 2.5).
#' @param m Rational approximation order (number of numerator / denominator
#'   factors) used for fractional `nu`. Higher `m` lowers the approximation
#'   error of the field's spectral density. Default 4.
#' @param lambda_range The generalized-eigenvalue spectrum `c(l_min, l_max)` of
#'   the FEM operator `CiL = C^{-1}(kappa^2 C + G)` over which the rational
#'   approximation is fitted. The approximation acts on the normalized interval
#'   `[l_min / l_max, 1]`. Used for fractional `nu` only.
#'
#' @return For integer `nu`, a list with `is_integer = TRUE`, the operator order
#'   `alpha`, `beta = alpha / 2`, `m = 0`, and empty `poles` / `weights`. For
#'   fractional `nu`, a list with `is_integer = FALSE`, `alpha`, `beta`, the
#'   rSPDE roots `rb` (denominator factors, drive `Pl`) and `rc` (numerator
#'   factors, drive `Pr`), the integer power `m_beta`, the remaining fractional
#'   exponent `beta_rem`, the scale constant `scale`, the rational order `m`, and
#'   the approximation `error`.
#'
#' @details
#' For integer `nu` (0, 1, 2, ...) the operator order `alpha = nu + 1` is an
#' integer and the precision is assembled directly from integer powers of the
#' FEM operator `L = kappa^2 C + G` -- an exact construction.
#'
#' For fractional `nu` the field uses the operator-based rational SPDE
#' approximation (Bolin & Kirchner 2020). With `beta = alpha / 2` and
#' `m_beta = max(1, floor(beta))`, the field-mode variance of the assembled
#' precision `Q = Pl' C^{-1} Pl` (field `u = Pr x`, `x ~ N(0, Q^{-1})`) tracks
#' the Matern spectral density `l^{-2 beta}` when
#' `prod(1 - l rc) / (l^{m_beta - 1} prod(1 - l rb))` approximates `l^{-beta_rem}`
#' on the scaled spectrum, `beta_rem = beta - (m_beta - 1)`. The roots come from
#' the degree-`(m, m)` best uniform (minimax) rational approximation of
#' `x^{-beta_rem}`, computed by the BRASIL algorithm (Hofreither 2021); its
#' numerator zeros map to `rc = 1 / zero` and denominator poles to `rb = 1 / pole`.
#' The field assembly from these roots is `.spde_rational_assemble()`; it is
#' validated against the Matern spectral density in `test-spde-rational.R`.
#'
#' @references
#' Bolin, D. & Kirchner, K. (2020). The rational SPDE approach for Gaussian
#' random fields with general smoothness. Journal of Computational and Graphical
#' Statistics, 29(2), 274-285.
#'
#' Hofreither, C. (2021). An algorithm for best rational approximation based on
#' barycentric rational interpolation. Numerical Algorithms, 88, 365-388.
#'
#' @keywords internal
rational_spde_coefficients <- function(nu, m = 4L, lambda_range = c(1e-4, 1e4)) {
  .validate_spde_nu(nu)

  alpha <- nu + 1  # d/2 = 1 in 2D
  beta  <- alpha / 2

  if (abs(alpha - round(alpha)) < 1e-10) {
    return(list(
      poles = numeric(0), weights = numeric(0),
      alpha = alpha, beta = beta, m = 0L, is_integer = TRUE
    ))
  }

  if (!is.numeric(lambda_range) || length(lambda_range) != 2L ||
      any(!is.finite(lambda_range)) || lambda_range[1] <= 0 ||
      lambda_range[2] <= lambda_range[1]) {
    stop("`lambda_range` must be a positive increasing pair c(l_min, l_max).",
         call. = FALSE)
  }
  ratio <- lambda_range[1] / lambda_range[2]
  rr <- .spde_rational_roots(order = as.integer(m), beta = beta,
                             spectrum_ratio = ratio)
  list(
    rb = rr$rb, rc = rr$rc, scale = rr$scale,
    m_beta = rr$m_beta, beta_rem = rr$beta_rem, error = rr$error,
    alpha = alpha, beta = beta, m = as.integer(m), is_integer = FALSE
  )
}

#' Validate the SPDE smoothness parameter
#'
#' `nu` must be a single finite non-negative number. Integer `nu` (0, 1, 2, ...)
#' gives an exact FEM construction; fractional `nu` uses the rational SPDE
#' approximation (see [rational_spde_coefficients()] and gcol33/tulpa#71).
#'
#' @param nu Candidate smoothness parameter.
#' @return Invisibly `TRUE`; raises an error on an invalid `nu`.
#' @keywords internal
.validate_spde_nu <- function(nu) {
  if (!is.numeric(nu) || length(nu) != 1L || !is.finite(nu)) {
    stop("SPDE `nu` must be a single finite number.", call. = FALSE)
  }
  if (nu < 0) {
    stop("SPDE `nu` must be non-negative.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Is an SPDE smoothness `nu` fractional (rational path) or integer (exact path)?
#'
#' @param nu Matern smoothness; `alpha = nu + 1` in 2D.
#' @return `TRUE` when `alpha` is non-integer (the rational SPDE path).
#' @keywords internal
.spde_nu_is_fractional <- function(nu) {
  alpha <- nu + 1
  abs(alpha - round(alpha)) > 1e-10
}

#' Build the sparse FEM stiffness / projector from a spatial_spde spec
#'
#' Prefers the stored `Matrix` objects (`spatial$G`, `spatial$A`); falls back to
#' rebuilding them from the pre-extracted CSC slots so a spec carrying only the
#' slots still assembles.
#' @keywords internal
.spde_fem_matrices <- function(spatial) {
  n_mesh <- spatial$n_mesh
  G <- spatial$G
  if (is.null(G)) {
    G <- Matrix::sparseMatrix(
      i = spatial$G1_i, p = spatial$G1_p, x = spatial$G1_x,
      dims = c(n_mesh, n_mesh), index1 = FALSE
    )
  }
  A <- spatial$A
  if (is.null(A)) {
    n_obs <- length(spatial$A_p) - 1L
    A <- Matrix::sparseMatrix(
      i = spatial$A_i, p = spatial$A_p, x = spatial$A_x,
      dims = c(n_obs, n_mesh), index1 = FALSE
    )
  }
  list(G = as(G, "CsparseMatrix"), A = as(A, "CsparseMatrix"))
}

# Number of probe vectors for the marginal-variance normalization, and the seed
# for the (fixed across the grid) Gaussian probe matrix. A fixed probe set makes
# the normalization constant a smooth, deterministic function of (kappa, sigma),
# so the nested-grid marginal stays smooth and the fit is reproducible.
.SPDE_VARNORM_NPROBE <- 100L
.SPDE_VARNORM_SEED   <- 20240601L

#' Mean marginal variance of the rSPDE field u = Pr x, x ~ N(0, Q^-1)
#'
#' Estimates `mean_i [Pr Q^{-1} Pr']_ii = tr(Pr Q^{-1} Pr') / n` by Hutchinson
#' probing: for `z ~ N(0, I)`, `a = Pr' z`, `Q v = a`, then `E[a' v] = tr(...)`.
#' The solve is against `Q` through the SAME sparse Cholesky the precomputed C++
#' fit uses, so the normalization is consistent with the fit even when the wide
#' rational spectrum makes `Q` ill-conditioned: a shared solver makes the implied
#' field covariance identical between the normalization and the likelihood, which
#' is what the nested `(range, sigma)` integration needs (an inconsistent solver
#' breaks the cross-grid marginal). The probe matrix is fixed across calls for a
#' deterministic, grid-smooth normalization.
#' @keywords internal
.spde_mean_marginal_var <- function(Q, Pr, C0, n_probe = .SPDE_VARNORM_NPROBE) {
  n <- length(C0)
  has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (has_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
  set.seed(.SPDE_VARNORM_SEED)
  Z <- matrix(stats::rnorm(n * n_probe), n, n_probe)
  if (!is.null(old_seed)) {
    assign(".Random.seed", old_seed, envir = .GlobalEnv)
  } else {
    rm(".Random.seed", envir = .GlobalEnv)
  }
  Achol <- Matrix::Cholesky(Q, LDL = FALSE, perm = TRUE)
  Aprz  <- Matrix::crossprod(Pr, Z)            # a = Pr' z  (n x n_probe)
  V     <- Matrix::solve(Achol, Aprz)          # Q v = a
  quad  <- colSums(as.matrix(Aprz) * as.matrix(V))   # a' v per probe
  mean(quad) / n
}

#' Assemble the fractional rSPDE precision and obs map at a given (range, sigma)
#'
#' Shared by every fractional-`nu` fit path (Laplace single-point, nested,
#' NUTS, marginal SEs). Maps `(range, sigma)` to the SPDE hyperparameters
#' `kappa = sqrt(8 nu) / range`, builds the latent precision `Q = Pl' C^{-1} Pl`
#' and field shift `Pr` (field `u = Pr x`) from the validated R oracle
#' `.spde_rational_assemble()`, then **normalizes the field to marginal variance
#' `sigma^2`**. The rational construction is correct in spectral shape but its
#' overall scale carries kappa-dependent constants (the `l_max` normalization,
#' the per-factor conditioning rescalings); without the variance normalization
#' the implied field variance is not `sigma^2` and varies with the range, which
#' biases the nested `(range, sigma)` integration. The normalization rescales
#' `Pr` (hence `A_eff = A Pr`) by `sigma / sqrt(mean marginal variance)`; `Q` is
#' untouched, so `logdet_Q` remains the correct prior normalizer for the
#' variance-normalized model.
#'
#' @param spatial A validated `spatial_spde` spec with fractional `nu`.
#' @param range,sigma Spatial range and marginal SD.
#' @param order Rational approximation order. Default 2 (the rSPDE convention),
#'   which keeps the rational precision's condition number tractable.
#' @return A list with `Q`, `Pr`, `A_eff`, `Pl` (all CSC, sized to the non-orphan
#'   submesh), `keep` (1-based indices of the retained mesh nodes), `n_mesh_full`
#'   (the full mesh size), `kappa`, `var_scale`, `l_max`, and `logdet_Q`.
#' @keywords internal
.spde_assemble_at <- function(spatial, range, sigma, order = 2L) {
  nu    <- spatial$nu
  kappa <- sqrt(8 * nu) / range

  fem <- .spde_fem_matrices(spatial)
  C0  <- as.numeric(spatial$C0_diag)
  n_full <- length(C0)

  # Drop orphan mesh nodes (zero lumped mass, e.g. mesh-extension nodes outside
  # the data hull). They carry no FEM mass -- Ci = 1/C0 would be infinite -- and
  # no observation projects onto them, so the rational operator factor Pl is
  # structurally singular over them. Assemble on the non-orphan submesh; the
  # field is identically zero at the dropped nodes.
  keep <- which(C0 > 1e-12)
  Gk   <- fem$G[keep, keep, drop = FALSE]
  Ak   <- fem$A[, keep, drop = FALSE]
  C0k  <- C0[keep]

  # tau = 1: the field scale is set by the explicit variance normalization below,
  # not the analytic tau (which is exact only for integer alpha).
  asm <- .spde_rational_assemble(
    C0 = C0k, G = as(Gk, "CsparseMatrix"),
    kappa = kappa, tau = 1, nu = nu, order = as.integer(order), d = 2
  )

  mean_var  <- .spde_mean_marginal_var(asm$Q, asm$Pr, C0k)
  var_scale <- sigma / sqrt(mean_var)
  Pr        <- as(var_scale * asm$Pr, "CsparseMatrix")

  A_eff    <- as(Ak %*% Pr, "CsparseMatrix")
  # log|Q| via the operator factor Pl (Q = Pl' C^{-1} Pl, so
  # log|Q| = 2 log|Pl| - sum log C0). cond(Pl) is the square root of cond(Q), so
  # this is far more accurate than factoring the ill-conditioned Q directly --
  # essential because the nested range identification rides on the prior
  # normalizer 0.5 log|Q|, whose tiny eigenvalues dominate a direct determinant.
  logPl    <- as.numeric(Matrix::determinant(asm$Pl, logarithm = TRUE)$modulus)
  logdet_Q <- 2 * logPl - sum(log(C0k))

  list(
    Q = as(asm$Q, "CsparseMatrix"), Pr = Pr, A_eff = A_eff,
    Pl = as(asm$Pl, "CsparseMatrix"),
    keep = keep, n_mesh_full = n_full,
    kappa = kappa, var_scale = var_scale, l_max = asm$l_max, logdet_Q = logdet_Q
  )
}

#' Numerically stable Laplace log-marginal for a fractional rSPDE at (range, sigma)
#'
#' Delegates the well-conditioned `B` / matrix-determinant-lemma marginal to C++
#' (`cpp_spde_fractional_logmarginal()`). The precision-space `0.5(log|Q| - log|H|)`
#' is corrupted in a range-dependent way by the rational precision's wide spectrum
#' (cond(Q) ~ 1e13+), so the marginal is formed through the obs-space
#' `B = (A_eff Pl^{-1}) C (A_eff Pl^{-1})' + X X'/tau_beta`, built through the
#' operator factor `Pl` (cond = sqrt cond(Q)), never an explicit `Q` inverse.
#' Gaussian is the exact conjugate marginal; non-gaussian uses the det-lemma at the
#' precomputed Laplace mode. `phi` is the Gaussian residual SD (variance `phi^2`),
#' consistent with the integer path and the engine family convention.
#'
#' @return A list with `log_marginal`, `n_iter`, `converged`, and the assembly.
#' @keywords internal
.spde_nested_logmarginal_at <- function(spatial, range, sigma, y, X, family, phi,
                                        n_trials, order, max_iter, tol, n_threads,
                                        offset, tau_beta = 1e-4) {
  n     <- length(y)
  p     <- ncol(X)
  asm   <- .spde_assemble_at(spatial, range, sigma, order = order)
  n_sub <- length(asm$keep)

  # Non-gaussian needs the Laplace mode (beta, auxiliary weights x); gaussian is
  # the exact conjugate marginal and needs none. A failed inner fit / factor gives
  # -Inf (zero nested weight) rather than crashing the grid.
  beta_hat  <- numeric(p)
  x_hat     <- numeric(n_sub)
  n_iter    <- 1L
  converged <- TRUE
  if (family != "gaussian") {
    Qg  <- as(asm$Q, "generalMatrix")
    fit <- tryCatch(cpp_laplace_fit_spde_precomputed(
      y = as.numeric(y), n_trials = as.integer(n_trials),
      X = as.matrix(X), re_idx = rep(0, n), n_re_groups = 0L, sigma_re = 1.0,
      n_obs = n, n_mesh = n_sub,
      Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
      Aeff_x = asm$A_eff@x, Aeff_i = asm$A_eff@i, Aeff_p = asm$A_eff@p,
      family = family, phi = phi,
      max_iter = as.integer(max_iter), tol = tol,
      n_threads = as.integer(n_threads),
      offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)),
      error = function(e) NULL)
    if (is.null(fit)) {
      return(list(log_marginal = -Inf, n_iter = 0L, converged = FALSE, asm = asm))
    }
    beta_hat  <- fit$mode[seq_len(p)]
    x_hat     <- fit$mode[(p + 1L):(p + n_sub)]
    n_iter    <- fit$n_iter
    converged <- isTRUE(fit$converged)
  }

  lm <- tryCatch(cpp_spde_fractional_logmarginal(
    y = as.numeric(y), X = as.matrix(X),
    A_eff = asm$A_eff, Pl = asm$Pl,
    C0sub = as.numeric(spatial$C0_diag)[asm$keep],
    family = family, phi = phi,
    beta_hat = as.numeric(beta_hat), x_hat = as.numeric(x_hat),
    n_trials = as.integer(n_trials),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset),
    tau_beta = tau_beta),
    error = function(e) -Inf)

  list(log_marginal = lm, n_iter = n_iter, converged = converged, asm = asm)
}

#' Fractional rSPDE single-point Laplace fit at a fixed (range, sigma)
#'
#' The fractional counterpart of the integer `cpp_laplace_fit_spde` branch in
#' [laplace_spde_at()]. Assembles `(Q, A_eff)` via [.spde_assemble_at()] and runs
#' the precomputed C++ solve, whose latent is the auxiliary weights `x`; the
#' returned `mode` mesh block is mapped back to the field `u = Pr x` so every
#' downstream consumer reads field-space mesh effects exactly as on the integer
#' path. The log-marginal is the well-conditioned `B` / determinant-lemma marginal
#' (`cpp_spde_fractional_logmarginal()`) for the RE-free case (comparable across
#' the `(range, sigma)` grid); a fit with an iid RE block keeps the precomputed
#' precision-space marginal.
#'
#' @keywords internal
.spde_laplace_fractional_at <- function(y, n_trials, X, spatial,
                                        family, phi, range, sigma,
                                        re_idx, n_re_groups, sigma_re,
                                        max_iter, tol, n_threads, offset,
                                        order = 4L) {
  asm <- .spde_assemble_at(spatial, range, sigma, order = order)
  Qg  <- as(asm$Q, "generalMatrix")
  n_sub <- length(asm$keep)

  if (is.null(re_idx)) re_idx <- rep(0L, length(y))

  result <- cpp_laplace_fit_spde_precomputed(
    y = as.numeric(y),
    n_trials = as.integer(n_trials %||% rep(1L, length(y))),
    X = as.matrix(X),
    re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups),
    sigma_re = sigma_re,
    n_obs = length(y), n_mesh = n_sub,
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = asm$A_eff@x, Aeff_i = asm$A_eff@i, Aeff_p = asm$A_eff@p,
    family = family, phi = phi,
    max_iter = as.integer(max_iter), tol = tol,
    n_threads = as.integer(n_threads),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)
  )

  # Rebuild the mode in the FULL mesh layout, field-space: the fit ran on the
  # non-orphan submesh, so map the auxiliary weights x to the field u = Pr x and
  # scatter into the kept nodes (orphan nodes carry zero field), matching the
  # integer path's [beta, re, field(n_mesh)] layout for downstream consumers.
  ms       <- ncol(X) + as.integer(n_re_groups)
  x_mesh   <- result$mode[(ms + 1L):(ms + n_sub)]

  # Recompute the single-fit log-marginal through the well-conditioned C++
  # B / determinant-lemma (cpp_spde_fractional_logmarginal), not the precomputed
  # fit's direct precision-space marginal (whose log|Q| / log|H| lose digits on
  # the rational precision's wide spectrum). The det-lemma marginal has no RE
  # term, so a fit carrying an iid RE block keeps the precomputed marginal.
  if (n_re_groups == 0L) {
    result$log_marginal <- tryCatch(cpp_spde_fractional_logmarginal(
      y = as.numeric(y), X = as.matrix(X),
      A_eff = asm$A_eff, Pl = asm$Pl,
      C0sub = as.numeric(spatial$C0_diag)[asm$keep],
      family = family, phi = phi,
      beta_hat = as.numeric(result$mode[seq_len(ncol(X))]),
      x_hat = as.numeric(x_mesh),
      n_trials = as.integer(n_trials %||% rep(1L, length(y))),
      offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)),
      error = function(e) result$log_marginal)
  }

  field_full <- numeric(spatial$n_mesh)
  field_full[asm$keep] <- as.numeric(asm$Pr %*% x_mesh)
  result$mode <- c(result$mode[seq_len(ms)], field_full)

  result$latent_mesh <- x_mesh
  result$keep <- asm$keep
  result$Pr <- asm$Pr
  result$kappa <- asm$kappa
  result$var_scale <- asm$var_scale
  result$range <- range
  result$sigma <- sigma
  result$spatial <- spatial
  result
}
