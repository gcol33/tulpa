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
#' The field assembly from these roots is [.spde_rational_assemble()]; it is
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

#' Mean marginal variance of the rSPDE field u = Pr x, x ~ N(0, Q^{-1})
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
#' [.spde_rational_assemble()], then **normalizes the field to marginal variance
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

# Per-observation GLM working weight w = -d^2 log p(y|eta) / d eta^2 at eta, and
# the log-likelihood, for the families the stable fractional-SPDE marginal
# supports. `phi` is the Gaussian residual variance / negbin reciprocal size.
.spde_family_wll <- function(family, y, eta, n_trials, phi) {
  if (family == "gaussian") {
    w  <- rep(1 / phi, length(y))
    ll <- sum(stats::dnorm(y, eta, sqrt(phi), log = TRUE))
  } else if (family == "poisson") {
    lam <- exp(eta)
    w   <- lam
    ll  <- sum(stats::dpois(y, lam, log = TRUE))
  } else if (family == "binomial") {
    p  <- stats::plogis(eta)
    w  <- n_trials * p * (1 - p)
    ll <- sum(stats::dbinom(y, n_trials, p, log = TRUE))
  } else {
    stop(sprintf(
      "Fractional-nu nested SPDE integration supports family in {gaussian, ",
      "poisson, binomial}; got '%s'. Supply explicit (range, sigma) for a ",
      "single-point fit, or use an integer nu.", family), call. = FALSE)
  }
  list(w = w, loglik = ll)
}

#' Numerically stable Laplace log-marginal for a fractional rSPDE at (range, sigma)
#'
#' The precision-space Laplace marginal needs `0.5(log|Q| - log|H|)`, whose two
#' determinants are individually corrupted by the rational precision's wide
#' spectrum (cond ~ 1e16) in a range-dependent way -- so range is unidentifiable
#' from it. This computes the same marginal through the matrix-determinant lemma:
#'   log|Q| - log|H| = -log|I + W^{1/2} (X X'/tau_beta + A_eff Q^{-1} A_eff') W^{1/2}|,
#' an `n_obs x n_obs` determinant of `I + PSD` that is well-conditioned. The cross
#' term `A_eff Q^{-1} A_eff' = (A_eff Pl^{-1}) C (A_eff Pl^{-1})'` is formed
#' through the operator factor `Pl` (condition number the square root of `Q`'s),
#' so no ill-conditioned `Q` inverse is ever taken. The mode comes from the
#' precomputed C++ fit (accurate despite the conditioning); only the determinant
#' term is recomputed here.
#'
#' @return A list with `log_marginal`, `n_iter`, `converged`, and the assembly.
#' @keywords internal
.spde_nested_logmarginal_at <- function(spatial, range, sigma, y, X, family, phi,
                                        n_trials, order, max_iter, tol, n_threads,
                                        offset, tau_beta = 1e-4) {
  n   <- length(y)
  p   <- ncol(X)
  asm <- .spde_assemble_at(spatial, range, sigma, order = order)
  n_sub <- length(asm$keep)
  C0sub <- as.numeric(spatial$C0_diag)[asm$keep]

  # B_field = A_eff Q^{-1} A_eff' = (A_eff Pl^{-1}) C (A_eff Pl^{-1})', formed
  # through Pl (cond = sqrt cond(Q)) so the wide rational spectrum is never
  # inverted. V_lat = X X'/tau_beta + B_field is the latent contribution to the
  # observation covariance (vague beta prior matching the precomputed fit). An
  # extreme grid cell (very small range -> very rough field) can push Pl past the
  # solver's singularity threshold; that cell is hopeless and gets -Inf (zero
  # nested weight) rather than crashing the grid.
  Mt <- tryCatch(as.matrix(Matrix::solve(asm$Pl, Matrix::t(asm$A_eff))),
                 error = function(e) NULL)
  if (is.null(Mt)) {
    return(list(log_marginal = -Inf, n_iter = 0L, converged = FALSE, asm = asm))
  }
  B   <- crossprod(Mt, C0sub * Mt)                                # n_obs x n_obs
  if (p > 0) B <- B + tcrossprod(X) / tau_beta
  off <- if (is.null(offset)) 0 else as.numeric(offset)

  if (family == "gaussian") {
    # Exact closed-form marginal: y - offset ~ N(0, V), V = V_lat + phi I.
    V  <- B + diag(phi, n)
    ch <- tryCatch(chol((V + t(V)) / 2), error = function(e) NULL)
    if (is.null(ch)) return(list(log_marginal = -Inf, n_iter = 1L, converged = FALSE))
    r  <- y - off
    lm <- -sum(log(diag(ch))) - 0.5 * sum(backsolve(ch, r, transpose = TRUE)^2)
    return(list(log_marginal = lm, n_iter = 1L, converged = TRUE, asm = asm))
  }

  # Non-Gaussian: Laplace marginal at the precomputed mode, with the
  # determinant ratio log|Q| - log|H| = -log|I + W^{1/2} V_lat W^{1/2}| (matrix
  # determinant lemma) computed from the well-conditioned B above. The prior
  # quadratic uses the Pl matvec x'Qx = ||C^{-1/2} Pl x||^2 (no inverse).
  Qg  <- as(asm$Q, "generalMatrix")
  fit <- cpp_laplace_fit_spde_precomputed(
    y = as.numeric(y), n_trials = as.integer(n_trials),
    X = as.matrix(X), re_idx = rep(0, n), n_re_groups = 0L, sigma_re = 1.0,
    n_obs = n, n_mesh = n_sub,
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = asm$A_eff@x, Aeff_i = asm$A_eff@i, Aeff_p = asm$A_eff@p,
    family = family, phi = phi,
    max_iter = as.integer(max_iter), tol = tol, n_threads = as.integer(n_threads),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)
  )
  beta_hat <- fit$mode[seq_len(p)]
  x_hat    <- fit$mode[(p + 1L):(p + n_sub)]
  eta_hat  <- as.numeric(X %*% beta_hat + asm$A_eff %*% x_hat) + off

  wll  <- .spde_family_wll(family, y, eta_hat, n_trials, phi)
  Plx  <- as.numeric(asm$Pl %*% x_hat)
  quad <- tau_beta * sum(beta_hat^2) + sum(Plx^2 / C0sub)
  sw   <- sqrt(wll$w)
  Gm   <- diag(n) + (sw %o% sw) * B
  ch   <- tryCatch(chol(Gm), error = function(e) NULL)
  if (is.null(ch)) return(list(log_marginal = -Inf, n_iter = fit$n_iter,
                               converged = FALSE, asm = asm, fit = fit))
  logdet_term <- 2 * sum(log(diag(ch)))

  list(
    log_marginal = wll$loglik - 0.5 * quad - 0.5 * logdet_term,
    n_iter = fit$n_iter, converged = isTRUE(fit$converged),
    asm = asm, fit = fit, beta = beta_hat, x = x_hat
  )
}

#' Fractional rSPDE single-point Laplace fit at a fixed (range, sigma)
#'
#' The fractional counterpart of the integer `cpp_laplace_fit_spde` branch in
#' [laplace_spde_at()]. Assembles `(Q, A_eff)` via [.spde_assemble_at()] and runs
#' the precomputed C++ solve, whose latent is the auxiliary weights `x`; the
#' returned `mode` mesh block is mapped back to the field `u = Pr x` so every
#' downstream consumer reads field-space mesh effects exactly as on the integer
#' path. The single-fit log-marginal is completed with `0.5 log|Q|` (the prior
#' normalizer the conditional solve omits) so values are comparable across the
#' nested `(range, sigma)` grid.
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

  # Complete the conditional log-marginal with the prior normalizer 0.5 log|Q|,
  # which varies with (kappa, tau) across the grid (the constant -0.5 n log 2pi
  # is grid-invariant and irrelevant to the nested weights).
  result$log_marginal <- result$log_marginal + 0.5 * asm$logdet_Q

  # Rebuild the mode in the FULL mesh layout, field-space: the fit ran on the
  # non-orphan submesh, so map the auxiliary weights x to the field u = Pr x and
  # scatter into the kept nodes (orphan nodes carry zero field), matching the
  # integer path's [beta, re, field(n_mesh)] layout for downstream consumers.
  ms       <- ncol(X) + as.integer(n_re_groups)
  x_mesh   <- result$mode[(ms + 1L):(ms + n_sub)]
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
