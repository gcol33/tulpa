#' Rational approximation coefficients for a fractional SPDE
#'
#' Returns the coefficient descriptor for an SPDE Matern field of operator order
#' `alpha = nu + 1` (in 2D). For integer `nu` the construction is exact and no
#' rational approximation is needed. For fractional `nu` it returns the
#' rational-SPDE roots from the BRASIL best-rational approximation (see Details).
#'
#' @param nu Matern smoothness parameter. A positive number; may be
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
#' For integer `nu` (1, 2, 3, ...) the operator order `alpha = nu + 1` is an
#' integer and the precision is assembled directly from integer powers of the
#' FEM operator `L = kappa^2 C + G` -- an exact construction.
#'
#' For fractional `nu` the roots are those of the degree-`(m, m)` best uniform
#' (minimax) rational approximation of `x^{-beta_rem}`, `beta = alpha / 2`,
#' `beta_rem = beta - (max(1, floor(beta)) - 1)`, computed by the BRASIL
#' algorithm (Hofreither 2021); its numerator zeros map to `rc = 1 / zero` and
#' denominator poles to `rb = 1 / pole`. The fitted fractional field is the
#' covariance-based rational SPDE (`.spde_rational_assemble()`), which takes its
#' own roots from `.spde_rational_terms()`.
#'
#' @references
#' Bolin, D. & Kirchner, K. (2020). The rational SPDE approach for Gaussian
#' random fields with general smoothness. Journal of Computational and Graphical
#' Statistics, 29(2), 274-285.
#'
#' Hofreither, C. (2021). An algorithm for best rational approximation based on
#' barycentric rational interpolation. Numerical Algorithms, 88, 365-388.
#'
#' @noRd
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
#' `nu` must be a single finite positive number. Integer `nu` (1, 2, 3, ...)
#' gives an exact FEM construction; fractional `nu` uses the rational SPDE
#' approximation (see `rational_spde_coefficients()`).
#'
#' @param nu Candidate smoothness parameter.
#' @return Invisibly `TRUE`; raises an error on an invalid `nu`.
#' @noRd
.validate_spde_nu <- function(nu) {
  if (!is.numeric(nu) || length(nu) != 1L || !is.finite(nu)) {
    stop("SPDE `nu` must be a single finite number.", call. = FALSE)
  }
  if (nu <= 0) {
    # The d = 2 Matern parameterisation carries a factor Gamma(nu) / Gamma(nu+1)
    # = 1 / nu in the marginal variance, so nu = 0 has no (kappa, tau):
    # kappa = sqrt(8 nu) / range is 0 and tau = 1 / (sqrt(4 pi nu) kappa^nu
    # sigma) is infinite. A fit at nu = 0 could only report an infeasible cell.
    stop("SPDE `nu` must be > 0. The Matern parameterisation is degenerate ",
         "at nu = 0: kappa = sqrt(8 * nu) / range is 0 and ",
         "tau = 1 / (sqrt(4 * pi * nu) * kappa^nu * sigma) is infinite, so no ",
         "precision can be built. Use nu = 1 for the standard second-order ",
         "operator.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Is an SPDE smoothness `nu` fractional (rational path) or integer (exact path)?
#'
#' @param nu Matern smoothness; `alpha = nu + 1` in 2D.
#' @return `TRUE` when `alpha` is non-integer (the rational SPDE path).
#' @noRd
.spde_nu_is_fractional <- function(nu) {
  alpha <- nu + 1
  abs(alpha - round(alpha)) > 1e-10
}

#' Build the sparse FEM stiffness / projector from a spatial_spde spec
#'
#' Prefers the stored `Matrix` objects (`spatial$G`, `spatial$A`); falls back to
#' rebuilding them from the pre-extracted CSC slots so a spec carrying only the
#' slots still assembles.
#' @noRd
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

# Order of the covariance-based rational approximation a fractional-nu field is
# fitted at, unless its spec carries `rational_order`. Measured on a 229-node
# mesh at nu = 0.5 (gaussian response, sigma = 1): the log marginal against the
# exact discrete fractional field (eigendecomposition of C^-1 G) is off by
# 0.04 / 0.22 / 2.75 / 19 nats at range 0.1 / 0.3 / 1 / 3 at order 2 and by
# 0.01 / 0.02 / 0.34 / 0.58 at order 3, for 1.4x the cost per cell.
.SPDE_RATIONAL_ORDER <- 3L

.spde_rational_order <- function(spatial) {
  as.integer(spatial$rational_order %||% .SPDE_RATIONAL_ORDER)
}

#' Assemble the fractional SPDE latent and obs map at a given (range, sigma)
#'
#' Shared by every fractional-`nu` fit path (Laplace single-point, nested,
#' NUTS, marginal SEs). Maps `(range, sigma)` to the SPDE hyperparameters
#' `kappa = sqrt(8 nu) / range` and `tau = 1 / (sqrt(4 pi nu) kappa^nu sigma)`,
#' the same map the integer-`nu` path uses, so a fractional field carries the
#' Matern marginal variance `sigma^2` away from the mesh boundary exactly as an
#' integer one does. The field is the covariance-based rational SPDE
#' (`.spde_rational_assemble()`): the latent `x` stacks the independent GMRF
#' components, `Q` is their block-diagonal precision and the field is
#' `u = Pr x`.
#'
#' The approximation interval is `[kappa^2, kappa^2 + lambda_G]`, with
#' `lambda_G` the Gershgorin bound on the spectrum of `C^-1 G` the spec carries
#' (`spatial$lambda_max_G`, computed once per mesh when the spec is built).
#'
#' Mesh nodes with zero lumped mass (mesh-extension nodes outside the data
#' hull, e.g.) carry no FEM mass and no observation projects onto them; the
#' assembly runs on the submesh of the others (`keep`) and the field is zero at
#' the dropped nodes.
#'
#' @param spatial A validated `spatial_spde` spec with fractional `nu`.
#' @param range,sigma Spatial range and marginal SD.
#' @param order Rational approximation order (`.spde_rational_order()`).
#' @return A list with `Q` (block-diagonal latent precision), `Pr` (latent ->
#'   field on the submesh), `A_eff = A Pr`, `keep` (1-based indices of the
#'   retained mesh nodes), `n_mesh_full`, `n_latent = ncol(Q)`, `kappa`, `tau`,
#'   `l_max` and the partial-fraction `terms`.
#' @noRd
.spde_assemble_at <- function(spatial, range, sigma,
                              order = .spde_rational_order(spatial)) {
  nu <- spatial$nu
  kt <- .spde_kappa_tau(range, sigma, nu)
  fem <- .spde_fem_matrices(spatial)
  C0  <- as.numeric(spatial$C0_diag)
  keep <- which(C0 > 1e-12)
  lambda_G <- spatial$lambda_max_G
  if (is.null(lambda_G)) {
    stop("A fractional-nu SPDE spec must carry `lambda_max_G`, the spectrum ",
         "bound spatial_spde() / spatial_spde_custom() attach to it.",
         call. = FALSE)
  }
  Gk <- as(fem$G[keep, keep, drop = FALSE], "CsparseMatrix")
  asm <- .spde_rational_assemble(
    C0 = C0[keep], G = Gk, kappa = kt$kappa, tau = kt$tau_spde, nu = nu,
    order = as.integer(order), d = 2, l_max = kt$kappa^2 + lambda_G)
  A_eff <- as(fem$A[, keep, drop = FALSE] %*% asm$Pr, "CsparseMatrix")
  list(Q = asm$Q, Pr = asm$Pr, A_eff = A_eff, keep = keep,
       n_mesh_full = length(C0), n_latent = ncol(asm$Q),
       kappa = kt$kappa, tau = kt$tau_spde, l_max = asm$l_max,
       terms = asm$terms)
}

#' Precomputed Laplace solve of the fractional SPDE latent at (range, sigma)
#'
#' One run of `cpp_laplace_fit_spde_precomputed()` on the assembly `asm`
#' (`.spde_assemble_at()`). Its log marginal is the precision-space Laplace
#' marginal of the stacked latent, which for a gaussian response is the exact
#' conjugate marginal. `phi` follows the engine convention (the Gaussian
#' residual variance) and is converted for the kernel here. `beta_prior` is
#' the normalized fixed-effect prior (`.normalize_beta_prior()`), `NULL` for
#' the engine default.
#' @noRd
.spde_fractional_solve <- function(asm, y, n_trials, X, family, phi,
                                   re_idx, n_re_groups, sigma_re,
                                   max_iter, tol, n_threads, offset,
                                   weights = NULL, beta_prior = NULL) {
  Qg <- as(asm$Q, "generalMatrix")
  n  <- length(y)
  cpp_laplace_fit_spde_precomputed(
    y = as.numeric(y),
    n_trials = as.integer(n_trials %||% rep(1L, n)),
    X = as.matrix(X),
    re_idx = as.numeric(re_idx %||% rep(0L, n)),
    n_re_groups = as.integer(n_re_groups),
    sigma_re = sigma_re,
    n_obs = n, n_mesh = asm$n_latent,
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = asm$A_eff@x, Aeff_i = asm$A_eff@i, Aeff_p = asm$A_eff@p,
    family = family, phi = .phi_to_kernel(family, phi),
    max_iter = as.integer(max_iter), tol = tol,
    n_threads = as.integer(n_threads),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset),
    weights_nullable = if (is.null(weights)) NULL else as.numeric(weights),
    beta_prior = beta_prior)
}

#' Laplace log-marginal of a fractional SPDE fit at (range, sigma)
#'
#' The nested `(range, sigma)` integrand. A cell whose rational approximation
#' is degenerate, or whose inner solve fails or does not reach its mode, has no
#' Laplace marginal and takes `-Inf` (zero nested weight).
#'
#' @return A list with `log_marginal`, `n_iter`, `converged`, and the assembly.
#' @noRd
.spde_nested_logmarginal_at <- function(spatial, range, sigma, y, X, family, phi,
                                        n_trials, order, max_iter, tol, n_threads,
                                        offset, beta_prior = NULL) {
  asm <- tryCatch(.spde_assemble_at(spatial, range, sigma, order = order),
                  error = function(e) NULL)
  fit <- if (!is.null(asm)) tryCatch(.spde_fractional_solve(
    asm, y = y, n_trials = n_trials, X = X, family = family, phi = phi,
    re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0, max_iter = max_iter,
    tol = tol, n_threads = n_threads, offset = offset,
    beta_prior = beta_prior),
    error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged) || !is.finite(fit$log_marginal)) {
    return(list(log_marginal = -Inf, n_iter = fit$n_iter %||% 0L,
                converged = FALSE, asm = asm))
  }
  list(log_marginal = fit$log_marginal, n_iter = fit$n_iter, converged = TRUE,
       asm = asm)
}

#' Fractional SPDE single-point Laplace fit at a fixed (range, sigma)
#'
#' The fractional counterpart of the integer `cpp_laplace_fit_spde` branch in
#' `laplace_spde_at()`. Assembles `(Q, A_eff)` via `.spde_assemble_at()` and runs
#' the precomputed C++ solve, whose latent is the stack of rational components;
#' the returned `mode` mesh block is the field `u = Pr x` scattered to the full
#' mesh, so every downstream consumer reads field-space mesh effects exactly as
#' on the integer path.
#'
#' @noRd
.spde_laplace_fractional_at <- function(y, n_trials, X, spatial,
                                        family, phi, range, sigma,
                                        re_idx, n_re_groups, sigma_re,
                                        max_iter, tol, n_threads, offset,
                                        weights = NULL, beta_prior = NULL,
                                        order = .spde_rational_order(spatial)) {
  asm <- .spde_assemble_at(spatial, range, sigma, order = order)
  result <- .spde_fractional_solve(
    asm, y = y, n_trials = n_trials, X = X, family = family, phi = phi,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    max_iter = max_iter, tol = tol, n_threads = n_threads, offset = offset,
    weights = weights, beta_prior = beta_prior)

  ms     <- ncol(X) + as.integer(n_re_groups)
  x_lat  <- result$mode[(ms + 1L):(ms + asm$n_latent)]
  field_full <- numeric(spatial$n_mesh)
  field_full[asm$keep] <- as.numeric(asm$Pr %*% x_lat)
  result$mode <- c(result$mode[seq_len(ms)], field_full)

  result$latent <- x_lat
  result$keep <- asm$keep
  result$Pr <- asm$Pr
  result$kappa <- asm$kappa
  result$range <- range
  result$sigma <- sigma
  result$spatial <- spatial
  result
}


# BRASIL roots for the rational SPDE. `.spde_rational_roots()` returns the
# degree-(m, m) best uniform rational approximation of x^{-beta_rem} on
# [spectrum_ratio, 1], as its numerator factors `rc` (= 1 / zero), denominator
# factors `rb` (= 1 / pole) and the scale constant, with
#   R(x) = scale * prod(x - 1 / rc) / prod(x - 1 / rb).
# `spectrum_ratio` = l_min / l_max of the operator whose fractional power is
# approximated; the approximation interval is [spectrum_ratio, 1] on the
# l / l_max-normalized axis. `.spde_rational_terms()` turns these roots into the
# partial-fraction covariance terms the field is assembled from.
#
# The fitted path is the compiled search (`src/brasil.h`): the outer integrator
# visits one spectrum interval per (range, sigma) cell -- 1102 of them for a
# single n = 40 fit, essentially all distinct -- so this runs once per cell and
# at R speed costs ~0.55 s apiece, the whole run time of a fractional fit
# (gcol33/tulpa#818). The R search it was ported from is the oracle in
# tests/testthat/helper-brasil-reference.R.
.spde_rational_roots <- function(order, beta, spectrum_ratio, tol = 1e-7) {
  cpp_spde_rational_roots(as.integer(order), as.numeric(beta),
                          as.numeric(spectrum_ratio), as.numeric(tol))
}

# Partial-fraction terms of the covariance-based rational SPDE (Bolin, Simas &
# Xiong 2024). The Matern field of operator order alpha = nu + d/2 has the
# covariance operator tau^-2 L^-alpha. With m = floor(alpha) and b = alpha - m in
# (0, 1), the fractional factor l^-b is a Stieltjes function, so its best
# rational approximation on the spectrum [kappa^2, l_max] has negative real poles
# and positive residues:
#   l^-b ~= c0 + sum_j B_j / (l + e_j),         c0 > 0, B_j > 0, e_j > 0,
# and therefore
#   l^-alpha ~= c0 l^-m + sum_j B_j l^-m / (l + e_j),
# a sum of m_rat + 1 covariance operators each of which is the covariance of a
# proper GMRF. The roots come from `.spde_rational_roots()` on the
# l / l_max-normalized axis: with zeros z = 1 / rc and poles p = 1 / rb of
#   R(x) = scale prod(x - z) / prod(x - p) ~= x^-b,  x = l / l_max,
# the partial fractions are R(x) = scale + sum_j rho_j / (x - p_j), and in l
#   c0 = scale l_max^-b,  B_j = rho_j l_max^(1 - b),  e_j = -l_max p_j.
# A degenerate approximation (non-finite or non-positive term) cannot be a
# covariance and stops the assembly.
.spde_rational_terms <- function(nu, order, kappa, l_max, d = 2) {
  alpha <- nu + d / 2
  m <- as.integer(floor(alpha))
  b <- alpha - m
  spectrum_ratio <- min(kappa^2 / l_max, 0.5)
  rr <- .spde_rational_roots(as.integer(order), b, spectrum_ratio)
  z <- 1 / rr$rc
  p <- 1 / rr$rb
  rho <- vapply(seq_along(p), function(j) {
    rr$scale * prod(p[j] - z) / prod(p[j] - p[-j])
  }, numeric(1))
  terms <- list(m = m, b = b, c0 = rr$scale * l_max^(-b),
                B = rho * l_max^(1 - b), e = -l_max * p,
                spectrum_ratio = spectrum_ratio, error = rr$error)
  ok <- is.finite(terms$c0) && terms$c0 > 0 &&
    all(is.finite(terms$B)) && all(terms$B > 0) &&
    all(is.finite(terms$e)) && all(terms$e > 0)
  if (!ok) {
    stop(sprintf(paste0(
      "The order-%d rational approximation of the fractional SPDE operator ",
      "is degenerate on the spectrum ratio %.3g (a pole or residue is not ",
      "positive), so it does not define a covariance; use a lower ",
      "`rational_order`."), as.integer(order), spectrum_ratio), call. = FALSE)
  }
  terms
}

# Assemble the fractional SPDE field from FEM matrices as a sum of independent
# GMRF components, one per term of `.spde_rational_terms()`. With the lumped
# mass C (diagonal `C0`), K = kappa^2 C + G and the discrete operator
# L = C^-1 K, the field covariance tau^-2 L^-alpha C^-1 is approximated by
#   u = u_0 + sum_j u_j,
#   Q_0 = (tau^2 / c0)  K (C^-1 K)^(m - 1),
#   Q_j = (tau^2 / B_j) (K + e_j C) C^-1 K (C^-1 K)^(m - 1),
# each u_k independent with precision Q_k. The latent is the stack
# x = (u_0, u_1, ...), its precision the block diagonal of the Q_k, and the
# field u = Pr x with Pr = [I, I, ...]. Every block is a polynomial of degree at
# most m + 1 in L with positive coefficients, so each is as well conditioned as
# the integer-order precision of that degree, and the Laplace marginal is formed
# in precision space like the integer path's. `l_max` bounds the spectrum of L
# from above (`.spde_operator_lambda_max()` plus kappa^2).
.spde_rational_assemble <- function(C0, G, kappa, tau, nu, order,
                                    d = 2, l_max) {
  n  <- length(C0)
  Cm <- Matrix::Diagonal(n, C0)
  Ci <- Matrix::Diagonal(n, 1 / C0)
  K  <- kappa^2 * Cm + G
  terms <- .spde_rational_terms(nu, order, kappa, l_max, d = d)

  base <- K
  if (terms$m > 1L) for (i in seq_len(terms$m - 1L)) base <- base %*% Ci %*% K
  sym <- function(M) as(Matrix::forceSymmetric(M), "CsparseMatrix")
  blocks <- c(
    list(sym((tau^2 / terms$c0) * base)),
    lapply(seq_along(terms$B), function(j) {
      sym((tau^2 / terms$B[j]) * (K + terms$e[j] * Cm) %*% Ci %*% base)
    }))
  n_comp <- length(blocks)
  Q  <- Matrix::bdiag(blocks)
  Pr <- do.call(cbind, rep(list(Matrix::Diagonal(n)), n_comp))
  list(Q = as(Matrix::forceSymmetric(Q), "CsparseMatrix"),
       Pr = as(Pr, "CsparseMatrix"), terms = terms, n_comp = n_comp,
       l_max = l_max, spectrum_ratio = terms$spectrum_ratio)
}

# Upper bound on the spectrum of C^-1 G over the mesh nodes that carry mass,
# by Gershgorin's theorem: every eigenvalue of C^-1 G lies below
# max_i sum_j |G_ij| / C_i. It is a property of the mesh, independent of
# (range, sigma); the operator L = kappa^2 I + C^-1 G then has
# l_max = kappa^2 + this bound, which the rational approximation interval
# covers.
.spde_operator_lambda_max <- function(C0, G) {
  keep <- which(C0 > 1e-12)
  Gk <- abs(as(G[keep, keep, drop = FALSE], "CsparseMatrix"))
  max(Matrix::rowSums(Gk) / C0[keep])
}
