# The 1% accuracy target of Riutort-Mayol et al. (2023), Eq. 18.
.HSGP_APPROX_ERROR <- 0.01

# Spectral variance outside the rectangular frequency set in the tensor
# expansion of Riutort-Mayol et al. (2023), Eqs. 9-12 and Section 4.2.
# For a 2-D Matern spectrum its radial tail beyond R is
# (1 + R^2 ell^2 / (2 nu))^(-nu). The rectangle's radial boundary in
# direction t is min(Omega_1 / cos(t), Omega_2 / sin(t)).
.hsgp_spectral_tail <- function(a, S, kernel) {
  if (a <= 0) return(1)
  omega <- pi * a * max(S) / (2 * S)
  nu <- .hsgp_kernel_info(kernel)$nu
  if (is.infinite(nu)) {
    return(1 - prod(2 * stats::pnorm(omega) - 1))
  }
  split <- atan(omega[2L] / omega[1L])
  radial <- function(t) {
    r <- pmin(omega[1L] / cos(t), omega[2L] / sin(t))
    (1 + r^2 / (2 * nu))^(-nu)
  }
  2 / pi * (stats::integrate(radial, 0, split)$value +
              stats::integrate(radial, split, pi / 2)$value)
}

# The coefficient m ell / (c max(S)) retaining 99% of the 2-D spectral
# variance, using the paper's 1% accuracy target (Eq. 18). A scalar S
# selects the paper's fitted 1-D rule. The covariance-error rule remains a
# lower bound, since spectral mass and covariance total variation differ.
.hsgp_m_coefficient <- local({
  cache <- new.env(parent = emptyenv())
  function(S, kernel) {
    paper <- .hsgp_kernel_info(kernel)$m_coef
    if (length(S) == 1L) return(paper)
    key <- paste(kernel, paste(format(S / max(S), digits = 16), collapse = ":"))
    if (exists(key, cache, inherits = FALSE)) return(cache[[key]])
    hi <- paper
    while (.hsgp_spectral_tail(hi, S, kernel) > .HSGP_APPROX_ERROR) hi <- 2 * hi
    a <- stats::uniroot(function(x) .hsgp_spectral_tail(x, S, kernel) - .HSGP_APPROX_ERROR,
                         c(0, hi), tol = 1e-8)$root
    cache[[key]] <- max(paper, a)
    cache[[key]]
  }
})

# The Dirichlet heat kernel is the free heat kernel times the probability
# that a Brownian bridge stays inside the rectangle. The reflection principle
# bounds the probability of hitting any face by the sum over faces. Mixing
# the heat kernels gives the corresponding stationary-kernel correlations at
# twice each distance to a face. The nearest corner maximizes this bound.
.hsgp_boundary_error <- function(c, S, ell, kernel) {
  corr <- .hsgp_kernel_info(kernel)$corr
  sum(corr(2 * (c - 1) * S / ell) + corr(2 * (c + 1) * S / ell))
}

.hsgp_boundary_factor <- function(c, S, ell, kernel) {
  if (length(S) == 1L || .hsgp_boundary_error(c, S, ell, kernel) <= .HSGP_APPROX_ERROR) return(c)
  hi <- 2 * c
  while (.hsgp_boundary_error(hi, S, ell, kernel) > .HSGP_APPROX_ERROR) hi <- 2 * hi
  stats::uniroot(function(x) .hsgp_boundary_error(x, S, ell, kernel) - .HSGP_APPROX_ERROR,
                  c(c, hi), tol = 1e-8)$root
}

.hsgp_boundary_lengthscale <- function(c, S, hi, kernel) {
  if (hi <= 0) return(0)
  if (length(S) == 1L || .hsgp_boundary_error(c, S, hi, kernel) <= .HSGP_APPROX_ERROR) return(hi)
  if (c <= 1) return(0)
  stats::uniroot(function(x) .hsgp_boundary_error(c, S, x, kernel) - .HSGP_APPROX_ERROR,
                  c(.Machine$double.eps * min(S), hi), tol = 1e-8)$root
}
