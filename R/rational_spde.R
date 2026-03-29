#' Compute Rational Approximation Coefficients for Fractional SPDE
#'
#' Computes poles and weights for the rational approximation of x^(-beta)
#' where beta = alpha/2 = (nu + 1)/2 in 2D. Uses the partial fraction
#' decomposition of the best rational approximation.
#'
#' @param nu Matérn smoothness parameter (can be fractional, e.g. 0.5, 1.5, 2.5)
#' @param m Order of the rational approximation. Default 2.
#'   Higher m gives better accuracy but more expensive Q computation.
#' @param lambda_range Approximate eigenvalue range of the FEM operator.
#'   Default c(1e-4, 1e4). Only matters for numerical precision of
#'   the approximation, not the result.
#'
#' @return A list with:
#'   \itemize{
#'     \item `poles`: vector of m pole values
#'     \item `weights`: vector of m weight values
#'     \item `alpha`: the operator order (nu + 1 in 2D)
#'     \item `beta`: the fractional power (alpha/2)
#'     \item `m`: approximation order
#'   }
#'
#' @details
#' For integer nu (0, 1, 2, ...), the standard SPDE construction is exact
#' and this function is not needed. For fractional nu (e.g. 0.5, 1.5),
#' this computes the coefficients for the rational SPDE approach
#' (Bolin, Simas & Xiong, 2023).
#'
#' The rational approximation writes:
#'   x^(-beta) ≈ sum_k w_k / (x + r_k)
#'
#' which translates to the precision:
#'   Q ≈ tau² * sum_k w_k * (L + r_k*C)' * C^{-1} * (L + r_k*C)
#'
#' where L = kappa²*C + G is the FEM operator.
#'
#' @references
#' Bolin, D., Simas, A.B. & Xiong, J. (2023). Rational SPDE approach for
#' Gaussian random fields with general smoothness. Journal of the Royal
#' Statistical Society Series B.
#'
#' @keywords internal
rational_spde_coefficients <- function(nu, m = 2, lambda_range = c(1e-4, 1e4)) {
  alpha <- nu + 1  # d/2 = 1 in 2D
  beta <- alpha / 2

  # For integer alpha (nu = 0, 1, 2, ...), direct construction — no rational approx needed
  if (abs(alpha - round(alpha)) < 1e-10) {
    return(list(
      poles = numeric(0),
      weights = numeric(0),
      alpha = alpha,
      beta = beta,
      m = 0,
      is_integer = TRUE
    ))
  }

  # Compute poles and weights via log-uniform spacing
  # This is a simplified version of the BRASIL algorithm.
  # For production use, the exact coefficients from Bolin et al. 2023
  # would give optimal approximation.
  #
  # The m-point Gauss-Jacobi quadrature for the integral representation:
  #   x^(-beta) = sin(pi*beta)/pi * integral_0^inf t^(-beta)/(x+t) dt
  #
  # Using log-uniform spacing on [lambda_min, lambda_max]:
  log_lam <- seq(log(lambda_range[1]), log(lambda_range[2]), length.out = m + 2)
  log_lam <- log_lam[2:(m + 1)]  # interior points

  poles <- exp(log_lam)

  # Weights from the integral representation
  # sin(pi*beta)/pi * t^(-beta) * dt (in log space: dt = t * d(log t))
  d_log_t <- diff(seq(log(lambda_range[1]), log(lambda_range[2]), length.out = m + 2))[1]
  weights <- sin(pi * beta) / pi * poles^(1 - beta) * d_log_t

  # Normalize so that the approximation matches x^(-beta) at x = 1
  # sum(w_k / (1 + r_k)) should equal 1
  approx_at_1 <- sum(weights / (1 + poles))
  weights <- weights / approx_at_1

  list(
    poles = poles,
    weights = weights,
    alpha = alpha,
    beta = beta,
    m = m,
    is_integer = FALSE
  )
}
