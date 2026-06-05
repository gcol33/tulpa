#' Rational Approximation Coefficients for Fractional SPDE (integer-only)
#'
#' Returns the FEM construction descriptor for an SPDE Matern field of operator
#' order `alpha = nu + 1` (in 2D). For integer `nu` the construction is exact and
#' no rational approximation is needed; fractional `nu` is not supported and
#' raises an error (see Details).
#'
#' @param nu Matern smoothness parameter. Must be a non-negative integer
#'   (0, 1, 2, ...).
#' @param m Reserved for a future rational implementation; ignored.
#' @param lambda_range Reserved for a future rational implementation; ignored.
#'
#' @return A list with:
#'   \itemize{
#'     \item `poles`: empty (no rational poles for integer `nu`)
#'     \item `weights`: empty
#'     \item `alpha`: the operator order (`nu + 1` in 2D)
#'     \item `beta`: `alpha / 2`
#'     \item `m`: 0
#'     \item `is_integer`: `TRUE`
#'   }
#'
#' @details
#' For integer `nu` (0, 1, 2, ...) the operator order `alpha = nu + 1` is an
#' integer and the precision is assembled directly from integer powers of the
#' FEM operator `L = kappa^2 C + G` -- an exact construction.
#'
#' Fractional `nu` is **not supported**. A faithful fractional field requires a
#' rational approximation of `x^(-alpha/2)` whose partial-fraction coefficients
#' assemble a precision with spectral symbol proportional to `l^alpha` (l the
#' generalized eigenvalue of `L` relative to `C`). The assembly currently wired
#' through the SPDE kernels,
#' `Q = tau^2 * sum_k w_k (L + r_k C)' C^{-1} (L + r_k C)`, has symbol
#' `tau^2 * sum_k w_k (l + r_k)^2`, which is a single quadratic in `l` for any
#' number of poles and therefore can only represent an `alpha = 2` field -- no
#' choice of poles/weights recovers fractional smoothness. Rather than return
#' coefficients that silently feed a mis-specified field, fractional `nu` errors.
#' Tracking a faithful rational SPDE assembly: gcol33/tulpa#71.
#'
#' @references
#' Bolin, D., Simas, A.B. & Xiong, J. (2023). Rational SPDE approach for
#' Gaussian random fields with general smoothness. Journal of the Royal
#' Statistical Society Series B. (The reference construction a future
#' fractional implementation should follow.)
#'
#' @keywords internal
rational_spde_coefficients <- function(nu, m = 2, lambda_range = c(1e-4, 1e4)) {
  .validate_spde_nu(nu)

  alpha <- nu + 1  # d/2 = 1 in 2D
  beta <- alpha / 2

  list(
    poles = numeric(0),
    weights = numeric(0),
    alpha = alpha,
    beta = beta,
    m = 0,
    is_integer = TRUE
  )
}

#' Validate the SPDE smoothness parameter
#'
#' Fractional `nu` is rejected because the wired rational assembly collapses to
#' an integer `alpha = 2` field (see [rational_spde_coefficients()] Details and
#' gcol33/tulpa#71). Integer `nu` (0, 1, 2, ...) is exact.
#'
#' @param nu Candidate smoothness parameter.
#' @return Invisibly `TRUE`; raises an error on a non-integer `nu`.
#' @keywords internal
.validate_spde_nu <- function(nu) {
  if (!is.numeric(nu) || length(nu) != 1L || !is.finite(nu)) {
    stop("SPDE `nu` must be a single finite number.", call. = FALSE)
  }
  alpha <- nu + 1
  if (abs(alpha - round(alpha)) > 1e-10 || nu < 0) {
    stop(sprintf(
      paste0(
        "Fractional SPDE smoothness (nu = %g) is not yet supported. The wired ",
        "rational assembly collapses to an integer alpha = 2 field, so it ",
        "cannot represent fractional smoothness (see gcol33/tulpa#71). Use an ",
        "integer nu (0, 1, 2, ...) for an exact FEM construction."),
      nu), call. = FALSE)
  }
  invisible(TRUE)
}
