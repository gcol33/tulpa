#' Central Composite Design (CCD) grid for nested-Laplace integration
#'
#' @description
#' Produces a structured set of standardised hyperparameter points
#' \eqn{z \in \mathbb{R}^k} for use as integration nodes in a nested
#' Laplace approximation when \eqn{k \ge 3}. CCD scales much better than
#' the full tensor `expand.grid()` used by the 1D and 2D backends:
#' a CCD has \eqn{1 + 2k + 2^{k - q}} points (1 centre, 2k axial,
#' \eqn{2^{k - q}} factorial), versus \eqn{m^k} for an `m`-per-axis
#' tensor product.
#'
#' Point layout (with centre+axial+factorial scaling \eqn{f_0}):
#'   * 1 centre point at the origin;
#'   * 2k axial points at \eqn{\pm f_0} along each coordinate axis;
#'   * \eqn{2^{k - q}} factorial points at corners of the hypercube,
#'     scaled to lie on a sphere of radius \eqn{f_0}.
#'
#' For \eqn{k \le 6} the factorial portion is the full \eqn{2^k}
#' design. For \eqn{k \ge 7} a half-fraction (\eqn{q = 1}) using the
#' defining word \eqn{x_1 \cdots x_k} keeps the point count reasonable
#' while preserving Resolution V.
#'
#' Used by [nested_laplace()] for higher-dimensional hyperparameter
#' blocks. The standardised z-coordinates are mapped to physical
#' hyperparameters \eqn{\theta} via [ccd_to_theta()].
#'
#' @param k Number of hyperparameters (length of θ). Must be ≥ 1.
#' @param f_0 Radius of the design sphere (default \eqn{\sqrt{k}}). Larger
#'   `f_0` spreads points further out; smaller concentrates near the
#'   centre. INLA's default scales like \eqn{\sqrt{k}}.
#'
#' @return A list with components:
#'   * `z`: numeric matrix `[n_points × k]` of standardised hyperparameter
#'     coordinates.
#'   * `n_points`: integer; total grid size.
#'   * `kind`: character vector labelling each point as
#'     `"center"`, `"axial"`, or `"factorial"`.
#'   * `f_0`: the sphere radius used.
#'
#' @seealso [ccd_to_theta()] to map z-coordinates to physical θ.
#' @keywords internal
#' @export
ccd_grid <- function(k, f_0 = sqrt(k)) {
  k <- as.integer(k)
  if (k < 1L) stop("`k` must be >= 1.", call. = FALSE)
  if (!is.finite(f_0) || f_0 <= 0)
    stop("`f_0` must be a positive finite scalar.", call. = FALSE)

  # Centre point.
  z_center <- matrix(0, nrow = 1L, ncol = k)
  kind <- "center"

  # Axial points: ±f_0 along each coordinate axis.
  if (k >= 1L) {
    z_ax <- matrix(0, nrow = 2L * k, ncol = k)
    for (j in seq_len(k)) {
      z_ax[2L * j - 1L, j] <- f_0
      z_ax[2L * j, j]     <- -f_0
    }
    kind <- c(kind, rep("axial", 2L * k))
  } else {
    z_ax <- matrix(0, nrow = 0L, ncol = k)
  }

  # Factorial points: corners of (±1)^k scaled so each corner has norm f_0.
  # For k <= 6 use the full 2^k design; for k >= 7 use a Resolution-V
  # half-fraction (defining word: product of all x_j == +1) so the
  # number of factorial points stays at 2^(k-1).
  # Skip for k == 1 since the factorial corners coincide with the axials.
  if (k <= 1L) {
    z_fac <- matrix(0, nrow = 0L, ncol = k)
  } else if (k <= 6L) {
    signs <- as.matrix(do.call(expand.grid, rep(list(c(-1, 1)), k)))
    dimnames(signs) <- NULL
    s <- f_0 / sqrt(k)
    z_fac <- s * signs
    kind <- c(kind, rep("factorial", nrow(z_fac)))
  } else {
    # Half-fraction via defining word x_1 * ... * x_k = +1.
    signs <- as.matrix(do.call(expand.grid, rep(list(c(-1, 1)), k - 1L)))
    dimnames(signs) <- NULL
    last <- apply(signs, 1L, prod)         # +1 fraction
    signs <- cbind(signs, last)
    s <- f_0 / sqrt(k)
    z_fac <- s * signs
    kind <- c(kind, rep("factorial", nrow(z_fac)))
  }

  z <- rbind(z_center, z_ax, z_fac)
  list(
    z = unname(z),
    n_points = nrow(z),
    kind = kind,
    f_0 = f_0
  )
}

#' Map standardised CCD coordinates to physical hyperparameters
#'
#' @description
#' Converts standardised CCD z-coordinates (in \eqn{\mathbb{R}^k}) to
#' physical hyperparameters \eqn{\theta} via the affine map
#' \eqn{\theta = \hat\theta + L \cdot z} with optional log-scale
#' transform per component. `L` is typically a Cholesky factor of the
#' negative Hessian inverse evaluated at the (working) mode \eqn{\hat\theta}
#' — i.e. it scales `z` to one posterior standard deviation per axis.
#'
#' @param z Matrix `[n_points × k]` of standardised coordinates from
#'   [ccd_grid()].
#' @param theta_hat Numeric vector of length `k`: centre of the design,
#'   in either physical or log-space (per `log_scale`).
#' @param L Numeric `[k × k]` matrix: scale/rotation applied to z. Pass
#'   `diag(sd)` for a diagonal axis-aligned grid where `sd` is the
#'   per-axis posterior SD; pass a Cholesky factor to capture
#'   correlations between hyperparameters.
#' @param log_scale Logical (or logical vector of length k). If `TRUE`
#'   for component j, `theta_hat[j]` and column j of the affine
#'   transform live on log-scale and are exponentiated afterward
#'   (useful for positive parameters like τ). Default `FALSE` everywhere.
#'
#' @return Numeric matrix `[n_points × k]` of physical θ-values.
#'
#' @keywords internal
#' @export
ccd_to_theta <- function(z, theta_hat, L, log_scale = FALSE) {
  if (!is.matrix(z)) stop("`z` must be a matrix.", call. = FALSE)
  k <- ncol(z)
  if (length(theta_hat) != k)
    stop("`theta_hat` must have length k = ncol(z).", call. = FALSE)
  if (!identical(dim(L), c(k, k)))
    stop("`L` must be k x k where k = ncol(z).", call. = FALSE)
  if (length(log_scale) == 1L) log_scale <- rep(log_scale, k)
  if (length(log_scale) != k)
    stop("`log_scale` must be length 1 or k.", call. = FALSE)

  theta <- sweep(z %*% t(L), 2L, theta_hat, FUN = "+")
  if (any(log_scale)) {
    cols <- which(log_scale)
    theta[, cols] <- exp(theta[, cols])
  }
  unname(theta)
}
