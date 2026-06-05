# Orthogonal projection onto the complement of col(X): P_perp = I - Q Q',
# Q an orthonormal basis for col(X) from the QR decomposition (more stable than
# a direct inverse). Single source of truth for the restricted spatial (RSR)
# and restricted temporal (RTR) regression projections.
.orthogonal_complement_projection <- function(X, label) {
  n <- nrow(X)
  p <- ncol(X)

  if (p >= n) {
    warning("More covariates than observations; ", label,
            " may not be effective", call. = FALSE)
  }

  Q <- qr.Q(qr(X))
  diag(n) - Q %*% t(Q)
}
