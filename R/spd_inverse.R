# Inverse of a symmetric positive-definite matrix through its Cholesky factor.
#
# Every covariance read off a precision or a negative Hessian goes through
# here, so the not-positive-definite path is defined once: `chol()` stops with
# an error, which callers already catch the way they caught a singular
# `solve()`. A precision that is only indefinite (a saddle, a Hessian at a
# boundary) therefore fails rather than returning a "covariance" with negative
# variances. The input is symmetrized first, since a Hessian assembled by
# finite differences or by `optim()` is symmetric only to rounding, and the
# dimnames of `A` are carried over. A sparse `Matrix` input is factored sparse
# (a supernodal CHOLMOD LL') and solved against the identity, so its
# fill-reducing ordering is kept; the result is a dense covariance in every
# caller. CHOLMOD stops on a non-positive pivot the way `chol()` does.
.spd_inverse <- function(A) {
  if (inherits(A, "sparseMatrix")) {
    if (!length(A)) return(as.matrix(A))
    dn <- dimnames(A)
    S <- Matrix::forceSymmetric((A + Matrix::t(A)) / 2)
    ch <- Matrix::Cholesky(S, LDL = FALSE, super = TRUE, perm = TRUE)
    V <- as.matrix(Matrix::solve(ch, Matrix::Diagonal(nrow(S))))
    V <- (V + t(V)) / 2
    dimnames(V) <- dn
    return(V)
  }
  A <- as.matrix(A)
  if (!length(A)) return(A)
  dn <- dimnames(A)
  V <- chol2inv(chol((A + t(A)) / 2))
  dimnames(V) <- dn
  V
}
