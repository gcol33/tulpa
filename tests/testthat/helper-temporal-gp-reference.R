temporal_gp_reference_rotation <- function(L) {
  v <- forwardsolve(L, rep(1,nrow(L)))
  v <- v/v[1]
  u <- v
  u[1] <- u[1]+sqrt(sum(v*v))
  diag(length(v))-2*tcrossprod(u)/sum(u*u)
}
