#' @param phi Dispersion passed to the family, held fixed. One convention at
#'   every door: for `gaussian` / `lognormal` this is the residual VARIANCE
#'   (the SD is `sqrt(phi)`), for `neg_binomial_2` the size, `gamma` the
#'   shape, `beta` the precision, `t` the scale; `binomial` and `poisson`
#'   ignore it. The compiled kernels parameterize the two variance families by
#'   the residual SD and are handed `sqrt(phi)` at the boundary.
