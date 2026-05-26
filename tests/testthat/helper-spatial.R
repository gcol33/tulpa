# helper-spatial.R
#
# Shared fixtures for the spatial test files (test-gibbs-spatial.R,
# test-tulpa-spatial-frontdoor.R). Sourced by testthat ahead of every test file.

# Rook-neighbour adjacency for an nr x nc grid (1-based units, column-major).
rook_adj <- function(nr, nc) {
  n <- nr * nc
  A <- matrix(0L, n, n)
  idx <- function(r, c) (c - 1L) * nr + r
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    i <- idx(r, c)
    if (r > 1L)  A[i, idx(r - 1L, c)] <- 1L
    if (r < nr)  A[i, idx(r + 1L, c)] <- 1L
    if (c > 1L)  A[i, idx(r, c - 1L)] <- 1L
    if (c < nc)  A[i, idx(r, c + 1L)] <- 1L
  }
  A
}
