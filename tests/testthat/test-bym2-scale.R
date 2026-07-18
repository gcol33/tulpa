# compute_bym2_scale() must follow the Riebler et al. (2016) generalized-variance
# convention: rescale so the geometric mean of the ICAR marginal variances
# diag(Q^+) equals 1. Since the BYM2 field enters as scale_factor * phi, the
# returned factor is 1 / sqrt(geomean(diag(Q^+))). This pins the computed
# default (all fitting tests pass scale_factor = 1 explicitly and never exercise
# it), so a regression to the old geomean-of-eigenvalues form is caught.

# Independent reference: generalized variance from the pseudo-inverse diagonal.
.ref_bym2_scale <- function(adj) {
  adj <- as.matrix(adj); diag(adj) <- 0
  Q <- diag(rowSums(adj)) - adj
  e <- eigen(Q, symmetric = TRUE)
  nz <- abs(e$values) > 1e-10
  Qinv <- e$vectors[, nz, drop = FALSE] %*%
          (t(e$vectors[, nz, drop = FALSE]) / e$values[nz])
  1 / sqrt(exp(mean(log(diag(Qinv)))))
}

test_that("compute_bym2_scale matches the Riebler generalized-variance factor", {
  # 3-node path
  a3 <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3)
  expect_equal(compute_bym2_scale(a3), .ref_bym2_scale(a3), tolerance = 1e-8)

  # 4-node ring (each node has two neighbours) -- an irregular vs regular check
  a4 <- matrix(0, 4, 4)
  for (i in 1:4) { j <- i %% 4 + 1; a4[i, j] <- 1; a4[j, i] <- 1 }
  expect_equal(compute_bym2_scale(a4), .ref_bym2_scale(a4), tolerance = 1e-8)

  # 6-node irregular tree (where geomean-of-eigenvalues and the true generalized
  # variance genuinely diverge, so the old formula would fail this)
  a6 <- matrix(0, 6, 6)
  edges <- rbind(c(1, 2), c(2, 3), c(2, 4), c(4, 5), c(4, 6))
  for (r in seq_len(nrow(edges))) { i <- edges[r, 1]; j <- edges[r, 2]; a6[i, j] <- 1; a6[j, i] <- 1 }
  expect_equal(compute_bym2_scale(a6), .ref_bym2_scale(a6), tolerance = 1e-8)

  # The scaled ICAR field has unit geometric-mean marginal variance by
  # construction: scale^2 * geomean(diag(Q^+)) == 1.
  gv <- function(adj) { s <- .ref_bym2_scale(adj); 1 / (s * s) }  # == geomean(diag(Qinv))
  expect_equal(compute_bym2_scale(a6)^2 * gv(a6), 1, tolerance = 1e-8)
})
