# The GP Gibbs sweep draws the field from its FULL conditional (gcol33/tulpa#427).
#
# The NNGP joint density factorizes over the Vecchia ordering as
#   p(w) = prod_i N(w_i | B_i' w_{N(i)}, sigma2 F_i),
# so the conditional of w_i given the rest carries its own factor AND every
# factor j for which i is a parent. The sweep used to carry the first alone,
# which under a nearest-neighbour ordering leaves most of the field
# under-constrained -- and the sigma2 and phi conditionals downstream then read
# a field drawn from the wrong law.
#
# The claim is a matrix identity: the pair the sweep draws from is row i of
# Lambda = (I - A)' D^-1 (I - A), the NNGP joint precision. The dense Lambda is
# assembled here from the kriging weights and conditional variances the probe
# hands back, so the arbiter is the definition rather than a second traversal of
# the same child lists.

.pgnngp_probe <- function(N = 20L, nn = 3L, seed = 4L, sigma2 = 0.7, phi = 0.35,
                          cov_type = 0L) {
  set.seed(seed)
  coords <- cbind(runif(N), runif(N))
  ni     <- compute_nngp_neighbors(coords, nn)
  order0 <- if (!is.null(ni$nn_order)) as.integer(ni$nn_order - 1L) else seq_len(N) - 1L
  w      <- rnorm(N)
  r <- cpp_test_pg_nngp_conditional(coords, ni$nn_idx, ni$nn_dist, order0,
                                    N, nn, w, sigma2, phi, cov_type)
  c(r, list(N = N, w = w, sigma2 = sigma2))
}

# Lambda in ORDERED coordinates, from the probe's own B / F.
.pgnngp_lambda <- function(r) {
  N <- r$N; nn <- r$nn
  A <- matrix(0, N, N)
  for (i in seq_len(N)) {
    m <- r$cnt[i]
    if (m > 0) for (t in seq_len(m)) {
      A[i, r$parent_pos[(i - 1L) * nn + t] + 1L] <- r$B[(i - 1L) * nn + t]
    }
  }
  t(diag(N) - A) %*% diag(1 / (r$sigma2 * r$F), N) %*% (diag(N) - A)
}

test_that("the field conditional is a row of the NNGP joint precision", {
  for (cov_type in 0:2) {
    r     <- .pgnngp_probe(cov_type = as.integer(cov_type))
    Lam   <- .pgnngp_lambda(r)
    w_ord <- r$w[r$orig + 1L]

    expect_equal(r$prec, diag(Lam), tolerance = 1e-12,
                 info = paste("precision, cov_type", cov_type))
    expect_equal(r$mean_num,
                 vapply(seq_len(r$N),
                        function(i) -sum(Lam[i, -i] * w_ord[-i]), numeric(1)),
                 tolerance = 1e-10,
                 info = paste("mean numerator, cov_type", cov_type))
  }
})

test_that("the child terms are most of the field, not an edge case", {
  # The negative control: without them the precision is the own factor alone.
  # Under a nearest-neighbour ordering nearly every location is somebody's
  # parent, so the omission was the rule rather than the exception.
  r <- .pgnngp_probe()
  parent_only <- 1 / (r$sigma2 * r$F)
  expect_gte(sum(r$prec > parent_only + 1e-10), 0.8 * r$N)
  expect_gt(mean(abs(r$prec - parent_only)), 0.5)
  # Every child term is a positive contribution to the precision, so the
  # parent-only reading can only ever be too small.
  expect_true(all(r$prec >= parent_only - 1e-12))
})

test_that("the assembled precision is the symmetric PD matrix the identity needs", {
  # If Lambda were not symmetric the row read and the column read would differ
  # and the comparison above would not be well posed.
  r   <- .pgnngp_probe()
  Lam <- .pgnngp_lambda(r)
  expect_lt(max(abs(Lam - t(Lam))), 1e-12)
  expect_gt(min(eigen(Lam, symmetric = TRUE, only.values = TRUE)$values), 0)
})

test_that("sigma2 scales the conditional the way the factorization says", {
  # F carries sigma2 linearly and B is scale-free, so Lambda is proportional to
  # 1 / sigma2 and both moments scale with it exactly.
  a <- .pgnngp_probe(sigma2 = 0.7)
  b <- .pgnngp_probe(sigma2 = 2.8)
  expect_equal(b$prec * 4, a$prec, tolerance = 1e-10)
  expect_equal(b$mean_num * 4, a$mean_num, tolerance = 1e-10)
})

test_that("a location with no parents and no children keeps its marginal", {
  # nn = 0 makes every factor marginal: Lambda is diagonal at 1 / sigma2 and the
  # mean numerator is zero everywhere.
  r <- .pgnngp_probe(nn = 0L)
  expect_equal(r$prec, rep(1 / r$sigma2, r$N), tolerance = 1e-12)
  expect_equal(r$mean_num, rep(0, r$N), tolerance = 1e-12)
})
