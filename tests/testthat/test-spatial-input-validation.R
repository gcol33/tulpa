# Input validation at the spatial / temporal constructors and the SPDE fit
# doors (gcol33/tulpa#908, gcol33/tulpa#909). Each input here is wrong or an
# edge case, and each used to reach internal code -- a C++ index check, the
# vendored triangulation, a foreign-function call on NaN -- instead of an error
# naming the argument. The assertions are on the message, so a regression back
# to the internal one fails.

.chain_adj <- function(n) {
  A <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- 1
  A
}

test_that("an adjacency with a missing or negative entry is refused (#909)", {
  A <- .chain_adj(5L)
  An <- A; An[1, 2] <- NA
  expect_error(spatial_car(An, group_var = "r"), "missing or non-finite")
  expect_error(spatial_bym2(An, group_var = "r"), "missing or non-finite")
  expect_warning(r <- check_adjacency(An), "no further check")
  expect_false(r$finite)
  expect_false(r$ok)

  Ai <- A; Ai[2, 3] <- Ai[3, 2] <- Inf
  expect_error(spatial_car(Ai, group_var = "r"), "non-finite")

  Aneg <- A; Aneg[1, 2] <- Aneg[2, 1] <- -1
  expect_error(spatial_car(Aneg, group_var = "r"), "2 negative entries")
  expect_warning(r <- check_adjacency(Aneg), "negative")
  expect_false(r$nonneg)
})

test_that("check_adjacency reports connected components (#902)", {
  A <- .chain_adj(4L)
  r <- check_adjacency(A)
  expect_identical(r$n_components, 1L)
  expect_true(r$ok)

  # Two chains of three and an island: two multi-node components are named,
  # the island separately, and the graph is no longer "ok".
  B <- as.matrix(Matrix::bdiag(.chain_adj(3L), .chain_adj(3L), matrix(0, 1, 1)))
  expect_warning(expect_warning(r2 <- check_adjacency(B), "isolated"),
                 "2 connected components of more than one node")
  expect_identical(r2$n_components, 3L)
  expect_identical(sort(r2$component_sizes), c(1L, 3L, 3L))
  expect_false(r2$ok)
  expect_output(print(r2), "components: 3 \\(sizes 3, 3, 1\\)")

  # The component labeller is the one .graph_components() reads: an edge kept
  # in one triangle only still joins its endpoints.
  U <- Matrix::Matrix(B, sparse = TRUE)
  U[lower.tri(U)] <- 0
  expect_identical(tulpa:::.graph_components(U),
                   list(1:3, 4:6, 7L))
})

test_that("coordinates must be finite and, when scaled, not constant (#909)", {
  set.seed(1)
  d <- data.frame(lon = runif(20), lat = 1, x = rnorm(20), y = rnorm(20))
  expect_error(validate_gp(spatial_gp(~ lon + lat, nn = 5), d),
               "coordinate `lat` is constant")
  expect_error(
    tulpa:::validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", m = 5), d),
    "coordinate `lat` is constant")
  # Unscaled, a constant column is only a degenerate axis, not an error.
  expect_no_error(validate_gp(spatial_gp(~ lon + lat, nn = 5,
                                         scale_coords = FALSE), d))

  d$lat <- runif(20); d$lat[3] <- Inf
  expect_error(validate_gp(spatial_gp(~ lon + lat, nn = 5), d),
               "non-finite value\\(s\\).*first at row 3")
  d$lat[3] <- NA
  expect_error(spatial_spde(~ lon + lat, data = d),
               "spatial_spde\\(\\).*missing or non-finite")
})

test_that("compute_nngp_neighbors() refuses a k no ordering can fill (#909)", {
  set.seed(2)
  co <- cbind(runif(10), runif(10))
  expect_error(compute_nngp_neighbors(co, 20), "`k` must be a whole number in \\[0, 9\\]")
  expect_error(compute_nngp_neighbors(co, -1), "\\[0, 9\\]")
  expect_error(compute_nngp_neighbors(co, 2.5), "whole number")
  co_na <- co; co_na[2, 1] <- NA
  expect_error(compute_nngp_neighbors(co_na, 3), "first at row 2")
  # nn_idx is 1-based: 0 marks an empty slot, never a neighbour.
  nb <- compute_nngp_neighbors(co, 3)
  expect_true(all(nb$nn_idx[1, ] == 0L))
  expect_true(all(nb$nn_idx[-1, 1] >= 1L))
  expect_identical(nb$k, 3L)
  # k = 0 is the independent field, and a lone location conditions on nothing.
  expect_true(all(compute_nngp_neighbors(co, 0)$nn_idx == 0L))
  expect_no_error(compute_nngp_neighbors(co[1, , drop = FALSE], 0))
})

test_that("a temporal field needs every time and enough of them (#909)", {
  d <- data.frame(t = 1, x = rnorm(6), y = rnorm(6))
  expect_error(validate_temporal(temporal_rw1("t"), d),
               "RW1 requires at least 2 distinct time points")
  expect_error(validate_temporal(temporal_ar1("t"), d),
               "AR1 requires at least 2")
  d2 <- data.frame(t = c(1:5, NA), x = rnorm(6), y = rnorm(6))
  expect_error(validate_temporal(temporal_rw1("t"), d2),
               "'t' has 1 missing value\\(s\\) \\(first at row 6\\)")
  d3 <- data.frame(t = rep(1:5, 4), x = rnorm(20))
  d3$t[3] <- NA
  expect_error(
    tulpa:::validate_tvc(temporal_tvc("t", terms = "x"), d3,
                         cbind(`(Intercept)` = 1, x = d3$x)),
    "'t' has 1 missing value")
})

test_that("spatial_spde() meshes repeated sites once and keeps every row (#909)", {
  set.seed(1)
  sites <- data.frame(lon = runif(25), lat = runif(25))
  d <- sites[rep(seq_len(25L), each = 2L), ]
  # The triangulation used to stop with "Duplicate vertex detected".
  sp <- spatial_spde(~ lon + lat, data = d)
  expect_identical(nrow(sp$A), 50L)
  expect_equal(unname(Matrix::rowSums(sp$A)), rep(1, 50))
  # The two observations at one site read the same projector row.
  expect_equal(as.matrix(sp$A[1, , drop = FALSE]),
               as.matrix(sp$A[2, , drop = FALSE]))
})

test_that("fit_spde() refuses a response whose length is not A's row count (#908)", {
  set.seed(1)
  d <- data.frame(lon = runif(40), lat = runif(40), x = rnorm(40))
  d$y <- rpois(40, 2)
  sp <- spatial_spde(~ lon + lat, data = d)
  # Longer: used to fit, with the extra rows given no field term.
  expect_error(fit_spde(rep(d$y, 2), cbind(1, rep(d$x, 2)), sp,
                        family = "poisson"),
               "A has 40 row\\(s\\) but the response has 80")
  # Shorter: used to reach the kernel's CSC check as a raw index error.
  expect_error(fit_spde(d$y[1:10], cbind(1, d$x[1:10]), sp, family = "poisson"),
               "A has 40 row\\(s\\) but the response has 10")
  expect_error(fit_spde(d$y[1:10], cbind(1, d$x[1:10]), sp, family = "poisson",
                        mode = "nuts"),
               "A has 40 row\\(s\\) but the response has 10")
  expect_error(fit_spde(d$y, cbind(1, d$x)[1:10, ], sp, family = "poisson"),
               "nrow\\(X\\) \\(10\\) must equal length\\(y\\) \\(40\\)")
})
