# gcol33/tulpa#900: an areal unit id given as labels is matched to the graph's
# nodes by rownames(adjacency), never by the labels' sort order, and a time id
# given as labels is put in time order, at every door.

chain_adj <- function(n, labels = NULL) {
  A <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- 1
  if (!is.null(labels)) rownames(A) <- colnames(A) <- labels
  A
}

test_that("labelled unit ids are matched to the adjacency by name", {
  lab <- paste0("r", 1:12)
  A <- chain_adj(12, lab)
  vals <- c("r1", "r10", "r2", "r12", "r11")
  expect_identical(tulpa:::.resolve_spatial_idx(vals, 12L, A, "region"),
                   c(1L, 10L, 2L, 12L, 11L))
  # A factor's level order does not move a named match.
  expect_identical(
    tulpa:::.resolve_spatial_idx(factor(vals, levels = rev(lab)), 12L, A, "region"),
    c(1L, 10L, 2L, 12L, 11L))
  expect_error(tulpa:::.resolve_spatial_idx(c("r1", "x9"), 12L, A, "region"),
               "not found in rownames")
})

test_that("unlabelled graphs take numeric labels as node numbers and refuse others", {
  A <- chain_adj(12)
  # "10" is node 10, not the second label in sort order.
  expect_identical(
    tulpa:::.resolve_spatial_idx(as.character(c(1, 10, 2, 12)), 12L, A, "region"),
    c(1L, 10L, 2L, 12L))
  expect_error(tulpa:::.resolve_spatial_idx(paste0("r", 1:12), 12L, A, "region"),
               "no rownames")
  # A factor states its order through its levels.
  f <- factor(paste0("r", 1:12), levels = paste0("r", 1:12))
  expect_identical(tulpa:::.resolve_spatial_idx(f, 12L, A, "region"), 1:12)
  expect_error(tulpa:::.resolve_spatial_idx(c(1, 1.5, 2), 12L, A, "region"),
               "whole-number")
})

test_that("labelled time ids are put in time order", {
  tv <- as.character(c(1, 10, 2, 9, 25))
  f <- tulpa:::.resolve_time_index(tv, "time")
  expect_identical(levels(f), c("1", "2", "9", "10", "25"))
  expect_identical(as.integer(f), c(1L, 4L, 2L, 3L, 5L))
  expect_error(tulpa:::.resolve_time_index(c("Jan", "Feb"), "month"),
               "carry no time order")
  fo <- factor(c("b", "a"), levels = c("b", "a"))
  expect_identical(tulpa:::.resolve_time_index(fo, "t"), fo)
  spec <- validate_temporal(temporal_rw2("time"),
                            data.frame(time = as.character(1:25)))
  expect_identical(spec$time_levels, as.character(1:25))
  expect_identical(spec$time_index, 1:25)
})

test_that("tulpa() attaches labelled observations to the named nodes", {
  skip_on_cran()
  set.seed(1)
  lab <- paste0("r", 1:12)
  A <- chain_adj(12, lab)
  d <- data.frame(region = paste0("r", rep(1:12, each = 3)), x = rnorm(36))
  d$y <- d$x + rnorm(36)
  f <- tulpa(y ~ x + spatial(region), d, phi = 1, mode = "laplace",
             spatial = spatial_car(A, group_var = "region"))
  expect_identical(rownames(A)[f$spatial$spatial_idx], d$region)
  # The same data without rownames has no way to place the labels.
  expect_error(
    tulpa(y ~ x + spatial(region), d, phi = 1, mode = "laplace",
          spatial = spatial_car(chain_adj(12), group_var = "region")),
    "no rownames")
})

test_that("a labelled lattice recovers the same field as integer ids", {
  skip_on_cran()
  nr <- 8L; S <- nr * nr; A <- matrix(0, S, S)
  id <- function(i, j) (j - 1L) * nr + i
  for (i in 1:nr) for (j in 1:nr) {
    if (i < nr) A[id(i, j), id(i + 1L, j)] <- A[id(i + 1L, j), id(i, j)] <- 1
    if (j < nr) A[id(i, j), id(i, j + 1L)] <- A[id(i, j + 1L), id(i, j)] <- 1
  }
  set.seed(5)
  Q <- diag(rowSums(A)) - A + diag(1e-3, S)
  phi <- as.numeric(backsolve(chol(Q), rnorm(S))); phi <- phi - mean(phi)
  d <- data.frame(node = rep(seq_len(S), each = 3L), x = rnorm(3L * S))
  d$y <- 0.5 * d$x + phi[d$node] + rnorm(nrow(d), sd = 0.3)
  lab <- sprintf("cell%d", seq_len(S))
  Al <- A; rownames(Al) <- colnames(Al) <- lab
  d$cell <- lab[d$node]
  f_int <- tulpa(y ~ x + spatial(node), d, phi = 0.09, mode = "laplace",
                 spatial = spatial_car(A, group_var = "node"))
  f_lab <- tulpa(y ~ x + spatial(cell), d, phi = 0.09, mode = "laplace",
                 spatial = spatial_car(Al, group_var = "cell"))
  expect_identical(f_lab$spatial$spatial_idx, f_int$spatial$spatial_idx)
  expect_equal(unname(coef(f_lab)), unname(coef(f_int)), tolerance = 1e-8)
})
