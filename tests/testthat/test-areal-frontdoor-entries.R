# What an areal spec states reaches the nested-Laplace block it becomes, or the
# call stops: its grid axes, its PC anchor, the BYM2 scaling, the unit labels
# of an adjacency() graph, and the outer-grid control knobs.

.fe_lattice <- function(k = 5L) {
  K <- k * k
  A <- matrix(0, K, K)
  for (i in seq_len(k)) for (j in seq_len(k)) {
    s <- (i - 1L) * k + j
    if (j < k) A[s, s + 1L] <- A[s + 1L, s] <- 1
    if (i < k) A[s, s + k] <- A[s + k, s] <- 1
  }
  A
}

.fe_data <- function(seed = 2) {
  set.seed(seed)
  A <- .fe_lattice(); K <- nrow(A)
  n <- 150L; reg <- sample(seq_len(K), n, TRUE); x <- rnorm(n)
  u <- 0.5 * as.numeric(scale(cumsum(rnorm(K))))
  list(A = A, d = data.frame(y = rpois(n, exp(0.3 + 0.4 * x + u[reg])),
                             x = x, region = reg))
}

.fe_ctl <- list(progress = FALSE)

test_that("check_adjacency() returns its report visibly and prints nothing", {
  A <- .fe_lattice(3L)
  out <- capture.output(r <- withVisible(check_adjacency(A)))
  expect_length(out, 0L)
  expect_true(r$visible)
  expect_s3_class(r$value, "tulpa_adjacency_check")
  expect_output(print(r$value), "adjacency check")
})

test_that("an adjacency() graph names its matrix by the unit identifiers", {
  ids <- as.character(37000 + 3 * seq_len(9))
  g <- adjacency(data.frame(x = rep(1:3, 3), y = rep(1:3, each = 3), id = ids),
                 id = "id", type = "rook")
  expect_identical(rownames(g$adjacency), ids)
  expect_identical(colnames(as.matrix(g$adjacency)), ids)
  lab <- factor(ids[c(9, 1, 5, 5)])
  expect_identical(
    tulpa:::.resolve_spatial_idx(lab, 9L, as.matrix(g$adjacency), "fips"),
    c(9L, 1L, 5L, 5L))
})
