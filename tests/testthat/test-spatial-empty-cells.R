# Tests for the spatial_car / spatial_bym2 empty-cell support (issue #25).
# Cells with no observations must be allowed: the ICAR / CAR / BYM2 prior is
# well-defined on every node of the graph regardless of whether data touches
# it, so unobserved cells should just contribute no likelihood term (matching
# INLA's `f(cell, model = "besag", graph = g)` behavior).

make_3x3_adjacency <- function() {
  adj <- matrix(0L, 9, 9)
  # Rook + diagonal connectivity for a 3x3 grid
  for (i in 1:3) for (j in 1:3) {
    c1 <- (i - 1) * 3 + j
    for (di in -1:1) for (dj in -1:1) {
      if (di == 0 && dj == 0) next
      ii <- i + di; jj <- j + dj
      if (ii >= 1 && ii <= 3 && jj >= 1 && jj <= 3) {
        c2 <- (ii - 1) * 3 + jj
        adj[c1, c2] <- 1L
      }
    }
  }
  adj
}

test_that("validate_spatial accepts data with empty cells (integer group_var)", {
  adj <- make_3x3_adjacency()
  dat <- data.frame(cell_idx = c(1L, 2L, 4L, 5L, 8L), y = rnorm(5))
  spec <- spatial_car(adj, level = "group", group_var = "cell_idx")
  expect_silent(validate_spatial(spec, dat))
})

test_that("validate_spatial rejects out-of-range integer group_var", {
  adj <- make_3x3_adjacency()
  dat <- data.frame(cell_idx = c(1L, 2L, 4L, 10L), y = rnorm(4))
  spec <- spatial_car(adj, level = "group", group_var = "cell_idx")
  expect_error(validate_spatial(spec, dat), "outside \\[1, n_spatial_units = 9\\]")
})

test_that("validate_spatial accepts character group_var matching rownames(adjacency)", {
  adj <- make_3x3_adjacency()
  rownames(adj) <- paste0("c", 1:9)
  colnames(adj) <- paste0("c", 1:9)
  dat <- data.frame(cell_id = c("c1", "c2", "c4", "c5", "c8"), y = rnorm(5))
  spec <- spatial_car(adj, level = "group", group_var = "cell_id")
  expect_silent(validate_spatial(spec, dat))
})

test_that("validate_spatial rejects unknown character group_var values", {
  adj <- make_3x3_adjacency()
  rownames(adj) <- paste0("c", 1:9)
  colnames(adj) <- paste0("c", 1:9)
  dat <- data.frame(cell_id = c("c1", "c2", "c4", "c99"), y = rnorm(4))
  spec <- spatial_car(adj, level = "group", group_var = "cell_id")
  expect_error(validate_spatial(spec, dat), "not found in rownames\\(adjacency\\)")
})

test_that("validate_spatial keeps existing factor-without-rownames check", {
  adj <- make_3x3_adjacency()
  # Factor levels exactly match adjacency size — should pass (legacy behavior).
  dat <- data.frame(site = factor(seq_len(9)), y = rnorm(9))
  spec <- spatial_car(adj, level = "group", group_var = "site")
  expect_silent(validate_spatial(spec, dat))

  # Factor levels < adjacency size and no rownames — must error with an
  # actionable message pointing to the integer / rownames alternatives.
  dat2 <- data.frame(site = factor(c("a", "b", "c", "a", "b")), y = rnorm(5))
  expect_error(validate_spatial(spec, dat2),
               "integer 1-based indices.*rownames\\(adjacency\\)")
})

test_that("prior_from_spec builds correct spatial_idx with empty cells", {
  adj <- make_3x3_adjacency()
  dat <- data.frame(cell_idx = c(1L, 2L, 4L, 5L, 8L), y = rnorm(5))
  spec <- spatial_car(adj, level = "group", group_var = "cell_idx")
  prior <- prior_from_spec(spec, dat)
  expect_equal(prior$spatial_idx, c(1L, 2L, 4L, 5L, 8L))
  expect_equal(prior$n_spatial_units, 9L)
})

test_that("prior_from_spec preserves cell identity via rownames", {
  adj <- make_3x3_adjacency()
  rownames(adj) <- paste0("c", 1:9)
  colnames(adj) <- paste0("c", 1:9)
  dat <- data.frame(cell_id = c("c8", "c2", "c4"), y = rnorm(3))
  spec <- spatial_car(adj, level = "group", group_var = "cell_id")
  prior <- prior_from_spec(spec, dat)
  # cell_id "c8" -> row 8, "c2" -> row 2, "c4" -> row 4
  expect_equal(prior$spatial_idx, c(8L, 2L, 4L))
})

test_that("prior_from_spec for spatial_bym2 with empty cells", {
  adj <- make_3x3_adjacency()
  dat <- data.frame(cell_idx = c(1L, 5L, 9L), y = rnorm(3))
  spec <- spatial_bym2(adj, level = "group", group_var = "cell_idx")
  prior <- prior_from_spec(spec, dat)
  expect_equal(prior$spatial_idx, c(1L, 5L, 9L))
  expect_equal(prior$type, "bym2")
})
