# Cross-constructor consistency for the field constructors (gcol33/tulpa#142 A8).
#
# Each field constructor carried its own copy of the variable coercion, the
# `shared = FALSE` warning, and the adjacency validation. The copies drifted:
# temporal_ar1() and spatial() never warned about `shared = FALSE` while their
# siblings did, and the inline adjacency checks were strictly weaker than the
# validator adjacency() / check_adjacency() already use. These tests assert the
# behaviour is the same whichever constructor a user reaches for.

chain_W <- function(n) {
  W <- matrix(0, n, n)
  for (i in seq_len(n - 1)) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
  W
}

test_that("every temporal constructor warns on shared = FALSE", {
  # temporal_ar1() silently accepted shared = FALSE while rw1/rw2 warned.
  for (f in list(temporal_rw1, temporal_rw2, temporal_ar1)) {
    expect_warning(f("year", shared = FALSE), "Non-shared temporal effects")
  }
  # and none warn when shared is left alone
  for (f in list(temporal_rw1, temporal_rw2, temporal_ar1)) {
    expect_silent(f("year"))
  }
})

test_that("every spatial constructor warns on shared = FALSE", {
  W <- chain_W(5)
  expect_warning(spatial_car(W, shared = FALSE), "Non-shared spatial effects")
  expect_warning(spatial_bym2(W, shared = FALSE), "Non-shared spatial effects")
  # spatial() silently accepted shared = FALSE.
  expect_warning(spatial(W, ~ 1 | g, shared = FALSE), "Non-shared spatial effects")
})

test_that("every temporal constructor coerces formula and string alike", {
  for (f in list(temporal_rw1, temporal_rw2, temporal_ar1)) {
    expect_identical(f(~ year)$time_var, "year")
    expect_identical(f("year")$time_var, "year")
    expect_identical(f("year", group_var = ~ site)$group_var, "site")
    expect_error(f(~ year + month), "exactly 1 variable")
    expect_error(f(1L), "single character string")
  }
})

test_that("the spatial constructors reject the same bad graphs", {
  ctors <- list(
    car  = function(W) spatial_car(W),
    bym2 = function(W) spatial_bym2(W),
    fld  = function(W) spatial(W, ~ 1 | g)
  )
  not_square <- matrix(0, 3, 4)
  asym <- chain_W(4); asym[1, 2] <- 0   # break symmetry on one side

  for (nm in names(ctors)) {
    expect_error(ctors[[nm]](not_square), "square", info = nm)
    expect_error(ctors[[nm]]("not a matrix"), "matrix", info = nm)
    expect_error(ctors[[nm]](asym), "symmetric", info = nm)
  }
})

test_that("the spatial constructors surface the same structural issues as check_adjacency", {
  # These were accepted silently: only check_adjacency() reported them, so a
  # raw matrix handed straight to a constructor built an improper field with
  # no warning.
  ctors <- list(
    car  = function(W) spatial_car(W),
    bym2 = function(W) spatial_bym2(W),
    fld  = function(W) spatial(W, ~ 1 | g)
  )

  isolated <- chain_W(5); isolated[4, 5] <- 0; isolated[5, 4] <- 0
  selfloop <- chain_W(5); diag(selfloop) <- 1
  weighted <- chain_W(5); weighted[1, 2] <- 2.5; weighted[2, 1] <- 2.5

  for (nm in names(ctors)) {
    expect_warning(ctors[[nm]](isolated), "isolated node", info = nm)
    expect_warning(ctors[[nm]](selfloop), "self-loops", info = nm)
    expect_warning(ctors[[nm]](weighted), "other than 0/1", info = nm)
  }
})

test_that("a validated tulpa_adjacency passes the constructors without re-warning", {
  W <- chain_W(5)
  adj <- suppressWarnings(adjacency(W))
  expect_silent(spatial_car(adj))
  expect_silent(spatial_bym2(adj))
  expect_silent(spatial(adj, ~ 1 | g))
})

test_that("a float-rounded symmetric graph is accepted, matching check_adjacency", {
  # The inline checks demanded exact symmetry via isSymmetric(), so a graph
  # check_adjacency() accepts was rejected by the constructors.
  W <- chain_W(5)
  W[1, 2] <- W[1, 2] + 1e-14
  expect_silent(spatial_car(W))
  expect_silent(spatial(W, ~ 1 | g))
})
