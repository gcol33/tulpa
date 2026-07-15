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

# spatial_car() / spatial_bym2() default to level = "group", which requires
# group_var; spatial() carries its grouping in the formula instead. The three
# are compared at the same grouping so the comparison is about the shared
# validation, not the signature.
car  <- function(W, ...) spatial_car(W, group_var = "g", ...)
bym2 <- function(W, ...) spatial_bym2(W, group_var = "g", ...)
fld  <- function(W, ...) spatial(W, ~ 1 | g, ...)

# A row of unit cells: rook adjacency on 1-D centroids is chain_W(n). Built
# through the documented door, since adjacency() constructs a graph from
# geometry and has no method for a bare matrix.
chain_adjacency <- function(n) {
  adjacency(data.frame(x = seq_len(n), y = 0), type = "rook")
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
  expect_warning(car(W, shared = FALSE), "Non-shared spatial effects")
  expect_warning(bym2(W, shared = FALSE), "Non-shared spatial effects")
  # spatial() silently accepted shared = FALSE.
  expect_warning(fld(W, shared = FALSE), "Non-shared spatial effects")
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
  ctors <- list(car = car, bym2 = bym2, fld = fld)
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
  ctors <- list(car = car, bym2 = bym2, fld = fld)

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
  # .validate_adjacency_arg() short-circuits on a tulpa_adjacency, so an object
  # adjacency() already vetted must not be re-checked or re-warned about.
  adj <- chain_adjacency(5)
  expect_s3_class(adj, "tulpa_adjacency")
  expect_equal(unname(as.matrix(adj$adjacency)), chain_W(5))
  expect_silent(car(adj))
  expect_silent(bym2(adj))
  expect_silent(fld(adj))
})

test_that("a float-rounded symmetric graph is accepted, matching check_adjacency", {
  # The inline checks demanded exact symmetry via isSymmetric(), so a graph
  # check_adjacency() accepts was rejected by the constructors. 1e-14 is inside
  # the 1e-8 tolerance the validator uses, so the graph must be accepted rather
  # than rejected on symmetry -- reaching the warning at all is the assertion.
  # The same perturbation puts one entry off exactly 1, which check_adjacency()
  # reports as a weighted graph as well, so the 0/1 warning is the two agreeing.
  W <- chain_W(5)
  W[1, 2] <- W[1, 2] + 1e-14
  expect_warning(car(W), "other than 0/1")
  expect_warning(fld(W), "other than 0/1")
})
