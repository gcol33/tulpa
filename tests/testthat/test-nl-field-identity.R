# NlFieldIdentity reproduces the hand-folded structural fingerprint bit for bit
# (gcol33/tulpa#286).
#
# The eleven cpp_nested_laplace_* entry points each folded their own structural
# fingerprint, which put the same byte-fold loops in eleven places. That
# fingerprint keys the grid checkpoint, so its value is a contract in both
# directions: a seed that shifts invalidates every checkpoint on disk, and a
# seed that stops distinguishing two structures makes a resumed run reuse cells
# computed under different inputs -- which passes every shape and finiteness
# check. cpp_test_nl_field_seed folds each group both ways and these tests
# require the two to agree.

seed_pair <- function(kind, tag,
                      n_spatial_units = 4L, scale_factor = 0.7,
                      adj_row_ptr = c(0L, 2L, 4L, 6L, 8L),
                      adj_col_idx = c(1L, 2L, 0L, 3L, 0L, 3L, 1L, 2L),
                      n_spatial = 5L, nn = 3L, cov_type = 1L,
                      coords = matrix(c(0.1, 0.4, 0.6, 0.2, 0.9,
                                        0.3, 0.8, 0.1, 0.5, 0.7), ncol = 2),
                      nn_idx = matrix(1:15, nrow = 5),
                      spatial_idx = c(1L, 2L, 3L, 4L, 5L, 1L),
                      M = 6L,
                      phi_basis = matrix(seq(0.01, 0.36, length.out = 36), nrow = 6),
                      lambda_eig = seq(0.5, 3.0, length.out = 6),
                      temporal_type = "ar1", n_times = 4L, cyclic = FALSE,
                      temporal_idx = c(1L, 2L, 3L, 4L, 1L, 2L),
                      n_groups = 2L, with_groups = FALSE) {
  cpp_test_nl_field_seed(kind, tag, n_spatial_units, scale_factor,
                         adj_row_ptr, adj_col_idx,
                         n_spatial, nn, cov_type, coords, nn_idx, spatial_idx,
                         M, phi_basis, lambda_eig,
                         temporal_type, n_times, cyclic, temporal_idx,
                         n_groups, with_groups)
}

KINDS <- list(
  list(kind = "areal",                 tag = "icar"),
  list(kind = "areal_scaled",          tag = "bym2"),
  list(kind = "areal",                 tag = "car_proper"),
  list(kind = "nngp",                  tag = "nngp"),
  list(kind = "hsgp",                  tag = "hsgp"),
  list(kind = "temporal",              tag = "temporal", with_groups = TRUE),
  list(kind = "areal+temporal",        tag = "st_icar"),
  list(kind = "areal+temporal",        tag = "st_car_proper"),
  list(kind = "areal_scaled+temporal", tag = "st_bym2"),
  list(kind = "hsgp+temporal",         tag = "st_hsgp"),
  list(kind = "nngp+temporal",         tag = "st_nngp")
)

test_that("every field model's builder seed matches the hand-folded reference", {
  for (k in KINDS) {
    got <- do.call(seed_pair, k)
    expect_identical(unname(got[["builder"]]), unname(got[["reference"]]),
                     info = k$tag)
  }
})

test_that("the eleven field models do not collide", {
  seeds <- vapply(KINDS, function(k) do.call(seed_pair, k)[["builder"]], "")
  expect_length(unique(seeds), length(KINDS))
})

test_that("changing any structural input moves the seed", {
  base <- seed_pair("areal+temporal", "st_icar")[["builder"]]

  # unit count
  expect_false(identical(
    seed_pair("areal+temporal", "st_icar", n_spatial_units = 5L)[["builder"]],
    base))
  # the adjacency itself, same length
  expect_false(identical(
    seed_pair("areal+temporal", "st_icar",
              adj_col_idx = c(1L, 3L, 0L, 3L, 0L, 3L, 1L, 2L))[["builder"]],
    base))
  # temporal structure, length, closure, node map
  expect_false(identical(
    seed_pair("areal+temporal", "st_icar", temporal_type = "rw2")[["builder"]],
    base))
  expect_false(identical(
    seed_pair("areal+temporal", "st_icar", n_times = 5L)[["builder"]],
    base))
  expect_false(identical(
    seed_pair("areal+temporal", "st_icar", cyclic = TRUE)[["builder"]],
    base))
  expect_false(identical(
    seed_pair("areal+temporal", "st_icar",
              temporal_idx = c(1L, 2L, 3L, 4L, 2L, 1L))[["builder"]],
    base))
})

test_that("the BYM2 mixing scale is part of the identity", {
  a <- seed_pair("areal_scaled", "bym2", scale_factor = 0.7)[["builder"]]
  b <- seed_pair("areal_scaled", "bym2", scale_factor = 0.9)[["builder"]]
  expect_false(identical(a, b))

  # and a scaled areal field is not the same identity as an unscaled one with
  # the same adjacency -- the slot the scale occupies is what separates them
  unscaled <- seed_pair("areal", "bym2")[["builder"]]
  expect_false(identical(a, unscaled))
})

test_that("the NNGP neighbour graph and its kernel are part of the identity", {
  base <- seed_pair("nngp", "nngp")[["builder"]]
  expect_false(identical(
    seed_pair("nngp", "nngp", cov_type = 2L)[["builder"]], base))
  expect_false(identical(
    seed_pair("nngp", "nngp", nn = 4L)[["builder"]], base))
  expect_false(identical(
    seed_pair("nngp", "nngp",
              coords = matrix(c(0.1, 0.4, 0.6, 0.2, 0.95,
                                0.3, 0.8, 0.1, 0.5, 0.7), ncol = 2))[["builder"]],
    base))
})

test_that("the HSGP basis and its eigenvalues are part of the identity", {
  base <- seed_pair("hsgp", "hsgp")[["builder"]]
  expect_false(identical(seed_pair("hsgp", "hsgp", M = 7L)[["builder"]], base))
  expect_false(identical(
    seed_pair("hsgp", "hsgp",
              lambda_eig = seq(0.5, 3.5, length.out = 6))[["builder"]],
    base))
})

test_that("the temporal group count occupies its own slot", {
  # The standalone temporal field folds n_groups between n_times and cyclic; the
  # spatiotemporal ones fold no group count at all. Folding it in the wrong slot
  # would still separate the two, so check the with-groups form differs from the
  # without-groups form AND tracks the value.
  with_g   <- seed_pair("temporal", "temporal", with_groups = TRUE)[["builder"]]
  without  <- seed_pair("temporal", "temporal", with_groups = FALSE)[["builder"]]
  other_g  <- seed_pair("temporal", "temporal", with_groups = TRUE,
                        n_groups = 3L)[["builder"]]
  expect_false(identical(with_g, without))
  expect_false(identical(with_g, other_g))
})
