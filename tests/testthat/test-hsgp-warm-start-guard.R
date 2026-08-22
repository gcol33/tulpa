# The HSGP warm start reads its indices, not only its flag (gcol33/tulpa#428).
#
# layout.is_hsgp follows data.spatial_type alone; the three HSGP index fields
# are assigned only when data.has_hsgp is set as well. A model that declares an
# HSGP field and carries no basis block therefore reaches warm_start_mass_matrix
# with the flag true and every index at its -1 sentinel, and the two scalar
# writes there used to be unguarded: inv_m[-1] = 1.0, an 8-byte store
# immediately before the vector's allocation.
#
# It is a heap buffer underflow rather than a wrong number, so the write lands
# in the allocator's bookkeeping and fails silently on some allocators and far
# from the write site on others. What R can pin is that the state is reachable
# at all -- no fitted path builds it, since the front door builds the flag and
# the basis together -- and that the call now returns with the diagonal intact.
# The rest of the same function already pairs every nullable index with a >= 0
# check; HSGP was the one unguarded pair.

test_that("the flag and the indices can disagree", {
  r <- cpp_test_hsgp_warm_start(has_hsgp = FALSE)
  expect_true(r$is_hsgp)
  expect_identical(r$sigma2_idx, -1L)
  expect_identical(r$ls_idx, -1L)
  # The basis-coefficient loop above the two scalar writes is harmless in the
  # same state: -1 < -1 is false, so it never runs.
  expect_identical(r$beta_start, -1L)
  expect_identical(r$beta_end, -1L)
  # ... and the indices the writes used are not positions this vector has.
  expect_gt(2L, r$total_params - 1L)
})

test_that("the warm start leaves that model's diagonal intact", {
  r <- cpp_test_hsgp_warm_start(has_hsgp = FALSE)
  expect_length(r$inv_mass, r$total_params)
  expect_true(all(is.finite(r$inv_mass)))
  expect_true(all(is.finite(r$sqrt_mass)))
  expect_true(all(r$inv_mass > 0))
})

test_that("the ordinary HSGP model still gets its indices and its warm start", {
  # The control: with the basis present the flag and the indices agree, the two
  # hyperparameters have slots, and the guard is an exact no-op.
  r <- cpp_test_hsgp_warm_start(has_hsgp = TRUE, n_basis = 4L, n_spatial = 6L)
  expect_true(r$is_hsgp)
  expect_gte(r$sigma2_idx, 0L)
  expect_gte(r$ls_idx, 0L)
  expect_lt(r$sigma2_idx, r$total_params)
  expect_lt(r$ls_idx, r$total_params)
  expect_identical(r$beta_end - r$beta_start, 4L)
  expect_length(r$inv_mass, r$total_params)
  expect_true(all(is.finite(r$inv_mass)))
})
