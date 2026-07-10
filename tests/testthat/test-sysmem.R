# System-memory queries and the outer-grid thread memory budget.
#
# The nested-Laplace joint grid caps its outer OpenMP width so the replicated
# per-thread working set fits available memory, then warns when it has to
# reduce the requested thread count (or, in the floor case, run best-effort at
# one thread). The clamp must budget against the memory that is actually FREE,
# not the installed total: half of a 64 GB install is 32 GB even when only 40 GB
# is free and 23 GB is already committed, which is what pushed a wide fit into
# swap / OOM. These tests pin (a) the RAM queries return sane byte counts and
# (b) the budget arithmetic tracks available RAM, floors at one thread, and is
# monotone in the per-thread footprint.

test_that("RAM queries return sane byte counts", {
  total <- tulpa:::cpp_total_ram_bytes()
  avail <- tulpa:::cpp_available_ram_bytes()

  expect_true(is.numeric(total) && length(total) == 1L)
  expect_true(is.numeric(avail) && length(avail) == 1L)
  expect_false(is.na(total))
  expect_false(is.na(avail))
  expect_gte(total, 0)
  expect_gte(avail, 0)

  # Free memory can never exceed the installed total (when both are detected).
  if (total > 0 && avail > 0) {
    expect_lte(avail, total)
  }
})

test_that("RAM queries resolve on this platform", {
  skip_on_cran()
  # Windows / Linux / macOS all implement both queries; a 0 here means the
  # platform query failed, which we want to catch in development rather than
  # silently fall back to the fixed 2 GB cap.
  total <- tulpa:::cpp_total_ram_bytes()
  avail <- tulpa:::cpp_available_ram_bytes()
  expect_gt(total, 1024^3)  # > 1 GB installed
  expect_gt(avail, 0)
})

test_that("outer-thread budget is sized against available RAM, not total", {
  gb <- 1024^3

  # The bug the clamp fixes: on a loaded box, free RAM is well below installed.
  # The budget must follow free RAM (a fraction of it), NOT half of the install.
  avail <- 40 * gb
  total <- 64 * gb
  budget <- tulpa:::cpp_test_outer_thread_mem_budget(avail, total)

  # Tracks available (0.6 * 40 = 24 GB), and is strictly below half of total
  # (32 GB) -- so more free memory would be needed than the old total-based rule
  # assumed, which is exactly why the old rule over-provisioned.
  expect_equal(budget, 0.6 * avail, tolerance = 1e-6)
  expect_lt(budget, total / 2)
})

test_that("outer-thread budget falls back cleanly when a query is unavailable", {
  gb <- 1024^3

  # available unknown (0), total known -> half of total.
  expect_equal(
    tulpa:::cpp_test_outer_thread_mem_budget(0, 64 * gb),
    32 * gb, tolerance = 1e-6
  )
  # both unknown -> fixed 2 GB cap.
  expect_equal(
    tulpa:::cpp_test_outer_thread_mem_budget(0, 0),
    2 * gb, tolerance = 1e-6
  )
})

test_that("outer-thread cap fits the budget, floors at one, and is monotone", {
  gb <- 1024^3
  budget <- 24 * gb

  # Comfortable fit: many threads.
  expect_equal(tulpa:::cpp_test_outer_thread_cap(budget, 1 * gb), 24L)

  # A larger per-thread footprint yields fewer threads (monotone decreasing).
  cap_small <- tulpa:::cpp_test_outer_thread_cap(budget, 1 * gb)
  cap_big   <- tulpa:::cpp_test_outer_thread_cap(budget, 3 * gb)
  expect_lt(cap_big, cap_small)
  expect_equal(cap_big, 8L)

  # Floor case: a single thread's working set exceeds the budget -> still 1
  # (the driver runs best-effort and warns rather than refusing to fit).
  expect_equal(tulpa:::cpp_test_outer_thread_cap(budget, 40 * gb), 1L)

  # Degenerate per-thread footprint of 0 -> 1 (never divide by zero).
  expect_equal(tulpa:::cpp_test_outer_thread_cap(budget, 0), 1L)
})
