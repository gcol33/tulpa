test_that("inference tier system is defined and every backend carries a valid tier", {
  reg <- tulpa:::BACKEND_REGISTRY
  expect_true(is.list(reg) && length(reg) > 0L)
  tiers <- vapply(reg, function(e)
    if (is.null(e$tier)) NA_character_ else e$tier, character(1))
  # Every registered backend carries one of the three correctness tiers.
  expect_false(anyNA(tiers))
  expect_true(all(tiers %in% c("exact", "structured", "optimized")))
  # Tier 1 (exact) and Tier 2 (structured) are both populated.
  expect_true(any(tiers == "exact"))
  expect_true(any(tiers == "structured"))
})

test_that("auto mode never selects a Tier 3 (VI / approximate) backend", {
  # Design invariant: auto may choose only Tier 1 (exact) or Tier 2 (structured),
  # never the approximate-MCMC / VI backends (registry tier "optimized"), which
  # carry no correctness guarantee and are explicit opt-in only.
  reg  <- tulpa:::BACKEND_REGISTRY
  auto <- tulpa:::get_mode_backends("auto")
  optimized <- names(Filter(function(e) identical(e$tier, "optimized"), reg))

  expect_true("vi" %in% optimized)               # VI is in the opt-in tier
  expect_false("vi" %in% auto)                    # and auto never picks it
  expect_length(intersect(auto, optimized), 0L)   # nor any other approximate one
})


test_that("every non-NULL registry $cabi name resolves as a registered C-ABI callable (gcol33/tulpa#775)", {
  # $cabi is documented as "the registered C-ABI callable backing the
  # backend" -- an R_RegisterCCallable() name, not merely an Rcpp export (an
  # R-level entry point with no such registration). Wrong values are metadata
  # today (assert_backend_reachable() is the only reader, and it only fires
  # for a backend with no R fitter, which none currently are), so nothing
  # catches drift without this test.
  reg <- tulpa:::BACKEND_REGISTRY
  for (backend in names(reg)) {
    cabi <- reg[[backend]]$cabi
    if (is.null(cabi)) next
    for (nm in cabi) {
      resolved <- isTRUE(tryCatch(tulpa:::cpp_test_ccallable_resolves(nm),
                                   error = function(e) FALSE))
      expect_true(resolved, info = sprintf("%s: cabi '%s'", backend, nm))
    }
  }
})


test_that("the unknown-mode message names every mode the registry defines", {
  # The message and the sibling that validates the same names both have to come
  # off INFERENCE_TIERS; a restated list goes stale the moment a tier is added.
  msg <- tryCatch(tulpa:::get_mode_backends("no-such-mode"),
                  error = conditionMessage)
  for (m in c("auto", names(tulpa:::INFERENCE_TIERS))) {
    expect_true(grepl(m, msg, fixed = TRUE), info = m)
  }
})

test_that("auto routes a continuous spatial field + (1 | g) to hmc, and explicit nested_laplace refuses it up front (gcol33/tulpa#794)", {
  skip_on_cran()
  set.seed(2)
  L <- cbind(lon = runif(60, 0, 10), lat = runif(60, 0, 10))
  g <- data.frame(L, x = rnorm(60), g = rep(1:5, 12))
  g$y <- rpois(60, exp(0.3 + 0.5 * g$x))

  fA <- suppressWarnings(tulpa(
    y ~ x + (1 | g), data = g, family = "poisson",
    spatial = spatial_gp(~ lon + lat, nn = 6)))
  expect_identical(fA$backend, "hmc")

  expect_error(
    tulpa(y ~ x + (1 | g), data = g, family = "poisson", mode = "nested_laplace",
          spatial = spatial_gp(~ lon + lat, nn = 6)),
    "nested_laplace")

  # An areal field + RE is unaffected -- it stays single-block-prior handling.
  W <- adjacency(expand.grid(x = 1:4, y = 1:4), x_coord = "x", y_coord = "y",
                 type = "rook")$adjacency
  d <- data.frame(region = rep(1:16, each = 4), g2 = rep(1:4, each = 16),
                  x = rnorm(64))
  d$y <- rpois(64, exp(0.3 + 0.5 * d$x))
  fI <- suppressWarnings(tulpa(
    y ~ x + (1 | g2) + spatial(region), data = d, family = "poisson",
    mode = "nested_laplace", spatial = list(type = "icar", adjacency = W)))
  expect_identical(fI$backend, "nested_laplace")
})
