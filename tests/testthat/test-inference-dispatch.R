# test-inference-dispatch.R
# Contract for the backend registry (single source of truth) and the
# dispatch spine: tier derivation, R-reachability, and fail-loud routing.

test_that("BACKEND_REGISTRY is the single source of truth for tiers", {
  # Every registry tier key is a known tier.
  for (b in names(BACKEND_REGISTRY)) {
    expect_true(BACKEND_REGISTRY[[b]]$tier %in% names(TIER_META),
                info = b)
  }
  # INFERENCE_TIERS membership is derived from the registry.
  for (tk in names(TIER_META)) {
    expect_setequal(INFERENCE_TIERS[[tk]]$backends, .tier_backends(tk))
  }
  # ALL_BACKENDS is exactly the registry keys.
  expect_setequal(ALL_BACKENDS, names(BACKEND_REGISTRY))
})

test_that("mclmc and smc are registered as Tier 1 (Exact)", {
  expect_true("mclmc" %in% INFERENCE_TIERS$exact$backends)
  expect_true("smc" %in% INFERENCE_TIERS$exact$backends)
  expect_equal(get_backend_tier("mclmc")$tier, 1L)
  expect_equal(get_backend_tier("smc")$tier, 1L)
})

test_that("get_backend_tier preserves its shape and errors on unknown", {
  ti <- get_backend_tier("laplace")
  expect_equal(ti$tier, 2L)
  expect_equal(ti$name, "Structured")
  expect_equal(ti$mode, "structured")
  expect_true(nzchar(ti$guarantee))
  expect_error(get_backend_tier("nope"), "Unknown backend")
})

test_that("reachability reflects whether an R fitter exists", {
  reachable <- c("gibbs", "imh_laplace", "mala", "laplace", "pathfinder", "agq")
  cabi_only <- c("hmc", "ess", "pg", "sghmc", "sgld", "mclmc", "smc", "vi")
  for (b in reachable) expect_true(backend_is_reachable(b), info = b)
  for (b in cabi_only) expect_false(backend_is_reachable(b), info = b)
})

test_that("selecting a C-ABI-only backend fails loudly, naming the kernel", {
  err <- expect_error(assert_backend_reachable("smc"))
  expect_match(conditionMessage(err), "no R-level fitter")
  expect_match(conditionMessage(err), "tulpa_smc_fit")

  err2 <- expect_error(tulpa_dispatch("ess", fitter_args = list()))
  expect_match(conditionMessage(err2), "tulpa_run_ess_sampler")
})

test_that("resolve_backend_fitter returns the real function for reachable backends", {
  expect_identical(resolve_backend_fitter("laplace"), tulpa_laplace)
  expect_identical(resolve_backend_fitter("mala"), mala)
  expect_identical(resolve_backend_fitter("pathfinder"), pathfinder)
})

test_that("tulpa_dispatch routes to the chosen backend and stamps the contract", {
  # mala consumes a log_posterior closure; a trivial isotropic Gaussian target.
  lp  <- function(theta) -0.5 * sum(theta^2)
  glp <- function(theta) -theta
  fit <- tulpa_dispatch(
    "mala",
    fitter_args = list(
      log_posterior = lp, grad_log_posterior = glp, init = c(0, 0),
      n_iter = 400L, warmup = 200L, epsilon = 0.3
    )
  )
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "mala")
  expect_equal(fit$inference_tier, 1L)
  expect_equal(fit$inference_mode, "exact")
  expect_true(nzchar(fit$selection_reason))
})

test_that("family support is derived from the registry", {
  binom <- list(name = "binomial", distribution = "binomial")
  gauss <- list(name = "gaussian", distribution = "gaussian")
  expect_true(backend_supports_family("gibbs", binom))
  expect_false(backend_supports_family("gibbs", gauss))
  # unrestricted backend accepts anything
  expect_true(backend_supports_family("laplace", gauss))
})
