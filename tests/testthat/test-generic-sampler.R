test_that("generic LikelihoodSpec path runs through production NUTS", {
  skip_on_cran()

  set.seed(11)
  n <- 12L
  x <- seq(-1, 1, length.out = n)
  X <- cbind(1, x)
  y <- 0.4 + 0.8 * x + rnorm(n, sd = 0.2)

  fit <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y,
    X_r = X,
    n_iter = 24L,
    n_warmup = 12L,
    max_treedepth = 3L,
    adapt_delta = 0.8,
    seed = 11L,
    verbose = FALSE
  )

  expect_equal(fit$sampler, "nuts")
  expect_equal(fit$n_params, 3L)
  expect_equal(ncol(fit$draws), 3L)
  expect_equal(nrow(fit$draws), 12L)
  expect_true(all(is.finite(fit$draws)))
  expect_true(all(is.finite(fit$log_prob)))
})

test_that("NUTS returns adapted metric + final position for resume (tulpa#29)", {
  skip_on_cran()

  set.seed(7)
  n <- 40L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.5, 1.2) + rnorm(n, sd = 0.3))

  fit <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X,
    n_iter = 400L, n_warmup = 200L,
    max_treedepth = 6L, adapt_delta = 0.8,
    seed = 7L, verbose = FALSE
  )

  np <- fit$n_params  # beta[1], beta[2], log_sigma

  # Shape + sanity
  expect_length(fit$inv_metric, np)
  expect_length(fit$final_position, np)
  expect_true(all(is.finite(fit$inv_metric)))
  expect_true(all(fit$inv_metric > 0))
  expect_true(all(is.finite(fit$final_position)))

  # final_position is the last raw draw (no NC-GP transform for this model)
  expect_equal(fit$final_position,
               as.numeric(fit$draws[nrow(fit$draws), ]),
               tolerance = 1e-10)
})

test_that("warm-start metric is honored and echoed back at n_warmup=0 (tulpa#29)", {
  skip_on_cran()

  set.seed(8)
  n <- 30L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.3, 0.9) + rnorm(n, sd = 0.4))

  metric_in <- c(0.05, 0.2, 0.1)        # all inside the [1e-3, 1e3] clamp window
  pos_in    <- c(0.3, 0.9, log(0.4))

  fit <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X,
    n_iter = 50L, n_warmup = 0L,
    max_treedepth = 6L, adapt_delta = 0.8,
    seed = 8L, verbose = FALSE,
    init = pos_in,
    inv_metric_init = metric_in
  )

  # n_warmup = 0 -> no mass adaptation -> the returned diagonal must equal the
  # supplied one bit-for-bit, proving the input is honored and the output
  # reports the metric actually used.
  expect_equal(fit$inv_metric, metric_in, tolerance = 1e-12)
  expect_length(fit$final_position, 3L)
  expect_true(all(is.finite(fit$final_position)))
})

test_that("warm-started continuation samples the same posterior (tulpa#29)", {
  skip_on_cran()

  set.seed(9)
  n <- 60L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.4, 1.0) + rnorm(n, sd = 0.3))

  ref <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X,
    n_iter = 1200L, n_warmup = 400L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 9L, verbose = FALSE
  )

  # Continue from the reported state + adapted geometry, zero warmup.
  cont <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X,
    n_iter = 1200L, n_warmup = 0L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 109L, verbose = FALSE,
    init = ref$final_position,
    inv_metric_init = ref$inv_metric
  )

  expect_true(all(is.finite(cont$draws)))
  # Same posterior: warm-started continuation means match the reference within
  # Monte Carlo error (beta posterior sd ~0.05 here).
  expect_equal(cont$means[["beta[1]"]], ref$means[["beta[1]"]], tolerance = 0.1)
  expect_equal(cont$means[["beta[2]"]], ref$means[["beta[2]"]], tolerance = 0.1)
})

# ---------------------------------------------------------------------------
# Multi-chain across-chain runner (gcol33/tulpa#30)
# ---------------------------------------------------------------------------

test_that("multi-chain runner returns chain-major draws + per-chain state (tulpa#30)", {
  skip_on_cran()

  set.seed(21)
  n <- 50L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.5, 1.1) + rnorm(n, sd = 0.3))

  nch <- 4L
  fit <- tulpa:::cpp_tulpa_fit_generic_chains(
    y_r = y, X_r = X, n_chains = nch,
    n_iter = 300L, n_warmup = 150L,
    max_treedepth = 6L, adapt_delta = 0.8,
    seed = 21L, verbose = FALSE
  )

  ns <- fit$n_samples  # per chain
  np <- fit$n_params

  expect_equal(fit$n_chains, nch)
  expect_equal(nrow(fit$draws), ns * nch)
  expect_equal(ncol(fit$draws), np)
  expect_equal(fit$chain_id, rep(seq_len(nch), each = ns))
  expect_true(all(is.finite(fit$draws)))

  # Per-chain summaries are matrices [n_chains x n_params], all finite/positive.
  expect_equal(dim(fit$inv_metric), c(nch, np))
  expect_equal(dim(fit$final_position), c(nch, np))
  expect_length(fit$epsilon, nch)
  expect_true(all(is.finite(fit$inv_metric)) && all(fit$inv_metric > 0))
  expect_true(all(is.finite(fit$final_position)))
  expect_true(all(fit$epsilon > 0))
})

test_that("each chain's final_position is its own last draw (tulpa#30)", {
  skip_on_cran()

  set.seed(22)
  n <- 40L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.4, 0.9) + rnorm(n, sd = 0.3))

  nch <- 3L
  fit <- tulpa:::cpp_tulpa_fit_generic_chains(
    y_r = y, X_r = X, n_chains = nch,
    n_iter = 200L, n_warmup = 100L,
    max_treedepth = 6L, adapt_delta = 0.8,
    seed = 22L, verbose = FALSE
  )

  ns <- fit$n_samples
  for (c in seq_len(nch)) {
    last_row <- fit$draws[c * ns, ]          # last iter of chain c's block
    expect_equal(as.numeric(fit$final_position[c, ]),
                 as.numeric(last_row), tolerance = 1e-10,
                 info = paste("chain", c))
  }
})

test_that("a shared init still yields independent chains (tulpa#30)", {
  skip_on_cran()

  set.seed(23)
  n <- 40L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.3, 1.0) + rnorm(n, sd = 0.3))

  nch <- 3L
  fit <- tulpa:::cpp_tulpa_fit_generic_chains(
    y_r = y, X_r = X, n_chains = nch,
    n_iter = 150L, n_warmup = 75L,
    max_treedepth = 6L, adapt_delta = 0.8,
    seed = 23L, verbose = FALSE
  )

  ns <- fit$n_samples
  block <- function(c) fit$draws[((c - 1L) * ns + 1L):(c * ns), , drop = FALSE]
  # Same origin init for every chain, but seed+chain_id diversification must
  # make the realised trajectories differ.
  expect_false(isTRUE(all.equal(block(1L), block(2L))))
  expect_false(isTRUE(all.equal(block(2L), block(3L))))
})

test_that("multi-chain fit is reproducible for a fixed seed (tulpa#30)", {
  skip_on_cran()

  set.seed(24)
  n <- 35L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.6, 0.8) + rnorm(n, sd = 0.3))

  args <- list(
    y_r = y, X_r = X, n_chains = 3L,
    n_iter = 150L, n_warmup = 75L,
    max_treedepth = 6L, adapt_delta = 0.8,
    seed = 24L, verbose = FALSE
  )
  a <- do.call(tulpa:::cpp_tulpa_fit_generic_chains, args)
  b <- do.call(tulpa:::cpp_tulpa_fit_generic_chains, args)

  # Each chain is independent (own RNG, own tape), so thread scheduling cannot
  # perturb the result: same seed -> identical draws.
  expect_equal(a$draws, b$draws, tolerance = 1e-10)
  expect_equal(a$final_position, b$final_position, tolerance = 1e-10)
})

test_that("mcmc_diagnostics consumes a native multi-chain fit (tulpa#26 via #30)", {
  skip_on_cran()

  set.seed(25)
  n <- 80L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.5, 1.0) + rnorm(n, sd = 0.3))

  nch <- 4L
  fit <- tulpa:::cpp_tulpa_fit_generic_chains(
    y_r = y, X_r = X, n_chains = nch,
    n_iter = 700L, n_warmup = 350L,
    max_treedepth = 8L, adapt_delta = 0.9,
    seed = 25L, verbose = FALSE
  )

  # The returned (draws, chain_id, n_chains) is exactly the layout
  # .tulpa_chain_list() expects — no R-side reshaping needed.
  diag <- tulpa::mcmc_diagnostics(
    list(draws = fit$draws, chain_id = fit$chain_id, n_chains = fit$n_chains),
    measures = c("rhat", "ess_bulk", "ess_tail")
  )

  expect_s3_class(diag, "data.frame")
  expect_setequal(diag$parameter, c("beta[1]", "beta[2]", "log_sigma"))
  expect_true(all(is.finite(diag$rhat)))
  expect_true(all(diag$rhat < 1.1))          # well-mixed Gaussian linear model
  expect_true(all(diag$ess_bulk > 0))
  expect_true(all(diag$ess_tail > 0))
})

test_that("per-chain warm-start resumes a multi-chain fit (tulpa#30 + #29)", {
  skip_on_cran()

  set.seed(26)
  n <- 60L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.4, 1.0) + rnorm(n, sd = 0.3))

  nch <- 3L
  ref <- tulpa:::cpp_tulpa_fit_generic_chains(
    y_r = y, X_r = X, n_chains = nch,
    n_iter = 800L, n_warmup = 400L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 26L, verbose = FALSE
  )

  # Continue every chain from its own final_position + adapted metric, zero
  # warmup. init / inv_metric_init are [n_chains x n_params], as ref emits.
  cont <- tulpa:::cpp_tulpa_fit_generic_chains(
    y_r = y, X_r = X, n_chains = nch,
    n_iter = 800L, n_warmup = 0L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 126L, verbose = FALSE,
    init = ref$final_position,
    inv_metric_init = ref$inv_metric
  )

  expect_true(all(is.finite(cont$draws)))

  # Pool both fits and compare posterior means within Monte Carlo error.
  ref_mean  <- colMeans(ref$draws)
  cont_mean <- colMeans(cont$draws)
  expect_equal(cont_mean[["beta[1]"]], ref_mean[["beta[1]"]], tolerance = 0.1)
  expect_equal(cont_mean[["beta[2]"]], ref_mean[["beta[2]"]], tolerance = 0.1)
})

# ---------------------------------------------------------------------------
# C-ABI round-trip for the resume outputs (gcol33/tulpa#29)
# ---------------------------------------------------------------------------
# The Rcpp wrappers above read inv_metric / final_position straight off
# HMCResultCpp; downstream packages instead reach tulpa through the registered
# C ABI (R_GetCCallable("tulpa", "tulpa_run_nuts_generic") -> NUTSResult).
# fill_nuts_result_from_cpp() copies the resume fields into the NUTSResult
# struct; these tests run the C ABI itself end-to-end so a regression to that
# copy fails here rather than only inside a downstream consumer.

test_that("C ABI tulpa_run_nuts_generic populates inv_metric_out + final_position (tulpa#29)", {
  skip_on_cran()

  set.seed(29)
  n <- 50L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.5, 1.0) + rnorm(n, sd = 0.3))

  abi <- tulpa:::cpp_test_c_abi_resume_roundtrip(
    y_r = y, X_r = X,
    n_iter = 400L, n_warmup = 200L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 29L
  )

  # Both resume fields must be length n_params and finite — the C ABI is
  # responsible for allocating + populating them in fill_nuts_result_from_cpp.
  expect_length(abi$inv_metric_out, abi$n_params)
  expect_length(abi$final_position, abi$n_params)
  expect_true(all(is.finite(abi$inv_metric_out)))
  expect_true(all(is.finite(abi$final_position)))

  # The inverse-mass diagonal should be strictly positive (a diagonal metric
  # entry of zero would be a broken sampler, not a valid resume state).
  expect_true(all(abi$inv_metric_out > 0))

  # final_position must match the last draw of the returned chain — the C ABI
  # advertises "init = final_position" as a valid continuation, so the field
  # has to be the actual last sampler position, not e.g. the posterior mean.
  last_row <- as.numeric(abi$draws[abi$n_samples, ])
  expect_equal(as.numeric(abi$final_position), last_row, tolerance = 1e-10)
})

test_that("C ABI resume continues a chain from inv_metric_out + final_position (tulpa#29)", {
  skip_on_cran()

  set.seed(30)
  n <- 60L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.4, 1.0) + rnorm(n, sd = 0.3))

  ref <- tulpa:::cpp_test_c_abi_resume_roundtrip(
    y_r = y, X_r = X,
    n_iter = 800L, n_warmup = 400L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 30L
  )

  # Hand the C-ABI resume fields back as warm-start inputs to a fresh C-ABI
  # call with n_warmup = 0 — the documented continuation contract.
  cont <- tulpa:::cpp_test_c_abi_resume_roundtrip(
    y_r = y, X_r = X,
    n_iter = 800L, n_warmup = 0L,
    max_treedepth = 7L, adapt_delta = 0.85,
    seed = 130L,
    init = ref$final_position,
    inv_metric_init = ref$inv_metric_out
  )

  expect_true(all(is.finite(cont$draws)))

  ref_mean  <- colMeans(ref$draws)
  cont_mean <- colMeans(cont$draws)
  expect_equal(cont_mean[["beta[1]"]], ref_mean[["beta[1]"]], tolerance = 0.1)
  expect_equal(cont_mean[["beta[2]"]], ref_mean[["beta[2]"]], tolerance = 0.1)
})
