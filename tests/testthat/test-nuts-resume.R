skip_on_cran()

.nuts_resume_data <- function() {
  set.seed(9)
  X <- cbind(1, seq(-1, 1, length.out = 60L))
  list(y_r = as.numeric(X %*% c(0.4, 1) + rnorm(60L, sd = 0.3)), X_r = X)
}

test_that("a zero-warmup chain uses the reference diagonal kernel", {
  skip_on_cran()
  load_cpp_fixture("nuts_resume")
  old <- tulpa_integrator("leapfrog")
  on.exit(tulpa_integrator(old), add = TRUE)
  for (step in c(0, 0.03)) for (seed in 109:113) {
    result <- cpp_test_nuts_resume_kernel(seed, step)
    expect_equal(result$draw, result$expected, tolerance = 1e-10)
    expect_equal(result$energy, result$expected_energy, tolerance = 1e-10)
    expect_equal(result$accept, result$expected_accept, tolerance = 1e-10)
    expect_identical(result$epsilon, result$expected_epsilon)
    expect_identical(result$inv_metric, result$expected_metric)
    expect_identical(result$divergences, 0L)
  }
})

test_that("R and C ABI resumes use supplied single-chain step sizes", {
  skip_on_cran()
  args <- c(.nuts_resume_data(), list(n_iter = 80L, n_warmup = 0L,
    max_treedepth = 7L, adapt_delta = 0.85, seed = 109L,
    init = c(0.4, 1, log(0.3)), inv_metric_init = c(0.002, 0.005, 0.007)))
  for (step in c(0.03, 0.5)) {
    args$epsilon_init <- step
    r <- do.call(tulpa:::cpp_tulpa_fit_generic, c(args, list(verbose = FALSE)))
    abi <- do.call(tulpa:::cpp_test_c_abi_resume_roundtrip, args)
    expect_identical(r$epsilon, step)
    expect_identical(abi$epsilon, step)
    expect_identical(r$draws, abi$draws)
    expect_equal(sum(r$divergent), 0)
    if (step == 0.03) expect_gt(mean(r$accept_prob), 0.99)
  }
})

test_that("R and C ABI resumes carry a step size for every chain", {
  skip_on_cran()
  args <- c(.nuts_resume_data(), list(n_chains = 2L, n_iter = 80L, n_warmup = 0L,
    max_treedepth = 7L, adapt_delta = 0.85, seed = 109L,
    init = matrix(rep(c(0.4, 1, log(0.3)), 2), 2, byrow = TRUE),
    inv_metric_init = matrix(rep(c(0.002, 0.005, 0.007), 2), 2, byrow = TRUE),
    epsilon_init = c(0.03, 0.5)))
  r <- do.call(tulpa:::cpp_tulpa_fit_generic_chains, c(args, list(verbose = FALSE)))
  abi <- do.call(tulpa:::cpp_test_c_abi_chains_roundtrip, args)
  expect_identical(r$epsilon, args$epsilon_init)
  expect_identical(abi$epsilon, args$epsilon_init)
  expect_identical(r$draws, abi$draws)
  expect_identical(r$final_position, abi$final_position)
  expect_equal(sum(r$divergent), 0)
})

test_that("step-size input validation reaches every resume door", {
  skip_on_cran()
  args <- c(.nuts_resume_data(), list(n_iter = 1L, n_warmup = 0L))
  for (entry in list(tulpa:::cpp_tulpa_fit_generic, tulpa:::cpp_test_c_abi_resume_roundtrip,
                    tulpa:::cpp_tulpa_fit_generic_chains, tulpa:::cpp_test_c_abi_chains_roundtrip)) {
    for (bad in c(-1, Inf, NA_real_, NaN))
      expect_error(do.call(entry, c(args, list(epsilon_init = bad))), "epsilon_init")
  }
  expect_error(do.call(tulpa:::cpp_tulpa_fit_generic_chains,
    c(args, list(n_chains = 2L, epsilon_init = c(0.1, 0.2, 0.3)))), "epsilon_init")
})

test_that("short warmups keep a fixed metric and a stable averaged step", {
  skip_on_cran()
  for (w in c(0L, 1L, 5L, 20L, 50L)) {
    args <- c(.nuts_resume_data(), list(n_iter = 300L + w, n_warmup = w,
      max_treedepth = 7L, adapt_delta = 0.85, seed = 109L, verbose = FALSE,
      init = c(0.4, 1, log(0.3))))
    f <- do.call(tulpa:::cpp_tulpa_fit_generic, args)
    expect_gt(mean(f$accept_prob), 0.7, label = paste("warmup", w))
    expect_equal(sum(f$divergent), 0, info = paste("warmup", w))
    expect_equal(f$inv_metric, c(1 / 60, 1 / sum(args$X_r[, 2]^2), 1))
  }
})

test_that("a checkpoint fingerprint includes each chain's initial step", {
  skip_on_cran()
  checkpoint <- tempfile()
  on.exit(unlink(checkpoint), add = TRUE)
  args <- c(.nuts_resume_data(), list(n_chains = 2L, n_iter = 3L, n_warmup = 0L,
    epsilon_init = c(0.01, 0.02), checkpoint_path = checkpoint, verbose = FALSE))
  first <- do.call(tulpa:::cpp_tulpa_fit_generic_chains, args)
  expect_identical(do.call(tulpa:::cpp_tulpa_fit_generic_chains, args)$draws, first$draws)
  args$epsilon_init <- c(0.01, 0.03)
  expect_error(do.call(tulpa:::cpp_tulpa_fit_generic_chains, args), "fingerprint|mismatch")
})

test_that("the GLMM front door carries per-chain resume state", {
  skip_on_cran()
  d <- .nuts_resume_data()
  ref <- tulpa:::tulpa_sample_glmm(d$y_r, NULL, d$X_r, "gaussian", "hmc", phi = 0.09,
    control = list(n_iter = 400L, warmup = 200L, n_chains = 2L, seed = 9L))
  f <- tulpa:::tulpa_sample_glmm(d$y_r, NULL, d$X_r, "gaussian", "hmc", phi = 0.09,
    warm_start = list(init = ref$final_position, inv_metric_diag = ref$inv_metric,
                      epsilon_init = ref$epsilon),
    control = list(n_iter = 200L, warmup = 0L, n_chains = 2L, seed = 109L))
  expect_identical(f$epsilon, ref$epsilon)
  expect_identical(f$inv_metric, ref$inv_metric)
  expect_equal(sum(f$divergent), 0)
  expect_equal(unname(colMeans(f$draws)), unname(colMeans(ref$draws)), tolerance = 0.03)
})
