test_that("the observation design rejects an inconsistent slope layout", {
  f <- fused_fixture("poisson", n = 40L)
  re <- list(idx = list(rep(1:5, each = 8)), ngroups = 5L, ncoefs = 2L,
             correlated = TRUE, Z = list(matrix(f$X[, 2L], ncol = 1L)))
  expect_error(tulpa:::cpp_device_model_gradient(f$y, f$trials, f$X, f$family,
    f$phi, numeric(), "cpu", re = re), "has 1 columns.*declares 2 coefficients")
})

test_that("resident family kernels match CPU values, scores, curvature and scatters", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  families <- c("gaussian", "binomial", "poisson", "neg_binomial_2", "gamma",
                "beta", "beta_binomial", "truncated_poisson",
                "truncated_neg_binomial_2", "neg_binomial_1")
  for (family in families) for (phi in c(.3, 2.3, 1e4, 1e12, 1e17)) {
    f <- fused_fixture(family, phi = phi)
    f$y[2L] <- NA_real_
    f$weights[3L] <- 0
    cpu <- fused_run(f, "cpu")
    cuda <- fused_run(f, "cuda")
    expect_true(cuda$used_cuda, info = family)
    expect_false(cpu$used_cuda)
    for (q in c("value", "gradient", "eta", "score", "curvature"))
      expect_device_equal(cuda[[q]], cpu[[q]], 2e-8,
                          sprintf("%s phi=%g %s", family, phi, q))
  }
})

test_that("ZI and hurdle layouts use the same mixture on CPU and CUDA", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  for (family in c("poisson", "binomial", "neg_binomial_2", "neg_binomial_1",
                    "beta_binomial", "truncated_poisson", "truncated_neg_binomial_2")) {
    f <- fused_fixture(family, zi = TRUE)
    cpu <- fused_run(f, "cpu"); cuda <- fused_run(f, "cuda")
    for (q in c("value", "gradient", "score", "curvature"))
      expect_device_equal(cuda[[q]], cpu[[q]], 2e-8, paste(family, q))
  }
})

test_that("a single binomial trial agrees with the Bernoulli likelihood", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- fused_fixture("binomial")
  f$trials[] <- 1L
  f$y[] <- rep(c(0, 1), length.out = length(f$y))
  cpu <- fused_run(f, "cpu"); cuda <- fused_run(f, "cuda")
  for (q in c("value", "gradient", "score", "curvature"))
    expect_device_equal(cuda[[q]], cpu[[q]], 1e-10, paste("Bernoulli", q))
})

test_that("large counts and large precision retain value and gradient equivalence", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  for (family in c("poisson", "neg_binomial_2", "beta_binomial", "gamma", "beta")) {
    f <- fused_fixture(family, phi = 1e14)
    f$coefficients[,] <- 0
    f$offset[] <- if (family %in% c("beta", "beta_binomial")) .2 else log(1e8)
    if (family == "beta_binomial") { f$trials[] <- 200000000L; f$y[] <- 100000000 }
    else if (family == "beta") f$y[] <- plogis(.2) + 1e-9
    else f$y[] <- 1e8 + 100
    cpu <- fused_run(f, "cpu"); cuda <- fused_run(f, "cuda")
    for (q in c("value", "gradient", "score"))
      expect_device_equal(cuda[[q]], cpu[[q]], 2e-8, paste("large", family, q))
  }
})

test_that("family gradients agree with independent finite differences", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  for (family in c("gaussian", "binomial", "poisson", "neg_binomial_2", "gamma", "beta", "beta_binomial",
                   "neg_binomial_1", "truncated_poisson", "truncated_neg_binomial_2")) {
    f <- fused_fixture(family, n = 13L)
    cuda <- fused_run(f, "cuda")
    numerical <- vapply(seq_len(nrow(f$coefficients)), function(j) {
      plus <- minus <- f
      h <- 1e-5
      plus$coefficients[j, 1L] <- plus$coefficients[j, 1L] + h
      minus$coefficients[j, 1L] <- minus$coefficients[j, 1L] - h
      (fused_run(plus, "cpu")$value - fused_run(minus, "cpu")$value) / (2 * h)
    }, 0)
    expect_device_equal(cuda$gradient, numerical, 2e-7, paste(family, "numerical"))
  }
})

test_that("gamma and beta retain near-mean precision in the concentrated regime", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  for (family in c("gamma", "beta")) for (phi in c(1e8, 1e10, 1e12, 1e14)) {
    f <- fused_fixture(family, phi = phi, n = 13L)
    f$coefficients[,] <- 0
    f$weights[] <- 1
    f$offset[] <- if (family == "gamma") 0 else qlogis(.3)
    f$y[] <- if (family == "gamma") 1 + .7 / sqrt(phi) else .3 + .7 * sqrt(.21 / (phi + 1))
    cpu <- fused_run(f, "cpu"); cuda <- fused_run(f, "cuda")
    for (q in c("value", "gradient", "score", "curvature"))
      expect_device_equal(cuda[[q]], cpu[[q]], 2e-8, paste("concentrated", family, phi, q))
  }
})

test_that("device families retain the CPU link guards at tail predictors", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  for (family in c("binomial", "poisson", "neg_binomial_2", "neg_binomial_1", "gamma", "beta",
                   "beta_binomial", "truncated_poisson", "truncated_neg_binomial_2")) {
    f <- fused_fixture(family, n = 13L)
    f$coefficients[,] <- 0
    f$offset[] <- rep(c(-100, -40, 40, 100), length.out = length(f$y))
    cpu <- fused_run(f, "cpu"); cuda <- fused_run(f, "cuda")
    for (q in c("value", "gradient", "score", "curvature"))
      expect_device_equal(cuda[[q]], cpu[[q]], 2e-8, paste("tail", family, q))
  }
})

test_that("one stream per chain and a batched chain kernel agree", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- fused_fixture("neg_binomial_2", n = 1031L, chains = 4L, zi = TRUE)
  batched <- fused_run(f, "cuda", batched = TRUE, repetitions = 3L)
  streams <- fused_run(f, "cuda", batched = FALSE, repetitions = 3L)
  cpu <- fused_run(f, "cpu")
  for (q in c("value", "gradient", "eta", "score", "curvature")) {
    expect_device_equal(streams[[q]], batched[[q]], 1e-13, paste("streams", q))
    expect_device_equal(batched[[q]], cpu[[q]], 1e-9, paste("chains", q))
  }
})

test_that("device gradients pass the production verification gate", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  for (family in c("gaussian", "binomial", "poisson", "neg_binomial_2", "gamma", "beta", "beta_binomial",
                   "neg_binomial_1", "truncated_poisson", "truncated_neg_binomial_2")) {
    f <- fused_fixture(family)
    run <- function(choice, device_slot = TRUE) tulpa:::cpp_device_model_gradient(
      f$y, f$trials, f$X, family, f$phi, numeric(), choice,
      verify = TRUE, device_slot = device_slot)
    cpu <- run("cpu"); cuda <- run("cuda")
    expect_true(cuda$used_cuda); expect_true(cuda$verified)
    expect_device_equal(cuda$value, cpu$value, 1e-9, paste(family, "posterior"))
    expect_device_equal(cuda$gradient, cpu$gradient, 1e-8, paste(family, "posterior gradient"))
    consumer <- run("cuda", FALSE)
    expect_false(consumer$used_cuda)
    expect_equal(consumer$gradient, cpu$gradient)
  }
})

test_that("spec Laplace uses fused device predictors and weights", {
  skip_on_cran()
  skip_if_no_cuda(c("nvrtc", "cublas"))
  set.seed(1183)
  d <- data.frame(x = rnorm(200))
  d$y <- rpois(nrow(d), exp(.3 + .2 * d$x))
  run <- function(choice) tulpa(y ~ x, d, family = "poisson", mode = "laplace",
                                control = list(device = choice))
  cpu <- run("cpu"); cuda <- run("cuda")
  expect_equal(coef(cuda), coef(cpu), tolerance = 1e-7)
  expect_true(any(cuda$device$stages$stage == "likelihood_weights" & cuda$device$stages$cuda_calls > 0))
})

test_that("parallel NUTS chains prepare device payloads without calling R", {
  skip_on_cran()
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- fused_fixture("binomial", n = 201L)
  fit <- tulpa:::tulpa_sample_glmm(f$y, f$trials, f$X, f$family, "hmc",
    control = list(device = "cuda", n_iter = 24L, warmup = 12L, n_chains = 2L,
                   max_treedepth = 2L, seed = 1183L))
  expect_equal(dim(fit$draws), c(24L, 3L))
  expect_equal(sort(unique(fit$chain_id)), 1:2)
  expect_true(all(is.finite(fit$draws)))
  stages <- fit$device$stages
  expect_gt(sum(stages$cuda_calls[stages$stage == "likelihood"]), 10)
})

test_that("the production verification gate rejects an incorrect device score", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- fused_fixture("poisson")
  expect_warning(bad <- tulpa:::cpp_device_model_gradient(f$y, f$trials, f$X,
    f$family, f$phi, numeric(), "cuda", corrupt = TRUE), "Gradient mismatch detected")
  expect_true(bad$verified)
  expect_false(bad$used_cuda)
})

test_that("device projection preserves RE slopes, BYM2 centering and temporal fields", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- fused_fixture("poisson", n = 40L)
  re <- list(idx = list(rep(1:5, each = 8)), ngroups = 5L, ncoefs = 2L,
             correlated = TRUE, Z = list(cbind(1, f$X[, 2L])))
  spatial <- list(type = "bym2", parameterization = "centered",
    spatial_idx = rep(1:4, each = 10), n_spatial_units = 4L,
    adj_row_ptr = c(0L, 1L, 3L, 5L, 6L), adj_col_idx = c(1L, 0L, 2L, 1L, 3L, 2L),
    n_neighbors = c(1L, 2L, 2L, 1L))
  temporal <- list(type = "rw1", parameterization = "centered", time_idx = rep(1:10, 4),
                   n_times = 10L, n_groups = 1L)
  for (structure in list(list(re = re), list(spatial = spatial), list(temporal = temporal),
                          list(re = re, spatial = spatial, temporal = temporal))) {
    run <- function(choice) do.call(tulpa:::cpp_device_model_gradient,
      c(list(y = f$y, trials = f$trials, X = f$X, family = f$family, phi = f$phi,
             params = numeric(), choice = choice), structure))
    cpu <- run("cpu"); cuda <- run("cuda")
    expect_true(cuda$used_cuda); expect_true(cuda$verified)
    expect_device_equal(cuda$value, cpu$value, 1e-9, "field posterior")
    expect_device_equal(cuda$gradient, cpu$gradient, 1e-8, "field chain rule")
  }
})

test_that("a consumer device slot differentiates direct extra parameters", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- fused_fixture("poisson")
  run <- function(choice) tulpa:::cpp_device_model_gradient(f$y, f$trials, f$X,
    f$family, f$phi, numeric(), choice, consumer_extra = TRUE)
  cpu <- run("cpu"); cuda <- run("cuda")
  expect_true(cuda$used_cuda); expect_true(cuda$verified)
  expect_device_equal(cuda$gradient, cpu$gradient, 1e-9, "consumer direct parameter")
  expect_device_equal(cuda$value, cpu$value, 1e-9, "consumer likelihood")
})

test_that("device scores propagate through GP and basis-field reconstruction", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  f <- smc_nngp_fixture("F1")
  for (parameterization in 0:1) {
    spatial <- f$sp
    spatial$gp_parameterization <- parameterization
    run <- function(choice) tulpa:::cpp_device_model_gradient(f$y, rep(1L, length(f$y)),
      f$X, "gaussian", sqrt(f$noise), numeric(), choice, spatial = spatial)
    cpu <- run("cpu"); cuda <- run("cuda")
    expect_true(cuda$used_cuda); expect_true(cuda$verified)
    expect_device_equal(cuda$value, cpu$value, 1e-9, "GP reconstruction value")
    expect_device_equal(cuda$gradient, cpu$gradient, 1e-8, "GP reconstruction gradient")
  }
  f <- fused_fixture("poisson", n = 40L)
  coords <- cbind(seq(0, 1, length.out = 40L), sin(seq_len(40L) / 5))
  for (kernel in 0:2) {
    spatial <- list(type = "hsgp", coords = coords, m = 3L, c = 1.5,
      kernel = kernel, lengthscale_prior_U = .3, lengthscale_prior_alpha = .1)
    run <- function(choice) tulpa:::cpp_device_model_gradient(f$y, f$trials,
      f$X, f$family, f$phi, numeric(), choice, spatial = spatial)
    cpu <- run("cpu"); cuda <- run("cuda")
    expect_true(cuda$used_cuda); expect_true(cuda$verified)
    expect_device_equal(cuda$value, cpu$value, 1e-9, "basis-field reconstruction value")
    expect_device_equal(cuda$gradient, cpu$gradient, 1e-8, "basis-field reconstruction gradient")
  }
})

test_that("automatic family routing agrees on both sides of every measured crossover", {
  skip_if_no_cuda(c("nvrtc", "cublas"))
  timings <- read.csv(system.file("device", "benchmarks.csv", package = "tulpa"))
  timings <- timings[timings$operation %in% c("likelihood", "likelihood_weights"), ]
  gates <- tulpa:::.device_thresholds(timings)
  if (!nrow(gates)) skip("likelihood crossovers have not been calibrated")
  families <- c("gaussian", "binomial", "poisson", "neg_binomial_2", "gamma", "beta",
                "beta_binomial", "truncated_poisson", "truncated_neg_binomial_2", "neg_binomial_1")
  for (i in seq_len(nrow(gates))) {
    gate <- gates[i, ]
    if (!is.finite(gate$threshold)) {
      expect_identical(tulpa:::cpp_device_probe(gate$operation, 1e7, gate$variant, "auto")$device, "cpu")
      next
    }
    family <- families[gate$variant %/% 2L + 1L]
    weights <- gate$operation == "likelihood_weights"
    for (n in as.integer(gate$threshold + c(-1L, 1L))) {
      f <- fused_fixture(family, n = n, zi = gate$variant %% 2L == 1L)
      cpu <- fused_run(f, "cpu", return_weights = weights)
      cuda <- fused_run(f, "cuda", return_weights = weights)
      auto <- fused_run(f, "auto", return_weights = weights)
      expect_identical(auto$used_cuda, n >= gate$threshold)
      for (q in c("value", "gradient")) {
        expect_device_equal(cuda[[q]], cpu[[q]], 2e-8, paste("threshold", family, n, q))
        expect_device_equal(auto[[q]], cpu[[q]], 2e-8, paste("auto", family, n, q))
      }
      rm(f, cpu, cuda, auto)
      gc(FALSE)
    }
  }
})
