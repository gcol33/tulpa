test_that("device choices are validated through the shared control reader", {
  for (x in list("bad", "CUDA", c("auto", "cpu"), NA_character_, TRUE, 1)) {
    expect_error(tulpa:::tulpa_check_control(list(device = x), "device", "test"),
                 "control\\$device must be")
  }
  for (x in c("auto", "cpu")) {
    expect_silent(tulpa:::tulpa_check_control(list(device = x), "device", "test"))
  }
  for (keys in tulpa:::.CONTROL_KEYS) expect_true("device" %in% keys)
  if (!gpu_info()$available) {
    reason <- gpu_info()$reason
    expect_error(tulpa:::cpp_device_validate("cuda"), reason, fixed = TRUE)
    expect_error(tulpa(y ~ 1, data.frame(y = 1:3), control = list(device = "cuda")),
                 reason, fixed = TRUE)
  }
})

test_that("automatic policy has a separate measured crossover per operation and order", {
  timings <- data.frame(operation = rep("nngp_cholesky", 6),
    variant = rep(c(10L, 15L), each = 3), size = rep(c(50, 100, 200), 2),
    cpu_seconds = 1, cuda_seconds = c(2, 0.8, 0.7, 0.8, 2, 1.2))
  settings <- tulpa:::.device_thresholds(timings)
  expect_identical(settings$threshold[settings$variant == 10L], 100)
  expect_identical(settings$threshold[settings$variant == 15L], Inf)
  expect_equal(nrow(tulpa:::.device_thresholds(timings[FALSE, ])), 0L)
  expect_error(tulpa:::.device_thresholds(transform(timings, cpu_seconds = 0)), "positive")
  expect_error(tulpa:::cpp_device_probe("unknown", 10, 0L, "cpu"), "unregistered operation")
  for (choice in c("auto", "cpu")) {
    expect_identical(tulpa:::cpp_device_probe("nngp_cholesky", 1e6, 987L, choice)$device, "cpu")
  }
})

test_that("the status reader turns library status into an R boundary error", {
  expect_error(tulpa:::cpp_device_test("status", numeric()),
               "CUDA::test_status failed \\(status 1\\): invalid value")
})

test_that("likelihood policy requires the compiler and BLAS libraries", {
  info <- gpu_info()
  reasons <- c(if (!isTRUE(info$available)) info$reason,
               info$library_reasons[c("nvrtc", "cublas")])
  reasons <- reasons[nzchar(reasons)]
  for (operation in c("likelihood", "likelihood_weights")) {
    expect_identical(tulpa:::cpp_device_probe(operation, 1e7, 0L, "cpu")$device, "cpu")
    if (length(reasons)) {
      expect_identical(tulpa:::cpp_device_probe(operation, 1e7, 0L, "auto")$device, "cpu")
      expect_error(tulpa:::cpp_device_probe(operation, 1e7, 0L, "cuda"))
    } else {
      expect_identical(tulpa:::cpp_device_probe(operation, 1e7, 0L, "cuda")$device, "cuda")
    }
  }
})

test_that("nested policies restore the enclosing choice and isolate fit counters", {
  outer <- tulpa:::cpp_device_begin("cpu")
  on.exit(tulpa:::cpp_device_end(outer), add = TRUE)
  inner <- tulpa:::cpp_device_begin("auto")
  expect_identical(tulpa:::cpp_device_snapshot()$requested, "cpu")
  tulpa:::cpp_device_end(inner)
  expect_identical(tulpa:::cpp_device_snapshot()$requested, "cpu")
  expect_identical(tulpa:::cpp_device_snapshot()$stages$device, "cpu")
})

test_that("diagnostic tables retain the fit's device record", {
  device <- list(requested = "cpu", stages = data.frame(stage = "inference",
    device = "cpu", cpu_calls = 1, cuda_calls = 0))
  fit <- structure(list(draws_kind = "chain", draws = matrix(sin(seq_len(800)), 400, 2),
                        device = device), class = "tulpa_fit")
  expect_output(result <- diagnostics(fit), "Device: inference=cpu")
  expect_identical(attr(result, "device"), device)
})

skip_on_cran()

test_that("every device control reaches a completed front-door fit", {
  skip_on_cran()
  d <- data.frame(y = c(-0.2, 0.1, 0.3, 0.5, 0.8), x = seq_len(5))
  info <- gpu_info()
  cuda <- isTRUE(info$available) &&
    !any(nzchar(info$library_reasons[c("nvrtc", "cublas")]))
  choices <- c("cpu", "auto", if (cuda) "cuda")
  for (choice in choices) {
    fit <- tulpa(y ~ x, d, family = "gaussian", phi = 1, mode = "laplace",
                 control = list(device = choice))
    expect_identical(fit$device$requested, choice)
    stage <- fit$device$stages
    on_device <- c("likelihood_weights", "dense_spec")
    expect_true(all(stage$device[!stage$stage %in% on_device] == "cpu"))
    if (any(stage$stage == "likelihood_weights")) {
      want <- tulpa:::cpp_device_probe("likelihood_weights", nrow(d), 0L, choice)$device
      expect_identical(stage$device[stage$stage == "likelihood_weights"], want)
    }
    if (choice == "cuda") {
      expect_true(any(stage$stage == "dense_spec" & stage$device == "cuda"))
    } else expect_true(all(stage$device[stage$stage == "dense_spec"] == "cpu"))
    expect_output(print(fit), "Device: inference=cpu")
    expect_message(expect_output(diag <- diagnostics(fit), "Device: inference=cpu"),
                    "carries no posterior draws")
    expect_null(diag)
  }
})
