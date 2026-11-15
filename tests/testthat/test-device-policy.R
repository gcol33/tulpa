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

test_that("the measured plan selects CPU, streams and batches independently", {
  timings <- data.frame(operation='laplace_scatter',variant=4L,size=c(50,100,200),
    cpu_seconds=1,cuda_seconds=c(2,0.8,0.7),stream_seconds=c(0.9,0.7,0.6))
  plan <- tulpa:::.device_thresholds(timings)
  expect_identical(plan$threshold,50)
  expect_identical(plan$batch_threshold,Inf)
  timings$stream_seconds <- c(1.5,0.9,0.8)
  plan <- tulpa:::.device_thresholds(timings)
  expect_identical(plan$threshold,100)
  expect_identical(plan$batch_threshold,100)
  expect_error(tulpa:::.device_thresholds(transform(timings,stream_seconds=-1)),
               'positive or infinite')
})

test_that("each operation uses its latest complete calibration without mixing binaries", {
  timings <- data.frame(operation=c('laplace_dense','laplace_dense','axpy'),
    variant=0L,size=100,cpu_seconds=c(2,0.5,2),cuda_seconds=1,
    calibration=c(0L,1L,0L))
  plan <- tulpa:::.device_thresholds(timings)
  expect_identical(plan$threshold[plan$operation=='laplace_dense'],Inf)
  expect_identical(plan$threshold[plan$operation=='axpy'],100)
  expect_error(tulpa:::.device_thresholds(transform(timings,calibration=0.5)),
    'non-negative integers')
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

test_that("an isolated shared solve replays its measured calls into its caller", {
  outer <- tulpa:::cpp_device_begin('cpu',inherit=FALSE)
  on.exit(tulpa:::cpp_device_end(outer))
  shared <- tulpa:::cpp_device_begin('cpu',inherit=FALSE)
  tulpa:::cpp_device_replay_stages(data.frame(stage='sparse_inverse_host_factor',
    cpu_calls=3,cuda_calls=0))
  measured <- tulpa:::cpp_device_snapshot()$stages
  tulpa:::cpp_device_end(shared)
  before <- tulpa:::cpp_device_snapshot()$stages
  expect_false('sparse_inverse_host_factor' %in% before$stage)
  tulpa:::cpp_device_replay_stages(measured)
  after <- tulpa:::cpp_device_snapshot()$stages
  expect_identical(after$cpu_calls[after$stage=='sparse_inverse_host_factor'],3)
  malformed <- data.frame(stage=c('laplace_dense','sparse_selected_inverse'),
    cpu_calls=c(1,-1),cuda_calls=0)
  expect_error(tulpa:::cpp_device_replay_stages(malformed),'non-negative integers')
  expect_identical(tulpa:::cpp_device_snapshot()$stages,after)
})

test_that("replaying device counters preserves an absent random seed", {
  had_seed <- exists('.Random.seed',envir=globalenv(),inherits=FALSE)
  if(had_seed) seed <- get('.Random.seed',envir=globalenv())
  session <- tulpa:::cpp_device_begin('cpu',inherit=FALSE)
  on.exit(tulpa:::cpp_device_end(session))
  on.exit({
    if(had_seed) assign('.Random.seed',seed,envir=globalenv())
    else if(exists('.Random.seed',envir=globalenv(),inherits=FALSE))
      rm('.Random.seed',envir=globalenv())
  },add=TRUE)
  if(exists('.Random.seed',envir=globalenv(),inherits=FALSE))rm('.Random.seed',envir=globalenv())
  tulpa:::cpp_device_replay_stages(data.frame(stage='laplace_dense',cpu_calls=1,cuda_calls=0))
  expect_false(exists('.Random.seed',envir=globalenv(),inherits=FALSE))
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
    !any(nzchar(info$library_reasons[c("cusolver", "nvrtc", "cublas")]))
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
