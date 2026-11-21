test_that("device choices are validated through the shared control reader", {
  for (x in list("bad", "CUDA", c("auto", "cpu"), NA_character_, TRUE, 1)) {
    expect_error(tulpa:::tulpa_check_control(list(device = x), "device", "test"),
                 "control\\$device must be")
  }
  for (x in c("auto", "cpu")) {
    expect_silent(tulpa:::tulpa_check_control(list(device = x), "device", "test"))
  }
  for (keys in tulpa:::.CONTROL_KEYS) expect_true("device" %in% keys)
  reason <- cuda_unavailable()
  if (!is.null(reason)) {
    expected <- if (requireNamespace("tulpaGPU", quietly = TRUE)) tulpaGPU::gpu_info()$reason else "tulpaGPU"
    expect_error(tulpa:::cpp_device_validate("cuda"), expected, fixed = TRUE)
    expect_error(tulpa(y ~ 1, data.frame(y = 1:3), control = list(device = "cuda")),
                 expected, fixed = TRUE)
  }
})

test_that("selection reads registered operations and keeps CPU choices on the CPU", {
  expect_error(tulpa:::cpp_device_probe("unknown", 10, 0L, "cpu"), "unregistered operation")
  for (choice in c("auto", "cpu")) {
    expect_identical(tulpa:::cpp_device_probe("nngp_cholesky", 1e6, 987L, choice)$device, "cpu")
  }
})

test_that("CPU device operations work without CUDA", {
  x <- c(-2, 0, 1, 5)
  result <- tulpa:::cpp_device_axpy(x, x, 0.25, "cpu")
  expect_false(result$used_cuda)
  expect_identical(result$value, x * 1.25)
  expect_error(tulpa:::cpp_device_axpy(1, 1:2, 1, "cpu"), "incompatible")
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
  tulpa:::cpp_device_replay_stages(data.frame(stage=c('sparse_inverse_host_factor','sparse_cholesky'),
    cpu_calls=c(3,2),cuda_calls=0,work=c(0,2.5e6)))
  measured <- tulpa:::cpp_device_snapshot()$stages
  tulpa:::cpp_device_end(shared)
  before <- tulpa:::cpp_device_snapshot()$stages
  expect_false('sparse_inverse_host_factor' %in% before$stage)
  tulpa:::cpp_device_replay_stages(measured)
  after <- tulpa:::cpp_device_snapshot()$stages
  expect_identical(after$cpu_calls[after$stage=='sparse_inverse_host_factor'],3)
  expect_identical(after$work[after$stage=='sparse_cholesky'],2.5e6)
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
  cuda <- is.null(cuda_unavailable(c("cusolver", "nvrtc", "cublas")))
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
