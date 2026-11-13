test_that("gradient names and enum codes are canonical", {
  load_cpp_fixture("gradient_modes")
  modes <- c("auto", "N", "numerical", "A_r", "arena", "H", "handcoded")
  expect_identical(tulpa:::cpp_gradient_mode_names(), modes)
  expect_identical(vapply(modes, cpp_test_gradient_mode, 0L),
                   setNames(c(0L, 1L, 1L, 3L, 3L, 5L, 5L), modes))
  on.exit(cpp_test_gradient_mode("auto", TRUE))
  for (mode in modes) expect_identical(cpp_test_gradient_mode(mode, TRUE), 0L)
})

test_that("removed and unknown names error at the parser and C ABI", {
  load_cpp_fixture("gradient_modes")
  valid <- "auto.*N.*numerical.*A_r.*arena.*H.*handcoded"
  on.exit(cpp_test_gradient_mode("auto", TRUE))
  for (mode in c("A", "A_t", "autodiff", "autodiff_tape", "forward",
                 "autodiff_arena", "analytical", "AUTO", "a_r", "numer", "", "typo")) {
    expect_error(cpp_test_gradient_mode(mode), valid)
    expect_error(cpp_test_gradient_mode(mode, TRUE), valid)
  }
  expect_error(cpp_test_gradient_mode(NULL, TRUE), valid)
})

test_that("R front doors reject removed and malformed gradient controls", {
  valid <- "auto.*N.*numerical.*A_r.*arena.*H.*handcoded"
  d <- data.frame(y = c(-1, 0, 1))
  for (mode in list("A", "A_t", "autodiff", "autodiff_tape", "forward",
                   "autodiff_arena", "analytical", "AUTO", "numer", "",
                   NA_character_, c("N", "H"), 1, NULL)) {
    expect_error(tulpa(y ~ 1, data = d, family = "gaussian", mode = "hmc",
                       control = list(gradient_mode = mode)), valid)
  }
})

test_that("R sampler carries each canonical gradient name", {
  skip_on_cran()
  y <- c(-1.2, -.3, .2, .8, 1.1)
  fits <- lapply(tulpa:::cpp_gradient_mode_names(), function(mode) {
    tulpa:::tulpa_sample_glmm(y, NULL, matrix(1, length(y), 1), "gaussian", "hmc",
      control = list(n_iter = 30L, warmup = 15L, n_chains = 1L,
        max_treedepth = 3L, seed = 18L, gradient_mode = mode))
  })
  for (fit in fits) expect_true(all(is.finite(fit$draws)))
  expect_equal(fits[[2]]$draws, fits[[3]]$draws)
  expect_equal(fits[[4]]$draws, fits[[5]]$draws)
  expect_equal(fits[[6]]$draws, fits[[7]]$draws)
})
