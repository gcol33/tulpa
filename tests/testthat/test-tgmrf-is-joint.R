.tgexact_summary <- function(fit, parameter) {
  z <- fit$draws[, parameter]
  statistic <- function(x) c(mean = mean(x), sd = sd(x),
    q10 = unname(quantile(x, .1)), q50 = median(x), q90 = unname(quantile(x, .9)))
  time <- fit$source_index %||% seq_along(z)
  groups <- cut(time, breaks = 30L, labels = FALSE)
  batches <- vapply(split(z, groups), statistic, numeric(5))
  list(value = statistic(z), mcse = apply(batches, 1L, sd) / sqrt(ncol(batches)))
}

test_that("pseudo-marginal and PSIS corrections agree with a true joint reference", {
  skip_if_not_slow()
  cpp <- testthat::test_path("fixtures", "tgmrf_exact_iid.cpp")
  blk <- tgmrf_cpp(cpp, "tgmrf_exact_iid", init = c(log_sigma = .5),
                    cache_dir = file.path(tempdir(), "tgmrf-exact-cache"))
  obs <- rep(seq_len(8), each = 2)
  blk$obs_idx <- obs
  fixtures <- list(
    poisson = c(0,0,0,0,0,0,0,1,1,2,3,5,0,0,0,1),
    binomial = c(0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1))
  for (family in names(fixtures)) {
    y <- fixtures[[family]]
    stopifnot(length(y) == 16L, length(unique(obs)) == 8L)
    args <- list(y = y, n_trials = rep(1L, length(y)), X = matrix(1, length(y), 1),
                 block = blk, family = family)
    reference <- do.call(tulpa_tgmrf, c(args, list(mode = "nuts_joint",
      control = list(n_iter = 30000L, warmup = 5000L, max_depth = 10L,
                     target_accept = .99, seed = 11741L))))
    expect_equal(reference$n_divergent, 0L)
    for (mode in c("imh", "nuts")) {
      set.seed(11740L + match(mode, c("imh", "nuts")))
      fit <- do.call(tulpa_tgmrf, c(args, list(mode = mode,
        control = list(n_iter = 8000L, warmup = 1500L))))
      if (mode == "imh") {
        expect_identical(fit$inference_mode, "exact")
        expect_identical(fit$inference_tier, 1L)
        expect_identical(fit$correction, "pseudo_marginal")
      } else {
        expect_true(fit$correction_applied)
        expect_lt(fit$pareto_k, fit$correction_k_threshold)
        expect_identical(fit$inference_tier, 2L)
      }
      expect_equal(ncol(fit$draws_z), 8L)
      expect_gt(fit$mean_accept, .1)
      for (parameter in c("beta_1", "log_sigma")) {
        a <- .tgexact_summary(fit, parameter)
        b <- .tgexact_summary(reference, parameter)
        error <- abs(a$value - b$value) / sqrt(a$mcse^2 + b$mcse^2)
        expect_true(max(error) < 5,
          info = paste(family, mode, parameter, paste(names(error), round(error, 2), collapse = "; ")))
      }
    }
  }
})

test_that("joint NUTS uses the engine beta prior and Gaussian variance convention", {
  skip_if_not_slow()
  blk <- tgmrf_cpp(testthat::test_path("fixtures", "tgmrf_exact_iid.cpp"),
    "tgmrf_exact_iid", init = c(log_sigma = .5),
    cache_dir = file.path(tempdir(), "tgmrf-exact-cache"))
  blk$obs_idx <- rep(seq_len(8), each = 2)
  y <- c(-.2,.1,.3,.2,-.4,-.3,.6,.4,-.2,-.3,.1,.2,.3,.4,-.1,.2)
  args <- list(y = y, n_trials = rep(1L, length(y)), X = matrix(1, length(y), 1),
               block = blk, family = "gaussian", phi = .4)
  joint <- do.call(tulpa_tgmrf, c(args, list(mode = "nuts_joint", control = list(
    n_iter = 20000L, warmup = 3000L, seed = 11741L, max_depth = 6L,
    debug_gradient_check = TRUE, gradient_check_tol = 1e-3))))
  set.seed(11743)
  pm <- do.call(tulpa_tgmrf, c(args, list(mode = "imh", control = list(
    n_iter = 8000L, warmup = 1500L))))
  for (parameter in c("beta_1", "log_sigma")) {
    a <- .tgexact_summary(pm, parameter); b <- .tgexact_summary(joint, parameter)
    expect_lt(max(abs(a$value - b$value) / sqrt(a$mcse^2 + b$mcse^2)), 5)
  }
})
