# Every sampler backend starts a non-centred field -- its values the scale
# times coordinates of unit prior scale -- with the scale at the median of its
# prior, and a centred field at the coordinate 0. The multi-chain start
# disperses each coordinate by Uniform(-2, 2), so a non-centred field starts
# moved by its scale times that offset; started at sigma = 1 above its
# posterior, a chain can spend its first trajectory's energy along the scale
# and stay far below the posterior through warmup.

start_fixture <- function(n = 24L) {
  set.seed(3)
  x <- rnorm(n)
  list(y = rnorm(n), X = cbind("(Intercept)" = 1, x = x), n = n)
}

start_of <- function(f, ...) {
  init_fn <- get("cpp_test_sampler_init", envir = parent.frame())
  init_fn(f$y, integer(0), f$X, "gaussian", 1, 2.5, 2.5, ...)
}

pc_median_log_sd <- function(U, alpha) log(log(2) * U / -log(alpha))

test_that("a BYM2 total scale starts at its stated PC median", {
  load_cpp_fixture("smc_normalizer")
  f <- start_fixture()
  sp <- list(type = "bym2", n_spatial_units = 2L,
    spatial_idx = as.integer(rep_len(1:2, f$n)),
    adj_row_ptr = c(0L, 1L, 2L), adj_col_idx = c(1L, 0L),
    n_neighbors = c(1L, 1L), sigma_prior_U = 0.8, sigma_prior_alpha = 0.1)
  s <- start_of(f, spatial_spec = sp)
  expect_equal(unname(s["log_sigma_spatial"]), pc_median_log_sd(0.8, 0.1),
               tolerance = 1e-12)
})

test_that("a non-centred random effect's scale starts at its half-Cauchy median", {
  load_cpp_fixture("smc_normalizer")
  f <- start_fixture()
  g <- rep_len(1:6, f$n)
  re <- list(idx = list(as.integer(g)), ngroups = 6L, ncoefs = 1L,
             correlated = FALSE, Z = list(NULL))
  s <- start_of(f, re_spec = re)
  sc <- grep("^log_sigma_re", names(s))
  expect_length(sc, 1L)
  # sigma_re_scale = 2.5: the half-Cauchy median is the scale.
  expect_equal(unname(s[sc]), log(2.5), tolerance = 1e-12)
})

test_that("a non-centred NNGP field's scale starts at its PC median; centred stays at 0", {
  load_cpp_fixture("smc_normalizer")
  f <- start_fixture()
  set.seed(4)
  loc <- rep_len(1:8, f$n)
  xy <- data.frame(lon = runif(8), lat = runif(8))
  d <- data.frame(lon = xy$lon[loc], lat = xy$lat[loc])
  sp <- tulpa:::.gp_sampler_spec(validate_gp(
    spatial_gp(~ lon + lat, nn = 4L, parameterization = "noncentered"), d))
  s <- start_of(f, spatial_spec = sp)
  expect_equal(unname(s["log_sigma2_gp"]),
               2 * pc_median_log_sd(sp$sigma2_prior_U, sp$sigma2_prior_alpha),
               tolerance = 1e-12)
  sp$gp_parameterization <- 0L
  expect_equal(unname(start_of(f, spatial_spec = sp)["log_sigma2_gp"]), 0)
})

test_that("every non-centred multiscale arm starts at its PC median; centred arms at 0", {
  load_cpp_fixture("smc_normalizer")
  f <- start_fixture()
  tp <- list(type = "multiscale", time_index = as.integer(rep_len(1:8, f$n)),
             n_times = 8L, n_groups = 1L, group_index = NULL, trend = "rw1",
             seasonal = 4L, short_term = "ar1", shared = TRUE,
             parameterization = "noncentered",
             sigma2_trend_prior_U = 2, sigma2_trend_prior_alpha = 0.05,
             sigma2_seasonal_prior_U = 0.5, sigma2_seasonal_prior_alpha = 0.01,
             sigma2_short_prior_U = 1, sigma2_short_prior_alpha = 0.1)
  s <- start_of(f, temporal_spec = tp)
  expect_equal(unname(s[c("log_sigma2_trend", "log_sigma2_seasonal",
                          "log_sigma2_short")]),
               2 * c(pc_median_log_sd(2, 0.05), pc_median_log_sd(0.5, 0.01),
                     pc_median_log_sd(1, 0.1)),
               tolerance = 1e-12)
  tp$parameterization <- "centered"
  expect_equal(unname(start_of(f, temporal_spec = tp)[
    c("log_sigma2_trend", "log_sigma2_seasonal", "log_sigma2_short")]),
    c(0, 0, 0))
})
