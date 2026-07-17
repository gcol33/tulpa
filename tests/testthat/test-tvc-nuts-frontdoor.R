# Front-door exact NUTS for temporally-varying coefficients (gcol33/tulpa#158d).
# tulpa(y ~ x, temporal = temporal_tvc("year", terms = ~ x - 1), mode = "exact")
# routes through the generic ModelData sampler, which allocates the per-term
# log_tau (+ logit_rho for AR1) and the [n_groups x n_tvc x n_times] field
# (keyed on data.has_tvc) and samples them jointly. eta_i += sum_j X_tvc[i,j]
# w_j(g_i, t_i). Under an RW1 walk the field level is confounded with the fixed
# coefficient, so the identified quantity is the TOTAL slope fixed_x + w_1(t).
# TVC is exact-only (no nested front door).

make_tvc_pois <- function(n_t = 20L, reps = 8L, seed = 505L) {
  set.seed(seed)
  walk <- cumsum(rnorm(n_t, 0, 0.35)); walk <- walk - mean(walk)   # centred RW1
  year <- rep(seq_len(n_t), each = reps); N <- length(year)
  x <- rnorm(N)
  d <- data.frame(year = year, x = x)
  d$y <- rpois(N, exp(0.3 + (0.5 + walk[year]) * x))
  list(d = d, walk = walk, n_t = n_t)
}

test_that("tvc requires an exact mode", {
  s <- make_tvc_pois(n_t = 6L, reps = 4L)
  expect_error(
    tulpa(y ~ x, data = s$d, family = "poisson",
          temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
          mode = "structured"),
    "Temporally-varying"
  )
})

test_that("tvc exact NUTS allocates the per-time field (structural)", {
  skip_if_not_slow()
  s <- make_tvc_pois(n_t = 8L, reps = 4L)
  fit <- tulpa(y ~ x, data = s$d, family = "poisson",
               temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
               mode = "exact",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L))
  pn <- colnames(fit$draws)
  expect_true("log_tau_tvc[1]" %in% pn)
  expect_equal(sum(grepl("^tvc_w\\[", pn)), 8L)     # one field value per time
  expect_equal(ncol(fit$draws), 3L + 8L)             # 2 fixed + log_tau + field
})

test_that("tvc exact NUTS recovers the varying-coefficient trajectory", {
  skip_if_not_slow()
  s <- make_tvc_pois()
  fit <- tulpa(y ~ x, data = s$d, family = "poisson",
               temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
               mode = "exact",
               control = list(n_iter = 350L, n_warmup = 175L, seed = 6L))
  wcol <- grep("^tvc_w\\[", colnames(fit$draws))
  fm <- colMeans(fit$draws[, wcol, drop = FALSE])       # w_1(t)
  # The total varying slope beta_x(t) = fixed_x + w_1(t) is the identified
  # quantity (RW1 level is confounded with the fixed coefficient).
  total <- unname(coef(fit)["x"]) + fm
  expect_gt(cor(total, 0.5 + s$walk), 0.8)
  expect_lt(abs(mean(total) - 0.5), 0.35)
})
