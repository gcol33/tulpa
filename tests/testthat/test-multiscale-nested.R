# Multi-scale fields on the nested-Laplace path: spatial_multiscale() as two
# field blocks of the multi-block driver (two NNGP blocks, or two HSGP blocks
# on one basis), temporal_multiscale() as one block per component (an RW1 /
# RW2 trend, a cyclic RW1 seasonal cycle, an AR1 or iid short-term component),
# each block with its own scale (and range / correlation) axes on the outer
# grid. Checked against a simulated truth.

ms_sim_spatial <- function(n = 150L, s2_loc = 0.4, phi_loc = 0.5, s2_reg = 1.0,
                           phi_reg = 3, seed = 11L) {
  set.seed(seed)
  lon <- runif(n, 0, 10); lat <- runif(n, 0, 10)
  D <- as.matrix(dist(cbind(lon, lat)))
  draw <- function(s2, phi) {
    as.numeric(t(chol(s2 * exp(-D / phi) + diag(1e-8, n))) %*% rnorm(n))
  }
  w_loc <- draw(s2_loc, phi_loc); w_reg <- draw(s2_reg, phi_reg)
  x <- rnorm(n)
  y <- rpois(n, exp(0.5 + 0.4 * x + w_loc + w_reg))
  list(d = data.frame(y, x, lon, lat), w_loc = w_loc, w_reg = w_reg,
       sigma = c(local = sqrt(s2_loc), regional = sqrt(s2_reg)),
       phi = c(local = phi_loc, regional = phi_reg))
}

ms_sim_temporal <- function(n_t = 80L, reps = 4L, seed = 21L, sd_rw2 = 0.02,
                            rho = 0.6, sd_ar = 0.3) {
  set.seed(seed)
  trend <- cumsum(cumsum(rnorm(n_t, 0, sd_rw2))); trend <- trend - mean(trend)
  short <- as.numeric(arima.sim(list(ar = rho), n_t, sd = sd_ar))
  time <- rep(seq_len(n_t), each = reps)
  x <- rnorm(length(time))
  y <- rpois(length(time), exp(1 + 0.4 * x + trend[time] + short[time]))
  list(d = data.frame(y, x, time), trend = trend, short = short,
       sigma = c(trend = sd_rw2, short_term = sd_ar), rho = rho)
}

ms_spec <- function(...) {
  spatial_multiscale(~ lon + lat, range_local = c(0.2, 1.5),
                     range_regional = c(1.5, 10), nn_local = 10L,
                     nn_regional = 15L, scale_coords = FALSE, ...)
}

test_that("a multi-scale spatial spec becomes two named NNGP blocks", {
  set.seed(1)
  d <- data.frame(lon = runif(30), lat = runif(30))
  spec <- validate_gp(spatial_multiscale(~ lon + lat, range_local = c(0.01, 0.2),
                                         range_regional = c(0.3, 2),
                                         nn_local = 4L, nn_regional = 6L,
                                         scale_coords = FALSE), d)
  blocks <- tulpa:::.spatial_spec_to_nl_blocks(spec)
  expect_length(blocks, 2L)
  expect_identical(vapply(blocks, `[[`, "", "type"), c("nngp", "nngp"))
  expect_identical(vapply(blocks, `[[`, "", "name"), c("local", "regional"))
  expect_identical(vapply(blocks, function(b) b$nn, 1L), c(4L, 6L))
  # Each scale's range prior is anchored at its own interval's lower bound,
  # and both routes read the one statement of the field's hyperpriors.
  expect_equal(blocks[[1L]]$prior_range, c(0.01, tulpa:::.MSGP_RANGE_ALPHA))
  expect_equal(blocks[[2L]]$prior_range, c(0.3, tulpa:::.MSGP_RANGE_ALPHA))
  ss <- tulpa:::.msgp_sampler_spec(spec)
  expect_equal(blocks[[1L]]$prior_sigma,
               c(ss$sigma2_local_prior_U, ss$sigma2_local_prior_alpha))
  expect_equal(ss$range_regional_prior_alpha, blocks[[2L]]$prior_range[2L])
})

test_that("a multi-scale temporal spec becomes one named block per component", {
  d <- data.frame(t = rep(1:24, 2))
  spec <- validate_temporal_multiscale(
    temporal_multiscale("t", trend = "rw2", seasonal = 12L, short_term = "ar1"), d)
  blocks <- tulpa:::.temporal_spec_to_nl_blocks(spec)
  expect_identical(vapply(blocks, `[[`, "", "type"), c("rw2", "rw1", "ar1"))
  expect_identical(vapply(blocks, `[[`, "", "name"),
                   c("trend", "seasonal", "short_term"))
  # The seasonal cycle is a ring over the period, read at (t - 1) %% period + 1,
  # the index the sampler reads it at.
  expect_true(blocks[[2L]]$cyclic)
  expect_identical(blocks[[2L]]$n_times, 12L)
  expect_identical(blocks[[2L]]$temporal_idx, (d$t - 1L) %% 12L + 1L)
  expect_equal(tulpa:::.ar1_rho_beta_ab(blocks[[3L]]$rho_prior),
               unname(tulpa:::.MS_TEMPORAL_RHO_BETA))
  iid <- tulpa:::.temporal_spec_to_nl_blocks(validate_temporal_multiscale(
    temporal_multiscale("t", trend = "none", short_term = "iid"), d))
  expect_identical(iid[[1L]]$type, "iid")
  expect_identical(iid[[1L]]$n_units, 24L)
  expect_error(tulpa:::.temporal_spec_to_nl_blocks(validate_temporal_multiscale(
    temporal_multiscale("t", group_var = "g"), cbind(d, g = rep(1:2, 24)))),
    "per-group")
})

test_that("auto and structured route multi-scale fields to nested Laplace", {
  fam <- list(name = "poisson", distribution = "poisson")
  sp <- tulpa:::select_inference_mode("auto", family = fam, n_obs = 100L,
                                      has_spatial = TRUE,
                                      spatial_type = "multiscale")
  expect_identical(sp$backend, "nested_laplace")
  expect_identical(
    tulpa:::select_backend_for_mode("structured", fam, 100L, TRUE, FALSE,
                                    spatial_type = "multiscale"),
    "nested_laplace")
  tp <- tulpa:::select_inference_mode("auto", family = fam, n_obs = 100L,
                                      has_temporal = TRUE,
                                      temporal = list(type = "multiscale"))
  expect_identical(tp$backend, "nested_laplace")
})

test_that("field_effects() refuses a component on a single-scale field", {
  fit <- structure(list(spatial = list(type = "icar")), class = "tulpa_fit")
  expect_error(field_effects(fit, component = "local"), "multi-scale")
})

skip_on_cran()

test_that("nested multi-scale spatial fit recovers both scales against the truth", {
  skip_on_cran()
  s <- ms_sim_spatial()
  fit <- suppressMessages(tulpa(y ~ x, data = s$d, family = "poisson",
                                spatial = ms_spec(),
                                control = list(progress = FALSE)))
  expect_identical(fit$backend, "nested_laplace")
  sr <- spatial_range(fit)
  expect_setequal(rownames(sr), c("sigma[local]", "range[local]",
                                  "sigma[regional]", "range[regional]"))
  # Every interval covers its truth; a range row is 3 phi.
  for (sc in c("local", "regional")) {
    expect_gte(s$sigma[[sc]], sr[sprintf("sigma[%s]", sc), "q2.5"])
    expect_lte(s$sigma[[sc]], sr[sprintf("sigma[%s]", sc), "q97.5"])
    expect_gte(3 * s$phi[[sc]], sr[sprintf("range[%s]", sc), "q2.5"])
    expect_lte(3 * s$phi[[sc]], sr[sprintf("range[%s]", sc), "q97.5"])
  }
  # The scales stay apart: the regional range sits above the local one.
  expect_gt(sr["range[regional]", "mean"], sr["range[local]", "mean"])
  # Component maps against the simulated fields, and the summed field.
  fl <- field_effects(fit, "local")
  fr <- field_effects(fit, "regional")
  fs <- field_effects(fit)
  expect_equal(nrow(fs), nrow(s$d))
  expect_gt(cor(fr$estimate, s$w_reg), 0.7)
  expect_gt(cor(fl$estimate, s$w_loc), 0.3)
  expect_gt(cor(fs$estimate, s$w_loc + s$w_reg), 0.8)
  # The sum is the loading of both blocks: its mean is the sum of the
  # component means, and its 95% band covers most of the true field.
  expect_equal(fs$estimate, fl$estimate + fr$estimate, tolerance = 1e-8)
  w <- s$w_loc + s$w_reg
  expect_gt(mean(w >= fs$conf.low & w <= fs$conf.high), 0.85)
})

test_that("nested multi-scale temporal fit recovers each component", {
  skip_on_cran()
  s <- ms_sim_temporal()
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$d, family = "poisson",
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "ar1"),
    control = list(progress = FALSE)))
  expect_identical(fit$backend, "nested_laplace")
  tg <- as.matrix(fit$theta_grid)
  expect_setequal(colnames(tg), c("b1.tau", "b2.tau", "b2.rho"))
  w <- fit$weights / sum(fit$weights)
  wq <- function(v, p) { o <- order(v); v[o][findInterval(p, cumsum(w[o])) + 1L] }
  covers <- function(v, truth) wq(v, 0.025) <= truth && truth <= wq(v, 0.975)
  expect_true(covers(1 / sqrt(tg[, "b1.tau"]), s$sigma[["trend"]]))
  expect_true(covers(1 / sqrt(tg[, "b2.tau"]), s$sigma[["short_term"]]))
  expect_true(covers(tg[, "b2.rho"], s$rho))
  tp <- temporal(fit)
  expect_setequal(names(tp$draws), c("trend", "short_term"))
  expect_gt(cor(colMeans(tp$draws$trend), s$trend), 0.95)
  expect_gt(cor(colMeans(tp$draws$short_term), s$short), 0.4)
  tc <- temporal_corr(fit)
  expect_true(all(c("precision[trend]", "precision[short_term]") %in% rownames(tc)))
})

test_that("an iid short-term component is the field's, not a random effect", {
  skip_on_cran()
  set.seed(4)
  n_t <- 30L; reps <- 4L
  time <- rep(seq_len(n_t), each = reps)
  g <- rep(1:6, length.out = length(time))
  short <- rnorm(n_t, 0, 0.4); u <- rnorm(6, 0, 0.5)
  d <- data.frame(time, g = factor(g), x = rnorm(length(time)))
  d$y <- rpois(nrow(d), exp(0.8 + 0.3 * d$x + short[time] + u[g]))
  fit <- suppressMessages(tulpa(
    y ~ x + (1 | g), data = d, family = "poisson",
    temporal = temporal_multiscale("time", trend = "rw1", short_term = "iid"),
    control = list(progress = FALSE)))
  re <- ranef(fit)
  expect_equal(nrow(as.data.frame(re[[1L]])), 6L)
  tp <- temporal(fit)
  sh <- colMeans(tp$draws$short_term)
  expect_length(sh, n_t)
  # The iid block is non-centred; its field is the coordinate times the cell's
  # SD, which is what tracks the simulated values.
  expect_gt(cor(sh, short), 0.5)
})

test_that("an HSGP multi-scale field fits on the nested path and nowhere else", {
  skip_on_cran()
  s <- ms_sim_spatial(n = 120L, seed = 3L)
  spec <- ms_spec(approx = "hsgp", m = 8L)
  fit <- suppressMessages(tulpa(y ~ x, data = s$d, family = "poisson",
                                spatial = spec, control = list(progress = FALSE)))
  expect_identical(fit$backend, "nested_laplace")
  expect_identical(vapply(fit$prior, `[[`, "", "type"), c("hsgp", "hsgp"))
  sr <- spatial_range(fit)
  expect_true(all(c("sigma[local]", "sigma[regional]") %in% rownames(sr)))
  expect_error(suppressMessages(tulpa(y ~ x, data = s$d, family = "poisson",
                                      spatial = spec, mode = "exact")),
               "nested-Laplace path")
})

test_that("mode = 'gibbs' runs the temporal Polya-Gamma sampler where it carries the field", {
  skip_on_cran()
  set.seed(2)
  tt <- rep(1:20, each = 4)
  d <- data.frame(time = tt, x = rnorm(80))
  d$y <- rbinom(80, 10, plogis(0.2 * d$x + sin(tt / 4)))
  fit <- suppressWarnings(tulpa(
    y ~ x, data = d, family = "binomial", n_trials = rep(10L, 80),
    temporal = temporal_multiscale("time", trend = "rw1", short_term = "none"),
    mode = "gibbs", control = list(n_iter = 60L, warmup = 30L, seed = 1L)))
  expect_identical(fit$backend, "gibbs")
  expect_true(any(grepl("^trend\\[", colnames(fit$draws))))
  # An RW2 trend is outside that sampler; the redirect says so.
  expect_warning(
    tulpa(y ~ x, data = d, family = "binomial", n_trials = rep(10L, 80),
          temporal = temporal_multiscale("time", trend = "rw2", short_term = "none"),
          mode = "gibbs"),
    "rw2 trend")
})
