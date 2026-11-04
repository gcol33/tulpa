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
  # The field is one walk per component on every route; there is no per-group
  # form to ask for.
  expect_error(temporal_multiscale("t", group_var = "g"), "unused argument")
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

test_that("summary() labels a named block's axes by the block's name", {
  # The driver addresses an axis by its block's position (`b1.sigma`); the
  # report names the scale, as spatial_range() does.
  blk <- function(nm) list(type = "nngp", name = nm)
  fit <- structure(list(
    prior = list(blk("local"), blk("regional")),
    theta_names = c("b1.sigma", "b1.phi_gp", "b2.sigma", "b2.phi_gp"),
    theta_mean = c(0.6, 0.5, 1, 3), theta_sd = rep(0.1, 4),
    theta_median = c(0.6, 0.5, 1, 3), theta_ci_lo = c(0.4, 0.3, 0.7, 2),
    theta_ci_hi = c(0.8, 0.7, 1.3, 4)), class = "tulpa_fit")
  expect_identical(tulpa:::.fit_hyper_table(fit)$parameter,
                   c("sigma[local]", "phi_gp[local]", "sigma[regional]",
                     "phi_gp[regional]"))
  # A block without a name keeps the positional label.
  fit$prior[[2L]]$name <- NULL
  expect_identical(tulpa:::.fit_hyper_table(fit)$parameter[3:4],
                   c("b2.sigma", "b2.phi_gp"))
})

test_that("the sampler route names a multi-scale field's rows as the nested one does", {
  set.seed(1)
  dr <- matrix(rnorm(400, -1, 0.3), 100, 4,
               dimnames = list(NULL, c("log_sigma2_gp_local", "log_phi_gp_local",
                                       "log_sigma2_gp_regional",
                                       "log_phi_gp_regional")))
  sr <- spatial_range(structure(list(draws = dr, spatial = list(type = "multiscale")),
                                class = "tulpa_fit"))
  expect_setequal(rownames(sr), c("sigma[local]", "range[local]",
                                  "sigma[regional]", "range[regional]"))
  expect_equal(sr["range[local]", "mean"], mean(3 * exp(dr[, 2])))
  dt <- matrix(rnorm(500, -1, 0.3), 100, 5,
               dimnames = list(NULL, c("log_sigma2_trend", "log_sigma2_seasonal",
                                       "log_sigma2_short", "logit_rho_short",
                                       "trend[1]")))
  tc <- temporal_corr(structure(list(draws = dt), class = "tulpa_fit"))
  expect_setequal(rownames(tc), c("sigma[trend]", "sigma[seasonal]",
                                  "sigma[short_term]", "rho[short_term]"))
  expect_equal(tc["sigma[trend]", "mean"], mean(exp(dt[, 1] / 2)))
  expect_equal(tc["rho[short_term]", "mean"], mean(2 * plogis(dt[, 4]) - 1))
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
  expect_true(all(c("sigma[trend]", "sigma[short_term]", "rho[short_term]") %in%
                    rownames(tc)))
})

test_that("nested multi-scale temporal intervals match the model's dense grid", {
  skip_on_cran()
  s <- ms_sim_temporal()
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$d, family = "poisson",
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "ar1"),
    control = list(progress = FALSE)))
  h <- summary(fit)$hyperparameters
  rownames(h) <- h$parameter
  # summary() names an axis by its block's name (trend, short_term).
  lab <- stats::setNames(
    tulpa:::.hyper_axis_labels(fit, c("b1.tau", "b2.tau", "b2.rho")),
    c("b1.tau", "b2.tau", "b2.rho"))
  got <- rbind(
    sigma_trend = 1 / sqrt(unlist(h[lab[["b1.tau"]], c("ci_hi", "ci_lo")])),
    sigma_short = 1 / sqrt(unlist(h[lab[["b2.tau"]], c("ci_hi", "ci_lo")])),
    rho = unlist(h[lab[["b2.rho"]], c("ci_lo", "ci_hi")]))
  # Reference: the same blocks and priors on a 31^3 tensor over log tau_trend
  # [3.5, 12], log tau_short [0.6, 3.6] and rho [-0.7, 0.985] (outer-node
  # masses at most 2e-4), each axis read by its cells' boxes; columns 95%
  # lower end, upper end and posterior SD. The cheap screen at a tolerance of
  # 1e-3 per cell dropped the tail cells of this grid and left rho's lower end
  # at 0.094.
  ref <- rbind(sigma_trend = c(0.0086, 0.0523, 0.0117),
               sigma_short = c(0.2561, 0.4593, 0.0519),
               rho         = c(0.0607, 0.7843, 0.1846))
  expect_true(all(abs(got[, 1L] - ref[, 1L]) < 0.15 * ref[, 3L]))
  expect_true(all(abs(got[, 2L] - ref[, 2L]) < 0.15 * ref[, 3L]))
})

test_that("temporal() draws the components of a row from one latent draw", {
  skip_on_cran()
  s <- ms_sim_temporal()
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$d, family = "poisson",
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "ar1"),
    control = list(progress = FALSE)))
  # 160 field columns over a few hundred cells: past the retained-covariance
  # limit, so the within-cell covariance is not kept on the fit.
  expect_null(fit$grid_field_cov)
  set.seed(3)
  tp <- temporal(fit)
  S <- tp$draws$trend + tp$draws$short_term
  # The exact moments of the fit's own posterior: each cell's constrained
  # covariance over both blocks, from its precision solved again.
  src <- tulpa:::.field_cells(fit)
  w <- fit$weights / sum(fit$weights)
  on <- which(w > 0)
  sol <- tulpa:::.field_cell_solve(fit, src, on)
  n_x <- ncol(fit$modes)
  idx <- c(tulpa:::.nl_block_cols(fit$block_latent_offsets, n_x, 1L, 2L),
           tulpa:::.nl_block_cols(fit$block_latent_offsets, n_x, 2L, 2L))
  V <- tulpa:::cpp_joint_inner_vcov_blocks(
    Q_p_per_grid = sol$Q_p, Q_i_per_grid = sol$Q_i, Q_x_per_grid = sol$Q_x,
    n_x = as.integer(n_x), idx = as.integer(idx), n_dense = length(idx),
    A_cols_list = tulpa:::.constraint_rows_for_cpp(sol$constraint_cols),
    field_marginal = FALSE, n_threads = 1L)
  ww <- w[on] / sum(w[on])
  M <- fit$modes[on, idx, drop = FALSE]
  lin_sd <- function(A) {
    mk <- M %*% A
    v <- Reduce(`+`, lapply(seq_along(on), function(i)
      ww[i] * (diag(t(A) %*% V[[i]] %*% A) + mk[i, ]^2)))
    sqrt(v - colSums(ww * mk)^2)
  }
  n_t <- ncol(S)
  # trend[t] + short[t] at each t, and the change between the first and the
  # last time point. Drawn per component with independent cells and
  # independent coordinates the first read 1.54x its exact SD on these data
  # and the second 1.48x; one latent draw per row reads 1.00 and 0.97.
  ex_t <- lin_sd(rbind(diag(n_t), diag(n_t)))
  expect_lt(abs(mean(apply(S, 2, sd) / ex_t) - 1), 0.04)
  a <- c(-1, rep(0, n_t - 2L), 1)
  ex_d <- lin_sd(matrix(c(a, a), ncol = 1L))
  expect_lt(abs(sd(S[, n_t] - S[, 1L]) / ex_d - 1), 0.1)
})

test_that("temporal_corr() reports the SD summary() reports for an axis", {
  skip_on_cran()
  s <- ms_sim_temporal(n_t = 40L)
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$d, family = "poisson",
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "ar1"),
    control = list(progress = FALSE)))
  sd_of <- function(f) {
    h <- summary(f)$hyperparameters
    tc <- temporal_corr(f)
    c(summary = h$sd[h$parameter == "rho[short_term]"],
      corr = tc["rho[short_term]", "sd"])
  }
  r <- sd_of(fit)
  expect_equal(r[["corr"]], r[["summary"]], tolerance = 1e-12)
  # An axis whose marginal is read by the parabola at its modal level reports
  # that SD (the stencil); temporal_corr() reported the nodes' weighted spread
  # beside it, 0.452 against summary()'s 0.561 on a fit that took it.
  st <- fit
  st$theta_sd_source[["b2.rho"]] <- "stencil"
  st$theta_sd[["b2.rho"]] <- 0.5
  r <- sd_of(st)
  expect_equal(r[["summary"]], 0.5)
  expect_equal(r[["corr"]], 0.5, tolerance = 1e-12)
  # A derived row carries the stencil's Gaussian to its own quantity: sigma =
  # tau^(-1/2) at the modal precision, |d sigma / d tau| = tau^(-3/2) / 2.
  st$theta_sd_source[["b2.tau"]] <- "stencil"
  st$theta_sd[["b2.tau"]] <- 3
  tg <- as.matrix(st$theta_grid)
  lm <- st$log_marginal + if (length(st$log_quad) == length(st$log_marginal))
    st$log_quad else 0
  marg <- tulpa:::.nl_axis_marginal_logdensity(tg[, "b2.tau"], lm)
  mode <- attr(tulpa:::.nl_laplace_at_mode_sd_axis(
    marg$vals, marg$log_marg, mass = length(st$log_quad) == length(lm)), "mode")
  expect_equal(temporal_corr(st)["sigma[short_term]", "sd"],
               3 * mode^(-1.5) / 2, tolerance = 1e-6)
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
  expect_length(coef(fit), 2L)
  expect_true(all(is.finite(coef(fit))))
  V <- vcov(fit)
  expect_equal(dim(V), c(2L, 2L))
  expect_true(all(is.finite(V)))
  expect_true(all(diag(V) > 0))
  expect_true(is.finite(fit$log_evidence))
  expect_true(all(is.finite(fit$weights)))
  expect_equal(sum(fit$weights), 1)
  expect_length(fit$grid_hessians, nrow(fit$theta_grid))
  unavailable <- !is.finite(fit$log_marginal)
  expect_equal(fit$weights[unavailable], rep(0, sum(unavailable)))
  expect_true(all(vapply(fit$grid_hessians[unavailable], is.null, logical(1))))
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


test_that("a nested fit's solved cells are kept for the next accessor call (#1125)", {
  skip_on_cran()
  s <- ms_sim_temporal()
  fit <- suppressMessages(tulpa(
    y ~ x, data = s$d, family = "poisson",
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "ar1"),
    control = list(progress = FALSE)))
  src <- tulpa:::.field_cells(fit)
  on <- which(fit$weights > 0)
  sol1 <- tulpa:::.field_cell_solve(fit, src, on)
  # Every cell is now solved: a further read does not reach the kernel, and
  # returns the precisions it solved.
  local_mocked_bindings(.field_cell_resolve = function(...) {
    stop("the kernel was reached")
  })
  sol2 <- tulpa:::.field_cell_solve(fit, src, rev(on))
  expect_identical(sol2$Q_x, rev(sol1$Q_x))
  expect_identical(sol2$constraint_cols, sol1$constraint_cols)
  expect_silent(temporal(fit))
  # A cell whose mode is no longer the one it was solved at is solved again.
  fit2 <- fit
  fit2$modes[on[1L], 1L] <- fit2$modes[on[1L], 1L] + 1e-3
  expect_error(tulpa:::.field_cell_solve(fit2, tulpa:::.field_cells(fit2), on),
               "the kernel was reached")
})
