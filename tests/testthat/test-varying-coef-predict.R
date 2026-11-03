# Varying-coefficient fields read off a sampler fit, and predicted at new
# coordinates and times.
#
# Three claims:
#   1. THE READ -- svc() / tvc() return the field the likelihood read at each
#      draw, on every field representation (HSGP basis coefficients, a
#      non-centered NNGP z, a centred w) and on every backend that stores the
#      draws differently. The check is the engine's own linear predictor: its
#      eta minus the fixed part is exactly sum_j x_ij w_j at every draw.
#   2. THE CONDITIONAL -- at a new coordinate / time each draw's coefficient is
#      drawn from the closed-form conditional given that draw: the NNGP / GP
#      kriging conditional, the Brownian bridge and random-walk steps, the AR1
#      recursion. A proper field's level is removed from eta, so the stored
#      field is centred and its level mbar is drawn from N(-1'Q w_c / 1'Q 1,
#      1 / 1'Q 1) before the conditional is applied; the reference below is the
#      dense Gaussian algebra of that, checked by Monte Carlo over one
#      replicated draw.
#   3. THE POINT -- on simulated data, predict() at held-out coordinates and
#      times covers the true linear predictor, and beats the average-site /
#      average-time line it used to return.

# The grid structures' placement of new times, read without a fit.
.tvcq_info <- function(structure = "rw1", levels = 1:6, group_var = NULL,
                       group_levels = NULL) {
  structure(list(time_var = "year", structure = structure,
                 level_values = as.numeric(levels),
                 time_levels = as.character(levels), n_times = length(levels),
                 group_var = group_var, group_levels = group_levels),
            class = c("tulpa_tvc", "tulpa_temporal", "list"))
}

test_that("new times are placed on the grid of the fitted levels", {
  q <- tulpa:::.temporal_query(.tvcq_info("rw1"),
                          data.frame(year = c(2, 3.25, 8, 0)))
  expect_identical(q$exact, c(1L, -1L, -1L, -1L))
  expect_equal(q$pos, c(1, 2.25, 7, -1))
  expect_identical(q$group, rep(0L, 4))

  # rw2 / ar1 have no value between grid positions, nor off the step lattice.
  expect_error(tulpa:::.temporal_query(.tvcq_info("rw2"), data.frame(year = 3.5)),
               "between two fitted levels")
  expect_error(tulpa:::.temporal_query(.tvcq_info("ar1"), data.frame(year = 7.5)),
               "fall off it")
  expect_equal(tulpa:::.temporal_query(.tvcq_info("ar1"), data.frame(year = 9))$pos, 8)

  # Unequal spacing: a gap is one step, so beyond the levels there is no
  # step count; between them rw1 still has a position.
  uneq <- .tvcq_info("rw1", levels = c(1, 2, 4, 7))
  expect_equal(tulpa:::.temporal_query(uneq, data.frame(year = 5.5))$pos, 2.5)
  expect_error(tulpa:::.temporal_query(uneq, data.frame(year = 8)), "not uniform")

  # A group without a fitted path, and a missing time column, are refused.
  grp <- .tvcq_info("rw1", group_var = "site", group_levels = c("a", "b"))
  expect_identical(
    tulpa:::.temporal_query(grp, data.frame(year = 1, site = "b"))$group, 1L)
  expect_error(tulpa:::.temporal_query(grp, data.frame(year = 1, site = "c")),
               "no field for: c")
  expect_error(tulpa:::.temporal_query(.tvcq_info(), data.frame(x = 1)),
               "missing the time variable")
})

skip_on_cran()

.vcp_data <- function(n = 120L, seed = 1L) {
  set.seed(seed)
  d <- data.frame(lon = stats::runif(n), lat = stats::runif(n),
                  x = stats::rnorm(n), z = stats::rnorm(n))
  d$y <- stats::rpois(n, exp(0.3 + 0.7 * d$x))
  d$year <- rep(seq_len(12L), length.out = n)[order(stats::runif(n))]
  d
}

.vcp_fit <- function(d, ..., mode = "exact") {
  ctl <- if (mode == "exact") list(n_iter = 40L, n_warmup = 30L, seed = 1L)
         else list(seed = 1L)
  suppressWarnings(tulpa(y ~ x, data = d, family = "poisson", mode = mode,
                         control = ctl, ...))
}

# eta the engine assembles at each draw, minus the fixed part, against the
# varying-coefficient contribution assembled from the accessor's fields.
.vcp_identity_gap <- function(fit) {
  E <- tulpa:::.sampler_engine_eta(fit, fit$draws, offset = NULL)
  beta <- tulpa:::.fixed_coef_draws(fit)$beta
  fixed <- beta %*% t(fit$model_inputs$X)
  vc <- tulpa:::.varying_coef_eta_contrib(fit, NULL)
  c(gap = max(abs(E - fixed - vc)), scale = max(abs(vc)))
}

test_that("svc() reads an HSGP fit through its basis, and it is the field eta read", {
  skip_on_cran()
  d <- .vcp_data()
  fit <- .vcp_fit(d, spatial = spatial_svc(~ lon + lat, terms = ~ x - 1,
                                           approx = "hsgp", m = 4L))
  post <- svc(fit)
  expect_identical(dim(post$draws), c(nrow(fit$draws), nrow(d), 1L))

  # The basis expansion recomputed from the draw columns with the R-facing
  # basis builder and the 2-D squared-exponential spectral density, centred
  # over the fitted rows: the field is proper, so its level aliases the slope
  # and is removed on its way into eta.
  cm <- tulpa:::.coords_plain(fit$spatial$coords_matrix)
  B <- tulpa:::cpp_hsgp_basis_2d(cm, 4L, fit$spatial$c_boundary)
  s2 <- exp(fit$draws[, "log_sigma2_svc[1]"])
  ls <- exp(fit$draws[, "log_phi_svc[1]"])
  beta <- tulpa:::.draws_by_prefix(fit$draws, "svc_w")
  ref <- t(vapply(seq_len(nrow(beta)), function(s) {
    S <- s2[s] * 2 * pi * ls[s]^2 * exp(-0.5 * ls[s]^2 * B$lambda_eig)
    f <- as.numeric(B$phi_basis %*% (sqrt(S) * beta[s, ]))
    f - mean(f)
  }, numeric(nrow(d))))
  expect_equal(unname(post$draws[, , 1L]), ref, tolerance = 1e-10)

  g <- .vcp_identity_gap(fit)
  expect_lt(g[["gap"]], 1e-10 * max(1, g[["scale"]]))

  # The training coordinates through the new-coordinate path: the same field.
  at_train <- tulpa:::.varying_coef_field_draws(fit, svc_coords = cm)$svc
  expect_equal(at_train, unname(post$draws[, , 1L]), tolerance = 1e-12)

  # Outside the basis domain the eigenfunctions approximate no GP.
  far <- data.frame(x = 1, lon = 5, lat = 5)
  expect_error(predict(fit, newdata = far), "outside the HSGP domain")
})

test_that("svc() / tvc() read the field the likelihood read on every backend", {
  skip_on_cran()
  d <- .vcp_data()
  fits <- list(
    nngp_nc = .vcp_fit(d, spatial = spatial_svc(~ lon + lat, terms = ~ x - 1,
                                                nn = 5L)),
    nngp_vi = .vcp_fit(d, spatial = spatial_svc(~ lon + lat, terms = ~ x - 1,
                                                nn = 5L), mode = "vi"),
    tvc_ar1_vi = .vcp_fit(d, temporal = temporal_tvc("year", terms = ~ x - 1,
                                                     structure = "ar1"),
                          mode = "vi"),
    tvc_gp = .vcp_fit(d, temporal = temporal_tvc("year", terms = ~ x - 1,
                                                 structure = "gp"))
  )
  for (nm in names(fits)) {
    g <- .vcp_identity_gap(fits[[nm]])
    expect_lt(g[["gap"]], 1e-9 * max(1, g[["scale"]]), label = nm)
  }
  # On VI, as on every ModelData backend, the stored NNGP slice is the
  # reconstructed, centred w (src/sampler_field_store.h), and the accessor's
  # field is that slice.
  w <- svc(fits$nngp_vi)$draws[, , 1L]
  expect_lt(max(abs(rowMeans(w))), 1e-10)
  stored <- tulpa:::.draws_by_prefix(fits$nngp_vi$draws, "svc_w")
  expect_lt(max(abs(w - stored)), 1e-10)
  # A VI TVC draw is stored centred, and the accessor's field with it.
  expect_lt(max(abs(apply(tvc(fits$tvc_ar1_vi)$draws[, , 1L], 1L, mean))), 1e-10)
})

test_that("tvc() reads a grouped field, keyed by group, time and term", {
  skip_on_cran()
  set.seed(5)
  n_t <- 6L
  d <- expand.grid(year = seq_len(n_t), site = c("a", "b", "c"), rep = 1:4)
  d$x <- stats::rnorm(nrow(d))
  d$y <- stats::rpois(nrow(d), exp(0.3 + 0.5 * d$x))
  fit <- .vcp_fit(d, temporal = temporal_tvc("year", terms = ~ x - 1,
                                             structure = "rw1",
                                             group_var = "site"))
  post <- tvc(fit)
  expect_identical(dim(post$draws), c(nrow(fit$draws), 3L * n_t, 1L))
  expect_identical(post$groups, rep(c("a", "b", "c"), each = n_t))

  # Each (group, time) unit is the (group, term, time) block the eta assembly
  # reads, unit for unit.
  raw <- tulpa:::.varying_coef_field_draws(fit)$tvc
  for (g in 1:3) {
    expect_equal(post$draws[, (g - 1L) * n_t + seq_len(n_t), 1L],
                 raw[, (g - 1L) * n_t + seq_len(n_t)])
  }
  s <- summary(post)
  expect_identical(names(s)[1:4], c("time_idx", "time", "group", "term"))
  expect_identical(nrow(s), 3L * n_t)
  expect_equal(s$mean[s$group == "b"],
               unname(colMeans(post$draws[, n_t + seq_len(n_t), 1L])))
  expect_no_error(plot(post))

  # predict() reads the same field at a fitted (group, year).
  nd <- data.frame(year = 2, site = "c", x = 1)
  ref <- mean(tulpa:::.fixed_coef_draws(fit)$beta %*% c(1, 1) +
                post$draws[, 2L * n_t + 2L, 1L])
  expect_equal(predict(fit, newdata = nd), ref, tolerance = 1e-10)
})

# Monte Carlo over one draw replicated S times, against the closed-form mean and
# variance of the predicted coefficient.
.vcp_replicate <- function(fit, S = 20000L, row = nrow(fit$draws)) {
  fit$draws <- fit$draws[rep(row, S), , drop = FALSE]
  fit
}

.vcp_expect_moments <- function(draws, mean_ref, var_ref, label) {
  S <- length(draws)
  expect_lt(abs(mean(draws) - mean_ref), 5 * sqrt(var_ref / S) + 1e-6,
            label = paste(label, "mean"))
  expect_lt(abs(stats::var(draws) / var_ref - 1), 0.06,
            label = paste(label, "variance"))
}

# A proper field of covariance K over the fitted points, centred on its way
# into eta: the predicted coefficient y0 = w(s0) - mean(w), conditioned on the
# n_nb points `nb` with cross-covariance k0 and prior variance v0.
.vcp_proper_ref <- function(K, wc, nb, k0, v0) {
  one <- rep(1, nrow(K))
  Ki1 <- solve(K, one)
  q <- sum(Ki1)
  mh <- -sum(Ki1 * wc) / q
  a <- solve(K[nb, nb, drop = FALSE], k0)
  list(mean = sum(a * (wc[nb] + mh)) - mh,
       var = v0 - sum(k0 * a) + (sum(a) - 1)^2 / q)
}

# The SVC NNGP density conditions every Vecchia factor with a diagonal nugget on
# the neighbour block and a blended floor on the conditional variance
# (`kSvcJitter`, `kSvcVarFloor` in src/hmc_svc.h), so its kriging conditional and
# its level forms 1'Q 1 / 1'Q w are those of that conditioned factorization. On
# a draw whose field variance is near the nugget (sigma2 ~ 1e-3), the exact
# dense-K conditional differs from the density's by more than the Monte Carlo
# error of the check, so the reference runs the same conditioning.
.vcp_svc_nugget <- 1e-4
.vcp_svc_floor  <- 1e-4

.vcp_blend_floor <- function(v) {
  if (v >= .vcp_svc_floor) return(v)
  if (v > 0) 0.99 * .vcp_svc_floor + 0.01 * v else 0.99 * .vcp_svc_floor
}

.vcp_cond <- function(Cnb, c0, v0) {
  a <- solve(Cnb + diag(.vcp_svc_nugget, nrow(Cnb)), c0)
  list(a = a, var = .vcp_blend_floor(v0 - sum(c0 * a)))
}

# 1'Q 1 and 1'Q w of the Vecchia precision, one factor per location in the
# fitted order (`nn_order`, 0-based; `nn_idx` rows hold 1-based positions in
# that order, 0 for none).
.vcp_nngp_level_forms <- function(ss, K, w) {
  ord <- as.integer(ss$nn_order) + 1L
  q <- 0; b <- 0
  for (i in seq_along(ord)) {
    loc <- ord[i]
    pos <- ss$nn_idx[i, ]
    pos <- pos[pos > 0L]
    r1 <- 1; rw <- w[loc]; dv <- K[loc, loc]
    if (length(pos)) {
      nl <- ord[pos]
      cc <- .vcp_cond(K[nl, nl, drop = FALSE], K[nl, loc], K[loc, loc])
      r1 <- 1 - sum(cc$a); rw <- w[loc] - sum(cc$a * w[nl]); dv <- cc$var
    }
    q <- q + r1 * r1 / dv
    b <- b + r1 * rw / dv
  }
  list(q = q, b = b)
}

# The predicted coefficient w(s0) - level, with the level drawn from its exact
# conditional N(-b / q, 1 / q) and w(s0) from the conditioned kriging step on
# the neighbours `nb`.
.vcp_nngp_ref <- function(K, wc, nb, k0, v0, lev) {
  cc <- .vcp_cond(K[nb, nb, drop = FALSE], k0, v0)
  mh <- -lev$b / lev$q
  list(mean = sum(cc$a * (wc[nb] + mh)) - mh,
       var = cc$var + (sum(cc$a) - 1)^2 / lev$q)
}

test_that("an NNGP coefficient is predicted from its kriging conditional", {
  skip_on_cran()
  set.seed(7)
  n <- 12L
  d <- data.frame(lon = stats::runif(n), lat = stats::runif(n),
                  x = stats::rnorm(n))
  d$y <- stats::rpois(n, exp(0.2 + 0.6 * d$x))
  # nn = n - 1: every Vecchia factor conditions on all earlier locations, so
  # the density is the full GP and 1'Q 1 / 1'Q w are the dense ones.
  fit <- .vcp_fit(d, spatial = spatial_svc(~ lon + lat, terms = ~ x - 1,
                                           nn = n - 1L))
  rep_fit <- .vcp_replicate(fit)
  r <- nrow(fit$draws)
  s2 <- exp(fit$draws[r, "log_sigma2_svc[1]"])
  ph <- exp(fit$draws[r, "log_phi_svc[1]"])
  uc <- tulpa:::.coords_plain(fit$spatial$unique_coords)
  wc <- svc(fit)$draws[r, , 1L]
  K <- s2 * exp(-as.matrix(stats::dist(uc)) / ph)

  newc <- tulpa:::.svc_query_coords(fit$spatial,
                                    data.frame(lon = c(0.5, 2.5), lat = c(0.5, 2.5)))
  pred <- tulpa:::.varying_coef_field_draws(rep_fit, svc_coords = newc)$svc
  lev <- .vcp_nngp_level_forms(fit$model_inputs$svc_spec, K, wc)
  for (p in seq_len(nrow(newc))) {
    dist0 <- sqrt(colSums((t(uc) - newc[p, ])^2))
    nb <- order(dist0)[seq_len(n - 1L)]
    ref <- .vcp_nngp_ref(K, wc, nb, s2 * exp(-dist0[nb] / ph), s2, lev)
    .vcp_expect_moments(pred[, p], ref$mean, ref$var, paste("site", p))
  }
  # Far from every location the coefficient reverts to the field's own level,
  # -mbar, with the prior variance plus the level's: the centred field's mean
  # is not where it goes.
  expect_gt(stats::var(pred[, 2L]), stats::var(pred[, 1L]))
  # A fitted location is its own stored draw.
  at_loc <- tulpa:::.varying_coef_field_draws(rep_fit, svc_coords = uc[1:2, ])$svc
  expect_equal(unname(at_loc[1L, ]), unname(wc[1:2]))
})

test_that("rw1 / rw2 / ar1 / gp coefficients are predicted from their conditionals", {
  skip_on_cran()
  set.seed(11)
  T_n <- 10L
  d <- data.frame(year = rep(seq_len(T_n), each = 6L), x = stats::rnorm(6L * T_n))
  d$y <- stats::rpois(nrow(d), exp(0.3 + 0.5 * d$x))
  fit_of <- function(...) .vcp_fit(d, temporal = temporal_tvc("year",
                                    terms = ~ x - 1, ...))
  new_years <- data.frame(year = c(4.5, 12, -1))

  # rw1: Brownian bridge inside, Brownian steps past either end.
  f <- fit_of(structure = "rw1"); r <- nrow(f$draws)
  tau <- exp(f$draws[r, "log_tau_tvc[1]"]); w <- tvc(f)$draws[r, , 1L]
  P <- tulpa:::.varying_coef_field_draws(
    .vcp_replicate(f), tvc_query = tulpa:::.temporal_query(f$temporal, new_years))$tvc
  .vcp_expect_moments(P[, 1L], 0.5 * (w[4] + w[5]), 0.25 / tau, "rw1 bridge")
  .vcp_expect_moments(P[, 2L], w[T_n], 2 / tau, "rw1 forecast")
  .vcp_expect_moments(P[, 3L], w[1], 2 / tau, "rw1 backcast")

  # rw2: the second-difference recursion, two steps ahead.
  f <- fit_of(structure = "rw2"); r <- nrow(f$draws)
  tau <- exp(f$draws[r, "log_tau_tvc[1]"]); w <- tvc(f)$draws[r, , 1L]
  P <- tulpa:::.varying_coef_field_draws(
    .vcp_replicate(f),
    tvc_query = tulpa:::.temporal_query(f$temporal, data.frame(year = 12)))$tvc
  .vcp_expect_moments(P[, 1L], w[T_n] + 2 * (w[T_n] - w[T_n - 1L]), 5 / tau,
                      "rw2 forecast")

  # ar1: proper, so the level is drawn before the recursion runs.
  f <- fit_of(structure = "ar1"); r <- nrow(f$draws)
  tau <- exp(f$draws[r, "log_tau_tvc[1]"])
  rho <- 2 * stats::plogis(f$draws[r, "logit_rho_tvc[1]"]) - 1
  w <- tvc(f)$draws[r, , 1L]
  v <- 1 / (tau * (1 - rho^2))
  K <- v * rho^abs(outer(seq_len(T_n), seq_len(T_n), "-"))
  P <- tulpa:::.varying_coef_field_draws(
    .vcp_replicate(f),
    tvc_query = tulpa:::.temporal_query(f$temporal, data.frame(year = c(13, -1))))$tvc
  for (k in 1:2) {
    pos <- c(13, -1)[k]
    ref <- .vcp_proper_ref(K, w, seq_len(T_n),
                           v * rho^abs(pos - seq_len(T_n)), v)
    .vcp_expect_moments(P[, k], ref$mean, ref$var, paste("ar1", pos))
  }

  # gp: the dense conditional on a Matern 5/2 kernel, and the bracketing
  # conditional on the exponential (Markov) one -- which has to be the dense
  # one too.
  for (kern in c("matern", "exponential")) {
    f <- if (kern == "matern") fit_of(structure = "gp", cov = "matern", nu = 2.5)
         else fit_of(structure = "gp", cov = "exponential")
    r <- nrow(f$draws); tv <- f$temporal
    s2 <- exp(f$draws[r, "log_sigma2_tvc_gp[1]"])
    ph <- tv$phi_prior_lower + (tv$phi_prior_upper - tv$phi_prior_lower) *
      stats::plogis(f$draws[r, "logit_phi_tvc_gp[1]"])
    kfun <- if (kern == "matern") {
      function(h) { u <- sqrt(5) * h / ph; s2 * (1 + u + u^2 / 3) * exp(-u) }
    } else function(h) s2 * exp(-h / ph)
    tt <- tv$time_values
    w <- tvc(f)$draws[r, , 1L]
    K <- kfun(abs(outer(tt, tt, "-"))) + diag(1e-8, T_n)
    q <- tulpa:::.temporal_query(tv, new_years)
    P <- tulpa:::.varying_coef_field_draws(.vcp_replicate(f), tvc_query = q)$tvc
    for (k in seq_len(nrow(new_years))) {
      ref <- .vcp_proper_ref(K, w, seq_len(T_n), kfun(abs(q$pos[k] - tt)),
                             s2 + 1e-8)
      .vcp_expect_moments(P[, k], ref$mean, ref$var,
                          paste("gp", kern, new_years$year[k]))
    }
  }
})

# Held-out truth: a smooth slope surface / path, fitted on part of the sites or
# years, predicted at the rest.
test_that("predict() covers a held-out SVC / TVC truth and beats the average line", {
  skip_on_cran()
  set.seed(21)
  n <- 220L
  lon <- stats::runif(n); lat <- stats::runif(n); x <- stats::rnorm(n)
  b <- 0.6 * sin(3 * lon) + 0.5 * cos(3 * lat); b <- b - mean(b)
  eta_true <- 0.3 + (0.7 + b) * x
  d <- data.frame(lon, lat, x, y = stats::rpois(n, exp(eta_true)))
  hold <- seq_len(40L)
  fit <- suppressWarnings(tulpa(
    y ~ x, data = d[-hold, ], family = "poisson",
    spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 8L),
    mode = "exact", control = list(n_iter = 400L, n_warmup = 300L,
                                   n_chains = 2L, seed = 1L)))
  p  <- predict(fit, newdata = d[hold, ], se.fit = TRUE, level = 0.9)
  p0 <- predict(fit, newdata = d[hold, ], include_field = FALSE)
  cover <- mean(eta_true[hold] >= p$lower & eta_true[hold] <= p$upper)
  rmse  <- sqrt(mean((p$fit - eta_true[hold])^2))
  rmse0 <- sqrt(mean((p0 - eta_true[hold])^2))
  expect_gt(cover, 0.75)
  expect_lt(rmse, 0.7 * rmse0)

  set.seed(22)
  n_t <- 30L; reps <- 8L
  wt <- 0.6 * sin(2 * pi * seq_len(n_t) / n_t); wt <- wt - mean(wt)
  year <- rep(seq_len(n_t), each = reps); x <- stats::rnorm(length(year))
  eta_t <- 0.3 + (0.5 + wt[year]) * x
  dt <- data.frame(year, x, y = stats::rpois(length(year), exp(eta_t)))
  ctl <- list(n_iter = 400L, n_warmup = 300L, n_chains = 2L, seed = 1L)
  check_heldout <- function(structure, hold_y) {
    keep <- !dt$year %in% hold_y
    f <- suppressWarnings(tulpa(
      y ~ x, data = dt[keep, ], family = "poisson",
      temporal = temporal_tvc("year", terms = ~ x - 1, structure = structure),
      mode = "exact", control = ctl))
    nd <- dt[!keep, ]
    p  <- predict(f, newdata = nd, se.fit = TRUE, level = 0.9)
    p0 <- predict(f, newdata = nd, include_field = FALSE)
    expect_gt(mean(eta_t[!keep] >= p$lower & eta_t[!keep] <= p$upper), 0.75,
              label = paste(structure, "coverage"))
    if (identical(structure, "gp")) {
      expect_lt(sqrt(mean((p$fit - eta_t[!keep])^2)),
                sqrt(mean((p0 - eta_t[!keep])^2)), label = "gp rmse")
    }
    f
  }
  # rw1 steps one grid position per fitted level, so it is held out past the
  # end only, where its forecast is the last level carried forward with a
  # widening band (coverage is the claim, not accuracy); gp reads the time
  # values, so it is held out inside as well and has to beat the average line.
  fit_t <- check_heldout("rw1", 27:30)
  check_heldout("gp", c(seq(4L, 24L, by = 4L), 28:30))
  # The forecast widens with the horizon.
  fc <- predict(fit_t, newdata = data.frame(year = 26 + c(1, 5, 15), x = 1),
                se.fit = TRUE)
  expect_true(all(diff(fc$se.fit) > 0))
})
