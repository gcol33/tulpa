# One draw path at new rows for every backend and every latent field
# (R/spatial_field_predict.R): a grid's hyperparameter cells, a conditional
# Laplace fit's one cell and a sampler's stored draws are mapped by the same
# functions, and the map at the training rows is the linear predictor each
# engine assembled itself.

.fpb_lattice <- function(side = 4L) {
  K <- side^2
  W <- matrix(0, K, K)
  for (i in 1:K) for (j in 1:K) {
    ri <- (i - 1) %/% side; ci <- (i - 1) %% side
    rj <- (j - 1) %/% side; cj <- (j - 1) %% side
    if (abs(ri - rj) + abs(ci - cj) == 1) W[i, j] <- 1
  }
  W
}

.fpb_areal <- function(seed = 5, n = 200L, side = 4L, n_t = 10L) {
  set.seed(seed)
  W <- .fpb_lattice(side)
  K <- nrow(W)
  a <- data.frame(region = sample(1:K, n, TRUE), t = sample(1:n_t, n, TRUE),
                  x = rnorm(n), g = factor(sample(1:4, n, TRUE)))
  u <- rnorm(K) * 0.5
  tr <- cumsum(rnorm(n_t, 0, 0.3))
  a$y <- rpois(n, exp(0.2 + 0.5 * a$x + u[a$region] + tr[a$t]))
  a$yb <- rbinom(n, 5, plogis(-0.3 + 0.5 * a$x + u[a$region]))
  list(W = W, d = a)
}

.fpb_cont <- function(seed = 31, n = 100L) {
  set.seed(seed)
  lon <- runif(n); lat <- runif(n)
  K <- exp(-as.matrix(dist(cbind(lon, lat))) / 0.25) * 0.6
  w <- as.numeric(t(chol(K + diag(1e-8, n))) %*% rnorm(n))
  x <- rnorm(n)
  data.frame(lon, lat, x, g = factor(sample(1:5, n, TRUE)), w,
             y = rpois(n, exp(0.4 + 0.5 * x + w)),
             yb = rbinom(n, 5, plogis(0.5 * x + w)))
}

# The map at the training rows, cell by cell (a stored draw is a cell).
.fpb_map_train <- function(fit) {
  src <- tulpa:::.field_source(fit)
  expect_false(is.null(src))
  k <- which(src$weights > 0)
  tulpa:::.field_eta_rows(fit, src, tulpa:::.tulpa_designs(fit, NULL, "test"),
                          NULL, src$modes[k, , drop = FALSE], k, draw = FALSE,
                          des = fit$re_design,
                          sel = tulpa:::.re_form_select(fit, NULL))
}

test_that("a sampler fit's fields map to the eta its engine assembled, draw by draw", {
  skip_on_cran()
  ar <- .fpb_areal()
  cd <- .fpb_cont()
  ctl <- list(n_chains = 1, n_iter = 200, warmup = 100, seed = 1)
  fits <- suppressWarnings(list(
    nngp_re = tulpa(y ~ x + (1 | g), data = cd, family = "poisson",
                    spatial = spatial_gp(~ lon + lat, nn = 8), mode = "hmc",
                    control = ctl),
    hsgp = tulpa(y ~ x, data = cd, family = "poisson",
                 spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 5),
                 mode = "hmc", control = ctl),
    icar_rw1_re = tulpa(y ~ x + spatial(region) + (1 | g), data = ar$d,
                        family = "poisson",
                        spatial = spatial_car(ar$W, group_var = "region"),
                        temporal = temporal_rw1("t"), mode = "hmc",
                        control = ctl),
    bym2 = tulpa(y ~ x + spatial(region), data = ar$d, family = "poisson",
                 spatial = spatial_bym2(ar$W, group_var = "region"),
                 mode = "hmc", control = ctl),
    ar1 = tulpa(y ~ x, data = ar$d, family = "poisson",
                temporal = temporal_ar1("t"), mode = "hmc", control = ctl)))
  for (nm in names(fits)) {
    fit <- suppressWarnings(fits[[nm]])
    E <- .fpb_map_train(fit)
    ref <- tulpa:::.tulpa_eta_draws_sampler(fit)
    expect_lt(max(abs(E - ref)), 1e-10, label = nm)
    # predict() at the training rows is the mean of those draws, read as
    # newdata or in sample.
    expect_lt(max(abs(predict(fit) - colMeans(ref))), 1e-10, label = nm)
    nd <- if (nm %in% c("nngp_re", "hsgp")) cd else ar$d
    expect_lt(max(abs(predict(fit, newdata = nd) - predict(fit))), 1e-10,
              label = nm)
  }
})

test_that("a sampler's multi-scale temporal field reaches every row at its time", {
  skip_on_cran()
  ar <- .fpb_areal(n_t = 12L)
  ctl <- list(n_chains = 1, n_iter = 200, warmup = 100, seed = 1)
  for (par in c("noncentered", "centered")) {
    fit <- suppressWarnings(tulpa(
      y ~ x, data = ar$d, family = "poisson",
      temporal = temporal_multiscale("t", trend = "rw1", seasonal = 4,
                                     short_term = "iid",
                                     parameterization = par),
      mode = "hmc", control = ctl))
    E <- .fpb_map_train(fit)
    ref <- tulpa:::.tulpa_eta_draws_sampler(fit)
    expect_lt(max(abs(E - ref)), 1e-10, label = par)
    expect_lt(max(abs(predict(fit) - colMeans(ref))), 1e-10, label = par)
    # A new row at a fitted time carries the field at that time: two rows at
    # the same x differ by the difference of the field's posterior means.
    nd <- data.frame(x = 0, t = sort(unique(ar$d$t)))
    p <- predict(fit, newdata = nd)
    i <- match(nd$t, ar$d$t)
    fe <- colMeans(ref)[i] - coef(fit)[["x"]] * ar$d$x[i]
    expect_lt(max(abs((p - p[1]) - (fe - fe[1]))), 1e-8, label = par)
    expect_gt(sd(p), 0.05, label = par)
  }
})

test_that("a grid or one-cell fit's areal and temporal fields map to its own eta", {
  skip_on_cran()
  ar <- .fpb_areal()
  sp <- function(f) f(ar$W, group_var = "region")
  fits <- suppressWarnings(list(
    nl_icar_re = tulpa(y ~ x + spatial(region) + (1 | g), data = ar$d,
                       family = "poisson", spatial = sp(spatial_car)),
    nl_bym2 = tulpa(y ~ x + spatial(region), data = ar$d, family = "poisson",
                    spatial = sp(spatial_bym2)),
    nl_car_proper = tulpa(y ~ x + spatial(region), data = ar$d,
                          family = "poisson", spatial = sp(spatial_car_proper)),
    nl_rw1_grouped = tulpa(y ~ x, data = ar$d, family = "poisson",
                           temporal = temporal_rw1("t", group_var = "g")),
    nl_icar_ar1 = tulpa(y ~ x + spatial(region), data = ar$d,
                        family = "poisson", spatial = sp(spatial_car),
                        temporal = temporal_ar1("t"))))
  for (nm in names(fits)) {
    fit <- suppressWarnings(fits[[nm]])
    k <- which(fit$weights > 0)
    expect_lt(max(abs(.fpb_map_train(fit) - fit$fitted_eta[k, ])), 1e-8,
              label = nm)
    expect_lt(max(abs(predict(fit, newdata = ar$d) - predict(fit))), 1e-10,
              label = nm)
    lp <- posterior_linpred(fit, newdata = ar$d[1:20, ], summary = TRUE,
                            ndraws = 2000, seed = 1)
    p <- predict(fit, newdata = ar$d[1:20, ], se.fit = TRUE)
    expect_lt(max(abs(lp$estimate - p$fit) / p$se.fit), 0.15, label = nm)
  }
})

test_that("a conditional-Laplace fit is one cell, solved again by its registry kernel", {
  skip_on_cran()
  ar <- .fpb_areal()
  cd <- .fpb_cont()
  fits <- suppressWarnings(list(
    nngp_re = tulpa(y ~ x + (1 | g), data = cd, family = "poisson",
                    spatial = spatial_gp(~ lon + lat, nn = 8), mode = "laplace"),
    hsgp = tulpa(y ~ x, data = cd, family = "poisson",
                 spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 5),
                 mode = "laplace"),
    icar = tulpa(y ~ x + spatial(region), data = ar$d, family = "poisson",
                 spatial = spatial_car(ar$W, group_var = "region"),
                 mode = "laplace"),
    bym2 = tulpa(y ~ x + spatial(region), data = ar$d, family = "poisson",
                 spatial = spatial_bym2(ar$W, group_var = "region"),
                 mode = "laplace")))
  for (nm in names(fits)) {
    fit <- suppressWarnings(fits[[nm]])
    src <- tulpa:::.field_source(fit)
    expect_equal(nrow(src$modes), 1L, label = nm)
    sol <- tulpa:::.field_cell_solve(fit, src, 1L)
    expect_lt(max(abs(sol$modes - src$modes)), 1e-5, label = nm)
    # The map at the mode is the conditional kernel's own eta: the fixed part
    # plus the field (and group effects) at the mode.
    p <- ncol(fit$model_matrix)
    lp <- tulpa:::.tulpa_point_linpred(fit, NULL, "test")$eta
    E <- .fpb_map_train(fit)
    expect_gt(max(abs(E - lp)), 0.1, label = nm)
    nd <- if (nm %in% c("nngp_re", "hsgp")) cd else ar$d
    pr <- predict(fit, newdata = nd[1:20, ], se.fit = TRUE)
    pf <- predict(fit, newdata = nd[1:20, ], include_field = FALSE)
    expect_gt(max(abs(pr$fit - pf)), 0.1, label = nm)
    expect_true(all(pr$se.fit > 0), label = nm)
  }
  # The areal kernel's eta at the mode: X beta + the field gathered by unit.
  fit <- suppressWarnings(fits$icar)
  lf <- tulpa:::.laplace_areal_field(fit, nrow(ar$d))
  ref <- as.numeric(fit$model_matrix %*% fit$mode[1:2]) +
    as.numeric(lf$design$Z_field %*% lf$u)
  expect_lt(max(abs(predict(fit) - ref)), 1e-10)
  # The NNGP kernel's: X beta + the group effect + w at the row's location.
  fit <- suppressWarnings(fits$nngp_re)
  m <- fit$mode
  ref <- as.numeric(fit$model_matrix %*% m[1:2]) + m[2 + as.integer(cd$g)] +
    m[7 + fit$spatial$obs_to_loc]
  expect_lt(max(abs(predict(fit) - ref)), 1e-10)
})

test_that("a new row's area or time is matched to the fitted ones; an unseen one is refused", {
  skip_on_cran()
  ar <- .fpb_areal()
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(region), data = ar$d, family = "poisson",
    spatial = spatial_car(ar$W, group_var = "region"),
    temporal = temporal_rw1("t")))
  # Two rows in the same area at the same time differ only by x.
  nd <- data.frame(region = c(3, 3, 7), t = c(2, 2, 2), x = c(0, 1, 0))
  p <- predict(fit, newdata = nd)
  b <- coef(fit)
  expect_equal(p[2] - p[1], unname(b["x"]), tolerance = 1e-10)
  fe <- field_effects(fit)
  expect_equal(p[3] - p[1], fe$estimate[7] - fe$estimate[3], tolerance = 1e-6)
  expect_error(predict(fit, newdata = data.frame(region = 3, t = 99, x = 0)),
               "time")
  expect_error(predict(fit, newdata = data.frame(region = 99, t = 2, x = 0)),
               "region")
  expect_error(predict(fit, newdata = data.frame(t = 2, x = 0)), "region")
  # Without the field the unit is not needed.
  expect_silent(predict(fit, newdata = data.frame(x = 0), include_field = FALSE))
})

test_that("a restricted continuous field is projected at every row", {
  skip_on_cran()
  cd <- .fpb_cont()
  fit <- suppressWarnings(tulpa(
    y ~ x, data = cd, family = "poisson",
    spatial = spatial_rsr(spatial_gp(~ lon + lat, nn = 8), ~ x)))
  k <- which(fit$weights > 0)
  expect_lt(max(abs(.fpb_map_train(fit) - fit$fitted_eta[k, ])), 1e-8)
  expect_lt(max(abs(predict(fit, newdata = cd) - predict(fit))), 1e-10)
  # At a fitted location the field's restricted value does not move with the
  # row's covariate; at a new one the restriction design is the row's own.
  src <- tulpa:::.field_source(fit)
  comp <- src$layout$fields[[1L]]
  Fr <- src$modes[k[1], comp$cols[[1L]], drop = FALSE]
  Xu <- comp$restrict$design
  delta <- qr.coef(qr(Xu), t(Fr))
  nd <- data.frame(lon = c(cd$lon[1], 0.5), lat = c(cd$lat[1], 0.5), x = c(3, 3))
  sh <- tulpa:::.field_restriction_shift(fit, comp, nd, Fr)
  expect_equal(as.numeric(sh), c(Xu[fit$spatial$obs_to_loc[1], ] %*% delta,
                                 c(1, 3) %*% delta), tolerance = 1e-10)
  p <- predict(fit, newdata = cd[1:10, ], se.fit = TRUE)
  pf <- predict(fit, newdata = cd[1:10, ], include_field = FALSE)
  expect_gt(max(abs(p$fit - pf)), 0.2)
})

test_that("a restricted Gibbs field is projected at every row from its raw draws", {
  skip_on_cran()
  cd <- .fpb_cont()
  fit <- suppressWarnings(tulpa(
    yb ~ x, data = cd, family = "binomial", n_trials = rep(5L, nrow(cd)),
    spatial = spatial_rsr(spatial_gp(~ lon + lat, nn = 8), ~ x),
    mode = "gibbs", control = list(n_iter = 300L, warmup = 150L, seed = 1L)))
  dr <- fit$draws
  # The map of the raw field reproduces the projection the kernel stored.
  E <- .fpb_map_train(fit)
  eta <- dr[, "(Intercept)"] + outer(dr[, "x"], cd$x) +
    dr[, grep("^gp_w\\[", colnames(dr))][, fit$spatial$obs_to_loc]
  expect_lt(max(abs(E - unname(eta))), 1e-10)
  nd <- data.frame(lon = c(0.2, 0.8), lat = c(0.3, 0.6), x = 0)
  p <- predict(fit, newdata = nd, se.fit = TRUE)
  pf <- predict(fit, newdata = nd, include_field = FALSE)
  expect_true(all(is.finite(p$fit)) && all(p$se.fit > 0))
  expect_gt(max(abs(p$fit - pf)), 0.05)
})

test_that("held-out 95% intervals cover the simulated linear predictor on every backend", {
  skip_if_not_slow()
  # Ten seeds of 150 fitted and 50 held-out locations, field variance 0.5 and
  # range 0.3 on the unit square; measured coverage 0.928 (hmc), 0.992
  # (conditional Laplace at its fixed hyperparameters), 0.958 (restricted
  # field on nested Laplace).
  hits <- list()
  for (s in 1:10) {
    set.seed(s)
    n <- 200L
    lon <- runif(n); lat <- runif(n)
    K <- exp(-as.matrix(dist(cbind(lon, lat))) / 0.3) * 0.5
    w <- as.numeric(t(chol(K + diag(1e-8, n))) %*% rnorm(n))
    x <- rnorm(n)
    eta <- 0.3 + 0.5 * x + w
    d <- data.frame(lon, lat, x, y = rpois(n, exp(eta)))
    obs <- d[1:150, ]; out <- d[151:200, ]; truth <- eta[151:200]
    fits <- suppressWarnings(list(
      nngp_hmc = tulpa(y ~ x, data = obs, family = "poisson",
                       spatial = spatial_gp(~ lon + lat, nn = 10), mode = "hmc",
                       control = list(n_chains = 2, n_iter = 1000, warmup = 500,
                                      seed = 1)),
      nngp_laplace = tulpa(y ~ x, data = obs, family = "poisson",
                           spatial = spatial_gp(~ lon + lat, nn = 10),
                           mode = "laplace"),
      gp_rsr = tulpa(y ~ x, data = obs, family = "poisson",
                     spatial = spatial_rsr(spatial_gp(~ lon + lat, nn = 10), ~ x))))
    for (nm in names(fits)) {
      p <- predict(fits[[nm]], newdata = out, se.fit = TRUE)
      hits[[nm]] <- c(hits[[nm]], p$lower <= truth & truth <= p$upper)
    }
  }
  for (nm in names(hits)) expect_gt(mean(hits[[nm]]), 0.85, label = nm)
})

test_that("inline spatial() / temporal() fields reach newdata, MCAR included", {
  skip_on_cran()
  set.seed(7)
  W <- .fpb_lattice(4L)
  K <- nrow(W)
  d <- data.frame(cell = sample(1:K, 300, TRUE), time = rnorm(300),
                  t = sample(1:8, 300, TRUE), g = factor(sample(1:4, 300, TRUE)))
  u <- rnorm(K) * 0.6; v <- rnorm(K) * 0.3; tr <- cumsum(rnorm(8, 0, 0.3))
  d$y <- rpois(300, exp(0.2 + 0.3 * d$time + u[d$cell] + v[d$cell] * d$time +
                          tr[d$t]))
  forms <- list(
    icar = y ~ time + spatial(graph = W, formula = ~ 1 + time || cell),
    mcar = y ~ time + spatial(graph = W, formula = ~ 1 + time | cell),
    rw1 = y ~ time + temporal(formula = ~ 1 || t, structure = "rw1") + (1 | g))
  for (nm in names(forms)) {
    fit <- suppressWarnings(tulpa(forms[[nm]], data = d, family = "poisson"))
    k <- which(fit$weights > 0)
    expect_lt(max(abs(.fpb_map_train(fit) - fit$fitted_eta[k, ])), 1e-8,
              label = nm)
    expect_lt(max(abs(predict(fit, newdata = d) - predict(fit))), 1e-10,
              label = nm)
    p <- predict(fit, newdata = d[1:30, ], se.fit = TRUE)
    pl <- posterior_linpred(fit, summary = TRUE)[1:30, ]
    # The draws at newdata are the posterior the grid mixture reports at the
    # same rows.
    expect_lt(max(abs(p$fit - pl$estimate) / pl$sd), 0.15, label = nm)
    expect_lt(max(abs(p$se.fit / pl$sd - 1)), 0.15, label = nm)
  }
})
