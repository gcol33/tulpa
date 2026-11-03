# Every field a Polya-Gamma Gibbs chain samples enters its linear predictor
# (`.gibbs_field_eta_contrib()`, R/posterior_predict.R): the in-sample eta the
# criteria score is the one the kernel's likelihood read at each draw, and a
# continuous field is drawn at new locations from its NNGP predictive.

.gfl_data <- function() {
  set.seed(11)
  n <- 100
  lon <- runif(n, 0, 10); lat <- runif(n, 0, 10)
  D <- as.matrix(dist(cbind(lon, lat)))
  w <- as.numeric(t(chol(exp(-D / 3) + diag(1e-8, n))) %*% rnorm(n))
  x <- rnorm(n)
  data.frame(y = rbinom(n, 20, plogis(-0.3 + 0.5 * x + w)), x, lon, lat,
             ntr = 20L)
}

.gfl_waic <- function(eta, y, ntr) {
  ll <- vapply(seq_along(y), function(i)
    stats::dbinom(y[i], ntr, stats::plogis(eta[, i]), log = TRUE),
    numeric(nrow(eta)))
  lppd <- sum(log(colMeans(exp(ll))))
  p <- sum(apply(ll, 2, stats::var))
  c(elpd_waic = lppd - p, p_waic = p)
}

test_that("gp and multiscale Gibbs fields enter eta, WAIC and predictions", {
  skip_on_cran()
  skip_if_not_installed("loo")
  d <- .gfl_data()
  specs <- list(spatial_gp(~ lon + lat, nn = 10, scale_coords = FALSE),
                suppressMessages(spatial_multiscale(~ lon + lat,
                                                    scale_coords = FALSE)))
  for (sp in specs) {
    fit <- suppressWarnings(tulpa(
      y ~ x, data = d, family = "binomial", n_trials = d$ntr, spatial = sp,
      mode = "gibbs", control = list(n_iter = 400L, warmup = 200L, seed = 1L)))
    dr <- as.matrix(fit$draws)
    fc <- grep("^(gp_w|gp_local|gp_regional)\\[", colnames(dr))
    stems <- split(fc, sub("\\[.*", "", colnames(dr)[fc]))
    fld <- Reduce(`+`, lapply(stems, function(cc) dr[, cc, drop = FALSE]))
    eta <- dr[, "(Intercept)"] + outer(dr[, "x"], d$x) + fld
    expect_equal(unname(posterior_linpred(fit)), unname(eta), tolerance = 1e-12)

    wt <- suppressWarnings(loo::waic(fit))
    ref <- .gfl_waic(eta, d$y, 20)
    expect_equal(unname(wt$estimates["elpd_waic", "Estimate"]),
                 unname(ref["elpd_waic"]), tolerance = 1e-8)
    expect_equal(unname(wt$estimates["p_waic", "Estimate"]),
                 unname(ref["p_waic"]), tolerance = 1e-8)

    nd <- data.frame(x = 0, lon = c(2, 5, 8), lat = c(3, 5, 7))
    p  <- predict(fit, newdata = nd, se.fit = TRUE)
    pf <- predict(fit, newdata = nd, include_field = FALSE)
    expect_gt(max(abs(p$fit - pf)), 0.1)
    lp <- posterior_linpred(fit, newdata = nd, summary = TRUE, seed = 4)
    expect_lt(max(abs(lp$estimate - p$fit) / p$se.fit), 0.3)
  }
})

test_that("a restricted Gibbs field at its fitted locations is the projection eta read", {
  skip_on_cran()
  d <- .gfl_data()
  fit <- suppressWarnings(tulpa(
    y ~ x, data = d, family = "binomial", n_trials = d$ntr,
    spatial = spatial_rsr(spatial_gp(~ lon + lat, nn = 10, scale_coords = FALSE),
                          restrict_to = ~ x),
    mode = "gibbs", control = list(n_iter = 200L, warmup = 100L, seed = 1L)))
  dr <- as.matrix(fit$draws)
  eta <- dr[, "(Intercept)"] + outer(dr[, "x"], d$x) +
    dr[, grep("^gp_w\\[", colnames(dr))]
  expect_equal(unname(posterior_linpred(fit)), unname(eta), tolerance = 1e-12)
  expect_equal(unname(predict(fit, newdata = d[1:3, ])),
               unname(colMeans(eta)[1:3]), tolerance = 1e-10)
})

test_that("a temporal Gibbs chain adds its trend at each row's time", {
  skip_on_cran()
  set.seed(2)
  tt <- rep(1:30, each = 6)
  xt <- rnorm(length(tt))
  yt <- rbinom(length(tt), 10, plogis(0.2 + 0.4 * xt + sin(2 * pi * tt / 30)))
  dt <- data.frame(y = yt, x = xt, time = tt)
  tm <- tulpa:::validate_temporal_multiscale(
    temporal_multiscale("time", trend = "rw1", short_term = "iid"), dt)
  fit <- tulpa_gibbs(dt$y, rep(10L, nrow(dt)), cbind("(Intercept)" = 1, x = dt$x),
                     rep(1L, nrow(dt)), 0L, temporal = tm,
                     control = list(n_iter = 300L, warmup = 150L, seed = 1L))
  dr <- as.matrix(fit$draws)
  eta <- dr[, "(Intercept)"] + outer(dr[, "x"], dt$x) +
    dr[, grep("^trend\\[", colnames(dr))][, tt] +
    dr[, grep("^short_term\\[", colnames(dr))][, tt]
  expect_equal(unname(posterior_linpred(fit)), unname(eta), tolerance = 1e-12)
})
