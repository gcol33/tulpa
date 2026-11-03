# A Gaussian residual variance the caller left out is integrated, not
# conditioned at 1 (gcol33/tulpa#1068). On the modes that integrate what the
# caller did not state (`auto`, `structured`, the nested-Laplace modes) the fit
# runs on the one-arm joint nested-Laplace door with the variance an outer-grid
# axis, placed at its posterior like a field SD; a supplied `phi` conditions as
# before, and the conditional `mode = "laplace"` still conditions and says so.

.pi_data <- function(s_eps, seed = 11L, ng = 30L, m = 10L) {
  set.seed(seed)
  g <- rep(seq_len(ng), each = m)
  x <- rnorm(ng * m)
  data.frame(x = x, g = factor(g),
             y = 1 + 0.5 * x + rnorm(ng, 0, 0.5)[g] + rnorm(ng * m, 0, s_eps))
}

.pi_warnings <- function(expr) {
  ws <- character(0)
  val <- withCallingHandlers(suppressMessages(expr), warning = function(w) {
    ws <<- c(ws, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = val, warnings = ws)
}

test_that("an unsupplied variance is an axis only where the mode integrates", {
  expect_true(tulpa:::.phi_axis_wanted("gaussian", FALSE, FALSE))
  expect_false(tulpa:::.phi_axis_wanted("gaussian", TRUE, FALSE))
  expect_false(tulpa:::.phi_axis_wanted("gaussian", FALSE, TRUE))
  expect_false(tulpa:::.phi_axis_wanted("poisson", FALSE, FALSE))
  expect_true(tulpa:::.phi_axis_wanted("interval_gaussian", FALSE, FALSE))
  expect_true(tulpa:::.phi_is_variance("truncated_gaussian"))
  ax <- tulpa:::.phi_axis_default("gaussian", c(1, 2, 4, 3, 7), cbind(1, 1:5))
  expect_true(is_auto_grid(ax))
  expect_true(all(ax > 0))
})

skip_on_cran()

test_that("auto integrates the residual variance at small and large noise", {
  skip_on_cran()
  for (s_eps in c(0.3, 3)) {
    d <- .pi_data(s_eps)
    out <- .pi_warnings(tulpa(y ~ x + (1 | g), d))
    f <- out$value
    expect_false(any(grepl("`phi` not supplied", out$warnings, fixed = TRUE)))
    expect_identical(f$backend, "nested_laplace_joint")
    expect_true(f$phi_estimated)
    lo <- sqrt(f$theta_ci_lo[["phi_y"]]); hi <- sqrt(f$theta_ci_hi[["phi_y"]])
    expect_true(lo < s_eps && s_eps < hi, info = paste("s_eps", s_eps))
    ci <- confint(f)
    expect_true(ci["x", 1] < 0.5 && 0.5 < ci["x", 2], info = paste("s_eps", s_eps))
    # The flat dispersion is the integrated one, on the variance scale.
    expect_equal(f$phi, f$theta_mean[["phi_y"]])
  }
})

test_that("a fixed-effect gaussian model integrates it too", {
  skip_on_cran()
  d <- .pi_data(0.3)
  d$y <- 1 + 0.5 * d$x + rnorm(nrow(d), 0, 0.3)
  f <- suppressMessages(tulpa(y ~ x, d))
  expect_identical(f$backend, "nested_laplace_joint")
  expect_equal(sqrt(f$phi), 0.3, tolerance = 0.15)
  vc <- VarCorr(f)
  expect_identical(vc$source, "estimated")
})

test_that("a supplied phi conditions, and mode = 'laplace' still conditions", {
  skip_on_cran()
  d <- .pi_data(0.3)
  f <- suppressWarnings(suppressMessages(
    tulpa(y ~ x + (1 | g), d, phi = 0.09)))
  expect_false(identical(f$backend, "nested_laplace_joint"))
  expect_false(isTRUE(f$phi_estimated))
  expect_equal(f$phi, 0.09)

  out <- .pi_warnings(tulpa(y ~ x, d, mode = "laplace"))
  expect_true(any(grepl("`phi` not supplied", out$warnings, fixed = TRUE)))
  expect_equal(out$value$phi, 1)
})

test_that("group effects and covariances are read on the effect scale", {
  skip_on_cran()
  set.seed(3)
  ng <- 40L; m <- 10L; g <- rep(seq_len(ng), each = m)
  x <- rnorm(ng * m)
  b0 <- rnorm(ng, 0, 0.6); b1 <- 0.5 * b0 + rnorm(ng, 0, 0.3)
  d <- data.frame(x = x, g = factor(g),
                  y = 1 + 0.5 * x + b0[g] + b1[g] * x + rnorm(ng * m, 0, 0.4))
  for (fm in list(y ~ x + (1 + x | g), y ~ x + (1 + x || g))) {
    f <- suppressMessages(tulpa(fm, d))
    expect_identical(f$backend, "nested_laplace_joint")
    re <- ranef(f)
    df <- if (is.data.frame(re)) re else re[[1L]]
    int <- startsWith(df$term, "g[")
    slo <- startsWith(df$term, "g.x[")
    expect_gt(stats::cor(df$estimate[int], b0), 0.9)
    expect_gt(stats::cor(df$estimate[slo], b1), 0.9)
    # Shrunk toward zero, never inflated past the truth's spread.
    expect_lt(stats::sd(df$estimate[int]), 1.2 * stats::sd(b0))
    vc <- VarCorr(f)
    expect_equal(vc$sd[vc$term == "Residual"], 0.4, tolerance = 0.15)
    expect_equal(vc$sd[vc$coef == "(Intercept)" & vc$term == "g"], 0.6,
                 tolerance = 0.3)
  }
})

test_that("coverage of beta and sigma at noise SD 0.3 and 3 holds over seeds", {
  skip_if_not_slow()
  for (s_eps in c(0.3, 3)) {
    cov <- t(vapply(seq_len(20L), function(s) {
      d <- .pi_data(s_eps, seed = 2000L + s)
      f <- suppressMessages(tulpa(y ~ x + (1 | g), d))
      ci <- confint(f)
      c(b1 = ci["x", 1] <= 0.5 && 0.5 <= ci["x", 2],
        s = sqrt(f$theta_ci_lo[["phi_y"]]) <= s_eps &&
          s_eps <= sqrt(f$theta_ci_hi[["phi_y"]]))
    }, logical(2)))
    expect_gte(mean(cov[, "b1"]), 0.85, label = paste("beta at", s_eps))
    expect_gte(mean(cov[, "s"]), 0.85, label = paste("sigma at", s_eps))
  }
})

# gcol33/tulpa#1122: a gaussian HSGP fit with phi left out routes to the joint
# door with phi as an axis. The noise variance and the field's variance and
# lengthscale each held their weight on one or two declared nodes; read on
# that declared grid they fire the placement pass, which lays all three at
# the outer mode, so the consistency pass does not bisect them level by level
# (885 cells before, 225 of them declared).
.hsgp_phi_fixture <- function() {
  set.seed(11)
  n <- 200
  d <- data.frame(lon = stats::runif(n), lat = stats::runif(n), x = stats::rnorm(n))
  D <- as.matrix(stats::dist(d[, c("lon", "lat")]))
  w <- as.numeric(t(chol(0.5 * exp(-D / 0.3) + diag(1e-8, n))) %*% stats::rnorm(n))
  d$y <- 1 + 0.5 * d$x + w + stats::rnorm(n, 0, 0.4)
  d
}

test_that("a collapsed noise axis and field axes are placed, not bisected", {
  skip_on_cran()
  d <- .hsgp_phi_fixture()
  sp <- spatial_gp(~ lon + lat, approx = "hsgp", m = 6)
  fit <- suppressWarnings(tulpa(y ~ x, data = d, family = "gaussian",
                                spatial = sp, mode = "auto",
                                control = list(diagnose_k = FALSE)))
  expect_identical(fit$outer_grid_placement, "auto_recentered")
  expect_setequal(fit$outer_grid_recenter_axes,
                  c("b1.sigma2", "b1.lengthscale", "phi_y"))
  expect_lt(nrow(fit$theta_grid), 450L)
  expect_false(isTRUE(fit$outer_refine_deferred))
})

test_that("the placed HSGP noise and field axes read a dense grid's interval ends", {
  skip_if_not_slow()
  d <- .hsgp_phi_fixture()
  sp <- spatial_gp(~ lon + lat, approx = "hsgp", m = 6)
  calls <- list()
  trace("tulpa_nested_laplace_joint", where = asNamespace("tulpa"), print = FALSE,
        tracer = quote(calls[[length(calls) + 1L]] <<- mget(
          c("responses", "prior", "copy", "phi_grid", "prior_sigma",
            "hyperprior", "control"))))
  fit <- suppressWarnings(tulpa(y ~ x, data = d, family = "gaussian",
                                spatial = sp, mode = "auto",
                                control = list(diagnose_k = FALSE)))
  untrace("tulpa_nested_laplace_joint", where = asNamespace("tulpa"))
  a <- calls[[length(calls)]]
  ls <- function(lo, hi, k) exp(seq(log(lo), log(hi), length.out = k))
  eg <- expand.grid(s = ls(0.05, 30, 32), l = ls(0.15, 10, 32))
  pr <- a$prior
  pr[[1]]$sigma2_grid <- eg$s
  pr[[1]]$lengthscale_grid <- eg$l
  dense <- suppressWarnings(tulpa_nested_laplace_joint(
    a$responses, pr, copy = a$copy, phi_grid = list(y = ls(0.13, 0.42, 20)),
    prior_sigma = a$prior_sigma, hyperprior = a$hyperprior,
    control = utils::modifyList(a$control, list(
      axis_refine = "none", var_of_means_consistency = FALSE,
      auto_recenter = FALSE, prune = FALSE, store_Q = FALSE))))
  tg <- as.matrix(dense$theta_grid)
  w <- exp(dense$log_marginal + dense$log_quad - max(dense$log_marginal + dense$log_quad))
  w <- w / sum(w)
  got <- summary(fit)$hyperparameters
  for (j in seq_len(ncol(tg))) {
    v <- log(tg[, j]); m <- tapply(w, v, sum); u <- as.numeric(names(m))
    h <- u[2] - u[1]
    q <- stats::approx(c(0, cumsum(m)), c(u - h / 2, max(u) + h / 2),
                       xout = c(0.025, 0.975), ties = "ordered")$y
    expect_lt(max(m[1], m[length(m)]), 1e-3)
    # summary() reports a spatial length in the coordinates' own units; the
    # grid carries it on the standardized coordinates the kernel reads.
    q <- q + log(tulpa:::.hyper_axis_data_scale(fit, colnames(tg)[j]))
    ref <- log(unlist(got[got$parameter == colnames(tg)[j], c("ci_lo", "ci_hi")]))
    expect_lt(max(abs(ref - q)), 0.06)
  }
})
