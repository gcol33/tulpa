
test_that("spatial_rsr() refuses a field its kernels cannot project", {
  # The projection lives in two Polya-Gamma kernels, one carrying an areal
  # neighbour list and one an NNGP field (gcol33/tulpa#848). A spec that is
  # neither used to be re-typed as areal by tulpa(), which then failed on the
  # missing adjacency with "non-numeric matrix extent" -- a message about
  # neither the spec nor the argument (gcol33/tulpa#815). Refused at
  # construction instead, where the argument that caused it is still in hand.
  expect_error(spatial_rsr(spatial_gp(~ lon + lat, approx = "hsgp"),
                           restrict_to = ~ x),
               "areal or NNGP")
  # The message names what to build it on instead.
  expect_error(spatial_rsr(spatial_gp(~ lon + lat, approx = "hsgp"),
                           restrict_to = ~ x),
               "spatial_car")
  # An NNGP field IS projectable, and is taken.
  expect_s3_class(spatial_rsr(spatial_gp(~ lon + lat), restrict_to = ~ x),
                  "tulpa_rsr")
})

test_that("spatial_rsr() still takes every areal field", {
  W <- matrix(0, 4, 4)
  for (i in 1:3) W[i, i + 1] <- W[i + 1, i] <- 1
  rsr <- spatial_rsr(spatial_car(W, level = "obs"), restrict_to = ~ depth)
  expect_s3_class(rsr, "tulpa_rsr")
  expect_true(isTRUE(rsr$rsr))
  expect_true(tolower(rsr$type) %in% .NL_FRONTDOOR_AREAL)
})

skip_on_cran_budget()
# Surface that the constructors used to accept and no backend could fit.
#
# Both cases had the same shape: an argument was documented, taken at
# construction, and then refused far downstream by a message about a backend
# the user never chose -- or, worse, by a message recommending a mode that
# refuses it for a second reason. The refusal now happens where the argument
# that caused it is still in hand (gcol33/tulpa#814, gcol33/tulpa#815).

test_that("temporal_tvc() takes the structures the block carries", {
  skip_on_cran_budget()
  # `"gp"` was documented and accepted here while no mode could fit it
  # (gcol33/tulpa#814); the constructor then took the three the block's density
  # had a branch for, and the GP branch was wired at gcol33/tulpa#847. The
  # declared set and the fittable set are the same set again.
  for (st in .TVC_STRUCTURES) {
    spec <- temporal_tvc("tidx", structure = st)
    expect_s3_class(spec, "tulpa_tvc")
    expect_identical(spec$structure, st)
  }
  expect_setequal(.TVC_STRUCTURES, c("rw1", "rw2", "ar1", "gp"))
})

test_that("the TVC structure predicate is the one the sampler entry asks", {
  skip_on_cran_budget()
  # One predicate, so the front door's wrong-mode message and the sampler spec
  # cannot disagree about what is fittable.
  for (st in .TVC_STRUCTURES) expect_identical(.tvc_structure_or_stop(st), st)
  # The refusal names the whole declared set, which is the fittable set.
  err <- tryCatch(.tvc_structure_or_stop("multiscale"),
                  error = conditionMessage)
  for (st in .TVC_STRUCTURES) expect_match(err, st, fixed = TRUE)
  expect_match(err, "multiscale", fixed = TRUE)
})

skip_on_cran()

test_that("a Laplace-family mode does not recommend a mode that also refuses", {
  skip_on_cran()
  set.seed(3)
  d <- data.frame(tidx = rep(1:20, each = 4), x = stats::rnorm(80))
  d$y <- stats::rpois(80, exp(0.3 + 0.5 * d$x))
  # A structure `exact` DOES take: the recommendation stands.
  expect_error(
    tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
          temporal = temporal_tvc("tidx", structure = "rw1")),
    "Use mode = 'exact'", fixed = TRUE)
})

test_that("an areal RSR fit still runs and still projects", {
  skip_on_cran()
  set.seed(404)
  n_regions <- 12
  W <- matrix(0, n_regions, n_regions)
  for (i in 1:(n_regions - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
  df <- data.frame(region = factor(rep(1:n_regions, each = 5)))
  df$x <- as.integer(df$region) / 4 + stats::rnorm(nrow(df), 0, 0.5)
  df$y <- stats::rbinom(nrow(df), 20, stats::plogis(-0.5 + 0.6 * df$x))
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(region), data = df, family = "binomial",
    n_trials = rep(20L, nrow(df)),
    spatial = spatial_rsr(spatial_car(W, level = "group",
                                      group_var = "region"),
                          restrict_to = ~ x),
    mode = "auto",
    control = list(n_iter = 400L, warmup = 200L, verbose = FALSE)))
  expect_identical(fit$backend, "gibbs")
  expect_lt(abs(coef(fit)[["x"]] - 0.6), 0.4)
})

test_that("a restricted areal field on conditional Laplace is the nested restricted fit at one point, and read at new rows (#1134)", {
  skip_on_cran()
  set.seed(2)
  nr <- 5L; nu <- nr * nr
  W <- matrix(0, nu, nu)
  for (i in seq_len(nu)) for (j in seq_len(nu)) {
    if (abs((i - 1) %/% nr - (j - 1) %/% nr) +
        abs((i - 1) %% nr - (j - 1) %% nr) == 1) W[i, j] <- 1
  }
  n <- 300L
  unit <- sample(nu, n, TRUE)
  x <- stats::rnorm(n)
  z <- stats::rnorm(nu, 0, 0.5)
  d <- data.frame(y = stats::rpois(n, exp(0.2 + 0.4 * x + z[unit] - mean(z))),
                  x = x, unit = unit)
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(unit), data = d, family = "poisson",
    spatial = spatial_rsr(spatial_car(W, level = "group", group_var = "unit"),
                          ~ x)))
  expect_identical(fit$backend, "nested_laplace")
  # Conditional Laplace holds the field precision at tau = 1 and reads the
  # restricted field through the multi-block driver's projector: the nested
  # restricted fit on the one-point grid tau = 1, to solver tolerance.
  X <- cbind(`(Intercept)` = 1, x = d$x)
  cond <- tulpa_laplace(y = d$y, n_trials = rep(1L, n), X = X,
                        family = "poisson", spatial = fit$spatial)
  blk <- tulpa:::.spatial_spec_to_nl_prior(fit$spatial)
  blk$tau_grid <- 1
  one <- suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, n), X = X, family = "poisson", prior = blk,
    control = list(axis_refine = "none", prune = FALSE, auto_recenter = FALSE,
                   diagnose_k = FALSE, diagnose_skew = FALSE)))
  expect_equal(nrow(one$modes), 1L)
  expect_equal(cond$mode, as.numeric(one$modes[1L, ]), tolerance = 1e-8)
  expect_equal(cond$log_marginal, one$log_marginal[1L], tolerance = 1e-8)
  # The field the rows receive is P z: nothing of it lies outside the
  # projector's range. The plain field's fit does not have that property, and
  # its mode differs.
  P <- fit$spatial$rsr_projection
  u <- as.numeric(P %*% utils::tail(cond$mode, nu))
  expect_lt(sqrt(sum((u - as.numeric(P %*% u))^2)), 1e-10)
  sp_plain <- fit$spatial
  sp_plain$rsr <- NULL
  plain <- tulpa_laplace(y = d$y, n_trials = rep(1L, n), X = X,
                         family = "poisson", spatial = sp_plain)
  u_plain <- utils::tail(plain$mode, nu)
  expect_gt(sqrt(sum((u_plain - as.numeric(P %*% u_plain))^2)), 0.1)
  expect_gt(max(abs(cond$mode[1:2] - plain$mode[1:2])), 1e-3)
  # The front door's conditional fit is the same cell, and at new rows it
  # reads X beta + (P z)[unit].
  fc <- suppressWarnings(tulpa(
    y ~ x + spatial(unit), data = d, family = "poisson", mode = "laplace",
    spatial = spatial_rsr(spatial_car(W, level = "group", group_var = "unit"),
                          ~ x)))
  expect_identical(fc$backend, "laplace")
  expect_equal(fc$mode, cond$mode, tolerance = 1e-8)
  nd0 <- d[c(5, 50, 120, 200), ]
  ref0 <- cond$mode[1L] + cond$mode[2L] * nd0$x + u[nd0$unit]
  expect_equal(as.numeric(predict(fc, newdata = nd0)), ref0, tolerance = 1e-8)
  # At new rows the field is P z at the row's unit: against the grid's own
  # posterior mean, sum_k w_k (X beta_k + (P z_k)[unit]).
  nd <- d[c(5, 50, 120, 200), ]
  w <- fit$weights / sum(fit$weights)
  P <- fit$spatial$rsr_projection
  cols <- ncol(fit$modes) - nu + seq_len(nu)
  ref <- colSums(w * t(vapply(seq_along(w), function(k) {
    m <- fit$modes[k, ]
    m[1L] + m[2L] * nd$x + as.numeric(P %*% m[cols])[nd$unit]
  }, numeric(nrow(nd)))))
  expect_equal(as.numeric(predict(fit, newdata = nd)), ref, tolerance = 1e-8)
})
