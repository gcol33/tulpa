# Restricted spatial regression on a CONTINUOUS field (gcol33/tulpa#848).
#
# RSR exists for spatially smooth covariates -- a climate surface, elevation --
# and those are where a continuous field is the natural prior and an areal one a
# discretisation of convenience, so the mitigation was available only on the
# field shape that needs it least (`spatial_rsr()` refused a continuous spec at
# gcol33/tulpa#815).
#
# Three claims:
#   1. THE PROJECTOR -- one builder over the observation -> field map, so the
#      areal and continuous routes differ in the map and in nothing else.
#   2. THE FIT -- the field a restricted continuous fit reports is exactly
#      orthogonal to the restricted design, where an unrestricted one is not.
#   3. THE ESTIMAND -- what the restriction buys is the MARGINAL association,
#      not a less biased version of the conditional one.
#
# The prior precision the projected field's dense conditional is assembled from
# is checked where the sparse one is, in `test-pg-nngp-conditional.R`: both are
# read against the same Lambda = (I - A)' D^-1 (I - A).

.rsrc_sim <- function(seed, n = 150L, beta = 1.0, gamma = 0.8, rho = 0.9,
                      range = 0.25) {
  set.seed(seed)
  lon <- stats::runif(n); lat <- stats::runif(n)
  K <- exp(-as.matrix(stats::dist(cbind(lon, lat))) / range) + diag(1e-6, n)
  s <- as.numeric(scale(as.numeric(t(chol(K)) %*% stats::rnorm(n))))
  x <- as.numeric(scale(rho * s + sqrt(1 - rho^2) * stats::rnorm(n)))
  d <- data.frame(lon = lon, lat = lat, x = x,
                  y = stats::rbinom(n, 25L, stats::plogis(-0.2 + beta * x +
                                                            gamma * s)))
  # The two estimands the arms below are scored against: the slope CONDITIONAL
  # on the field, and the MARGINAL association, which differ by exactly the
  # covariate's projection onto the field.
  list(d = d, n = n, beta_c = beta, beta_m = beta + gamma * stats::cor(x, s))
}

.rsrc_fit <- function(sim, spatial, n_iter = 1200L) {
  suppressWarnings(tulpa(
    y ~ x, data = sim$d, family = "binomial", n_trials = rep(25L, sim$n),
    spatial = spatial, mode = "gibbs",
    control = list(n_iter = n_iter, warmup = as.integer(n_iter / 2),
                   verbose = FALSE)))
}

# max |w' X| relative to the field's own scale, over the retained draws.
.rsrc_orthogonality <- function(fit, data) {
  W <- fit$draws[, grep("^gp_w\\[", colnames(fit$draws)), drop = FALSE]
  Xr <- stats::model.matrix(~ x, data = data)
  max(abs(W %*% Xr)) / max(abs(W))
}


test_that("spatial_rsr() takes the fields a kernel carries a precision for", {
  expect_s3_class(spatial_rsr(spatial_gp(~ lon + lat), restrict_to = ~ x),
                  "tulpa_rsr")
  W <- matrix(0, 4, 4)
  for (i in 1:3) W[i, i + 1] <- W[i + 1, i] <- 1
  expect_s3_class(spatial_rsr(spatial_car(W, level = "obs"), restrict_to = ~ x),
                  "tulpa_rsr")

  # An HSGP basis and an SPDE mesh are neither an adjacency nor Vecchia
  # factors, so no kernel can apply the projection to them.
  expect_error(spatial_rsr(spatial_gp(~ lon + lat, approx = "hsgp"),
                           restrict_to = ~ x), "areal or NNGP")
  expect_setequal(.RSR_FIELDS, c("icar", "car", "bym2", "car_proper",
                                 "gp", "nngp"))
})


test_that("one projector builder over the observation -> field map", {
  # The areal route passes `spatial_idx`, the continuous one `obs_to_loc`; both
  # are 1-based maps into the field's own coordinates, and the projector is the
  # same construction over either.
  set.seed(11)
  X <- cbind(1, stats::rnorm(12))
  map <- rep(1:4, each = 3)

  P <- tulpa:::.rsr_unit_projection(X, map, 4L)
  expect_equal(dim(P), c(4L, 4L))
  expect_equal(P, t(P))                    # symmetric
  expect_equal(P %*% P, P, tolerance = 1e-10)   # idempotent
  # It annihilates the aggregated design it was built against.
  X_unit <- do.call(rbind, lapply(1:4, function(u) colMeans(X[map == u, ,
                                                              drop = FALSE])))
  expect_lt(max(abs(P %*% X_unit)), 1e-10)

  # One observation per coordinate -- the continuous case -- is the identity
  # map, and the projector is then the plain one on X itself.
  expect_equal(tulpa:::.rsr_unit_projection(X, seq_len(12), 12L),
               compute_rsr_projection(X), tolerance = 1e-10)
})


test_that("a restricted continuous fit reports an orthogonal field", {
  skip_on_cran()
  sim <- .rsrc_sim(1L)
  f_rsr   <- .rsrc_fit(sim, spatial_rsr(spatial_gp(~ lon + lat),
                                        restrict_to = ~ x))
  f_plain <- .rsrc_fit(sim, spatial_gp(~ lon + lat))

  expect_identical(f_rsr$backend, "gibbs")
  # Measured: 6.2e-15 restricted against 14.3 unrestricted. The gate is the
  # separation, not the tolerance.
  expect_lt(.rsrc_orthogonality(f_rsr, sim$d), 1e-8)
  expect_gt(.rsrc_orthogonality(f_plain, sim$d), 1e-2)
})


test_that("a backend without the projection refuses a restricted continuous field", {
  skip_on_cran()
  # The projection lives in the Polya-Gamma kernel and the nested-Laplace
  # multi-block driver; every other backend reads the underlying $type and
  # would fit the plain field while still reporting $spatial$rsr = TRUE
  # (gcol33/tulpa#792).
  sim <- .rsrc_sim(1L, n = 40L)
  expect_error(
    tulpa(y ~ x, data = sim$d, family = "binomial",
          n_trials = rep(25L, sim$n),
          spatial = spatial_rsr(spatial_gp(~ lon + lat), restrict_to = ~ x),
          mode = "laplace"),
    "would fit it unprojected")
})


# Several observations per location, so the observation -> location incidence
# is not the identity and a projector differs from the gather it replaces.
.rsrc_nngp_data <- function(seed = 3L, n_loc = 40L, reps = 2L) {
  set.seed(seed)
  loc <- data.frame(lon = stats::runif(n_loc), lat = stats::runif(n_loc))
  d <- loc[rep(seq_len(n_loc), each = reps), ]
  rownames(d) <- NULL
  d$x <- stats::rnorm(nrow(d))
  d$count <- stats::rpois(nrow(d), exp(0.3 + 0.5 * d$x + 0.5 * sin(4 * d$lon)))
  d
}

.rsrc_nngp_block <- function(d, nn = 6L) {
  tulpa:::.spatial_spec_to_nl_prior(
    tulpa:::validate_gp(spatial_gp(~ lon + lat, nn = nn), d))
}


test_that("an NNGP block read through its incidence fits the gathered model", {
  skip_on_cran()
  # Three routes to one model: the single-block NNGP kernel, the multi-block
  # driver gathering the field through spatial_idx, and the same driver reading
  # it through the incidence S as a projector.
  d <- .rsrc_nngp_data()
  N <- nrow(d)
  X <- stats::model.matrix(~ x, d)
  blk <- .rsrc_nngp_block(d)
  g <- expand.grid(s = c(0.2, 0.6), p = c(0.1, 0.3, 0.8))
  blk$sigma2_grid <- g$s
  blk$phi_gp_grid <- g$p
  ctl <- list(axis_refine = "none", prune = FALSE, diagnose_k = FALSE)
  fit_with <- function(prior) {
    tulpa_nested_laplace(y = d$count, n_trials = rep(1L, N), X = X,
                         family = "poisson", prior = prior, control = ctl)
  }
  S <- matrix(0, N, blk$n_spatial)
  S[cbind(seq_len(N), blk$spatial_idx)] <- 1
  via_S <- blk
  via_S$spatial_idx <- NULL
  via_S$projector <- S

  single   <- fit_with(blk)
  gathered <- fit_with(list(blk))
  incident <- fit_with(via_S)
  for (fit in list(gathered, incident)) {
    expect_equal(fit$log_marginal, single$log_marginal, tolerance = 1e-6)
    expect_equal(unname(coef(fit)), unname(coef(single)), tolerance = 1e-6)
  }
  expect_error(fit_with(c(via_S, list(spatial_idx = blk$spatial_idx))),
               "either `projector` or `spatial_idx`")
})


test_that("a non-binomial restricted NNGP field is fit by nested Laplace through A = S P", {
  skip_on_cran()
  d <- .rsrc_nngp_data()
  sp <- spatial_rsr(spatial_gp(~ lon + lat, nn = 6), restrict_to = ~ x)
  via <- tulpa(count ~ x, data = d, family = "poisson", spatial = sp,
               mode = "auto")
  expect_equal(via$backend, "nested_laplace")
  # The same fit as the registry door handed the projector directly: row i of
  # A is the location-level projector's row for observation i's location.
  X <- stats::model.matrix(~ x, d)
  blk <- .rsrc_nngp_block(d)
  P <- tulpa:::.rsr_unit_projection(X, blk$spatial_idx, blk$n_spatial)
  blk$projector <- P[blk$spatial_idx, , drop = FALSE]
  blk$spatial_idx <- NULL
  direct <- tulpa_nested_laplace(y = d$count, n_trials = rep(1L, nrow(d)),
                                 X = X, family = "poisson", prior = blk)
  expect_equal(via$log_marginal, direct$log_marginal, tolerance = 1e-8)
  expect_equal(unname(coef(via)), unname(coef(direct)), tolerance = 1e-8)
})


test_that("a restricted NNGP field shares the nested stack with a (1 | g) term", {
  skip_on_cran()
  d <- .rsrc_nngp_data()
  d$g <- rep(1:5, length.out = nrow(d))
  fit <- tulpa(count ~ x + (1 | g), data = d, family = "poisson",
               spatial = spatial_rsr(spatial_gp(~ lon + lat, nn = 6),
                                     restrict_to = ~ x),
               mode = "nested_laplace")
  expect_equal(fit$backend, "nested_laplace")
  expect_true(all(is.finite(coef(fit))))
})


test_that("a binomial restricted NNGP field on nested Laplace agrees with the Gibbs kernel", {
  skip_on_cran()
  sim <- .rsrc_sim(1L)
  sp <- spatial_rsr(spatial_gp(~ lon + lat), restrict_to = ~ x)
  nested <- tulpa(y ~ x, data = sim$d, family = "binomial",
                  n_trials = rep(25L, sim$n), spatial = sp,
                  mode = "nested_laplace")
  gibbs <- .rsrc_fit(sim, sp, n_iter = 4000L)
  expect_equal(nested$backend, "nested_laplace")
  # Measured over seeds 1-4 at 8000 Gibbs iterations, the slope's offset in
  # Gibbs posterior SDs: -0.69 / -0.28 / -0.23 / -0.17 restricted, against
  # -1.34 / -0.50 / 0.07 / -0.34 for the unrestricted NNGP field against its
  # own Gibbs run, with the restricted SDs within 2%
  # (dev_notes/issue942/gibbs_vs_nested.R). Seed 1 reads 0.72 here.
  b_gibbs <- fixed_draws(gibbs)[, "x"]
  z <- (coef(nested)[["x"]] - mean(b_gibbs)) / stats::sd(b_gibbs)
  expect_lt(abs(z), 1)
  ci <- confint(nested)["x", ]
  expect_equal(unname(diff(ci)) / (2 * 1.96), stats::sd(b_gibbs),
               tolerance = 0.1)
  expect_lt(abs(coef(nested)[["x"]] - sim$beta_m), 0.2)
})


test_that("the restriction buys the MARGINAL association, not a better one", {
  skip_if_not_slow()
  # What RSR targets. The restriction puts no part of the shared smooth signal
  # in the field, so the fixed effect takes all of it: the restricted fit
  # tracks the marginal association and NOT the slope conditional on the field.
  # That is an estimand difference rather than a bias one (Bradley 2024), and
  # it is why Khan & Calder (2022) find a non-spatial model competitive with
  # RSR -- the non-spatial fit estimates the same marginal quantity.
  #
  # Measured over these 5 seeds: mean |error| against the marginal value 0.060
  # restricted / 0.055 non-spatial / 0.097 unrestricted; against the
  # conditional one 0.743 / 0.713 / 0.626.
  seeds <- 1:5
  err <- t(vapply(seeds, function(sd) {
    sim <- .rsrc_sim(sd)
    b_rsr <- coef(.rsrc_fit(sim, spatial_rsr(spatial_gp(~ lon + lat),
                                             restrict_to = ~ x)))[["x"]]
    b_pl  <- coef(.rsrc_fit(sim, spatial_gp(~ lon + lat)))[["x"]]
    c(rsr_marg = abs(b_rsr - sim$beta_m), rsr_cond = abs(b_rsr - sim$beta_c),
      pl_marg  = abs(b_pl  - sim$beta_m))
  }, numeric(3)))
  m <- colMeans(err)

  # The restricted fit is an order of magnitude closer to the marginal value
  # than to the conditional one.
  expect_lt(m[["rsr_marg"]], 0.2)
  expect_gt(m[["rsr_cond"]], 0.5)
  expect_lt(m[["rsr_marg"]], 0.3 * m[["rsr_cond"]])
  # And closer to it than the unrestricted fit, which keeps part of the shared
  # signal in the field.
  expect_lt(m[["rsr_marg"]], m[["pl_marg"]])
})
