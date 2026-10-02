# Refinement changes the quadrature error, not the measure (gcol33/tulpa#620).
#
# The outer nodes are a quadrature rule for a prior declared before the fit, so
# a grid that gains nodes after seeing the data integrates the same measure it
# started with -- only more accurately. These tests run the whole driver on a
# fixture whose fixed-measure answer is available to any accuracy: a Gaussian
# sample with known mean, whose residual-scale marginal likelihood is exact.

# Exact log marginal of y ~ N(0, sigma^2) at one sigma, up to a constant that
# cancels in the weights.
gaussian_sigma_inner <- function(y) {
  n <- length(y); s2 <- sum(y^2)
  function(hypers) {
    sigma <- as.numeric(hypers[["sigma"]])
    if (!is.finite(sigma) || sigma <= 0) return(list(log_marginal = -Inf))
    list(log_marginal = -n * log(sigma) - 0.5 * s2 / sigma^2,
         beta_mean = c(mu = 0), beta_cov = matrix(sigma^2 / n, 1, 1))
  }
}

# The declared prior, and the support it is normalised over. Both fixed before
# the fit, so they are the same measure whatever nodes end up integrating it.
LOG_PRIOR <- function(s) stats::dexp(s, 1, log = TRUE)
SLAB <- c(0.2, 6)

# Moments of the posterior, integrated densely over the declared support.
#
# The integration coordinate of a `log_scale` axis is `log sigma`, and a spec's
# `log_prior` enters the target there: the engine weights
# `exp(log_marginal + log_prior)` by cell widths measured in `log sigma` and
# applies no volume element, the same convention the outer Pareto-k target
# carries. So the reference is a dense rule on `log sigma`, which is what the
# nodes are a quadrature rule FOR.
reference_moments <- function(y, slab = SLAB, n = 200001L) {
  u  <- seq(log(slab[1L]), log(slab[2L]), length.out = n)
  s  <- exp(u)
  inner <- gaussian_sigma_inner(y)
  lm <- vapply(s, function(v) inner(c(sigma = v))$log_marginal, numeric(1)) +
        LOG_PRIOR(s)
  w <- exp(lm - max(lm)); w <- w / sum(w)
  c(mean = sum(w * s), sd = sqrt(sum(w * s^2) - sum(w * s)^2))
}

slab_grid <- function(m, slab = SLAB) {
  exp(seq(log(slab[1L]), log(slab[2L]), length.out = m))
}

fit_sigma <- function(y, grid, control, slab = SLAB) {
  specs <- list(hyper_axis_spec("sigma", grid = grid, log_scale = TRUE,
                                bounds = c(0, Inf), refinable = TRUE,
                                log_prior = LOG_PRIOR, slab_bounds = slab))
  tulpa_hyper_grid(specs, gaussian_sigma_inner(y), combine = "none",
                   n_draws = 0L, control = control)
}

REFINE <- list(adaptive_grid = TRUE, adaptive_grid_edge_thresh = 1e-6,
               adaptive_grid_max_passes = 3L,
               var_of_means_consistency = TRUE)
PINNED <- list(adaptive_grid = FALSE, var_of_means_consistency = FALSE)

test_that("refinement moves the answer towards the measure, not away from it", {
  skip_on_cran()
  set.seed(620)
  y <- stats::rnorm(60, 0, 1.3)
  ref <- reference_moments(y)

  for (m in c(7L, 13L)) {
    pinned  <- fit_sigma(y, slab_grid(m), PINNED)
    refined <- fit_sigma(y, slab_grid(m), REFINE)
    # The arms differ in their node sets, which is the premise of the test.
    expect_gt(nrow(refined$theta_grid), nrow(pinned$theta_grid))
    err <- function(f) abs(f$theta_mean[["sigma"]] - ref[["mean"]])
    expect_lt(err(refined), err(pinned), label = sprintf("m = %d", m))
    expect_lt(err(refined) / ref[["mean"]], 0.03)
  }
})

test_that("two grids that resolve the measure agree on the posterior", {
  # The invariance the issue asks for: a coarse grid refined after seeing the
  # data and a fine grid pinned before it integrate ONE measure, so they report
  # one posterior. Before the nodes carried integration weights, the added
  # nodes reweighted the prior instead and the reported SD of a refinable axis
  # moved by more than 100 %.
  set.seed(620)
  y <- stats::rnorm(60, 0, 1.3)
  refined <- fit_sigma(y, slab_grid(7L), REFINE)
  fine    <- fit_sigma(y, slab_grid(21L), PINNED)
  expect_equal(unname(refined$theta_mean[["sigma"]]),
               unname(fine$theta_mean[["sigma"]]), tolerance = 0.02)
  expect_equal(unname(refined$theta_sd[["sigma"]]),
               unname(fine$theta_sd[["sigma"]]), tolerance = 0.15)
})

test_that("a finer grid integrates the declared measure more accurately", {
  skip_on_cran()
  set.seed(6201)
  y <- stats::rnorm(40, 0, 0.8)
  ref <- reference_moments(y)
  err <- function(m) abs(fit_sigma(y, slab_grid(m), PINNED)$theta_mean[["sigma"]] -
                         ref[["mean"]])
  e <- vapply(c(5L, 13L, 41L), err, numeric(1))
  expect_lt(e[3L], e[2L])
  expect_lt(e[2L], e[1L])
  expect_lt(e[3L] / ref[["mean"]], 0.01)
})

test_that("the nodes are a quadrature rule for the declared prior", {
  # A flat inner fit, so the weights ARE the measure: refining then has one
  # visible consequence, the same declared prior integrated more accurately.
  flat <- function(hypers) list(log_marginal = 0)
  u <- seq(log(SLAB[1L]), log(SLAB[2L]), length.out = 400001L)
  w <- exp(LOG_PRIOR(exp(u))); w <- w / sum(w)
  ref_mean <- sum(w * exp(u))

  got <- vapply(c(5L, 11L, 41L), function(m) {
    sp <- list(hyper_axis_spec("sigma", grid = slab_grid(m), log_scale = TRUE,
                               bounds = c(0, Inf), log_prior = LOG_PRIOR,
                               slab_bounds = SLAB))
    tulpa_hyper_grid(sp, flat, combine = "none",
                     n_draws = 0L)$theta_mean[["sigma"]]
  }, numeric(1))

  err <- abs(got - ref_mean)
  expect_lt(err[3L], err[2L])
  expect_lt(err[2L], err[1L])
  expect_lt(err[3L] / ref_mean, 0.002)
})

test_that("nodes outside the declared support are not proposed", {
  # The support is a prior choice, so refinement may only reduce the quadrature
  # error inside it. An axis whose posterior presses on the ceiling reports the
  # boundary mass rather than moving the support outward.
  set.seed(6202)
  y <- stats::rnorm(50, 0, 3.5)
  slab <- c(0.3, 2)
  fit <- fit_sigma(y, slab_grid(5L, slab), REFINE, slab = slab)
  expect_true(all(fit$theta_grid[, "sigma"] >= slab[1L]))
  expect_true(all(fit$theta_grid[, "sigma"] <= slab[2L]))
  expect_identical(.nl_edge_mass_axes(fit), "sigma:upper")
})

test_that("an evenly spaced grid keeps the equal weights it always had", {
  # The rule the engine applied before the nodes carried weights, so an
  # unrefined grid over a flat declared measure is unchanged.
  lev <- exp(seq(log(0.2), log(5), length.out = 6L))
  spec <- hyper_axis_spec("sigma", grid = lev, log_scale = TRUE,
                          bounds = c(0, Inf))
  w <- .hyper_axis_level_weights(lev, spec)
  expect_equal(unname(w), rep(1 / 6, 6), tolerance = 1e-12)
})

test_that("tulpa_hyper_grid refuses an unknown control knob", {
  # The consistency-pass trigger moved from a ratio against the parabola to an
  # ESS on the weights, so the retired spelling has to say so rather than be
  # ignored.
  expect_true("var_of_means_min_ess" %in% .CONTROL_KEYS$hyper_grid)
  expect_false("var_of_means_tolerance" %in% .CONTROL_KEYS$hyper_grid)
  expect_error(
    tulpa_hyper_grid(list(hyper_axis_spec("sigma", grid = c(1, 2))),
                     function(hypers) list(log_marginal = 0),
                     control = list(var_of_means_tolerance = 0.7)),
    "Unknown control knob")
})

test_that("log_prior_coord says which coordinate the declared prior lives on", {
  # `Exponential(1)` written as a density on sigma. Read on the integration
  # coordinate (the default) the nodes integrate it as a density on log sigma;
  # declared `"natural"` they integrate the density the caller wrote, because
  # the change of variables is carried across the way `slab_log_density`'s is.
  flat <- function(hypers) list(log_marginal = 0)
  ref <- function(coord) {
    u <- seq(log(SLAB[1L]), log(SLAB[2L]), length.out = 400001L)
    s <- exp(u)
    lw <- LOG_PRIOR(s) + if (coord == "natural") u else 0
    w <- exp(lw - max(lw)); w <- w / sum(w)
    sum(w * s)
  }
  got <- function(coord) {
    sp <- list(hyper_axis_spec("sigma", grid = slab_grid(41L), log_scale = TRUE,
                               bounds = c(0, Inf), log_prior = LOG_PRIOR,
                               slab_bounds = SLAB, log_prior_coord = coord))
    tulpa_hyper_grid(sp, flat, combine = "none",
                     n_draws = 0L)$theta_mean[["sigma"]]
  }
  for (coord in c("integration", "natural")) {
    expect_lt(abs(got(coord) - ref(coord)) / ref(coord), 0.01, label = coord)
  }
  # The two readings are different measures, so the test is not vacuous.
  expect_gt(got("natural"), 1.15 * got("integration"))

  # The default is the reading every existing fit was taken under.
  expect_identical(hyper_axis_spec("s", grid = c(1, 2))$log_prior_coord,
                   "integration")
  expect_error(hyper_axis_spec("s", grid = c(1, 2), log_prior_coord = "log"),
               "should be one of")
  # Inert on a linear axis, where the two coordinates coincide.
  lin <- function(coord) {
    sp <- list(hyper_axis_spec("m", grid = seq(-2, 2, length.out = 9L),
                               log_prior = function(x) stats::dnorm(x, 0, 1,
                                                                    log = TRUE),
                               log_prior_coord = coord))
    tulpa_hyper_grid(sp, flat, combine = "none", n_draws = 0L)$theta_sd[["m"]]
  }
  expect_identical(lin("natural"), lin("integration"))
})

# --------------------------------------------------------------------------- #
# Refinement adds levels of the tensor (gcol33/tulpa#932)                      #
# --------------------------------------------------------------------------- #
#
# A refinement pass lays each new level in every row of the other axes, so the
# grid stays a tensor and the product rule measures it. The `refining` tags say
# which levels were declared: those fix the span the outer cells reach and any
# prior read off the nodes.

LEVEL_AXES <- list(sigma   = exp(seq(log(0.1), log(3), length.out = 5)),
                   phi_pos = exp(seq(log(1),   log(60), length.out = 5)))
LEVEL_TENSOR <- as.matrix(expand.grid(LEVEL_AXES))
FLAT_LEVEL_SPECS <- list(list(name = "sigma", log_scale = TRUE),
                         list(name = "phi_pos", log_scale = TRUE))

# `LEVEL_TENSOR` with `pts` added as levels of `axis`, tagged as a pass tags
# them.
add_levels <- function(g, ref, axis, pts) {
  others <- setdiff(colnames(g), axis)
  rows <- unique(g[, others, drop = FALSE])
  new <- matrix(NA_real_, nrow(rows) * length(pts), ncol(g),
                dimnames = list(NULL, colnames(g)))
  for (b in others) new[, b] <- rep(rows[, b], each = length(pts))
  new[, axis] <- rep(pts, nrow(rows))
  list(g = rbind(g, new), ref = c(ref, rep(axis, nrow(new))))
}

test_that("a new level is laid in every row of the grid", {
  lm <- rep(-10, 25L); lm[13L] <- 0
  pk <- .hyper_tensor_level_cells(LEVEL_TENSOR, lm, "sigma", c(0.2, 0.4))
  expect_identical(nrow(pk$new_cells), 2L * length(LEVEL_AXES$phi_pos))
  expect_setequal(pk$new_cells[, "phi_pos"], LEVEL_AXES$phi_pos)
  expect_identical(pk$warm_start_idx, 13L)
  # Rows through a level an earlier pass added to another axis are rows too.
  x <- add_levels(LEVEL_TENSOR, rep("", 25L), "phi_pos", 2.5)
  pk2 <- .hyper_tensor_level_cells(x$g, c(lm, rep(-10, 5L)), "sigma", 0.2)
  expect_true(2.5 %in% pk2$new_cells[, "phi_pos"])
  # An unsolved cell holds no row open.
  lm2 <- lm; lm2[LEVEL_TENSOR[, "phi_pos"] == LEVEL_AXES$phi_pos[5]] <- -Inf
  pk3 <- .hyper_tensor_level_cells(LEVEL_TENSOR, lm2, "sigma", 0.2)
  expect_false(LEVEL_AXES$phi_pos[5] %in% pk3$new_cells[, "phi_pos"])
})

test_that("a new level skips only the rows holding the tail of the mass", {
  lm <- rep(0, 25L)
  row_mass <- c(0.5, 0.3, 0.1989, 1e-3, 1e-4)
  phi <- LEVEL_TENSOR[, "phi_pos"]
  lw <- log(row_mass[match(phi, LEVEL_AXES$phi_pos)] / 5)
  pk <- .hyper_tensor_level_cells(LEVEL_TENSOR, lm, "sigma", 0.2,
                                  log_weight = lw, row_tail = 5e-4)
  # The three heaviest rows hold 0.9989 of the mass; the fourth closes it past
  # 1 - 5e-4, and the lightest is left out.
  expect_setequal(pk$new_cells[, "phi_pos"], LEVEL_AXES$phi_pos[1:4])
  # No tail, or no weights, refines every row.
  for (pk0 in list(.hyper_tensor_level_cells(LEVEL_TENSOR, lm, "sigma", 0.2,
                                             log_weight = lw, row_tail = 0),
                   .hyper_tensor_level_cells(LEVEL_TENSOR, lm, "sigma", 0.2))) {
    expect_setequal(pk0$new_cells[, "phi_pos"], LEVEL_AXES$phi_pos)
  }
})

test_that("a solved level grows into the rows next to the ones holding its mass", {
  phi <- LEVEL_AXES$phi_pos
  # sigma 0.2 laid in the first two phi rows; its mass sits in the second.
  g <- rbind(LEVEL_TENSOR, cbind(sigma = 0.2, phi_pos = phi[1:2]))
  ref <- c(rep("", 25L), "sigma", "sigma")
  lm <- c(rep(-50, 25L), -50, 0)
  grow <- .hyper_level_frontier(g, lm, FLAT_LEVEL_SPECS, ref, "sigma", 0.2)
  expect_equal(unname(grow[, "phi_pos"]), phi[3])
  expect_equal(unname(grow[, "sigma"]), 0.2)
  # A row holding none of it passes nothing on, and a neighbour that already
  # holds the level is not laid again.
  lm[27L] <- -50; lm[26L] <- 0
  expect_null(.hyper_level_frontier(g, lm, FLAT_LEVEL_SPECS, ref, "sigma", 0.2))
})

test_that("interior levels are measured by the product rule over all levels", {
  lp <- log(LEVEL_AXES$phi_pos)
  x <- add_levels(LEVEL_TENSOR, rep("", 25L), "phi_pos",
                  exp((lp[2] + lp[3]) / 2))
  for (specs in list(FLAT_LEVEL_SPECS,
                     .joint_axis_specs_from_grid(LEVEL_TENSOR))) {
    expect_equal(.hyper_log_quad_weights(x$g, specs, refining = x$ref),
                 .hyper_log_quad_weights(x$g, specs), tolerance = 1e-14)
  }
  # A grid nothing refined takes the product rule unchanged.
  specs <- .joint_axis_specs_from_grid(LEVEL_TENSOR)
  expect_identical(.hyper_log_quad_weights(LEVEL_TENSOR, specs,
                                           refining = rep("", 25L)),
                   .hyper_log_quad_weights(LEVEL_TENSOR, specs))
})

test_that("an unrefined grid reports the support of its levels", {
  for (specs in list(FLAT_LEVEL_SPECS,
                     .joint_axis_specs_from_grid(LEVEL_TENSOR))) {
    sup <- .hyper_grid_supports(LEVEL_TENSOR, specs)
    expect_identical(.hyper_grid_supports(LEVEL_TENSOR, specs,
                                          refining = rep("", 25L)), sup)
    for (spec in specs) {
      expect_identical(sup[[spec$name]],
                       .hyper_axis_support(LEVEL_TENSOR[, spec$name], spec))
    }
    span <- .joint_axis_span(LEVEL_TENSOR, LEVEL_TENSOR, specs,
                             refining = rep("", 25L))
    for (a in names(span)) {
      expect_identical(span[[a]]$integrated, sup[[a]])
      expect_identical(span[[a]]$integrated, span[[a]]$declared)
    }
  }
})

test_that("densifying beside an outer node leaves the span the declared nodes had", {
  lp <- log(LEVEL_AXES$phi_pos); ls <- log(LEVEL_AXES$sigma)
  x <- add_levels(LEVEL_TENSOR, rep("", 25L), "phi_pos",
                  exp(c((lp[4] + 3 * lp[5]) / 4, (3 * lp[1] + lp[2]) / 4)))
  x <- add_levels(x$g, x$ref, "sigma", exp((ls[4] + ls[5]) / 2))
  tensor <- .hyper_grid_supports(LEVEL_TENSOR, FLAT_LEVEL_SPECS)
  sup <- .hyper_grid_supports(x$g, FLAT_LEVEL_SPECS, refining = x$ref)
  expect_identical(sup, tensor)
  # The half step of the joined levels alone is narrower than the declared
  # span: that is the truncation the declared levels' edges prevent.
  lev <- .hyper_axis_support(x$g[, "phi_pos"], FLAT_LEVEL_SPECS[[2L]])
  expect_lt(diff(log(lev)), diff(log(sup$phi_pos)))
  # The measure integrates exactly the declared tensor's area.
  area <- prod(vapply(tensor, function(s) diff(log(s)), numeric(1)))
  w <- exp(.hyper_log_quad_weights(x$g, FLAT_LEVEL_SPECS, refining = x$ref,
                                   absolute = TRUE))
  expect_equal(sum(w), area, tolerance = 1e-12)
  span <- .joint_axis_span(LEVEL_TENSOR, x$g, FLAT_LEVEL_SPECS,
                           refining = x$ref)
  expect_identical(span$phi_pos$integrated, span$phi_pos$declared)
})

test_that("an extension widens the span to the added level's own mirror", {
  lp <- log(LEVEL_AXES$phi_pos); ls <- log(LEVEL_AXES$sigma)
  tensor <- .hyper_grid_supports(LEVEL_TENSOR, FLAT_LEVEL_SPECS)
  area0 <- prod(vapply(tensor, function(s) diff(log(s)), numeric(1)))
  x_ext <- exp(lp[5] + (lp[5] - lp[4]))
  x <- add_levels(LEVEL_TENSOR, rep("", 25L), "phi_pos", x_ext)
  sup <- .hyper_grid_supports(x$g, FLAT_LEVEL_SPECS, refining = x$ref)
  expect_identical(sup$phi_pos[1L], tensor$phi_pos[1L])
  expect_equal(log(sup$phi_pos[2L]), log(x_ext) + (log(x_ext) - lp[5]) / 2,
               tolerance = 1e-14)
  expect_identical(sup$sigma, tensor$sigma)
  w <- exp(.hyper_log_quad_weights(x$g, FLAT_LEVEL_SPECS, refining = x$ref,
                                   absolute = TRUE))
  expect_equal(sum(w),
               area0 + (log(sup$phi_pos[2L]) - log(tensor$phi_pos[2L])) *
                 diff(log(tensor$sigma)),
               tolerance = 1e-12)
  span <- .joint_axis_span(LEVEL_TENSOR, x$g, FLAT_LEVEL_SPECS, refining = x$ref)
  expect_identical(span$phi_pos$integrated, sup$phi_pos)
  expect_gt(span$phi_pos$integrated[2L], span$phi_pos$declared[2L])
})

test_that("an added level does not move a prior read off the declared nodes", {
  # The copy scale's exponential rate is set by its largest declared node, so
  # a level a pass laid past it must not reach the specs rebuilt from the grid.
  tg <- as.matrix(expand.grid(sigma = LEVEL_AXES$sigma,
                              alpha = c(0, 0.3, 0.6, 1, 1.5)))
  x <- add_levels(tg, rep("", nrow(tg)), "alpha", 2.2)
  lq <- .nl_grid_log_quad(x$g, refining = x$ref)
  declared <- .joint_axis_specs_from_grid(tg)
  expect_equal(lq, .hyper_log_quad_weights(x$g, declared, refining = x$ref),
               tolerance = 1e-14)
  expect_false(isTRUE(all.equal(lq, .nl_grid_log_quad(x$g))))
})

# Correlated two-axis posterior: log sigma and a location m, so a refinement on
# sigma at the modal row is the configuration where a mask or a stand-in term
# would move the reported m.
two_axis_inner <- function(hypers) {
  u <- log(as.numeric(hypers[["sigma"]])); m <- as.numeric(hypers[["m"]])
  z1 <- (u - 0.2) / 0.35; z2 <- (m - 0.6) / 0.8
  list(log_marginal = -0.5 * (z1^2 - 2 * 0.7 * z1 * z2 + z2^2) / (1 - 0.7^2))
}
fit_two_axis <- function(n_sigma, control) {
  specs <- list(
    hyper_axis_spec("sigma", grid = exp(seq(-1.5, 1.9, length.out = n_sigma)),
                    log_scale = TRUE, bounds = c(0, Inf), refinable = TRUE,
                    log_prior = function(s) 0, slab_bounds = exp(c(-1.8, 2.2))),
    hyper_axis_spec("m", grid = seq(-3, 4, length.out = 15L),
                    log_prior = function(x) stats::dnorm(x, 0, 3, log = TRUE)))
  tulpa_hyper_grid(specs, two_axis_inner, combine = "none", n_draws = 0L,
                   control = control)
}

test_that("a refined grid reports the posterior its own cell weights define", {
  coarse  <- fit_two_axis(5L, PINNED)
  refined <- fit_two_axis(5L, REFINE)
  fine    <- fit_two_axis(61L, PINNED)
  expect_gt(nrow(refined$theta_grid), nrow(coarse$theta_grid))
  # Every pass adds levels of the tensor, tagged with the axis they refine.
  expect_true(all(refined$refining_axis %in% c("", "sigma")))
  expect_true(any(refined$refining_axis == "sigma"))
  # One posterior per fit: the reported means are the weighted means over every
  # cell, the same cells and weights the fit's draws are taken from, on the axis
  # that was refined and on the one that was not.
  w <- refined$weights
  for (ax in c("sigma", "m")) {
    expect_equal(refined$theta_mean[[ax]], sum(w * refined$theta_grid[, ax]),
                 tolerance = 1e-12, label = ax)
  }
  # The refined grid integrates the same measure at a finer resolution, so the
  # refined axis and the one correlated with it move towards the fine
  # tensor's. The evidence is not asserted: the bisection stops on the axis
  # marginal's ESS and leaves the outer gaps, where the rows away from the
  # mode carry their conditional, as coarse as they were.
  closer <- function(f) abs(refined[[f]] - fine[[f]]) < abs(coarse[[f]] - fine[[f]])
  expect_true(all(closer("theta_mean")))
  expect_true(all(closer("theta_sd")))
  expect_true(closer("theta_ci_hi")[["sigma"]])
})

