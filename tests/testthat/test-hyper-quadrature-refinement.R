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
# Slice cells are measured by the box they own (gcol33/tulpa#733)             #
# --------------------------------------------------------------------------- #
#
# A refinement pass adds a level on one axis at ONE combination of the others.
# The product rule is a tensor measure, so on a grid carrying such cells every
# cell is measured by its own box instead.

SLICE_AXES <- list(sigma   = exp(seq(log(0.1), log(3), length.out = 5)),
                   phi_pos = exp(seq(log(1),   log(60), length.out = 5)))
SLICE_TENSOR <- as.matrix(expand.grid(SLICE_AXES))

test_that("a slice cell leaves every row of the base tensor its mass", {
  specs <- .joint_axis_specs_from_grid(SLICE_TENSOR)
  mid <- sqrt(SLICE_AXES$phi_pos[4] * SLICE_AXES$phi_pos[5])
  g <- rbind(SLICE_TENSOR, c(SLICE_AXES$sigma[3], mid))
  ref <- c(rep("", 25L), "phi_pos")
  w <- exp(.hyper_log_quad_weights(g, specs, refining = ref))
  expect_equal(sum(w), 1, tolerance = 1e-12)
  rows <- as.numeric(tapply(w, signif(g[, "sigma"], 6), sum))
  expect_equal(unname(rows), rep(0.2, 5), tolerance = 1e-12)
  # The tensor itself takes the product rule unchanged.
  expect_identical(.hyper_log_quad_weights(SLICE_TENSOR, specs,
                                           refining = rep("", 25L)),
                   .hyper_log_quad_weights(SLICE_TENSOR, specs))
})

test_that("crossing refinements and an extension conserve the base area exactly", {
  # Flat log axes with no declared span: the absolute measure is coordinate
  # area, so the total is the tensor's area plus the one extension region.
  sp <- list(list(name = "sigma", log_scale = TRUE),
             list(name = "phi_pos", log_scale = TRUE))
  ls <- log(SLICE_AXES$sigma); lp <- log(SLICE_AXES$phi_pos)
  area0 <- prod(c(diff(range(ls)), diff(range(lp))) * 5 / 4)
  tensor_area <- sum(exp(.hyper_log_quad_weights(SLICE_TENSOR, sp,
                                                 absolute = TRUE)))
  expect_equal(tensor_area, area0, tolerance = 1e-12)

  ext_phi <- exp(lp[5] + (lp[5] - lp[4]))
  g <- rbind(SLICE_TENSOR,
             c(SLICE_AXES$sigma[3], exp((lp[4] + lp[5]) / 2)),
             c(SLICE_AXES$sigma[3], ext_phi),
             c(exp((ls[3] + ls[4]) / 2), SLICE_AXES$phi_pos[4]),
             c(exp((ls[2] + ls[3]) / 2), SLICE_AXES$phi_pos[4]))
  ref <- c(rep("", 25L), "phi_pos", "phi_pos", "sigma", "consistency_sigma")
  w <- exp(.hyper_log_quad_weights(g, sp, refining = ref, absolute = TRUE))
  expect_true(all(w > 0))
  old_edge <- lp[5] + (lp[5] - lp[4]) / 2
  new_edge <- log(ext_phi) + (log(ext_phi) - lp[5]) / 2
  expect_equal(sum(w), area0 + (new_edge - old_edge) * (ls[2] - ls[1]),
               tolerance = 1e-12)
})

test_that("a slice cell off the base levels of another axis is refused", {
  specs <- .joint_axis_specs_from_grid(SLICE_TENSOR)
  g <- rbind(SLICE_TENSOR, c(0.5, 30))
  expect_error(.hyper_log_quad_weights(g, specs,
                                       refining = c(rep("", 25L), "phi_pos")),
               "off the base levels")
})

test_that("refinement never anchors a slice at a cell placed on another axis", {
  g <- rbind(SLICE_TENSOR, c(SLICE_AXES$sigma[2], 5))
  ref <- c(rep("", 25L), "phi_pos")
  lm <- rep(-10, 26L); lm[26L] <- 0
  pk <- .hyper_new_mode_tracked_triples(g, lm, NULL, "sigma", 0.2,
                                        anchor_lev = SLICE_AXES$sigma[2],
                                        refining_axis = ref)
  expect_false(pk$warm_start_idx == 26L)
  expect_true(pk$new_cells[1L, "phi_pos"] %in% SLICE_AXES$phi_pos)
  expect_null(pk$calibration)
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
  expect_true(any(nzchar(refined$refining_axis)))
  # One posterior per fit: the reported means are the weighted means over every
  # cell, the same cells and weights the fit's draws are taken from, on the axis
  # that was refined and on the one that was not.
  w <- refined$weights
  for (ax in c("sigma", "m")) {
    expect_equal(refined$theta_mean[[ax]], sum(w * refined$theta_grid[, ax]),
                 tolerance = 1e-12, label = ax)
  }
  # The refined grid integrates the same measure at a finer resolution, so its
  # evidence and the refined axis move towards the fine tensor's.
  expect_lt(abs(refined$log_evidence - fine$log_evidence),
            abs(coarse$log_evidence - fine$log_evidence))
  expect_lt(abs(refined$theta_mean[["sigma"]] - fine$theta_mean[["sigma"]]),
            abs(coarse$theta_mean[["sigma"]] - fine$theta_mean[["sigma"]]))
})

# --------------------------------------------------------------------------- #
# The support a refined grid reports is the region its measure integrates     #
# --------------------------------------------------------------------------- #

FLAT_SLICE_SPECS <- list(list(name = "sigma", log_scale = TRUE),
                         list(name = "phi_pos", log_scale = TRUE))

# The span along `spec`'s axis that `.hyper_refined_log_quad()` integrates, read
# off the construction the measure itself uses: the base measure's outer edges
# and the cell every admitted slice point owns in its re-tiled row.
measure_span <- function(g, spec, ref) {
  home <- .hyper_slice_home(ref, nrow(g))
  v <- as.numeric(g[, spec$name])
  m <- .hyper_axis_measure(v[!nzchar(home)], spec, .hyper_axis_atom_mass(spec))
  m$spec <- spec
  cells <- lapply(.hyper_slice_rows(g, home, spec$name), function(i) {
    tl <- .hyper_fibre_tiling(m, v[i], close_domain = TRUE)
    tl$cell[tl$ok, , drop = FALSE]
  })
  u <- range(c(m$edges, unlist(cells)))
  if (isTRUE(spec$log_scale)) exp(u) else u
}

test_that("an unrefined grid reports the support of its levels", {
  for (specs in list(FLAT_SLICE_SPECS,
                     .joint_axis_specs_from_grid(SLICE_TENSOR))) {
    sup <- .hyper_grid_supports(SLICE_TENSOR, specs)
    expect_identical(.hyper_grid_supports(SLICE_TENSOR, specs,
                                          refining = rep("", 25L)), sup)
    for (spec in specs) {
      expect_identical(sup[[spec$name]],
                       .hyper_axis_support(SLICE_TENSOR[, spec$name], spec))
    }
    span <- .joint_axis_span(SLICE_TENSOR, SLICE_TENSOR, specs,
                             refining = rep("", 25L))
    for (a in names(span)) {
      expect_identical(span[[a]]$integrated, sup[[a]])
      expect_identical(span[[a]]$integrated, span[[a]]$declared)
    }
  }
})

test_that("densifying a row leaves the span the declared nodes had", {
  lp <- log(SLICE_AXES$phi_pos); ls <- log(SLICE_AXES$sigma)
  g <- rbind(SLICE_TENSOR,
             c(SLICE_AXES$sigma[3], exp((lp[4] + lp[5]) / 2)),
             c(SLICE_AXES$sigma[3], exp((3 * lp[1] + lp[2]) / 4)),
             c(exp((ls[4] + ls[5]) / 2), SLICE_AXES$phi_pos[2]))
  ref <- c(rep("", 25L), "phi_pos", "consistency_phi_pos", "sigma")
  tensor <- .hyper_grid_supports(SLICE_TENSOR, FLAT_SLICE_SPECS)
  sup <- .hyper_grid_supports(g, FLAT_SLICE_SPECS, refining = ref)
  expect_identical(sup, tensor)
  for (spec in FLAT_SLICE_SPECS) {
    expect_equal(sup[[spec$name]], measure_span(g, spec, ref),
                 tolerance = 1e-14)
    # The half step of the joined levels is narrower than what is integrated.
    lev <- .hyper_axis_support(g[, spec$name], spec)
    expect_lt(diff(log(lev)), diff(log(sup[[spec$name]])))
  }
  # The measure integrates exactly the tensor's area over that span.
  area <- prod(vapply(tensor, function(s) diff(log(s)), numeric(1)))
  w <- exp(.hyper_log_quad_weights(g, FLAT_SLICE_SPECS, refining = ref,
                                   absolute = TRUE))
  expect_equal(sum(w), area, tolerance = 1e-12)

  span <- .joint_axis_span(SLICE_TENSOR, g, FLAT_SLICE_SPECS, refining = ref)
  expect_identical(span$phi_pos$integrated, span$phi_pos$declared)
  expect_identical(span$sigma$integrated, sup$sigma)
})

test_that("an extension widens the span to the cell the extended row integrates", {
  lp <- log(SLICE_AXES$phi_pos); ls <- log(SLICE_AXES$sigma)
  tensor <- .hyper_grid_supports(SLICE_TENSOR, FLAT_SLICE_SPECS)
  area0 <- prod(vapply(tensor, function(s) diff(log(s)), numeric(1)))
  for (tag in c("phi_pos", "consistency_phi_pos")) {
    x_ext <- exp(lp[5] + (lp[5] - lp[4]))
    g <- rbind(SLICE_TENSOR, c(SLICE_AXES$sigma[3], x_ext),
               c(SLICE_AXES$sigma[2], exp((lp[2] + lp[3]) / 2)))
    ref <- c(rep("", 25L), tag, "phi_pos")
    sup <- .hyper_grid_supports(g, FLAT_SLICE_SPECS, refining = ref)
    expect_equal(sup$phi_pos,
                 measure_span(g, FLAT_SLICE_SPECS[[2L]], ref), tolerance = 1e-14)
    expect_identical(sup$phi_pos[1L], tensor$phi_pos[1L])
    expect_equal(log(sup$phi_pos[2L]), log(x_ext) + (log(x_ext) - lp[5]) / 2,
                 tolerance = 1e-14)
    expect_identical(sup$sigma, tensor$sigma)
    # What the widened span adds is the extension region of that one row.
    w <- exp(.hyper_log_quad_weights(g, FLAT_SLICE_SPECS, refining = ref,
                                     absolute = TRUE))
    expect_equal(sum(w),
                 area0 + (log(sup$phi_pos[2L]) - log(tensor$phi_pos[2L])) *
                   (ls[2L] - ls[1L]),
                 tolerance = 1e-12, label = tag)
    span <- .joint_axis_span(SLICE_TENSOR, g, FLAT_SLICE_SPECS, refining = ref)
    expect_identical(span$phi_pos$integrated, sup$phi_pos)
    expect_gt(span$phi_pos$integrated[2L], span$phi_pos$declared[2L])
  }
})

test_that("a fibre cell a base node owns past a closed edge adds no span", {
  # On a correlation the outer cell closes inside the domain, so a densified
  # row's half-step mirror can land past the base edge; the base node keeps
  # only its own base cell there, and the span stays where the mass stops.
  rho <- c(0.1, 0.4, 0.7, 0.9)
  sg <- exp(seq(log(0.2), log(2), length.out = 4))
  tg <- as.matrix(expand.grid(sigma = sg, rho_car = rho))
  specs <- list(hyper_axis_spec("sigma", sg, log_scale = TRUE),
                hyper_axis_spec("rho_car", rho))
  tensor <- .hyper_grid_supports(tg, specs)
  tensor_area <- sum(exp(.hyper_log_quad_weights(tg, specs, absolute = TRUE)))
  g <- rbind(tg, c(sg[2], 0.75))
  ref <- c(rep("", 16L), "rho_car")
  sup <- .hyper_grid_supports(g, specs, refining = ref)
  expect_identical(sup, tensor)
  expect_equal(sup$rho_car, measure_span(g, specs[[2L]], ref), tolerance = 1e-14)
  expect_gt(.hyper_axis_support(g[, "rho_car"], specs[[2L]])[2L],
            tensor$rho_car[2L])
  w <- exp(.hyper_log_quad_weights(g, specs, refining = ref, absolute = TRUE))
  expect_equal(sum(w), tensor_area, tolerance = 1e-12)

  # A slice point past the outermost node does extend it, inside the domain.
  g2 <- rbind(tg, c(sg[2], 0.97))
  sup2 <- .hyper_grid_supports(g2, specs, refining = ref)
  expect_equal(sup2$rho_car, c(tensor$rho_car[1L], 0.985), tolerance = 1e-14)
  expect_lt(sup2$rho_car[2L], 1)
})

# --------------------------------------------------------------------------- #
# A row reaching past the base span is tiled without a gap (gcol33/tulpa#740)  #
# --------------------------------------------------------------------------- #

# The pieces a re-tiled row's nodes own, on the integration coordinate: each
# base node's retained part of its base cell and its extension past the base
# span, and each admitted slice point's cell.
row_pieces <- function(g, spec, ref, row_value, row_axis) {
  home <- .hyper_slice_home(ref, nrow(g))
  v <- as.numeric(g[, spec$name])
  m <- .hyper_axis_measure(v[!nzchar(home)], spec, .hyper_axis_atom_mass(spec))
  m$spec <- spec
  in_row <- which(home == spec$name & g[, row_axis] == row_value)
  tl <- .hyper_fibre_tiling(m, v[in_row], close_domain = TRUE)
  list(tl = tl, owned = sum(tl$retained * diff(m$edges)) + sum(tl$base_ext) +
         sum(tl$cell[tl$ok, "hi"] - tl$cell[tl$ok, "lo"]),
       base_edges = range(m$edges))
}

# Absolute flat measure of a grid with one re-tiled phi_pos row at sigma[3]:
# the tensor's area plus that row's region past the base span, times the row's
# sigma cell width, and the row itself carrying its region times that width.
expect_row_conserved <- function(g, ref) {
  sp <- FLAT_SLICE_SPECS
  ls <- log(SLICE_AXES$sigma)
  sw <- ls[2L] - ls[1L]
  tensor <- .hyper_grid_supports(SLICE_TENSOR, sp)
  area0 <- prod(vapply(tensor, function(s) diff(log(s)), numeric(1)))
  pc <- row_pieces(g, sp[[2L]], ref, SLICE_AXES$sigma[3L], "sigma")
  region <- pc$tl$region
  expect_equal(pc$owned, diff(region), tolerance = 1e-12)
  sup <- .hyper_grid_supports(g, sp, refining = ref)
  expect_equal(log(sup$phi_pos), region, tolerance = 1e-14)
  w <- exp(.hyper_log_quad_weights(g, sp, refining = ref, absolute = TRUE))
  expect_equal(sum(w), area0 + (diff(region) - diff(pc$base_edges)) * sw,
               tolerance = 1e-12)
  expect_equal(sum(w[g[, "sigma"] == SLICE_AXES$sigma[3L]]), diff(region) * sw,
               tolerance = 1e-12)
  invisible(pc)
}

test_that("an extension k base steps out integrates its row without a gap", {
  lp <- log(SLICE_AXES$phi_pos)
  h <- lp[5L] - lp[4L]
  for (k in c(2, 3)) {
    g <- rbind(SLICE_TENSOR, c(SLICE_AXES$sigma[3L], exp(lp[5L] + k * h)))
    ref <- c(rep("", 25L), "phi_pos")
    pc <- expect_row_conserved(g, ref)
    # The outermost base node owns up to the midpoint towards the slice point,
    # (k - 1) / 2 steps past the base edge; the slice point owns the rest.
    expect_equal(pc$tl$base_ext[5L], (k - 1) / 2 * h, tolerance = 1e-12,
                 label = sprintf("k = %g", k))
    expect_equal(unname(pc$tl$cell[1L, "lo"]), lp[5L] + k * h / 2,
                 tolerance = 1e-12)
    # Both sides at once, with a densified cell inside the base span.
    g2 <- rbind(g, c(SLICE_AXES$sigma[3L], exp(lp[1L] - k * h)),
                c(SLICE_AXES$sigma[3L], exp((lp[2L] + lp[3L]) / 2)))
    ref2 <- c(ref, "phi_pos", "consistency_phi_pos")
    pc2 <- expect_row_conserved(g2, ref2)
    expect_equal(pc2$tl$base_ext[c(1L, 5L)], rep((k - 1) / 2 * h, 2L),
                 tolerance = 1e-12)
  }
})

test_that("a consistency point more than a step out is integrated without a gap", {
  lp <- log(SLICE_AXES$phi_pos)
  h <- lp[5L] - lp[4L]
  spec <- hyper_axis_spec("phi_pos", SLICE_AXES$phi_pos, log_scale = TRUE,
                          refinable = TRUE)
  mu <- SLICE_AXES$phi_pos[5L]
  pts <- .hyper_propose_consistency_points(spec, mu, sd = 1.6 * h * mu,
                                           lev = SLICE_AXES$phi_pos)
  expect_gt(max(log(pts)) - lp[5L], h)
  g <- rbind(SLICE_TENSOR, cbind(sigma = SLICE_AXES$sigma[3L], phi_pos = pts))
  ref <- c(rep("", 25L), rep("consistency_phi_pos", length(pts)))
  pc <- expect_row_conserved(g, ref)
  expect_gt(pc$tl$base_ext[5L], 0)
})

test_that("an evenly spaced extension leaves the base node no sliver past the edge", {
  # These levels put the midpoint towards the extension point one ulp away from
  # the base edge, the two being one number computed two ways.
  lev <- c(0.07, 0.19, 0.5, 1.3, 2.9)
  spec <- list(name = "sigma", log_scale = TRUE)
  m <- .hyper_axis_measure(lev, spec)
  m$spec <- spec
  p <- .hyper_propose_axis_extension(hyper_axis_spec("sigma", lev,
                                                     log_scale = TRUE),
                                     lev, "min")
  p <- p[p < min(lev)]
  tl <- .hyper_fibre_tiling(m, p, close_domain = TRUE)
  expect_true(all(tl$base_ext == 0))
  expect_equal(sum(tl$retained * diff(m$edges)) +
                 sum(tl$cell[tl$ok, "hi"] - tl$cell[tl$ok, "lo"]),
               diff(tl$region), tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# Rows extended past the base span on several axes tile their corner (#741)    #
# --------------------------------------------------------------------------- #

# Pointwise ownership of a refined grid on flat log axes, independent of the
# closed forms: each point of a lattice whose cell boundaries carry every edge
# of the tiling is handed to its owners, and its volume is summed into per-cell
# measures. Its base cell is the one whose box, widened past the base span on
# the side of each outermost node, contains it. Along every axis the point lies
# in the cell of its nearest node in the fibre through that base cell; it is the
# base cell's when every axis gives a base node, split equally among the slice
# cells otherwise. A point outside its fibres' regions belongs to no cell.
pointwise_measure <- function(g, ref, step) {
  U <- log(g)
  axes <- colnames(g)
  home <- .hyper_slice_home(ref, nrow(g))
  base <- which(!nzchar(home))
  lev <- lapply(stats::setNames(axes, axes),
                function(a) sort(unique(U[base, a])))
  mirror <- function(x) {
    n <- length(x)
    c(x[1L] - (x[2L] - x[1L]) / 2, (x[-n] + x[-1L]) / 2,
      x[n] + (x[n] - x[n - 1L]) / 2)
  }
  bedge <- lapply(lev, mirror)
  fibre <- function(c0, a) {
    in_row <- which(home == a)
    for (b in setdiff(axes, a)) {
      in_row <- in_row[abs(U[in_row, b] - lev[[b]][c0[[b]]]) < 1e-9]
    }
    x <- c(lev[[a]], U[in_row, a])
    id <- c(rep(0L, length(lev[[a]])), in_row)
    o <- order(x)
    x <- x[o]
    id <- id[o]
    ed <- mirror(x)
    eb <- range(bedge[[a]])
    ed[1L] <- min(ed[1L], eb[1L])
    ed[length(ed)] <- max(ed[length(ed)], eb[2L])
    s <- id > 0L
    list(id = id, edges = ed,
         region = c(min(eb[1L], ed[-length(ed)][s]), max(eb[2L], ed[-1L][s])))
  }
  level_of <- function(v, a) which.min(abs(lev[[a]] - v))
  span <- lapply(stats::setNames(axes, axes), function(a) {
    r <- range(bedge[[a]])
    for (i in which(home == a)) {
      c0 <- vapply(axes, function(b) level_of(U[i, b], b), integer(1))
      r <- range(r, fibre(c0, a)$region)
    }
    r
  })
  lattice <- lapply(stats::setNames(axes, axes), function(a) {
    o <- bedge[[a]][1L]
    s <- step[[a]]
    lo <- o + floor((span[[a]][1L] - o) / s + 1e-9) * s
    hi <- o + ceiling((span[[a]][2L] - o) / s - 1e-9) * s
    seq(lo + s / 2, hi - s / 2, by = s)
  })
  pts <- as.matrix(expand.grid(lattice))
  vol <- prod(unlist(step))
  base_key <- vapply(base, function(i) {
    paste(vapply(axes, function(a) level_of(U[i, a], a), integer(1)),
          collapse = ",")
  }, character(1))
  cache <- new.env()
  mass <- numeric(nrow(g))
  for (p in seq_len(nrow(pts))) {
    xp <- pts[p, ]
    c0 <- stats::setNames(integer(length(axes)), axes)
    for (k in seq_along(axes)) {
      eb <- range(bedge[[k]])
      c0[k] <- if (xp[k] > eb[2L]) length(lev[[k]])
               else if (xp[k] < eb[1L]) 1L
               else findInterval(xp[k], bedge[[k]], all.inside = TRUE)
    }
    owners <- integer(0)
    inside <- TRUE
    for (k in seq_along(axes)) {
      key <- paste(k, paste(c0, collapse = ","))
      fb <- cache[[key]]
      if (is.null(fb)) cache[[key]] <- fb <- fibre(c0, axes[k])
      if (xp[k] < fb$region[1L] || xp[k] > fb$region[2L]) {
        inside <- FALSE
        break
      }
      j <- findInterval(xp[k], fb$edges, all.inside = TRUE)
      if (fb$id[j] > 0L) owners <- c(owners, fb$id[j])
    }
    if (!inside) next
    if (length(owners)) {
      mass[owners] <- mass[owners] + vol / length(owners)
    } else {
      i0 <- base[base_key == paste(c0, collapse = ",")]
      mass[i0] <- mass[i0] + vol
    }
  }
  mass
}

# Flat log axes `axes` with, for each axis in `ext_axes`, one row through the
# corner base node extended `k` steps past the maximum base edge, followed by
# the slice cells `extra` tagged `extra_ref`.
corner_grid <- function(axes, ext_axes, k, extra = NULL, extra_ref = NULL) {
  T0 <- as.matrix(expand.grid(axes))
  top <- vapply(axes, max, numeric(1))
  rows <- lapply(ext_axes, function(a) {
    u <- log(axes[[a]])
    n <- length(u)
    v <- top
    v[[a]] <- exp(u[n] + k * (u[n] - u[n - 1L]))
    v
  })
  list(g = rbind(T0, do.call(rbind, rows), extra),
       ref = c(rep("", nrow(T0)), ext_axes, extra_ref))
}

expect_pointwise <- function(cg, specs, step) {
  w <- exp(.hyper_log_quad_weights(cg$g, specs, refining = cg$ref,
                                   absolute = TRUE))
  mass <- pointwise_measure(cg$g, cg$ref, step)
  expect_true(all(mass > 0))
  expect_equal(w, mass, tolerance = 1e-12)
  invisible(w)
}

test_that("one-, two- and three-axis extensions tile their region cell by cell", {
  quarter <- lapply(SLICE_AXES, function(x) diff(log(x))[1L] / 4)
  ls <- log(SLICE_AXES$sigma)
  lp <- log(SLICE_AXES$phi_pos)
  for (k in 1:3) {
    expect_pointwise(corner_grid(SLICE_AXES, "phi_pos", k), FLAT_SLICE_SPECS,
                     quarter)
    expect_pointwise(corner_grid(SLICE_AXES, c("phi_pos", "sigma"), k),
                     FLAT_SLICE_SPECS, quarter)
    # Densified cells inside the corner base cell on both axes, and a second
    # slice past the edge on the extended phi_pos row.
    extra <- rbind(
      c(SLICE_AXES$sigma[5L], exp((lp[4L] + lp[5L]) / 2)),
      c(exp((ls[4L] + ls[5L]) / 2), SLICE_AXES$phi_pos[5L]),
      c(SLICE_AXES$sigma[5L], exp(lp[5L] + (k + 1) * (lp[5L] - lp[4L]))))
    expect_pointwise(corner_grid(SLICE_AXES, c("phi_pos", "sigma"), k, extra,
                                 c("consistency_phi_pos", "sigma", "phi_pos")),
                     FLAT_SLICE_SPECS, quarter)
  }
  ax3 <- list(sigma   = exp(seq(log(0.1), log(3), length.out = 4)),
              phi_pos = exp(seq(log(1), log(60), length.out = 5)),
              tau     = exp(seq(log(0.5), log(8), length.out = 3)))
  sp3 <- lapply(names(ax3), function(a) list(name = a, log_scale = TRUE))
  half <- lapply(ax3, function(x) diff(log(x))[1L] / 2)
  for (k in 1:3) {
    expect_pointwise(corner_grid(ax3, names(ax3), k), sp3, half)
  }
})

test_that("rows extended along every face integrate the product of the reported spans", {
  ls <- log(SLICE_AXES$sigma)
  lp <- log(SLICE_AXES$phi_pos)
  hs <- ls[2L] - ls[1L]
  hp <- lp[2L] - lp[1L]
  for (k in 1:3) {
    g <- SLICE_TENSOR
    ref <- rep("", 25L)
    for (j in 1:5) {
      g <- rbind(g, c(SLICE_AXES$sigma[j], exp(lp[5L] + k * hp)),
                 c(exp(ls[5L] + k * hs), SLICE_AXES$phi_pos[j]),
                 c(exp(ls[1L] - k * hs), SLICE_AXES$phi_pos[j]))
      ref <- c(ref, "phi_pos", "sigma", "sigma")
    }
    w <- exp(.hyper_log_quad_weights(g, FLAT_SLICE_SPECS, refining = ref,
                                     absolute = TRUE))
    sup <- .hyper_grid_supports(g, FLAT_SLICE_SPECS, refining = ref)
    expect_equal(sum(w),
                 prod(vapply(sup, function(s) diff(log(s)), numeric(1))),
                 tolerance = 1e-12, label = sprintf("k = %d", k))
  }
})

test_that("the corner past two base edges goes to its nearest nodes", {
  ls <- log(SLICE_AXES$sigma)
  lp <- log(SLICE_AXES$phi_pos)
  box <- (ls[2L] - ls[1L]) * (lp[2L] - lp[1L])
  w_of <- function(k) {
    cg <- corner_grid(SLICE_AXES, c("phi_pos", "sigma"), k)
    exp(.hyper_log_quad_weights(cg$g, FLAT_SLICE_SPECS, refining = cg$ref,
                                absolute = TRUE))
  }
  # k = 1: the base node owns nothing past its edges, and in the one-step
  # corner square a slice cell is nearest along each axis, so each takes half.
  w <- w_of(1)
  expect_equal(w[25L] / box, 1, tolerance = 1e-12)
  expect_equal(w[26:27] / box, c(1.5, 1.5), tolerance = 1e-12)
  # k = 2: the base node owns half a step past each edge and with it the
  # half-step corner square. Each slice owns two steps along its own axis: over
  # the base cell, over the other axis's half step the base node owns, and
  # half of the two-step square the two slices share.
  w <- w_of(2)
  expect_equal(w[25L] / box, 1 + 0.5 + 0.5 + 0.25, tolerance = 1e-12)
  expect_equal(w[26:27] / box, rep(2 + 2 * 0.5 + 2 * 2 / 2, 2L),
               tolerance = 1e-12)
  expect_equal(sum(w) / box, 25 + 2 * 2.5 + 2.5^2, tolerance = 1e-12)
})

test_that("a slice inside the base span owns the strip past another axis's edge where it is nearest", {
  ls <- log(SLICE_AXES$sigma)
  lp <- log(SLICE_AXES$phi_pos)
  box <- (ls[2L] - ls[1L]) * (lp[2L] - lp[1L])
  # phi_pos extended two steps past its edge on the sigma[5] row, and a sigma
  # slice at the midpoint of sigma[4] and sigma[5] on the phi_pos[5] row.
  cg <- corner_grid(SLICE_AXES, "phi_pos", 2,
                    extra = rbind(c(exp((ls[4L] + ls[5L]) / 2),
                                    SLICE_AXES$phi_pos[5L])),
                    extra_ref = "sigma")
  w <- exp(.hyper_log_quad_weights(cg$g, FLAT_SLICE_SPECS, refining = cg$ref,
                                   absolute = TRUE))
  # The corner base cell keeps three quarters of its sigma cell and half a step
  # past the phi_pos edge.
  expect_equal(w[25L] / box, 0.75 * 1.5, tolerance = 1e-12)
  # The sigma slice owns a quarter step in each of the two base cells it
  # splits; in the corner base cell that quarter reaches over the whole
  # extended phi_pos interval of 3.5 steps, alone over the node's 1.5 and half
  # of the two the phi_pos slice owns.
  expect_equal(w[27L] / box, 0.25 + 0.25 * (1.5 + 2 / 2), tolerance = 1e-12)
  # The phi_pos slice owns two steps, over the node's 0.75 of the sigma cell
  # alone and half of the sigma slice's quarter.
  expect_equal(w[26L] / box, 2 * (0.75 + 0.25 / 2), tolerance = 1e-12)
  expect_equal(sum(w) / box, 25 + 2.5, tolerance = 1e-12)
  expect_pointwise(cg, FLAT_SLICE_SPECS,
                   lapply(SLICE_AXES, function(x) diff(log(x))[1L] / 4))
})
