# The outer-grid dump / rebuild harness (gcol33/tulpa#322).
#
# The harness itself is in helper-outer-grid-dump.R. What is asserted here is the
# guarantee it rests on: a dump rebuilt with the fit's own weights returns the
# read the fit shipped. Everything an offline weight experiment concludes is
# attributable to the weights only while that holds, so it is checked on every
# node set the joint driver can leave behind -- a tensor grid whose cells carry
# their own mass, a central-composite design whose nodes carry a design weight,
# and a locally refined grid carrying both.

.ogd_cache <- new.env(parent = emptyenv())

.ogd_memo <- function(name, make) {
  if (!exists(name, envir = .ogd_cache, inherits = FALSE)) {
    assign(name, make(), envir = .ogd_cache)
  }
  get(name, envir = .ogd_cache, inherits = FALSE)
}

# Crossed iid random effects on one gaussian arm: each block contributes one
# outer sigma axis, so the axis count is the block count and the same simulation
# serves the d = 3 fits and the d = 4 one local CCD needs.
.ogd_sim <- function(sd_true, seed = 4242L, G = 30L, N = 600L) {
  set.seed(seed)
  grp <- lapply(seq_along(sd_true), function(k) sample.int(G, N, replace = TRUE))
  X <- cbind(1, stats::rnorm(N))
  eta <- as.numeric(X %*% c(0.2, 0.6))
  for (k in seq_along(sd_true)) {
    eta <- eta + stats::rnorm(G, 0, sd_true[k])[grp[[k]]]
  }
  list(y = eta + stats::rnorm(N, 0, 0.5), X = X, grp = grp,
       N = N, G = G, sd_true = sd_true)
}

.ogd_fit <- function(sim, control = list(), levels = 3L, spread = 3) {
  prior <- lapply(seq_along(sim$grp), function(k) {
    s <- sim$sd_true[k]
    list(type = "iid", obs_idx = list(sim$grp[[k]]), n_units = sim$G,
         sigma_grid = exp(seq(log(s / spread), log(s * spread),
                              length.out = levels)))
  })
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
                              family = "gaussian", phi = 0.25)),
    prior = prior,
    control = utils::modifyList(
      list(n_threads = 1L, diagnose_k = FALSE, max_iter = 100L, tol = 1e-8),
      control)))
}

# A tensor grid: every cell carries its own mass, no design weight anywhere.
.ogd_fit_grid <- function() {
  .ogd_memo("grid", function() {
    .ogd_fit(.ogd_sim(c(0.8, 0.5, 0.3)), control = list(integration = "grid"))
  })
}

# A global central-composite design: every node carries a design weight, so
# `dnode` is populated and the interval is read off a moment rule.
.ogd_fit_ccd <- function() {
  .ogd_memo("ccd", function() {
    .ogd_fit(.ogd_sim(c(0.8, 0.5, 0.3)), control = list(integration = "ccd"),
             levels = 7L)
  })
}

# A locally CCD-refined tensor grid: the one node set carrying both kinds at
# once. `skew_max = Inf` opens the engagement gate, which this fixture's own cell
# would otherwise be declined by -- the harness has to handle the mixed support
# whether or not this particular posterior earns the refinement.
.ogd_fit_local <- function() {
  .ogd_memo("local", function() {
    .ogd_fit(.ogd_sim(c(0.8, 0.5, 0.3, 0.2)),
             control = list(integration = "grid",
                            local_ccd = list(max_cells = 4L, skew_max = Inf)))
  })
}

# The largest absolute disagreement between two per-axis reads, over every
# reported number: 3 x n_axes of them.
.ogd_max_dev <- function(a, b) {
  max(abs(c(a$median - b$median, a$ci_lo - b$ci_lo, a$ci_hi - b$ci_hi)))
}

# --------------------------------------------------------------------------- #
# The round trip                                                              #
# --------------------------------------------------------------------------- #

test_that("a dump rebuilt with its own weights returns the read the fit shipped", {
  skip_on_cran()
  for (fit in list(.ogd_fit_grid(), .ogd_fit_ccd(), .ogd_fit_local())) {
    d <- outer_grid_dump(fit)
    # The regime is asserted before the agreement: a read that is all NA would
    # round-trip perfectly and mean nothing.
    expect_true(all(is.finite(d$reported$median)))
    expect_true(all(is.finite(d$reported$ci_lo)))
    expect_true(all(is.finite(d$reported$ci_hi)))

    rb <- outer_grid_rebuild(d)
    expect_named(rb$median, d$axis_names)
    # Measured: 0 on every axis of all three fits. The tolerance is floating
    # slack, not a fitted allowance -- the rebuild runs the same routine on the
    # same numbers, so anything above it means the harness and the fit have
    # stopped agreeing on what the read is.
    expect_lt(.ogd_max_dev(rb, d$reported), 1e-10)
    expect_equal(rb$median, d$reported$median, tolerance = 1e-12)
    expect_equal(rb$ci_lo,  d$reported$ci_lo,  tolerance = 1e-12)
    expect_equal(rb$ci_hi,  d$reported$ci_hi,  tolerance = 1e-12)
  }
})

test_that("the round trip survives the RDS", {
  skip_on_cran()
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  fit <- .ogd_fit_local()
  d <- outer_grid_dump(fit, file = f)
  expect_true(file.exists(f))
  d2 <- outer_grid_load(f)
  expect_identical(d2$support, d$support)
  expect_equal(d2$joint_grid, d$joint_grid)
  expect_equal(d2$log_marginal, d$log_marginal)
  expect_equal(d2$dnode, d$dnode)
  expect_identical(d2$weight_kind, d$weight_kind)
  expect_equal(.ogd_max_dev(outer_grid_rebuild(d2), d$reported), 0,
               tolerance = 1e-12)
})

test_that("the floor's per-axis view is the same read as the whole-grid one", {
  skip_on_cran()
  # `outer_grid_noise_floor()` reads one axis at a time and so mirrors the
  # per-axis cell mask `.nl_axis_quantiles()` applies. At stride 1 that read must
  # BE the whole-grid read; this is what fails if the two masks ever drift.
  for (fit in list(.ogd_fit_grid(), .ogd_fit_ccd(), .ogd_fit_local())) {
    d <- outer_grid_dump(fit)
    expect_lt(.ogd_max_dev(.ogd_read_at(d, d$weights, stride = 1L),
                           outer_grid_rebuild(d)), 1e-12)
  }
})

# --------------------------------------------------------------------------- #
# What the dump carries                                                       #
# --------------------------------------------------------------------------- #

test_that("the dump carries the grid state each node set actually left", {
  skip_on_cran()
  g <- outer_grid_dump(.ogd_fit_grid())
  expect_null(g$dnode)                                  # uniform tensor cells
  expect_identical(unique(g$weight_kind), "mass")
  expect_identical(g$support, "density")
  expect_null(g$axis_domains)
  expect_null(g$local_ccd_info)
  expect_length(g$log_marginal, nrow(g$joint_grid))
  expect_length(g$weights, nrow(g$joint_grid))
  expect_length(g$refining_axis, nrow(g$joint_grid))
  expect_identical(g$axis_tags, rep("log", ncol(g$joint_grid)),
                   ignore_attr = TRUE)

  c_ <- outer_grid_dump(.ogd_fit_ccd())
  # The design has to have engaged, or the dnode check below passes for the
  # wrong reason.
  expect_identical(c_$integration, "ccd")
  expect_length(c_$dnode, nrow(c_$joint_grid))
  expect_identical(unique(c_$weight_kind), "design")
  expect_identical(c_$support, "moment_rule")
  expect_identical(c_$axis_domains, rep("positive", ncol(c_$joint_grid)))
  # The dumped design weights are the ones the fit integrated with: they
  # reproduce its own weight vector through the engine's own weighting.
  expect_equal(outer_grid_weights(c_, c_$dnode), c_$weights, tolerance = 1e-12)

  l <- outer_grid_dump(.ogd_fit_local())
  expect_false(is.null(l$local_ccd_info))
  expect_gte(l$local_ccd_info$n_cells_refined, 1L)
  expect_setequal(unique(l$weight_kind), c("mass", "design"))
  expect_identical(l$support, "mixed")
  expect_length(l$dnode, nrow(l$joint_grid))
  expect_equal(outer_grid_weights(l, l$dnode), l$weights, tolerance = 1e-12)
  expect_equal(l$theta_interval_design_mass, l$local_ccd_info$design_mass)
})

test_that("a fit with no outer grid is refused, not silently dumped", {
  expect_error(outer_grid_dump(list()), "no outer hyperparameter grid")
  expect_error(outer_grid_dump("not a fit"), "must be a tulpa fit")
  tg <- matrix(1:6, 3L, 2L, dimnames = list(NULL, c("a", "b")))
  expect_error(outer_grid_dump(list(theta_grid = tg)), "log_marginal")
  expect_error(
    outer_grid_dump(list(theta_grid = tg, log_marginal = c(1, 2, 3))),
    "weights")
  expect_error(
    outer_grid_dump(list(theta_grid = matrix(numeric(0), 0L, 0L))),
    "no outer hyperparameter grid")
})

test_that("a rebuild refuses a weight vector of the wrong length", {
  skip_on_cran()
  d <- outer_grid_dump(.ogd_fit_grid())
  expect_error(outer_grid_rebuild(d, weights = c(1, 2, 3)), "cell")
})

# --------------------------------------------------------------------------- #
# The noise floor                                                             #
# --------------------------------------------------------------------------- #

# A one-axis dump with a known Gaussian outer log-marginal on `m` levels, built
# directly rather than fitted: the floor estimator's own behaviour is a property
# of the grid, so it is checked where the grid can be varied at no cost.
.ogd_fake_dump <- function(m, reach = 4) {
  v  <- seq(-reach, reach, length.out = m)
  lm <- -0.5 * v^2
  w  <- exp(lm - max(lm)); w <- w / sum(w)
  structure(list(
    joint_grid = matrix(v, ncol = 1L, dimnames = list(NULL, "a")),
    log_marginal = lm, dnode = NULL, weight_kind = rep("mass", m),
    weights = w, refining_axis = rep("", m), axis_names = "a",
    axis_tags = c(a = "identity"), axis_domains = NULL,
    support = "density", probs = OGD_PROBS),
    class = c("tulpa_outer_grid_dump", "list"))
}

test_that("the floor is the grid's own resolution, and falls as the grid resolves", {
  coarse <- outer_grid_noise_floor(.ogd_fake_dump(9L))
  fine   <- outer_grid_noise_floor(.ogd_fake_dump(81L))
  expect_gt(coarse$endpoints, 0)
  expect_gt(coarse$widths, 0)
  # Coarsening a grid that already resolves the read moves it far less than
  # coarsening one that does not, which is what makes the number a floor rather
  # than a constant.
  expect_lt(fine$endpoints, coarse$endpoints)
  expect_lt(fine$widths, coarse$widths)
  # Every stride / offset combination is scored, not just one arbitrary pairing.
  expect_identical(coarse$n_perturbations, 5L)
})

test_that("the floor bounds the read's own discretisation error", {
  # This grid's exact answer is known: the outer log-marginal is a standard
  # Gaussian, so the axis's 95% endpoints are qnorm(0.025) / qnorm(0.975) and the
  # read's distance from them is the discretisation error the floor is standing
  # in for. Measured at 9 / 15 / 21 / 41 / 81 levels, the error is 0.2646 /
  # 0.1263 / 0.0300 / 0.0113 / 0.0035 against a floor of 0.5519 / 0.2814 /
  # 0.1959 / 0.0562 / 0.0135: bounded at every resolution, and by a factor
  # between 2 and 7 rather than by orders of magnitude.
  exact <- stats::qnorm(c(0.025, 0.975))
  for (m in c(9L, 15L, 21L, 41L, 81L)) {
    d  <- .ogd_fake_dump(m)
    rb <- outer_grid_rebuild(d)
    err <- mean(abs(c(rb$ci_lo, rb$ci_hi) - exact))
    fl  <- outer_grid_noise_floor(d)$endpoints
    expect_lt(err, fl)
    expect_gt(err / fl, 0.1)
  }
})

test_that("a perturbation that should not matter sits below the floor", {
  skip_on_cran()
  d <- outer_grid_dump(.ogd_fit_local())
  fl <- outer_grid_noise_floor(d)
  expect_gt(fl$endpoints, 0)

  # Cell order carries no information: permuting the grid and everything aligned
  # with it is the same integration written down in a different order. The read
  # is invariant to it up to summation order, so it lands orders of magnitude
  # below the floor -- which is what the floor being a floor means.
  set.seed(7L)
  p <- sample.int(nrow(d$joint_grid))
  dp <- d
  dp$joint_grid    <- d$joint_grid[p, , drop = FALSE]
  dp$log_marginal  <- d$log_marginal[p]
  dp$weights       <- d$weights[p]
  dp$weight_kind   <- d$weight_kind[p]
  dp$refining_axis <- d$refining_axis[p]
  moved <- outer_grid_read_diff(outer_grid_rebuild(d), outer_grid_rebuild(dp))
  expect_lt(moved$endpoints, 1e-12)
  expect_lt(moved$endpoints, fl$endpoints)
})

test_that("a candidate weight rule is reported against the floor, not on its own", {
  skip_on_cran()
  d <- outer_grid_dump(.ogd_fit_local())

  # The rule that changes nothing: the fit's own design weights re-entered
  # through the engine's own weighting. Its difference is exactly zero and the
  # verdict says so.
  same <- outer_grid_weight_report(d, outer_grid_weights(d, d$dnode))
  expect_equal(same$diff$endpoints, 0, tolerance = 1e-12)
  expect_false(same$endpoints_above_floor)
  expect_false(same$widths_above_floor)

  # This fit's base grid holds three levels per axis, and one step of coarsening
  # moves its read by 0.105 -- so on THIS grid even a heavy re-weighting
  # (tempering the integrand, which concentrates the posterior) moves the
  # endpoints by 0.039 and is not resolved. Reporting that 0.039 on its own would
  # read as a difference; against the floor it reads as a grid too coarse to
  # tell, which is the whole point of carrying the floor.
  hot <- outer_grid_weight_report(
    d, outer_grid_weights(d, dnode = d$dnode, log_marginal = 3 * d$log_marginal),
    floor = same$floor)
  expect_gt(hot$diff$endpoints, 0)
  expect_false(hot$endpoints_above_floor)

  # Where the grid does resolve the read, the same rule is reported as the
  # difference it is: at 81 levels the floor is 0.0135 and tempering moves the
  # endpoints by 0.825.
  fine <- .ogd_fake_dump(81L)
  loud <- outer_grid_weight_report(
    fine, outer_grid_weights(fine, log_marginal = 3 * fine$log_marginal))
  expect_gt(loud$diff$endpoints, 0.5)
  expect_true(loud$endpoints_above_floor)
  expect_true(loud$widths_above_floor)
})
