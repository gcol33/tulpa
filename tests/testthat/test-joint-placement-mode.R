# gcol33/tulpa#919: a railed grid's heaviest cell is its boundary node, so the
# outer mode a placement lays an axis around is FOUND (`.joint_placement_mode()`)
# rather than read off the grid it railed on. No inner solve runs here: the
# outer log-posterior is an analytic quadratic in the placement coordinate, on
# which the mode and the curvature the mode-find reaches are known exactly.

# The copy scale's atom (alpha = 0) carries `atom_lp` plus the (sigma, phi_pos)
# marginal of the same quadratic, so the surface a search held at the atom moves
# along is known too.
.pm_quadratic <- function(mode_u, cov_u, atom_lp = -1e6) {
  P  <- solve(cov_u)
  Pa <- solve(cov_u[c(1L, 3L), c(1L, 3L)])
  function(theta) {
    u  <- cbind(log(theta[, "sigma"]), log(theta[, "alpha"]),
                log(theta[, "phi_pos"]))
    dz <- sweep(u, 2L, mode_u)
    at <- theta[, "alpha"] == 0
    lp <- numeric(nrow(u))
    lp[!at] <- -0.5 * rowSums((dz[!at, , drop = FALSE] %*% P) *
                                dz[!at, , drop = FALSE])
    da <- dz[at, c(1L, 3L), drop = FALSE]
    lp[at] <- atom_lp - 0.5 * rowSums((da %*% Pa) * da)
    lp
  }
}

# The shared25 default axes: field SD on [0.1, 3], the copy scale's atom plus a
# nine-node slab, and the Beta precision on [1, 60].
.pm_fit <- function(lp_fn,
                    sig = exp(seq(log(0.1), log(3), length.out = 5L)),
                    alp = c(0, exp(seq(log(0.1), log(3), length.out = 9L))),
                    phi = exp(seq(log(1), log(60), length.out = 4L))) {
  tg <- as.matrix(expand.grid(sigma = sig, alpha = alp, phi_pos = phi))
  lm <- lp_fn(tg)
  w  <- exp(lm - max(lm))
  list(theta_grid = tg, log_marginal = lm, weights = w / sum(w),
       prior = list(type = "icar"))
}

.pm_mode <- c(log(3.1), log(0.234), log(3.9))
.pm_cov  <- local({
  s <- c(0.03, 0.05, 0.005)
  R <- matrix(c(1, -0.9, 0, -0.9, 1, 0, 0, 0, 1), 3L)
  diag(s) %*% R %*% diag(s)
})

test_that("placement finds an outer mode past the grid's own ceiling", {
  lp  <- .pm_quadratic(.pm_mode, .pm_cov)
  res <- .pm_fit(lp)
  # The grid rails: its heaviest cell sits on the top field-SD node.
  expect_equal(unname(res$theta_grid[which.max(res$weights), "sigma"]), 3)

  pm <- .joint_placement_mode(res, lp)
  expect_identical(pm$status, "ok")
  expect_identical(pm$names, c("sigma", "alpha", "phi_pos"))
  # The copy scale is searched on its log continuum.
  expect_identical(pm$tags, c("log", "log", "log"))
  expect_lt(max(abs(pm$mode_u - .pm_mode)), 1e-4)
  expect_lt(max(abs(pm$cov_u - .pm_cov)) / max(abs(.pm_cov)), 1e-3)

  # The axis the single-block rescue lays from it brackets the mode, where the
  # argmax node read it off the ceiling.
  rc <- .nl_axis_recenter_from_fit_full(pm$mode_u, pm$cov_u, pm$tags,
                                        pm$names, "sigma")
  expect_lt(min(rc$nodes), 3.1)
  expect_gt(max(rc$nodes), 3.1)
})

test_that("a collapsed field SD is placed when another axis still spreads", {
  # The dense arm of #919: the weight sits on one field-SD node while the
  # Beta-precision axis carries spread across two of its nodes. The weighted
  # covariance then had zero variance on sigma alone, and the placement
  # declined for want of a curvature.
  cov_u <- .pm_cov
  cov_u[3L, 3L] <- 1.2^2
  lp  <- .pm_quadratic(c(log(3.1), log(0.234), log(2)), cov_u)
  res <- .pm_fit(lp)
  res$pareto_k_regime <- "collapsed_edge"
  w_phi <- tapply(res$weights, res$theta_grid[, "phi_pos"], sum)
  expect_gt(sort(w_phi, decreasing = TRUE)[2L], 0.05)

  out <- .joint_attach_placement(res, lp)
  expect_null(out$outer_mode_declined)
  expect_identical(out$outer_mode_status, "ok")
  expect_equal(sqrt(out$outer_mode_cov_u[1L, 1L]), 0.03, tolerance = 1e-3)
  expect_lt(abs(out$outer_mode_u[1L] - log(3.1)), 1e-4)
})

test_that("a copy scale whose heaviest cell is its atom is held there", {
  # All the weight on alpha = 0: the "no coupling" model. The Newton step runs
  # over the continuum axes only.
  lp  <- .pm_quadratic(c(log(0.5), log(0.234), log(3.9)), .pm_cov,
                       atom_lp = 50)
  res <- .pm_fit(lp)
  expect_identical(unname(res$theta_grid[which.max(res$weights), "alpha"]), 0)

  pm <- .joint_placement_mode(res, lp)
  expect_identical(pm$status, "ok")
  expect_identical(pm$tags[[2L]], "identity")
  expect_identical(pm$mode_u[[2L]], 0)
  expect_true(all(pm$cov_u[2L, ] == 0) && all(pm$cov_u[, 2L] == 0))
  expect_lt(abs(pm$mode_u[[1L]] - log(0.5)), 1e-4)
  expect_lt(abs(pm$mode_u[[3L]] - log(3.9)), 1e-4)
})

test_that("a grid no rescue would place carries no placement mode", {
  lp  <- .pm_quadratic(c(log(0.5), log(0.234), log(3.9)), .pm_cov)
  res <- .pm_fit(lp)
  res$pareto_k_regime <- "spread"
  calls <- 0L
  out <- .joint_attach_placement(res, function(theta) {
    calls <<- calls + 1L
    lp(theta)
  })
  expect_identical(calls, 0L)
  expect_null(out$outer_mode_u)
})

test_that("a default axis left railed after placement is named in a warning", {
  res <- list(outer_grid_railed_axes = "sigma:upper",
              outer_grid_axis_declined = c(sigma = "attempts_exhausted"))
  expect_warning(.nl_warn_unplaced_rail(res, "f()"),
                 "`sigma` \\(upper edge: attempts_exhausted\\)")

  # The caller's own pin is the caller's statement: recorded, not warned.
  for (why in .NL_RAIL_CALLER_HOLDS) {
    res$outer_grid_axis_declined <- c(sigma = why)
    expect_silent(.nl_warn_unplaced_rail(res, "f()"))
  }
  expect_silent(.nl_warn_unplaced_rail(
    list(outer_grid_railed_axes = character(0)), "f()"))
})
