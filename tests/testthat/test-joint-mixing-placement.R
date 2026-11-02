# A BYM2 field's mixing weight is a movable axis of the joint placement pass
# (`.joint_placement_families()`: the `mixing` row, placed on its logit), and
# integration = "auto" at four or more axes keeps the placed tensor while the
# declared tensor stays within the dense-grid cap (`.joint_ccd_engage()`).

test_that("auto keeps the tensor at four axes within the cap, the CCD past it", {
  cap <- .NL_DENSE_GRID_WARN_CELLS
  expect_false(.joint_ccd_engage("auto", 3L, 10))
  expect_false(.joint_ccd_engage("auto", 4L, cap))
  expect_true(.joint_ccd_engage("auto", 4L, cap + 1))
  expect_true(.joint_ccd_engage("auto", 5L))
  expect_true(.joint_ccd_engage("ccd", 3L, 10))
  expect_false(.joint_ccd_engage("grid", 6L, cap * 10))
})

test_that("every BYM2 mixing weight the caller left is movable", {
  bym <- list(type = "bym2")
  iid <- list(type = "iid")
  expect_identical(.joint_movable_mixing_axes(list(prior = bym)), "rho")
  expect_identical(.joint_movable_mixing_axes(list(prior = list(iid, bym))),
                   "b2.rho")
  stated <- list(bym, utils::modifyList(bym, list(rho_grid = c(0.3, 0.6))))
  expect_identical(.joint_movable_mixing_axes(list(prior = stated)), "b1.rho")
  expect_identical(.joint_movable_mixing_axes(list(prior = list(type = "icar"))),
                   character(0))
  expect_true(.joint_axis_refine_eligible("b1.rho"))
})

skip_on_cran()

.mix_fixture <- function(seed) {
  src <- readLines(testthat::test_path("test-nested-laplace-joint-ccd.R"))
  env <- new.env()
  eval(parse(text = src[11:54]), envir = env)
  sim <- env$.sim_joint_ccd(seed, N = 800L, n_s = 40L)
  G <- 25L
  set.seed(seed + 1L)
  o_occ <- sample.int(G, length(sim$responses$occ$y), replace = TRUE)
  u <- rnorm(G, 0, 0.5)
  p <- pmin(pmax(sim$responses$occ$y * 0.8 + 0.1, 1e-3), 1 - 1e-3)
  sim$responses$occ$y <- as.numeric(rbinom(length(o_occ), 1,
                                           plogis(qlogis(p) + u[o_occ])))
  o_pos <- sample.int(G, length(sim$responses$pos$y), replace = TRUE)
  sim$responses$pos$y <- sim$responses$pos$y + u[o_pos]
  sp <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
  bym <- env$.bym2_copy_block(sim$adj, NULL, NULL, sp)
  bym$sigma_grid <- NULL
  bym$rho_grid <- NULL
  list(responses = sim$responses,
       prior = list(bym, list(type = "iid", n_units = G,
                              obs_idx = list(o_occ, o_pos))))
}

.mix_fit <- function(fx, ...) suppressWarnings(suppressMessages(
  tulpa_nested_laplace_joint(fx$responses, fx$prior,
                             copy = list(arm = "pos", block = 1L),
                             control = c(list(diagnose_k = FALSE,
                                              progress = FALSE), list(...)))))

test_that("a mixing weight below its default ladder is placed, not cut", {
  skip_on_cran()
  fx <- .mix_fixture(11L)
  f <- .mix_fit(fx)
  expect_identical(f$integration, "grid")
  expect_true("b1.rho" %in% f$outer_grid_recenter_axes)
  # The default ladder's lowest node is 0.2, whose box ends near 0.12; the
  # placed axis reads the 2.5% bound a 13^4 tensor reads (0.051).
  expect_lt(f$theta_ci_lo[["b1.rho"]], 0.08)
})

test_that("the placed four-axis default reads a dense tensor's bounds", {
  skip_if_not_slow()
  fx <- .mix_fixture(11L)
  f <- .mix_fit(fx)
  mk <- function(a, b, n) exp(seq(log(a), log(b), length.out = n))
  pr <- fx$prior
  pr[[1L]]$sigma_grid <- mk(0.25, 1.3, 13L)
  pr[[1L]]$rho_grid <- stats::plogis(seq(-3.5, 7, length.out = 13L))
  pr[[2L]]$sigma_grid <- mk(0.2, 1.3, 13L)
  ref <- suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
    fx$responses, pr,
    copy = list(arm = "pos", block = 1L, alpha_grid = c(0, mk(0.6, 2.6, 13L), 3)),
    control = list(integration = "grid", diagnose_k = FALSE, progress = FALSE,
                   axis_refine = "none", var_of_means_consistency = FALSE,
                   auto_recenter = FALSE))))
  ax <- names(ref$theta_mean)
  # Reference (seed 11): b1.rho (0.051, 0.864); the CCD read (0.055, 0.706).
  expect_true(all(abs(f$theta_ci_hi[ax] / ref$theta_ci_hi[ax] - 1) < 0.1))
  expect_true(all(abs(f$theta_ci_lo[ax] - ref$theta_ci_lo[ax]) < 0.05))
})
