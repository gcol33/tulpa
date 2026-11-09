
test_that("the PC range prior integrates to one", {
  load_cpp_fixture("pc_prior_equiv")
  U <- 0.2; alpha <- 0.05
  dens <- function(r) exp(cpp_test_log_prior_range_pc(r, U, alpha))
  total <- stats::integrate(dens, lower = 0, upper = Inf,
                            rel.tol = 1e-10)$value
  expect_equal(total, 1, tolerance = 1e-6)
})

test_that("the C++ range prior matches its closed form", {
  # p(r) = lambda r^-2 exp(-lambda / r), lambda = -U log(alpha), written out here
  # rather than read from an engine helper.
  load_cpp_fixture("pc_prior_equiv")
  U <- 0.3; alpha <- 0.5
  r <- c(0.05, 0.15, 0.3, 0.9, 2.5, 7.0)
  lambda <- -U * log(alpha)
  expect_equal(cpp_test_log_prior_range_pc(r, U, alpha),
               log(lambda) - 2 * log(r) - lambda / r, tolerance = 1e-12)
})

test_that("the PC range prior prefers the truth over the old Uniform mean", {
  # The #144 failure: with coords on the unit square and a truth of phi = 0.25,
  # a Uniform(0.01, 10) on phi carries mean ~5 and a weakly informative binary
  # likelihood let the posterior sit there. Under an anchor placed for unit-square
  # coordinates the density at the truth has to dominate the density out at 4.
  load_cpp_fixture("pc_prior_equiv")
  U <- 0.1; alpha <- 0.05
  at_truth <- cpp_test_log_prior_range_pc(0.25, U, alpha)
  at_rail  <- cpp_test_log_prior_range_pc(4.00, U, alpha)
  expect_gt(at_truth, at_rail)
})

test_that("a GP/SVC NNGP block requires its range anchors", {
  # ModelData ships the anchors unset (-1). The engine refuses rather than
  # substituting a default, because a silent default is a prior -- which is how
  # phi came to sit at the Uniform mean in the first place.
  load_cpp_fixture("pc_prior_equiv")
  expect_error(cpp_test_gp_layout_anchor_check(-1, -1),
               "PC prior on the range")
  expect_error(cpp_test_gp_layout_anchor_check(0.1, -1),
               "PC prior on the range")
  expect_error(cpp_test_gp_layout_anchor_check(0.1, 1.0),
               "PC prior on the range")
  expect_error(cpp_test_gp_layout_anchor_check(0, 0.05),
               "PC prior on the range")
  expect_silent(cpp_test_gp_layout_anchor_check(0.1, 0.05))
})

test_that("the PC scale density matches its definition on sigma", {
  load_cpp_fixture("pc_prior_equiv")
  x <- c(-4, 0, 3.2)
  sigma <- exp(x / 2)
  for (anchor in list(c(1, 0.01), c(2.5, 0.2))) {
    rate <- -log(anchor[2]) / anchor[1]
    twin <- dexp(sigma, rate, log = TRUE) - log(2) + 0.5 * x
    expect_equal(cpp_test_log_prior_log_sigma2_pc(x, anchor[1], anchor[2]),
                 twin, tolerance = 1e-12)
  }
})

skip_on_cran_budget()
# PC range prior (gcol33/tulpa#144) and the bounded-parameter map.

test_that("the PC range prior honours P(range < U) = alpha", {
  skip_on_cran_budget()
  # The defining contract of the anchor, and the one the removed
  # log_prior_phi_pc violated: it set rate = -log(alpha)/U where its own
  # derivation, exp(-rate/U) = alpha, gives rate = -U * log(alpha). Integrating
  # the density is what tells the two apart.
  load_cpp_fixture("pc_prior_equiv")
  for (anchor in list(c(0.1, 0.05), c(0.3, 0.5), c(2.0, 0.01), c(0.75, 0.25))) {
    U <- anchor[1]; alpha <- anchor[2]
    dens <- function(r) exp(cpp_test_log_prior_range_pc(r, U, alpha))
    p_below <- stats::integrate(dens, lower = 0, upper = U,
                                rel.tol = 1e-10)$value
    expect_equal(p_below, alpha, tolerance = 1e-6,
                 info = sprintf("U = %g, alpha = %g", U, alpha))
  }
})

test_that("the log-range entry point agrees with the range entry point", {
  skip_on_cran_budget()
  load_cpp_fixture("pc_prior_equiv")
  U <- 0.4; alpha <- 0.1
  r <- c(0.02, 0.1, 0.5, 1.0, 3.0, 10.0)
  expect_equal(
    cpp_test_log_prior_range_pc_at_log(log(r), U, alpha),
    cpp_test_log_prior_range_pc(r, U, alpha),
    tolerance = 1e-12
  )
})

test_that("the bounded map stays inside its interval and its Jacobian is right", {
  skip_on_cran_budget()
  load_cpp_fixture("pc_prior_equiv")
  lower <- 0.01; upper <- 10.0

  # The map never leaves [lower, upper]. Past |u| ~ 37 the logistic saturates in
  # double precision and phi lands exactly on an endpoint, where the Jacobian is
  # -Inf. That is the density correctly vanishing at the edge of its support,
  # not the #144 wall: there -INFINITY was returned across a reachable region
  # the density was perfectly happy in.
  u_wide <- c(-40, -8, -1, 0, 0.5, 3, 12, 40)
  phi_wide <- cpp_test_bounded_from_logit(u_wide, lower, upper)
  expect_true(all(phi_wide >= lower))
  expect_true(all(phi_wide <= upper))
  expect_true(all(is.finite(phi_wide)))

  # Strictly interior wherever the sampler realistically lives.
  u <- c(-8, -1, 0, 0.5, 3, 12)
  phi <- cpp_test_bounded_from_logit(u, lower, upper)
  expect_true(all(phi > lower))
  expect_true(all(phi < upper))

  # log|dphi/du| against a numerical derivative of the map itself.
  h <- 1e-6
  u_fd <- c(-2, -0.5, 0, 0.7, 2.5)
  num <- (cpp_test_bounded_from_logit(u_fd + h, lower, upper) -
          cpp_test_bounded_from_logit(u_fd - h, lower, upper)) / (2 * h)
  ana <- exp(cpp_test_log_jacobian_bounded(
    cpp_test_bounded_from_logit(u_fd, lower, upper), lower, upper))
  expect_equal(ana, num, tolerance = 1e-6)
})

test_that("the PC scale prior is normalized on the sampled log variance", {
  skip_on_cran_budget()
  load_cpp_fixture("pc_prior_equiv")
  for (anchor in list(c(1, 0.01), c(2.5, 0.2))) {
    dens <- function(x) {
      exp(cpp_test_log_prior_log_sigma2_pc(x, anchor[1], anchor[2]))
    }
    total <- stats::integrate(dens, lower = -80, upper = 80,
                              rel.tol = 1e-10)$value
    expect_equal(total, 1, tolerance = 1e-6)
  }
})
