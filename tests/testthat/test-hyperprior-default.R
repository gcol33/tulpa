# The default hyperpriors on the outer axes (gcol33/tulpa#730).
#
# Every default outer axis carries a proper, normalised density on its
# integration coordinate, so a grid's evidence is the evidence under that prior
# and does not depend on where the nodes were laid. The arbiters here are the
# densities' own integrals and anchors, written without the engine's code, and
# closed forms of each change of variables.

.hp_anchor <- tulpa:::.nl_scale_anchor()
.hp_lambda <- -log(.hp_anchor[2]) / .hp_anchor[1]

test_that("every scale density integrates to one on the log coordinate", {
  for (kind in c("sd", "variance", "precision")) {
    f <- function(u) exp(tulpa:::cpp_hyperprior_log_density(
      u, kind, "log", .hp_anchor[1], .hp_anchor[2]))
    expect_equal(stats::integrate(f, -40, 40, rel.tol = 1e-10,
                                  subdivisions = 2000L)$value,
                 1, tolerance = 1e-6, info = kind)
  }
})

test_that("the scale densities are the PC prior on sigma carried by hand", {
  x <- c(0.01, 0.3, 1, 4, 50)
  pc_sd <- function(s) log(.hp_lambda) - .hp_lambda * s
  expect_equal(tulpa:::cpp_hyperprior_log_density(x, "sd", "natural",
                                                  .hp_anchor[1], .hp_anchor[2]),
               pc_sd(x), tolerance = 1e-12)
  # variance v = sigma^2: dsigma/dv = 1 / (2 sqrt(v)).
  expect_equal(tulpa:::cpp_hyperprior_log_density(x, "variance", "natural",
                                                  .hp_anchor[1], .hp_anchor[2]),
               pc_sd(sqrt(x)) - log(2 * sqrt(x)), tolerance = 1e-12)
  # precision tau = sigma^-2: |dsigma/dtau| = tau^(-3/2) / 2.
  expect_equal(tulpa:::cpp_hyperprior_log_density(x, "precision", "natural",
                                                  .hp_anchor[1], .hp_anchor[2]),
               pc_sd(1 / sqrt(x)) - log(2) - 1.5 * log(x), tolerance = 1e-12)
  # The log coordinate adds log(theta).
  expect_equal(tulpa:::cpp_hyperprior_log_density(log(x), "precision", "log",
                                                  .hp_anchor[1], .hp_anchor[2]),
               pc_sd(1 / sqrt(x)) - log(2) - 1.5 * log(x) + log(x),
               tolerance = 1e-12)
})

test_that("the range PC prior is a density with its anchor in every dimension", {
  for (d in 1:3) for (anchor in list(c(0.2, 0.5), c(1.5, 0.05))) {
    f <- function(r) exp(tulpa:::cpp_hyperprior_log_density(
      r, "range", "natural", anchor[1], anchor[2], d))
    info <- sprintf("d = %d, U = %g, alpha = %g", d, anchor[1], anchor[2])
    expect_equal(stats::integrate(f, 0, Inf, rel.tol = 1e-10)$value, 1,
                 tolerance = 1e-6, info = info)
    expect_equal(stats::integrate(f, 0, anchor[1], rel.tol = 1e-10)$value,
                 anchor[2], tolerance = 1e-6, info = info)
  }
  # d = 2 is the form the SPDE path has always used.
  r <- c(0.05, 0.4, 2, 9)
  expect_equal(tulpa:::cpp_hyperprior_log_density(r, "range", "natural", 0.3, 0.5, 2),
               tulpa:::cpp_test_log_prior_range_pc(r, 0.3, 0.5), tolerance = 1e-12)
})

test_that("the LKJ normaliser makes the density integrate to one", {
  # d = 2: p(r) = (1 - r^2)^(eta - 1) / c_2.
  for (eta in c(1, 2, 3.5)) {
    f <- function(r) exp((eta - 1) * log1p(-r^2) - tulpa:::.lkj_log_normaliser(2L, eta))
    expect_equal(stats::integrate(f, -1, 1, rel.tol = 1e-10)$value, 1,
                 tolerance = 1e-6, info = eta)
  }
  # d = 3: midpoint rule over (r12, r13, r23), zero outside the positive
  # definite region. The integrand is continuous and vanishes on that boundary
  # at eta = 2, so the rule converges at the resolution taken here.
  n <- 160L
  g <- -1 + (seq_len(n) - 0.5) * 2 / n
  gr <- expand.grid(a = g, b = g, c = g)
  det3 <- 1 - gr$a^2 - gr$b^2 - gr$c^2 + 2 * gr$a * gr$b * gr$c
  eta <- 2
  mass <- sum(pmax(det3, 0)^(eta - 1)) * (2 / n)^3
  expect_equal(mass * exp(-tulpa:::.lkj_log_normaliser(3L, eta)), 1,
               tolerance = 5e-3)
})

test_that("each engine axis resolves to its density or a named decline", {
  hp <- tulpa:::.hp_axis_default
  expect_identical(hp("phi_arm", family = "gamma")$reason,
                   "dispersion_prior_unsourced")
  expect_true(is.function(hp("phi_arm", family = "gaussian")$fn))
  expect_identical(hp("phi_gp", list(type = "nngp"))$reason, "range_extent_unknown")
  expect_identical(hp("L21", list(type = "mcar"))$reason, "logchol_design_measure")
  expect_identical(hp("zeta", list(type = "icar"))$reason, "unclassified_axis")
  expect_true(isTRUE(hp("alpha")$spec))

  # The range anchor reads the coordinates' bounding-box diagonal.
  co <- cbind(c(0, 3, 1), c(0, 4, 2))
  fn <- hp("phi_gp", list(type = "nngp", coords = co))$fn
  U <- tulpa:::.nl_hyperprior("range_extent_fraction") * 5
  a <- tulpa:::.nl_hyperprior("range_alpha")
  expect_equal(fn(c(0.4, 2)),
               tulpa:::cpp_hyperprior_log_density(log(c(0.4, 2)), "range", "log",
                                                  U, a, 2),
               tolerance = 1e-12)

  # Bounded axes: the uniform on the domain, the AR1 Beta normalised on rho.
  expect_equal(hp("rho", list(type = "bym2"))$fn(c(0.2, 0.9)), c(0, 0))
  for (ab in list(c(1, 1), c(10, 1), c(2, 5))) {
    f <- hp("rho", list(type = "ar1", rho_prior = prior_beta(ab[1], ab[2])))$fn
    expect_equal(stats::integrate(function(x) exp(f(x)), -1, 1,
                                  rel.tol = 1e-10)$value, 1, tolerance = 1e-6)
  }
})

test_that("the SPDE hyperprior is the PC density carried to log range, log sigma", {
  sp <- list(prior_range = c(0.3, 0.5), prior_sigma = c(1, 0.01))
  r <- c(0.1, 0.5, 2); s <- c(0.2, 1, 3)
  lr <- -0.3 * log(0.5)
  ls <- -log(0.01)
  ref <- log(lr) - 2 * log(r) - lr / r + log(r) + log(ls) - ls * s + log(s)
  expect_equal(tulpa:::.spde_log_hyperprior(r, s, sp), ref, tolerance = 1e-12)
})

test_that("the negative-binomial size prior is R-INLA's pc.mgamma density", {
  lambda <- tulpa:::.nl_hyperprior("nb_size_lambda")
  # inla.pc.dgamma() on the overdispersion a = 1 / size, written out.
  pc_a <- function(a) {
    z <- 1 / a
    d <- sqrt(2 * (log(z) - digamma(z)))
    -lambda * d - log(d) + log(lambda) - 2 * log(a) + log(trigamma(z) - a)
  }
  size <- c(0.3, 2, 20, 400)
  expect_equal(tulpa:::cpp_hyperprior_log_density(size, "nb_size", "natural",
                                                  1, 0.5, lambda = lambda),
               pc_a(1 / size) - 2 * log(size), tolerance = 1e-8)
  # Proper on log(size), across the switch to the asymptotic series at 1e4.
  f <- function(u) exp(tulpa:::cpp_hyperprior_log_density(
    u, "nb_size", "log", 1, 0.5, lambda = lambda))
  expect_equal(stats::integrate(f, -30, 60, rel.tol = 1e-9,
                                subdivisions = 4000L)$value, 1, tolerance = 1e-5)
  u_switch <- log(1e4)
  expect_equal(f(u_switch - 1e-9), f(u_switch + 1e-9), tolerance = 1e-6)
  # Bound to the size families only; other dispersions decline by name.
  expect_true(is.function(tulpa:::.hp_axis_default("phi_c",
                                                   family = "neg_binomial_2")$fn))
  expect_identical(tulpa:::.hp_axis_default("phi_c", family = "beta")$reason,
                   "dispersion_prior_unsourced")
})
