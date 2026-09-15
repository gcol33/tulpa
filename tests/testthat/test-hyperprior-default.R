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

test_that("the range prior's dimension is the one the coordinates span", {
  set.seed(8)
  x <- runif(40)
  ex <- function(co) tulpa:::.hp_block_extent(list(coords = co))
  expect_identical(ex(cbind(x))$dim, 1L)
  # A constant or collinear column adds no direction, and leaves the extent
  # bit-identical, so a field over the same points carries the same prior.
  expect_identical(ex(cbind(x, 0)), ex(cbind(x)))
  expect_identical(ex(cbind(x, 1e6))$dim, 1L)
  expect_identical(ex(cbind(x, 2 * x))$dim, 1L)
  expect_identical(ex(cbind(x, runif(40)))$dim, 2L)
  expect_identical(ex(cbind(x, runif(40), 3.5))$dim, 2L)
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
  expect_identical(hp("phi_arm", family = "inverse_gaussian")$reason,
                   "dispersion_prior_unsourced")
  expect_true(is.function(hp("phi_arm", family = "gaussian")$fn))
  expect_identical(hp("phi_gp", list(type = "nngp"))$reason, "range_extent_unknown")
  expect_identical(hp("L21", list(type = "mcar"))$group, "logchol")
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
  expect_identical(tulpa:::.hp_axis_default("phi_c", family = "t")$reason,
                   "dispersion_prior_unsourced")
})

test_that("a gamma shape and a beta precision carry R-INLA's exponential defaults (#736)", {
  # loggamma(1, 0.01) on the gamma family's precision parameter and
  # loggamma(1, 0.1) on the beta family's, i.e. Exponential(rate) on phi,
  # carried to log phi where the axis integrates.
  for (fam in c("gamma", "gamma_inverse", "beta")) {
    rate <- if (startsWith(fam, "gamma")) 0.01 else 0.1
    fn <- tulpa:::.hp_axis_default("phi_c", family = fam)$fn
    expect_true(is.function(fn), info = fam)
    x <- c(0.5, 3, 40, 900)
    expect_equal(fn(x), stats::dexp(x, rate, log = TRUE) + log(x),
                 tolerance = 1e-12, info = fam)
    f <- function(u) exp(fn(exp(u)))
    expect_equal(stats::integrate(f, -40, 12, rel.tol = 1e-10)$value, 1,
                 tolerance = 1e-6, info = fam)
  }
})

# --------------------------------------------------------------------------- #
# Free-covariance blocks: one prior and one measure per block (#735)          #
# --------------------------------------------------------------------------- #

test_that("the two-field default grid carries PC x PC x LKJ on its own coordinates", {
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  d <- tulpa:::.hp_logchol_design(g)
  expect_identical(d$design, "sd_rho")
  lp <- tulpa:::.hp_logchol_log_density(g)$lp
  # The same PC + LKJ prior re_cov_pc_lkj_prior() puts on the log-Cholesky
  # coordinates, carried to (log sigma_1, log sigma_2, rho) by the Jacobian of
  # L11 = log s1, L21 = rho s2, L22 = log s2 + log(1 - rho^2) / 2, whose
  # determinant is s2 / (1 - rho^2).
  f <- tulpa:::.re_cov_block_logprior(2L, TRUE, tulpa:::.nl_scale_anchor(),
                                      tulpa:::.nl_hyperprior("lkj_eta"))
  s2 <- exp(d$coords[, 2L]); rho <- d$coords[, 3L]
  expect_equal(lp, apply(g, 1L, f) + log(s2 / (1 - rho^2)), tolerance = 1e-12)

  # A proper prior: on a tensor wide enough to hold it, the density times the
  # absolute cell measure sums to one.
  ls <- seq(-9, 3, length.out = 70L); r <- seq(-0.995, 0.995, length.out = 81L)
  gg <- expand.grid(a = ls, b = ls, rho = r)
  M <- cbind(L11 = gg$a, L21 = gg$rho * exp(gg$b),
             L22 = gg$b + 0.5 * log1p(-gg$rho^2))
  mass <- sum(exp(tulpa:::.hp_logchol_log_density(M)$lp +
                  tulpa:::.hyper_logchol_log_measure(M, absolute = TRUE)))
  expect_equal(mass, 1, tolerance = 2e-3)
})

test_that("a log-Cholesky tensor is measured by its own column widths", {
  G1 <- matrix(seq(-9, 3, length.out = 200L), ncol = 1L,
               dimnames = list(NULL, "L11"))
  expect_identical(tulpa:::.hp_logchol_design(G1)$design, "logchol")
  mass <- sum(exp(tulpa:::.hp_logchol_log_density(G1)$lp +
                  tulpa:::.hyper_logchol_log_measure(G1, absolute = TRUE)))
  expect_equal(mass, 1, tolerance = 2e-3)

  G <- as.matrix(expand.grid(L11 = c(-1, 0, 1), L21 = c(-0.5, 0.5),
                             L22 = c(-1, 0.5)))
  expect_identical(tulpa:::.hp_logchol_design(G)$design, "logchol")
  w <- function(x) { e <- c(x[1] - diff(x)[1] / 2, (x[-1] + x[-length(x)]) / 2,
                            x[length(x)] + diff(x)[length(x) - 1] / 2); diff(e) }
  ref <- log(w(c(-1, 0, 1)))[match(G[, 1], c(-1, 0, 1))] +
         log(w(c(-0.5, 0.5)))[match(G[, 2], c(-0.5, 0.5))] +
         log(w(c(-1, 0.5)))[match(G[, 3], c(-1, 0.5))]
  expect_equal(tulpa:::.hyper_logchol_log_measure(G, absolute = TRUE), ref,
               tolerance = 1e-12)
})

test_that("a free-covariance block the grid does not measure declines by name", {
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  pinned <- g
  pinned[, "L22"] <- 0
  expect_identical(tulpa:::.hp_logchol_log_density(pinned)$reason,
                   "logchol_partial_block")
  set.seed(735)
  scattered <- cbind(L11 = rnorm(12), L21 = rnorm(12), L22 = rnorm(12))
  expect_identical(tulpa:::.hp_logchol_log_density(scattered)$reason,
                   "logchol_design_measure")
  tg <- scattered; colnames(tg) <- paste0("b1.", colnames(tg))
  sp <- tulpa:::.joint_axis_specs_from_grid(tg)
  expect_true(all(is.na(tulpa:::.hyper_log_quad_weights(tg, sp, absolute = TRUE))))

  # Through the joint collector, a measured block folds once over all its axes.
  tg <- g; colnames(tg) <- paste0("b1.", colnames(g))
  hp <- tulpa:::.joint_hyperprior(tg, list(list(type = "mcar")),
                                  axes = tulpa:::.hp_integrated_axes(tg))
  expect_setequal(hp$axes, colnames(tg))
  expect_length(hp$declined, 0L)
  expect_equal(hp$lp, tulpa:::.hp_logchol_log_density(g)$lp)
  sp <- tulpa:::.joint_axis_specs_from_grid(tg)
  expect_equal(tulpa:::.hyper_log_quad_weights(tg, sp, absolute = TRUE),
               tulpa:::.hyper_logchol_log_measure(g, absolute = TRUE))
})

test_that("a batch of cells reads the prior on the fit's axes, not on its own spread", {
  # Refinement slices, CCD points and importance draws are evaluated as batches
  # that share coordinates on axes the fit integrates. Each cell's prior is a
  # property of the cell and the fit, so a batch must read what the whole grid
  # reads at those rows (gcol33/tulpa#760).
  tg <- as.matrix(expand.grid(sigma = c(0.3, 0.7, 1.5), alpha = c(0, 0.5, 1.2),
                              phi_pos = c(5, 20, 40), KEEP.OUT.ATTRS = FALSE))
  blocks <- list(list(type = "icar"))
  fam <- c(occ = "bernoulli", pos = "beta")
  axes <- tulpa:::.hp_integrated_axes(tg)
  whole <- tulpa:::.joint_hyperprior(tg, blocks, fam, axes = axes)
  expect_setequal(whole$axes, c("sigma", "phi_pos"))

  slice <- which(tg[, "sigma"] == 0.7 & tg[, "phi_pos"] == 20)
  one <- which(tg[, "sigma"] == 1.5 & tg[, "alpha"] == 0.5 & tg[, "phi_pos"] == 40)
  for (rows in list(slice, one)) {
    part <- tulpa:::.joint_hyperprior(tg[rows, , drop = FALSE], blocks, fam,
                                      axes = axes)
    expect_equal(part$lp, whole$lp[rows])
    expect_setequal(part$axes, whole$axes)
  }

  # Reading the axes off the batch drops every density a batch holds constant,
  # which is exactly the difference the fix removes.
  own <- tulpa:::.joint_hyperprior(tg[slice, , drop = FALSE], blocks, fam,
                                   axes = tulpa:::.hp_integrated_axes(tg[slice, , drop = FALSE]))
  expect_length(own$axes, 0L)
  expect_equal(own$lp, rep(0, length(slice)))
  expect_true(all(whole$lp[slice] != 0))
})

# --------------------------------------------------------------------------- #
# The `hyperprior` choice at the two nested-Laplace front doors               #
# --------------------------------------------------------------------------- #

test_that("hyperprior = \"flat\" keeps only a density the call states", {
  flat <- function(bare, block = list(), family = NULL)
    tulpa:::.hp_axis_prior(bare, block, family, "flat")
  for (a in c("sigma", "sigma2", "tau", "range")) {
    expect_identical(flat(a)$reason, "flat_hyperprior", info = a)
  }
  expect_identical(flat("phi_pos", family = "gaussian")$reason, "flat_hyperprior")
  expect_identical(flat("rho", list(type = "bym2"))$reason, "flat_hyperprior")
  expect_identical(flat("rho", list(type = "ar1"))$reason, "flat_hyperprior")
  expect_identical(flat("L21", list(type = "mcar"))$reason, "flat_hyperprior")
  # The copy scale's slab is declared on its axis spec under either choice.
  expect_true(isTRUE(flat("alpha")$spec))
  # A block that states a density keeps it.
  x <- c(0.3, 0.9, 2.5)
  ar1 <- list(type = "ar1", rho_prior = list(alpha = 2, beta = 3))
  expect_identical(flat("rho", ar1)$fn(x - 1),
                   tulpa:::.hp_axis_default("rho", ar1)$fn(x - 1))
  spde <- list(type = "spde", prior_range = c(0.5, 0.1), prior_sigma = c(2, 0.05))
  for (a in c("range", "sigma")) {
    expect_identical(flat(a, spde)$fn(x), tulpa:::.hp_axis_default(a, spde)$fn(x),
                     info = a)
  }
  expect_identical(tulpa:::.hp_axis_prior("sigma", hyperprior = "proper")$fn(x),
                   tulpa:::.hp_axis_default("sigma")$fn(x))
  expect_error(tulpa:::.hp_choice("pc_lkj"), "hyperprior")
})

test_that("a flat joint fit is the proper fit with the default density taken out", {
  skip_on_cran()
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))
  fp <- ogd_fixture_fit(sim, 4L, hyperprior = "proper")
  ff <- ogd_fixture_fit(sim, 4L, hyperprior = "flat")
  expect_identical(ff$theta_grid, fp$theta_grid)
  expect_true(all(fp$log_hyperprior != 0))
  expect_true(all(ff$log_hyperprior == 0))
  expect_equal(ff$log_marginal, fp$log_marginal - fp$log_hyperprior,
               tolerance = 1e-12)
  expect_setequal(names(ff$log_hyperprior_declined), colnames(ff$theta_grid))
  expect_true(all(unlist(ff$log_hyperprior_declined) == "flat_hyperprior"))
  expect_true(is.na(ff$log_evidence))
  expect_identical(ff$log_evidence_declined, "improper_hyperprior")
  expect_true(is.finite(fp$log_evidence))
  expect_error(ogd_fixture_fit(sim, 4L, hyperprior = "pc_lkj"), "should be one of")
})

test_that("both doors apply a stated density under hyperprior = \"flat\"", {
  skip_on_cran()
  set.seed(11)
  S <- 30L
  nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- lengths(nb)
  site <- rep(seq_len(S), each = 5L)
  X <- cbind(1, rnorm(length(site)))
  field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
  y <- rbinom(length(site), 1L, plogis(-0.2 + 0.6 * X[, 2] + field[site]))
  adj <- list(n_spatial_units = S, adj_row_ptr = c(0L, cumsum(nn)),
              adj_col_idx = unlist(nb) - 1L, n_neighbors = nn)

  # Single-block door on a pinned precision axis: the flat fit folds nothing,
  # and its marginal is the proper fit's less the PC density on tau.
  icar <- c(list(type = "icar", spatial_idx = site,
                 tau_grid = c(0.5, 1, 2, 4, 8)), adj)
  np <- tulpa_nested_laplace(y, rep(1L, length(y)), X, prior = icar,
                             family = "binomial",
                             control = list(n_threads = 1L, diagnose_k = FALSE))
  nf <- tulpa_nested_laplace(y, rep(1L, length(y)), X, prior = icar,
                             family = "binomial", hyperprior = "flat",
                             control = list(n_threads = 1L, diagnose_k = FALSE))
  expect_identical(nf$theta_grid, np$theta_grid)
  expect_true(all(nf$log_hyperprior == 0))
  expect_equal(nf$log_marginal, np$log_marginal - np$log_hyperprior,
               tolerance = 1e-12)
  expect_identical(nf$log_evidence_declined, "improper_hyperprior")

  # Joint door: a `prior_sigma` the caller states is folded under either choice.
  arm <- list(y = y, n_trials = rep(1L, length(y)), X = X,
              spatial_idx = as.integer(site), family = "binomial")
  pr <- c(list(type = "icar", sigma_grid = c(0.3, 0.6, 1.2, 2.4)), adj)
  ps <- list("pc.prec", c(1, 0.01))
  jp <- tulpa_nested_laplace_joint(list(occ = arm), pr, prior_sigma = ps,
                                   control = list(n_threads = 1L, diagnose_k = FALSE))
  jf <- tulpa_nested_laplace_joint(list(occ = arm), pr, prior_sigma = ps,
                                   hyperprior = "flat",
                                   control = list(n_threads = 1L, diagnose_k = FALSE))
  expect_identical(jf$theta_grid, jp$theta_grid)
  expect_true("sigma" %in% jf$log_hyperprior_axes)
  expect_equal(jf$log_hyperprior, jp$log_hyperprior, tolerance = 1e-12)
  expect_equal(jf$log_marginal, jp$log_marginal, tolerance = 1e-12)
})
