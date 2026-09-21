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
  lp <- tulpa:::.hp_logchol_log_density(g, d)$lp
  # The same PC + LKJ prior re_cov_pc_lkj_prior() puts on the log-Cholesky
  # coordinates, carried to (log sigma_1, log sigma_2, rho) by the Jacobian of
  # L11 = log s1, L21 = rho s2, L22 = log s2 + log(1 - rho^2) / 2, whose
  # determinant is s2 / (1 - rho^2).
  f <- tulpa:::.re_cov_block_logprior(2L, TRUE, tulpa:::.nl_scale_anchor(),
                                      tulpa:::.nl_hyperprior("lkj_eta"))
  C <- tulpa:::.hp_logchol_coords(g, d)
  s2 <- exp(C[, 2L]); rho <- C[, 3L]
  expect_equal(lp, apply(g, 1L, f) + log(s2 / (1 - rho^2)), tolerance = 1e-12)

  # A proper prior: on a tensor wide enough to hold it, the density times the
  # absolute cell measure sums to one. The resolution only sets the quadrature
  # error, measured at 5.0e-4 on 70 x 70 x 81 and 6.1e-4 on the 50 x 50 x 61 CRAN
  # reads, both inside the 2e-3 the identity is held to.
  n_ls <- if (cran_fixture()) 50L else 70L
  n_r  <- if (cran_fixture()) 61L else 81L
  ls <- seq(-9, 3, length.out = n_ls); r <- seq(-0.995, 0.995, length.out = n_r)
  gg <- expand.grid(a = ls, b = ls, rho = r)
  M <- cbind(L11 = gg$a, L21 = gg$rho * exp(gg$b),
             L22 = gg$b + 0.5 * log1p(-gg$rho^2))
  dM <- tulpa:::.hp_logchol_design(M)
  mass <- sum(exp(tulpa:::.hp_logchol_log_density(M, dM)$lp +
                  tulpa:::.hyper_logchol_log_measure(M, dM, absolute = TRUE)))
  expect_equal(mass, 1, tolerance = 2e-3)
})

test_that("a log-Cholesky tensor is measured by its own column widths", {
  G1 <- matrix(seq(-9, 3, length.out = 200L), ncol = 1L,
               dimnames = list(NULL, "L11"))
  d1 <- tulpa:::.hp_logchol_design(G1)
  expect_identical(d1$design, "logchol")
  mass <- sum(exp(tulpa:::.hp_logchol_log_density(G1, d1)$lp +
                  tulpa:::.hyper_logchol_log_measure(G1, d1, absolute = TRUE)))
  expect_equal(mass, 1, tolerance = 2e-3)

  G <- as.matrix(expand.grid(L11 = c(-1, 0, 1), L21 = c(-0.5, 0.5),
                             L22 = c(-1, 0.5)))
  dG <- tulpa:::.hp_logchol_design(G)
  expect_identical(dG$design, "logchol")
  w <- function(x) { e <- c(x[1] - diff(x)[1] / 2, (x[-1] + x[-length(x)]) / 2,
                            x[length(x)] + diff(x)[length(x) - 1] / 2); diff(e) }
  ref <- log(w(c(-1, 0, 1)))[match(G[, 1], c(-1, 0, 1))] +
         log(w(c(-0.5, 0.5)))[match(G[, 2], c(-0.5, 0.5))] +
         log(w(c(-1, 0.5)))[match(G[, 3], c(-1, 0.5))]
  expect_equal(tulpa:::.hyper_logchol_log_measure(G, dG, absolute = TRUE), ref,
               tolerance = 1e-12)
})

test_that("a free-covariance block the grid does not measure declines by name", {
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  pinned <- g
  pinned[, "L22"] <- 0
  expect_identical(tulpa:::.hp_logchol_design(pinned)$reason,
                   "logchol_partial_block")
  set.seed(735)
  scattered <- cbind(L11 = rnorm(12), L21 = rnorm(12), L22 = rnorm(12))
  expect_identical(tulpa:::.hp_logchol_design(scattered)$reason,
                   "logchol_design_measure")
  tg <- scattered; colnames(tg) <- paste0("b1.", colnames(tg))
  sp <- tulpa:::.joint_axis_specs_from_grid(tg)
  expect_true(all(is.na(tulpa:::.hyper_log_quad_weights(tg, sp, absolute = TRUE))))

  # Through the joint collector, a measured block folds once over all its axes.
  tg <- g; colnames(tg) <- paste0("b1.", colnames(g))
  hp <- tulpa:::.joint_hyperprior(tg, list(list(type = "mcar")),
                                  declared = tulpa:::.hp_declare(tg))
  expect_setequal(hp$axes, colnames(tg))
  expect_length(hp$declined, 0L)
  dg <- tulpa:::.hp_logchol_design(g)
  expect_equal(hp$lp, tulpa:::.hp_logchol_log_density(g, dg)$lp)
  sp <- tulpa:::.joint_axis_specs_from_grid(tg)
  expect_equal(tulpa:::.hyper_log_quad_weights(tg, sp, absolute = TRUE),
               tulpa:::.hyper_logchol_log_measure(g, dg, absolute = TRUE))
})

test_that("a batch of a free-covariance block's rows reads the declared design", {
  # gcol33/tulpa#762: the design was inferred from the rows handed to the fold,
  # so a batch of rows of a tensor grid -- importance draws, adaptive seeds, a
  # local-CCD cloud -- is a tensor in no coordinates and lost the Sigma prior.
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  colnames(g) <- paste0("b1.", colnames(g))
  bl <- list(list(type = "mcar"))
  declared <- tulpa:::.hp_declare(g)
  whole <- tulpa:::.joint_hyperprior(g, bl, declared = declared)
  set.seed(762)
  for (rows in list(sample(nrow(g), 7L), 5L, c(1L, nrow(g)))) {
    part <- tulpa:::.joint_hyperprior(g[rows, , drop = FALSE], bl,
                                      declared = declared)
    expect_equal(part$lp, whole$lp[rows], tolerance = 0)
    expect_setequal(unname(part$axes), unname(whole$axes))
    expect_length(part$declined, 0L)
  }

  # A fold handed no declaration for the block refuses rather than inferring one.
  expect_error(
    tulpa:::.joint_hyperprior(g[1:3, , drop = FALSE], bl,
                              declared = list(axes = colnames(g))),
    "no declared design")
})

test_that("a joint tensor with a free-covariance block beside another folds and measures it", {
  # gcol33/tulpa#762: the joint grid repeats the block's rows once per row of the
  # other block, which is still the block's tensor.
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  colnames(g) <- paste0("b1.", colnames(g))
  tg <- cbind(g[rep(seq_len(nrow(g)), 2L), ],
              b2.sigma = rep(c(0.5, 1), each = nrow(g)))
  blocks <- list(list(type = "mcar"), list(type = "icar"))
  hp <- tulpa:::.joint_hyperprior(tg, blocks, declared = tulpa:::.hp_declare(tg))
  expect_setequal(hp$axes, colnames(tg))
  expect_length(hp$declined, 0L)

  dg <- tulpa:::.hp_logchol_design(g)
  expect_identical(dg$design, "sd_rho")
  lp_block <- tulpa:::.hp_logchol_log_density(g, dg)$lp
  lp_sigma <- tulpa:::.joint_hyperprior(
    tg[, "b2.sigma", drop = FALSE], blocks[2L],
    declared = list(axes = "b2.sigma"))$lp
  expect_equal(hp$lp, rep(lp_block, 2L) + lp_sigma, tolerance = 1e-12)

  sp <- tulpa:::.joint_axis_specs_from_grid(tg, folded_axes = hp$axes)
  lq <- tulpa:::.hyper_log_quad_weights(tg, sp, absolute = TRUE)
  expect_true(all(is.finite(lq)))
  # What is left after the block's own measure is the sigma axis's cell width,
  # one value per sigma level.
  rest <- lq - rep(tulpa:::.hyper_logchol_log_measure(g, dg, absolute = TRUE), 2L)
  expect_equal(as.numeric(tapply(rest, tg[, "b2.sigma"], function(v) diff(range(v)))),
               c(0, 0), tolerance = 1e-12)
})

test_that("a subset of the declared tensor is measured under the declared design", {
  # An adaptive lattice keeps a subset of the dense tensor. Its rows are a tensor
  # in no coordinates, and the measure reads the design the fit declared.
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  colnames(g) <- paste0("b1.", colnames(g))
  declared <- tulpa:::.hp_declare(g)
  keep <- setdiff(seq_len(nrow(g)), c(1L, 7L, 30L))
  sub <- g[keep, , drop = FALSE]
  expect_identical(tulpa:::.hp_logchol_designs(sub)[[1L]]$reason,
                   "logchol_design_measure")
  sp <- tulpa:::.joint_axis_specs_from_grid(sub, logchol = declared$logchol)
  lq <- tulpa:::.hyper_log_quad_weights(sub, sp, absolute = TRUE)
  # Every level survives the three dropped cells, so each kept cell keeps the
  # measure it has on the whole grid.
  whole <- tulpa:::.hyper_log_quad_weights(
    g, tulpa:::.joint_axis_specs_from_grid(g), absolute = TRUE)
  expect_equal(lq, whole[keep], tolerance = 1e-12)
})

test_that("points laid in column coordinates read the density carried by the Jacobian", {
  # A CCD design and importance draws are laid in the log-Cholesky columns; the
  # column declaration is the sd_rho density plus log|d(sd_rho)/dL|, so a density
  # over those points is one in the coordinates they were laid in.
  g <- tulpa:::.mcar_default_logchol_grid(2L)
  colnames(g) <- paste0("b1.", colnames(g))
  bl <- list(list(type = "mcar"))
  declared <- tulpa:::.hp_declare(g)
  cols <- tulpa:::.hp_declare_on_columns(declared)
  set.seed(7621)
  pts <- cbind(b1.L11 = rnorm(9), b1.L21 = rnorm(9), b1.L22 = rnorm(9))
  a <- tulpa:::.joint_hyperprior(pts, bl, declared = declared)$lp
  b <- tulpa:::.joint_hyperprior(pts, bl, declared = cols)$lp
  C <- tulpa:::.hp_logchol_coords(pts, list(design = "sd_rho"))
  expect_equal(b, a - log(exp(C[, 2L]) / (1 - C[, 3L]^2)), tolerance = 1e-10)
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
  axes <- tulpa:::.hp_declare(tg)
  whole <- tulpa:::.joint_hyperprior(tg, blocks, fam, declared = axes)
  expect_setequal(whole$axes, c("sigma", "phi_pos"))

  slice <- which(tg[, "sigma"] == 0.7 & tg[, "phi_pos"] == 20)
  one <- which(tg[, "sigma"] == 1.5 & tg[, "alpha"] == 0.5 & tg[, "phi_pos"] == 40)
  for (rows in list(slice, one)) {
    part <- tulpa:::.joint_hyperprior(tg[rows, , drop = FALSE], blocks, fam,
                                      declared = axes)
    expect_equal(part$lp, whole$lp[rows])
    expect_setequal(part$axes, whole$axes)
  }

  # Reading the axes off the batch drops every density a batch holds constant,
  # which is exactly the difference the fix removes.
  own <- tulpa:::.joint_hyperprior(tg[slice, , drop = FALSE], blocks, fam,
                                   declared = tulpa:::.hp_declare(tg[slice, , drop = FALSE]))
  expect_length(own$axes, 0L)
  expect_equal(own$lp, rep(0, length(slice)))
  expect_true(all(whole$lp[slice] != 0))
})

test_that("a stencil or a probe row on a registry block reads the prior of its grid", {
  skip_on_cran()
  # The placement stencil writes its rows onto a block's grid fields, and the
  # inner-skew probe narrows them to the modal row. Both are scored against the
  # hyperprior of the grid the fit integrates: a stencil that moves a column the
  # grid holds fixed adds no density on it, and a single row keeps the density of
  # every column the grid integrates (gcol33/tulpa#760).
  S <- 30L
  set.seed(760)
  idx <- rep(seq_len(S), each = 5L)
  X <- cbind(1, rnorm(length(idx)))
  eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE))
  y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] + rnorm(length(idx), 0, 0.7)
  nbr <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- lengths(nbr)
  graph <- list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
                adj_col_idx = as.integer(unlist(nbr)) - 1L,
                n_neighbors = as.integer(nn))
  a <- list(y = y, n = rep(1L, length(y)), offset_nullable = NULL, X = X,
            re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1,
            family = "gaussian", phi = tulpa:::.phi_to_kernel("gaussian", 0.5),
            max_iter = 200L, tol = 1e-9, n_threads = 1L, x_init_nullable = NULL,
            store_Q = FALSE, checkpoint_path = "", prune_tol = 0,
            screen_iters = tulpa:::.nl_screen("iters"), compute_fitted_var = FALSE,
            hyperprior = "proper")
  bym2 <- function(sigma, rho)
    c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx,
           scale_factor = 1, sigma_grid = sigma, rho_grid = rho), graph)
  rows_at <- function(blk, i) bym2(blk$sigma_grid[i], blk$rho_grid[i])

  # sigma held at one value, rho integrated: the stencil moves sigma too.
  rg <- c(0.2, 0.4, 0.6, 0.8, 0.95)
  held_sigma <- bym2(rep(0.8, 5L), rg)
  grid <- tulpa:::.nl_dispatch("bym2", a, held_sigma, held_sigma)
  expect_identical(grid$log_hyperprior_axes, "rho")
  m <- which.max(grid$log_marginal)
  stencil <- bym2(0.8 * exp(c(0, 0.1, -0.1)), rep(rg[m], 3L))
  st <- tulpa:::.nl_dispatch("bym2", a, stencil, held_sigma)
  expect_equal(st$log_marginal[1L], grid$log_marginal[m], tolerance = 1e-8)
  expect_identical(st$log_hyperprior_axes, "rho")

  # rho held at one value, sigma integrated: the probe row holds sigma constant.
  sg <- c(0.3, 0.6, 1, 1.6, 2.5)
  held_rho <- bym2(sg, rep(0.5, 5L))
  grid <- tulpa:::.nl_dispatch("bym2", a, held_rho, held_rho)
  m <- which.max(grid$log_marginal)
  probe <- tulpa:::.nl_dispatch("bym2", a, rows_at(held_rho, m), held_rho)
  expect_equal(probe$log_marginal, grid$log_marginal[m], tolerance = 1e-8)
  expect_identical(probe$log_hyperprior_axes, "sigma")

  # The multi-block dispatch evaluates an override against the prior it is handed.
  multi <- tulpa:::.nl_dispatch_multi(a, list(held_rho))
  tm <- multi$theta_grid[m, , drop = FALSE]
  one <- tulpa:::.nl_dispatch_multi(a, list(held_rho), theta_grid_override = tm)
  expect_equal(one$log_marginal, multi$log_marginal[m], tolerance = 1e-8)
  multi <- tulpa:::.nl_dispatch_multi(a, list(held_sigma))
  m <- which.max(multi$log_marginal)
  ov <- cbind(0.8 * exp(c(0, 0.1, -0.1)), rep(rg[m], 3L))
  st <- tulpa:::.nl_dispatch_multi(a, list(held_sigma), theta_grid_override = ov)
  expect_equal(st$log_marginal[1L], multi$log_marginal[m], tolerance = 1e-8)
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
