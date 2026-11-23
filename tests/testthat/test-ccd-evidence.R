# The evidence read off a central composite design. Each node's moment-rule
# weight, calibrated against the Gaussian the design was laid on
# (`.ccd_log_measure()`), is exact when the target over that Gaussian is a
# cubic in the whitened coordinates; the fits below are checked against a
# tensor grid dense enough to be its own reference.

test_that(".ccd_log_measure integrates a Gaussian exactly at any centre and scale", {
  set.seed(3)
  for (k in 1:4) {
    ccd <- tulpa:::ccd_grid(k, f_0 = sqrt(k) * 1.1)
    A <- matrix(rnorm(k * k), k)
    L <- t(chol(crossprod(A) + diag(k)))
    mu <- rnorm(k)
    th <- tulpa:::ccd_to_theta(ccd$z, mu, L)
    # log of c * N(th; mu, L L'), whose integral is c.
    lc <- 2.7
    Si <- solve(L %*% t(L))
    lg <- lc - 0.5 * k * log(2 * pi) - sum(log(diag(L))) -
      0.5 * rowSums((sweep(th, 2, mu) %*% Si) * sweep(th, 2, mu))
    ev <- tulpa:::.nl_outer_log_evidence(lg, tulpa:::.ccd_log_measure(ccd, L))
    expect_equal(ev, lc, tolerance = 1e-10)
  }
})

test_that("the CCD rule is exact for a cubic ratio to its Gaussian", {
  k <- 3L
  ccd <- tulpa:::ccd_grid(k, f_0 = sqrt(k) * 1.1)
  z <- ccd$z
  # r(z) = 1 + 0.3 z1 + 0.2 z2^2 - 0.1 z1 z3 + 0.05 z2^3 has E_phi[r] = 1.2.
  r <- 1 + 0.3 * z[, 1] + 0.2 * z[, 2]^2 - 0.1 * z[, 1] * z[, 3] +
    0.05 * z[, 2]^3
  lg <- log(r) - 0.5 * rowSums(z^2)
  ev <- tulpa:::.nl_outer_log_evidence(lg, tulpa:::.ccd_log_measure(ccd, diag(k)))
  expect_equal(ev, log(1.2) + 0.5 * k * log(2 * pi), tolerance = 1e-12)
})

skip_on_cran()

.ccde_slope_data <- function(seed = 4L, G = 30L, npg = 10L) {
  set.seed(seed)
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(G * npg)
  S <- matrix(c(0.49, 0.14, 0.14, 0.25), 2)
  u <- t(t(chol(S)) %*% matrix(rnorm(2 * G), 2))
  y <- rpois(G * npg, exp(0.2 + 0.5 * x + u[grp, 1] + u[grp, 2] * x))
  data.frame(y = y, x = x, g = factor(grp))
}

# The same model on the joint door with every axis stated as a dense tensor
# over the span the default fit's cells cover, widened by half on each side,
# every cell solved: its evidence is the reference.
.ccde_dense_evidence <- function(fit, d, n_per_axis) {
  tg <- fit$theta_grid
  ax <- lapply(seq_len(ncol(tg)), function(j) {
    r <- range(tg[, j]); h <- diff(r) / 2
    seq(r[1] - h, r[2] + h, length.out = n_per_axis)
  })
  G <- as.matrix(expand.grid(ax))
  nc <- if (ncol(tg) == 1L) 1L else 2L
  colnames(G) <- if (nc == 1L) "L11" else c("L11", "L21", "L22")
  g <- as.integer(d$g); N <- nrow(d)
  w <- if (nc == 1L) list(list(rep(1, N))) else list(list(rep(1, N)), list(d$x))
  blk <- list(type = "miid", obs_idx = g, n_groups = nlevels(d$g), n_fields = nc,
              field_weight = w, logchol_prior = re_cov_pc_lkj_prior(nc),
              logchol_grid = G)
  tulpa_nested_laplace_joint(
    list(y = list(y = d$y, X = cbind(1, d$x), family = "poisson")),
    prior = list(blk),
    control = list(progress = FALSE, diagnose_k = FALSE, axis_refine = "none",
                   var_of_means_consistency = FALSE, prune = FALSE,
                   auto_recenter = FALSE, max_grid_cells = 1e5))$log_evidence
}

test_that("logLik() of a re_cov fit matches a dense tensor grid", {
  d <- .ccde_slope_data()
  fit <- function(f, ctl) tulpa(f, data = d, family = "poisson", mode = "laplace",
                                control = c(list(diagnose_k = FALSE), ctl))
  slope <- fit(y ~ x + (1 + x | g), list())
  expect_true(is.finite(as.numeric(logLik(slope))))
  expect_null(attr(logLik(slope), "declined"))
  expect_lt(abs(as.numeric(logLik(slope)) - .ccde_dense_evidence(slope, d, 21L)),
            0.1)
  int <- fit(y ~ x + (1 | g), list(re_cov = "nested"))
  expect_lt(abs(as.numeric(logLik(int)) - .ccde_dense_evidence(int, d, 201L)),
            0.05)
})

test_that("the evidence of an SPDE fit matches a covering dense grid", {
  skip_if_not_installed("fmesher")
  set.seed(7)
  n_obs <- 150L
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.4), cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- methods::as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                              prior_range = c(0.1, 0.5),
                              prior_sigma = c(0.6, 0.05))
  w <- as.numeric(rnorm(spec$n_mesh, 0, 0.6)); w <- w - mean(w)
  x <- rnorm(n_obs)
  y <- rpois(n_obs, exp(2 + 0.5 * x + as.numeric(spec$A %*% w)))
  X <- cbind(1, x)
  fc <- suppressWarnings(fit_spde(y = y, X = X, spatial = spec, family = "poisson",
                                  control = list(progress = FALSE)))
  expect_true(is.finite(fc$log_evidence))
  # The reference: the same per-cell marginal and hyperprior on a dense 41 x 41
  # grid in (log range, log sigma) covering +/- 6 posterior SDs, its evidence
  # the cell-width-weighted sum.
  L <- log(fc$nested$theta_grid); wg <- fc$nested$weights
  ctr <- as.numeric(crossprod(wg, L))
  sdl <- sqrt(as.numeric(crossprod(wg, sweep(L, 2L, ctr)^2)))
  g1 <- seq(ctr[1] - 6 * sdl[1], ctr[1] + 6 * sdl[1], length.out = 41L)
  g2 <- seq(ctr[2] - 6 * sdl[2], ctr[2] + 6 * sdl[2], length.out = 41L)
  G <- expand.grid(a = g1, b = g2)
  solve <- list(re_idx = rep(0L, n_obs), n_re_groups = 0L, sigma_re = 1,
                max_iter = 100L, tol = 1e-6, beta_prior = NULL)
  lm <- tulpa:::.spde_grid_solve(spec, y = y, n_trials = rep(1L, n_obs), X = X,
                                 family = "poisson", phi = 1, offset = NULL,
                                 solve = solve, range_vec = exp(G$a),
                                 sigma_vec = exp(G$b),
                                 extra = list(progress = FALSE))$log_marginal
  lp <- lm + tulpa:::.spde_log_hyperprior(exp(G$a), exp(G$b), spec)
  mx <- max(lp)
  m <- matrix(exp(lp - mx) / sum(exp(lp - mx)), 41L)
  expect_lt(max(sum(m[1, ]), sum(m[41, ]), sum(m[, 1]), sum(m[, 41])), 1e-3)
  ref <- mx + log(sum(exp(lp - mx))) + log(diff(g1[1:2]) * diff(g2[1:2]))
  expect_lt(abs(fc$log_evidence - ref), 0.1)
})

test_that("a joint CCD design reads its evidence on the axes' own coordinates", {
  src <- testthat::test_path("test-nested-laplace-joint-ccd.R")
  env <- new.env()
  eval(parse(text = readLines(src)[11:54]), envir = env)
  sim <- env$.sim_joint_ccd(2024L, N = 800L, n_s = 40L)
  sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
  blk <- env$.bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
  jc <- suppressMessages(tulpa_nested_laplace_joint(sim$responses, list(blk),
    copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
    control = list(integration = "ccd", diagnose_k = FALSE, verbose = FALSE,
                   var_of_means_consistency = FALSE)))
  mk <- function(a, b, n) exp(seq(log(a), log(b), length.out = n))
  blk2 <- env$.bym2_copy_block(sim$adj, mk(0.3, 1.8, 9L),
                               stats::plogis(seq(-4, 5, length.out = 9L)), sp)
  jt <- suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
    sim$responses, list(blk2),
    copy = list(arm = "pos", block = 1L, alpha_grid = mk(0.45, 1.9, 9L)),
    control = list(integration = "grid", diagnose_k = FALSE, verbose = FALSE,
                   var_of_means_consistency = FALSE, axis_refine = "none"))))
  expect_identical(jc$integration, "ccd")
  expect_true(is.finite(jc$log_evidence))
  # The mixing weight is designed on logit(rho) and its prior is uniform on
  # rho, so the design carries log(rho (1 - rho)). The tolerance is the
  # design's own error on a posterior whose rho spreads over a third of the
  # unit interval, about 1 nat on this data.
  expect_lt(abs(jc$log_evidence - jt$log_evidence), 1.5)
  expect_lt(abs(unname(jc$theta_mean)[3] - unname(jt$theta_mean)[3]), 0.05)
})
