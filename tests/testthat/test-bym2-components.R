# BYM2 on a disconnected graph (gcol33/tulpa#902). Riebler's scale factor is
# 1 / sqrt(generalised variance), the geometric mean of diag(Q^+); over a whole
# graph with an island that mean took log(0) and the scale was Inf, after which
# nested_laplace returned NA coefficients and the sampler froze. The graph is
# now scaled per connected component and an island gets unit structured
# variance (Freni-Sterrantino, Ventrucci & Rue 2018): the scalar scale_factor
# is the largest component's, and the rest rides on phi's prior as a per-node
# precision multiplier (.bym2_component_scaling).

.chain <- function(n) {
  A <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- 1
  A
}

.gen_var <- function(A) {
  L <- diag(rowSums(A)) - A
  e <- eigen(L, TRUE)
  nz <- abs(e$values) > 1e-10
  Lp <- e$vectors[, nz] %*% (t(e$vectors[, nz]) / e$values[nz])
  exp(mean(log(diag(Lp))))
}

test_that("a connected graph keeps its Riebler scale and carries no multipliers", {
  A <- .chain(12L)
  sc <- tulpa:::.bym2_component_scaling(A)
  expect_null(sc$node_prec)
  expect_equal(sc$scale_factor, 1 / sqrt(.gen_var(A)), tolerance = 1e-10)
  expect_equal(compute_bym2_scale(A), sc$scale_factor)
})

test_that("an island no longer makes the scale infinite (#902)", {
  A <- matrix(0, 16, 16)
  A[1:15, 1:15] <- .chain(15L)                 # node 16 isolated
  s <- suppressWarnings(spatial_bym2(A, group_var = "region"))
  expect_true(is.finite(s$scale_factor))
  expect_equal(s$scale_factor, 1 / sqrt(.gen_var(.chain(15L))), tolerance = 1e-10)
  # The island's structured part has unit variance after scaling:
  # (scale_factor * phi) with phi ~ N(0, 1 / node_prec).
  expect_equal(s$scale_factor^2 / s$node_prec[16], 1)
  expect_equal(s$node_prec[1:15], rep(1, 15))
  expect_output(print(s), "per connected component")
})

test_that("each component of a disconnected graph reaches unit generalised variance", {
  # A 10-chain, a 3 x 3 lattice (a different geometry, so a different scale)
  # and an island.
  lat <- matrix(0, 9, 9)
  id <- function(i, j) (j - 1L) * 3L + i
  for (i in 1:3) for (j in 1:3) {
    if (i < 3) lat[id(i, j), id(i + 1, j)] <- lat[id(i + 1, j), id(i, j)] <- 1
    if (j < 3) lat[id(i, j), id(i, j + 1)] <- lat[id(i, j + 1), id(i, j)] <- 1
  }
  A <- as.matrix(Matrix::bdiag(.chain(10L), lat, matrix(0, 1, 1)))
  sc <- tulpa:::.bym2_component_scaling(A)
  comps <- sc$components
  for (cc in comps[lengths(comps) > 1L]) {
    # Precision of scale_factor * phi on the component: L_c * w_c / s^2.
    w <- unique(sc$node_prec[cc])
    expect_length(w, 1L)
    gv_scaled <- .gen_var(A[cc, cc]) * sc$scale_factor^2 / w
    expect_equal(gv_scaled, 1, tolerance = 1e-10)
  }
  expect_equal(sc$scale_factor^2 / sc$node_prec[20], 1)
})

.island_data <- function() {
  A <- matrix(0, 16, 16)
  A[1:15, 1:15] <- .chain(15L)
  set.seed(1)
  d <- data.frame(region = rep(1:16, each = 4), x = rnorm(64))
  d$y <- rpois(64, exp(0.3 + 0.5 * d$x))
  list(A = A, d = d)
}

test_that("the kernels read the per-node multipliers: exact gaussian check", {
  skip_on_cran()
  # A gaussian BYM2 at the conditional kernel's (sigma = 1, rho = 0.5) is a
  # Gaussian model, so the fitted linear predictor is the closed-form posterior
  # mean under the weighted structured prior -- a kernel that ignored node_prec
  # (or read it for the wrong nodes) would not reproduce it.
  A <- as.matrix(Matrix::bdiag(.chain(6L), .chain(4L), matrix(0, 1, 1)))
  n <- nrow(A)
  set.seed(2)
  d <- data.frame(region = rep(seq_len(n), each = 3L), x = rnorm(3L * n))
  d$y <- 0.2 + 0.5 * d$x + rnorm(nrow(d), 0, 0.5)
  sp <- suppressWarnings(spatial_bym2(A, group_var = "region"))
  expect_false(is.null(sp$node_prec))
  f <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d, phi = 0.25,
                              mode = "laplace", spatial = sp))

  Q <- as.matrix(tulpa:::.icar_precision_Q(sp, sp$node_prec))
  Z <- model.matrix(~ factor(region) - 1, d)
  X <- cbind(1, d$x)
  W <- cbind(X, sqrt(0.5) * sp$scale_factor * Z, sqrt(0.5) * Z)
  P <- as.matrix(Matrix::bdiag(diag(1e-4, 2), Q, diag(n)))
  post_mean <- solve(crossprod(W) / 0.25 + P, crossprod(W, d$y) / 0.25)
  eta_exact <- as.numeric(W %*% post_mean)

  m <- f$mode %||% f$fit$mode
  eta_fit <- as.numeric(W %*% m[seq_len(ncol(W))])
  expect_equal(eta_fit, eta_exact, tolerance = 1e-5)
})

test_that("nested_laplace fits a BYM2 field with an island (#902)", {
  skip_on_cran()
  dd <- .island_data()
  f <- suppressWarnings(tulpa(
    y ~ x + spatial(region), dd$d, family = "poisson", mode = "nested_laplace",
    spatial = spatial_bym2(dd$A, group_var = "region"),
    control = list(n_threads = 1L)))
  cf <- coef(f)
  # Was NA NA, with "did not return Q. This is a tulpa bug".
  expect_true(all(is.finite(cf)))
  expect_equal(unname(cf[2]), 0.5, tolerance = 0.3)
  expect_true(all(is.finite(summary(f)[, 2])))
})

test_that("the Polya-Gamma Gibbs BYM2 gives the island its unit prior", {
  skip_on_cran()
  dd <- .island_data()
  set.seed(3)
  dd$d$yb <- rbinom(64, 1, plogis(0.3 + 0.5 * dd$d$x))
  sp <- suppressWarnings(spatial_bym2(dd$A, group_var = "region"))
  f <- tulpa(yb ~ x + spatial(region), dd$d, family = "binomial",
             mode = "gibbs", spatial = sp,
             control = list(n_iter = 400, warmup = 200, seed = 1))
  expect_true(all(is.finite(summary(f)[, 2])))
  # The island's structured draw is N(0, 1 / node_prec) a priori, where the
  # unweighted kernel's near-flat PG_ICAR_ISOLATED_PREC let it wander.
  ph <- f$draws[, "phi_spatial[16]"]
  expect_lt(stats::sd(ph), 5 / sqrt(sp$node_prec[16]))
})

test_that("the ModelData sampler's BYM2 prior is diag(node_prec) Q_aug (#902)", {
  # Shifting phi by a constant leaves the centred field -- and so eta and the
  # likelihood -- unchanged, so the log-posterior difference is the prior's
  # alone. Q 1 = 0 on every component, so what moves is each component's
  # augmentation, sum_c w_c (S_c + c J_c)^2 / J_c: weighted by the component's
  # multiplier, where the one-scale prior weighted every component by 1.
  dd <- .island_data()
  sp <- suppressWarnings(spatial_bym2(dd$A, group_var = "region"))
  sp$spatial_idx <- dd$d$region
  nl <- tulpa:::.spatial_spec_to_nl_prior(sp)
  spec <- list(type = "bym2", spatial_idx = nl$spatial_idx,
               n_spatial_units = nl$n_spatial_units,
               adj_row_ptr = nl$adj_row_ptr, adj_col_idx = nl$adj_col_idx,
               n_neighbors = nl$n_neighbors, scale_factor = sp$scale_factor,
               node_prec = sp$node_prec)
  X <- cbind(1, dd$d$x)
  S <- nrow(dd$A)
  set.seed(4)
  base <- c(0.3, 0.5, log(0.8), 0.2, rnorm(S, 0, 0.3), rnorm(S, 0, 0.3))
  shift <- 0.7
  up <- base; up[4L + seq_len(S)] <- up[4L + seq_len(S)] + shift
  lp <- function(sp_spec) tulpa:::cpp_tulpa_glmm_log_prob_draws(
    rbind(base, up), dd$d$y, rep(1L, nrow(dd$d)), X, "poisson",
    spatial_spec = sp_spec)
  aug <- function(phi, w) {
    comps <- list(1:15, 16L)
    sum(vapply(comps, function(cc) w[cc[1]] * sum(phi[cc])^2 / length(cc),
               numeric(1)))
  }
  phi0 <- base[4L + seq_len(S)]
  expect_equal(diff(lp(spec)),
               -0.5 * (aug(phi0 + shift, sp$node_prec) - aug(phi0, sp$node_prec)),
               tolerance = 1e-8)
  # The one-scale prior (no node_prec) weights the island by 1 instead, and
  # the island's multiplier is not 1 on this graph.
  expect_false(isTRUE(all.equal(sp$node_prec[16], 1)))
  spec1 <- spec; spec1$node_prec <- NULL
  expect_equal(diff(lp(spec1)),
               -0.5 * (aug(phi0 + shift, rep(1, S)) - aug(phi0, rep(1, S))),
               tolerance = 1e-8)
  # A multiplier that is not constant within a component is refused.
  bad <- spec; bad$node_prec[1] <- 2
  expect_error(lp(bad), "constant within each connected component")
})

test_that("mode = 'hmc' fits a BYM2 field with an island (#902)", {
  skip_on_cran()
  dd <- .island_data()
  sp <- suppressWarnings(spatial_bym2(dd$A, group_var = "region"))
  f <- suppressWarnings(tulpa(
    y ~ x + spatial(region), dd$d, family = "poisson", mode = "hmc",
    spatial = sp,
    control = list(n_chains = 2, n_iter = 1000, warmup = 500, seed = 1)))
  cf <- coef(f)
  expect_true(all(is.finite(cf)))
  expect_equal(unname(cf[2]), 0.5, tolerance = 0.3)
  # The island's structured draw is N(0, 1 / node_prec) a priori; the data
  # barely inform it, so its spread is the prior's and not a frozen chain.
  ph <- f$draws[, "phi_spatial[16]"]
  expect_gt(stats::sd(ph), 0.2 / sqrt(sp$node_prec[16]))
  expect_lt(stats::sd(ph), 5 / sqrt(sp$node_prec[16]))
  # Agrees with the nested-Laplace fit of the same per-component model.
  fn <- suppressWarnings(tulpa(
    y ~ x + spatial(region), dd$d, family = "poisson", mode = "nested_laplace",
    spatial = sp, control = list(n_threads = 1L)))
  expect_lt(max(abs(coef(f) - coef(fn))), 0.15)
})
