# Per-observation likelihood weights on the SPATIAL tulpa_laplace() route
# (gcol33/tulpa#385).
#
# `weights` used to reach only the non-spatial route. dispatch_laplace_spatial()
# carried no `weights` argument at all, so the spatial kernels found the
# UNWEIGHTED mode -- and then .marginal_H_beta_*(), which has always consumed
# `weights`, built the WEIGHTED marginal precision at that point. The reported
# mode and the reported precision described two different models: on the ICAR
# fixture below the weighted score at the reported mode was 14.86 rather than 0,
# and the slope moved 19% once the weight reached the Newton loop.
#
# The channel was already there -- BuiltinFamilyResponse::weights scales each
# row's log-density, score and Fisher curvature alike -- so the fix is the
# argument reaching the kernels, not new arithmetic. The gates here are
# therefore (a) an all-ones weight is EXACTLY a no-op on every kernel, (b) an
# integer weight is exactly row replication, and (c) the weighted mode is the
# stationary point of the weighted penalized objective, written out
# independently of the engine.

# The penalized objective the ICAR kernel maximizes, from the definition:
#   sum_i w_i * ll(y_i | eta_i)
#   - 0.5 * tau_beta * ||beta||^2            (sigma_beta = 100 in the kernel)
#   - 0.5 * tau * u' (D - A) u               (ICAR structure, tau_spatial = 1)
#   - 0.5 * (tau / J) * (sum_c u)^2          (sum-to-zero pin, J the component
#                                             size; inst/include/tulpa/sum_to_zero.h)
.icar_ref <- function(y, ntr, X, W, unit, w) {
  n_units <- nrow(W); N <- length(unit); p <- ncol(X)
  Q <- diag(rowSums(W)) - W + matrix(1 / n_units, n_units, n_units)
  Z <- matrix(0, N, n_units); Z[cbind(seq_len(N), unit)] <- 1
  M <- cbind(X, Z)
  Qj <- as.matrix(Matrix::bdiag(Matrix::Diagonal(p, 1 / 100^2),
                                Matrix::Matrix(Q, sparse = TRUE)))
  score <- function(th) {
    e <- as.numeric(M %*% th)
    as.numeric(crossprod(M, w * (y - ntr * plogis(e)))) - as.numeric(Qj %*% th)
  }
  hess <- function(th) {
    e <- as.numeric(M %*% th); pr <- plogis(e)
    crossprod(M, (w * ntr * pr * (1 - pr)) * M) + Qj
  }
  th <- rep(0, p + n_units)
  for (it in seq_len(300)) {
    st <- solve(hess(th), score(th))
    th <- th + st
    if (max(abs(st)) < 1e-12) break
  }
  # The kernel centres the field after the Newton loop and folds the removed
  # mean into the intercept, leaving eta unchanged; match that convention.
  m <- mean(th[p + seq_len(n_units)])
  th[1] <- th[1] + m
  th[p + seq_len(n_units)] <- th[p + seq_len(n_units)] - m
  list(theta = th, score = score, hess = hess, M = M, Qj = Qj, p = p,
       n_units = n_units)
}

.areal_fixture <- function(seed, nr = 5L, nc = 5L, N = 300L) {
  set.seed(seed)
  W <- rook_adj(nr, nc)
  n_units <- nrow(W)
  unit <- sample.int(n_units, N, replace = TRUE)
  X <- cbind(1, rnorm(N), rnorm(N))
  u <- rnorm(n_units, 0, 0.7); u <- u - mean(u)
  ntr <- rep(1L, N)
  y <- rbinom(N, ntr, plogis(as.numeric(X %*% c(-0.4, 0.8, -0.5)) + u[unit]))
  list(W = W, unit = unit, X = X, y = as.numeric(y), ntr = ntr, N = N,
       spatial = list(type = "icar", adjacency = W, spatial_idx = unit,
                      scale_factor = 1.0))
}


# A validated NNGP spec: site i conditions on its `nn` nearest predecessors in
# the natural order, the layout laplace_gp_at() reads (nn_order 1-based, nn_idx
# 1-based into that order).
.nngp_fixture <- function(ng, seed, nn = 5L) {
  set.seed(seed)
  coords <- cbind(runif(ng), runif(ng))
  fld <- as.numeric(scale(sin(3 * coords[, 1]) + coords[, 2]))
  y <- as.numeric(rbinom(ng, 1, plogis(fld)))
  X <- cbind(1, rnorm(ng))
  d <- as.matrix(dist(coords))
  nn_idx <- matrix(0L, ng, nn); nn_dist <- matrix(0, ng, nn)
  for (i in seq_len(ng)) {
    prev <- seq_len(i - 1L); if (!length(prev)) next
    k <- min(nn, length(prev)); o <- order(d[i, prev])[seq_len(k)]
    nn_idx[i, seq_len(k)] <- as.integer(prev[o])
    nn_dist[i, seq_len(k)] <- d[i, prev[o]]
  }
  list(y = y, X = X, ntr = rep(1L, ng),
       spatial = list(type = "gp", cov = "exponential",
                      unique_coords = coords, n_spatial = ng, nn = nn,
                      obs_to_loc = seq_len(ng),
                      neighbor_info = list(nn_idx = nn_idx, nn_dist = nn_dist,
                                           nn_order = seq_len(ng)),
                      sigma2_gp = 0.9, phi_gp = 0.4))
}


test_that("an all-ones weight vector is exactly a no-op on every spatial kernel", {
  skip_on_cran()
  # This is the gate that protects every existing spatial fit: nothing may move
  # when no weight is supplied, so the comparison is at tolerance = 0.
  set.seed(700)
  a <- .areal_fixture(700L, N = 180L)
  ones <- rep(1, a$N)

  for (ty in c("icar", "car", "bym2", "car_proper")) {
    sp <- if (ty == "car_proper")
      list(type = "car_proper", adjacency = a$W, spatial_idx = a$unit,
           tau = 1.0, rho = 0.4)
    else list(type = ty, adjacency = a$W, spatial_idx = a$unit,
              scale_factor = 1.0)
    f0 <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial", spatial = sp)
    f1 <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial", spatial = sp,
                        weights = ones)
    expect_equal(f0$mode, f1$mode, tolerance = 0, label = ty)
    expect_equal(f0$log_marginal, f1$log_marginal, tolerance = 0, label = ty)
    expect_equal(f0$H_beta, f1$H_beta, tolerance = 0, label = ty)
  }

  # HSGP (dense basis) and NNGP (Vecchia, sparse Newton) take different
  # scatters, so both are checked rather than one standing for the other.
  coords <- cbind(runif(a$N), runif(a$N))
  sp_h <- list(type = "hsgp", coords_matrix = coords, m = 6L, c = 1.5,
               sigma2 = 0.8, lengthscale = 0.5)
  h0 <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial", spatial = sp_h)
  h1 <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial", spatial = sp_h,
                      weights = ones)
  expect_equal(h0$mode, h1$mode, tolerance = 0)
  expect_equal(h0$log_marginal, h1$log_marginal, tolerance = 0)

  g <- .nngp_fixture(90L, 12L)
  n0 <- tulpa_laplace(g$y, g$ntr, g$X, family = "binomial", spatial = g$spatial)
  n1 <- tulpa_laplace(g$y, g$ntr, g$X, family = "binomial", spatial = g$spatial,
                      weights = rep(1, length(g$y)))
  expect_equal(n0$mode, n1$mode, tolerance = 0)
  expect_equal(n0$log_marginal, n1$log_marginal, tolerance = 0)
})


test_that("an integer weight is exactly row replication (areal + NNGP)", {
  skip_on_cran()
  # sum_i w_i ll_i is the log-likelihood of the data set with row i repeated
  # w_i times, and neither the field prior nor the fixed-effect prior depends on
  # N -- so the two penalized objectives are the same function and must have the
  # same mode. This reads the weighted path against the engine's own unweighted
  # solver on a different data set, so a weight applied to the score but not the
  # curvature (or to neither) fails it.
  a <- .areal_fixture(701L, N = 150L)
  w <- rep(c(1, 2, 3), length.out = a$N)
  rep_idx <- rep(seq_len(a$N), times = w)

  for (ty in c("icar", "bym2", "car_proper")) {
    sp_w <- if (ty == "car_proper")
      list(type = "car_proper", adjacency = a$W, spatial_idx = a$unit,
           tau = 1.0, rho = 0.4)
    else list(type = ty, adjacency = a$W, spatial_idx = a$unit,
              scale_factor = 1.0)
    sp_r <- sp_w
    sp_r$spatial_idx <- a$unit[rep_idx]

    fw <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial", spatial = sp_w,
                        weights = w, max_iter = 300L, tol = 1e-12)
    fr <- tulpa_laplace(a$y[rep_idx], a$ntr[rep_idx],
                        a$X[rep_idx, , drop = FALSE], family = "binomial",
                        spatial = sp_r, max_iter = 300L, tol = 1e-12)
    expect_equal(fw$mode, fr$mode, tolerance = 1e-6, label = ty)
    expect_equal(fw$H_beta, fr$H_beta, tolerance = 1e-6, label = ty)
  }

  g <- .nngp_fixture(80L, 13L)
  n <- length(g$y)
  wg <- rep(c(1, 3), length.out = n)
  ri <- rep(seq_len(n), times = wg)
  sp_r <- g$spatial
  sp_r$obs_to_loc <- ri                  # replicated rows share their field node
  fw <- tulpa_laplace(g$y, g$ntr, g$X, family = "binomial",
                      spatial = g$spatial, weights = wg,
                      max_iter = 300L, tol = 1e-12)
  fr <- tulpa_laplace(g$y[ri], g$ntr[ri], g$X[ri, , drop = FALSE],
                      family = "binomial", spatial = sp_r,
                      max_iter = 300L, tol = 1e-12)
  expect_equal(fw$mode, fr$mode, tolerance = 1e-6)
})


test_that("the weighted spatial mode is the weighted penalized optimum", {
  skip_on_cran()
  # The arbiter is an R Newton solve on the objective written from its
  # definition, not on anything the engine computes.
  a <- .areal_fixture(20385L)
  w <- ifelse(seq_len(a$N) %% 2L == 0L, 4, 1)

  fit <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial",
                       spatial = a$spatial, weights = w,
                       max_iter = 300L, tol = 1e-12)
  ref_w <- .icar_ref(a$y, a$ntr, a$X, a$W, a$unit, w)
  ref_u <- .icar_ref(a$y, a$ntr, a$X, a$W, a$unit, rep(1, a$N))

  p <- ncol(a$X)
  expect_lt(max(abs(fit$mode[seq_len(p)] - ref_w$theta[seq_len(p)])), 1e-6)

  # The weighted and unweighted optima are genuinely apart on this fixture, so
  # the assertion above is not satisfied by simply ignoring the weight.
  expect_gt(max(abs(ref_w$theta[seq_len(p)] - ref_u$theta[seq_len(p)])), 0.05)

  # Stationarity of the weighted objective at the point the fit reports. The
  # Newton loop stops on its step, so this residual settles at a floor set by
  # conditioning rather than at zero (~1e-06 here). What the bound separates is
  # a settled weighted mode from the unweighted one, which left this same
  # quantity at 14.86 -- seven orders of magnitude up.
  expect_lt(max(abs(ref_w$score(fit$mode))), 1e-4)
})


test_that("the spatial mode and its H_beta describe the same weighted model", {
  skip_on_cran()
  # #385 itself: H_beta was already the weighted Schur, the mode was not. The
  # pair is only a posterior once both read the same w.
  a <- .areal_fixture(20386L, N = 260L)
  w <- runif(a$N, 0.2, 3)

  fit <- tulpa_laplace(a$y, a$ntr, a$X, family = "binomial",
                       spatial = a$spatial, weights = w,
                       max_iter = 300L, tol = 1e-12)
  ref <- .icar_ref(a$y, a$ntr, a$X, a$W, a$unit, w)
  p <- ncol(a$X)

  H <- ref$hess(fit$mode)
  idx_b <- seq_len(p); idx_u <- p + seq_len(ref$n_units)
  schur <- H[idx_b, idx_b] -
    H[idx_b, idx_u] %*% solve(H[idx_u, idx_u], t(H[idx_b, idx_u]))
  # The engine's marginal omits the weak fixed-effect ridge the mode carries.
  schur <- schur - diag(1 / 100^2, p)

  expect_lt(max(abs(fit$H_beta - schur)) / max(abs(schur)), 1e-6)
  expect_lt(max(abs(ref$score(fit$mode))), 1e-4)
})


test_that("a spatial fit rejects a weights vector of the wrong length", {
  a <- .areal_fixture(702L, N = 60L)
  expect_error(
    tulpa_laplace(a$y, a$ntr, a$X, family = "binomial", spatial = a$spatial,
                  weights = rep(1, a$N - 1L)),
    regexp = "length"
  )
})
