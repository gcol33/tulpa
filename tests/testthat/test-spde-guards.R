# The SPDE boundary guards, and the zero-mass mesh vertex (gcol33/tulpa#590).
#
# Split out of gcol33/tulpa#528, #530 and #534: in each the guard was written
# and wired and nothing in tests/ reached it, so a regression would have been
# silent.
#
#   1. The orphan ridge. A vertex with zero FEM mass reaches the assembly from
#      upstream refiners that emit Steiner points without retriangulating.
#      `spde_zero_mass.h` floors its inverse mass and places a theta-independent
#      unit precision on its diagonal, and BOTH assemblies do it -- the Laplace
#      path through SpdeQBuilder and the non-centered NUTS path through
#      SpdeNcTransform. The point of the fix is that the two agree on such a
#      mesh, so both are exercised on the same one.
#
#   2. The four structural validators. Each reports the offending length or
#      index, and none had ever fired.
#
#   3. The indefinite-H guard in the implicit-diff gradient is deliberately NOT
#      pinned; see the note at the bottom.

skip_on_cran()

# A valid small mesh, and the same mesh with one vertex's FEM mass set to zero.
# Zeroing an entry of C0_diag is exactly the shape the refiner produces: the
# vertex keeps its G1 connectivity and loses its mass, which is the harder of
# the two flavours spde_zero_mass.h describes (a truly disconnected vertex has
# no G1 row either).
.spde_guard_fem <- function(n_obs = 80L, seed = 590L) {
  skip_if_not_installed("fmesher")
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.30, 0.60),
                              cutoff = 0.10)
  fem  <- fmesher::fm_fem(mesh)
  G1   <- as(fem$g1, "CsparseMatrix")
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  list(n_obs = n_obs, coords = coords, n_mesh = mesh$n,
       C0_diag = as.numeric(Matrix::diag(fem$c0)),
       G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
       A_x = A@x, A_i = A@i, A_p = A@p)
}

# The orphan is the LAST vertex, which is where a refiner appends a Steiner
# point, and it is chosen to still carry G1 connectivity.
.spde_orphan <- function(fem) {
  j <- fem$n_mesh
  expect_gt(fem$G1_p[j + 1L] - fem$G1_p[j], 1L)   # the flavour we mean
  fem$C0_diag[j] <- 0
  fem$orphan <- j
  fem
}

# --------------------------------------------------------------------------- #
# 1. Zero-mass mesh vertex: both assemblies                                    #
# --------------------------------------------------------------------------- #

# The precision SpdeQBuilder assembles at alpha = 2, written independently in R
# from the operator chain's own definition: Q = tau^2 (k2^2 C + 2 k2 G + G D G),
# with D the FLOORED inverse mass.
.spde_expanded_Q <- function(fem, kappa, tau) {
  n <- fem$n_mesh
  G <- Matrix::sparseMatrix(i = fem$G1_i + 1L, p = fem$G1_p, x = fem$G1_x,
                            dims = c(n, n), index1 = TRUE)
  D <- ifelse(fem$C0_diag > 1e-15, 1 / fem$C0_diag, 0)
  k2 <- kappa^2
  Q <- tau^2 * (k2^2 * Matrix::Diagonal(x = fem$C0_diag) + 2 * k2 * G +
                  G %*% Matrix::Diagonal(x = D) %*% G)
  # The theta-independent unit ridge on orphan diagonals.
  orph <- which(fem$C0_diag <= 1e-15)
  if (length(orph)) Q <- Q + Matrix::sparseMatrix(i = orph, j = orph,
                                                  x = 1, dims = c(n, n))
  as.matrix(Q)
}

# The precision SpdeQBuilder assembles for the Laplace path. It derives
# (kappa, tau) from (range, sigma, nu) through the shared Matern conversion, so
# the pair it used comes back with the matrix and is checked against the one the
# other two assemblies were handed.
.spde_qb_Q <- function(fem, range = 0.4, sigma = 1) {
  qb <- cpp_test_spde_assemble(
    C0_diag = fem$C0_diag, G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    range = range, sigma = sigma, nu = 1, alpha = 2L)
  qb$Q <- as.matrix(Matrix::sparseMatrix(i = qb$Q_i + 1L, p = qb$Q_p,
                                         x = qb$Q_x,
                                         dims = c(fem$n_mesh, fem$n_mesh),
                                         index1 = TRUE))
  qb
}

# The product form tau^2 K diag(1/C0) K, which is what the two agree on where
# C diag(1/C0) = I. Carried as the negative control below.
.spde_product_Q <- function(fem, kappa, tau) {
  n <- fem$n_mesh
  G <- Matrix::sparseMatrix(i = fem$G1_i + 1L, p = fem$G1_p, x = fem$G1_x,
                            dims = c(n, n), index1 = TRUE)
  D <- ifelse(fem$C0_diag > 1e-15, 1 / fem$C0_diag, 0)
  K <- kappa^2 * Matrix::Diagonal(x = fem$C0_diag) + G
  Q <- tau^2 * (K %*% Matrix::Diagonal(x = D) %*% K)
  orph <- which(fem$C0_diag <= 1e-15)
  if (length(orph)) Q <- Q + Matrix::sparseMatrix(i = orph, j = orph,
                                                  x = 1, dims = c(n, n))
  as.matrix(Q)
}

.spde_nc_Q <- function(fem, kappa, tau) {
  q <- cpp_test_spde_nc_transform_Q(
    C0_diag = fem$C0_diag, G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    log_kappa_val = log(kappa), log_tau_val = log(tau))
  as.matrix(Matrix::sparseMatrix(i = q$i + 1L, p = q$p, x = q$x,
                                 dims = c(q$n, q$n), index1 = TRUE))
}

test_that("both assemblies build the same Q on a mesh with a zero-mass vertex", {
  # This is the point of the orphan ridge: a field fitted non-centered and the
  # same field fitted centered must describe the same posterior. The two took
  # different routes to Q -- SpdeQBuilder the operator-chain EXPANSION, the
  # non-centered transform the product K diag(1/C0) K -- and those agree only
  # where C diag(1/C0) = I, which a floored orphan row is not.
  fem <- .spde_orphan(.spde_guard_fem())
  kappa <- sqrt(8) / 0.4
  tau   <- 1 / (sqrt(4 * pi) * kappa)
  Q_nc  <- .spde_nc_Q(fem, kappa, tau)
  Q_ref <- .spde_expanded_Q(fem, kappa, tau)
  expect_equal(max(abs(Q_nc - Q_ref)), 0, tolerance = 1e-12)

  # SpdeQBuilder itself, at the same (kappa, tau). The R expansion is a third
  # party both are scored against; this is the pair the fit actually uses.
  qb <- .spde_qb_Q(fem)
  expect_equal(qb$kappa, kappa)
  expect_equal(qb$tau, tau)
  expect_equal(max(abs(qb$Q - Q_ref)), 0, tolerance = 1e-12)
  expect_equal(max(abs(Q_nc - qb$Q)) / max(abs(qb$Q)), 0, tolerance = 1e-12)

  # The disagreement was confined to the orphan's row and column, so a check
  # that ignored them would have passed throughout.
  j <- fem$orphan
  expect_gt(max(abs(Q_ref[j, -j])), 1e-3)
  expect_equal(max(abs(Q_nc[j, ] - Q_ref[j, ])), 0, tolerance = 1e-12)

  # The negative control, and what makes the assertions above non-vacuous: the
  # product form differs on the orphan's row by exactly tau^2 * 2 kappa^2 G[j, ],
  # 6% of max|Q| on this mesh. A regression back to it fails here.
  G <- Matrix::sparseMatrix(i = fem$G1_i + 1L, p = fem$G1_p, x = fem$G1_x,
                            dims = c(fem$n_mesh, fem$n_mesh), index1 = TRUE)
  Q_prod <- .spde_product_Q(fem, kappa, tau)
  expect_equal(max(abs(Q_prod[j, ] - Q_ref[j, ])),
               tau^2 * 2 * kappa^2 * max(abs(G[j, ])), tolerance = 1e-10)
  expect_gt(max(abs(Q_prod - Q_ref)) / max(abs(Q_ref)), 0.05)
  expect_true(cpp_test_spde_nc_transform_Q(
    C0_diag = fem$C0_diag, G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    log_kappa_val = log(kappa), log_tau_val = log(tau))$has_orphans)
})

test_that("both assemblies agree on a healthy mesh too", {
  fem <- .spde_guard_fem()
  kappa <- sqrt(8) / 0.4
  tau   <- 1 / (sqrt(4 * pi) * kappa)
  Q_ref <- .spde_expanded_Q(fem, kappa, tau)
  Q_nc  <- .spde_nc_Q(fem, kappa, tau)
  qb    <- .spde_qb_Q(fem)
  expect_equal(max(abs(Q_nc - Q_ref)), 0, tolerance = 1e-12)
  expect_equal(max(abs(qb$Q - Q_ref)), 0, tolerance = 1e-12)
  expect_equal(max(abs(Q_nc - qb$Q)) / max(abs(qb$Q)), 0, tolerance = 1e-12)

  # With every mass positive the floor never fires, so the product form is the
  # same matrix. That is the whole of what the orphan row breaks.
  expect_equal(max(abs(.spde_product_Q(fem, kappa, tau) - Q_ref)) /
                 max(abs(Q_ref)), 0, tolerance = 1e-12)
})

test_that("the Laplace fit runs to convergence on a mesh with a zero-mass vertex", {
  fem <- .spde_orphan(.spde_guard_fem())
  set.seed(1L)
  w <- stats::rnorm(fem$n_mesh, 0, 0.3); w <- w - mean(w)
  A <- Matrix::sparseMatrix(i = fem$A_i + 1L, p = fem$A_p, x = fem$A_x,
                            dims = c(fem$n_obs, fem$n_mesh), index1 = TRUE)
  eta <- -0.3 + as.numeric(A %*% w)
  y <- stats::rbinom(fem$n_obs, 1L, stats::plogis(eta))

  fit <- cpp_laplace_fit_spde(
    y = as.numeric(y), n_trials = as.integer(rep(1L, fem$n_obs)),
    X = matrix(1, fem$n_obs, 1L),
    re_idx = rep(0, fem$n_obs), n_re_groups = 0L, sigma_re = 1,
    A_x = fem$A_x, A_i = fem$A_i, A_p = fem$A_p,
    n_obs = fem$n_obs, n_mesh = fem$n_mesh, C0_diag = fem$C0_diag,
    G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    kappa = sqrt(8) / 0.4, tau_spde = 1 / (sqrt(4 * pi) * (sqrt(8) / 0.4)),
    family = "binomial", phi = 1, alpha = 2L, max_iter = 100L, tol = 1e-8)

  # Without the ridge Q is singular at the orphan and CHOLMOD reports "not
  # positive definite", so no solver on the model makes progress.
  expect_true(fit$converged)
  expect_true(is.finite(fit$log_marginal))
  expect_false(anyNA(fit$mode))
})

test_that("the non-centered assembly factors the same mesh and its adjoint holds", {
  fem <- .spde_orphan(.spde_guard_fem())
  set.seed(7L)
  z <- stats::rnorm(fem$n_mesh, sd = 0.5)
  kappa <- sqrt(8) / 0.4
  res <- cpp_test_spde_nc_transform_grad(
    C0_diag = fem$C0_diag, G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    z_init = z, log_kappa_val = log(kappa),
    log_tau_val = log(1 / (sqrt(4 * pi) * kappa)), fd_eps = 1e-5)

  # Without the ridge Q is singular at every (kappa, tau) and the forward
  # Cholesky fails on every proposal, while the Laplace path over the same mesh
  # fits -- which is the disagreement the fix exists to prevent.
  expect_true(is.finite(res$L) && res$L > 0)
  expect_false(anyNA(res$grad_z_arena))

  # The ridge is theta-INDEPENDENT, so both hyper derivatives subtract it before
  # applying dQ/dlog_tau = 2 Q. That is the part a finite difference can see.
  rel <- function(a, b) abs(a - b) / max(1e-8, abs(a), abs(b))
  expect_lt(rel(res$grad_log_kappa_arena, res$grad_log_kappa_fd), 1e-4)
  expect_lt(rel(res$grad_log_tau_arena, res$grad_log_tau_fd), 1e-4)
  err <- abs(res$grad_z_arena - res$grad_z_fd)
  sc  <- pmax(1e-8, pmax(abs(res$grad_z_arena), abs(res$grad_z_fd)))
  expect_lt(max(err / sc), 1e-4)
})

test_that("a mesh with no orphan is unchanged by the ridge machinery", {
  # The negative control: the ridge must not touch a healthy mesh, or the
  # assertions above would pass for the wrong reason.
  fem <- .spde_guard_fem()
  expect_true(all(fem$C0_diag > 1e-15))
  set.seed(7L)
  z <- stats::rnorm(fem$n_mesh, sd = 0.5)
  kappa <- sqrt(8) / 0.4
  res <- cpp_test_spde_nc_transform_grad(
    C0_diag = fem$C0_diag, G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    z_init = z, log_kappa_val = log(kappa),
    log_tau_val = log(1 / (sqrt(4 * pi) * kappa)), fd_eps = 1e-5)
  expect_true(is.finite(res$L) && res$L > 0)
  # Zeroing one mass genuinely changes the operator, so the two are not the
  # same fit dressed differently.
  fem2 <- .spde_orphan(.spde_guard_fem())
  res2 <- cpp_test_spde_nc_transform_grad(
    C0_diag = fem2$C0_diag, G1_x = fem2$G1_x, G1_i = fem2$G1_i,
    G1_p = fem2$G1_p, z_init = z, log_kappa_val = log(kappa),
    log_tau_val = log(1 / (sqrt(4 * pi) * kappa)), fd_eps = 1e-5)
  expect_false(isTRUE(all.equal(res$L, res2$L)))
})

# --------------------------------------------------------------------------- #
# 2. The structural validators                                                 #
# --------------------------------------------------------------------------- #

# Both R-callable SPDE entries run spde_validate_operators before touching the
# arrays, so one malformed field is driven through each. The entry is the
# boundary the guard exists for: every loop downstream sizes itself from the
# caller's n_mesh / n_obs rather than from the vectors, so a short C0_diag is an
# out-of-range read and a bad column pointer is an out-of-range WRITE.

.spde_grad_call <- function(fem, ...) {
  args <- list(
    y = as.numeric(rep(0, fem$n_obs)),
    n_trials = as.integer(rep(1L, fem$n_obs)),
    X = matrix(1, fem$n_obs, 1L),
    A_x = fem$A_x, A_i = fem$A_i, A_p = fem$A_p,
    n_obs = fem$n_obs, n_mesh = fem$n_mesh, C0_diag = fem$C0_diag,
    G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    log_range = log(0.4), log_sigma = log(1), nu = 1, family = "binomial")
  do.call(cpp_spde_laplace_gradient, utils::modifyList(args, list(...)))
}

test_that("a short C0_diag is refused, naming both lengths", {
  fem <- .spde_guard_fem()
  expect_error(.spde_grad_call(fem, C0_diag = fem$C0_diag[-1L]),
               "length\\(C0_diag\\)")
  expect_error(.spde_grad_call(fem, C0_diag = fem$C0_diag[-1L]),
               as.character(fem$n_mesh), fixed = TRUE)
})

test_that("a wrong-length G1_p is refused", {
  fem <- .spde_guard_fem()
  expect_error(.spde_grad_call(fem, G1_p = fem$G1_p[-1L]),
               "length\\(G1_p\\)")
})

test_that("mismatched G1 value and row-index lengths are refused", {
  fem <- .spde_guard_fem()
  expect_error(.spde_grad_call(fem, G1_x = fem$G1_x[-1L]),
               "same length")
})

test_that("a malformed G1 column pointer is refused", {
  fem <- .spde_guard_fem()
  # Not starting at 0.
  bad0 <- fem$G1_p; bad0[1L] <- 1L
  expect_error(.spde_grad_call(fem, G1_p = bad0), "must start at 0")
  # Decreasing.
  dec <- fem$G1_p; dec[3L] <- dec[2L] - 1L
  expect_error(.spde_grad_call(fem, G1_p = dec), "non-decreasing")
  # Last pointer disagreeing with the value count.
  last <- fem$G1_p; last[length(last)] <- last[length(last)] - 1L
  expect_error(.spde_grad_call(fem, G1_p = last), "last column pointer")
})

test_that("an out-of-range G1 row index is refused", {
  fem <- .spde_guard_fem()
  bad <- fem$G1_i; bad[1L] <- fem$n_mesh          # 0-based, so n_mesh is past
  expect_error(.spde_grad_call(fem, G1_i = bad), "is outside")
  neg <- fem$G1_i; neg[1L] <- -1L
  expect_error(.spde_grad_call(fem, G1_i = neg), "is outside")
})

test_that("a malformed projector is refused by the same checks", {
  fem <- .spde_guard_fem()
  expect_error(.spde_grad_call(fem, A_p = fem$A_p[-1L]), "length\\(A_p\\)")
  expect_error(.spde_grad_call(fem, A_x = fem$A_x[-1L]), "same length")
  bad <- fem$A_i; bad[1L] <- fem$n_obs            # A's row space is n_obs
  expect_error(.spde_grad_call(fem, A_i = bad), "is outside")
})

test_that("a non-positive n_mesh or n_obs is refused before any array is read", {
  fem <- .spde_guard_fem()
  expect_error(.spde_grad_call(fem, n_mesh = 0L), "n_mesh must be positive")
  expect_error(.spde_grad_call(fem, n_obs = 0L),
               "n_obs must be positive|must equal n_obs")
})

test_that("the Laplace entry validates the same way as the gradient entry", {
  # spde_validate_operators is one definition reached from both entries, so the
  # second is checked once rather than field by field.
  fem <- .spde_guard_fem()
  expect_error(cpp_laplace_fit_spde(
    y = as.numeric(rep(0, fem$n_obs)),
    n_trials = as.integer(rep(1L, fem$n_obs)),
    X = matrix(1, fem$n_obs, 1L),
    re_idx = rep(0, fem$n_obs), n_re_groups = 0L, sigma_re = 1,
    A_x = fem$A_x, A_i = fem$A_i, A_p = fem$A_p,
    n_obs = fem$n_obs, n_mesh = fem$n_mesh,
    C0_diag = fem$C0_diag[-1L],
    G1_x = fem$G1_x, G1_i = fem$G1_i, G1_p = fem$G1_p,
    kappa = 1, tau_spde = 1, family = "binomial"),
    "length\\(C0_diag\\)")
})

# --------------------------------------------------------------------------- #
# 3. Deliberately untested: the indefinite-H guard in the implicit-diff        #
#    gradient (from gcol33/tulpa#534).                                          #
#                                                                              #
# `factorize()`'s return is read and both gradients stay NaN on failure, but no #
# valid input reaches it: the entry refuses a non-PD Q earlier, and A'WA is PSD #
# for every family the entry gates to, so H = Q + A'WA is PD by construction.   #
# It is defensive against numerical failure at extreme conditioning, which no   #
# fixture pins deterministically -- a fixture that reached it would be pinning  #
# the conditioning of a particular BLAS rather than the guard. Recorded here so #
# the absence reads as a decision rather than an oversight.                     #
# --------------------------------------------------------------------------- #
