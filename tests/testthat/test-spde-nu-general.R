# SPDE smoothness handling: gcol33/tulpa#279, #280, #281.
#
# Together these cover the claim that an integer `nu` other than 1 is a real
# model rather than a relabelled nu = 1 fit:
#   #281  nu <= 0 is refused at the front door (the parameterisation is
#         degenerate there, so no precision exists).
#   #280  the FEM assembly expands K (C^-1 K)^(alpha-1) for the alpha it was
#         asked for, instead of falling through to alpha = 2.
#   #279  the (range, sigma) -> (kappa, tau) map carries nu in BOTH
#         coordinates, so a requested sigma is the field's marginal SD.

# --- fixtures ---------------------------------------------------------------

.nu_fixture <- function(seed = 1L, n = 120L) {
  set.seed(seed)
  coords <- cbind(runif(n), runif(n))
  spatial_spde(coords, max_edge = c(0.12, 0.35))
}

# The stiffness matrix exactly as the C++ side receives it, so the reference
# below is independent of how the mesh builder stores its triangle.
.nu_G <- function(spec) {
  Matrix::sparseMatrix(i = spec$G1_i, p = spec$G1_p, x = spec$G1_x,
                       dims = c(spec$n_mesh, spec$n_mesh), index1 = FALSE)
}

.nu_from_csc <- function(res, n) {
  Matrix::sparseMatrix(i = res$Q_i, p = res$Q_p, x = res$Q_x,
                       dims = c(n, n), index1 = FALSE)
}

# Q = tau^2 K (C^-1 K)^(alpha-1) built as an explicit matrix product, plus the
# orphan ridge. Written from the operator definition rather than from the
# builder's binomial expansion, so agreement is evidence about the expansion.
.nu_ref_Q <- function(spec, kappa, tau, alpha) {
  n  <- spec$n_mesh
  C0 <- as.numeric(spec$C0_diag)
  K  <- kappa^2 * Matrix::Diagonal(n, C0) + .nu_G(spec)
  Cinv <- Matrix::Diagonal(n, ifelse(C0 > 1e-15, 1 / C0, 0))
  Qop <- K
  for (i in seq_len(alpha - 1L)) Qop <- Qop %*% Cinv %*% K
  Q <- tau^2 * Qop
  orph <- which(C0 <= 1e-15)
  if (length(orph)) {
    Q <- Q + Matrix::sparseMatrix(i = orph, j = orph, x = 1, dims = c(n, n))
  }
  Q
}

# --- #281: nu <= 0 is not a model -------------------------------------------

test_that("nu <= 0 is refused with the reason (gcol33/tulpa#281)", {
  coords <- cbind(runif(20), runif(20))
  expect_error(spatial_spde(coords, nu = 0), "degenerate at nu = 0")
  expect_error(spatial_spde(coords, nu = -1), "must be > 0")
  # The message names both broken quantities, so a caller can act on it.
  err <- tryCatch(spatial_spde(coords, nu = 0), error = function(e) conditionMessage(e))
  expect_match(err, "kappa")
  expect_match(err, "tau")
})

# --- #279: nu enters both kappa and tau --------------------------------------

test_that("the C++ (range, sigma) -> (kappa, tau) map matches the R one at every nu", {
  spec <- .nu_fixture()
  for (nu in c(0.5, 1, 1.5, 2, 3)) {
    for (range in c(0.2, 0.8)) for (sigma in c(0.4, 1.7)) {
      alpha <- max(1L, as.integer(round(nu)) + 1L)
      res <- cpp_test_spde_assemble(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
                                    range = range, sigma = sigma, nu = nu,
                                    alpha = alpha)
      kt <- tulpa:::.spde_kappa_tau(range, sigma, nu)
      expect_equal(res$kappa, kt$kappa, tolerance = 1e-14,
                   info = sprintf("nu=%g range=%g", nu, range))
      expect_equal(res$tau, kt$tau_spde, tolerance = 1e-14,
                   info = sprintf("nu=%g sigma=%g", nu, sigma))
    }
  }
})

test_that("nu = 1 keeps the old hardcoded conversion exactly (gcol33/tulpa#279)", {
  # The nu-general tau must reduce to 1 / (sqrt(4 pi) kappa sigma) at nu = 1,
  # so the default SPDE path is unmoved by the fix.
  spec <- .nu_fixture()
  for (range in c(0.2, 0.5, 1.3)) for (sigma in c(0.3, 1.0)) {
    res <- cpp_test_spde_assemble(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
                                  range = range, sigma = sigma, nu = 1,
                                  alpha = 2L)
    kappa_old <- sqrt(8 * 1) / range
    expect_identical(res$kappa, kappa_old)
    expect_identical(res$tau, 1 / (sqrt(4 * pi) * kappa_old * sigma))
  }
})

test_that("(kappa, tau) invert the d = 2 Matern marginal variance", {
  # sigma^2 = 1 / (4 pi nu kappa^(2 nu) tau^2). A nu-independent tau satisfies
  # this only at nu = 1.
  spec <- .nu_fixture()
  for (nu in c(0.5, 1, 2, 3)) {
    res <- cpp_test_spde_assemble(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
                                  range = 0.6, sigma = 0.75, nu = nu,
                                  alpha = max(1L, as.integer(round(nu)) + 1L))
    var_closed <- 1 / (4 * pi * nu * res$kappa^(2 * nu) * res$tau^2)
    expect_equal(sqrt(var_closed), 0.75, tolerance = 1e-10,
                 info = sprintf("nu=%g", nu))
  }
})

# --- #280: the assembly is the alpha it was asked for ------------------------

test_that("the FEM assembly reproduces K (C^-1 K)^(alpha-1) at every integer alpha", {
  spec <- .nu_fixture()
  n <- spec$n_mesh
  for (alpha in 1:4) {
    nu  <- max(alpha - 1, 1e-6)
    res <- cpp_test_spde_assemble(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
                                  range = 0.4, sigma = 0.8, nu = nu,
                                  alpha = alpha)
    expect_identical(res$alpha, alpha)
    Qc <- .nu_from_csc(res, n)
    Qr <- .nu_ref_Q(spec, res$kappa, res$tau, alpha)
    expect_lt(max(abs(Qc - Qr)) / max(abs(Qr)), 1e-12)
  }
})

test_that("alpha >= 3 is assembled, not silently folded into alpha = 2 (gcol33/tulpa#280)", {
  # The pre-fix `if (alpha == 1) ... else <alpha 2>` returned the alpha = 2
  # precision for every alpha >= 2, on the alpha = 2 sparsity pattern. The
  # higher-order operator has a strictly wider stencil, so the pattern alone
  # separates them.
  spec <- .nu_fixture()
  n <- spec$n_mesh
  args <- list(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
               range = 0.4, sigma = 0.8, nu = 2)
  a2 <- do.call(cpp_test_spde_assemble, c(args, alpha = 2L))
  a3 <- do.call(cpp_test_spde_assemble, c(args, alpha = 3L))
  a4 <- do.call(cpp_test_spde_assemble, c(args, alpha = 4L))
  expect_gt(length(a3$Q_x), length(a2$Q_x))
  expect_gt(length(a4$Q_x), length(a3$Q_x))
  expect_gt(max(abs(.nu_from_csc(a3, n) - .nu_from_csc(a2, n))), 1e-6)

  # alpha < 1 has no operator to build.
  expect_error(do.call(cpp_test_spde_assemble, c(args, alpha = 0L)),
               "alpha >= 1")
})

test_that("the R precision mirror agrees with the compiled assembly at nu = 1 and nu = 2", {
  # .spde_precision_Q (the analytic marginal-SE path) and the compiled builder
  # are two implementations of the same operator; #279/#280 had them disagree
  # at nu = 2 on both the tau map and the assembly order.
  spec_shared <- .nu_fixture()
  n <- spec_shared$n_mesh
  for (nu in c(1, 2)) {
    spec <- spec_shared
    spec$nu <- nu
    alpha <- as.integer(round(nu)) + 1L
    res <- cpp_test_spde_assemble(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
                                  range = 0.45, sigma = 0.9, nu = nu,
                                  alpha = alpha)
    kt <- tulpa:::.spde_kappa_tau(0.45, 0.9, nu)
    Qr <- tulpa:::.spde_precision_Q(spec, kt$kappa, kt$tau_spde)
    Qc <- .nu_from_csc(res, n)
    expect_lt(max(abs(Qc - Qr)) / max(abs(Qr)), 1e-12,
              label = sprintf("nu=%g precision mismatch", nu))
  }
})

# --- #279 + #280 together: a requested sigma is the field's marginal SD ------

test_that("the assembled field carries the requested marginal SD at nu = 1 and nu = 2", {
  skip_on_cran()
  # The end the two fixes serve: solve the assembled precision and read the
  # marginal SD at interior mesh nodes. The FEM discretisation inflates it by a
  # mesh-dependent factor, but that factor must not depend on nu -- a wrong tau
  # or a wrong operator order shows up as a factor that moves with nu.
  spec <- .nu_fixture()
  n  <- spec$n_mesh
  mv <- spec$mesh$vertices
  interior <- mv[, 1] > 0.15 & mv[, 1] < 0.85 & mv[, 2] > 0.15 & mv[, 2] < 0.85
  sigma_target <- 0.8

  ratio_at <- function(nu, tau_override = NULL) {
    alpha <- as.integer(round(nu)) + 1L
    res <- cpp_test_spde_assemble(spec$C0_diag, spec$G1_x, spec$G1_i, spec$G1_p,
                                  range = 0.4, sigma = sigma_target, nu = nu,
                                  alpha = alpha)
    tau <- tau_override %||% res$tau
    Q <- .nu_ref_Q(spec, res$kappa, tau, alpha)
    v <- diag(solve(as.matrix(Q)))
    sqrt(median(v[interior])) / sigma_target
  }

  r1 <- ratio_at(1)
  r2 <- ratio_at(2)
  expect_gt(r1, 0.8); expect_lt(r1, 1.3)
  expect_gt(r2, 0.8); expect_lt(r2, 1.3)
  expect_lt(abs(r2 / r1 - 1), 0.10)

  # What the pre-fix nu-independent tau produced at nu = 2: the same operator
  # scaled by a tau roughly 14x too large, so the marginal SD collapses.
  kappa2   <- sqrt(8 * 2) / 0.4
  tau_old  <- 1 / (sqrt(4 * pi) * kappa2 * sigma_target)
  expect_gt(tau_old / tulpa:::.spde_kappa_tau(0.4, sigma_target, 2)$tau_spde, 10)
  expect_lt(ratio_at(2, tau_override = tau_old), 0.2)
})

test_that("fit_spde runs the nu = 2 operator end to end (gcol33/tulpa#279, #280)", {
  skip_on_cran()
  # Simulating from the nu = 2 precision and refitting recovers the marginal SD
  # it was drawn at. Before the fix the nu = 2 grid mapped sigma through a
  # nu = 1 tau onto an alpha = 2 operator, so neither the scale nor the
  # smoothness of the fitted field was the requested one.
  set.seed(7)
  n_obs <- 250
  coords <- cbind(runif(n_obs), runif(n_obs))
  sigma_true <- 1.0

  spec <- spatial_spde(coords, max_edge = c(0.1, 0.3), nu = 2)
  kt <- tulpa:::.spde_kappa_tau(0.35, sigma_true, 2)
  Q  <- tulpa:::.spde_precision_Q(spec, kt$kappa, kt$tau_spde)
  L  <- Matrix::Cholesky(Matrix::forceSymmetric(Q), LDL = FALSE, perm = TRUE)
  w  <- as.numeric(Matrix::solve(L, Matrix::solve(L, rnorm(spec$n_mesh),
                                                  system = "Lt"), system = "Pt"))
  eta <- as.numeric(spec$A %*% w)
  y <- rnorm(n_obs, mean = 0.5 + eta, sd = 0.3)

  fit <- fit_spde(y, matrix(1, n_obs, 1), spec, family = "gaussian", phi = 0.09,
                  control = list(n_grid = 7L))
  expect_true(fit$converged)
  expect_true(all(is.finite(fit$log_marginal)))
  expect_gt(fit$sigma, 0.6 * sigma_true)
  expect_lt(fit$sigma, 1.6 * sigma_true)
})

test_that("joint NUTS refuses an integer alpha it cannot differentiate (gcol33/tulpa#280)", {
  # The non-centered transform's adjoint is written for the alpha = 2 assembly.
  # Sampling nu = 2 there would silently be a nu = 1 fit.
  set.seed(3)
  coords <- cbind(runif(30), runif(30))
  spec <- spatial_spde(coords, nu = 2)
  expect_error(
    tulpa_nuts_spde(y = rnorm(30), X = matrix(1, 30, 1), spatial = spec,
                    family = "gaussian", joint = TRUE),
    "alpha = 2"
  )
})

# --- #437: the joint-NUTS hyper prior reads the same map ---------------------

test_that("the PC hyper prior maps (log_kappa, log_tau) to sigma at every nu", {
  # compute_spde_hyper_prior is the only prior on (log_kappa, log_tau) in
  # joint-NUTS mode, and it has to reach sigma through the same map the sampler
  # reports sigma_draws with. The nu = 1 special case
  # sigma = exp(-log_kappa - log_tau) / sqrt(4 pi) drops the nu inside the log
  # and the nu factor on log_kappa, so the PC density P(sigma > sigma_0) =
  # alpha_s is evaluated at a quantity that is not the field's marginal SD --
  # the hyper posterior shifts by (nu - 1) log_kappa + 0.5 log(nu) in the
  # exponent while the reported summaries stay on the correct map.
  #
  # The reference reads .spde_range_sigma(), the one R-side map, rather than
  # writing the formula out again.
  pc_ref <- function(log_kappa, log_tau, nu, range_0, alpha_r,
                     sigma_0, alpha_s) {
    rs    <- tulpa:::.spde_range_sigma(exp(log_kappa), exp(log_tau), nu)
    range <- rs$range
    sigma <- rs$sigma
    lambda_r <- -log(alpha_r) * range_0
    lambda_s <- -log(alpha_s) / sigma_0
    log_pi_range <- log(lambda_r) - 2 * log(range) - lambda_r / range
    log_pi_sigma <- log(lambda_s) - lambda_s * sigma
    # log|J| = log_range + log_sigma: J is triangular for any nu, so the
    # Jacobian is what the fix leaves alone.
    log_pi_range + log_pi_sigma + log(range) + log(sigma)
  }

  grid <- expand.grid(log_kappa = c(-1.0, 0.0, 0.7, 1.5),
                      log_tau   = c(-1.2, 0.0, 0.4),
                      nu        = c(1, 2, 3))
  for (i in seq_len(nrow(grid))) {
    got <- tulpa:::cpp_spde_hyper_prior_probe(
      log_kappa = grid$log_kappa[i], log_tau = grid$log_tau[i],
      nu = grid$nu[i],
      prior_range_0 = 0.2, prior_range_alpha = 0.05,
      prior_sigma_0 = 1.0, prior_sigma_alpha = 0.05
    )$prior_val
    expect_equal(got,
                 pc_ref(grid$log_kappa[i], grid$log_tau[i], grid$nu[i],
                        range_0 = 0.2, alpha_r = 0.05,
                        sigma_0 = 1.0, alpha_s = 0.05),
                 tolerance = 1e-10,
                 info = sprintf("nu=%g log_kappa=%g log_tau=%g",
                                grid$nu[i], grid$log_kappa[i], grid$log_tau[i]))
  }
})

test_that("nu != 1 moves the hyper prior away from the nu = 1 map", {
  # The negative control for the test above: at nu = 1 the two references
  # coincide, so a test run only at nu = 1 passes against either map. This
  # states how far apart they are at nu = 2, which is reachable with no user
  # action (cpp_tulpa_fit_spde_nuts defaults nu to alpha - 1, so alpha = 3
  # gives nu = 2).
  nu1_sigma_map <- function(log_kappa, log_tau) {
    exp(-log_kappa - log_tau) / sqrt(4 * pi)
  }
  for (nu in c(2, 3)) {
    lk <- 0.7; lt <- -0.4
    rs <- tulpa:::.spde_range_sigma(exp(lk), exp(lt), nu)
    expect_false(isTRUE(all.equal(rs$sigma, nu1_sigma_map(lk, lt))),
                 info = sprintf("nu=%g", nu))
  }
  # ... and at nu = 1 they are the same number, so nothing about the default
  # path moved.
  expect_equal(tulpa:::.spde_range_sigma(exp(0.7), exp(-0.4), 1)$sigma,
               nu1_sigma_map(0.7, -0.4), tolerance = 1e-14)
})

test_that("the hyper prior is flat when either hyper slot is absent", {
  # compute_spde_hyper_prior reads params[log_tau_spde_idx] after checking
  # log_kappa_spde_idx, so a layout carrying one slot and not the other used to
  # index at -1. Both guards are in place; with joint_hypers off the layout
  # reserves neither and the prior is flat.
  expect_equal(
    tulpa:::cpp_spde_hyper_prior_probe(
      log_kappa = 0.3, log_tau = -0.2, nu = 2,
      prior_range_0 = 0.2, prior_range_alpha = 0.05,
      prior_sigma_0 = 1.0, prior_sigma_alpha = -1.0
    )$prior_val, 0.0, tolerance = 1e-12)
})
