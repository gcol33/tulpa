# test-spde-nc-transform.R
# Sparse-Cholesky non-centered SPDE transform: reverse-mode adjoint vs.
# central differences. The tested object is `spde_nc_transform_arena`,
# which builds Q(kappa, tau) from FEM matrices, factors L L^T = Q, and
# returns w = L^{-T} z as arena Vars whose adjoints flow back to z and
# (log_kappa, log_tau) via the new custom_backward block.
#
# This is the (a.ii) checkpoint for the joint-NUTS SPDE arc — once this
# adjoint matches finite differences, (a.iii) can wire it into the actual
# cpp_tulpa_fit_spde_nuts sampler.

helper_make_small_spde_fem <- function(n_obs = 60, max_edge = c(0.3, 0.6),
                                       cutoff = 0.10, seed = 42L) {
  skip_if_not_installed("fmesher")
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = max_edge,
                              cutoff = cutoff)
  fem  <- fmesher::fm_fem(mesh)
  C0   <- as.numeric(Matrix::diag(fem$c0))
  G1   <- as(fem$g1, "CsparseMatrix")
  list(
    n_mesh = mesh$n,
    C0_diag = C0,
    G1_x    = G1@x,
    G1_i    = G1@i,
    G1_p    = G1@p
  )
}

test_that("SPDE NC transform: arena adjoint matches finite differences", {
  fem <- helper_make_small_spde_fem(n_obs = 80, max_edge = c(0.30, 0.60),
                                    cutoff = 0.10, seed = 42L)
  n   <- fem$n_mesh
  expect_true(n >= 8)

  # Sanity: pick reasonable hypers (range ~0.4, sigma ~1) on the unit square.
  range_true <- 0.4
  sigma_true <- 1.0
  nu         <- 1
  kappa      <- sqrt(8 * nu) / range_true
  tau        <- 1.0 / (sqrt(4 * pi) * kappa * sigma_true)
  log_kappa  <- log(kappa)
  log_tau    <- log(tau)

  set.seed(7L)
  z_init <- rnorm(n, sd = 0.5)

  res <- cpp_test_spde_nc_transform_grad(
    C0_diag       = fem$C0_diag,
    G1_x          = fem$G1_x,
    G1_i          = fem$G1_i,
    G1_p          = fem$G1_p,
    z_init        = z_init,
    log_kappa_val = log_kappa,
    log_tau_val   = log_tau,
    fd_eps        = 1e-5
  )

  # Loss is sum(w^2) — should be positive, finite.
  expect_true(is.finite(res$L) && res$L > 0)

  # Per-element relative error on z.
  err_z <- abs(res$grad_z_arena - res$grad_z_fd)
  scale_z <- pmax(1e-8, pmax(abs(res$grad_z_arena), abs(res$grad_z_fd)))
  rel_z <- err_z / scale_z
  worst <- which.max(rel_z)
  expect_lt(max(rel_z), 1e-4,
            label = sprintf("max rel err in dz: %.3e at idx %d (arena %.6e vs fd %.6e)",
                            max(rel_z), worst,
                            res$grad_z_arena[worst], res$grad_z_fd[worst]))

  # log_kappa and log_tau gradients.
  rel_lk <- abs(res$grad_log_kappa_arena - res$grad_log_kappa_fd) /
              max(1e-8, max(abs(res$grad_log_kappa_arena),
                            abs(res$grad_log_kappa_fd)))
  rel_lt <- abs(res$grad_log_tau_arena - res$grad_log_tau_fd) /
              max(1e-8, max(abs(res$grad_log_tau_arena),
                            abs(res$grad_log_tau_fd)))
  expect_lt(rel_lk, 1e-4,
            label = sprintf("dlog_kappa: arena %.6e vs fd %.6e (rel %.3e)",
                            res$grad_log_kappa_arena,
                            res$grad_log_kappa_fd, rel_lk))
  expect_lt(rel_lt, 1e-4,
            label = sprintf("dlog_tau: arena %.6e vs fd %.6e (rel %.3e)",
                            res$grad_log_tau_arena,
                            res$grad_log_tau_fd, rel_lt))
})
