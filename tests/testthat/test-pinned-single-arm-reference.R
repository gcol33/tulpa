skip_on_cran()

for (type in c("icar", "bym2", "rw1")) {
  for (n_s in c(40L, 250L)) {
    test_that(sprintf("a weighted %s field retains its level variance (%d nodes)", type, n_s), {
      skip_on_cran()
      set.seed(19)
      idx <- rep(seq_len(n_s), each = 6L)
      N <- length(idx)
      x <- rnorm(N)
      X <- cbind(`(Intercept)` = 1, x = x)
      phi <- 0.25
      tau <- 2
      field <- cumsum(rnorm(n_s, sd = 0.1))
      y <- rnorm(N, 0.2 + x * (0.5 + field[idx]), sqrt(phi))
      nbr <- lapply(seq_len(n_s), function(s) {
        setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L))
      })
      nn <- vapply(nbr, length, integer(1))
      prior <- list(type = "icar", n_spatial_units = n_s,
                    adj_row_ptr = as.integer(c(0L, cumsum(nn))),
                    adj_col_idx = as.integer(unlist(nbr)) - 1L,
                    n_neighbors = nn, tau_grid = tau, spatial_idx = idx,
                    svc_weight = x)
      if (type == "rw1") {
        prior <- list(type = "rw1", n_times = n_s, temporal_idx = idx,
                      tau_grid = tau, svc_weight = x)
      }

      # The field's constant direction has precision tau under the augmented
      # ICAR prior. No level row transfers that direction to the fixed effects.
      S <- outer(idx, seq_len(n_s), "==") * x
      A <- cbind(X, S)
      R <- crossprod(diff(diag(n_s)))
      fb <- 2L + seq_len(n_s)
      field_precision <- tau * (R + 1 / n_s)
      field_load <- diag(n_s)
      pin_precision <- tau / n_s
      if (type == "bym2") {
        prior$type <- "bym2"
        prior$tau_grid <- NULL
        prior$sigma_grid <- 0.7
        prior$rho_grid <- 0.4
        prior$scale_factor <- 0.8
        a <- 0.7 * sqrt(0.4) * 0.8
        b <- 0.7 * sqrt(1 - 0.4)
        A <- cbind(X, a * S, b * S)
        field_precision <- matrix(0, 2L * n_s, 2L * n_s)
        field_precision[seq_len(n_s), seq_len(n_s)] <- R + 1 / n_s
        ub <- n_s + seq_len(n_s)
        field_precision[ub, ub] <- diag(n_s)
        field_load <- cbind(a * diag(n_s), b * diag(n_s))
        pin_precision <- 1 / n_s
      }
      P <- crossprod(A) / phi
      P[1:2, 1:2] <- P[1:2, 1:2] + diag(1 / 2.5^2, 2L)
      fc <- 3L:ncol(A)
      P[fc, fc] <- P[fc, fc] + field_precision
      V <- solve(P)
      m <- drop(V %*% crossprod(A, y) / phi)
      sd_ref <- sqrt(diag(V))

      captured <- new.env(parent = emptyenv())
      attach <- tulpa:::.nl_attach_grid_hessians
      testthat::local_mocked_bindings(
        .nl_attach_grid_hessians = function(res, ...) {
          captured$res <- res
          attach(res, ...)
        }, .package = "tulpa")
      fit <- suppressWarnings(tulpa_nested_laplace(
        y, rep(1L, N), X, prior = list(prior), family = "gaussian", phi = phi,
        control = list(max_iter = 60L, tol = 1e-12, axis_refine = "none",
                       auto_recenter = FALSE, prune = FALSE,
                       diagnose_k = FALSE, diagnose_skew = FALSE)))
      raw <- captured$res
      expect_true(all(raw$converged))
      expect_equal(ncol(raw$modes), ncol(A))
      expect_length(raw$constraint_cols, 0L)
      expect_length(raw$Q_soft_cols, if (n_s == 250L) 1L else 0L)
      if (n_s == 250L) {
        expect_identical(raw$Q_soft_cols[[1L]], fb)
        expect_equal(as.numeric(raw$Q_soft_D_per_grid[[1L]]), pin_precision)
      }
      cons <- tulpa:::.inner_constraint_cpp(tulpa:::.inner_constraint(raw))
      got <- tulpa:::cpp_joint_inner_vcov_blocks(
        raw$Q_csc_p_per_grid, raw$Q_csc_i_per_grid, raw$Q_csc_x_per_grid,
        n_x = ncol(A), idx = seq_len(ncol(A)), n_dense = ncol(A),
        A_cols_list = cons$A_cols_list, field_marginal = FALSE, n_threads = 1L,
        soft_cols_list = cons$soft_cols_list,
        soft_D_per_grid = cons$soft_D_per_grid)[[1L]]
      expect_lt(max(abs((got - V) / outer(sd_ref, sd_ref))), 1e-8)
      expect_lt(abs(sqrt(sum(got[fb, fb]) / sum(V[fb, fb])) - 1), 1e-8)
      expect_lt(max(abs(fit$modes[1L, ] - m) / sd_ref), 1e-8)
      expect_lt(max(abs(sqrt(diag(vcov(fit))) / sd_ref[1:2] - 1)), 1e-8)
      expect_lt(max(abs(sqrt(diag(solve(fit$grid_hessians[[1L]]))) /
                          sd_ref[1:2] - 1)), 1e-8)
      if (type != "rw1") {
        field_sd <- sqrt(diag(field_load %*% V[fc, fc] %*% t(field_load)))
        expect_lt(max(abs(sqrt(fit$grid_spatial_var[1L, ]) / field_sd - 1)), 1e-8)
        expect_lt(max(abs(fit$grid_spatial_mean[1L, ] -
                            drop(field_load %*% m[fc])) / field_sd), 1e-8)
      }
      eta_sd <- sqrt(rowSums((A %*% V) * A))
      expect_lt(max(abs(fit$fitted_eta[1L, ] - drop(A %*% m)) / eta_sd), 1e-8)
      expect_lt(max(abs(sqrt(fit$fitted_eta_var[1L, ]) / eta_sd - 1)), 1e-8)
    })
  }
}
