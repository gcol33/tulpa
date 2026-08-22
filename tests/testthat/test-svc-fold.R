# Weighted-entry (areal SVC / temporal TVC) intrinsic blocks alias their constant
# null direction with the COEFFICIENT on the covariate they ride on, not the arm
# intercept (gcol33/tulpa#242). A weighted field contributes eta_i += w_i * f[s_i],
# so a constant added to the field shifts eta along the covariate column w, not
# uniformly. The centerer therefore folds the field's global level into the
# declared covariate column (svc_beta_offset), where the coefficient absorbs it;
# folding into the intercept (the old behavior) would corrupt the reported mode's
# eta reconstruction. The fold is a post-hoc, eta-preserving reparameterization of
# the reported mode, so it leaves the log-marginal untouched.

.chain_adj_svc <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn),
       n_spatial_units = n_s)
}

.svc_icar_block <- function(adj, s_idx, w, tau_grid, beta_offset = NULL) {
  b <- list(type = "icar",
            n_spatial_units = adj$n_spatial_units,
            adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
            n_neighbors = adj$n_neighbors,
            tau_grid = tau_grid,
            spatial_idx = list(s_idx),
            svc_weight = list(w))
  if (!is.null(beta_offset)) b$svc_beta_offset <- beta_offset
  b
}

.svc_fit <- function(arm, block) {
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = list(block),
    control = list(max_iter = 60L, tol = 1e-7)))
}

# Weighted mean over the outer grid of a slice of the mode matrix.
.wmean_cols <- function(fit, cols) {
  w <- fit$weights
  as.numeric(crossprod(w, fit$modes[, cols, drop = FALSE]))
}

test_that("areal SVC: the constant folds into the covariate coefficient, preserving eta", {
  skip_on_cran()
  set.seed(1)
  n_s <- 40L
  adj <- .chain_adj_svc(n_s)
  f <- cumsum(rnorm(n_s)); f <- f - mean(f); f <- f / stats::sd(f)   # centered field
  level <- 1.5                                                       # x-slope level
  g <- f + level                                                     # the field the data sees
  N <- 800L
  s <- sample.int(n_s, N, replace = TRUE)
  x <- rnorm(N)
  # eta = intercept + x * g[s]: the whole x main effect lives in g's level, so a
  # correct fold must move `level` into the x coefficient (not the intercept).
  eta <- 0.3 + x * g[s]
  y <- rnorm(N, eta, 0.4)
  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "gaussian", phi = 1.0)
  tau_grid <- 1 / c(0.6, 1.0, 1.4)^2

  # x is column 1 (0-based) of X; the field rides on x, so its constant aliases
  # with beta[1] (the x coefficient), NOT beta[0] (the intercept).
  fit_fold <- .svc_fit(arm, .svc_icar_block(adj, s, x, tau_grid, beta_offset = 1L))
  fit_none <- .svc_fit(arm, .svc_icar_block(adj, s, x, tau_grid))

  bstart <- fit_fold$arm_layout$beta_start[[1L]]     # 0-based
  fstart <- fit_fold$arm_layout$field_starts[[1L]]   # 0-based
  b0_fold  <- .wmean_cols(fit_fold, bstart + 1L)     # intercept (0-based col 0)
  b1_fold  <- .wmean_cols(fit_fold, bstart + 2L)     # x coefficient (0-based col 1)
  b0_none  <- .wmean_cols(fit_none, bstart + 1L)
  b1_none  <- .wmean_cols(fit_none, bstart + 2L)
  f_fold   <- .wmean_cols(fit_fold, fstart + seq_len(n_s))
  f_none   <- .wmean_cols(fit_none, fstart + seq_len(n_s))

  # (1) The fold is post-hoc and eta-preserving: it does not touch the inner
  # Laplace marginal, so the per-cell log-marginal is identical to the no-fold fit.
  expect_equal(fit_fold$log_marginal, fit_none$log_marginal, tolerance = 1e-8)

  # (2) Reconstructing the fitted eta from the reported mode (intercept + b1 x +
  # x f[s]) gives the SAME values whether or not the level was folded out -- the
  # fold moved it into the x column, which multiplies the same weight the field
  # does, so eta is preserved. Folding into the intercept instead would shift the
  # reconstruction by level*(1 - x_i).
  eta_fold <- b0_fold + b1_fold * x + x * f_fold[s]
  eta_none <- b0_none + b1_none * x + x * f_none[s]
  expect_lt(max(abs(eta_fold - eta_none)), 1e-5)

  # (3) With the fold the reported field is centred (its global constant removed)
  # and the x coefficient absorbs the level, recovering the true x-slope (1.5).
  expect_lt(abs(mean(f_fold)), 1e-8)
  expect_lt(abs(b1_fold - level), 0.4)               # aliased-coefficient recovery

  # (4) The field SHAPE recovers under either reporting (correlation is shift-
  # invariant): the fold changes only where the level is reported, not the fit.
  expect_gt(abs(cor(f_fold, f)), 0.7)
  expect_gt(abs(cor(f_none, f)), 0.7)
})

test_that("areal SVC: a declared column that does not match the weight errors", {
  set.seed(2)
  n_s <- 20L
  adj <- .chain_adj_svc(n_s)
  N <- 200L
  s <- sample.int(n_s, N, replace = TRUE)
  x <- rnorm(N)
  y <- rnorm(N, 0.2 + x, 0.4)
  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "gaussian", phi = 1.0)
  # The field rides on x, but declare the INTERCEPT column (0, all-ones) as its
  # alias -- a mismatch that would silently shift eta. It must error, not fit.
  expect_error(
    .svc_fit(arm, .svc_icar_block(adj, s, x, 1 / c(0.8, 1.2)^2, beta_offset = 0L)),
    "must carry the field weight")
})
