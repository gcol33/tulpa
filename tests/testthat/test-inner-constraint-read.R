# Every read of a joint fit's stored per-cell precision -- the posterior draws,
# the mixture moments, the retained fixed-effect block -- reads the Gaussian
# the inner solve built: the stored Q plus the sum-to-zero pins a sparse solve
# folds at solve time (`fit$Q_soft_cols`, `fit$Q_soft_D_per_grid`),
# conditioned on the hard rows the solve carried (`fit$constraint_cols`). A
# pinned intrinsic field with no level row (a varying-coefficient ICAR whose
# covariate column is not declared) keeps its level under the augmented
# prior's precision; conditioning it to zero drew the slope it aliases with at
# 0.29 of its posterior SD.
#
# The fixture is gaussian with a known residual variance on one field
# precision, so the inner Gaussian is the exact posterior and the reference is
# the dense inverse of the full precision.

skip_on_cran()

.icr_chain <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

.icr_case <- function(n_s, seed = 1) {
  set.seed(seed)
  adj <- .icr_chain(n_s)
  N <- 6L * n_s
  s <- sample.int(n_s, N, replace = TRUE)
  x <- rnorm(N)
  f <- cumsum(rnorm(n_s)) * 0.1
  phi <- 0.25
  tau <- 2
  y <- rnorm(N, 0.2 + x * (0.5 + f[s]), sqrt(phi))
  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "gaussian", phi = phi)
  blk <- list(type = "icar", n_spatial_units = n_s,
              adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
              n_neighbors = adj$n_neighbors, tau_grid = tau,
              spatial_idx = list(s), svc_weight = list(x))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = list(blk),
    control = list(max_iter = 60L, tol = 1e-9, store_Q = TRUE,
                   progress = FALSE)))
  # Exact posterior precision: likelihood, the fixed effects' N(0, 2.5^2)
  # prior, and the field's augmented ICAR prior tau (R + 1 1' / n).
  A <- cbind(1, x, outer(s, seq_len(n_s), "==") * x)
  R <- -outer(seq_len(n_s), seq_len(n_s), function(i, j) abs(i - j) == 1) * 1
  diag(R) <- adj$n_neighbors
  fb <- 2L + seq_len(n_s)
  P <- crossprod(A) / phi
  P[1:2, 1:2] <- P[1:2, 1:2] + diag(1 / 2.5^2, 2)
  P[fb, fb] <- P[fb, fb] + tau * R + tau / n_s
  list(fit = fit, V = solve(P), m = drop(solve(P, crossprod(A, y) / phi)),
       fb = fb)
}

for (n_s in c(40L, 250L)) {
  test_that(sprintf("a pinned field with no level row reads its own Gaussian (%d nodes)", n_s), {
    cs <- .icr_case(n_s)
    fit <- cs$fit
    # 40 nodes: the dense driver writes the pin into Q. 250: the sparse
    # driver folds it, and it travels as a soft row.
    expect_length(fit$constraint_cols, 0L)
    expect_length(fit$Q_soft_cols, if (n_s == 250L) 1L else 0L)
    expect_lt(max(abs(fit$modes[1, ] - cs$m)), 1e-8)

    sd_ex <- sqrt(diag(cs$V))
    mm <- tulpa:::.joint_mixture_moments(fit)
    expect_lt(max(abs(sqrt(diag(mm$Sigma)) / sd_ex - 1)), 1e-8)
    blk <- solve(fit$grid_hessians[[1L]])
    expect_lt(max(abs(sqrt(diag(blk)) / sd_ex[1:2] - 1)), 1e-8)

    set.seed(3)
    dr <- tulpa_posterior_draws(fit, n = 8000L)
    # 8000 draws: an SD ratio carries a standard error of about 0.008.
    expect_lt(max(abs(apply(dr, 2, stats::sd) / sd_ex - 1)), 0.04)
    lev <- rowMeans(dr[, cs$fb])
    lev_sd <- sqrt(sum(cs$V[cs$fb, cs$fb]) / n_s^2)
    expect_lt(abs(stats::sd(lev) / lev_sd - 1), 0.04)
  })
}
