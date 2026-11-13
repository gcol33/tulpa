.subspace_exact_data <- function() {
  list(y = c(0, 0, 0, 1, 1, 4, 0, 0), n_trials = rep(1L, 8),
       X = cbind(intercept = 1, slope = rep(c(-1, 1), 4)))
}

.subspace_exact_block <- function(cpp_file = testthat::test_path(
    "fixtures", "tgmrf_subspace_slope.cpp"), cache_dir = file.path(tempdir(), "subspace-slope")) {
  tgmrf_cpp(cpp_file, "tgmrf_subspace_slope", init = c(log_sigma = .5),
             cache_dir = cache_dir, obs_idx = seq_len(8L))
}

.subspace_exact_fit <- function(block, method, seed, n_grid = 101L,
                                 n_iter = 8000L, warmup = 1000L,
                                 n_draws = 10000L) {
  d <- .subspace_exact_data()
  stopifnot(length(d$y) == 8L, nrow(d$X) == 8L, block$n_latent == 8L)
  block$theta_grid_built <- matrix(seq(-2, 3, length.out = n_grid), ncol = 1L,
                                  dimnames = list(NULL, "log_sigma"))
  set.seed(seed)
  tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = block, family = "poisson",
    control = list(max_iter = 100L, tol = 1e-8, diagnose_k = FALSE,
      diagnose_skew = FALSE, skew_correct = FALSE, progress = FALSE,
      axis_refine = "none", subspace_debias = list(method = method, idx = 1L,
        n_iter = n_iter, warmup = warmup, thin = 2L, n_draws = n_draws)))
}

.subspace_fast_joint <- function(fit, block, n_draws) {
  d <- .subspace_exact_data()
  lm <- tulpa:::.tgmrf_make_log_marginal(d$y, d$n_trials, d$X, block, seq_len(8L),
    NULL, 0L, 1, "poisson", 1, 100L, 1e-8, 1L)
  record <- fit$subspace_debias
  picks <- sample.int(length(fit$weights), n_draws, replace = TRUE, prob = fit$weights)
  out <- matrix(NA_real_, n_draws, 10L)
  for (k in unique(picks)) {
    at <- which(picks == k)
    proposal <- lm$conditional(as.numeric(fit$theta_grid[k, ]))
    out[at, ] <- tulpa:::.subspace_node_draws(record$cell_modes[k, ],
      as.matrix(Matrix::solve(proposal$precision)), record$idx,
      record$cell_draws[[k]], length(at))
  }
  out
}

# Two observed predictor values identify each group's intercept and slope.
.subspace_slope_coordinates <- function(latent) {
  out <- latent
  for (g in seq_len(4L)) {
    cols <- 2L + c(2L * g - 1L, 2L * g)
    out[, cols[1L]] <- (latent[, cols[1L]] + latent[, cols[2L]]) / 2
    out[, cols[2L]] <- (latent[, cols[2L]] - latent[, cols[1L]]) / 2
  }
  colnames(out) <- c("beta_intercept", "beta_slope",
    as.vector(rbind(paste0("b0_g", 1:4), paste0("b1_g", 1:4))))
  out
}

.subspace_summary_values <- function(x) {
  c(mean = mean(x), sd = sd(x), q025 = unname(quantile(x, .025)),
     median = median(x), q975 = unname(quantile(x, .975)))
}

.subspace_fit_statistics <- function(fit, latent) {
  x <- .subspace_slope_coordinates(latent)
  out <- vapply(seq_len(ncol(x)), function(j) .subspace_summary_values(x[, j]), numeric(5))
  colnames(out) <- colnames(x)
  cbind(out, log_sigma = c(fit$theta_mean[1], fit$theta_sd[1],
    fit$theta_ci_lo[1], fit$theta_median[1], fit$theta_ci_hi[1]))
}

.subspace_reference_statistics <- function(reference, n_batches = 40L) {
  latent <- .subspace_slope_coordinates(cbind(reference$draws_beta, reference$draws_z))
  x <- cbind(latent, log_sigma = reference$draws_theta[, 1])
  value <- vapply(seq_len(ncol(x)), function(j) .subspace_summary_values(x[, j]), numeric(5))
  colnames(value) <- colnames(x)
  groups <- cut(seq_len(nrow(x)), n_batches, labels = FALSE)
  mcse <- vapply(seq_len(ncol(x)), function(j) {
    batches <- vapply(split(x[, j], groups), .subspace_summary_values, numeric(5))
    apply(batches, 1L, sd) / sqrt(ncol(batches))
  }, numeric(5))
  dimnames(mcse) <- dimnames(value)
  list(value = value, mcse = mcse)
}
