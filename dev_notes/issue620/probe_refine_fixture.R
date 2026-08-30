# Pick the fit fixture for the refinement-invariance test: nodes spanning the
# declared support, pinned against adaptively refined, both against the dense
# reference for the same measure.
suppressMessages(devtools::load_all(".", quiet = TRUE))

inner_for <- function(y) { n <- length(y); s2 <- sum(y^2)
  function(h) { s <- as.numeric(h[["sigma"]])
    if (!is.finite(s) || s <= 0) return(list(log_marginal = -Inf))
    list(log_marginal = -n * log(s) - 0.5 * s2 / s^2) } }
LP <- function(s) stats::dexp(s, 1, log = TRUE)

ref_moments <- function(y, slab) {
  u <- seq(log(slab[1]), log(slab[2]), length.out = 200001L); s <- exp(u)
  f <- inner_for(y)
  lm <- vapply(s, function(v) f(c(sigma = v))$log_marginal, numeric(1)) + LP(s)
  w <- exp(lm - max(lm)); w <- w / sum(w)
  c(mean = sum(w * s), sd = sqrt(sum(w * s^2) - sum(w * s)^2))
}

fit <- function(y, grid, slab, ctrl) {
  sp <- list(hyper_axis_spec("sigma", grid = grid, log_scale = TRUE,
                             bounds = c(0, Inf), refinable = TRUE,
                             log_prior = LP, slab_bounds = slab))
  tulpa_hyper_grid(sp, inner_for(y), combine = "none", n_draws = 0L,
                   control = ctrl)
}

set.seed(620); y <- stats::rnorm(60, 0, 1.3)
slab <- c(0.2, 6)
r <- ref_moments(y, slab)
cat(sprintf("reference mean %.4f sd %.4f\n", r[["mean"]], r[["sd"]]))
for (m in c(7L, 9L, 13L, 21L)) {
  g <- exp(seq(log(slab[1]), log(slab[2]), length.out = m))
  a <- fit(y, g, slab, list(adaptive_grid = FALSE,
                            var_of_means_consistency = FALSE))
  b <- fit(y, g, slab, list(adaptive_grid = TRUE,
                            adaptive_grid_edge_thresh = 1e-6,
                            adaptive_grid_max_passes = 3L,
                            var_of_means_consistency = TRUE))
  cat(sprintf("m %2d | pinned %.4f/%.4f (%2d cells, %.1f%%) | refined %.4f/%.4f (%2d cells, %.1f%%)\n",
              m, a$theta_mean[["sigma"]], a$theta_sd[["sigma"]],
              nrow(a$theta_grid),
              100 * abs(a$theta_mean[["sigma"]] - r[["mean"]]) / r[["mean"]],
              b$theta_mean[["sigma"]], b$theta_sd[["sigma"]],
              nrow(b$theta_grid),
              100 * abs(b$theta_mean[["sigma"]] - r[["mean"]]) / r[["mean"]]))
}
