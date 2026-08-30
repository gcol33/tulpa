# How much of the posterior moves when the grid is refined, against the densely
# integrated answer for the same declared measure.
suppressMessages(devtools::load_all(".", quiet = TRUE))

inner_for <- function(y) { n <- length(y); s2 <- sum(y^2)
  function(h) { s <- as.numeric(h[["sigma"]])
    if (!is.finite(s) || s <= 0) return(list(log_marginal = -Inf))
    list(log_marginal = -n * log(s) - 0.5 * s2 / s^2) } }
LP <- function(s) stats::dexp(s, 1, log = TRUE)

ref_moments <- function(y, lo, hi) {
  s <- seq(lo, hi, length.out = 200001L); f <- inner_for(y)
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
for (slab in list(c(0.05, 20), c(0.2, 6))) {
  r <- ref_moments(y, slab[1], slab[2])
  for (m in c(6L, 9L, 15L)) {
    g <- exp(seq(log(slab[1] * 1.2), log(slab[2] * 0.8), length.out = m))
    a <- fit(y, g, slab, list(adaptive_grid = FALSE,
                              var_of_means_consistency = FALSE))
    b <- fit(y, g, slab, list(adaptive_grid = TRUE,
                              adaptive_grid_edge_thresh = 1e-6,
                              adaptive_grid_max_passes = 3L,
                              var_of_means_consistency = TRUE))
    cat(sprintf("slab [%.2f, %5.2f] m %2d -> ref %.4f/%.4f | pinned %.4f/%.4f (%2d) | refined %.4f/%.4f (%2d)\n",
                slab[1], slab[2], m, r[["mean"]], r[["sd"]],
                a$theta_mean[["sigma"]], a$theta_sd[["sigma"]], nrow(a$theta_grid),
                b$theta_mean[["sigma"]], b$theta_sd[["sigma"]], nrow(b$theta_grid)))
  }
}

cat("\n== the prior alone: a flat inner, so the weights ARE the measure ==\n")
flat <- function(h) list(log_marginal = 0)
for (m in c(5L, 9L, 21L)) {
  g <- exp(seq(log(0.1), log(8), length.out = m))
  sp <- list(hyper_axis_spec("sigma", grid = g, log_scale = TRUE,
                             bounds = c(0, Inf), log_prior = LP,
                             slab_bounds = c(0.05, 20)))
  f <- tulpa_hyper_grid(sp, flat, combine = "none", n_draws = 0L)
  v <- as.numeric(f$theta_grid[, "sigma"])
  cat(sprintf("m %2d  P(sigma > 2) %.4f   mean %.4f\n", m,
              sum(f$weights[v > 2]), f$theta_mean[["sigma"]]))
}
ex <- function(a, b) stats::pexp(b, 1) - stats::pexp(a, 1)
cat(sprintf("exact truncated P(sigma > 2) %.4f  mean %.4f\n",
            ex(2, 20) / ex(0.05, 20),
            {s <- seq(0.05, 20, length.out = 400001L)
             w <- stats::dexp(s, 1); w <- w / sum(w); sum(w * s)}))
