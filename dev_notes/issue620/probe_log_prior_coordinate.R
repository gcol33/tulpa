# Which coordinate a spec's `log_prior` is read as a density on.
#
# `.hyper_axis_level_weights()` carries a declared `slab_log_density` to the
# integration coordinate explicitly (`ld + log(x)` on a log axis). A spec's
# `log_prior` goes the other way in: it is added to `log_marginal` by `hp_fn`,
# and the cell widths it is multiplied by are widths in `log x`. So the two
# declarations are read on two coordinates. This measures which.
suppressMessages(devtools::load_all(".", quiet = TRUE))

LP   <- function(s) stats::dexp(s, 1, log = TRUE)
flat <- function(h) list(log_marginal = 0)
SLAB <- c(0.05, 20)

grid_mean <- function(m) {
  g <- exp(seq(log(0.1), log(8), length.out = m))
  sp <- list(hyper_axis_spec("sigma", grid = g, log_scale = TRUE,
                             bounds = c(0, Inf), log_prior = LP,
                             slab_bounds = SLAB))
  f <- tulpa_hyper_grid(sp, flat, combine = "none", n_draws = 0L)
  v <- as.numeric(f$theta_grid[, "sigma"])
  c(mean = f$theta_mean[["sigma"]], p_gt2 = sum(f$weights[v > 2]))
}

# Reference A: the declared density read on the NATURAL coordinate.
s  <- seq(SLAB[1], SLAB[2], length.out = 400001L)
wA <- exp(LP(s)); wA <- wA / sum(wA)
# Reference B: the same expression read as a density on log sigma.
u  <- seq(log(SLAB[1]), log(SLAB[2]), length.out = 400001L)
wB <- exp(LP(exp(u))); wB <- wB / sum(wB)

cat(sprintf("reference A (density on sigma)     mean %.4f  P(>2) %.4f\n",
            sum(wA * s), sum(wA[s > 2])))
cat(sprintf("reference B (density on log sigma) mean %.4f  P(>2) %.4f\n",
            sum(wB * exp(u)), sum(wB[exp(u) > 2])))
for (m in c(5L, 9L, 21L, 61L, 201L)) {
  r <- grid_mean(m)
  cat(sprintf("grid m %3d                         mean %.4f  P(>2) %.4f\n",
              m, r[["mean"]], r[["p_gt2"]]))
}
