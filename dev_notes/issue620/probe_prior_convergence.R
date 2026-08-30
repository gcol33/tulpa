# Does the node set converge to the declared prior when it spans the support?
suppressMessages(devtools::load_all(".", quiet = TRUE))
LP   <- function(s) stats::dexp(s, 1, log = TRUE)
flat <- function(h) list(log_marginal = 0)
SLAB <- c(0.05, 20)

u <- seq(log(SLAB[1]), log(SLAB[2]), length.out = 400001L)
w <- exp(LP(exp(u))); w <- w / sum(w)
ref <- sum(w * exp(u))
cat(sprintf("reference (density on log sigma over the slab) mean %.5f\n", ref))

for (span in list(c(0.1, 8), SLAB)) {
  for (m in c(5L, 11L, 21L, 41L, 81L)) {
    g <- exp(seq(log(span[1]), log(span[2]), length.out = m))
    sp <- list(hyper_axis_spec("sigma", grid = g, log_scale = TRUE,
                               bounds = c(0, Inf), log_prior = LP,
                               slab_bounds = SLAB))
    f <- tulpa_hyper_grid(sp, flat, combine = "none", n_draws = 0L)
    cat(sprintf("span [%.2f, %5.2f] m %3d  mean %.5f  err %.5f\n",
                span[1], span[2], m, f$theta_mean[["sigma"]],
                abs(f$theta_mean[["sigma"]] - ref)))
  }
}
