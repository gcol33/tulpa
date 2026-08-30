# Fixture check for the coarse-vs-refined assertion: a right-skewed marginal
# with an interior mode, read by both estimators on three grids.
suppressMessages(devtools::load_all(".", quiet = TRUE))
k <- 1.5; th <- 2
ld <- function(v) ifelse(v > 0, (k - 1) * log(v) - v / th, -Inf)
truth_sd <- sqrt(k) * th
grids <- list(coarse  = seq(0.2, 12, length.out = 15L),
              fine    = seq(0.2, 12, length.out = 45L),
              shifted = seq(0.2, 12, length.out = 15L) + 0.5 * (11.8 / 14))
for (nm in names(grids)) {
  v <- grids[[nm]]; lm <- ld(v)
  ch <- .nl_axis_sd_choice(v, lm, log_axis = FALSE)
  st <- as.numeric(.nl_laplace_at_mode_sd_axis(v, lm, log_axis = FALSE))
  cat(sprintf("%-8s ess %5.2f  source %-8s sd %.4f  stencil %.4f  (truth %.4f)\n",
              nm, ch$ess, ch$source, ch$sd, st, truth_sd))
}
