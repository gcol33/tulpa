# Which per-axis SD estimator to report, and at what resolution.
#
# Two estimators of the same quantity, the posterior SD of one outer axis:
#   * the grid-weighted SD, a quadrature of the axis marginal over its nodes;
#   * the 3-point parabola at the modal node (the shipped `theta_sd`).
#
# The first is consistent as the grid refines and is exact for the measure the
# nodes integrate; the second reads only the curvature at the mode, so it is a
# Gaussian summary and it moves with the spacing of the three nodes it reads.
# The question is where each is right, and what statistic READ OFF THE WEIGHTS
# separates the two regimes -- the issue's own requirement, so the gate does not
# compare one estimator against the other.
#
# Sweeps h / s (node spacing over posterior SD) and the mode's offset inside its
# cell, on a Gaussian axis marginal (where both estimators target the same
# number) and on a skewed one (where the parabola targets a different number by
# construction). Reports the axis quadrature ESS, 1 / sum(p^2), alongside.

suppressMessages(devtools::load_all(".", quiet = TRUE))

# Weighted SD over a set of levels carrying log weights.
wtd_sd <- function(v, lw) {
  m <- max(lw); w <- exp(lw - m); w <- w / sum(w)
  sqrt(max(0, sum(w * v^2) - sum(w * v)^2))
}

axis_ess <- function(lw) {
  m <- max(lw); p <- exp(lw - m); p <- p / sum(p)
  1 / sum(p^2)
}

# ---- Gaussian axis ---------------------------------------------------------
# Marginal N(0, 1) on the integration coordinate; nodes at spacing h, the grid
# offset by `off` * h so the mode is not always on a node.
gauss_row <- function(h, off, K = 41L) {
  u <- (seq_len(K) - (K + 1) / 2) * h + off * h
  lw <- -0.5 * u^2
  s_w <- wtd_sd(u, lw)
  s_p <- .nl_laplace_at_mode_sd_axis(u, lw, log_axis = FALSE, return_u_sd = TRUE)
  data.frame(kind = "gaussian", h = h, off = off,
             ess = axis_ess(lw), sd_wtd = s_w, sd_sten = as.numeric(s_p))
}

# ---- Skewed axis -----------------------------------------------------------
# log-density of a shifted lognormal in the integration coordinate: sharply
# peaked near the mode, heavy to the right, which is the copy-axis shape the
# issue reports (stencil four times below the weighted SD).
skew_row <- function(h, off, K = 41L, s = 0.9) {
  u <- (seq_len(K) - (K + 1) / 2) * h + off * h
  x <- u + 6
  lw <- ifelse(x > 0, -log(pmax(x, 1e-12)) - 0.5 * (log(pmax(x, 1e-12)) / s)^2,
               -Inf)
  # Reference SD of the same density, integrated finely on the same support.
  ug <- seq(min(u), max(u), length.out = 20001L)
  xg <- ug + 6
  lg <- ifelse(xg > 0, -log(pmax(xg, 1e-12)) - 0.5 * (log(pmax(xg, 1e-12)) / s)^2,
               -Inf)
  data.frame(kind = "skew", h = h, off = off,
             ess = axis_ess(lw), sd_wtd = wtd_sd(u, lw),
             sd_sten = as.numeric(.nl_laplace_at_mode_sd_axis(
               u, lw, log_axis = FALSE, return_u_sd = TRUE)),
             sd_ref = wtd_sd(ug, lg))
}

hs   <- c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4)
offs <- c(0, 0.15, 0.3, 0.5)

g <- do.call(rbind, lapply(hs, function(h)
  do.call(rbind, lapply(offs, function(o) gauss_row(h, o)))))
g$err_wtd  <- g$sd_wtd - 1
g$err_sten <- g$sd_sten - 1

cat("== Gaussian axis: both estimators target SD = 1 ==\n")
agg <- do.call(rbind, lapply(split(g, g$h), function(d) data.frame(
  h = d$h[1], ess_min = min(d$ess), ess_max = max(d$ess),
  wtd_min = min(d$sd_wtd), wtd_max = max(d$sd_wtd),
  sten_min = min(d$sd_sten, na.rm = TRUE), sten_max = max(d$sd_sten, na.rm = TRUE))))
print(agg, digits = 4, row.names = FALSE)

cat("\nWorst |weighted SD - 1| by ESS band:\n")
g$band <- cut(g$ess, breaks = c(0, 1.5, 2, 2.5, 3, 4, 6, Inf))
print(do.call(rbind, lapply(split(g, g$band), function(d) if (!nrow(d)) NULL else
  data.frame(band = d$band[1], n = nrow(d),
             worst_wtd = max(abs(d$err_wtd)),
             worst_sten = max(abs(d$err_sten), na.rm = TRUE)))),
  digits = 4, row.names = FALSE)

s <- do.call(rbind, lapply(hs, function(h)
  do.call(rbind, lapply(offs, function(o) skew_row(h, o)))))
cat("\n== Skewed axis: the fine-grid reference is the target ==\n")
s$rel_wtd  <- s$sd_wtd / s$sd_ref
s$rel_sten <- s$sd_sten / s$sd_ref
print(s[, c("h", "off", "ess", "sd_ref", "sd_wtd", "sd_sten", "rel_wtd",
            "rel_sten")], digits = 4, row.names = FALSE)

cat("\n== Grid dependence on one density: spread of each estimator over h ==\n")
for (kind in c("gaussian", "skew")) {
  d <- if (kind == "gaussian") g else s
  fine <- d[d$ess >= 3, ]
  cat(sprintf("%-8s ESS >= 3 rows %2d  weighted range [%.4f, %.4f]  stencil range [%.4f, %.4f]\n",
              kind, nrow(fine), min(fine$sd_wtd), max(fine$sd_wtd),
              min(fine$sd_sten, na.rm = TRUE), max(fine$sd_sten, na.rm = TRUE)))
}
