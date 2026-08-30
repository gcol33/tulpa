# What a boundary node's own weight says about the tail the grid leaves out.
#
# The label #622 asks for names an axis holding a materially large share of its
# own marginal on a boundary node, whether or not that node is the argmax. The
# threshold has to be in the SAME currency as the rail's -- `lift = m * w_edge`,
# the boundary node's weight against what a flat marginal would put there --
# because a fixed share makes a longer axis a weaker detector at the same
# posterior (`.NL_RECENTER$edge_mass_mult`).
#
# What the label is ABOUT is the mass the span does not reach, so the threshold
# is read off that: sweep marginals and spans, record the lift at the outer node
# and the true mass beyond the outer CELL EDGE (half a step past it, the support
# the reported interval extends to), and find where a lift cutoff separates a
# grid that truncates from one that does not.

suppressMessages(devtools::load_all(".", quiet = TRUE))

# One arrangement: `m` nodes on [lo, hi] in the integration coordinate, marginal
# `ld` (log density there). Returns the outer-node lift on each side and the true
# mass beyond each outer cell edge.
arrangement <- function(ld, lo, hi, m) {
  u  <- seq(lo, hi, length.out = m)
  lw <- ld(u)
  w  <- exp(lw - max(lw)); w <- w / sum(w)
  h  <- (hi - lo) / (m - 1)
  # Reference mass, integrated finely well past both edges.
  ug <- seq(lo - 12, hi + 12, length.out = 240001L)
  g  <- exp(ld(ug) - max(ld(ug)))
  tot <- sum(g)
  tail_lo <- sum(g[ug < lo - h / 2]) / tot
  tail_hi <- sum(g[ug > hi + h / 2]) / tot
  data.frame(m = m, lift_lo = m * w[1L], lift_hi = m * w[m],
             tail_lo = tail_lo, tail_hi = tail_hi)
}

densities <- list(
  gaussian  = function(u) -0.5 * u^2,
  laplace   = function(u) -abs(u),
  student3  = function(u) -2 * log1p(u^2 / 3),
  lognormal = function(u) ifelse(u + 8 > 0,
                 -log(pmax(u + 8, 1e-12)) -
                 0.5 * (log(pmax(u + 8, 1e-12)) / 0.9)^2, -Inf)
)

rows <- list()
for (nm in names(densities)) {
  ld <- densities[[nm]]
  for (m in c(5L, 7L, 9L, 12L, 15L)) {
    for (lo in c(-6, -4, -3, -2, -1)) {
      for (hi in c(0.5, 1, 1.5, 2, 3, 4, 6)) {
        r <- arrangement(ld, lo, hi, m)
        rows[[length(rows) + 1L]] <- cbind(kind = nm, r)
      }
    }
  }
}
d <- do.call(rbind, rows)

# One row per (arrangement, side).
long <- rbind(
  data.frame(kind = d$kind, m = d$m, lift = d$lift_lo, tail = d$tail_lo),
  data.frame(kind = d$kind, m = d$m, lift = d$lift_hi, tail = d$tail_hi))
long <- long[is.finite(long$lift) & is.finite(long$tail), ]

cat("== lift against the mass the span does not reach ==\n")
long$tband <- cut(long$tail, c(-Inf, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, Inf))
print(do.call(rbind, lapply(split(long, long$tband), function(x)
  if (!nrow(x)) NULL else data.frame(
    tail_band = x$tband[1], n = nrow(x),
    lift_min = min(x$lift), lift_med = stats::median(x$lift),
    lift_max = max(x$lift)))), digits = 3, row.names = FALSE)

cat("\n== a lift cutoff as a detector of 'more than TAIL left outside' ==\n")
for (tt in c(0.01, 0.05)) {
  cat(sprintf("\ntruncating := tail > %.2f  (%d of %d arrangements)\n",
              tt, sum(long$tail > tt), nrow(long)))
  lad <- do.call(rbind, lapply(c(0.5, 0.75, 1, 1.25, 1.5, 2, 3), function(cut) {
    hit <- long$lift >= cut
    pos <- long$tail > tt
    data.frame(cutoff = cut,
               caught = mean(hit[pos]),
               false_alarm = mean(hit[!pos]),
               worst_missed_tail = if (any(pos & !hit)) max(long$tail[pos & !hit]) else 0,
               worst_flagged_tail = if (any(hit & !pos)) max(long$tail[hit & !pos]) else NA)
  }))
  print(lad, digits = 3, row.names = FALSE)
}

cat("\n== the issue's own cases, at m = 9 ==\n")
cases <- list(
  "observed fit"       = c(0.0181,0.0444,0.1534,0.0655,0.1242,0.1186,0.1160,0.2907,0.0691),
  "20% on the ceiling" = c(0.010,0.020,0.050,0.060,0.080,0.100,0.120,0.360,0.200),
  "34% on the ceiling" = c(0.005,0.010,0.020,0.030,0.050,0.070,0.130,0.345,0.340),
  "ceiling is mode"    = c(0.005,0.010,0.020,0.030,0.050,0.070,0.130,0.245,0.440))
for (nm in names(cases)) {
  w <- cases[[nm]] / sum(cases[[nm]])
  cat(sprintf("%-20s ceiling %5.1f%%  lift %.2f\n", nm, 100 * w[9], 9 * w[9]))
}

cat("\n== what a flagged axis is guaranteed to be leaving out ==\n")
for (cut in c(0.75, 1, 1.25, 1.5)) {
  f <- long[long$lift >= cut, ]
  cat(sprintf("lift >= %.2f : n %4d  tail min %.4f  q10 %.4f  median %.4f\n",
              cut, nrow(f), min(f$tail), stats::quantile(f$tail, 0.1),
              stats::median(f$tail)))
}
