# The ESS a reported axis SD has to clear, as a ladder rather than a choice.
#
# For each candidate threshold, the WORST relative error either estimator makes
# over every (spacing, offset) arrangement that clears it. The Gaussian axis is
# where the parabola is exact by construction, so it is the arm that says what
# the weighted read costs; the skewed axis is where the parabola targets a
# different number, so it is the arm that says what the parabola costs.

suppressMessages(devtools::load_all(".", quiet = TRUE))

wtd_sd <- function(v, lw) {
  m <- max(lw); w <- exp(lw - m); w <- w / sum(w)
  sqrt(max(0, sum(w * v^2) - sum(w * v)^2))
}
axis_ess <- function(lw) {
  m <- max(lw); p <- exp(lw - m); p <- p / sum(p); 1 / sum(p^2)
}
sten <- function(u, lw)
  as.numeric(.nl_laplace_at_mode_sd_axis(u, lw, log_axis = FALSE,
                                         return_u_sd = TRUE))

hs   <- seq(0.25, 4, by = 0.05)
offs <- seq(0, 0.9, by = 0.1)

row_for <- function(kind, h, off, K = 61L) {
  u <- (seq_len(K) - (K + 1) / 2) * h + off * h
  if (kind == "gaussian") {
    lw  <- -0.5 * u^2
    ref <- 1
  } else {
    x  <- u + 8
    ld <- function(z) ifelse(z > 0,
      -log(pmax(z, 1e-12)) - 0.5 * (log(pmax(z, 1e-12)) / 0.9)^2, -Inf)
    lw <- ld(x)
    ug <- seq(min(u), max(u), length.out = 40001L)
    ref <- wtd_sd(ug, ld(ug + 8))
  }
  data.frame(kind = kind, h = h, off = off, ess = axis_ess(lw),
             rel_wtd = wtd_sd(u, lw) / ref, rel_sten = sten(u, lw) / ref)
}

d <- do.call(rbind, lapply(c("gaussian", "skew"), function(k)
  do.call(rbind, lapply(hs, function(h)
    do.call(rbind, lapply(offs, function(o) row_for(k, h, o)))))))

cat("== worst relative error among arrangements clearing each ESS threshold ==\n")
lad <- do.call(rbind, lapply(c(1.5, 2, 2.5, 3, 3.5, 4, 5, 6), function(t) {
  g <- d[d$kind == "gaussian" & d$ess >= t, ]
  s <- d[d$kind == "skew"     & d$ess >= t, ]
  data.frame(threshold = t,
             n_gauss = nrow(g), wtd_gauss = max(abs(g$rel_wtd - 1)),
             sten_gauss = max(abs(g$rel_sten - 1), na.rm = TRUE),
             n_skew = nrow(s), wtd_skew = max(abs(s$rel_wtd - 1)),
             sten_skew = max(abs(s$rel_sten - 1), na.rm = TRUE))
}))
print(lad, digits = 3, row.names = FALSE)

cat("\n== below the threshold: what the parabola rescues ==\n")
lo <- do.call(rbind, lapply(c(1.2, 1.5, 2, 2.5, 3), function(t) {
  g <- d[d$kind == "gaussian" & d$ess < t, ]
  data.frame(band = sprintf("ess < %.1f", t), n = nrow(g),
             worst_wtd = max(abs(g$rel_wtd - 1)),
             worst_sten = max(abs(g$rel_sten - 1), na.rm = TRUE),
             sten_na = mean(is.na(g$rel_sten)))
}))
print(lo, digits = 3, row.names = FALSE)

cat("\n== how far each estimator moves across grids of ONE density (ess >= 3) ==\n")
for (k in c("gaussian", "skew")) {
  x <- d[d$kind == k & d$ess >= 3, ]
  cat(sprintf("%-8s n %4d  weighted [%.4f, %.4f]  parabola [%.4f, %.4f]\n",
              k, nrow(x), min(x$rel_wtd), max(x$rel_wtd),
              min(x$rel_sten, na.rm = TRUE), max(x$rel_sten, na.rm = TRUE)))
}
