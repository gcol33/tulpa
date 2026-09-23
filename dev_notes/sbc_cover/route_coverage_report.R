# Read whatever route_coverage.R has written and report coverage per route.
#
#   Rscript route_coverage_report.R [out_dir]
#
# The gate is the one in ~/.claude/skills/research-method-rules: a nominal 95%
# interval should contain the truth in >= 85% of >= 20 seeds. Reported per
# coefficient and per route, with a binomial CI so a shortfall is separated
# from seed noise, plus mean interval width so an interval that covers by being
# enormous is distinguishable from one that covers by being right.

args <- commandArgs(trailingOnly = TRUE)
d <- if (length(args) >= 1) args[1] else
     "dev_notes/sbc_cover/route_coverage"
fs <- list.files(d, pattern = "^cov_.*[.]rds$", full.names = TRUE)
if (!length(fs)) stop("no cells in ", d)

rows <- list(); tim <- list()
for (f in fs) {
  r <- readRDS(f)
  for (rt in c("nested_laplace", "nuts")) {
    ci <- r[[paste0(rt, "_ci")]]
    if (is.null(ci)) next
    ci$route <- rt; ci$sigma_true <- r$sigma_true; ci$seed <- r$seed
    ci$n_det <- r$n_det_sites
    ci$field_sd <- r[[paste0(rt, "_field_sd")]] %||% NA_real_
    ci$field_sd_realized <- r$field_sd_realized
    rows[[length(rows) + 1L]] <- ci
    tim[[length(tim) + 1L]] <- data.frame(
      route = rt, secs = r[[paste0(rt, "_secs")]] %||% NA_real_)
  }
}
`%||%` <- function(x, y) if (is.null(x)) y else x
df <- do.call(rbind, rows)
tm <- do.call(rbind, tim)

cat(sprintf("cells: %d | timings: grid %.0fs nuts %.0fs per fit\n\n",
            length(fs),
            mean(tm$secs[tm$route == "nested_laplace"], na.rm = TRUE),
            mean(tm$secs[tm$route == "nuts"], na.rm = TRUE)))

bin_ci <- function(k, n) {
  if (n == 0) return(c(NA, NA))
  bt <- stats::binom.test(k, n)$conf.int
  c(bt[1], bt[2])
}

cat("=== 95% interval coverage against simulated truth ===\n")
cat(sprintf("%-18s %-15s %5s %6s %7s %-14s %8s %8s\n",
            "term", "route", "n", "cov", "rate", "95% CI", "mean_w", "med_w"))
for (tmn in unique(df$term)) {
  for (rt in c("nested_laplace", "nuts")) {
    s <- df[df$term == tmn & df$route == rt, ]
    if (!nrow(s)) next
    k <- sum(s$cov, na.rm = TRUE); n <- sum(!is.na(s$cov))
    ci <- bin_ci(k, n)
    cat(sprintf("%-18s %-15s %5d %6d %6.1f%% [%.2f, %.2f] %8.3f %8.3f\n",
                tmn, rt, n, k, 100 * k / n, ci[1], ci[2],
                mean(s$width, na.rm = TRUE), stats::median(s$width, na.rm = TRUE)))
  }
}

cat("\n=== by true field scale, psi_(Intercept) ===\n")
psi <- df[df$term == "psi_(Intercept)", ]
for (sg in sort(unique(psi$sigma_true))) {
  for (rt in c("nested_laplace", "nuts")) {
    s <- psi[psi$sigma_true == sg & psi$route == rt, ]
    if (!nrow(s)) next
    cat(sprintf("sigma %.1f  %-15s n %2d  cov %5.1f%%  mean width %6.3f  mean |bias| %6.3f\n",
                sg, rt, nrow(s), 100 * mean(s$cov, na.rm = TRUE),
                mean(s$width, na.rm = TRUE),
                mean(abs(s$est - s$truth), na.rm = TRUE)))
  }
}

cat("\n=== field_sd recovery (mean estimate / realized field SD) ===\n")
fsd <- df[df$term == "psi_(Intercept)", ]
for (sg in sort(unique(fsd$sigma_true))) {
  for (rt in c("nested_laplace", "nuts")) {
    s <- fsd[fsd$sigma_true == sg & fsd$route == rt, ]
    if (!nrow(s)) next
    cat(sprintf("sigma %.1f  %-15s n %2d  est %6.3f  realized %6.3f  ratio %5.2f\n",
                sg, rt, nrow(s), mean(s$field_sd, na.rm = TRUE),
                mean(s$field_sd_realized, na.rm = TRUE),
                mean(s$field_sd, na.rm = TRUE) /
                  mean(s$field_sd_realized, na.rm = TRUE)))
  }
}

cat("\n=== width spread across seeds (does the interval track the data?) ===\n")
for (tmn in c("psi_(Intercept)", "psi_occ_cov1", "p_(Intercept)")) {
  for (rt in c("nested_laplace", "nuts")) {
    s <- df[df$term == tmn & df$route == rt, ]
    if (nrow(s) < 2) next
    cat(sprintf("%-18s %-15s width min %6.3f max %6.3f  span %5.2fx\n",
                tmn, rt, min(s$width, na.rm = TRUE), max(s$width, na.rm = TRUE),
                max(s$width, na.rm = TRUE) / min(s$width, na.rm = TRUE)))
  }
}
