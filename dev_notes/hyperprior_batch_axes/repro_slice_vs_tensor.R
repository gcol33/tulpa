# gcol33/tulpa#760: a refinement slice against a tensor cell at the same coordinates.
# Rscript repro_slice_vs_tensor.R tests/testthat/test-joint-axis-refinable.R
# Before the fix the slice reads +1.003292 (sigma PC density dropped); base cells agree to 2e-13.

suppressMessages(library(tulpa))
args <- commandArgs(trailingOnly = TRUE)
cat("tulpa", as.character(packageVersion("tulpa")), "from", find.package("tulpa"), "\n")
ex <- parse(args[1])
for (e in ex) if (is.call(e) && identical(e[[1]], as.name("<-")) &&
                  as.character(e[[2]]) %in% c(".axr_sim", ".axr_fit")) eval(e, globalenv())
sim <- .axr_sim()
fit <- .axr_fit(sim, c(0.2, 0.4, 0.6))
tag <- fit$refining_axis
tensor <- .axr_fit(sim, sort(unique(as.numeric(fit$theta_grid[, "alpha"]))),
                   control = list(adaptive_grid = FALSE, axis_refine = c(alpha = "none")))
key <- function(g) sprintf("%.12g|%.12g", g[, "sigma"], g[, "alpha"])
m <- match(key(fit$theta_grid), key(tensor$theta_grid))
d <- fit$log_marginal - tensor$log_marginal[m]
cat("slices:", sum(nzchar(tag)), "| unmatched:", sum(is.na(m)), "\n")
cat("slice diff:", sprintf("%.6f", d[nzchar(tag)]), "\n")
cat("base max |diff|:", max(abs(d[!nzchar(tag)])), "\n")
