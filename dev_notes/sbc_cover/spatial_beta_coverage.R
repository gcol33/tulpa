# Sizes the fixed-effect coverage gate that runs with a SPATIAL field present
# (gcol33/tulpa#862). The gate itself is `test-spatial-beta-coverage.R`.
#
# The gap it closes. tulpa gates the nested-Laplace fixed-effect interval's
# coverage only on an IID region-grouped RE block
# (`test-nested-laplace-recovery.R`); the spatial recovery tests gate the
# HYPERPARAMETERS and never beta; the joint-multi sweep gates (sigma, alpha)
# and never beta. So nothing asserted that a fixed-effect interval is
# calibrated when a spatial field is in the model -- the exact configuration
# gcol33/tulpa#862 spent eight rounds of route-versus-route comparison on, with
# no reference to consult. Route A disagreeing with route B says nothing about
# which is wrong; coverage against a simulated truth does.
#
# Simulator, fit and regimes come from `tests/testthat/helper-spatial-beta-coverage.R`,
# so this driver and the gate it sizes cannot drift apart.
#
#   Rscript dev_notes/sbc_cover/spatial_beta_coverage.R [n_seed]

# The installed build, not `load_all()`: this may run beside the route-coverage
# sweep and a source reload would recompile under that load.
suppressPackageStartupMessages(library(tulpa))

source("tests/testthat/helper-spatial-grid.R")
source("tests/testthat/helper-spatial-beta-coverage.R")

n_seed <- if (length(commandArgs(trailingOnly = TRUE)))
  as.integer(commandArgs(trailingOnly = TRUE)[1]) else 30L
seeds <- seq_len(n_seed)

cat(sprintf("tulpa %s | %d seeds per regime | %s\n\n",
            packageVersion("tulpa"), n_seed, format(Sys.time())))

pooled <- integer(0)
for (rn in names(.SPATIAL_BETA_REGIMES)) {
  r  <- .SPATIAL_BETA_REGIMES[[rn]]
  sw <- .spatial_beta_sweep(rn, seeds)
  pooled <- c(pooled, as.vector(sw$cov[!is.na(sw$cov)]))
  cat(sprintf("regime %-6s %dx%d lattice (%d sites) reps=%d amp=%.1f\n",
              rn, r$nr, r$nc, r$nr * r$nc, r$reps, r$amp))
  for (j in seq_along(.SPATIAL_BETA_TRUTH)) {
    k <- sum(sw$cov[, j], na.rm = TRUE); n <- sw$n[j]
    bt <- stats::binom.test(k, n)$conf.int
    cat(sprintf("   beta%d truth %+0.2f  n %2d  coverage %5.1f%% [%.2f, %.2f]  mean width %6.3f\n",
                j, .SPATIAL_BETA_TRUTH[j], n, 100 * sw$rate[j], bt[1], bt[2],
                mean(sw$width[, j], na.rm = TRUE)))
  }
  cat("\n")
}

bt <- stats::binom.test(sum(pooled), length(pooled))$conf.int
cat(sprintf("AGGREGATE pooled coverage: %.3f [%.3f, %.3f] over %d trials\n",
            mean(pooled), bt[1], bt[2], length(pooled)))
