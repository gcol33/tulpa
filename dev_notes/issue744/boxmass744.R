# gcol33/tulpa#744: every figure the box-mass test file quotes.
#
#   Rscript boxmass744.R <lib> <repo> <hyperprior>
#
# The test file fits `ogd_fixture_fit()` at its own defaults, so its figures are
# read under the fixture's stated prior and within-cell read; this script prints
# them under `hyperprior` and under both reads.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; hyperprior <- args[3L]
stopifnot(hyperprior %in% c("flat", "proper"))

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
env <- harness_load(lib, repo, "test-nested-laplace-joint-box-mass.R")
id <- harness_identity(lib, hyperprior, "box_uniform and chord")
harness_print_identity(id)

local({
  f4 <- function(x) formatC(x, format = "f", digits = 4)
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))
  cat("fixture residual variance", sim$phi, "\n")
  fit <- function(lv, wc) ogd_fixture_fit(sim, lv, hyperprior = hyperprior,
                                          within_cell = wc)
  for (wc in c("box_uniform", "chord")) {
    cat("\n== within_cell", wc, "==\n")
    d5 <- outer_grid_dump(fit(5L, wc))
    b5 <- .bxm_weights(d5)
    r5 <- outer_grid_weight_report(d5, b5$w)
    cat("L5 computed", sum(b5$bm$computed), " interior weight",
        f4(sum(d5$weights[b5$bm$computed])), "\n")
    for (p in names(OGD_PARTS)) {
      cat(sprintf("L5 %-9s diff %s floor %s above %s\n", p, f4(r5$diff[[p]]),
                  f4(r5$floor[[p]]), r5$above_floor[[p]]))
    }
    d4 <- outer_grid_dump(fit(4L, wc))
    b4 <- .bxm_weights(d4)
    r4 <- outer_grid_weight_report(d4, b4$w)
    cat("L4 computed", sum(b4$bm$computed), " max log multiplier",
        f4(max(b4$bm$log_box_ratio)), "\n")
    for (p in names(OGD_PARTS)) {
      cat(sprintf("L4 %-9s diff %s floor %s above %s\n", p, f4(r4$diff[[p]]),
                  f4(r4$floor[[p]]), r4$above_floor[[p]]))
    }
    ref <- outer_grid_rebuild(outer_grid_dump(fit(12L, wc)))
    e0 <- outer_grid_read_diff(ref, outer_grid_rebuild(d4))
    e1 <- outer_grid_read_diff(ref, outer_grid_rebuild(d4, b4$w))
    for (p in names(OGD_PARTS)) {
      cat(sprintf("L4 vs L12 %-9s shipped %s  box rule %s\n", p, f4(e0[[p]]),
                  f4(e1[[p]])))
    }
  }
}, envir = env)
