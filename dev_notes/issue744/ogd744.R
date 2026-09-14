# gcol33/tulpa#744: the figures test-outer-grid-dump.R quotes off its own
# locally CCD-refined fixture (`.ogd_fit_local()`, four crossed iid blocks).
#
#   Rscript ogd744.R <lib> <repo>
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
env <- harness_load(lib, repo, "test-outer-grid-dump.R")
id <- harness_identity(lib, "proper", "box_uniform")
harness_print_identity(id)

local({
  f4 <- function(x) formatC(x, format = "f", digits = 4)
  sim <- .ogd_sim(c(0.8, 0.5, 0.3, 0.2))
  cat("fixture residual variance", sim$phi, "\n")
  d <- outer_grid_dump(.ogd_fit_local())
  base <- outer_grid_rebuild_fixed(d)
  hot <- outer_grid_rebuild_fixed(
    d, outer_grid_weights(d, dnode = d$dnode, log_marginal = 3 * d$log_marginal))
  cat("\n== fixed-effect read under tempering (x3) ==\n")
  cat("se base", f4(base$se), " hot", f4(hot$se), "\n")
  cat("se change", signif(hot$se - base$se, 3), "\n")
  cat("mean change", signif(hot$mean - base$mean, 3), "\n")
  cat("grid ESS", f4(1 / sum(d$weights^2)), " cells", nrow(d$joint_grid), "\n")

  cat("\n== tempering against the floor ==\n")
  same <- outer_grid_weight_report(d, outer_grid_weights(d, d$dnode))
  h <- outer_grid_weight_report(
    d, outer_grid_weights(d, dnode = d$dnode, log_marginal = 3 * d$log_marginal),
    floor = same$floor)
  for (p in names(OGD_PARTS)) {
    cat(sprintf("%-9s diff %s floor %s above %s\n", p, f4(h$diff[[p]]),
                f4(h$floor[[p]]), h$above_floor[[p]]))
  }

  cat("\n== tempering exponent against the floor ==\n")
  for (k in c(1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 3)) {
    wk <- outer_grid_weights(d, dnode = d$dnode, log_marginal = k * d$log_marginal)
    hk <- outer_grid_weight_report(d, wk, floor = same$floor)
    fk <- outer_grid_rebuild_fixed(d, wk)
    cat(sprintf("k %.2f endpoints %s widths %s median %s | above %s | se1 change %s\n",
                k, f4(hk$diff$endpoints), f4(hk$diff$widths), f4(hk$diff$median),
                paste(unlist(hk$above_floor), collapse = "/"),
                signif(fk$se[[1L]] - base$se[[1L]], 3)))
  }
}, envir = env)
