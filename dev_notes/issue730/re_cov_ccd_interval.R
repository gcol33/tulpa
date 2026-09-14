# The CCD scale interval of test-re-cov-nested.R ("a CCD fit reports a scale
# interval that is not the node extent") under both hyperpriors: the design's
# node extent on log(sigma_1), the reported interval, and the moment-rule
# interval m +- z s it is built as.
#
#   Rscript re_cov_ccd_interval.R <lib> <repo>
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]
source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
env <- harness_load(lib, repo, "test-re-cov-nested.R")
for (hp in c("flat", "pc_lkj")) {
  harness_print_identity(harness_identity(lib, hp, "moment rule (CCD)"))
  d <- env$sim_corr_recov(77L, G = 60L, npg = 12L)
  rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             hyperprior = hp, control = list(diagnose_k = FALSE))
  th <- as.numeric(res$theta_grid[, 1L])
  w <- res$weights / sum(res$weights)
  m <- sum(w * th); s <- sqrt(sum(w * th^2) - m^2)
  row <- res$posterior[res$posterior$parameter == "sigma_1", ]
  ci <- log(c(row$ci_lo, row$ci_hi))
  cat(sprintf("%-6s nodes [%.5f, %.5f]  interval [%.5f, %.5f]  |interval - (m +- z s)| %.2e\n",
              hp, min(th), max(th), ci[1], ci[2],
              max(abs(ci - (m + stats::qnorm(c(0.025, 0.975)) * s)))))
}
