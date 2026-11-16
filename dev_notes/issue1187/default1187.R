# gcol33/tulpa#1187: the #331 four-axis crossed fixture (`LCCD_SIM`, `LCCD_CFG`,
# seed offset 8100) fitted the way a user fits it -- every block's `sigma_grid`
# left out, so the placement pass lays each axis at the SD the mode-find
# measured, and the engine's own integration, screen, refinement and within-cell
# read -- and scored on the interval the fit itself reports for sigma_1 and the
# two fixed effects.
#
#   Rscript default1187.R <lib> <repo> <hyperprior> <n_seed> <out.csv> [workers=8]
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; hyperprior <- args[3L]
n_seed <- as.integer(args[4L]); out <- args[5L]
workers <- if (length(args) >= 6L) as.integer(args[6L]) else 8L
stopifnot(hyperprior %in% c("flat", "proper"))

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
REC_FILE <- "test-nested-laplace-recovery.R"
invisible(harness_load(lib, repo, REC_FILE))
id <- harness_identity(lib, hyperprior, "engine default")
harness_print_identity(id)

one_seed <- function(s, hyperprior) {
  env <- list2env(list(s = s, hyperprior = hyperprior),
                  parent = get(".cov_env", envir = globalenv()))
  eval(quote({
    d <- LCCD_SIM(8100L + s, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr, LCCD_CFG$ntr,
                  LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
    f <- suppressWarnings(tulpa_nested_laplace_joint(
      responses = list(a = list(y = as.numeric(d$y),
                                n_trials = rep(LCCD_CFG$ntr, d$N), X = d$X,
                                family = "gaussian", phi = LCCD_CFG$phi)),
      prior = lapply(d$regions, function(g)
        list(type = "iid", obs_idx = list(g), n_units = d$nr)),
      hyperprior = hyperprior,
      control = list(n_threads = 1L)))
    bp <- beta_post(f)
    geo <- .nl_axis_geometry(f)
    rs  <- .nl_axis_resolution(f$theta_grid, f$log_marginal, geo$domain, geo$atom)
    dec <- f$theta_within_cell_declined
    data.frame(seed = s,
               s_lo = f$theta_ci_lo[[1L]], s_hi = f$theta_ci_hi[[1L]],
               s_med = f$theta_median[[1L]],
               b0_lo = bp$lo[1L], b0_hi = bp$hi[1L],
               b1_lo = bp$lo[2L], b1_hi = bp$hi[2L],
               n_cells = NROW(f$theta_grid),
               integration = paste(format(f$integration %||% NA), collapse = ""),
               h_sd_1 = rs$h_over_sd[1L], h_sd_max = max(rs$h_over_sd),
               declined_1 = if (length(dec)) paste(format(dec[[1L]]), collapse = "") else "")
  }), env)
}

cl <- parallel::makePSOCKcluster(workers)
cat("worker pids:", paste(unlist(parallel::clusterEvalQ(cl, Sys.getpid())), collapse = " "), "\n")
parallel::clusterExport(cl, c("lib", "repo", "REC_FILE"))
invisible(parallel::clusterEvalQ(cl, {
  source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
  assign(".cov_env", harness_load(lib, repo, REC_FILE), envir = globalenv())
  NULL
}))
t0 <- Sys.time()
R <- do.call(rbind, parallel::parLapplyLB(cl, seq_len(n_seed), one_seed,
                                          hyperprior = hyperprior))
parallel::stopCluster(cl)
R$hyperprior <- hyperprior; R$build <- harness_build_tag(id)
utils::write.csv(R, out, row.names = FALSE)
cat(sprintf("%d seeds in %.1f min -> %s\n", n_seed,
            as.numeric(difftime(Sys.time(), t0, units = "mins")), out))

su <- 0.7; beta <- c(-0.2, 0.7)
cv <- function(lo, hi, t) sum(lo <= t & hi >= t)
cat(sprintf("\nhyperprior %s, %d seeds, engine default path\n", hyperprior, n_seed))
cat(sprintf("  sigma_1    %3d  width %.4f  (misses high %d, low %d)\n",
            cv(R$s_lo, R$s_hi, su), mean(R$s_hi - R$s_lo),
            sum(R$s_lo > su), sum(R$s_hi < su)))
cat(sprintf("  intercept  %3d  width %.4f\n", cv(R$b0_lo, R$b0_hi, beta[1]), mean(R$b0_hi - R$b0_lo)))
cat(sprintf("  slope      %3d  width %.4f\n", cv(R$b1_lo, R$b1_hi, beta[2]), mean(R$b1_hi - R$b1_lo)))
cat(sprintf("  cells median %g [%g, %g]; h/sd axis 1 median %.2f, max over axes median %.2f\n",
            stats::median(R$n_cells), min(R$n_cells), max(R$n_cells),
            stats::median(R$h_sd_1), stats::median(R$h_sd_max)))
print(table(integration = R$integration, declined_1 = R$declined_1))
writeLines(format(Sys.time()), paste0(out, ".done"))
