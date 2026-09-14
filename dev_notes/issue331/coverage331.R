# gcol33/tulpa#331: the four cell-rule arms (shipped, mass, location, pair)
# judged by coverage, the table in tests/testthat/test-nested-laplace-recovery.R.
#
# Per seed, one fit of the four-axis crossed recovery fixture (`LCCD_SIM`,
# `LCCD_CFG`, seed offset 8100) at four, five and six levels per axis
# (`recov_fit_joint_coarse()`), all four arms read off each fit through
# `lccd_arms()` / `lccd_arm_read()` with the within-cell read held at `chord`,
# exactly as `lccd_arm_sweep()` reads them.
#
#   Rscript coverage331.R <lib> <repo> <hyperprior> <n_seed> <out.csv> [workers=8]
#
# Writes one row per seed x resolution x arm, stamped with the prior arm, the
# within-cell read and the build, then prints the table.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; hyperprior <- args[3L]
n_seed <- as.integer(args[4L]); out <- args[5L]
workers <- if (length(args) >= 6L) as.integer(args[6L]) else 8L
stopifnot(hyperprior %in% c("flat", "proper"))

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
REC_FILE <- "test-nested-laplace-recovery.R"
invisible(harness_load(lib, repo, REC_FILE))
id <- harness_identity(lib, hyperprior, "chord")
harness_print_identity(id)

one_seed <- function(s, hyperprior) {
  env <- list2env(list(s = s, hyperprior = hyperprior),
                  parent = get(".cov_env", envir = globalenv()))
  eval(quote({
    sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
    d  <- LCCD_SIM(8100L + s, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr, LCCD_CFG$ntr,
                   LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
    do.call(rbind, lapply(4:6, function(lev) {
      dm <- outer_grid_dump(recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG,
                                                   levels = lev,
                                                   hyperprior = hyperprior))
      dm$within <- "chord"
      A <- lccd_arms(dm)
      do.call(rbind, lapply(names(A), function(a) {
        r <- lccd_arm_read(dm, A[[a]])
        data.frame(seed = s, levels = lev, arm = a,
                   s_lo = r$s_lo, s_hi = r$s_hi, s_med = r$s_med,
                   b0_lo = r$lo[1L], b0_hi = r$hi[1L],
                   b1_lo = r$lo[2L], b1_hi = r$hi[2L])
      }))
    }))
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
R$hyperprior <- hyperprior; R$within_cell <- "chord"; R$build <- harness_build_tag(id)
utils::write.csv(R, out, row.names = FALSE)
cat(sprintf("%d seeds in %.1f min -> %s\n", n_seed,
            as.numeric(difftime(Sys.time(), t0, units = "mins")), out))

su <- 0.7; beta <- c(-0.2, 0.7)
R$s_cov <- R$s_lo <= su & R$s_hi >= su;   R$s_w <- R$s_hi - R$s_lo
R$b0_cov <- R$b0_lo <= beta[1] & R$b0_hi >= beta[1]; R$b0_w <- R$b0_hi - R$b0_lo
R$b1_cov <- R$b1_lo <= beta[2] & R$b1_hi >= beta[2]; R$b1_w <- R$b1_hi - R$b1_lo
arms <- c("shipped", "mass", "location", "pair")
get_arm <- function(lev, a) { z <- R[R$levels == lev & R$arm == a, ]; z[order(z$seed), ] }
cat(sprintf("\nhyperprior %s, %d seeds; coverage count and mean width\n", hyperprior, n_seed))
for (q in list(c("s", "sigma_1"), c("b0", "intercept"), c("b1", "slope"))) {
  cat(" ", q[2L], "\n")
  for (a in arms) {
    cat(sprintf("    %-9s", a))
    for (lev in 4:6) {
      z <- get_arm(lev, a)
      cat(sprintf("  %3d  %.4f", sum(z[[paste0(q[1L], "_cov")]]), mean(z[[paste0(q[1L], "_w")]])))
    }
    cat("\n")
  }
}
cat("\ndiscordance against shipped on sigma_1 (lost / gained), and misses high / low\n")
for (a in c("mass", "location", "pair")) for (lev in 4:6) {
  sh <- get_arm(lev, "shipped"); z <- get_arm(lev, a)
  lost <- sh$s_cov & !z$s_cov; gained <- !sh$s_cov & z$s_cov
  ex <- if (sum(lost) + sum(gained) > 0)
    stats::binom.test(sum(lost), sum(lost) + sum(gained))$p.value else NA
  cat(sprintf("  %-9s L%d  -%d / +%d  (misses high %d, low %d)  exact p %.3g  width ratio %.4f\n",
              a, lev, sum(lost), sum(gained), sum(!z$s_cov & z$s_lo > su),
              sum(!z$s_cov & z$s_hi < su), ex, mean(z$s_w) / mean(sh$s_w)))
}
cat("\nfixed-effect discordance, mass against shipped\n")
for (lev in 4:6) {
  sh <- get_arm(lev, "shipped"); z <- get_arm(lev, "mass")
  cat(sprintf("  L%d intercept %d  slope %d  | se ratio intercept %.4f slope %.4f\n", lev,
              sum(sh$b0_cov != z$b0_cov), sum(sh$b1_cov != z$b1_cov),
              mean(z$b0_w) / mean(sh$b0_w), mean(z$b1_w) / mean(sh$b1_w)))
}
cat("\nper-seed |width - six-level shipped width|\n")
ref <- get_arm(6L, "shipped")
for (lev in 4:5) for (a in arms) {
  z <- get_arm(lev, a)
  cat(sprintf("  L%d %-9s sigma_1 %.4f  intercept %.5f\n", lev, a,
              mean(abs(z$s_w - ref$s_w)), mean(abs(z$b0_w - ref$b0_w))))
}
writeLines(format(Sys.time()), paste0(out, ".done"))
