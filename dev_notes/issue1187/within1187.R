# gcol33/tulpa#1187: on the #331 four-axis fixture, whether the log-quadratic
# read differs from the box read, and on how many rows it resolves its own
# conditional (h / sd <= 2, where it joins quadratics rather than boxes).
#
#   Rscript within1187.R [<repo>] [<seed>]
args <- commandArgs(trailingOnly = TRUE)
repo <- if (length(args) >= 1L) args[1L] else "C:/GillesC/Documents/dev/tulpa"
s    <- if (length(args) >= 2L) as.integer(args[2L]) else 1L
suppressMessages(devtools::load_all(repo, quiet = TRUE))
tdir <- file.path(repo, "tests", "testthat")
env <- new.env(parent = asNamespace("tulpa"))
for (h in list.files(tdir, "^helper-.*[.]R$", full.names = TRUE)) sys.source(h, envir = env)
for (e in parse(file.path(tdir, "test-nested-laplace-recovery.R"), keep.source = FALSE))
  if (is.call(e) && identical(e[[1L]], as.name("<-"))) eval(e, env)
env$s <- s
evalq({
  sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
  d  <- LCCD_SIM(8100L + s, "gaussian", LCCD_CFG$nr, LCCD_CFG$spr, LCCD_CFG$ntr,
                 LCCD_CFG$beta, LCCD_CFG$su, LCCD_CFG$phi)
  for (lev in 4:6) {
    f  <- recov_fit_joint_coarse(d, sg, "gaussian", LCCD_CFG, levels = lev,
                                 hyperprior = "flat")
    dm <- outer_grid_dump(f)
    A  <- lccd_arms(dm)
    for (wc in tulpa:::.NL_WITHIN_CELL) {
      dm$within <- wc
      r <- lccd_arm_read(dm, A$shipped)
      cat(sprintf("L%d %-13s sigma_1 [%.4f, %.4f] median %.4f\n", lev, wc,
                  r$s_lo, r$s_hi, r$s_med))
    }
    geo <- .nl_axis_geometry(f)
    for (wc in tulpa:::.NL_WITHIN_CELL) {
      q <- .nl_axis_quantiles(f$theta_grid, f$log_marginal, f$refining_axis,
                              probs = c(0.025, 0.5, 0.975), domains = geo$domain,
                              atoms = geo$atom, within = wc)
      cat(sprintf("L%d engine %-13s sigma_1 [%.4f, %.4f]\n", lev, wc,
                  as.numeric(q$ci_lo[[1L]]), as.numeric(q$ci_hi[[1L]])))
    }
    rs <- .nl_axis_resolution(f$theta_grid, f$log_marginal, geo$domain, geo$atom)
    cat(sprintf("L%d h/sd per axis: %s\n", lev,
                paste(sprintf("%.2f", rs$h_over_sd), collapse = " ")))
    for (nm in grep("within|construction", names(f), value = TRUE)) {
      cat(sprintf("L%d f$%s: %s\n", lev, nm,
                  paste(format(unlist(f[[nm]])), collapse = " ")))
    }
  }
}, env)
