# gcol33/tulpa#1187: the floor figures quoted in test-nested-laplace-joint-box-mass.R,
# test-nested-laplace-joint-barycentre.R and test-outer-grid-dump.R, read through
# `outer_grid_noise_floor()` with runs inside one level taking that level exactly.
#
#   Rscript floor1187.R [<repo>]
#
# Prints, per candidate rule, resolution, prior and within-cell read, the
# difference against the shipped read and the floor for each part of the read,
# then the tempering ladder on the local-CCD fixture.
args <- commandArgs(trailingOnly = TRUE)
repo <- if (length(args)) args[1L] else "C:/GillesC/Documents/dev/tulpa"
suppressMessages(devtools::load_all(repo, quiet = TRUE))
tdir <- file.path(repo, "tests", "testthat")
env <- new.env(parent = asNamespace("tulpa"))
for (h in list.files(tdir, "^helper-.*[.]R$", full.names = TRUE)) sys.source(h, envir = env)
for (f in c("test-nested-laplace-joint-box-mass.R", "test-nested-laplace-joint-barycentre.R",
            "test-outer-grid-dump.R")) {
  for (e in parse(file.path(tdir, f), keep.source = FALSE))
    if (is.call(e) && identical(e[[1L]], as.name("<-"))) eval(e, env)
}
env$show <- function(lab, r) {
  cat(sprintf("%-34s", lab))
  for (p in c("endpoints", "widths", "median"))
    cat(sprintf("  %s %.4f / %.4f (%.2fx)", substr(p, 1, 3), r$diff[[p]], r$floor[[p]],
                r$diff[[p]] / r$floor[[p]]))
  cat("\n")
}
env$quiet <- function(expr) {
  r <- NULL
  invisible(capture.output(r <- suppressMessages(expr), type = "output"))
  r
}
evalq({
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))
  for (wc in c("log_quadratic", "chord")) {
    f <- function(lv, hyp = "proper")
      quiet(outer_grid_dump(ogd_fixture_fit(sim, lv, within_cell = wc, hyperprior = hyp)))
    d5 <- f(5L); d4 <- f(4L)
    show(paste("box-mass L5 proper", wc), outer_grid_weight_report(d5, .bxm_weights(d5)$w))
    show(paste("box-mass L4 proper", wc), outer_grid_weight_report(d4, .bxm_weights(d4)$w))
    for (hp in c("flat", "proper")) {
      b4 <- if (hp == "flat") f(4L, .BAR_HYPERPRIOR) else d4
      b5 <- if (hp == "flat") f(5L, .BAR_HYPERPRIOR) else d5
      show(paste("barycentre L4", hp, wc),
           outer_grid_weight_report(b4, joint_grid = .bar_place_g(b4)$joint_grid))
      show(paste("barycentre L5", hp, wc),
           outer_grid_weight_report(b5, joint_grid = .bar_place_g(b5)$joint_grid))
    }
  }
  d <- quiet(outer_grid_dump(.ogd_fit_local()))
  same <- outer_grid_weight_report(d, outer_grid_weights(d, d$dnode))
  for (k in c(1.1, 1.3, 1.5, 2, 3)) {
    show(sprintf("local-CCD temper %.1f", k), outer_grid_weight_report(
      d, outer_grid_weights(d, dnode = d$dnode, log_marginal = k * d$log_marginal),
      floor = same$floor))
  }
  fine <- .ogd_fake_dump(81L)
  show("fake 81 levels temper 3", outer_grid_weight_report(
    fine, outer_grid_weights(fine, log_marginal = 3 * fine$log_marginal)))
}, env)
