# gcol33/tulpa#327: every figure the barycentre test file quotes.
#
#   Rscript bary327.R <lib> <repo> <hyperprior> [within_cell=box_uniform]
#
# `hyperprior` is the outer prior every ogd fixture fit states ("flat" or
# "proper"), `within_cell` the read the four-arm tables are scored under. The closed-form section fits nothing, so it does not depend on the
# prior; it names the platform its numeric-route figure was measured on.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; hyperprior <- args[3L]
table_within <- if (length(args) >= 4L) args[4L] else "box_uniform"
stopifnot(hyperprior %in% c("flat", "proper"))

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
env <- harness_load(lib, repo, "test-nested-laplace-joint-barycentre.R")
id <- harness_identity(lib, hyperprior, table_within)
harness_print_identity(id)

local({
  f4 <- function(x) formatC(x, format = "f", digits = 4)
  cat("\n== closed form against quadrature, 648 combinations ==\n")
  gr <- expand.grid(g = c(0, 0.25, 1, 2, 4.44, -3, 10, -8),
                    a = c(-2, -0.5, -0.05, 0, 0.05, 0.5, 1, 3, 20),
                    h_lo = c(0.25, 1, 2), h_hi = c(0.25, 1, 2))
  ei <- es <- ia <- numeric(nrow(gr)); route <- character(nrow(gr))
  for (i in seq_len(nrow(gr))) {
    f  <- .joint_local_ccd_axis_bary(gr$g[i], gr$a[i], gr$h_lo[i], gr$h_hi[i])
    bi <- .bar_integrate(gr$g[i], gr$a[i], gr$h_lo[i], gr$h_hi[i])
    bs <- .bar_simpson(gr$g[i], gr$a[i], gr$h_lo[i], gr$h_hi[i])
    route[i] <- f$route
    ei[i] <- abs(f$u_bar - bi); es[i] <- abs(f$u_bar - bs); ia[i] <- abs(bi - bs)
  }
  cat("platform", R.version$platform, "\n")
  cat("max e_int", signif(max(ei), 3), " max e_sim", signif(max(es), 3),
      " max |integrate - simpson|", signif(max(ia), 3), "\n")
  k <- which.max(ei); w <- gr$h_lo[k] + gr$h_hi[k]
  P <- 0.5 * gr$g[k]^2 / gr$a[k]; mu <- gr$g[k] / gr$a[k]
  cat("worst e_int at g", gr$g[k], "a", gr$a[k], "h", gr$h_lo[k], gr$h_hi[k],
      "route", route[k], " err/width", signif(ei[k] / w, 3),
      " eps * cancellation product", signif(.Machine$double.eps * P * abs(mu) / w, 3), "\n")
  cat("numeric route: max e_sim", signif(max(es[route == "numeric"]), 3),
      " max e_int", signif(max(ei[route == "numeric"]), 3),
      " bound 8 eps^0.75", signif(8 * .Machine$double.eps^0.75, 3), "\n")

  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))
  fit <- function(s, lv, within_cell = table_within)
    ogd_fixture_fit(s, lv, hyperprior = hyperprior, within_cell = within_cell)

  cat("\n== round trip, five levels ==\n")
  d <- outer_grid_dump(fit(sim, 5L))
  rb <- outer_grid_rebuild(d, d$weights, d$joint_grid)
  cat("max |rebuild - reported|",
      max(abs(unlist(rb[c("median", "ci_lo", "ci_hi")]) -
                unlist(d$reported[c("median", "ci_lo", "ci_hi")]))), "\n")

  cat("\n== placement against the grid's own resolution ==\n")
  for (wc in c("box_uniform", "chord")) for (lv in c(4L, 5L)) {
    dd <- outer_grid_dump(fit(sim, lv, wc))
    bc <- .bar_place_g(dd)
    r  <- outer_grid_weight_report(dd, joint_grid = bc$joint_grid)
    cat(sprintf("%-11s L%d computed %d  max shift %s |", wc, lv, sum(bc$computed),
                f4(max(bc$bary_shift, na.rm = TRUE))))
    for (p in names(OGD_PARTS)) {
      cat(sprintf(" %s %s / floor %s (%.2fx)", p, f4(r$diff[[p]]), f4(r$floor[[p]]),
                  r$diff[[p]] / r$floor[[p]]))
    }
    cat("\n")
  }

  cat("\n== the four arms against a twelve-level reference, seed 4242 ==\n")
  tab <- function(s, lv, ref) {
    dd <- outer_grid_dump(fit(s, lv))
    w <- .bar_mass_w(dd); g <- .bar_place_g(dd)$joint_grid
    err <- function(ww, gg) outer_grid_read_diff(ref, outer_grid_rebuild(dd, ww, gg))
    out <- list(shipped = err(NULL, NULL), mass = err(w, NULL),
                location = err(NULL, g), pair = err(w, g))
    t(sapply(out, function(e) unlist(e[names(OGD_PARTS)])))
  }
  ref <- outer_grid_rebuild(outer_grid_dump(fit(sim, 12L)))
  T4 <- tab(sim, 4L, ref); T5 <- tab(sim, 5L, ref)
  cat("four levels\n"); print(round(T4, 4))
  cat("five levels\n"); print(round(T5, 4))
  cat("five levels: shipped / location median", f4(T5["shipped", "median"] / T5["location", "median"]), "\n")

  cat("\n== seeds 1 to 5 ==\n")
  A4 <- A5 <- list()
  for (sd_ in 1:5) {
    s <- ogd_fixture_sim(c(0.8, 0.5, 0.3), sd_)
    r <- outer_grid_rebuild(outer_grid_dump(fit(s, 12L)))
    A4[[sd_]] <- tab(s, 4L, r); A5[[sd_]] <- tab(s, 5L, r)
  }
  for (nm in c("mass", "location", "pair")) for (L in list(list("L4", A4), list("L5", A5))) {
    A <- L[[2L]]
    cat(sprintf("%s %-8s wins against shipped: endpoints %d  widths %d  median %d of 5\n",
                L[[1L]], nm,
                sum(sapply(A, function(t) t[nm, "endpoints"] < t["shipped", "endpoints"])),
                sum(sapply(A, function(t) t[nm, "widths"] < t["shipped", "widths"])),
                sum(sapply(A, function(t) t[nm, "median"] < t["shipped", "median"]))))
  }
  cat("four-level means\n"); print(round(Reduce(`+`, A4) / 5, 4))
  cat("five-level means\n"); print(round(Reduce(`+`, A5) / 5, 4))
}, envir = env)
