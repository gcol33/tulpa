# gcol33/tulpa#333: every figure the descriptor-plane test file quotes, read off
# units written by plane333.R.
#
#   Rscript analyse333.R <lib> <repo> <units.rds> [spreads=all] [seeds=all]
#
# The subset `spread 3, seeds 1:8, gaussian` is exactly the `.dp_sweep()` the
# test file asserts on. The identity of the build and the prior arm are printed
# from the units file, and every row carries them as stamp columns.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; units_file <- args[3L]
spreads <- if (length(args) >= 4L && args[4L] != "all") as.numeric(strsplit(args[4L], ",")[[1L]]) else NULL
seeds   <- if (length(args) >= 5L && args[5L] != "all") eval(parse(text = args[5L])) else NULL

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
env <- harness_load(lib, repo, "test-nested-laplace-joint-descriptor-plane.R")
U <- readRDS(units_file)
harness_print_identity(U$identity)

local({
  keep <- vapply(U$units, function(u) {
    s <- u[[1L]]$cells
    (is.null(spreads) || s$spread[1L] %in% spreads) &&
      (is.null(seeds) || s$seed[1L] %in% seeds)
  }, logical(1))
  z <- .dp_combine(U$units[keep], 3L * sum(keep))
  D <- z$cells; W <- z$whole
  stopifnot(length(unique(D$hyperprior)) == 1L, length(unique(D$build)) == 1L)
  f4 <- function(x) formatC(x, format = "f", digits = 4)
  sp <- function(x, y) stats::cor(x, y, method = "spearman")
  D$fit <- paste(D$family, D$spread, D$seed, D$levels)
  cat(sprintf("hyperprior %s  family %s  spreads %s  seeds %s\n", D$hyperprior[1L],
              paste(unique(D$family), collapse = ","),
              paste(sort(unique(D$spread)), collapse = ","),
              paste(range(D$seed), collapse = "-")))
  cat("cells", nrow(D), " fits", z$n_fits, " coarse fits", length(unique(D$fit)), "\n")
  d4 <- D[D$levels == 4L, ]; d5 <- D[D$levels == 5L, ]

  cat("\n== separation ==\n")
  cat("rho(R_M,R_L) pooled", f4(sp(D$R_M, D$R_L)), " L4", f4(sp(d4$R_M, d4$R_L)),
      " L5", f4(sp(d5$R_M, d5$R_L)), "\n")
  pf <- vapply(split(D, D$fit), function(s) if (nrow(s) > 2L) sp(s$R_M, s$R_L) else NA_real_,
               numeric(1))
  cat("per-fit median", f4(stats::median(pf, na.rm = TRUE)), "\n")
  qm <- cut(D$R_M, stats::quantile(D$R_M, 0:4 / 4), include.lowest = TRUE)
  ql <- cut(D$R_L, stats::quantile(D$R_L, 0:4 / 4), include.lowest = TRUE)
  cat("off quartile diagonal", f4(1 - sum(diag(table(qm, ql))) / nrow(D)), "\n")
  cat("R_M range", f4(min(D$R_M)), f4(max(D$R_M)), " R_L range", f4(min(D$R_L)),
      f4(max(D$R_L)), " bound", f4(sqrt(3)), " max R_L/bound", f4(max(D$R_L) / sqrt(3)),
      " max R_L_max", f4(max(D$R_L_max)), "\n")
  cat("rho(R_M,R_L_A) pooled", f4(sp(D$R_M, D$R_L_A)), " L4", f4(sp(d4$R_M, d4$R_L_A)),
      " L5", f4(sp(d5$R_M, d5$R_L_A)), "\n")

  cat("\n== resolutions ==\n")
  cat("median R_M", f4(stats::median(d4$R_M)), f4(stats::median(d5$R_M)), " MW p",
      f4(stats::wilcox.test(d4$R_M, d5$R_M)$p.value), "\n")
  cat("median R_L", f4(stats::median(d4$R_L)), f4(stats::median(d5$R_L)), " MW p",
      f4(stats::wilcox.test(d4$R_L, d5$R_L)$p.value), " shift",
      f4(stats::median(d5$R_L) / stats::median(d4$R_L) - 1), "\n")
  for (s in list(L4 = d4, L5 = d5)) {
    q <- factor(.dp_quad(s), levels = c("M+L+", "M+L-", "M-L+", "M-L-"))
    print(round(100 * prop.table(table(q)), 1))
  }
  wm <- function(s, v) sum(s[[v]] * s$w) / sum(s$w)
  cat("weighted R_M", f4(wm(d4, "R_M")), f4(wm(d5, "R_M")), " ratio",
      f4(wm(d4, "R_M") / wm(d5, "R_M")), " weighted R_L", f4(wm(d4, "R_L")),
      f4(wm(d5, "R_L")), "\n")

  cat("\n== resolvable ==\n")
  res <- function(s, p) mean(s[[paste0("res_mass_", p)]] | s[[paste0("res_loc_", p)]] |
                               s[[paste0("res_both_", p)]])
  for (p in names(OGD_PARTS)) cat(p, " L4", f4(res(d4, p)), " L5", f4(res(d5, p)), "\n")

  cat("\n== quadrant gain against its permutation null (seed 327, 999 shuffles) ==\n")
  gain_at <- function(q, wins) {
    tb <- table(q, wins)
    sum(apply(tb, 1L, max)) / sum(tb) - max(table(wins)) / length(wins)
  }
  set.seed(327L); n_perm <- 999L
  rows <- list()
  for (p in names(OGD_PARTS)) for (lv in c(4L, 5L)) {
    s <- D[D$levels == lv, ]
    s <- s[s[[paste0("win_", p)]] != "unresolved", , drop = FALSE]
    if (nrow(s) < 20L) { cat(p, lv, "n resolved", nrow(s), "(not scored)\n"); next }
    wins <- s[[paste0("win_", p)]]
    for (rl in c("R_L", "R_L_max", "R_L_A")) {
      q <- .dp_quad(s, rl); obs <- gain_at(q, wins)
      null <- replicate(n_perm, gain_at(sample(q), wins))
      rows[[length(rows) + 1L]] <- data.frame(
        part = p, lv = lv, rl = rl, n = nrow(s), gain = round(obs, 4),
        p = round(mean(null >= obs), 4), null_mean = round(mean(null), 4),
        null_q95 = round(unname(stats::quantile(null, 0.95)), 4))
    }
  }
  G <- do.call(rbind, rows); print(G, row.names = FALSE)
  cat("scored", nrow(G), " gain exactly 0 in", sum(G$gain == 0), "\n")
  for (lv in c(4L, 5L)) {
    s <- D[D$levels == lv & D$win_median != "unresolved", ]
    tb <- table(.dp_quad(s), s$win_median)
    ct <- suppressWarnings(stats::chisq.test(tb))
    V <- sqrt(unname(ct$statistic) / (sum(tb) * (min(dim(tb)) - 1)))
    cat("median contingency L", lv, " n", sum(tb), " p", signif(ct$p.value, 3), " V", f4(V), "\n")
    print(tb)
    cat("argmax per quadrant:", paste(rownames(tb), colnames(tb)[apply(tb, 1, which.max)],
                                      collapse = "; "), "\n")
  }
  for (lv in c(4L, 5L)) for (p in names(OGD_PARTS)) {
    s <- D[D$levels == lv & D[[paste0("win_", p)]] != "unresolved", ]
    if (!nrow(s)) next
    tb <- prop.table(table(s[[paste0("win_", p)]]))
    cat("labels L", lv, p, " n", nrow(s), ":", paste(names(tb), round(tb, 3), collapse = " "), "\n")
  }

  cat("\n== loc versus mass ==\n")
  got <- numeric(0)
  for (lv in c(4L, 5L)) for (p in names(OGD_PARTS)) {
    s <- D[D$levels == lv, ]
    s <- s[s[[paste0("res_mass_", p)]] | s[[paste0("res_loc_", p)]], ]
    if (nrow(s) < 4L) { cat(sprintf("L%d %-9s n=%d\n", lv, p, nrow(s))); next }
    x <- rank(s$R_L) - rank(s$R_M)
    y <- s[[paste0("dL_loc_", p)]] - s[[paste0("dL_mass_", p)]]
    ct <- suppressWarnings(stats::cor.test(x, y, method = "spearman"))
    rl <- rank(s$R_L); rm <- rank(s$R_M); ry <- rank(y)
    pr <- stats::cor(stats::resid(stats::lm(rl ~ rm)), stats::resid(stats::lm(ry ~ rm)))
    if (nrow(s) >= 25L) got <- c(got, unname(ct$estimate))
    cat(sprintf("L%d %-9s n=%d rho=%+.4f p=%.3g partial(R_L | R_M)=%+.3f%s\n", lv, p,
                nrow(s), unname(ct$estimate), ct$p.value, pr,
                if (nrow(s) >= 25L) "  [scored by the test]" else ""))
  }
  cat("test-scored strata", length(got), " max |rho|", f4(max(abs(got))),
      " median |rho|", f4(stats::median(abs(got))), "\n")

  cat("\n== composition (per-seed means of summed one-cell against whole-grid) ==\n")
  ratio <- function(p, lv) {
    s <- D[D$levels == lv, ]
    per <- vapply(split(s, paste(s$spread, s$seed)),
                  function(x) sum(x[[paste0("dL_both_", p)]]), numeric(1))
    w <- W[W$part == p & W$levels == lv, ]
    c(per = mean(per), whole = mean(w$shipped - w$both))
  }
  for (p in names(OGD_PARTS)) for (lv in c(4L, 5L)) {
    r <- ratio(p, lv)
    cat(sprintf("%-9s L%d per %+.4f whole %+.4f ratio %.2f\n", p, lv, r[["per"]],
                r[["whole"]], r[["per"]] / r[["whole"]]))
  }
}, envir = env)
