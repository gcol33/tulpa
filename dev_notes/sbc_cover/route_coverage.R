# Which route is right? Coverage against SIMULATED TRUTH (gcol33/tulpa#862, #865)
#
# Every comparison in #862 measures the nested-Laplace marginal against the NUTS
# marginal. #865 then established that the sampler is itself biased on the field
# scale (1.3-2.7x high at every true value), so that reference does not support
# the conclusion "the grid is too narrow" -- two estimates disagreeing says
# nothing about which one is wrong.
#
# The measurement that does: simulate from a KNOWN truth and ask, per route,
# whether the reported 95% interval contains it. That is the coverage gate in
# ~/.claude/skills/research-method-rules (>= 85% of >= 20 seeds for a nominal
# 95% interval), applied to both routes on identical data.
#
# Identification note. `simulate_occu_cover` builds eta as
#   beta_occ[1] + beta_occ[2] * occ_cov1 + sigma * f
# (verified numerically against truth$psi). An ICAR field is identified only up
# to a constant, which the engine's sum-to-zero constraint moves into the
# intercept, so the truth a fitted `psi_(Intercept)` is estimating is
#   beta_occ[1] + sigma * mean(f)
# NOT the bare beta_occ[1]. Both are recorded; `*_cov` uses the identified one
# and `*_cov_raw` the bare one, so the choice is visible rather than baked in.
#
#   Rscript route_coverage.R <out_dir> [seeds] [sigmas] [routes]
#     seeds  "1:30"          seed range
#     sigmas "0.3,0.7,1.5"   true field scales
#
# One .rds per (sigma, seed) so an interrupted run resumes and costs only what
# is missing. Cells are visited SEED-MAJOR within a level, and levels are meant
# to be run as separate processes, so an interruption leaves the three levels at
# comparable depth instead of one level complete and the others empty.

suppressPackageStartupMessages({
  library(tulpa)
  library(tulpaObs)
})
`%||%` <- function(x, y) if (is.null(x)) y else x

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "route_coverage"
seeds   <- if (length(args) >= 2) eval(parse(text = args[2])) else 1:30
SIGMAS  <- if (length(args) >= 3) as.numeric(strsplit(args[3], ",")[[1]]) else
           c(0.3, 0.7, 1.5)
routes  <- if (length(args) >= 4) strsplit(args[4], ",")[[1]] else
           c("nested_laplace", "nuts")

GRID <- 8L; J <- 3L; N_CH <- 2L; N_IT <- 3000L; N_WU <- 1500L
N_THREADS <- 6L                      # 3 levels x 6 = 18 of 32, leaves the box usable
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Which engine produced these numbers. The first run of this sweep measured
# INSTALLED tulpa 0.5.4 while the repo was at 0.6.0 -- and 0.6.0 is exactly
# where the corrections reaching the occu_cover route landed, so those cells
# scored a different reporting path from the shipped one and the directory name
# said nothing about it. The identity is asserted here and STAMPED into every
# cell, so it travels with the data rather than with the filename.
ENGINE <- list(
  tulpa    = as.character(utils::packageVersion("tulpa")),
  tulpaObs = as.character(utils::packageVersion("tulpaObs")),
  git      = tryCatch(system2("git", c("rev-parse", "--short", "HEAD"),
                              stdout = TRUE, stderr = FALSE),
                      error = function(e) NA_character_)
)
EXPECT_TULPA <- Sys.getenv("TULPA_COV_EXPECT", "0.6.0")
if (!identical(ENGINE$tulpa, EXPECT_TULPA)) {
  stop(sprintf(
    "engine identity: tulpa %s installed, %s expected. Install the tree under test, or set TULPA_COV_EXPECT.",
    ENGINE$tulpa, EXPECT_TULPA), call. = FALSE)
}

rook_adj <- function(g) {
  N <- g * g; A <- matrix(0L, N, N); idx <- function(r, c) (r - 1L) * g + c
  for (r in seq_len(g)) for (c in seq_len(g)) {
    s <- idx(r, c)
    if (r > 1L) A[s, idx(r - 1L, c)] <- 1L
    if (r < g) A[s, idx(r + 1L, c)] <- 1L
    if (c > 1L) A[s, idx(r, c - 1L)] <- 1L
    if (c < g) A[s, idx(r, c + 1L)] <- 1L
  }
  A
}
adj <- rook_adj(GRID); N <- nrow(adj)

fit_route <- function(sim, route) {
  long <- data.frame(site_id = rep(seq_len(N), each = J),
                     visit = rep(seq_len(J), times = N),
                     y = as.vector(t(sim$y)),
                     det_cov1 = sim$visit_data$det_cov1,
                     pos_cov1 = sim$visit_data$pos_cov1)
  od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                  det.covs = c("det_cov1", "pos_cov1"))
  y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0
  ctl <- if (identical(route, "nuts"))
    list(n.iter = N_IT, n.warmup = N_WU, n.chains = N_CH, verbose = FALSE)
  else
    list(engine = "joint", verbose = FALSE, n.threads.outer = N_THREADS,
         sigma.grid = exp(seq(log(0.05), log(6), length.out = 15L)),
         phi.grid.pos = exp(seq(log(1), log(60), length.out = 11L)))
  tobs(~ occ_cov1 + icar(graph = adj),
       data = cbind(data.frame(site_id = seq_len(N)), sim$data),
       family = occu_cover("beta"), detection = ~ det_cov1,
       positive = ~ pos_cov1 + share(spatial()),
       y = od$y, y_pos = y_pos, visits = od$det.covs,
       method = route, control = ctl)
}

read_field_sd <- function(fit, route) {
  if (identical(route, "nuts")) {
    hd <- fit$hyper_draws
    k  <- match("field_sd", colnames(hd) %||% character(0))
    if (is.na(k)) return(c(NA_real_, NA_real_))
    return(c(mean(hd[, k]), stats::sd(hd[, k])))
  }
  c(fit$spatial$field_sd_mean %||% NA_real_,
    fit$spatial$field_sd_sd   %||% NA_real_)
}

# The user-facing door: whatever confint() reports IS the interval under test.
#
# `confint()` returns the two bounds only, so the point estimate comes from
# `coef()` and is matched BY NAME, never by row order -- the two doors need not
# order their terms alike. `mid` is recorded unconditionally as a bound-only
# location summary, so a route whose `coef()` is unavailable still yields a
# location error rather than a hole.
read_interval <- function(fit) {
  ci <- try(stats::confint(fit), silent = TRUE)
  if (inherits(ci, "try-error") || is.null(ci)) return(NULL)
  ci <- as.matrix(ci)
  nm <- rownames(ci)
  lo <- as.numeric(ci[, ncol(ci) - 1L]); hi <- as.numeric(ci[, ncol(ci)])

  cf <- try(stats::coef(fit), silent = TRUE)
  est <- rep(NA_real_, length(nm))
  if (!inherits(cf, "try-error") && !is.null(cf)) {
    cf <- unlist(cf)
    k <- match(nm, names(cf))
    est <- as.numeric(cf[k])
  }
  data.frame(term = nm, est = est, mid = (lo + hi) / 2,
             lo = lo, hi = hi, stringsAsFactors = FALSE)
}

cat(sprintf("tulpa %s | tulpaObs %s | git %s | seeds %s | sigmas %s | %s\n",
            ENGINE$tulpa, ENGINE$tulpaObs, ENGINE$git %||% "NA",
            paste(range(seeds), collapse = "-"),
            paste(SIGMAS, collapse = ","), format(Sys.time())))
flush.console()

for (sg in SIGMAS) for (s in seeds) {
  f <- file.path(out_dir, sprintf("cov_s%03.1f_seed%03d.rds", sg, s))
  if (file.exists(f)) next
  sim <- simulate_occu_cover(N = N, J = J, positive = "beta", adj = adj,
                             sigma = sg, alpha = 1, seed = s)
  tr <- sim$truth
  fbar <- mean(tr$f)

  # Truth on the scale each fitted coefficient estimates. The ICAR level is
  # absorbed into the intercept, so the psi intercept's target carries it.
  truth_vec <- c(
    "psi_(Intercept)"  = tr$beta_occ[1] + sg * fbar,
    "psi_occ_cov1"     = tr$beta_occ[2],
    "p_(Intercept)"    = tr$beta_p[1],
    "p_det_cov1"       = tr$beta_p[2],
    "pos_(Intercept)"  = tr$beta_pos[1],
    "pos_pos_cov1"     = tr$beta_pos[2]
  )
  truth_raw <- truth_vec; truth_raw[["psi_(Intercept)"]] <- tr$beta_occ[1]

  row <- list(sigma_true = sg, seed = s, engine = ENGINE,
              field_sd_realized = sg * stats::sd(tr$f),
              f_mean = fbar, f_sd = stats::sd(tr$f),
              truth = truth_vec, truth_raw = truth_raw,
              n_det_sites = sum(rowSums(sim$y, na.rm = TRUE) > 0))

  for (r in routes) {
    t0  <- Sys.time()
    fit <- try(fit_route(sim, r), silent = TRUE)
    if (inherits(fit, "try-error")) {
      row[[paste0(r, "_err")]] <- sub("\n.*", "", as.character(fit))
      next
    }
    fs <- read_field_sd(fit, r)
    row[[paste0(r, "_field_sd")]]    <- fs[1]
    row[[paste0(r, "_field_sd_sd")]] <- fs[2]

    iv <- read_interval(fit)
    if (!is.null(iv)) {
      keep <- iv$term %in% names(truth_vec)
      iv   <- iv[keep, , drop = FALSE]
      tv   <- truth_vec[iv$term]
      trw  <- truth_raw[iv$term]
      iv$truth     <- as.numeric(tv)
      iv$truth_raw <- as.numeric(trw)
      iv$cov     <- iv$lo <= iv$truth     & iv$truth     <= iv$hi
      iv$cov_raw <- iv$lo <= iv$truth_raw & iv$truth_raw <= iv$hi
      iv$width   <- iv$hi - iv$lo
      row[[paste0(r, "_ci")]] <- iv
    }
    row[[paste0(r, "_secs")]] <-
      as.numeric(difftime(Sys.time(), t0, units = "secs"))
    rm(fit); gc(verbose = FALSE)
  }
  saveRDS(row, f)

  gci <- row$nested_laplace_ci; nci <- row$nuts_ci
  pick <- function(d, col) if (is.null(d)) NA else
    d[[col]][match("psi_(Intercept)", d$term)]
  cat(sprintf(
    "sigma %.1f seed %3d | det %2d | grid fs %6.3f psi w %6.3f cov %-5s | nuts fs %6.3f psi w %6.3f cov %-5s\n",
    sg, s, row$n_det_sites,
    row$nested_laplace_field_sd %||% NA_real_, pick(gci, "width"), pick(gci, "cov"),
    row$nuts_field_sd %||% NA_real_, pick(nci, "width"), pick(nci, "cov")))
  flush.console()
}
# One marker per LEVEL. The three levels run as separate processes against one
# `out_dir`, so a single shared `DONE` would be written by whichever finished
# first and read as the whole sweep being complete.
writeLines(format(Sys.time()),
           file.path(out_dir, sprintf("DONE_s%s",
                                      paste(SIGMAS, collapse = "_"))))
cat("done\n")
