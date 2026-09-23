# Which route recovers the field scale? (gcol33/tulpa#865, and with it #862)
#
# Every component of the nested-Laplace fixed-effect marginal has been verified
# correct, and the whole remaining difference from the sampler is WHERE the
# field-scale posterior sits: on one fixture the grid read field_sd 0.42 and
# the sampler 1.39, against a simulated 0.7. Three seeds at one true value
# cannot say which is biased -- a single realized field has its own SD.
#
# This simulates at SEVERAL known field scales and asks which route tracks
# truth. Both are compared on `field_sd`, the quantity ?simulate_occu_cover
# prescribes (its `sigma` is the Sorbye-Rue geo-mean-marginal-SD convention,
# NEVER the raw `sigma` a nested_laplace fit reports).
#
#   Rscript field_scale_calibration.R <out_dir> [n_seeds] [routes]
#
# Writes one row per (sigma_true, seed, route) so an interrupted run resumes.

suppressPackageStartupMessages({
  library(tulpa)
  library(tulpaObs)
})
`%||%` <- function(x, y) if (is.null(x)) y else x

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "field_scale_calibration"
n_seeds <- if (length(args) >= 2) as.integer(args[2]) else 5L
routes  <- if (length(args) >= 3) strsplit(args[3], ",")[[1]] else
           c("nested_laplace", "nuts")

SIGMAS <- c(0.3, 0.7, 1.5)
GRID <- 8L; J <- 3L; N_CH <- 2L; N_IT <- 3000L; N_WU <- 1500L
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

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
    list(engine = "joint", verbose = FALSE, n.threads.outer = 8L,
         sigma.grid = exp(seq(log(0.05), log(6), length.out = 15L)),
         phi.grid.pos = exp(seq(log(1), log(60), length.out = 11L)))
  tobs(~ occ_cov1 + icar(graph = adj),
       data = cbind(data.frame(site_id = seq_len(N)), sim$data),
       family = occu_cover("beta"), detection = ~ det_cov1,
       positive = ~ pos_cov1 + share(spatial()),
       y = od$y, y_pos = y_pos, visits = od$det.covs,
       method = route, control = ctl)
}

# `field_sd` on either route, the comparable quantity.
read_field_sd <- function(fit, route) {
  if (identical(route, "nuts")) {
    hd <- fit$hyper_draws
    k  <- match("field_sd", colnames(hd) %||% character(0))
    if (is.na(k)) return(c(NA_real_, NA_real_))
    return(c(mean(hd[, k]), stats::sd(hd[, k])))
  }
  c(fit$spatial$field_sd_mean %||% NA_real_, fit$spatial$field_sd_sd %||% NA_real_)
}

cat(sprintf("tulpa %s | tulpaObs %s | %d seeds x %d sigmas | %s\n",
            packageVersion("tulpa"), packageVersion("tulpaObs"),
            n_seeds, length(SIGMAS), format(Sys.time())))

for (sg in SIGMAS) for (s in seq_len(n_seeds)) {
  f <- file.path(out_dir, sprintf("s%03.1f_seed%03d.rds", sg, s))
  if (file.exists(f)) next
  sim <- simulate_occu_cover(N = N, J = J, positive = "beta", adj = adj,
                             sigma = sg, alpha = 1, seed = s)
  row <- list(sigma_true = sg, seed = s)
  for (r in routes) {
    t0  <- Sys.time()
    fit <- try(fit_route(sim, r), silent = TRUE)
    if (inherits(fit, "try-error")) {
      row[[paste0(r, "_err")]] <- sub("\n.*", "", as.character(fit)); next
    }
    fs <- read_field_sd(fit, r)
    row[[paste0(r, "_field_sd")]]    <- fs[1]
    row[[paste0(r, "_field_sd_sd")]] <- fs[2]
    row[[paste0(r, "_secs")]] <-
      as.numeric(difftime(Sys.time(), t0, units = "secs"))
  }
  saveRDS(row, f)
  cat(sprintf("sigma %.1f seed %2d | grid %6.3f | nuts %6.3f\n", sg, s,
              row$nested_laplace_field_sd %||% NA_real_,
              row$nuts_field_sd %||% NA_real_))
  flush.console()
}
cat("done\n")
