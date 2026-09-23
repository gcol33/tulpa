# Grid against sampler on the SAME dataset: how wide is each reported marginal?
#
# The SBC arms cannot answer this. Posterior-predictive SBC draws its generating
# truths from the base posterior it is checking, so a scale error shared by the
# base fit and the refits is largely invisible to it -- J10 reports a calibrated
# sd(qnorm(PIT)) of 1.009 on a base posterior that is 27% narrower than the
# sampler's. This fits both routes on one dataset at a time and compares the
# reported sd directly.
#
# One dataset per seed, both routes, written per seed so an interrupted run
# costs the seed in flight and resumes. Everything except `method` and the
# control list it implies is held at the values sbc_occu_cover.R uses.
#
#   Rscript width_grid_vs_nuts.R <out_dir> [n_seeds] [J]

suppressPackageStartupMessages({
  library(tulpa)
  library(tulpaObs)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "width_grid_vs_nuts"
n_seeds <- if (length(args) >= 2) as.integer(args[2]) else 20L
J       <- if (length(args) >= 3) as.integer(args[3]) else 3L

FAMILY  <- "beta"
GRID    <- 8L
K_SIGMA <- 13L
K_PHI   <- 11L
N_ITER  <- 2000L
N_WARMUP <- 1000L
N_THREADS <- 8L

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("tulpa %s | tulpaObs %s | J=%d | %d seeds | %s\n",
            packageVersion("tulpa"), packageVersion("tulpaObs"),
            J, n_seeds, format(Sys.time())))

rook_adj <- function(g) {
  N <- g * g
  A <- matrix(0L, N, N)
  idx <- function(r, c) (r - 1L) * g + c
  for (r in seq_len(g)) for (c in seq_len(g)) {
    s <- idx(r, c)
    if (r > 1L) A[s, idx(r - 1L, c)] <- 1L
    if (r < g) A[s, idx(r + 1L, c)] <- 1L
    if (c > 1L) A[s, idx(r, c - 1L)] <- 1L
    if (c < g) A[s, idx(r, c + 1L)] <- 1L
  }
  A
}

adj <- rook_adj(GRID)
N   <- nrow(adj)

ctl_grid <- list(
  engine = "joint", verbose = FALSE, n.threads.outer = N_THREADS,
  sigma.grid   = exp(seq(log(0.10), log(3.00), length.out = K_SIGMA)),
  phi.grid.pos = exp(seq(log(1.00), log(60.00), length.out = K_PHI))
)
ctl_nuts <- list(n.iter = N_ITER, n.warmup = N_WARMUP, n.chains = 1L,
                 verbose = FALSE)

fit_one <- function(sim, method) {
  long <- data.frame(site_id = rep(seq_len(N), each = J),
                     visit   = rep(seq_len(J), times = N),
                     y       = as.vector(t(sim$y)),
                     det_cov1 = sim$visit_data$det_cov1,
                     pos_cov1 = sim$visit_data$pos_cov1)
  od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                  det.covs = c("det_cov1", "pos_cov1"))
  y_pos <- sim$y_pos
  y_pos[is.na(y_pos)] <- 0
  tobs(~ occ_cov1 + icar(graph = adj),
       data      = cbind(data.frame(site_id = seq_len(N)), sim$data),
       family    = occu_cover(FAMILY),
       detection = ~ det_cov1,
       positive  = ~ pos_cov1 + share(spatial()),
       y = od$y, y_pos = y_pos, visits = od$det.covs,
       method = method,
       control = if (identical(method, "nuts")) ctl_nuts else ctl_grid)
}

read_marginals <- function(fit) {
  d <- fit$draws
  if (is.null(d)) return(NULL)
  nm <- fit$param_names
  data.frame(param = nm,
             mean  = apply(d, 2L, mean),
             sd    = apply(d, 2L, stats::sd),
             stringsAsFactors = FALSE)
}

for (s in seq_len(n_seeds)) {
  f <- file.path(out_dir, sprintf("seed%03d.rds", s))
  if (file.exists(f)) { cat(sprintf("seed %3d  already done\n", s)); next }
  t0 <- Sys.time()
  sim <- simulate_occu_cover(N = N, J = J, positive = FAMILY, adj = adj,
                             sigma = 0.7, alpha = 1, seed = s)
  res <- list(seed = s, J = J,
              tulpa = as.character(packageVersion("tulpa")),
              tulpaObs = as.character(packageVersion("tulpaObs")))
  for (m in c("nested_laplace", "nuts")) {
    fit <- try(fit_one(sim, m), silent = TRUE)
    if (inherits(fit, "try-error")) {
      res[[m]] <- NULL
      res[[paste0(m, "_error")]] <- as.character(fit)
      next
    }
    res[[m]] <- read_marginals(fit)
    if (identical(m, "nuts")) res$divergent <- sum(fit$divergent %||% 0L)
  }
  saveRDS(res, f)
  p <- function(x, q) if (is.null(x)) NA_real_ else x$sd[match(q, x$param)]
  cat(sprintf("seed %3d  psi_int sd: grid %.4f  nuts %.4f  ratio %.3f  div %s  (%.1f min)\n",
              s, p(res$nested_laplace, "psi_(Intercept)"), p(res$nuts, "psi_(Intercept)"),
              p(res$nested_laplace, "psi_(Intercept)") / p(res$nuts, "psi_(Intercept)"),
              res$divergent %||% NA,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  flush.console()
}

writeLines(format(Sys.time()), file.path(out_dir, "DONE"))
cat("done\n")
