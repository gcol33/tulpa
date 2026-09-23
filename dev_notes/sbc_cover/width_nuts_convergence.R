# Does the sampler arm in width_grid_vs_nuts.R actually mix?
#
# That script runs one chain at 2000 iterations. A chain that wanders reports a
# marginal sd larger than the posterior's, which would inflate the grid/nuts
# width ratio for reasons that have nothing to do with the grid. This refits the
# named seeds at four chains and twice the length and reports split-Rhat and
# bulk ESS beside the sd, so the two readings can be compared directly.
#
#   Rscript width_nuts_convergence.R <out_dir> <seed> [<seed> ...]

suppressPackageStartupMessages({
  library(tulpa)
  library(tulpaObs)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
seeds <- as.integer(args[-1])

FAMILY <- "beta"; GRID <- 8L; J <- 3L
N_ITER <- 4000L; N_WARMUP <- 2000L; N_CHAINS <- 4L

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

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
adj <- rook_adj(GRID); N <- nrow(adj)

# Split-Rhat and bulk ESS on a draws matrix split into `nc` equal chains.
split_diag <- function(x, nc) {
  n <- length(x)
  if (n < 4L * nc) return(c(rhat = NA_real_, ess = NA_real_))
  m <- (n %/% nc) %/% 2L * 2L
  ch <- lapply(seq_len(nc), function(k) x[((k - 1L) * m + 1L):(k * m)])
  ch <- unlist(lapply(ch, function(z) list(z[1:(m / 2)], z[(m / 2 + 1):m])), FALSE)
  M <- length(ch); L <- m / 2
  mu <- sapply(ch, mean); s2 <- sapply(ch, stats::var)
  W <- mean(s2); B <- L * stats::var(mu)
  vhat <- ((L - 1) * W + B) / L
  rhat <- sqrt(vhat / W)
  rho <- sapply(ch, function(z) {
    a <- stats::acf(z, lag.max = min(200L, L - 2L), plot = FALSE)$acf[-1]
    t <- 1; s <- 0
    while (t < length(a) - 1 && (a[t] + a[t + 1]) > 0) { s <- s + a[t] + a[t + 1]; t <- t + 2 }
    s
  })
  c(rhat = rhat, ess = M * L / (1 + 2 * mean(rho)))
}

fit_nuts <- function(sim) {
  long <- data.frame(site_id = rep(seq_len(N), each = J),
                     visit = rep(seq_len(J), times = N),
                     y = as.vector(t(sim$y)),
                     det_cov1 = sim$visit_data$det_cov1,
                     pos_cov1 = sim$visit_data$pos_cov1)
  od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                  det.covs = c("det_cov1", "pos_cov1"))
  y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0
  tobs(~ occ_cov1 + icar(graph = adj),
       data = cbind(data.frame(site_id = seq_len(N)), sim$data),
       family = occu_cover(FAMILY), detection = ~ det_cov1,
       positive = ~ pos_cov1 + share(spatial()),
       y = od$y, y_pos = y_pos, visits = od$det.covs,
       method = "nuts",
       control = list(n.iter = N_ITER, n.warmup = N_WARMUP,
                      n.chains = N_CHAINS, verbose = FALSE))
}

qs <- c("psi_(Intercept)", "psi_occ_cov1", "p_(Intercept)", "pos_(Intercept)")
for (s in seeds) {
  f <- file.path(out_dir, sprintf("conv%03d.rds", s))
  if (file.exists(f)) { cat(sprintf("seed %3d already done\n", s)); next }
  sim <- simulate_occu_cover(N = N, J = J, positive = FAMILY, adj = adj,
                             sigma = 0.7, alpha = 1, seed = s)
  t0 <- Sys.time()
  fit <- fit_nuts(sim)
  d <- fit$draws; nm <- fit$param_names
  tab <- do.call(rbind, lapply(qs, function(q) {
    i <- match(q, nm)
    dg <- split_diag(d[, i], N_CHAINS)
    data.frame(param = q, sd = stats::sd(d[, i]), mean = mean(d[, i]),
               rhat = dg[["rhat"]], ess = dg[["ess"]])
  }))
  res <- list(seed = s, n_iter = N_ITER, n_warmup = N_WARMUP,
              n_chains = N_CHAINS, divergent = sum(fit$divergent %||% 0L),
              tab = tab)
  saveRDS(res, f)
  cat(sprintf("seed %3d  div %3d  (%.1f min)\n", s, res$divergent,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  print(tab, row.names = FALSE, digits = 4)
  flush.console()
}
cat("done\n")
