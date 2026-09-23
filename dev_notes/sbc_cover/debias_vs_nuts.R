# gcol33/tulpa#862: does an inner debias close the psi width gap, measured
# against a sampler on the SAME dataset?
#
# Arms: the plain grid, the grid with the subspace debias, the same with the
# coupling closure (which needs the modal-cell joint precision the engine now
# retains), and NUTS as the reference. Seeds 2 and 4 are weakly identified
# (grid/nuts 0.230 and 0.340); seed 1 is well identified (0.698) and is the
# OVERSHOOT control -- a correction that widens a marginal the Gaussian already
# fit is not a fix.

suppressMessages(devtools::load_all(".", quiet = TRUE))
suppressPackageStartupMessages(library(tulpaObs))

`%||%` <- function(x, y) if (is.null(x)) y else x

FAMILY  <- "beta"
GRID    <- 8L
J       <- 3L
K_SIGMA <- 13L
K_PHI   <- 11L

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
  engine = "joint", verbose = FALSE, n.threads.outer = 8L,
  sigma.grid   = exp(seq(log(0.10), log(3.00), length.out = K_SIGMA)),
  phi.grid.pos = exp(seq(log(1.00), log(60.00), length.out = K_PHI))
)
ctl_nuts <- list(n.iter = 2000L, n.warmup = 1000L, n.chains = 1L, verbose = FALSE)

fit_one <- function(sim, method, extra = NULL) {
  long <- data.frame(site_id = rep(seq_len(N), each = J),
                     visit   = rep(seq_len(J), times = N),
                     y       = as.vector(t(sim$y)),
                     det_cov1 = sim$visit_data$det_cov1,
                     pos_cov1 = sim$visit_data$pos_cov1)
  od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                  det.covs = c("det_cov1", "pos_cov1"))
  y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0
  ctl <- if (identical(method, "nuts")) ctl_nuts else c(ctl_grid, extra)
  tobs(~ occ_cov1 + icar(graph = adj),
       data      = cbind(data.frame(site_id = seq_len(N)), sim$data),
       family    = occu_cover(FAMILY),
       detection = ~ det_cov1,
       positive  = ~ pos_cov1 + share(spatial()),
       y = od$y, y_pos = y_pos, visits = od$det.covs,
       method = method, control = ctl)
}

psi_sd <- function(fit) {
  d <- fit$draws
  if (is.null(d)) return(NA_real_)
  j <- match("psi_(Intercept)", fit$param_names)
  if (is.na(j)) return(NA_real_)
  stats::sd(d[, j])
}

arms <- list(
  plain   = NULL,
  debias  = list(subspace.debias = TRUE),
  closure = list(subspace.debias = list(closure = TRUE)),
  clos_gd = list(subspace.debias = list(band = "good", closure = TRUE))
)

cat(sprintf("%-5s %-9s %9s %9s %8s\n", "seed", "arm", "psi_sd", "vs_nuts", "secs"))
for (s in c(2L, 4L, 1L)) {
  sim <- simulate_occu_cover(N = N, J = J, positive = FAMILY, adj = adj,
                             sigma = 0.7, alpha = 1, seed = s)
  fn <- try(fit_one(sim, "nuts"), silent = TRUE)
  ref <- if (inherits(fn, "try-error")) NA_real_ else psi_sd(fn)
  cat(sprintf("%-5d %-9s %9.4f %9s %8s\n", s, "nuts", ref, "1.000", "-"))
  for (a in names(arms)) {
    t0 <- Sys.time()
    f  <- try(fit_one(sim, "nested_laplace", arms[[a]]), silent = TRUE)
    if (inherits(f, "try-error")) {
      cat(sprintf("%-5d %-9s     ERROR %s\n", s, a,
                  sub("\n.*", "", as.character(f)))); next
    }
    v <- psi_sd(f)
    cat(sprintf("%-5d %-9s %9.4f %9.3f %8.1f\n", s, a, v, v / ref,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    flush.console()
  }
}
