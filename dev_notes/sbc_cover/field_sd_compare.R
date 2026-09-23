# The comparison the simulator's own docs prescribe: `field_sd`, not the raw
# `sigma`. `simulate_occu_cover(sigma = 0.7)` is in the geo-mean-marginal-SD
# (Sorbye-Rue) convention, comparable to `fit$spatial$field_sd_mean` on the
# nested_laplace path and `fit$hyper_draws[, "field_sd"]` on nuts, and NEVER to
# the raw sigma a nested_laplace fit reports.

suppressMessages(devtools::load_all(".", quiet = TRUE))
suppressPackageStartupMessages(library(tulpaObs))
`%||%` <- function(x, y) if (is.null(x)) y else x

GRID <- 8L; J <- 3L
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

cat(sprintf("%-5s %-9s %10s %10s %10s\n",
            "seed", "route", "field_sd", "raw_sigma", "psi_sd"))
for (s in c(2L, 4L, 1L)) {
  sim <- simulate_occu_cover(N = N, J = J, positive = "beta", adj = adj,
                             sigma = 0.7, alpha = 1, seed = s)
  long <- data.frame(site_id = rep(seq_len(N), each = J),
                     visit = rep(seq_len(J), times = N),
                     y = as.vector(t(sim$y)),
                     det_cov1 = sim$visit_data$det_cov1,
                     pos_cov1 = sim$visit_data$pos_cov1)
  od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                  det.covs = c("det_cov1", "pos_cov1"))
  y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0
  dat <- cbind(data.frame(site_id = seq_len(N)), sim$data)
  common <- list(y = od$y, y_pos = y_pos, visits = od$det.covs)
  base <- list(~ occ_cov1 + icar(graph = adj), data = dat,
               family = occu_cover("beta"), detection = ~ det_cov1,
               positive = ~ pos_cov1 + share(spatial()))

  gr <- do.call(tobs, c(base, common, list(method = "nested_laplace",
        control = list(engine = "joint", verbose = FALSE, n.threads.outer = 8L,
                       sigma.grid = exp(seq(log(0.1), log(3), length.out = 13L)),
                       phi.grid.pos = exp(seq(log(1), log(60), length.out = 11L))))))
  nu <- do.call(tobs, c(base, common, list(method = "nuts",
        control = list(n.iter = 4000L, n.warmup = 2000L, n.chains = 4L,
                       verbose = FALSE))))

  gj <- match("psi_(Intercept)", gr$param_names)
  nj <- match("psi_(Intercept)", nu$param_names)
  hd <- nu$hyper_draws
  k  <- match("field_sd", colnames(hd) %||% character(0))

  cat(sprintf("%-5d %-9s %10.4f %10.4f %10.4f\n", s, "grid",
              gr$spatial$field_sd_mean %||% NA_real_,
              gr$spatial$sigma_mean    %||% NA_real_,
              stats::sd(gr$draws[, gj])))
  cat(sprintf("%-5d %-9s %10.4f %10.4f %10.4f\n", s, "nuts",
              if (is.na(k)) NA_real_ else mean(hd[, k]),
              nu$spatial$sigma_mean %||% NA_real_,
              stats::sd(nu$draws[, nj])))
  cat(sprintf("%-5s %-9s %10.4f %10s %10s   <- simulated\n", "", "truth",
              0.7, "-", "-"))
  flush.console()
}
