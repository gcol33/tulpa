# gcol33/tulpa#633 -- the copy_alpha axis does not densify with its posterior.
#
# Sweeps the requested node count on the two axes a caller CAN set (sigma.grid,
# phi.grid.pos) and reads the base fit's outer-grid quadrature ESS plus the
# surviving node count per axis. The copy_alpha axis is not settable here: the
# positive formula carries copy(), under which tulpaObs rejects control$alpha.grid
# (family_cover_hurdle.R:109-113).
#
# Expectation if the axes all responded: ESS rises with the node count at every J.
# Observed: it does at J = 10 and stops at J = 30, where alpha's surviving count
# pins near 13 while sigma and phi_pos track the request exactly.
#
# Rscript dev_notes/issue633/probe_alpha_axis.R

suppressMessages(library(tulpaObs))

J_SET    <- c(10L, 30L)
NODE_SET <- c(13L, 17L, 21L, 25L, 29L)
GRID     <- 8L

cat(sprintf("tulpa %s | tulpaObs %s\n",
            packageVersion("tulpa"), packageVersion("tulpaObs")))
cat(sprintf("declared copy_alpha axis: %d nodes [%g, %g]\n",
            length(tulpa:::.nl_grid_axis("copy_alpha")),
            min(tulpa:::.nl_grid_axis("copy_alpha")),
            max(tulpa:::.nl_grid_axis("copy_alpha"))))
cat(sprintf("declared field_sd   axis: %d nodes [%g, %g]\n\n",
            length(tulpa:::.nl_grid_axis("field_sd")),
            min(tulpa:::.nl_grid_axis("field_sd")),
            max(tulpa:::.nl_grid_axis("field_sd"))))

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
rows <- list()

for (J in J_SET) {
  sim <- simulate_occu_cover(N = N, J = J, positive = "lognormal",
                             adj = adj, sigma = 0.7, alpha = 1, seed = 1L)
  long <- data.frame(site_id = rep(seq_len(N), each = J),
                     visit = rep(seq_len(J), times = N),
                     y = as.vector(t(sim$y)),
                     det_cov1 = sim$visit_data$det_cov1,
                     pos_cov1 = sim$visit_data$pos_cov1)
  od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                  det.covs = c("det_cov1", "pos_cov1"))
  y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0

  for (k in NODE_SET) {
    ctl <- list(engine = "joint", verbose = FALSE,
                sigma.grid   = exp(seq(log(0.10), log(3.00), length.out = k)),
                phi.grid.pos = exp(seq(log(0.20), log(0.90), length.out = k)))
    fit <- tobs(~ occ_cov1 + icar(graph = adj),
                data = cbind(data.frame(site_id = seq_len(N)), sim$data),
                family = occu_cover("lognormal"),
                detection = ~ det_cov1,
                positive = ~ pos_cov1 + copy(spatial()),
                y = od$y, y_pos = y_pos, visits = od$det.covs,
                method = "nested_laplace", control = ctl)
    og    <- fit$joint_fit
    cells <- nrow(og$theta_grid)
    ess   <- 1 / sum(og$weights^2)
    naxis <- vapply(colnames(og$theta_grid),
                    function(a) length(unique(og$theta_grid[, a])), integer(1))
    cat(sprintf("J=%3d nodes=%2d  cells=%5d  ess=%5.1f  surviving: %s\n",
                J, k, cells, ess,
                paste(sprintf("%s=%d", names(naxis), naxis), collapse = " ")))
    rows[[length(rows) + 1L]] <- data.frame(
      J = J, nodes = k, cells = cells, ess = ess,
      axes = paste(sprintf("%s=%d", names(naxis), naxis), collapse = " "))
  }
}

res <- do.call(rbind, rows)
write.csv(res, file.path("dev_notes", "issue633", "probe_alpha_axis.csv"),
          row.names = FALSE)
cat("\n")
print(res[, c("J", "nodes", "cells", "ess")], row.names = FALSE)
