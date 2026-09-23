# Is the grid's low field_sd the hyperprior shrinking the field scale?
#
# The grid reads field_sd BELOW the simulated 0.7 on every seed while the
# sampler reads above it, and the fixed-effect marginal width tracks the scale
# on both sides. A PC prior on the field SD pulls toward the base model
# (sigma = 0), so an over-strong default would produce exactly this.
#
# Vary the sigma hyperprior and watch field_sd and psi_sd move together.

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

arms <- list(
  default    = NULL,
  pc_U3_a01  = list(prior.sigma = list("pc.prec", c(3, 0.01))),
  pc_U10_a01 = list(prior.sigma = list("pc.prec", c(10, 0.01))),
  pc_U30_a05 = list(prior.sigma = list("pc.prec", c(30, 0.5)))
)

cat(sprintf("%-5s %-11s %10s %10s %10s\n",
            "seed", "prior", "field_sd", "raw_sigma", "psi_sd"))
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

  for (a in names(arms)) {
    ctl <- c(list(engine = "joint", verbose = FALSE, n.threads.outer = 8L,
                  sigma.grid = exp(seq(log(0.1), log(3), length.out = 13L)),
                  phi.grid.pos = exp(seq(log(1), log(60), length.out = 11L))),
             arms[[a]])
    f <- try(tobs(~ occ_cov1 + icar(graph = adj), data = dat,
                  family = occu_cover("beta"), detection = ~ det_cov1,
                  positive = ~ pos_cov1 + share(spatial()),
                  y = od$y, y_pos = y_pos, visits = od$det.covs,
                  method = "nested_laplace", control = ctl), silent = TRUE)
    if (inherits(f, "try-error")) {
      cat(sprintf("%-5d %-11s ERROR %s\n", s, a,
                  sub("\n.*", "", as.character(f)))); next
    }
    j <- match("psi_(Intercept)", f$param_names)
    cat(sprintf("%-5d %-11s %10.4f %10.4f %10.4f\n", s, a,
                f$spatial$field_sd_mean %||% NA_real_,
                f$spatial$sigma_mean    %||% NA_real_,
                stats::sd(f$draws[, j])))
    flush.console()
  }
  cat(sprintf("%-5s %-11s %10.4f %10s %10.4f   <- truth / nuts\n", "", "ref",
              0.7, "-", c(`2` = 1.6209, `4` = 0.8166, `1` = 0.4391)[[as.character(s)]]))
}
