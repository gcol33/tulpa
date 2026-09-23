# Does the per-cell fixed-effect variance grow with that cell's field scale?
#
# The reported marginal is the law-of-total-variance recombination over the
# outer grid: sum_k w_k (V_k + mu_k mu_k') - mean mean'. As sigma grows the
# field's prior precision falls and the fixed effects become LESS identified,
# so V_k must grow with the cell's sigma. If V_k is flat in sigma, the marginal
# cannot respond to the scale -- which is what gcol33/tulpa#862 measures.

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
sim <- simulate_occu_cover(N = N, J = J, positive = "beta", adj = adj,
                           sigma = 0.7, alpha = 1, seed = 2L)
long <- data.frame(site_id = rep(seq_len(N), each = J),
                   visit = rep(seq_len(J), times = N),
                   y = as.vector(t(sim$y)),
                   det_cov1 = sim$visit_data$det_cov1,
                   pos_cov1 = sim$visit_data$pos_cov1)
od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                det.covs = c("det_cov1", "pos_cov1"))
y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0

f <- tobs(~ occ_cov1 + icar(graph = adj),
          data = cbind(data.frame(site_id = seq_len(N)), sim$data),
          family = occu_cover("beta"), detection = ~ det_cov1,
          positive = ~ pos_cov1 + share(spatial()),
          y = od$y, y_pos = y_pos, visits = od$det.covs,
          method = "nested_laplace",
          control = list(engine = "joint", verbose = FALSE, n.threads.outer = 8L,
                         sigma.grid = exp(seq(log(0.1), log(3), length.out = 13L)),
                         phi.grid.pos = exp(seq(log(1), log(60), length.out = 11L))))

jf <- f$joint_fit
H  <- jf$grid_hessians
M  <- jf$grid_modes
w  <- jf$weights; w <- w / sum(w[is.finite(w) & w > 0], na.rm = TRUE)
sg <- jf$theta_grid[, "sigma"]

ok <- which(is.finite(w) & w > 0 &
            !vapply(M, is.null, logical(1)) & !vapply(H, is.null, logical(1)))
cat(sprintf("cells solved=%d  with weight=%d\n", length(w), length(ok)))

# psi_(Intercept) is the first fixed coordinate on this layout.
v1 <- vapply(ok, function(k) {
  Vk <- tryCatch(solve(as.matrix(H[[k]])), error = function(e) NULL)
  if (is.null(Vk)) NA_real_ else Vk[1, 1]
}, numeric(1))
m1 <- vapply(ok, function(k) as.numeric(M[[k]])[1], numeric(1))

o <- order(sg[ok])
cat("\n  sigma   weight   V_k[1,1]   sd_k    mode_k\n")
for (i in o) {
  cat(sprintf("%7.3f %8.4f %10.5f %7.4f %9.4f\n",
              sg[ok][i], w[ok][i], v1[i], sqrt(max(v1[i], 0)), m1[i]))
}

within  <- sum(w[ok] * v1, na.rm = TRUE)
mbar    <- sum(w[ok] * m1, na.rm = TRUE)
between <- sum(w[ok] * (m1 - mbar)^2, na.rm = TRUE)
cat(sprintf("\nwithin  = sum w_k V_k        = %.5f  (sd %.4f)\n", within, sqrt(within)))
cat(sprintf("between = sum w_k (mu_k-mu)^2 = %.5f  (sd %.4f)\n", between, sqrt(between)))
cat(sprintf("total                          = %.5f  (sd %.4f)\n",
            within + between, sqrt(within + between)))
cat(sprintf("reported sd(draws[, psi_int])  = %.4f\n",
            stats::sd(f$draws[, match("psi_(Intercept)", f$param_names)])))
cat(sprintf("\ncor(sigma, V_k[1,1]) over weighted cells = %+.4f\n",
            suppressWarnings(stats::cor(sg[ok], v1, use = "complete.obs"))))
