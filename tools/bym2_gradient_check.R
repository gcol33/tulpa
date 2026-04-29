# BYM2 H-mode gradient verification.
#
# TODO.md item #1 claims BYM2 H-mode is a numerical fallback, but
# compute_laplace_gradient_bym2_H exists fully implemented at
# src/hmc_icar_collapsed.h:1606.  This script checks whether the H-mode
# dispatched for is_bym2_collapsed agrees with the numerical reference.

suppressPackageStartupMessages({
  devtools::load_all(quiet = TRUE)
})

set.seed(2026)

# 5x5 grid (25 sites)
grid_adj <- function(nx, ny) {
  S <- nx * ny
  A <- matrix(0L, S, S)
  for (i in seq_len(nx)) for (j in seq_len(ny)) {
    s <- (i - 1L) * ny + j
    if (i > 1L)  A[s, (i - 2L) * ny + j] <- 1L
    if (i < nx)  A[s, i * ny + j]         <- 1L
    if (j > 1L)  A[s, (i - 1L) * ny + (j - 1L)] <- 1L
    if (j < ny)  A[s, (i - 1L) * ny + (j + 1L)] <- 1L
  }
  A
}
adj <- grid_adj(5, 5)
S <- nrow(adj)

adj_to_csr <- function(A) {
  S <- nrow(A)
  n_neigh <- rowSums(A != 0)
  row_ptr <- as.integer(c(0L, cumsum(n_neigh)))
  col_idx <- integer(0)
  for (i in seq_len(S)) col_idx <- c(col_idx, which(A[i, ] != 0L) - 1L)
  list(row_ptr = row_ptr, col_idx = as.integer(col_idx),
       n_neighbors = as.integer(n_neigh))
}
csr <- adj_to_csr(adj)

# Sample data: binomial logit with BYM2 spatial term
sigma_total_true <- 0.7
rho_true <- 0.6
n_per <- 8L
N <- S * n_per
site <- rep(seq_len(S), each = n_per)
x <- scale(rnorm(N))[, 1]
phi_true <- rnorm(S, sd = sigma_total_true)  # rough proxy
beta_num <- c(0.0, 0.4)
eta <- beta_num[1] + beta_num[2] * x + phi_true[site]
prob <- 1 / (1 + exp(-eta))
trials <- 10L
y_num <- rbinom(N, size = trials, prob = prob)
y_denom <- rep(trials, N)

X_num <- cbind(1, x)
X_denom <- matrix(1, N, 1)

# bym2_scale_factor: geometric mean of marginal variances for ICAR on this graph.
# Quick numerical estimate: use 1.0 (only affects scaling, not gradient correctness).
bym2_scale <- 1.0

spatial_params <- list(
  type = "bym2",
  group = as.integer(site),
  n_units = S,
  adj_row_ptr = csr$row_ptr,
  adj_col_idx = csr$col_idx,
  n_neighbors = csr$n_neighbors,
  bym2_scale = bym2_scale,
  rho_lower = 0.0,
  rho_upper = 1.0,
  parameterization = "collapsed"
)

re_params <- list(
  group = integer(N), n_groups = 0L, n_terms = 0L,
  parameterization = 1L,
  group_matrix = matrix(0L, 1, 1),
  n_groups_vec = integer(0),
  has_slopes = FALSE, has_correlated_slopes = FALSE,
  n_coefs_vec = integer(0), correlated_vec = integer(0),
  n_chol_vec = integer(0), slope_matrices = list()
)
temporal_params <- list(
  type = "none", time_idx = integer(N), group_idx = integer(N),
  n_times = 0L, n_groups = 0L, n_params = 0L,
  cyclic = FALSE, shared = TRUE,
  tau_shape = 1.0, tau_rate = 0.01
)
prior_params <- list(
  sigma_beta = 2.5, sigma_re_scale = 1.0,
  phi_shape = 1.0, phi_rate = 0.01,
  tau_spatial_shape = 1.0, tau_spatial_rate = 0.01
)
zi_params <- list(type = "none", X = matrix(1, 1, 1), prior_sd = 2.5,
                  X_oi = matrix(1, 1, 1), p_oi = 0L, oi_prior_sd = 2.5,
                  p_zi = 0L)
latent_params <- list(has_latent = FALSE, n_factors = 0L,
                      shared = TRUE, scale = TRUE,
                      constraint = 0L, sigma_prior_rate = 1.0)
st_params <- list(has_spatiotemporal = FALSE)
tvc_params <- list(has_tvc = FALSE)
svc_params <- list(has_svc = FALSE)

# Collapsed BYM2 layout (binomial, no RE, no temporal):
#   [beta_num(2), beta_denom(1), log_sigma_bym2, logit_rho_bym2]
# phi_star and theta_star are inner-mode-found, not parameters.
# Collapsed BYM2 layout (binomial, no RE):
#   [beta_num(2), beta_denom(1), log_sigma_bym2, logit_rho_bym2]
# (phi*, theta* are mode-found inside, not in the param vector)
init <- c(0.0, 0.4,
          0.0,
          log(sigma_total_true),
          log(rho_true / (1 - rho_true)))

cat("Initial param vector length:", length(init), "\n")
cat("N=", N, "S=", S, "\n")

res <- tryCatch(tulpa:::cpp_hmc_fit(
  q_init = init,
  y_num = as.integer(y_num),
  y_denom = as.integer(y_denom),
  y_num_cont = numeric(N),
  y_denom_cont = numeric(N),
  X_num = X_num,
  X_denom = X_denom,
  model_type_str = "binomial",
  re_params = re_params,
  spatial_params = spatial_params,
  temporal_params = temporal_params,
  prior_params = prior_params,
  zi_params = zi_params,
  latent_params = latent_params,
  st_params = st_params,
  tvc_params = tvc_params,
  svc_params = svc_params,
  n_iter = 1L, n_warmup = 0L, L = 1L,
  n_chains = 1L, seed = 1L,
  n_threads = 1L, verbose = TRUE,
  gradient_mode_str = "H",
  max_treedepth = 4L, metric_str = "diag",
  adapt_delta = -1.0, riemannian = -1L,
  gradient_check_only = TRUE
), error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); NULL })

if (is.null(res)) { cat("FAILED\n"); quit(save = "no", status = 1) }

cat("\n=== BYM2 H-mode gradient check ===\n")
cat("h_vs_n :", format(res$h_vs_n, digits = 6), "\n")
cat("ar_vs_n:", format(res$ar_vs_n, digits = 6), "\n")
cat("a_vs_n :", format(res$a_vs_n, digits = 6), "\n")
cat("h_vs_ar:", format(res$h_vs_ar, digits = 6), "\n")
cat("tol    :", res$tol, "\n")
cat("h_ok   :", res$h_ok, "\n")
