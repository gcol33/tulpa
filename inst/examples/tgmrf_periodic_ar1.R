# Worked example: user-defined periodic-AR1 GMRF latent block via tgmrf().
#
# Defines the periodic-AR1 precision in a single closure, fits a Poisson GLM
# with the AR1 block as a latent term using the nested-Laplace adapter, and
# reports posterior moments. Mirrors generic-todo.md section 7 (the
# canonical worked example for the tgmrf extensibility surface).
#
# Run:
#   Rscript inst/examples/tgmrf_periodic_ar1.R

library(tulpa)

# -- 1. The user closure: periodic AR(1) precision -----------------------------
periodic_ar1 <- function(n) {
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1])
      rho   <- tanh(theta[2])
      tau   <- 1 / sigma^2
      d <- rep(tau * (1 + rho^2), n)
      o <- rep(-tau * rho, n - 1L)
      M <- Matrix::bandSparse(n, k = c(-1L, 0L, 1L),
                              diagonals = list(o, d, o))
      M[1, n] <- -tau * rho
      M[n, 1] <- -tau * rho
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior = function(theta) {
      stats::dnorm(theta[1], 0, 1, log = TRUE) +
        stats::dnorm(theta[2], 0, 1, log = TRUE)
    },
    init   = c(log_sigma = 0, atanh_rho = atanh(0.5)),
    bounds = list(lower = c(log(0.3), atanh(0.0)),
                  upper = c(log(3.0), atanh(0.95))),
    name   = "periodic_ar1"
  )
}

# -- 2. Simulate Poisson y from the true AR(1) latent -------------------------
set.seed(2026)
n <- 80L
sigma_true <- 0.8
rho_true   <- 0.7
tau_true   <- 1 / sigma_true^2

z <- numeric(n)
z[1] <- rnorm(1, 0, sigma_true)
for (t in 2:n) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, sigma_true * sqrt(1 - rho_true^2))
X <- matrix(1, nrow = n, ncol = 1L)
beta_true <- 0.4
eta <- beta_true + z
y <- rpois(n, exp(eta))

# -- 3. Build the block and fit ------------------------------------------------
blk <- periodic_ar1(n)
print(blk)

fit <- tulpa_nested_laplace(
  y        = y,
  n_trials = rep(1L, n),
  X        = X,
  prior    = blk,           # bare tgmrf — auto-wraps into a 1-block prior
  family   = "poisson",
  control  = list(max_iter = 100L, tol = 1e-8)
)

# -- 4. Report -----------------------------------------------------------------
cat("\nPosterior moments (block parameterisation):\n")
print(rbind(mean = fit$theta_mean, sd = fit$theta_sd))
cat("\nMapped to (sigma, rho):\n")
sigma_hat <- exp(fit$theta_mean[1])
rho_hat   <- tanh(fit$theta_mean[2])
cat(sprintf("  sigma_hat = %.3f  (true %.3f)\n", sigma_hat, sigma_true))
cat(sprintf("  rho_hat   = %.3f  (true %.3f)\n", rho_hat,   rho_true))

cat("\nlog_marginal range across the outer grid:",
    paste(format(range(fit$log_marginal), digits = 5), collapse = " -- "),
    "\n")
