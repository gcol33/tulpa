# Worked example: periodic-AR1 GMRF latent block via tgmrf_cpp() (the
# compiled-C++ analogue of inst/examples/tgmrf_periodic_ar1.R).
#
# Mirrors the R-closure example end-to-end -- same data, same outer grid,
# same Poisson likelihood. The only difference is that Q(theta) is built
# in compiled C++ from inst/examples/tgmrf_periodic_ar1.cpp instead of
# from an R closure.
#
# Run:
#   Rscript inst/examples/tgmrf_periodic_ar1_cpp.R
#
# First call compiles the .cpp and caches the resulting DLL under
# tulpa_cache_dir(); subsequent calls hit the cache and load in < 1 s.

library(tulpa)

cpp_file <- system.file("examples", "tgmrf_periodic_ar1.cpp",
                        package = "tulpa")
if (!nzchar(cpp_file)) {
  # When running from the source tree (devtools::load_all), system.file
  # returns ""; fall back to the absolute repo path.
  cpp_file <- "inst/examples/tgmrf_periodic_ar1.cpp"
}

# -- 1. Compile + register the block -------------------------------------------
# n must match kN inside the .cpp (fixed at 80). The R constructor verifies
# this by evaluating Q(init) once at registration time.
blk <- tgmrf_cpp(
  cpp_file = cpp_file,
  id       = "tgmrf_periodic_ar1",
  init     = c(log_sigma = 0, atanh_rho = atanh(0.5)),
  bounds   = list(lower = c(log(0.3), atanh(0.0)),
                  upper = c(log(3.0), atanh(0.95))),
  name     = "periodic_ar1_cpp"
)
stopifnot(blk$backend == "cpp")
stopifnot(blk$n_latent == 80L)

# -- 2. Simulate Poisson y from the true AR(1) latent --------------------------
set.seed(2026)
n <- blk$n_latent
sigma_true <- 0.8
rho_true   <- 0.7
z <- numeric(n)
z[1] <- rnorm(1, 0, sigma_true)
for (t in 2:n) {
  z[t] <- rho_true * z[t - 1] + rnorm(1, 0, sigma_true * sqrt(1 - rho_true^2))
}
X <- matrix(1, nrow = n, ncol = 1L)
beta_true <- 0.4
eta <- beta_true + z
y <- rpois(n, exp(eta))

# -- 3. Fit ---------------------------------------------------------------------
fit <- tulpa_nested_laplace(
  y        = y,
  n_trials = rep(1L, n),
  X        = X,
  prior    = blk,
  family   = "poisson",
  max_iter = 100L,
  tol      = 1e-8
)

# -- 4. Report ------------------------------------------------------------------
cat("\nPosterior moments (C++ backend):\n")
print(rbind(mean = fit$theta_mean, sd = fit$theta_sd))
cat("\nMapped to (sigma, rho):\n")
sigma_hat <- exp(fit$theta_mean[1])
rho_hat   <- tanh(fit$theta_mean[2])
cat(sprintf("  sigma_hat = %.3f  (true %.3f)\n", sigma_hat, sigma_true))
cat(sprintf("  rho_hat   = %.3f  (true %.3f)\n", rho_hat,   rho_true))
cat("\nlog_marginal range across the outer grid:",
    paste(format(range(fit$log_marginal), digits = 5), collapse = " -- "),
    "\n")
