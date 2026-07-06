# Regenerate the stored leapfrog draws reference used by test-integrator.R.
# Run after an intentional, verified change to the default leapfrog path:
#   Rscript tools/gen_leapfrog_ref.R
# It writes tests/testthat/_ref/leapfrog_ref.rds (self-contained: data + fit
# settings + draws), which the test refits and compares against.

suppressMessages(pkgload::load_all(quiet = TRUE))

tulpa_integrator("leapfrog")

set.seed(1)
n <- 40L
X <- cbind(1, seq(-1, 1, length.out = n))
y <- as.numeric(X %*% c(0.5, 1.2) + rnorm(n, sd = 0.3))

settings <- list(
  n_iter = 600L, n_warmup = 300L,   # 300 retained post-warmup draws
  max_treedepth = 8L, adapt_delta = 0.8, seed = 42L
)

fit <- tulpa:::cpp_tulpa_fit_generic(
  y_r = y, X_r = X, n_iter = settings$n_iter, n_warmup = settings$n_warmup,
  max_treedepth = settings$max_treedepth, adapt_delta = settings$adapt_delta,
  seed = settings$seed, verbose = FALSE)

ref <- c(list(y = y, X = X, draws = fit$draws), settings)

out_dir <- file.path("tests", "testthat", "_ref")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(ref, file.path(out_dir, "leapfrog_ref.rds"))
cat("wrote", file.path(out_dir, "leapfrog_ref.rds"),
    "- draws dim", paste(dim(fit$draws), collapse = "x"), "\n")
