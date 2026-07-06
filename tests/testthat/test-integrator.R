# Symplectic integrator selection (SIMP-backed), gcol33/tulpa.

test_that("tulpa_integrator gets, sets, and validates", {
  old <- tulpa_integrator()
  on.exit(tulpa_integrator(old), add = TRUE)

  expect_type(tulpa_integrator(), "character")

  prev <- tulpa_integrator("yoshida4")
  expect_identical(tulpa_integrator(), "yoshida4")
  expect_identical(prev, old)

  tulpa_integrator("leapfrog")
  expect_identical(tulpa_integrator(), "leapfrog")

  expect_error(tulpa_integrator("nope"), "unknown integrator")
  # A bad name must not change the active integrator.
  expect_identical(tulpa_integrator(), "leapfrog")
})

test_that("adaptive selections round-trip by their own name", {
  old <- tulpa_integrator()
  on.exit(tulpa_integrator(old), add = TRUE)

  # The selection name is reported back (not the resolved placeholder), so a
  # save/restore of the current integrator is faithful.
  prev <- tulpa_integrator("adaptive2")
  expect_identical(tulpa_integrator(), "adaptive2")
  expect_identical(prev, old)

  tulpa_integrator("adaptive3")
  expect_identical(tulpa_integrator(), "adaptive3")

  tulpa_integrator("mts", mts_substeps = 3L)
  expect_identical(tulpa_integrator(), "mts")
  expect_error(tulpa_integrator("mts", mts_substeps = 0L), "mts_substeps")

  tulpa_integrator(old)
  expect_identical(tulpa_integrator(), old)
})

test_that("leapfrog is the default and reproduces trajectories bit-for-bit", {
  skip_if_not_slow()
  old <- tulpa_integrator()
  on.exit(tulpa_integrator(old), add = TRUE)
  tulpa_integrator("leapfrog")

  set.seed(1)
  n <- 40L
  X <- cbind(1, seq(-1, 1, length.out = n))
  y <- as.numeric(X %*% c(0.5, 1.2) + rnorm(n, sd = 0.3))
  fit1 <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X, n_iter = 300L, n_warmup = 300L,
    max_treedepth = 8L, adapt_delta = 0.8, seed = 42L, verbose = FALSE)
  fit2 <- tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X, n_iter = 300L, n_warmup = 300L,
    max_treedepth = 8L, adapt_delta = 0.8, seed = 42L, verbose = FALSE)

  expect_identical(fit1$draws, fit2$draws)  # same seed -> byte-identical
  expect_equal(sum(fit1$divergent), 0)
})

test_that("default leapfrog draws match a stored reference (guards integrator drift)", {
  # The same-seed determinism above only proves the build reproduces itself. It
  # does not catch a *change* to the default leapfrog stepper: a refactor that
  # silently alters the trajectory still reproduces its own (new) draws. This
  # test pins the draws to a committed snapshot, so any numeric change to the
  # default path (the "byte-for-byte unchanged" NEWS claim) fails loudly.
  skip_if_not_slow()
  ref_path <- test_path("_ref", "leapfrog_ref.rds")
  skip_if_not(file.exists(ref_path),
              "no stored leapfrog reference (regenerate via tools/gen_leapfrog_ref.R)")
  ref <- readRDS(ref_path)

  old <- tulpa_integrator()
  on.exit(tulpa_integrator(old), add = TRUE)
  tulpa_integrator("leapfrog")

  fit <- tulpa:::cpp_tulpa_fit_generic(
    y_r = ref$y, X_r = ref$X, n_iter = ref$n_iter, n_warmup = ref$n_warmup,
    max_treedepth = ref$max_treedepth, adapt_delta = ref$adapt_delta,
    seed = ref$seed, verbose = FALSE)

  # Byte-identical on the same build (the determinism test proves it); 1e-6
  # tolerates cross-compiler FP reordering while still catching any real change
  # to the leapfrog stepper, which moves draws by >> 1e-3.
  expect_equal(dim(fit$draws), dim(ref$draws))
  expect_equal(fit$draws, ref$draws, tolerance = 1e-6)
})

test_that("leapfrog and yoshida4 both recover a linear-Gaussian model", {
  skip_if_not_slow()
  old <- tulpa_integrator()
  on.exit(tulpa_integrator(old), add = TRUE)

  set.seed(1)
  n <- 60L
  X <- cbind(1, seq(-1, 1, length.out = n))
  beta_true <- c(0.5, 1.2); sigma_true <- 0.3
  y <- as.numeric(X %*% beta_true + rnorm(n, sd = sigma_true))

  fit_one <- function() tulpa:::cpp_tulpa_fit_generic(
    y_r = y, X_r = X, n_iter = 800L, n_warmup = 500L,
    max_treedepth = 8L, adapt_delta = 0.8, seed = 42L, verbose = FALSE)

  for (scheme in c("leapfrog", "minerror2", "adaptive2", "adaptive3", "mts", "yoshida4")) {
    tulpa_integrator(scheme)
    fit <- fit_one()
    beta_hat <- colMeans(fit$draws[, 1:2])
    sigma_hat <- exp(mean(fit$draws[, 3]))
    expect_lt(max(abs(beta_hat - beta_true)), 0.15,
              label = paste(scheme, "beta error"))
    expect_lt(abs(sigma_hat - sigma_true), 0.1,
              label = paste(scheme, "sigma error"))
    expect_equal(sum(fit$divergent), 0, info = scheme)
  }
})
