# The NUTS/HMC mass metric, reachable from the sampler front door
# (gcol33/tulpa#545). One string -> MassMatrixType map (tulpa_hmc::
# parse_metric_type) serves every entry point, and one detector
# (detect_mass_blocks) serves the AUTO resolution and an explicit BLOCK_DIAG
# request, so the two spellings of the same request cannot resolve to different
# metrics.
#
# Structural (tier 1) except the two recovery blocks at the end.

.mm_binom <- function(n = 200L, seed = 5L) {
  set.seed(seed)
  x <- rnorm(n)
  X <- cbind(1, x)
  y <- rbinom(n, 1L, plogis(-0.3 + 0.8 * x))
  list(y = as.numeric(y), X = X, n = n)
}

test_that("an unrecognised metric name is rejected rather than defaulted", {
  d <- .mm_binom(60L)
  expect_error(
    tulpa_sample_glmm(d$y, NULL, d$X, "binomial", "hmc",
                      control = list(n_iter = 40L, warmup = 20L, n_chains = 1L,
                                     seed = 1L, mass_matrix = "banded")),
    "should be one of")
})

test_that("every accepted metric name reaches the kernel", {
  d <- .mm_binom(120L)
  for (mm in c("diag", "dense", "block_diag", "auto")) {
    fit <- tulpa_sample_glmm(
      d$y, NULL, d$X, "binomial", "hmc",
      fixed_names = c("(Intercept)", "x"),
      control = list(n_iter = 200L, warmup = 100L, n_chains = 1L,
                     seed = 4L, mass_matrix = mm))
    expect_true(all(is.finite(fit$means)), info = mm)
    expect_identical(ncol(fit$draws), 2L, info = mm)
  }
})

test_that("the default metric is diag, and naming it changes nothing", {
  d <- .mm_binom(150L)
  ctrl <- list(n_iter = 300L, warmup = 150L, n_chains = 2L, seed = 21L)
  a <- tulpa_sample_glmm(d$y, NULL, d$X, "binomial", "hmc", control = ctrl)
  b <- tulpa_sample_glmm(d$y, NULL, d$X, "binomial", "hmc",
                         control = c(ctrl, list(mass_matrix = "diag")))
  expect_identical(a$draws, b$draws)
})

test_that("a backend with no mass matrix refuses a non-default metric", {
  d <- .mm_binom(80L)
  for (backend in c("ess", "vi")) {
    expect_error(
      tulpa_sample_glmm(d$y, NULL, d$X, "binomial", backend,
                        control = list(n_iter = 60L, warmup = 30L, seed = 2L,
                                       vi_max_iter = 50L,
                                       mass_matrix = "dense")),
      "carries no mass matrix", info = backend)
    # The default is still accepted there, so the knob is not a trap for a
    # caller who passes one control list to several backends.
    expect_no_error(
      tulpa_sample_glmm(d$y, NULL, d$X, "binomial", backend,
                        control = list(n_iter = 60L, warmup = 30L, seed = 2L,
                                       vi_max_iter = 50L,
                                       mass_matrix = "diag")))
  }
})

test_that("mass_matrix is a declared control key", {
  expect_true("mass_matrix" %in% .CONTROL_KEYS$sample_glmm)
  d <- .mm_binom(40L)
  expect_error(
    tulpa_sample_glmm(d$y, NULL, d$X, "binomial", "hmc",
                      control = list(n_iter = 20L, mass_matirx = "auto")),
    "mass_matirx")
})

# A model whose layout carries a correlated hyperparameter pair is what the
# block detector was written for: BYM2 lays log_sigma_bym2 immediately before
# logit_rho_bym2, so AUTO gives that pair its own 2x2 block instead of two
# diagonal entries. The assertion is on the fit, not on the metric name the
# kernel logged: the metric is an internal choice, and what has to hold is that
# every one of them samples the same posterior.
test_that("the metrics agree on a BYM2 posterior", {
  skip_if_not_slow()
  set.seed(9)
  S <- 24L
  adj <- matrix(0, S, S)
  for (i in 1:(S - 1)) { adj[i, i + 1] <- 1; adj[i + 1, i] <- 1 }
  phi <- cumsum(rnorm(S, 0, 0.3)); phi <- phi - mean(phi)
  reps <- 8L; N <- S * reps
  unit <- rep(seq_len(S), each = reps)
  x <- rnorm(N)
  b <- c(0.2, 0.6)
  y <- rpois(N, exp(b[1] + b[2] * x + phi[unit]))
  csr <- adjacency_to_csr_tulpa(adj)
  sp <- list(type = "bym2", spatial_idx = as.integer(unit),
             n_spatial_units = S, adj_row_ptr = as.integer(csr$row_ptr),
             adj_col_idx = as.integer(csr$col_idx),
             n_neighbors = as.integer(csr$n_neighbors), scale_factor = 1.0)

  fits <- lapply(c("diag", "auto", "block_diag"), function(mm)
    tulpa_sample_glmm(y, NULL, cbind(1, x), "poisson", "hmc",
                      spatial_spec = sp,
                      fixed_names = c("(Intercept)", "x"),
                      control = list(n_iter = 1200L, warmup = 600L,
                                     n_chains = 2L, seed = 13L,
                                     mass_matrix = mm)))
  slopes <- vapply(fits, function(f) unname(f$means[["x"]]), numeric(1))
  expect_lt(max(abs(slopes - b[2])), 0.20)
  expect_lt(max(abs(slopes - slopes[1])), 0.10)
})
