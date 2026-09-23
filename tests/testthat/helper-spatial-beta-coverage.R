# Fixture for the FIXED-EFFECT coverage gate with a spatial field present
# (gcol33/tulpa#862).
#
# Why it is its own fixture. tulpa gates the nested-Laplace fixed-effect
# interval's coverage only on an IID region-grouped RE block
# (`test-nested-laplace-recovery.R`). The spatial recovery tests gate the
# HYPERPARAMETERS -- tau, rho, sigma2, range -- and never beta; the joint-multi
# sweep gates (sigma, alpha) and never beta. So the combination "spatial field
# in the model, is the fixed-effect interval calibrated" had no gate anywhere,
# which is why gcol33/tulpa#862 could run eight rounds of route-versus-route
# comparison with no reference to consult.
#
# It lives in a helper rather than in the test file so the driver that SIZED
# the gate (`dev_notes/sbc_cover/spatial_beta_coverage.R`) and the gate itself
# run the same simulator and the same fit. A threshold sized against a
# different fixture from the one it guards is not a measured threshold.
#
# Adjacency comes from `make_grid_adjacency()` in helper-spatial-grid.R -- the
# lattice CSR the icar prior takes directly -- rather than a further copy of
# the `.chain_adj` that ten test files each define for themselves.

.SPATIAL_BETA_TRUTH <- c(-0.4, 0.6)

# `weak` is the identification regime gcol33/tulpa#862's fixture sits in: an
# 8x8 rook lattice, 64 sites, three replicates per site, a field larger than
# the fixed effects. `strong` is the well-identified control, so a coverage
# loss that is really a loss shows in both and one that is regime-specific is
# visible as such.
.SPATIAL_BETA_REGIMES <- list(
  strong = list(nr = 10L, nc = 10L, reps = 10L, amp = 0.8),
  weak   = list(nr =  8L, nc =  8L, reps =  3L, amp = 1.6)
)

# A spatially smooth unit-variance field, mean-centred: the ICAR's own level is
# not identified, so leaving a non-zero mean in the field would fold an unknown
# constant into the intercept's truth and make its coverage unreadable.
.spatial_beta_field <- function(S, seed) {
  set.seed(seed)
  f <- as.numeric(stats::filter(stats::rnorm(S), rep(1 / 3, 3), circular = TRUE))
  f <- f - mean(f)
  f / stats::sd(f)
}

# One (simulate, fit, read interval) cell. Returns the per-coefficient coverage
# indicator and interval width, or NULL if the fit failed.
.spatial_beta_cell <- function(seed, regime) {
  r    <- .SPATIAL_BETA_REGIMES[[regime]]
  S    <- r$nr * r$nc
  adj  <- make_grid_adjacency(r$nr, r$nc)
  sidx <- rep(seq_len(S), each = r$reps)
  N    <- length(sidx)

  phi <- .spatial_beta_field(S, seed)
  set.seed(seed + 77L)
  X   <- cbind(1, stats::rnorm(N))
  eta <- as.numeric(X %*% .SPATIAL_BETA_TRUTH) + r$amp * phi[sidx]
  y   <- stats::rbinom(N, 1L, stats::plogis(eta))

  prior <- list(type = "icar", spatial_idx = sidx, n_spatial_units = S,
                adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                n_neighbors = adj$n_neighbors,
                tau_grid = 1 / exp(seq(log(0.2), log(2.5), length.out = 7))^2)

  fit <- try(suppressWarnings(tulpa_nested_laplace(
    as.integer(y), rep(1L, N), X, prior = prior, family = "binomial",
    control = list(diagnose_k = FALSE))), silent = TRUE)
  if (inherits(fit, "try-error")) return(NULL)

  ci <- try(as.matrix(stats::confint(fit)), silent = TRUE)
  if (inherits(ci, "try-error")) return(NULL)
  lo <- as.numeric(ci[, ncol(ci) - 1L])
  hi <- as.numeric(ci[, ncol(ci)])
  list(cov = as.integer(.SPATIAL_BETA_TRUTH >= lo & .SPATIAL_BETA_TRUTH <= hi),
       width = hi - lo)
}

# Sweep one regime over `seeds`; returns the per-coefficient coverage rate, the
# raw indicators (so an aggregate can pool trials rather than average rates)
# and the mean widths.
.spatial_beta_sweep <- function(regime, seeds) {
  p  <- length(.SPATIAL_BETA_TRUTH)
  cv <- matrix(NA_integer_, length(seeds), p)
  wd <- matrix(NA_real_,    length(seeds), p)
  off <- 1000L * match(regime, names(.SPATIAL_BETA_REGIMES))
  for (i in seq_along(seeds)) {
    o <- .spatial_beta_cell(off + seeds[i], regime)
    if (is.null(o)) next
    cv[i, ] <- o$cov; wd[i, ] <- o$width
  }
  list(cov = cv, width = wd,
       rate = colMeans(cv, na.rm = TRUE),
       n    = colSums(!is.na(cv)))
}
