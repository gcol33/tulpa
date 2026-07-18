# Guards the cyclic RW1 / RW2 rank normalizer that both the production
# exact-NUTS log-posterior (compute_temporal_prior) and the double twin share
# via the single-source tulpa_temporal::rw1_rank / rw2_rank helpers. A cyclic
# RW1 is the cycle-graph Laplacian (one null direction -> rank T-1); a cyclic
# RW2 annihilates only constants on a ring (a linear ramp is not periodic ->
# rank T-1). A regression to `rank = T` in the cyclic branch would bias tau
# high / sigma_temporal low on seasonal fits.

test_that("cyclic RW rank helper returns T-1 for both RW1 and RW2", {
  T_len <- 12L
  expect_equal(tulpa:::cpp_test_temporal_rank("rw1", T_len, TRUE),  T_len - 1L)
  expect_equal(tulpa:::cpp_test_temporal_rank("rw1", T_len, FALSE), T_len - 1L)
  expect_equal(tulpa:::cpp_test_temporal_rank("rw2", T_len, TRUE),  T_len - 1L)  # ring: T-1
  expect_equal(tulpa:::cpp_test_temporal_rank("rw2", T_len, FALSE), T_len - 2L)  # path: T-2
})

# Independent R references for the intrinsic-GMRF log-prior as implemented by
# tulpa_temporal::temporal_log_prior: 0.5 * rank * log(tau) - 0.5 * tau * phi'Qphi
# (the improper-prior normalizer drops the 2pi constant).
.rw1_quad <- function(phi, cyclic) {
  T <- length(phi)
  q <- sum(diff(phi)^2)
  if (cyclic) q <- q + (phi[1] - phi[T])^2
  q
}
.rw2_quad <- function(phi, cyclic) {
  T <- length(phi)
  q <- sum((phi[1:(T - 2)] - 2 * phi[2:(T - 1)] + phi[3:T])^2)
  if (cyclic) {
    q <- q + (phi[T - 1] - 2 * phi[T] + phi[1])^2
    q <- q + (phi[T]     - 2 * phi[1] + phi[2])^2
  }
  q
}
.ref_lp <- function(phi, type, tau, cyclic) {
  T <- length(phi)
  if (type == "rw1") { rank <- T - 1;                q <- .rw1_quad(phi, cyclic) }
  else               { rank <- if (cyclic) T - 1 else T - 2; q <- .rw2_quad(phi, cyclic) }
  0.5 * rank * log(tau) - 0.5 * tau * q
}

test_that("temporal log-prior matches the intrinsic-GMRF normalizer (cyclic + acyclic)", {
  set.seed(11)
  phi <- rnorm(12)
  for (cyc in c(FALSE, TRUE)) {
    for (ty in c("rw1", "rw2")) {
      got <- tulpa:::cpp_test_temporal_log_prior(phi, ty, tau = 1.7, rho = 0, cyclic = cyc)
      expect_equal(got, .ref_lp(phi, ty, 1.7, cyc), tolerance = 1e-9,
                   info = paste(ty, "cyclic =", cyc))
    }
  }
})
