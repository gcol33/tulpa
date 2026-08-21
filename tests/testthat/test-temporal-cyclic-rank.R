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

# Independent R references for the temporal log-priors. These are the
# DEFINITION: a kernel that disagrees with them is the defect.
#
# An intrinsic GMRF with precision tau * Q0 and rank(Q0) = r has density
#
#   log p = -0.5*r*log(2*pi) + 0.5*r*log(tau) + 0.5*log|Q0|_+ - 0.5*tau*x'Q0x
#
# The pseudo-determinant 0.5*log|Q0|_+ is constant in tau and is deliberately
# omitted by tulpa_temporal::gmrf_log_norm, so it is omitted here too: every
# quantity within one model is on a common scale, and a log-marginal is not
# comparable across temporal structures. The 2*pi term is NOT omitted -- it is
# what puts the Laplace and sampler tiers on one scale.
.LOG_2PI <- log(2 * pi)

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
  0.5 * rank * (log(tau) - .LOG_2PI) - 0.5 * tau * q
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

test_that("the RW normalizer carries the 2*pi term at every rank", {
  # The offset a missing 2*pi leaves is 0.5 * rank * log(2*pi), which tracks the
  # structure rather than being one constant: 10.1 nats for a rank-11 RW1 on
  # T = 12, 9.2 for the rank-10 acyclic RW2 on the same field. Pinned so a copy
  # that normalizes on tau alone is named by its size, not just by a mismatch.
  set.seed(12)
  phi <- rnorm(12)
  for (cyc in c(FALSE, TRUE)) {
    for (ty in c("rw1", "rw2")) {
      rank <- if (ty == "rw1") 11 else if (cyc) 11 else 10
      got <- tulpa:::cpp_test_temporal_log_prior(phi, ty, tau = 1.7, rho = 0, cyclic = cyc)
      no_2pi <- 0.5 * rank * log(1.7) -
        0.5 * 1.7 * (if (ty == "rw1") .rw1_quad(phi, cyc) else .rw2_quad(phi, cyc))
      expect_equal(no_2pi - got, 0.5 * rank * .LOG_2PI, tolerance = 1e-9,
                   info = paste(ty, "cyclic =", cyc))
    }
  }
})

# ---------------------------------------------------------------------------
# AR1: proper, rank T, and one floor on the stationary factor 1 - rho^2.
# ---------------------------------------------------------------------------

# The engine floors 1 - rho^2 at kAr1StationaryFloor (hmc_temporal.h) rather
# than flooring the stationary precision tau * (1 - rho^2), so where the guard
# engages is a property of rho alone and does not move with tau.
.AR1_FLOOR <- 1e-10

.ref_ar1 <- function(phi, tau, rho) {
  T <- length(phi)
  omr2 <- max(1 - rho * rho, .AR1_FLOOR)
  q <- omr2 * phi[1]^2 + sum((phi[-1] - rho * phi[-T])^2)
  0.5 * T * (log(tau) - .LOG_2PI) + 0.5 * log(omr2) - 0.5 * tau * q
}

# Fully independent arbiter for the interior of the stationary region: the
# dense multivariate normal with the AR1 covariance, built from the correlation
# matrix rather than from any factorization the kernel uses.
.ref_ar1_dense <- function(phi, tau, rho) {
  T <- length(phi)
  d <- abs(outer(seq_len(T), seq_len(T), "-"))
  Sigma <- (1 / tau) / (1 - rho^2) * rho^d
  ch <- chol(Sigma)
  z <- backsolve(ch, phi, transpose = TRUE)
  -0.5 * T * .LOG_2PI - sum(log(diag(ch))) - 0.5 * sum(z^2)
}

test_that("AR1 log-prior equals the dense multivariate normal it factorizes", {
  set.seed(13)
  phi <- rnorm(9)
  for (tau in c(0.4, 1.7, 12)) {
    for (rho in c(-0.85, -0.2, 0, 0.3, 0.95)) {
      got <- tulpa:::cpp_test_temporal_log_prior(phi, "ar1", tau = tau,
                                                 rho = rho, cyclic = FALSE)
      expect_equal(got, .ref_ar1_dense(phi, tau, rho), tolerance = 1e-9,
                   info = paste("tau =", tau, "rho =", rho))
      expect_equal(got, .ref_ar1(phi, tau, rho), tolerance = 1e-9,
                   info = paste("tau =", tau, "rho =", rho))
    }
  }
})

test_that("AR1 log-prior stays finite at and past the stationarity boundary", {
  # An unregularized stationary variance sigma^2 / (1 - rho^2) diverges here and
  # took the log-prior to -Inf just inside the boundary.
  set.seed(14)
  phi <- rnorm(9)
  for (tau in c(0.01, 1, 100)) {
    for (rho in c(0.999999, 1 - 1e-9, 1 - 1e-13, 1, -1)) {
      got <- tulpa:::cpp_test_temporal_log_prior(phi, "ar1", tau = tau,
                                                 rho = rho, cyclic = FALSE)
      expect_true(is.finite(got), info = paste("tau =", tau, "rho =", rho))
      expect_equal(got, .ref_ar1(phi, tau, rho), tolerance = 1e-9,
                   info = paste("tau =", tau, "rho =", rho))
    }
  }
})

test_that("the AR1 floor engages on rho alone, not on the stationary precision", {
  # A floor on tau * (1 - rho^2) would engage at 1 - rho^2 = 1e-12 when
  # tau = 100 and at 1e-8 when tau = 0.01, so the same rho would be regularized
  # in one fit and not in another. Holding phi and rho fixed, the difference
  # between two taus must be exactly the normalizer's, with no floor term left
  # over.
  set.seed(15)
  phi <- rnorm(7)
  # 1 - rho^2 is about 1e-9 here: above the correlation floor at either tau, and
  # below a precision floor at tau = 0.01 but not at tau = 100.
  rho <- 1 - 5e-10
  lp <- function(tau) {
    tulpa:::cpp_test_temporal_log_prior(phi, "ar1", tau = tau, rho = rho,
                                        cyclic = FALSE)
  }
  omr2 <- 1 - rho * rho
  expect_gt(omr2, .AR1_FLOOR)
  q <- omr2 * phi[1]^2 + sum((phi[-1] - rho * phi[-length(phi)])^2)
  expect_equal(lp(100) - lp(0.01),
               0.5 * length(phi) * log(100 / 0.01) - 0.5 * (100 - 0.01) * q,
               tolerance = 1e-9)
})
