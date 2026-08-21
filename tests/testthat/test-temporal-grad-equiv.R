# The sigma2-parameterized temporal gradient wrappers agree with the canonical
# tau kernels (gcol33/tulpa#142 A7).
#
# RW1 / RW2 / AR1 gradients existed in two independent transcriptions -- one in
# the precision (tau) parameterization, one in the variance (sigma2) one. Only
# RW1 had been made a wrapper; RW2 and AR1 were re-derived by hand and had
# already drifted on their guards, leaving two out-of-bounds accesses on the tau
# side (rw1_grad_w read w[1] at n = 1; rw2_grad_w wrote d[1] into an n-sized
# buffer). RW2 and AR1 are now wrappers too, and these tests pin the agreement.

grad_num <- function(f, w, h = 1e-6) {
  vapply(seq_along(w), function(i) {
    wp <- w; wm <- w
    wp[i] <- wp[i] + h; wm[i] <- wm[i] - h
    (f(wp) - f(wm)) / (2 * h)
  }, numeric(1))
}

test_that("the sigma2 wrappers equal the canonical tau kernels", {
  # The export converts with the exact tau = 1 / sigma2, so the two sides are
  # two parameterizations of one kernel rather than a wrapper compared against
  # its own internal arithmetic. A conversion that reproduced whatever the
  # wrapper does internally would make this pass for any wrapper.
  set.seed(21)
  for (n in c(3L, 5L, 12L)) {
    for (sigma2 in c(0.25, 1.0, 4.0)) {
      for (rho in c(-0.6, 0.0, 0.85)) {
        w <- rnorm(n)
        g <- cpp_test_temporal_grad_equiv(w, sigma2, rho)
        info <- paste("n:", n, "sigma2:", sigma2, "rho:", rho)
        expect_equal(g$rw1_sig, g$rw1_tau, tolerance = 1e-11, info = info)
        expect_equal(g$rw2_sig, g$rw2_tau, tolerance = 1e-11, info = info)
        expect_equal(g$ar1_sig, g$ar1_tau, tolerance = 1e-11, info = info)
        expect_equal(g$rho_sig, g$rho_tau, tolerance = 1e-11, info = info)
      }
    }
  }
})

test_that("the temporal gradients match numerical derivatives of their priors", {
  set.seed(22)
  n <- 8L; sigma2 <- 1.7; rho <- 0.6
  tau <- 1 / sigma2
  w <- rnorm(n)
  g <- cpp_test_temporal_grad_equiv(w, sigma2, rho)

  # RW1: -0.5 * tau * sum(diff(w)^2)
  rw1_lp <- function(v) -0.5 * tau * sum(diff(v)^2)
  expect_equal(g$rw1_tau, grad_num(rw1_lp, w), tolerance = 1e-5)

  # RW2: -0.5 * tau * sum(second differences^2)
  rw2_lp <- function(v) -0.5 * tau * sum(diff(v, differences = 2)^2)
  expect_equal(g$rw2_tau, grad_num(rw2_lp, w), tolerance = 1e-5)

  # AR1: stationary first term + conditional increments
  ar1_lp <- function(v) {
    -0.5 * tau * (1 - rho^2) * v[1]^2 -
      0.5 * tau * sum((v[-1] - rho * v[-length(v)])^2)
  }
  expect_equal(g$ar1_tau, grad_num(ar1_lp, w), tolerance = 1e-5)
})

test_that("degenerate lengths are flat rather than out of bounds", {
  # rw1_grad_w read w[1] at n = 1; rw2_grad_w wrote d[1] into an n-sized buffer.
  # Both now return a zero gradient, matching the sigma2 copies' guards.
  for (n in c(1L, 2L)) {
    g <- cpp_test_temporal_grad_equiv(rnorm(n), 1.0, 0.5)
    expect_equal(g$rw2_tau, rep(0, n), info = paste("n:", n))
    expect_equal(g$rw2_sig, rep(0, n), info = paste("n:", n))
    if (n == 1L) {
      expect_equal(g$rw1_tau, 0, info = "rw1 at n = 1")
      expect_equal(g$rw1_sig, 0, info = "rw1 at n = 1")
    }
  }
})

test_that("the AR1 rho gradient stays finite at the stationarity boundary", {
  # -rho / (1 - rho^2) diverges as rho -> +-1; the tau copy had no guard.
  for (rho in c(-0.999999999, 0.999999999, -1, 1)) {
    g <- cpp_test_temporal_grad_equiv(rnorm(6), 1.0, rho)
    expect_true(is.finite(g$rho_tau), info = paste("rho:", rho))
    expect_true(all(is.finite(g$ar1_tau)), info = paste("rho:", rho))
  }
})

# ---------------------------------------------------------------------------
# Value against gradient: the check the cross-parameterization probe cannot make
# ---------------------------------------------------------------------------

# Comparing two wrappers over one kernel says nothing about the density the
# kernel is supposed to differentiate: a value function that divides by a
# different quantity than its gradient passes that comparison. Central
# differencing the multiscale value functions is the arbiter that sees it, and
# sigma2 = 1e-6 is where an offset of 1e-10 on the precision shows at 1e-4
# relative -- past any tolerance a gradient check would use.

.tg_arms <- list(
  rw1  = list(val = "rw1_val",  gphi = "rw1_sig",  gls2 = "rw1_dls2"),
  rw1c = list(val = "rw1c_val", gphi = "rw1c_sig", gls2 = "rw1c_dls2"),
  rw2  = list(val = "rw2_val",  gphi = "rw2_sig",  gls2 = "rw2_dls2"),
  ar1  = list(val = "ar1_val",  gphi = "ar1_sig",  gls2 = "ar1_dls2"),
  iid  = list(val = "iid_val",  gphi = "iid_sig",  gls2 = "iid_dls2")
)

test_that("each multiscale value function is differentiated by its own gradient", {
  set.seed(23)
  rho <- 0.6
  val <- function(field, w, sigma2) {
    cpp_test_temporal_grad_equiv(w, sigma2, rho)[[field]]
  }
  for (n in c(5L, 12L)) {
    w <- rnorm(n)
    for (sigma2 in c(1e-6, 1e-2, 1.0, 4.0)) {
      g <- cpp_test_temporal_grad_equiv(w, sigma2, rho)
      for (nm in names(.tg_arms)) {
        a <- .tg_arms[[nm]]
        info <- paste(nm, "n:", n, "sigma2:", sigma2)

        h <- 1e-5
        fd_ls2 <- (val(a$val, w, sigma2 * exp(h)) -
                     val(a$val, w, sigma2 * exp(-h))) / (2 * h)
        expect_equal(g[[a$gls2]], fd_ls2, tolerance = 1e-6, info = info)

        fd_phi <- vapply(seq_len(n), function(i) {
          hw <- 1e-6 * max(1, abs(w[i]))
          wp <- w; wm <- w
          wp[i] <- wp[i] + hw; wm[i] <- wm[i] - hw
          (val(a$val, wp, sigma2) - val(a$val, wm, sigma2)) / (2 * hw)
        }, numeric(1))
        expect_equal(g[[a$gphi]], fd_phi, tolerance = 1e-5, info = info)
      }
    }
  }
})

test_that("the AR1 logit-rho gradient differentiates the AR1 density", {
  # rho = 2 * plogis(logit_rho) - 1 is the map the sampler carries, so the
  # reported gradient is on logit_rho and the finite difference has to be taken
  # there too.
  set.seed(24)
  w <- rnorm(9)
  to_rho <- function(l) 2 * plogis(l) - 1
  for (sigma2 in c(1e-6, 1.0)) {
    for (rho in c(-0.3, 0.6, 0.85)) {
      l <- qlogis((rho + 1) / 2)
      h <- 1e-5
      up <- cpp_test_temporal_grad_equiv(w, sigma2, to_rho(l + h))$ar1_val
      dn <- cpp_test_temporal_grad_equiv(w, sigma2, to_rho(l - h))$ar1_val
      got <- cpp_test_temporal_grad_equiv(w, sigma2, rho)$rho_sig
      expect_equal(got, (up - dn) / (2 * h), tolerance = 1e-5,
                   info = paste("sigma2:", sigma2, "rho:", rho))
    }
  }
})
