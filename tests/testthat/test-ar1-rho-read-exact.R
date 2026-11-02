# The AR(1) autocorrelation read off the registry door's default grid, against
# the exact posterior. With a gaussian response at a known residual variance the
# latent field and the fixed effects integrate out in closed form, so the
# hyperparameter posterior is exact up to a dense quadrature over (log tau,
# atanh rho) under the engine's own hyperpriors.

skip_on_cran()

.ar1_exact_fixture <- fixture_once(function() {
  set.seed(24)
  Tn <- 60L; per <- 8L; phi <- 0.49
  idx <- rep(seq_len(Tn), each = per)
  eff <- as.numeric(stats::arima.sim(list(ar = 0.8), Tn)) * 0.4
  X <- cbind(1, rnorm(length(idx)))
  y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] + rnorm(length(idx), 0, 0.7)
  n <- length(y)
  A <- matrix(0, n, Tn); A[cbind(seq_len(n), idx)] <- 1
  W <- cbind(X, A); WtW <- crossprod(W); Wty <- crossprod(W, y)
  q_ar1 <- function(rho) {
    Q <- diag(c(1, rep(1 + rho^2, Tn - 2L), 1))
    Q[cbind(1:(Tn - 1L), 2:Tn)] <- -rho
    Q[cbind(2:Tn, 1:(Tn - 1L))] <- -rho
    Q
  }
  # log p(y | tau, rho) up to a constant, the fixed effects under a vague
  # Gaussian prior, as the registry door integrates them.
  loglik <- function(tau, rho) {
    Qz <- matrix(0, Tn + 2L, Tn + 2L)
    Qz[1:2, 1:2] <- diag(1e-6, 2)
    Qz[3:(Tn + 2L), 3:(Tn + 2L)] <- tau * q_ar1(rho)
    R <- chol(Qz + WtW / phi)
    m <- backsolve(R, forwardsolve(t(R), Wty / phi))
    0.5 * (Tn * log(tau) + log(1 - rho^2)) - sum(log(diag(R))) +
      0.5 * sum(m * Wty / phi)
  }
  hp_tau <- .hp_axis_default("tau", list(type = "ar1"))$fn
  hp_rho <- .hp_axis_default("rho", list(type = "ar1"))$fn
  G <- expand.grid(lt = seq(log(0.3), log(100), length.out = 121),
                   zr = seq(atanh(-0.6), atanh(0.995), length.out = 121))
  tau <- exp(G$lt); rho <- tanh(G$zr)
  lp <- mapply(loglik, tau, rho) + hp_tau(tau) + hp_rho(rho) + log(1 - rho^2)
  w <- exp(lp - max(lp)); w <- w / sum(w)
  o <- order(rho); cw <- cumsum(w[o])
  exact <- c(mean = sum(w * rho),
             lo = stats::approx(cw, rho[o], 0.025, ties = "ordered")$y,
             hi = stats::approx(cw, rho[o], 0.975, ties = "ordered")$y)
  fit <- function(ctrl = list()) suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    prior = list(type = "ar1", temporal_idx = idx, n_times = Tn),
    family = "gaussian", phi = phi,
    control = utils::modifyList(list(n_threads = 1L, diagnose_k = FALSE,
                                     diagnose_skew = FALSE, progress = FALSE),
                                ctrl)))
  list(exact = exact, default = fit(),
       recentred = fit(list(auto_recenter = "always")))
})

test_that("the default AR(1) grid reads rho's interval near the exact one", {
  skip_on_cran()
  fx <- .ar1_exact_fixture()
  ex <- fx$exact
  # Exact: mean 0.835, 95% interval (0.663, 0.976).
  for (f in list(fx$default, fx$recentred)) {
    expect_lt(abs(f$theta_mean[["rho"]] - ex[["mean"]]), 0.01)
    expect_lt(abs(f$theta_ci_hi[["rho"]] - ex[["hi"]]), 0.01)
  }
  # The lower bound is where a row whose cell at rho = 0.4 was pruned carried
  # its quadratic tail across that cell: 0.333 on the default grid and 0.586
  # recentred, now 0.613 and 0.650.
  expect_lt(abs(fx$default$theta_ci_lo[["rho"]] - ex[["lo"]]), 0.06)
  expect_lt(abs(fx$recentred$theta_ci_lo[["rho"]] - ex[["lo"]]), 0.03)
})
