# A sampler fit of a temporal walk forecasts from each draw's own path and
# hyperparameters; the nested-Laplace fit with the future times in the field
# reads its posterior at them. On a gaussian response with known noise both
# are checked against the exact posterior: the fixed effects and the field
# integrated in closed form, the precision and correlation on a dense grid,
# under the prior both routes carry (a PC prior on the field SD with
# P(sigma > 3) = 0.01, a uniform rho).

.ar1_exact_eta <- function(tr, Tn, phi, sdb = 2.5, U = 3, alpha = 0.01) {
  n <- nrow(tr)
  A <- cbind(1, tr$x, outer(tr$t, seq_len(Tn), "==") * 1)
  AtA <- crossprod(A) / phi; b <- crossprod(A, tr$y) / phi
  yy <- sum(tr$y^2) / phi
  lam <- -log(alpha) / U
  C <- cbind(1, 0, diag(Tn))
  lw <- numeric(0); M <- list(); V <- list()
  for (lt in seq(-4, 10, by = 0.1)) for (l in seq(-7, 7, by = 0.1)) {
    u <- plogis(l); rho <- 2 * u - 1
    P <- diag(c(1, rep(1 + rho^2, Tn - 2), 1))
    P[cbind(1:(Tn - 1), 2:Tn)] <- P[cbind(2:Tn, 1:(Tn - 1))] <- -rho
    Qp <- matrix(0, Tn + 2, Tn + 2); Qp[1:2, 1:2] <- diag(1 / sdb^2, 2)
    Qp[3:(Tn + 2), 3:(Tn + 2)] <- exp(lt) * P
    R <- chol(Qp + AtA)
    m <- backsolve(R, backsolve(R, b, transpose = TRUE))
    s <- exp(-lt / 2)
    lw <- c(lw, 0.5 * (-4 * log(sdb) + Tn * lt + log(1 - rho^2)) -
      sum(log(diag(R))) - 0.5 * (yy - sum(b * m)) +
      log(lam) - lam * s + log(0.5 * s) + log(u) + log1p(-u))
    M[[length(M) + 1L]] <- as.numeric(C %*% m)
    V[[length(V) + 1L]] <- colSums(backsolve(R, t(C), transpose = TRUE)^2)
  }
  w <- exp(lw - max(lw)); w <- w / sum(w)
  Mm <- do.call(rbind, M); Vm <- do.call(rbind, V)
  mu <- colSums(w * Mm)
  list(mean = mu, sd = sqrt(colSums(w * (Vm + Mm^2)) - mu^2))
}

test_that("an AR1 forecast is the exact posterior on the sampler and the nested route", {
  skip_if_not_slow()
  set.seed(303)
  Tn <- 30L; Tfit <- 24L; phi <- 0.25
  f <- as.numeric(stats::arima.sim(list(ar = 0.8), Tn, sd = 0.3))
  t <- rep(seq_len(Tn), each = 8L); x <- rnorm(length(t))
  d <- data.frame(t = t, x = x, y = 0.5 + 0.4 * x + f[t] + rnorm(length(t), 0, sqrt(phi)))
  tr <- d[d$t <= Tfit, ]
  ex <- .ar1_exact_eta(tr, Tn, phi)
  fut <- (Tfit + 1L):Tn
  hmc <- suppressWarnings(tulpa(y ~ x, data = tr, family = "gaussian", phi = phi,
    temporal = temporal_ar1("t"), mode = "hmc",
    control = list(n_chains = 4L, n_iter = 3000L, n_warmup = 1000L, seed = 1L,
                   verbose = FALSE)))
  p <- predict(hmc, newdata = data.frame(t = fut, x = 0), se.fit = TRUE)
  # Measured over ten such data sets: forecast SD 0.97-1.04 of the exact one,
  # means within 0.05 SD. A Gamma(1, 0.01) on the precision in place of the
  # PC prior gives an exact forecast SD 0.92 of this one, so the check tells
  # the two priors apart.
  expect_lt(max(abs(p$se.fit / ex$sd[fut] - 1)), 0.08)
  expect_lt(max(abs(p$fit - ex$mean[fut]) / ex$sd[fut]), 0.2)
  dna <- rbind(tr, data.frame(t = fut, x = 0, y = NA))
  nl <- suppressWarnings(suppressMessages(tulpa(y ~ x, data = dna,
    family = "gaussian", phi = phi, temporal = temporal_ar1("t"),
    mode = "nested_laplace")))
  pr <- posterior_linpred(nl, summary = TRUE)[nrow(tr) + seq_along(fut), ]
  expect_lt(max(abs(pr$sd / ex$sd[fut] - 1)), 0.03)
  expect_lt(max(abs(pr$estimate - ex$mean[fut]) / ex$sd[fut]), 0.05)
})
