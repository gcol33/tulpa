# Finite-difference check of the baseline-category multinomial logit kernel
# (gcol33/tulpaObs#106). eta is length K-1 (non-baseline class predictors), class
# c in 1..K (c == K is the baseline). The kernel returns (ll, grad[K-1],
# neg_hess[K-1 x K-1]); each is checked against central differences of an
# independent R softmax evaluation.

.ml_ll_R <- function(eta, cls) {
  K <- length(eta) + 1L
  denom <- 1 + sum(exp(eta))
  p <- c(exp(eta) / denom, 1 / denom)
  log(p[cls])
}

test_that("grad / neg_hess match central differences across K and class", {
  h <- 1e-5
  for (eta in list(c(0.4), c(-1.2, 0.8), c(0.3, -0.5, 1.1), c(-2, 0, 1.5, 0.2))) {
    Km1 <- length(eta); K <- Km1 + 1L
    for (cls in seq_len(K)) {
      out <- tulpa:::cpp_multinomial_logit_terms(eta, cls)
      expect_equal(out$ll, .ml_ll_R(eta, cls), tolerance = 1e-10)

      g_fd <- vapply(seq_len(Km1), function(j) {
        ep <- eta; em <- eta; ep[j] <- ep[j] + h; em[j] <- em[j] - h
        (.ml_ll_R(ep, cls) - .ml_ll_R(em, cls)) / (2 * h)
      }, numeric(1))
      expect_equal(as.numeric(out$grad), g_fd, tolerance = 1e-5)

      H_fd <- matrix(0, Km1, Km1)
      for (j in seq_len(Km1)) for (l in seq_len(Km1)) {
        epp <- eta; epp[j] <- epp[j] + h; epp[l] <- epp[l] + h
        epm <- eta; epm[j] <- epm[j] + h; epm[l] <- epm[l] - h
        emp <- eta; emp[j] <- emp[j] - h; emp[l] <- emp[l] + h
        emm <- eta; emm[j] <- emm[j] - h; emm[l] <- emm[l] - h
        H_fd[j, l] <- (.ml_ll_R(epp, cls) - .ml_ll_R(epm, cls) -
                       .ml_ll_R(emp, cls) + .ml_ll_R(emm, cls)) / (4 * h^2)
      }
      expect_equal(as.matrix(out$neg_hess), -H_fd, tolerance = 1e-3)
    }
  }
})

test_that("negative Hessian is the multinomial covariance (PSD, data-independent)", {
  for (eta in list(c(-1.2, 0.8), c(0.3, -0.5, 1.1))) {
    Km1 <- length(eta)
    denom <- 1 + sum(exp(eta)); p <- exp(eta) / denom
    expect_equal(as.matrix(tulpa:::cpp_multinomial_logit_terms(eta, 1)$neg_hess),
                 diag(p, Km1) - outer(p, p), tolerance = 1e-12)
    # same Hessian regardless of observed class (multinomial logit info is data-free)
    for (cls in 2:(Km1 + 1L))
      expect_equal(as.matrix(tulpa:::cpp_multinomial_logit_terms(eta, cls)$neg_hess),
                   diag(p, Km1) - outer(p, p), tolerance = 1e-12)
    expect_gte(min(eigen(diag(p, Km1) - outer(p, p), only.values = TRUE)$values), -1e-12)
  }
})

test_that("class probabilities are a valid simplex and overflow-safe", {
  big <- c(800, -800, 200)          # would overflow a naive exp()
  out <- tulpa:::cpp_multinomial_logit_terms(big, 1)
  expect_true(is.finite(out$ll))
  # ll of the dominant class approx 0 (prob approx 1)
  expect_lt(abs(out$ll), 1e-6)
})
