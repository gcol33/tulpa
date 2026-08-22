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

# --------------------------------------------------------------------------- #
# (gcol33/tulpa#453) A separated class has a finite log-likelihood             #
#                                                                              #
# The softmax is overflow-safe on the way in and was not underflow-safe on the #
# way back out: p_j = exp(eta_j - m) / denom flushes to exactly 0 once eta_j   #
# sits about 745 below the shift, and log(0) is -Inf where the algebra gives a #
# finite number near eta_c - m - log(denom). Separation in a categorical arm   #
# is what drives one class's eta far below the others, so this is reachable    #
# from a fit: one -Inf observation makes the whole data log-likelihood -Inf,   #
# the Newton line search never accepts a non-finite trial, and the solve       #
# backtracks off a point the model is perfectly well defined at.               #
# --------------------------------------------------------------------------- #

test_that("a class the softmax underflows still reports its finite value", {
  # eta_c - logsumexp(0, eta), computed without forming a probability.
  ll_stable <- function(eta, cls) {
    m <- max(0, eta)
    (if (cls <= length(eta)) eta[cls] else 0) - m - log(sum(exp(c(0, eta) - m)))
  }
  for (gap in c(600, 745, 800, 5000)) {
    eta <- c(-gap, 0.4, -0.7)
    for (cls in seq_len(length(eta) + 1L)) {
      out <- tulpa:::cpp_multinomial_logit_terms(eta, cls)
      expect_true(is.finite(out$ll),
                  label = paste("ll finite at gap", gap, "class", cls))
      expect_equal(out$ll, ll_stable(eta, cls), tolerance = 1e-12)
    }
  }
  # The separated class itself: log p is about -gap, not -Inf.
  expect_equal(tulpa:::cpp_multinomial_logit_terms(c(-800, 0.4, -0.7), 1)$ll,
               ll_stable(c(-800, 0.4, -0.7), 1), tolerance = 1e-12)
})

test_that("cls outside 1..K is refused rather than read as the baseline", {
  # cls <= 0 indexed p[cls - 1], before the start of the buffer; cls > K
  # silently returned log(p_K), scoring an observation of the baseline that
  # was never made.
  for (bad in c(-1L, 0L, 5L, 100L)) {
    expect_error(tulpa:::cpp_multinomial_logit_terms(c(0.3, -0.5, 1.1), bad),
                 "1[.][.]4")
  }
  expect_error(tulpa:::cpp_multinomial_logit_terms(numeric(0), 1L),
               "at least one entry")
})
