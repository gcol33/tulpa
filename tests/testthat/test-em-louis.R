# Standard errors of tulpa_em_laplace(): the complete-data SE on every block
# fit, and Louis' observed-information SE, checked against the exact observed
# information of a zero-inflated Poisson (the Hessian of its observed-data log
# posterior at the optimum).

.zip_em_fixture <- function(n = 300L, seed = 20261102L) {
  set.seed(seed)
  x <- rnorm(n); v <- rnorm(n)
  z <- rbinom(n, 1, plogis(0.8 - 1.0 * v))
  y <- z * rpois(n, exp(1.0 + 0.5 * x))
  X <- cbind(1, x); Z <- cbind(1, v)
  e_step <- function(fits, ...) {
    if (!length(fits)) return(list(weights = ifelse(y > 0, 1, 0.5)))
    psi <- plogis(drop(Z %*% fits$occ$mode))
    p0  <- psi * exp(-exp(drop(X %*% fits$abund$mode)))
    list(weights = ifelse(y > 0, 1, p0 / (p0 + 1 - psi)))
  }
  m_step_encode <- function(weights, ...) list(
    occ   = list(y = rep(c(1, 0), each = n), X = rbind(Z, Z),
                 weights = c(weights, 1 - weights), family = "binomial"),
    abund = list(y = y, X = X, weights = weights, family = "poisson"))
  # Observed-data negative log posterior, with the engine's default
  # fixed-effect prior N(0, s^2).
  s_b <- tulpa:::cpp_test_default_sigma_beta()
  nlp <- function(th) {
    psi <- plogis(drop(Z %*% th[1:2])); lam <- exp(drop(X %*% th[3:4]))
    ll <- ifelse(y == 0, log(1 - psi + psi * exp(-lam)),
                 log(psi) + stats::dpois(y, lam, log = TRUE))
    -sum(ll) + sum(th^2) / (2 * s_b^2)
  }
  list(e_step = e_step, m_step_encode = m_step_encode, nlp = nlp, n = n)
}

test_that("every block fit carries beta and its complete-data se", {
  skip_if_fast()
  d <- .zip_em_fixture(n = 120L)
  set.seed(1)
  res <- tulpa_em_laplace(d$e_step, d$m_step_encode, verbose = FALSE,
                          n_louis = 0L)
  for (nm in c("occ", "abund")) {
    f <- res$fits[[nm]]
    expect_equal(f$beta, f$mode[1:2])
    expect_equal(f$se, sqrt(diag(solve(f$H_beta))), tolerance = 1e-8)
  }
  expect_null(res$vcov)
})

test_that("Louis standard errors match the exact observed information", {
  skip_on_cran()
  d <- .zip_em_fixture()
  op <- stats::optim(c(0, 0, 0, 0), d$nlp, method = "BFGS",
                     control = list(reltol = 1e-14, maxit = 2000))
  se_exact <- sqrt(diag(solve(stats::optimHess(op$par, d$nlp))))
  set.seed(7)
  res <- tulpa_em_laplace(d$e_step, d$m_step_encode, verbose = FALSE,
                          tol = 1e-9, max_iter = 500L)
  est <- c(res$fits$occ$beta, res$fits$abund$beta)
  expect_lt(max(abs(est - op$par) / se_exact), 1e-3)

  se_louis <- c(res$se$occ, res$se$abund)
  se_complete <- c(res$fits$occ$se, res$fits$abund$se)
  # The complete-data SE leaves out the missing information...
  expect_true(all(se_complete < se_exact))
  # ...which Louis' identity restores, to Monte Carlo error on n_louis = 1000.
  expect_lt(max(abs(se_louis / se_exact - 1)), 0.03)
  expect_equal(dim(res$vcov), c(4L, 4L))
  expect_identical(rownames(res$vcov)[1], "occ:beta1")
  expect_true(all(res$louis$missing_fraction > 0 &
                    res$louis$missing_fraction < 1))
})

test_that("Louis standard errors decline on a block with latent structure", {
  skip_if_fast()
  set.seed(3)
  n <- 60L; g <- rep(1:6, each = 10); x <- rnorm(n)
  y <- rpois(n, exp(0.5 + 0.3 * x + rnorm(6, 0, 0.5)[g]))
  e_step <- function(fits, ...) list(weights = rep(1, n))
  m_step_encode <- function(weights, ...) list(
    a = list(y = y, X = cbind(1, x), family = "poisson", weights = weights,
             re_list = list(list(idx = g, n_groups = 6L, sigma = 0.5))))
  res <- tulpa_em_laplace(e_step, m_step_encode, verbose = FALSE,
                          max_iter = 3L)
  expect_null(res$se)
  expect_match(res$louis$unavailable, "latent structure")
  expect_true(all(is.finite(res$fits$a$se)))
})
