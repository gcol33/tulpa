# Multi-arm / general (make_group) path of tulpa_re_aghq().
#
# tulpa_re_aghq integrates a grouped random effect against an abstract per-group
# conditional likelihood. The familiar `make_site` form is the single-arm,
# per-row-separable case; `make_group` lets the caller supply the b-space oracle
# directly, so non-separable units and random effects on several coupled arms
# (e.g. a community N-mixture: species priors on BOTH the abundance and the
# detection coefficients) integrate with no engine change.
#
# The integration core is arm-agnostic -- it only ever sees b in R^d and the
# oracle. Correctness is established by composition: make_group reproduces
# make_site EXACTLY at d = 1 and d = 2 (correlated slope) -- the d-dimensional
# quadrature core is already recovery-validated for make_site in test-re-aghq.R,
# so the equivalence inherits that. The two-arm N-mixture oracle (species priors
# on both the abundance and detection coefficients) and its community parameter
# recovery are a model-adapter concern (tulpaObs ms_abun), tested there, not in
# the generic engine.

cl <- function(e) pmin(pmax(e, -30), 30)
l1pe <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))   # log(1+e^x)

# ---- make_group == make_site on a shared single-arm problem ------------------

test_that("make_group reproduces make_site on a single-arm Poisson random intercept", {
  set.seed(1)
  ng <- 40L; ni <- 8L
  grp <- rep(seq_len(ng), each = ni); n <- ng * ni
  x   <- rnorm(n)
  beta_true <- c(0.4, 0.6)
  eta <- beta_true[1] + beta_true[2] * x + rnorm(ng, 0, 0.7)[grp]
  y   <- rpois(n, exp(eta)); Xfe <- cbind(1, x)

  re_terms     <- list(list(idx = grp, n_groups = ng, n_coefs = 1L))
  re_terms_grp <- list(list(n_groups = ng, n_coefs = 1L))   # covariance-only
  Sigma0 <- list(matrix(0.5^2, 1, 1))

  make_site <- function(theta) {
    eta_fe <- as.numeric(Xfe %*% theta)
    list(eta_re = eta_fe,
         deriv = function(rows, e) {
           lam <- exp(cl(e))
           list(logL = dpois(y[rows], lam, log = TRUE), d1 = y[rows] - lam, d2 = -lam)
         },
         lmat = function(rows, ETA) {
           y[rows] * cl(ETA) - exp(cl(ETA)) - lgamma(y[rows] + 1)
         })
  }
  rows_by_g <- split(seq_len(n), grp)
  make_group <- function(theta) {
    eta_fe <- as.numeric(Xfe %*% theta)
    list(grad_hess = function(g, b) {
           rows <- rows_by_g[[g]]; lam <- exp(cl(eta_fe[rows] + b[1]))
           list(logL = sum(dpois(y[rows], lam, log = TRUE)),
                grad = sum(y[rows] - lam), negH = matrix(sum(lam), 1, 1))
         },
         node_ll = function(g, B) {
           rows <- rows_by_g[[g]]
           vapply(seq_len(nrow(B)), function(k) {
             e <- cl(eta_fe[rows] + B[k, 1])
             sum(y[rows] * e - exp(e) - lgamma(y[rows] + 1))
           }, numeric(1))
         })
  }

  fit_s <- tulpa_re_aghq(beta_true, re_terms, Sigma0, make_site = make_site,
                         n_obs = n, n_quad = 5L)
  fit_g <- tulpa_re_aghq(beta_true, re_terms_grp, Sigma0, make_group = make_group,
                         n_quad = 5L)
  expect_false(is.null(fit_s)); expect_false(is.null(fit_g))
  expect_equal(fit_g$theta, fit_s$theta, tolerance = 1e-6)
  expect_equal(fit_g$Sigma_list[[1]], fit_s$Sigma_list[[1]], tolerance = 1e-6)
  expect_equal(fit_g$theta_se, fit_s$theta_se, tolerance = 1e-5)
  expect_equal(fit_g$blup[[1]], fit_s$blup[[1]], tolerance = 1e-5)
})

test_that("make_group reproduces make_site on a correlated random slope (d = 2)", {
  # Binomial GLMM (bounded, numerically stable at d = 2) with a correlated
  # (1 + x | g) block -- the same family/structure test-re-aghq.R validates for
  # make_site. The equivalence holds for any likelihood; binomial avoids the
  # ill-conditioning a Poisson random slope can throw at the marginal Hessian.
  set.seed(11)
  ng <- 50L; ni <- 20L
  grp <- rep(seq_len(ng), each = ni); n <- ng * ni
  x   <- rnorm(n); X <- cbind(1, x); nt <- rep(4L, n)
  beta_true <- c(0.2, -0.4)
  Sig_true <- matrix(c(0.6, 0.3, 0.3, 0.4), 2, 2)
  bg <- matrix(rnorm(ng * 2), ng, 2) %*% chol(Sig_true)
  eta <- beta_true[1] + beta_true[2] * x + bg[grp, 1] + bg[grp, 2] * x
  y   <- rbinom(n, nt, plogis(eta))

  re_terms     <- list(list(idx = grp, n_groups = ng, n_coefs = 2L,
                            Z = X, correlated = TRUE))
  re_terms_grp <- list(list(n_groups = ng, n_coefs = 2L, correlated = TRUE))
  Sigma0 <- list(diag(0.25, 2))

  make_site <- function(theta) {
    eta_fe <- as.numeric(X %*% theta)
    list(eta_re = eta_fe,
         deriv = function(rows, e) {
           p <- plogis(e)
           list(logL = y[rows] * e - nt[rows] * l1pe(e),
                d1 = y[rows] - nt[rows] * p, d2 = -nt[rows] * p * (1 - p))
         },
         lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
  }
  rows_by_g <- split(seq_len(n), grp)
  make_group <- function(theta) {
    eta_fe <- as.numeric(X %*% theta)
    list(grad_hess = function(g, b) {
           rows <- rows_by_g[[g]]; Zg <- X[rows, , drop = FALSE]
           e <- cl(eta_fe[rows] + as.numeric(Zg %*% b)); p <- plogis(e)
           list(logL = sum(y[rows] * e - nt[rows] * l1pe(e)),
                grad = as.numeric(crossprod(Zg, y[rows] - nt[rows] * p)),
                negH = crossprod(Zg, (nt[rows] * p * (1 - p)) * Zg))
         },
         node_ll = function(g, B) {
           rows <- rows_by_g[[g]]; Zg <- X[rows, , drop = FALSE]
           ETA <- cl(matrix(eta_fe[rows], length(rows), nrow(B)) + Zg %*% t(B))
           colSums(y[rows] * ETA - nt[rows] * l1pe(ETA))
         })
  }

  fit_s <- tulpa_re_aghq(beta_true, re_terms, Sigma0, make_site = make_site,
                         n_obs = n, n_quad = 7L)
  fit_g <- tulpa_re_aghq(beta_true, re_terms_grp, Sigma0, make_group = make_group,
                         n_quad = 7L)
  expect_false(is.null(fit_s)); expect_false(is.null(fit_g))
  expect_equal(fit_g$theta, fit_s$theta, tolerance = 1e-5)
  expect_equal(fit_g$Sigma_list[[1]], fit_s$Sigma_list[[1]], tolerance = 1e-5)
  expect_equal(fit_g$blup[[1]], fit_s$blup[[1]], tolerance = 1e-4)
})

test_that("make_site / make_group are mutually exclusive", {
  re_terms <- list(list(idx = rep(1:2, each = 3), n_groups = 2L, n_coefs = 1L))
  Sigma0 <- list(matrix(1, 1, 1))
  expect_error(
    tulpa_re_aghq(0, re_terms, Sigma0, n_obs = 6L), "exactly one")
  expect_error(
    tulpa_re_aghq(0, re_terms, Sigma0, make_site = function(th) NULL,
                  make_group = function(th) NULL, n_obs = 6L), "exactly one")
})
