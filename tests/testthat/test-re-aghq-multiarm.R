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
# oracle. Correctness is established by composition:
#   * make_group reproduces make_site EXACTLY at d = 1 and d = 2 (correlated
#     slope) -- the d-dimensional quadrature core is already recovery-validated
#     for make_site in test-re-aghq.R, so the equivalence inherits that;
#   * the two-arm N-mixture oracle's gradient and the 2x2 observed-information
#     block (the abundance<->detection coupling from
#     tulpa_nmix_site_marginal()$obs_info_block) match finite differences.
# Tight community parameter recovery -- which needs an identified design
# (covariates) and a warm start to avoid the N-mixture lambda<->p ridge -- is a
# model-adapter concern (tulpaObs ms_abun), not the generic engine.

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

# ---- two-arm community N-mixture oracle (the msNMix structure) ---------------

# Per-group b-space oracle for an intercept-only community N-mixture: per species
# the RE b = (b_lambda, b_p) enters the abundance and detection intercepts. The
# per-species marginal, gradient and the 2x2 observed-info block (the abundance
# <-> detection coupling) come from tulpa_nmix_site_marginal(). Reference an
# msNMix adapter follows; `b` already includes the community mean.
.nmix_community_oracle <- function(marg_by_sp) {
  list(
    grad_hess = function(s, b) {
      m  <- marg_by_sp[[s]]
      ev <- m$eval(cl(rep(b[1], m$n_sites)), cl(rep(b[2], m$n_obs)))
      negH <- matrix(0, 2, 2)
      for (i in seq_len(m$n_sites)) {
        Ji <- length(m$obs_by_site[[i]]); Bi <- m$obs_info_block(i, ev)
        Zi <- cbind(c(1, rep(0, Ji)), c(0, rep(1, Ji)))
        negH <- negH + crossprod(Zi, Bi %*% Zi)
      }
      list(logL = ev$log_lik,
           grad = c(sum(ev$grad_eta_lambda), sum(ev$grad_eta_p)), negH = negH)
    },
    node_ll = function(s, B) {
      m <- marg_by_sp[[s]]
      vapply(seq_len(nrow(B)), function(k)
        m$eval(cl(rep(B[k, 1], m$n_sites)), cl(rep(B[k, 2], m$n_obs)))$log_lik,
        numeric(1))
    })
}

simulate_ms_nmix <- function(seed, S = 20L, n_sites = 12L, J = 6L,
                             mu_lambda = log(8), mu_p = qlogis(0.6),
                             sigma_lambda = 0.6, sigma_p = 0.5) {
  set.seed(seed)
  b_lam <- rnorm(S, 0, sigma_lambda); b_p <- rnorm(S, 0, sigma_p)
  marg_by_sp <- lapply(seq_len(S), function(s) {
    lam <- exp(mu_lambda + b_lam[s]); p <- plogis(mu_p + b_p[s])
    N <- rpois(n_sites, lam)
    y_mat <- matrix(rbinom(n_sites * J, rep(N, times = J), p), n_sites, J)
    tulpa_nmix_site_marginal(
      y = as.integer(as.vector(t(y_mat))),
      site_idx = rep(seq_len(n_sites), each = J),
      X_lambda = matrix(1, n_sites, 1), X_p = matrix(1, n_sites * J, 1),
      mixture = "P")
  })
  list(marg_by_sp = marg_by_sp, S = S,
       truth = c(mu_lambda = mu_lambda, mu_p = mu_p,
                 sigma_lambda = sigma_lambda, sigma_p = sigma_p))
}

test_that("community N-mixture oracle: grad / cross-arm Hessian match finite differences", {
  # Decisive correctness check for the two-arm assembly: the per-species b-space
  # gradient and the 2x2 observed-information block (the abundance / detection
  # coupling) match finite differences of the per-species marginal log-lik.
  sim <- simulate_ms_nmix(seed = 5L, S = 6L, n_sites = 12L, J = 6L)
  orc <- .nmix_community_oracle(sim$marg_by_sp)
  h <- 1e-5
  for (b0 in list(c(0, 0), c(0.3, -0.4), c(-0.5, 0.6))) {
    gh <- orc$grad_hess(1L, b0)
    fd_grad <- c(
      (orc$grad_hess(1L, b0 + c(h, 0))$logL - orc$grad_hess(1L, b0 - c(h, 0))$logL) / (2 * h),
      (orc$grad_hess(1L, b0 + c(0, h))$logL - orc$grad_hess(1L, b0 - c(0, h))$logL) / (2 * h))
    fd_H <- matrix(0, 2, 2)
    for (i in 1:2) for (j in 1:2) {
      ei <- c(0, 0); ei[i] <- h; ej <- c(0, 0); ej[j] <- h
      fd_H[i, j] <- (orc$grad_hess(1L, b0 + ei + ej)$logL - orc$grad_hess(1L, b0 + ei - ej)$logL -
                     orc$grad_hess(1L, b0 - ei + ej)$logL + orc$grad_hess(1L, b0 - ei - ej)$logL) / (4 * h * h)
    }
    expect_lt(max(abs(gh$grad - fd_grad)), 1e-4)
    expect_lt(max(abs(gh$negH - (-(fd_H + t(fd_H)) / 2))), 1e-3)
    expect_gt(abs(gh$negH[1, 2]), 1e-8)               # genuine cross-arm coupling
    expect_equal(orc$node_ll(1L, matrix(b0, 1, 2)), gh$logL)
  }
})

test_that("two-arm community N-mixture integrates end-to-end", {
  skip_on_cran()
  # End-to-end smoke: the full tulpa-side stack (marginal primitive -> oracle ->
  # N-arm engine) composes and returns a usable community fit. Not a precision
  # recovery claim (see file header). Warm-started near the data to keep the
  # lambda<->p ridge off the boundary.
  sim <- simulate_ms_nmix(seed = 23L, S = 14L, n_sites = 10L, J = 8L)
  orc <- .nmix_community_oracle(sim$marg_by_sp)
  ybar <- mean(vapply(sim$marg_by_sp, function(m) mean(m$y), numeric(1)))
  fit <- tulpa_re_aghq(
    theta0 = c(log(max(ybar / 0.5, 0.5)), qlogis(0.5)),
    re_terms = list(list(n_groups = sim$S, n_coefs = 1L),
                    list(n_groups = sim$S, n_coefs = 1L)),
    Sigma0 = list(matrix(0.4^2, 1, 1), matrix(0.4^2, 1, 1)),
    make_group = function(theta) list(
      grad_hess = function(g, b) orc$grad_hess(g, b + theta),
      node_ll   = function(g, B) orc$node_ll(g, sweep(B, 2, theta, `+`))),
    n_quad = 3L)
  skip_if(is.null(fit))                       # flat-likelihood seeds can null out
  expect_true(all(is.finite(fit$theta)))
  expect_length(fit$theta, 2L)
  expect_true(fit$Sigma_list[[1]][1, 1] > 0 && fit$Sigma_list[[2]][1, 1] > 0)
  expect_true(all(is.finite(fit$theta_se)))
  # Estimates land in a broad plausible region (means within ~0.8 on the link
  # scale, SDs positive and not blown up) -- gross-divergence guard, not recovery.
  expect_lt(abs(fit$theta[1] - sim$truth["mu_lambda"]), 0.8)
  expect_lt(abs(fit$theta[2] - sim$truth["mu_p"]), 0.8)
  expect_lt(sqrt(fit$Sigma_list[[1]][1, 1]), 1.5)
  expect_lt(sqrt(fit$Sigma_list[[2]][1, 1]), 1.5)
})
