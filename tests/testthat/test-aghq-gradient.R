# Analytic AGHQ gradient (cpp_aghq_objective_grad) vs central finite difference
# of the objective (cpp_aghq_objective). The analytic gradient is the
# Fisher-identity gradient of the true marginal; it agrees with the FD of the
# AGHQ objective as n_quad grows (the omitted node-placement terms are O(AGHQ
# truncation)), so these checks run at the AGHQ default n_quad = 9 where the two
# match tightly. Per-parameter-block tolerance; a sign flip fails loudly.

.sim_community_nmix <- function(seed, S, R, J, mu_lambda, mu_p, Sig_l, Sig_p) {
  set.seed(seed)
  p_lam <- length(mu_lambda); p_p <- length(mu_p)
  Ll <- t(chol(Sig_l)); Lp <- t(chol(Sig_p))
  X_lambda <- cbind(1, as.numeric(scale(rnorm(R))))[, seq_len(p_lam), drop = FALSE]
  y <- integer(0); site_idx <- integer(0); species_idx <- integer(0)
  xp_rows <- list()
  for (s in seq_len(S)) {
    cf_l <- mu_lambda + as.numeric(Ll %*% rnorm(p_lam))
    cf_p <- mu_p      + as.numeric(Lp %*% rnorm(p_p))
    for (i in seq_len(R)) {
      lam <- exp(sum(X_lambda[i, ] * cf_l))
      N   <- rpois(1, lam)
      for (j in seq_len(J)) {
        xp  <- c(1, rnorm(1))[seq_len(p_p)]
        pij <- stats::plogis(sum(xp * cf_p))
        y           <- c(y, stats::rbinom(1, N, pij))
        site_idx    <- c(site_idx, i)
        species_idx <- c(species_idx, s)
        xp_rows[[length(xp_rows) + 1L]] <- xp
      }
    }
  }
  list(y = y, site_idx = site_idx, species_idx = species_idx,
       X_lambda = X_lambda, X_p = do.call(rbind, xp_rows), R = R, S = S)
}

# Central FD of the exposed objective at every coordinate of par.
.aghq_fd_grad <- function(par, orc, nc, full, nq, lkj, h = 1e-4) {
  vapply(seq_along(par), function(jj) {
    pp <- par; pp[jj] <- pp[jj] + h
    pm <- par; pm[jj] <- pm[jj] - h
    (tulpa:::cpp_aghq_objective(pp, orc, nc, full, nq, lkj) -
     tulpa:::cpp_aghq_objective(pm, orc, nc, full, nq, lkj)) / (2 * h)
  }, numeric(1))
}

test_that("analytic AGHQ gradient matches central FD (Poisson community N-mixture)", {
  skip_on_cran()
  p_lam <- 2L; p_p <- 2L
  d <- .sim_community_nmix(
    seed = 404L, S = 5L, R = 8L, J = 3L,
    mu_lambda = c(1.0, 0.4), mu_p = c(0.2, -0.3),
    Sig_l = matrix(c(0.30, 0.10, 0.10, 0.20), 2, 2),
    Sig_p = matrix(c(0.25, -0.05, -0.05, 0.15), 2, 2))
  K_max <- max(d$y) + 25L

  orc <- tulpa:::cpp_nmix_community_oracle(
    d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p, d$R, d$S, K_max)
  nc   <- c(p_lam, p_p)
  full <- c(p_lam > 1L, p_p > 1L)

  # Pack Sigma with the engine's own log-Cholesky packer (no convention drift).
  layout <- tulpa:::.re_cov_block_layout(
    list(list(n_groups = d$S, n_coefs = p_lam, correlated = TRUE),
         list(n_groups = d$S, n_coefs = p_p,   correlated = TRUE)), NULL)
  re_par <- tulpa:::.re_cov_L_list_to_theta(
    lapply(list(matrix(c(0.30, 0.10, 0.10, 0.20), 2, 2),
                matrix(c(0.25, -0.05, -0.05, 0.15), 2, 2)), tulpa:::.re_chol_spd),
    layout)

  # Test point OFF the optimum so the gradient is non-trivial.
  par0 <- c(c(1.0, 0.4) + 0.10, c(0.2, -0.3) - 0.10, re_par)
  nq <- 9L; lkj <- 1

  r <- tulpa:::cpp_aghq_objective_grad(par0, orc, nc, full, nq, lkj)
  expect_true(isTRUE(r$ok))
  ana <- r$grad
  fd  <- .aghq_fd_grad(par0, orc, nc, full, nq, lkj)

  th_idx  <- seq_len(p_lam + p_p)               # theta block (mu, link scale)
  sig_idx <- setdiff(seq_along(par0), th_idx)   # log-Cholesky Sigma block

  # Fail loudly on a sign flip among entries large enough to have a definite sign.
  big <- abs(fd) > 1e-3
  expect_true(all(sign(ana[big]) == sign(fd[big])),
              info = paste0("gradient sign disagrees with FD:\n",
                            "  analytic = ", paste(signif(ana, 4), collapse = ", "),
                            "\n  FD       = ", paste(signif(fd, 4), collapse = ", ")))

  expect_lt(max(abs(ana[th_idx]  - fd[th_idx])),  1e-4)   # theta (mu) block
  expect_lt(max(abs(ana[sig_idx] - fd[sig_idx])), 1e-4)   # log-Cholesky Sigma block
})

# Same analytic-vs-FD gradient check for the native single-arm GLMM oracle
# (cpp_glmm_oracle_make), which also implements theta_score. d_re = 2 exercises
# a correlated (full 2x2) RE-covariance block; d_re = 1 the scalar block.
.glmm_grad_check <- function(family, seed, d_re) {
  set.seed(seed)
  ng <- 8L; nper <- 6L; n <- ng * nper
  group <- rep(seq_len(ng), each = nper)
  x <- as.numeric(scale(rnorm(n)))
  X <- cbind(1, x)                                   # fixed: intercept + slope
  Z <- if (d_re == 2L) cbind(1, x) else matrix(1, n, 1)
  beta <- c(0.3, 0.5)
  if (d_re == 2L) {
    Sig <- matrix(c(0.40, 0.10, 0.10, 0.30), 2, 2)
    B   <- matrix(rnorm(ng * 2L), ng, 2L) %*% t(chol(Sig))
  } else {
    Sig <- matrix(0.5, 1, 1)
    B   <- matrix(rnorm(ng) * sqrt(0.5), ng, 1L)
  }
  eta <- as.numeric(X %*% beta) + rowSums(Z * B[group, , drop = FALSE])
  n_trials <- rep(1, n)
  y <- switch(family,
    binomial = stats::rbinom(n, 1, stats::plogis(eta)),
    poisson  = stats::rpois(n, exp(pmin(eta, 5))),
    gaussian = eta + stats::rnorm(n, 0, 0.5))
  phi <- if (family == "gaussian") 0.25 else 1
  orc <- tulpa:::cpp_glmm_oracle_make(family, phi, as.numeric(y),
                                      as.numeric(n_trials), X, Z, group, ng)
  nc <- ncol(Z); full <- nc > 1L
  layout <- tulpa:::.re_cov_block_layout(
    list(list(n_groups = ng, n_coefs = nc, correlated = full)), NULL)
  re_par <- tulpa:::.re_cov_L_list_to_theta(list(tulpa:::.re_chol_spd(Sig)), layout)
  par0 <- c(beta + 0.10, re_par)
  ana <- tulpa:::cpp_aghq_objective_grad(par0, orc, nc, full, 9L, 1)$grad
  fd  <- .aghq_fd_grad(par0, orc, nc, full, 9L, 1)
  list(ana = ana, fd = fd)
}

test_that("analytic AGHQ gradient matches FD with NB dispersion (community, incl log_r)", {
  skip_on_cran()
  p_lam <- 2L; p_p <- 2L
  d <- .sim_community_nmix(
    seed = 505L, S = 5L, R = 8L, J = 3L,
    mu_lambda = c(1.0, 0.4), mu_p = c(0.2, -0.3),
    Sig_l = matrix(c(0.30, 0.10, 0.10, 0.20), 2, 2),
    Sig_p = matrix(c(0.25, -0.05, -0.05, 0.15), 2, 2))
  K_max <- max(d$y) + 25L

  # nb = TRUE: the oracle exposes n_theta = d + 1, carrying log_r as theta[d].
  orc <- tulpa:::cpp_nmix_community_oracle(
    d$y, d$site_idx, d$species_idx, d$X_lambda, d$X_p, d$R, d$S, K_max, nb = TRUE)
  nc <- c(p_lam, p_p); full <- c(TRUE, TRUE)
  layout <- tulpa:::.re_cov_block_layout(
    list(list(n_groups = d$S, n_coefs = p_lam, correlated = TRUE),
         list(n_groups = d$S, n_coefs = p_p,   correlated = TRUE)), NULL)
  re_par <- tulpa:::.re_cov_L_list_to_theta(
    lapply(list(matrix(c(0.30, 0.10, 0.10, 0.20), 2, 2),
                matrix(c(0.25, -0.05, -0.05, 0.15), 2, 2)), tulpa:::.re_chol_spd),
    layout)
  # theta = [mu_lambda(2), mu_p(2), log_r(1)], then the log-Cholesky Sigma coords.
  par0 <- c(c(1.0, 0.4) + 0.10, c(0.2, -0.3) - 0.10, log(8), re_par)
  nq <- 9L; lkj <- 1

  ana <- tulpa:::cpp_aghq_objective_grad(par0, orc, nc, full, nq, lkj)$grad
  fd  <- .aghq_fd_grad(par0, orc, nc, full, nq, lkj)

  th_idx  <- seq_len(p_lam + p_p + 1L)            # mu + log_r
  sig_idx <- setdiff(seq_along(par0), th_idx)

  big <- abs(fd) > 1e-3
  expect_true(all(sign(ana[big]) == sign(fd[big])),
              info = paste0("NB gradient sign disagrees with FD (incl log_r):\n  ana=",
                            paste(signif(ana, 4), collapse = ", "),
                            "\n  fd =", paste(signif(fd, 4), collapse = ", ")))
  expect_lt(max(abs(ana[th_idx]  - fd[th_idx])),  5e-4)   # mu + log_r block
  expect_lt(max(abs(ana[sig_idx] - fd[sig_idx])), 5e-4)   # log-Cholesky Sigma block
})

for (fam_dre in list(c("binomial", "2"), c("poisson", "1"), c("gaussian", "1"))) {
  local({
    family <- fam_dre[1]; d_re <- as.integer(fam_dre[2])
    test_that(sprintf("analytic AGHQ gradient matches FD (GLMM oracle, %s d=%d)",
                      family, d_re), {
      skip_on_cran()
      r <- .glmm_grad_check(family, seed = 11L, d_re = d_re)
      big <- abs(r$fd) > 1e-3
      expect_true(all(sign(r$ana[big]) == sign(r$fd[big])),
                  info = paste0(family, ": sign disagrees\n  ana=",
                                paste(signif(r$ana, 4), collapse = ", "),
                                "\n  fd =", paste(signif(r$fd, 4), collapse = ", ")))
      # 5e-4 is the n_quad = 9 AGHQ-truncation floor (the analytic gradient is the
      # true-marginal gradient; its gap to the objective's FD is O(quadrature
      # truncation) and varies by integrand -- the poisson case sits at ~1e-4
      # here). Still ~4 orders below the gradient scale, so a sign / scale bug
      # (O(0.1)+) is caught with wide margin.
      expect_lt(max(abs(r$ana - r$fd)), 5e-4)
    })
  })
}
