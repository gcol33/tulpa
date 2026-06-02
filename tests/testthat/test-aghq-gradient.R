# Analytic AGHQ gradient (cpp_aghq_objective_grad) vs central finite difference
# of the objective (cpp_aghq_objective). The analytic gradient is the
# Fisher-identity gradient of the true marginal; it agrees with the FD of the
# AGHQ objective as n_quad grows (the omitted node-placement terms are O(AGHQ
# truncation)), so these checks run at the AGHQ default n_quad = 9 where the two
# match tightly. Per-parameter-block tolerance; a sign flip fails loudly.
#
# This file covers the engine's native single-arm GLMM oracle. The
# community / multispecies N-mixture oracle is consumer code in tulpaObs and the
# corresponding gradient check lives there.

# Central FD of the exposed objective at every coordinate of par.
.aghq_fd_grad <- function(par, orc, nc, full, nq, lkj, h = 1e-4) {
  vapply(seq_along(par), function(jj) {
    pp <- par; pp[jj] <- pp[jj] + h
    pm <- par; pm[jj] <- pm[jj] - h
    (tulpa:::cpp_aghq_objective(pp, orc, nc, full, nq, lkj) -
     tulpa:::cpp_aghq_objective(pm, orc, nc, full, nq, lkj)) / (2 * h)
  }, numeric(1))
}

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

for (fam_dre in list(c("binomial", "2"), c("poisson", "1"), c("gaussian", "1"))) {
  local({
    family <- fam_dre[1]; d_re <- as.integer(fam_dre[2])
    test_that(sprintf("analytic AGHQ gradient matches FD (GLMM oracle, %s d=%d)",
                      family, d_re), {
      skip_on_cran()
      skip_if_fast()
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
