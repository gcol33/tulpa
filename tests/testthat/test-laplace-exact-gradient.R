# The exact outer gradient (R/laplace_gradient.R) and the curvature-derivative
# ladder it stands on (src/laplace_family_curvature.h).
#
# The load-bearing check is agreement with a central difference of tulpa's OWN
# Laplace log-marginal: the derivation can be right on paper and still be wired
# to the wrong latent ordering, the wrong weight convention, or the wrong sign.

.exg_sim <- function(seed = 5L, G = 10L, per = 7L, nc = 1L, fam = "poisson") {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  X <- cbind(1, rnorm(n))
  Zc <- if (nc == 1L) NULL else cbind(1, rnorm(n))
  Sig <- if (nc == 1L) matrix(0.5, 1, 1) else
    matrix(c(0.5, 0.15, 0.15, 0.35), 2, 2)
  b <- t(chol(Sig)) %*% matrix(rnorm(nc * G), nc, G)
  zrow <- if (nc == 1L) matrix(1, n, 1) else Zc
  eta <- as.numeric(X %*% c(0.3, 0.5)) +
    rowSums(zrow * t(b)[grp, , drop = FALSE])
  y <- switch(fam,
              poisson  = rpois(n, exp(eta)),
              binomial = rbinom(n, 1L, plogis(eta)),
              gaussian = rnorm(n, eta, 0.7))
  list(y = y, X = X, grp = grp, G = G, n = n, nc = nc, Zc = Zc, fam = fam,
       n_trials = rep(1L, n))
}

.exg_theta_to_L <- function(theta, nc, full) {
  L <- matrix(0, nc, nc)
  if (!full) { diag(L) <- exp(theta); return(L) }
  i <- 0
  for (cc in seq_len(nc)) for (rr in cc:nc) {
    i <- i + 1
    L[rr, cc] <- if (rr == cc) exp(theta[i]) else theta[i]
  }
  L
}

.exg_layout <- function(d, full) {
  list(list(nc = d$nc, full = full,
            k = if (full) d$nc * (d$nc + 1) / 2 else d$nc,
            n_groups = d$G, idx = d$grp, Z = d$Zc))
}

# Analytic gradient vs a central difference of tulpa_laplace()$log_marginal.
.exg_compare <- function(d, theta, full = FALSE, phi = 1.0) {
  layout <- .exg_layout(d, full)
  fit_at <- function(th, joint) {
    L <- .exg_theta_to_L(th, d$nc, full)
    tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                  re_list = .re_cov_build_re_list(list(L), layout),
                  family = d$fam, phi = phi, return_hessian = TRUE,
                  return_joint_hessian = joint,
                  max_iter = 300L, tol = 1e-12)
  }
  L0 <- .exg_theta_to_L(theta, d$nc, full)
  ga <- .laplace_exact_re_grad(
    fit = fit_at(theta, TRUE), y = d$y, X = d$X, n_trials = d$n_trials,
    offset = NULL, weights = NULL,
    re_list = .re_cov_build_re_list(list(L0), layout),
    layout = layout, L_list = list(L0), family = d$fam, phi = phi)
  if (is.null(ga)) return(NULL)
  h <- 1e-5
  gf <- vapply(seq_along(theta), function(j) {
    tp <- theta; tp[j] <- tp[j] + h
    tm <- theta; tm[j] <- tm[j] - h
    (fit_at(tp, FALSE)$log_marginal - fit_at(tm, FALSE)$log_marginal) / (2 * h)
  }, numeric(1))
  list(analytic = ga, fd = gf)
}


test_that("the exact gradient matches a difference of tulpa's own marginal", {
  for (cs in list(list(fam = "poisson",  th = log(0.7)),
                  list(fam = "binomial", th = log(0.7)),
                  list(fam = "poisson",  th = log(1.6)))) {
    d <- .exg_sim(fam = cs$fam, G = if (cs$fam == "binomial") 14L else 10L,
                  per = if (cs$fam == "binomial") 10L else 7L)
    r <- .exg_compare(d, cs$th)
    skip_if(is.null(r), "gradient unavailable")
    expect_equal(r$analytic, r$fd, tolerance = 1e-6,
                 info = paste(cs$fam, "at theta =", signif(cs$th, 4)))
  }
})

test_that("the exact gradient matches for a correlated block", {
  d <- .exg_sim(seed = 5L, G = 12L, per = 8L, nc = 2L)
  r <- .exg_compare(d, c(log(0.7), 0.2, log(0.6)), full = TRUE)
  skip_if(is.null(r), "gradient unavailable")
  expect_equal(r$analytic, r$fd, tolerance = 1e-6)
  expect_length(r$analytic, 3L)
})

test_that("a gaussian response is the control: dw/deta is zero throughout", {
  # The curvature-movement term vanishes identically here, so this case is
  # exact with or without it -- which is why it cannot be the only check.
  d <- .exg_sim(fam = "gaussian")
  r <- .exg_compare(d, log(0.7), phi = 0.49)
  skip_if(is.null(r), "gradient unavailable")
  expect_equal(r$analytic, r$fd, tolerance = 1e-8)
  expect_true(all(cpp_family_curvature_deta_vec(
    d$y, d$n_trials, rnorm(d$n), "gaussian", 0.7) == 0))
})

test_that("dw/deta differentiates the weight the Newton system uses", {
  cases <- list(
    list(family = "poisson",                  y = 4,   n = 1L,  phi = 1.0),
    list(family = "binomial",                 y = 3,   n = 10L, phi = 1.0),
    list(family = "neg_binomial_2",           y = 6,   n = 1L,  phi = 2.5),
    list(family = "neg_binomial_1",           y = 6,   n = 1L,  phi = 1.7),
    list(family = "truncated_poisson",        y = 3,   n = 1L,  phi = 1.0),
    list(family = "truncated_neg_binomial_2", y = 3,   n = 1L,  phi = 2.0),
    list(family = "beta_binomial",            y = 3,   n = 10L, phi = 4.0),
    list(family = "gamma",                    y = 2.0, n = 1L,  phi = 3.0),
    list(family = "inverse_gaussian",         y = 1.4, n = 1L,  phi = 1.1),
    list(family = "beta",                     y = 0.6, n = 1L,  phi = 6.0),
    list(family = "binomial_probit",          y = 3,   n = 10L, phi = 1.0),
    list(family = "binomial_cloglog",         y = 3,   n = 10L, phi = 1.0),
    list(family = "gaussian_log",             y = 1.2, n = 1L,  phi = 1.5)
  )
  h <- 1e-5
  for (cs in cases) {
    for (eta in c(-0.6, 0.0, 0.9)) {
      w <- function(e) unname(
        cpp_family_terms(cs$y, cs$n, e, cs$family, cs$phi)[["neg_hess"]])
      got <- unname(cpp_family_curvature_deta(
        cs$y, cs$n, eta, cs$family, cs$phi)[["dw_deta"]])
      num <- (w(eta + h) - w(eta - h)) / (2 * h)
      # Scaled by the larger magnitude with a floor: several families have a
      # weight that is genuinely constant in eta, where the answer is 0 and a
      # ratio against finite-difference noise is meaningless.
      scale <- max(abs(num), abs(got), 1e-6)
      expect_lt(abs(got - num) / scale, 1e-4)
    }
  }
})

test_that("families with an exact curvature derivative are gated honestly", {
  for (f in c("poisson", "binomial", "neg_binomial_2", "gamma",
              "binomial_probit", "gaussian_log")) {
    expect_true(cpp_family_has_curvature_derivative(f), info = f)
  }
  # The gate is what stops the exact path being offered for a family whose
  # weight has no closed-form eta-derivative here.
  expect_false(cpp_family_has_curvature_derivative("not_a_family"))
})

test_that("the vectorized curvature derivative matches the scalar probe", {
  set.seed(2)
  eta <- rnorm(25)
  y   <- rpois(25, 2)
  vec <- cpp_family_curvature_deta_vec(y, 1L, eta, "poisson", 1.0)
  sc  <- vapply(seq_along(eta), function(i)
    unname(cpp_family_curvature_deta(y[i], 1L, eta[i], "poisson", 1.0)[["dw_deta"]]),
    numeric(1))
  expect_equal(vec, sc)
  expect_error(cpp_family_curvature_deta_vec(y[1:3], 1L, eta, "poisson", 1.0),
               "differ in length")
})

test_that("the chain rule export agrees with a direct dSigma trace", {
  # cpp_recov_block_grad must equal 0.5 tr(dSigma_j . core) with Smat = 0.5 core.
  set.seed(4)
  nc <- 2L
  L <- matrix(c(0.8, 0.25, 0, 0.6), 2, 2)
  core <- crossprod(matrix(rnorm(4), 2, 2))
  Smat <- 0.5 * core
  got <- cpp_recov_block_grad(Smat, L, TRUE, 1.0)
  dS <- .re_block_dSigma(L, nc, TRUE)
  want <- vapply(dS, function(d) 0.5 * sum(diag(d %*% core)), numeric(1))
  expect_equal(as.numeric(got), want, tolerance = 1e-10)
})

test_that("the joint Hessian round-trips as a symmetric matrix", {
  d <- .exg_sim()
  layout <- .exg_layout(d, FALSE)
  L <- .exg_theta_to_L(log(0.7), 1L, FALSE)
  f <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                     re_list = .re_cov_build_re_list(list(L), layout),
                     family = "poisson", return_hessian = TRUE,
                     return_joint_hessian = TRUE)
  H <- f$H_joint
  expect_false(is.null(H))
  expect_equal(nrow(H), length(f$mode))
  expect_equal(ncol(H), length(f$mode))
  expect_true(Matrix::isSymmetric(H))
  # It is a posterior precision, so it must be positive definite.
  expect_true(min(eigen(as.matrix(H), symmetric = TRUE,
                        only.values = TRUE)$values) > 0)
  # Absent unless asked for, so no other caller pays for it.
  f0 <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                      re_list = .re_cov_build_re_list(list(L), layout),
                      family = "poisson", return_hessian = TRUE)
  expect_null(f0$H_joint)
})

test_that("the gradient-driven outer fit lands where the derivative-free one does", {
  d <- .exg_sim(seed = 7L, G = 20L, per = 8L)
  re <- list(idx = d$grp, n_groups = d$G, n_coefs = 1L)
  a <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson")

  orig <- cpp_family_has_curvature_derivative
  on.exit(assignInNamespace("cpp_family_has_curvature_derivative", orig,
                            ns = "tulpa"), add = TRUE)
  assignInNamespace("cpp_family_has_curvature_derivative",
                    function(family) FALSE, ns = "tulpa")
  b <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson")

  expect_equal(unname(coef(a)), unname(coef(b)), tolerance = 1e-4)
  expect_equal(unname(sqrt(diag(vcov(a)))), unname(sqrt(diag(vcov(b)))),
               tolerance = 1e-4)
})

test_that("the exact and finite-difference stencils give the same correction", {
  d <- .exg_sim(seed = 3L, G = 20L, per = 8L)
  re <- list(idx = d$grp, n_groups = d$G, n_coefs = 1L)
  a <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson", marginal = TRUE)
  skip_if(is.null(a$cov_marginal), "correction did not form")

  orig <- cpp_family_has_curvature_derivative
  on.exit(assignInNamespace("cpp_family_has_curvature_derivative", orig,
                            ns = "tulpa"), add = TRUE)
  assignInNamespace("cpp_family_has_curvature_derivative",
                    function(family) FALSE, ns = "tulpa")
  b <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson", marginal = TRUE)
  skip_if(is.null(b$cov_marginal), "fallback correction did not form")

  expect_equal(as.numeric(a$H_theta), as.numeric(b$H_theta), tolerance = 1e-4)
  expect_equal(sqrt(diag(a$cov_marginal)), sqrt(diag(b$cov_marginal)),
               tolerance = 1e-4)
})
