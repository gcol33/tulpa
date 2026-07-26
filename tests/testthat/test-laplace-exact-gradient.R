# The exact outer gradient (R/laplace_gradient.R) and the curvature-derivative
# ladder it stands on (src/laplace_family_curvature.h).
#
# The load-bearing check is agreement with a central difference of tulpa's OWN
# Laplace log-marginal: the derivation can be right on paper and still be wired
# to the wrong latent ordering, the wrong weight convention, or the wrong sign.

.exg_sim <- function(seed = 5L, G = 10L, per = 7L, nc = 1L, fam = "poisson",
                     phi = 2.5) {
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
  mu <- exp(eta)
  # zero-truncated draw: reject the empty class the family conditions away.
  pos <- function(rng) vapply(seq_len(n),
    function(i) { repeat { v <- rng(i); if (v >= 1L) return(v) } }, numeric(1))
  y <- switch(fam,
              poisson  = rpois(n, mu),
              binomial = rbinom(n, 1L, plogis(eta)),
              gaussian = rnorm(n, eta, 0.7),
              neg_binomial_2 = rnbinom(n, size = phi, mu = mu),
              neg_binomial_1 = rnbinom(n, size = mu / phi, mu = mu),
              truncated_poisson        = pos(function(i) rpois(1, mu[i])),
              truncated_neg_binomial_2 = pos(function(i) rnbinom(1, size = phi, mu = mu[i])))
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

# Closed-form outer Hessian vs a central difference of the analytic gradient --
# the Hessian analogue of .exg_compare. The gradient it differences is itself
# checked against tulpa's own marginal above, so this pins the second-derivative
# assembly (R/laplace_gradient.R) onto a verified first derivative.
.exg_hess_compare <- function(d, theta, full = FALSE, phi = 1.0) {
  layout <- .exg_layout(d, full)
  fit_at <- function(th) {
    L <- .exg_theta_to_L(th, d$nc, full)
    tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                  re_list = .re_cov_build_re_list(list(L), layout),
                  family = d$fam, phi = phi, return_hessian = TRUE,
                  return_joint_hessian = TRUE, max_iter = 300L, tol = 1e-12)
  }
  grad_at <- function(th) {
    L <- .exg_theta_to_L(th, d$nc, full)
    .laplace_exact_re_grad(
      fit = fit_at(th), y = d$y, X = d$X, n_trials = d$n_trials,
      offset = NULL, weights = NULL,
      re_list = .re_cov_build_re_list(list(L), layout),
      layout = layout, L_list = list(L), family = d$fam, phi = phi)
  }
  L0 <- .exg_theta_to_L(theta, d$nc, full)
  r <- .laplace_exact_re_grad(
    fit = fit_at(theta), y = d$y, X = d$X, n_trials = d$n_trials,
    offset = NULL, weights = NULL,
    re_list = .re_cov_build_re_list(list(L0), layout),
    layout = layout, L_list = list(L0), family = d$fam, phi = phi,
    want_hessian = TRUE)
  if (is.null(r) || is.null(r$H)) return(NULL)
  h <- 1e-5
  k <- length(theta)
  Hf <- matrix(0, k, k)
  for (j in seq_len(k)) {
    tp <- theta; tp[j] <- tp[j] + h
    tm <- theta; tm[j] <- tm[j] - h
    gp <- grad_at(tp); gm <- grad_at(tm)
    if (is.null(gp) || is.null(gm)) return(NULL)
    Hf[, j] <- (gp - gm) / (2 * h)
  }
  list(analytic = r$H, fd = (Hf + t(Hf)) / 2)
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

test_that("the closed-form Hessian matches a difference of the analytic gradient", {
  for (cs in list(list(fam = "poisson",  th = log(0.7)),
                  list(fam = "binomial", th = log(0.7)),
                  list(fam = "poisson",  th = log(1.6)))) {
    d <- .exg_sim(fam = cs$fam, G = if (cs$fam == "binomial") 14L else 10L,
                  per = if (cs$fam == "binomial") 10L else 7L)
    r <- .exg_hess_compare(d, cs$th)
    skip_if(is.null(r), "hessian unavailable")
    expect_equal(dim(r$analytic), c(1L, 1L))
    expect_equal(r$analytic, r$fd, tolerance = 1e-5,
                 info = paste(cs$fam, "at theta =", signif(cs$th, 4)))
  }
})

test_that("the closed-form Hessian matches for a correlated block", {
  d <- .exg_sim(seed = 5L, G = 12L, per = 8L, nc = 2L)
  r <- .exg_hess_compare(d, c(log(0.7), 0.2, log(0.6)), full = TRUE)
  skip_if(is.null(r), "hessian unavailable")
  expect_equal(dim(r$analytic), c(3L, 3L))
  expect_equal(r$analytic, r$fd, tolerance = 1e-5)
})

test_that("the closed Hessian declines where the working weight differs from observed", {
  # neg_binomial_1 and truncated_neg_binomial_2 build H from a working weight
  # that is NOT the observed curvature. The gradient's dW channel is
  # v_r' (dx_hat/dtheta), and the mode motion follows the TRUE stationarity
  # condition, so u is formed on the observed-curvature inverse Hinv_mode. The
  # closed Hessian's du/dtheta differentiates u through Hinv instead, which only
  # matches where the two inverses coincide. Forced on for these two families it
  # misses the exact H_theta by 2.6e-2 (nb1) and 2.7e-4 (tnb2), so it declines
  # and the caller central-differences the exact gradient.
  #
  # Asserted as a refusal rather than skipped: a silent skip would let the route
  # be re-enabled without anyone noticing it had gone back to being inexact.
  for (cs in list(list(fam = "neg_binomial_1",          phi = 2.5),
                  list(fam = "truncated_neg_binomial_2", phi = 2.5))) {
    d <- .exg_sim(fam = cs$fam, G = 24L, per = 10L, phi = cs$phi)
    layout <- .exg_layout(d, FALSE)
    L0 <- .exg_theta_to_L(log(0.7), d$nc, FALSE)
    fit <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                         re_list = .re_cov_build_re_list(list(L0), layout),
                         family = d$fam, phi = cs$phi, return_hessian = FALSE,
                         return_joint_hessian = TRUE, max_iter = 300L,
                         tol = 1e-12)
    r <- .laplace_exact_re_grad(
      fit = fit, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
      weights = NULL, re_list = .re_cov_build_re_list(list(L0), layout),
      layout = layout, L_list = list(L0), family = d$fam, phi = cs$phi,
      want_jacobian = TRUE, want_hessian = TRUE)
    expect_false(is.null(r), info = cs$fam)
    # Gradient and mode Jacobian stay exact; only the Hessian returns NULL.
    expect_true(all(is.finite(r$grad)), info = cs$fam)
    expect_true(all(is.finite(r$J)), info = cs$fam)
    expect_null(r$H, info = cs$fam)
  }
})

test_that("H_theta from the gradient stencil is exact where the closed route declines", {
  # What the declining families fall back to has to be right, or the refusal
  # above just moves the error. The stencil differences the exact gradient, so
  # it is checked against a second difference of the objective itself.
  for (cs in list(list(fam = "neg_binomial_1",          phi = 2.5),
                  list(fam = "truncated_neg_binomial_2", phi = 2.5))) {
    d <- .exg_sim(fam = cs$fam, G = 24L, per = 10L, phi = cs$phi)
    layout <- .exg_layout(d, FALSE)
    obj <- function(th) {
      L <- .exg_theta_to_L(th, d$nc, FALSE)
      tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                    re_list = .re_cov_build_re_list(list(L), layout),
                    family = d$fam, phi = cs$phi, return_hessian = FALSE,
                    max_iter = 300L, tol = 1e-12)$log_marginal
    }
    grad <- function(th) {
      L <- .exg_theta_to_L(th, d$nc, FALSE)
      fit <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                           re_list = .re_cov_build_re_list(list(L), layout),
                           family = d$fam, phi = cs$phi, return_hessian = FALSE,
                           return_joint_hessian = TRUE, max_iter = 300L,
                           tol = 1e-12)
      .laplace_exact_re_grad(
        fit = fit, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
        weights = NULL, re_list = .re_cov_build_re_list(list(L), layout),
        layout = layout, L_list = list(L), family = d$fam, phi = cs$phi)
    }
    th <- log(0.7); h <- 1e-4
    stencil <- (grad(th + h) - grad(th - h)) / (2 * h)
    second  <- (obj(th + h) - 2 * obj(th) + obj(th - h)) / (h * h)
    expect_equal(stencil, second, tolerance = 2e-3, info = cs$fam)
  }
})

test_that("the analytic mode Jacobian equals the differenced true inner mode", {
  # J = dx_hat/dtheta from true score stationarity carries the observed
  # curvature; the check differences tulpa's own inner mode, catching a
  # working-weight Jacobian directly rather than only through the Hessian.
  for (cs in list(list(fam = "neg_binomial_2",          phi = 2.5),
                  list(fam = "neg_binomial_1",          phi = 2.5),
                  list(fam = "truncated_neg_binomial_2", phi = 2.5))) {
    d <- .exg_sim(fam = cs$fam, G = 24L, per = 10L, phi = cs$phi)
    layout <- .exg_layout(d, FALSE)
    fit_at <- function(th) {
      L <- .exg_theta_to_L(th, 1L, FALSE)
      tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                    re_list = .re_cov_build_re_list(list(L), layout),
                    family = d$fam, phi = cs$phi, return_hessian = TRUE,
                    return_joint_hessian = TRUE, max_iter = 300L, tol = 1e-12)
    }
    L0 <- .exg_theta_to_L(log(0.7), 1L, FALSE)
    r <- .laplace_exact_re_grad(
      fit = fit_at(log(0.7)), y = d$y, X = d$X, n_trials = d$n_trials,
      offset = NULL, weights = NULL,
      re_list = .re_cov_build_re_list(list(L0), layout),
      layout = layout, L_list = list(L0), family = d$fam, phi = cs$phi,
      want_jacobian = TRUE)
    skip_if(is.null(r) || is.null(r$J), "jacobian unavailable")
    h <- 1e-5
    Jtrue <- (fit_at(log(0.7) + h)$mode - fit_at(log(0.7) - h)$mode) / (2 * h)
    expect_equal(as.numeric(r$J), Jtrue, tolerance = 1e-5, info = cs$fam)
  }
})

test_that("a gaussian response has a closed Hessian with the curvature at zero", {
  # The control: dw/deta and d2w/deta2 are identically zero, so the Hessian is
  # exercised through J, dR and dV alone -- the analogue of the gradient control.
  d <- .exg_sim(fam = "gaussian")
  layout <- .exg_layout(d, FALSE)
  L <- .exg_theta_to_L(log(0.7), 1L, FALSE)
  fit <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                       re_list = .re_cov_build_re_list(list(L), layout),
                       family = "gaussian", phi = 0.49,
                       return_hessian = TRUE, return_joint_hessian = TRUE)
  r <- .laplace_exact_re_grad(
    fit = fit, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
    weights = NULL, re_list = .re_cov_build_re_list(list(L), layout),
    layout = layout, L_list = list(L), family = "gaussian", phi = 0.49,
    want_hessian = TRUE)
  skip_if(is.null(r) || is.null(r$H), "hessian unavailable")
  expect_equal(dim(r$H), c(1L, 1L))
  expect_true(all(is.finite(r$H)))
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

test_that("d2w/deta2 differentiates dw/deta (curvature the theta-Hessian needs)", {
  # Same load-bearing idea as the dw/deta test: a central difference of the
  # verified first curvature derivative is the ground truth the closed-form
  # second derivative must reproduce, family by family.
  cases <- list(
    list(family = "poisson",          y = 4,   n = 1L,  phi = 1.0),
    list(family = "binomial",         y = 3,   n = 10L, phi = 1.0),
    list(family = "neg_binomial_2",   y = 6,   n = 1L,  phi = 2.5),
    list(family = "neg_binomial_1",   y = 6,   n = 1L,  phi = 1.7),
    list(family = "beta_binomial",    y = 3,   n = 10L, phi = 4.0),
    list(family = "tweedie",          y = 2.0, n = 1L,  phi = 1.3, phi2 = 1.6),
    list(family = "gamma",            y = 2.0, n = 1L,  phi = 3.0),
    list(family = "inverse_gaussian", y = 1.4, n = 1L,  phi = 1.1),
    list(family = "beta",             y = 0.6, n = 1L,  phi = 6.0),
    list(family = "binomial_probit",  y = 3,   n = 10L, phi = 1.0),
    list(family = "binomial_cloglog", y = 3,   n = 10L, phi = 1.0),
    list(family = "gaussian_log",     y = 1.2, n = 1L,  phi = 1.5)
  )
  h <- 1e-5
  for (cs in cases) {
    p2 <- if (is.null(cs$phi2)) NA_real_ else cs$phi2
    for (eta in c(-0.6, 0.0, 0.9)) {
      dw <- function(e) unname(cpp_family_curvature_deta(
        cs$y, cs$n, e, cs$family, cs$phi, p2)[["dw_deta"]])
      got <- unname(cpp_family_curvature_deta2(
        cs$y, cs$n, eta, cs$family, cs$phi, p2)[["d2w_deta2"]])
      num <- (dw(eta + h) - dw(eta - h)) / (2 * h)
      # Floor as before: gamma+log has dw==0 identically, so this is the
      # 0-vs-FD-noise case where a ratio would be meaningless.
      scale <- max(abs(num), abs(got), 1e-6)
      expect_lt(abs(got - num) / scale, 1e-4)
    }
  }
})

test_that("the second curvature derivative gate covers every family with a first", {
  # The truncated pair carry the third truncation-shape derivative d3a, so their
  # second eta-derivative is closed-form alongside every other family.
  for (f in c("poisson", "binomial", "neg_binomial_2", "neg_binomial_1",
              "beta_binomial", "t", "tweedie", "gamma", "beta",
              "binomial_probit", "gaussian_log",
              "truncated_poisson", "truncated_neg_binomial_2"))
    expect_true(cpp_family_has_curvature_2nd_derivative(f), info = f)
  expect_false(cpp_family_has_curvature_2nd_derivative("not_a_family"))
  # The truncated second eta-derivative FD-checks against the first, the same
  # gate proto_truncated_curvature.R pins in base R.
  h <- 1e-6
  for (cs in list(list(f = "truncated_poisson", phi = 1.0),
                  list(f = "truncated_neg_binomial_2", phi = 2.5))) {
    for (eta in c(-0.6, 0.2, 0.9)) {
      dw <- function(e) unname(cpp_family_curvature_deta(
        3, 1L, e, cs$f, cs$phi, NA_real_)[["dw_deta"]])
      got <- unname(cpp_family_curvature_deta2(
        3, 1L, eta, cs$f, cs$phi, NA_real_)[["d2w_deta2"]])
      num <- (dw(eta + h) - dw(eta - h)) / (2 * h)
      expect_lt(abs(got - num) / max(abs(num), abs(got), 1e-6), 1e-4, label = cs$f)
    }
  }
})

test_that("the exact mode-Jacobian gate tracks whether the observed curvature is available", {
  # Working weight equals observed curvature (canonical / constant), or an exact
  # observed form exists (has_observed_curvature): the analytic mode Jacobian is
  # exact and the closed route may run.
  for (f in c("poisson", "binomial", "neg_binomial_2", "gaussian",
              "truncated_poisson", "neg_binomial_1", "truncated_neg_binomial_2"))
    expect_true(cpp_family_has_exact_mode_jacobian(f), info = f)
  # Working weight differs from the observed curvature and no exact observed form
  # exists: the Jacobian cannot be formed, so the correction differences the mode.
  for (f in c("beta_binomial", "t", "tweedie"))
    expect_false(cpp_family_has_exact_mode_jacobian(f), info = f)
  # The observed-minus-working delta is identically zero where they coincide and
  # nonzero exactly for the two families whose observed curvature carries y.
  eta <- c(-0.5, 0.3, 1.1); y <- c(2, 3, 1)
  expect_true(all(cpp_family_obs_curvature_delta_vec(y, 1L, eta, "poisson", 1) == 0))
  expect_true(all(cpp_family_obs_curvature_delta_vec(y, 1L, eta, "neg_binomial_2", 2.5) == 0))
  expect_true(any(abs(cpp_family_obs_curvature_delta_vec(
    y, 1L, eta, "truncated_neg_binomial_2", 2.5)) > 1e-8))
  expect_true(any(abs(cpp_family_obs_curvature_delta_vec(
    y, 1L, eta, "neg_binomial_1", 2.5)) > 1e-8))
})

test_that("the vectorized second curvature derivative matches the scalar probe", {
  set.seed(3)
  eta <- rnorm(20)
  y   <- rpois(20, 2)
  vec <- cpp_family_curvature_deta2_vec(y, 1L, eta, "poisson", 1.0)
  sc  <- vapply(seq_along(eta), function(i) unname(cpp_family_curvature_deta2(
    y[i], 1L, eta[i], "poisson", 1.0)[["d2w_deta2"]]), numeric(1))
  expect_equal(vec, sc)
  expect_error(cpp_family_curvature_deta2_vec(y[1:3], 1L, eta, "poisson", 1.0),
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

test_that("the closed, exact-stencil and finite-difference corrections agree", {
  d <- .exg_sim(seed = 3L, G = 20L, per = 8L)
  re <- list(idx = d$grp, n_groups = d$G, n_coefs = 1L)
  # Default route: the closed-form outer Hessian, no differencing of any solve.
  a <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson", marginal = TRUE)
  skip_if(is.null(a$cov_marginal), "closed correction did not form")

  orig2 <- cpp_family_has_curvature_2nd_derivative
  orig1 <- cpp_family_has_curvature_derivative
  on.exit({
    assignInNamespace("cpp_family_has_curvature_2nd_derivative", orig2, ns = "tulpa")
    assignInNamespace("cpp_family_has_curvature_derivative", orig1, ns = "tulpa")
  }, add = TRUE)

  # Hide only the second curvature derivative -> the closed route declines and
  # the exact stencil (difference the analytic gradient) runs.
  assignInNamespace("cpp_family_has_curvature_2nd_derivative",
                    function(family) FALSE, ns = "tulpa")
  b <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson", marginal = TRUE)
  skip_if(is.null(b$cov_marginal), "exact-stencil correction did not form")

  # Hide the first as well -> the finite-difference stencil.
  assignInNamespace("cpp_family_has_curvature_derivative",
                    function(family) FALSE, ns = "tulpa")
  cc <- tulpa_eb(d$y, NULL, d$X, re, family = "poisson", marginal = TRUE)
  skip_if(is.null(cc$cov_marginal), "fallback correction did not form")

  # Closed vs exact stencil: both differentiate the same analytic gradient, so
  # they agree to the stencil's O(step^2) truncation. Against the objective-
  # differencing fallback the tolerance is looser.
  expect_equal(as.numeric(a$H_theta), as.numeric(b$H_theta), tolerance = 1e-4)
  expect_equal(as.numeric(a$H_theta), as.numeric(cc$H_theta), tolerance = 1e-4)
  expect_equal(sqrt(diag(a$cov_marginal)), sqrt(diag(cc$cov_marginal)),
               tolerance = 1e-4)
})
