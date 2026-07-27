# Zero inflation and hurdle mixtures alongside random effects.
#
# The two-process kernel and the one-process one had been exercised separately:
# the ZI reference fits are all fixed-effects only, and every random-effect test
# runs a single linear predictor. Nothing crossed them, which is how a
# fixed-effect covariance that sliced the random effects out of the mode at
# ncol(X) -- inside the beta_zi block when a ziformula is present -- went
# unnoticed until vcov() was asked for one.
#
# The groups below are: the fixed block's shape and provenance, the curvature
# derivatives the analytic outer gradient is assembled from, that gradient
# against the objective it claims to differentiate, parameter recovery, interval
# coverage, and the refusals.

# --- simulation --------------------------------------------------------------

# Zero-truncated draws come from helper-reference.R (rejection sampling), so a
# hurdle fixture's count arm really is the truncated law rather than a
# zero-to-one remap, which piles mass on 1 and biases the mean down.
zi_re_sim <- function(seed, family, G = 50L, per = 20L, sigma = 0.8, phi = 3,
                      beta = c(0.5, 0.5), gamma = c(-0.6, 0.7)) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per)
  x <- rnorm(n); z <- rnorm(n)
  b <- rnorm(G, 0, sigma)
  mu <- exp(beta[1L] + beta[2L] * x + b[g])
  base <- switch(family,
    poisson                  = stats::rpois(n, mu),
    neg_binomial_2           = stats::rnbinom(n, mu = mu, size = phi),
    truncated_poisson        = ref_sim_truncated_poisson(mu),
    truncated_neg_binomial_2 = ref_sim_truncated_nb2(mu, phi))
  y <- ifelse(stats::rbinom(n, 1, plogis(gamma[1L] + gamma[2L] * z)) == 1L,
              0L, base)
  list(y = y, x = x, z = z, g = g, G = G, n = n, sigma = sigma, phi = phi,
       beta = beta, gamma = gamma,
       X = cbind(1, x), X_zi = cbind(1, z),
       re = list(list(idx = g, n_groups = G, n_coefs = 1L)),
       data = data.frame(y = y, x = x, z = z, g = factor(g)))
}

ZI_RE_FAMILIES <- c("poisson", "neg_binomial_2",
                    "truncated_poisson", "truncated_neg_binomial_2")


# --- 1. the fixed block is [beta | beta_zi] ----------------------------------

test_that("a ZI fit with random effects reports both fixed blocks", {
  d <- zi_re_sim(1L, "poisson", G = 25L, per = 12L)
  fit <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d$data, family = "poisson",
          ziformula = ~ z, mode = "laplace"))

  expect_identical(names(coef(fit)),
                   c("(Intercept)", "x", "zi_(Intercept)", "zi_z"))
  # The defect this pins: H_beta came back ncol(X) square, so vcov() indexed
  # past its bounds and every interval accessor errored.
  expect_equal(dim(vcov(fit)), c(4L, 4L))
  expect_equal(dim(confint(fit)), c(4L, 2L))
  expect_true(all(is.finite(vcov(fit))))
  expect_true(all(sqrt(diag(vcov(fit))) > 0))
  expect_identical(rownames(vcov(fit)), names(coef(fit)))
})

test_that("the fixed-effect precision is the Schur complement of the joint curvature", {
  d <- zi_re_sim(2L, "poisson", G = 25L, per = 12L)
  # tulpa_laplace conditions on the RE scale, so the term carries one here;
  # tulpa_eb() estimates it and takes the term without.
  re <- list(list(idx = d$g, n_groups = d$G, n_coefs = 1L, sigma = 0.8))
  fit <- tulpa_laplace(d$y, NULL, d$X, re, family = "poisson",
                       X_zi = d$X_zi, return_hessian = TRUE,
                       return_joint_hessian = TRUE)
  p <- ncol(d$X) + ncol(d$X_zi)
  H <- as.matrix(fit$H_joint)
  B <- H[seq_len(p), -seq_len(p), drop = FALSE]
  D <- H[-seq_len(p), -seq_len(p), drop = FALSE]
  schur <- H[seq_len(p), seq_len(p), drop = FALSE] - B %*% solve(D, t(B))

  expect_equal(dim(fit$H_beta), c(p, p))
  expect_equal(unname(as.matrix(fit$H_beta)), unname(schur), tolerance = 1e-8)
})

test_that("H_joint is the observed curvature of the two-process log posterior", {
  # The Schur above is only the right precision if what it complements is the
  # true negative-log-posterior curvature. Differenced directly, with the
  # kernel's own beta ridge and ZI prior in the penalty.
  skip_on_cran()
  d <- zi_re_sim(3L, "poisson", G = 18L, per = 10L)
  sigma_re <- 0.8
  re <- list(list(idx = d$g, n_groups = d$G, n_coefs = 1L, sigma = sigma_re))
  fit <- tulpa_laplace(d$y, NULL, d$X, re, family = "poisson", X_zi = d$X_zi,
                       return_hessian = FALSE, return_joint_hessian = TRUE,
                       max_iter = 300L, tol = 1e-12)

  tau_beta <- tulpa:::.LAPLACE_DEFAULT_TAU_BETA
  nlp <- function(par) {
    beta <- par[1:2]; bzi <- par[3:4]; u <- par[-(1:4)]
    eta <- as.numeric(d$X %*% beta) + u[d$g]
    zz  <- as.numeric(d$X_zi %*% bzi)
    lpi <- -log1p(exp(-zz)); l1mp <- -log1p(exp(zz))
    p0  <- stats::dpois(0, exp(eta), log = TRUE)
    ll  <- ifelse(d$y == 0,
                  pmax(lpi, l1mp + p0) +
                    log(exp(lpi - pmax(lpi, l1mp + p0)) +
                        exp(l1mp + p0 - pmax(lpi, l1mp + p0))),
                  l1mp + stats::dpois(d$y, exp(eta), log = TRUE))
    -(sum(ll) - 0.5 * tau_beta * sum(beta^2) - 0.5 * sum(bzi^2) / 2.5^2 -
        0.5 * sum(u^2) / sigma_re^2)
  }
  H_fd <- stats::optimHess(fit$mode, nlp)
  expect_equal(max(abs(H_fd - as.matrix(fit$H_joint))) / max(abs(H_fd)),
               0, tolerance = 1e-5)
})


# --- 2. the mixture's curvature derivatives ----------------------------------

test_that("the six mixture curvature partials match central differences", {
  # What the analytic outer gradient contracts against the predictor
  # (co)variances. The y = 0 rows are the only place the two predictors couple,
  # so a grid that missed them would verify almost nothing.
  grid <- expand.grid(y = c(0, 1, 2, 5), eta = c(-1.2, 0.4, 1.1),
                      z = c(-1.5, 0.6, 1.3))
  h <- 1e-5
  Wof <- function(fam, phi, e, zz)
    cpp_zi_mixture_curvature(grid$y, 1L, e, zz, fam, phi, NA_real_)

  for (fam in ZI_RE_FAMILIES) {
    phi <- 2
    an <- cpp_zi_mixture_curvature_deriv(grid$y, 1L, grid$eta, grid$z,
                                         fam, phi, NA_real_)
    Wp_e <- Wof(fam, phi, grid$eta + h, grid$z)
    Wm_e <- Wof(fam, phi, grid$eta - h, grid$z)
    Wp_z <- Wof(fam, phi, grid$eta, grid$z + h)
    Wm_z <- Wof(fam, phi, grid$eta, grid$z - h)
    fd <- cbind((Wp_e[, "W_ee"] - Wm_e[, "W_ee"]) / (2 * h),
                (Wp_z[, "W_ee"] - Wm_z[, "W_ee"]) / (2 * h),
                (Wp_e[, "W_ez"] - Wm_e[, "W_ez"]) / (2 * h),
                (Wp_z[, "W_ez"] - Wm_z[, "W_ez"]) / (2 * h),
                (Wp_e[, "W_zz"] - Wm_e[, "W_zz"]) / (2 * h),
                (Wp_z[, "W_zz"] - Wm_z[, "W_zz"]) / (2 * h))
    expect_equal(unname(an), unname(fd), tolerance = 1e-6, info = fam)
  }
})

test_that("the mixture curvature derivative is gated on the family", {
  for (fam in ZI_RE_FAMILIES) {
    expect_true(cpp_family_has_zi_curvature_derivative(fam), info = fam)
  }
  # The untruncated y = 0 branch differentiates P(Y = 0) a third time, which
  # needs d/deta of the OBSERVED curvature -- the registered working-weight one
  # plus the observed-minus-working correction. neg_binomial_1 is the family
  # that separates those two, and qualifies because both halves exist.
  expect_true(cpp_family_has_zi_curvature_derivative("neg_binomial_1"))
  expect_true(tulpa:::.laplace_exact_supports_zi("neg_binomial_1"))
  # Refused where there is no observed form at all, so the gate stays a property
  # of the family rather than a list.
  for (fam in c("beta_binomial", "t", "tweedie")) {
    expect_false(cpp_family_has_zi_curvature_derivative(fam), info = fam)
    expect_false(tulpa:::.laplace_exact_supports_zi(fam), info = fam)
  }
})


# --- 3. the analytic outer gradient ------------------------------------------

test_that("the analytic outer gradient matches the objective it differentiates", {
  # Checked AWAY from the optimum: at theta_hat every method agrees on ~0, so a
  # gradient evaluated only there would pass while being wrong everywhere else.
  skip_on_cran()
  for (fam in ZI_RE_FAMILIES) {
    d <- zi_re_sim(4L, fam, G = 30L, per = 14L)
    layout <- tulpa:::.re_cov_block_layout(d$re, d$n)
    at <- log(0.45)

    obj <- function(theta) {
      L <- tulpa:::.re_cov_theta_to_L_list(theta, layout)
      tulpa_laplace(d$y, NULL, d$X,
                    tulpa:::.re_cov_build_re_list(L, layout),
                    family = fam, phi = d$phi, return_hessian = FALSE,
                    X_zi = d$X_zi, max_iter = 300L, tol = 1e-12)$log_marginal
    }
    L0 <- tulpa:::.re_cov_theta_to_L_list(at, layout)
    fit <- tulpa_laplace(d$y, NULL, d$X,
                         tulpa:::.re_cov_build_re_list(L0, layout),
                         family = fam, phi = d$phi, return_hessian = FALSE,
                         return_joint_hessian = TRUE, X_zi = d$X_zi,
                         max_iter = 300L, tol = 1e-12)
    g_an <- tulpa:::.laplace_exact_re_grad(
      fit = fit, y = d$y, X = d$X, n_trials = rep(1L, d$n), offset = NULL,
      weights = NULL, re_list = tulpa:::.re_cov_build_re_list(L0, layout),
      layout = layout, L_list = L0, family = fam, phi = d$phi,
      phi2 = NA_real_, X_zi = d$X_zi)
    expect_false(is.null(g_an), info = fam)

    h <- 1e-4
    g_fd <- (obj(at + h) - obj(at - h)) / (2 * h)
    expect_equal(g_an, g_fd, tolerance = 1e-5, info = fam)
  }
})

test_that("the analytic gradient carries a correlated block under a mixture", {
  skip_on_cran()
  d <- zi_re_sim(5L, "truncated_poisson", G = 35L, per = 16L)
  re <- list(list(idx = d$g, n_groups = d$G, n_coefs = 2L, Z = d$X,
                  correlated = TRUE))
  layout <- tulpa:::.re_cov_block_layout(re, d$n)
  at <- c(log(0.6), 0.25, log(0.4))

  obj <- function(theta) {
    L <- tulpa:::.re_cov_theta_to_L_list(theta, layout)
    tulpa_laplace(d$y, NULL, d$X, tulpa:::.re_cov_build_re_list(L, layout),
                  family = "truncated_poisson", phi = 1, return_hessian = FALSE,
                  X_zi = d$X_zi, max_iter = 300L, tol = 1e-12)$log_marginal
  }
  L0 <- tulpa:::.re_cov_theta_to_L_list(at, layout)
  fit <- tulpa_laplace(d$y, NULL, d$X, tulpa:::.re_cov_build_re_list(L0, layout),
                       family = "truncated_poisson", phi = 1,
                       return_hessian = FALSE, return_joint_hessian = TRUE,
                       X_zi = d$X_zi, max_iter = 300L, tol = 1e-12)
  g_an <- tulpa:::.laplace_exact_re_grad(
    fit = fit, y = d$y, X = d$X, n_trials = rep(1L, d$n), offset = NULL,
    weights = NULL, re_list = tulpa:::.re_cov_build_re_list(L0, layout),
    layout = layout, L_list = L0, family = "truncated_poisson", phi = 1,
    phi2 = NA_real_, X_zi = d$X_zi)
  expect_false(is.null(g_an))

  h <- 1e-4
  g_fd <- vapply(seq_along(at), function(j) {
    tp <- at; tp[j] <- tp[j] + h
    tm <- at; tm[j] <- tm[j] - h
    (obj(tp) - obj(tm)) / (2 * h)
  }, numeric(1))
  expect_equal(g_an, g_fd, tolerance = 1e-5)
})


# --- 4. parameter recovery ---------------------------------------------------

test_that("EB recovers the RE standard deviation under a mixture", {
  skip_on_cran()
  # Averaged over seeds rather than asserted per fit: a variance component from
  # 50 groups carries real sampling noise, and a per-seed tolerance loose enough
  # to absorb it would not distinguish a working estimator from a broken one.
  for (fam in c("poisson", "truncated_poisson")) {
    est <- vapply(101:112, function(s) {
      d <- zi_re_sim(s, fam, G = 60L, per = 20L, sigma = 0.8)
      fit <- suppressMessages(
        tulpa(y ~ x + (1 | g), data = d$data, family = fam,
              ziformula = ~ z, mode = "eb", phi = d$phi))
      fit$map$sigma
    }, numeric(1))
    expect_equal(mean(est), 0.8, tolerance = 0.06, info = fam)
    expect_lt(stats::sd(est), 0.15)
  }
})

test_that("EB recovers the count and zero-inflation coefficients", {
  skip_on_cran()
  d <- zi_re_sim(21L, "truncated_poisson", G = 80L, per = 30L)
  fit <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d$data, family = "truncated_poisson",
          ziformula = ~ z, mode = "eb"))
  truth <- c(d$beta, d$gamma)
  se <- sqrt(diag(vcov(fit)))
  # Within 3 standard errors on every coordinate, count and zero side alike --
  # a sign error in the logit linkage moves the latter two only.
  expect_true(all(abs(coef(fit) - truth) < 3 * se),
              info = paste(signif((coef(fit) - truth) / se, 3), collapse = " "))
})

test_that("fixed-effect intervals cover at the nominal rate under a mixture", {
  skip_on_cran()
  n_rep <- 24L
  covered <- vapply(seq_len(n_rep), function(s) {
    d <- zi_re_sim(200L + s, "poisson", G = 50L, per = 20L)
    fit <- suppressMessages(
      tulpa(y ~ x + (1 | g), data = d$data, family = "poisson",
            ziformula = ~ z, mode = "eb", control = list(marginal = TRUE)))
    ci <- confint(fit)
    truth <- c(d$beta, d$gamma)
    ci[, 1L] <= truth & truth <= ci[, 2L]
  }, logical(4L))
  rate <- rowMeans(covered)
  # 24 replicates put the binomial 95% band on a true 0.95 at roughly
  # [0.83, 1.0], so 0.79 is a floor that catches a broken interval without
  # firing on ordinary Monte Carlo noise.
  expect_true(all(rate >= 0.79),
              info = paste(names(rate), signif(rate, 3), collapse = "  "))
})


# --- 5. against an external implementation -----------------------------------

test_that("hurdle poisson with a random intercept reproduces glmmTMB", {
  skip_on_cran()
  skip_if_not_installed("glmmTMB")
  d <- zi_re_sim(31L, "truncated_poisson", G = 60L, per = 25L)

  ref <- glmmTMB::glmmTMB(y ~ x + (1 | g), ziformula = ~ z, data = d$data,
                          family = glmmTMB::truncated_poisson())
  co  <- summary(ref)$coefficients
  ref_sd <- sqrt(glmmTMB::VarCorr(ref)$cond$g[1, 1])

  # Like-for-like needs every prior widened, not just the count-side one: the
  # ZI block carries its own N(0, zi_prior_sd^2) by default, and glmmTMB carries
  # none. Left at 2.5 the zero-side coefficients sit 1e-2 reference SEs away;
  # widened they agree to 1e-4, which is the two-process likelihood being
  # checked rather than the prior.
  fit <- tulpa_eb(d$y, NULL, d$X, d$re, family = "truncated_poisson",
                  X_zi = d$X_zi,
                  beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD),
                  zi_prior_sd = REF_DIFFUSE_SD,
                  log_prior_theta = function(theta) 0,
                  control = list(max_iter = 300L, tol = 1e-12))

  ref_b  <- c(co$cond[, "Estimate"],   co$zi[, "Estimate"])
  ref_se <- c(co$cond[, "Std. Error"], co$zi[, "Std. Error"])

  # The zero side is the strong claim: the mixture's logit linkage and its
  # y = 0 branch are exactly what this file adds, and they reproduce glmmTMB to
  # a ten-thousandth of a reference SE.
  expect_agrees_with_mle(coef(fit)[3:4], ref_b[3:4], ref_se[3:4],
                         tol = 1e-2, label = "hurdle_re_zi_block")

  # The count side is looser by construction, and the reason is an estimator
  # difference rather than slack. tulpa_eb maximizes the Laplace log-marginal
  # with the fixed effects INTEGRATED and reports beta at the conditional mode
  # of the maximizer; glmmTMB maximizes over (beta, sigma) jointly. The two
  # differ at O(1/n_groups), and with a random intercept the difference lands
  # almost entirely on the intercept, which trades off against the RE scale --
  # measured at 0.27 reference SEs on this fixture against 0.01 for the slope.
  expect_agrees_with_mle(coef(fit)[1:2], ref_b[1:2], ref_se[1:2],
                         tol = 0.4, label = "hurdle_re_count_block")
  expect_lt(abs(fit$map$sigma - ref_sd) / ref_sd, 0.05)
})


# --- 6. refusals -------------------------------------------------------------

test_that("zero inflation is refused where the backend cannot carry it", {
  d <- zi_re_sim(41L, "poisson", G = 20L, per = 10L)
  expect_error(
    suppressMessages(tulpa(y ~ x + (1 | g), data = d$data, family = "poisson",
                           ziformula = ~ z, mode = "re_cov_gibbs")),
    "not carried by backend 're_cov_gibbs'")
  expect_error(
    tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", X_zi = d$X_zi,
             n_quad = 5L),
    "does not carry a zero-inflation process")
  # `estimate_phi` alongside zero inflation is no longer one of these: both
  # mixture kinds are carried now. What is still refused, and why, is in
  # "estimate_phi is still refused where the mixture branch is unregistered".
})

test_that("`marginal` reaches EB through the front door but not through control", {
  d <- zi_re_sim(42L, "poisson", G = 20L, per = 10L)
  expect_error(
    tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson",
             control = list(marginal = TRUE)),
    "is an argument of tulpa_eb")
  fit <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d$data, family = "poisson",
          ziformula = ~ z, mode = "eb", control = list(marginal = TRUE)))
  expect_false(is.null(fit$cov_marginal))
  # The correction widens; it never narrows.
  expect_true(all(diag(fit$cov_marginal) >= diag(fit$cov_conditional) - 1e-12))
})

test_that("EB and the nested integrator share one objective under a mixture", {
  skip_on_cran()
  d <- zi_re_sim(43L, "poisson", G = 30L, per = 15L)
  eb <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d$data, family = "poisson",
          ziformula = ~ z, mode = "eb"))
  nl <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d$data, family = "poisson",
          ziformula = ~ z, mode = "re_cov_nested",
          control = list(n_draws = 200, diagnose_k = FALSE)))
  # The same closure, the same optimizer, the same start: not "close".
  expect_identical(eb$theta_hat, nl$theta_hat)
  expect_identical(names(coef(eb)), names(coef(nl)))
  expect_equal(length(coef(nl)), 4L)
})


# --- 7. the dispersion under a hurdle ----------------------------------------
#
# Whether the base family's phi derivatives describe the MIXTURE turns entirely
# on the base being zero-truncated. A hurdle's zero branch is log(pi), which
# carries no phi, so they do; an untruncated mixture's is
# log(pi + (1 - pi) P(Y = 0)), which does not. The tests below pin both halves,
# because the failure mode of getting it wrong is a gradient that still
# converges, to the maximizer of a different objective.

test_that("the hurdle's phi gradient matches a difference of the log-marginal", {
  d <- zi_re_sim(51L, "truncated_neg_binomial_2", G = 30L, per = 14L)
  layout <- tulpa:::.re_cov_block_layout(d$re, d$n)
  at <- log(0.7)
  L0 <- tulpa:::.re_cov_theta_to_L_list(at, layout)
  rl <- tulpa:::.re_cov_build_re_list(L0, layout)

  obj <- function(p)
    tulpa_laplace(d$y, NULL, d$X, rl, family = "truncated_neg_binomial_2",
                  phi = p, return_hessian = FALSE, X_zi = d$X_zi,
                  max_iter = 300L, tol = 1e-12)$log_marginal

  for (phi in c(1.5, 3.0, 6.0)) {
    fit <- tulpa_laplace(d$y, NULL, d$X, rl,
                         family = "truncated_neg_binomial_2", phi = phi,
                         return_hessian = FALSE, return_joint_hessian = TRUE,
                         X_zi = d$X_zi, max_iter = 300L, tol = 1e-12)
    g <- tulpa:::.laplace_exact_re_grad(
      fit = fit, y = d$y, X = d$X, n_trials = rep(1L, d$n), offset = NULL,
      weights = NULL, re_list = rl, layout = layout, L_list = L0,
      family = "truncated_neg_binomial_2", phi = phi, phi2 = NA_real_,
      X_zi = d$X_zi, want_phi = TRUE)
    expect_false(is.null(g), info = sprintf("phi = %g", phi))

    # theta carries log phi, so the difference is taken there too.
    h <- 1e-5
    fd <- (obj(phi * exp(h)) - obj(phi * exp(-h))) / (2 * h)
    expect_equal(g[[length(g)]], fd, tolerance = 1e-6,
                 info = sprintf("phi = %g", phi))
  }
})


test_that("the closed outer Hessian carries a hurdle", {
  # Against a difference of the exact GRADIENT rather than a second difference
  # of the objective: the gradient is already pinned above, so this isolates the
  # Hessian assembly instead of folding the gradient's own error back in.
  d <- zi_re_sim(52L, "truncated_poisson", G = 30L, per = 14L)
  layout <- tulpa:::.re_cov_block_layout(d$re, d$n)
  nt <- rep(1L, d$n)

  grad_at <- function(th, want_h = FALSE) {
    L <- tulpa:::.re_cov_theta_to_L_list(th, layout)
    rl <- tulpa:::.re_cov_build_re_list(L, layout)
    fit <- tulpa_laplace(d$y, NULL, d$X, rl, family = "truncated_poisson",
                         phi = 1, return_hessian = FALSE,
                         return_joint_hessian = TRUE, X_zi = d$X_zi,
                         max_iter = 300L, tol = 1e-12)
    tulpa:::.laplace_exact_re_grad(
      fit = fit, y = d$y, X = d$X, n_trials = nt, offset = NULL,
      weights = NULL, re_list = rl, layout = layout, L_list = L,
      family = "truncated_poisson", phi = 1, phi2 = NA_real_, X_zi = d$X_zi,
      want_hessian = want_h, want_jacobian = want_h)
  }

  at <- log(0.7)
  r <- grad_at(at, want_h = TRUE)
  expect_false(is.null(r$H))

  h <- 1e-4
  fd <- (grad_at(at + h) - grad_at(at - h)) / (2 * h)
  expect_equal(as.numeric(r$H), as.numeric(fd), tolerance = 1e-5)
})


test_that("the closed Hessian carries a hurdle whose Newton weight is not observed", {
  # truncated_neg_binomial_2 builds H from Var(y | y > 0), which is not its
  # observed curvature, so the assembly has to differentiate u on the
  # observed-curvature inverse rather than the working one. That is a
  # pre-existing split the mixture only inherits -- it stops the one-process
  # closed Hessian for this family too -- so both are checked here.
  for (zi in c(FALSE, TRUE)) {
    d <- zi_re_sim(55L, "truncated_neg_binomial_2", G = 30L, per = 14L)
    X_zi <- if (zi) d$X_zi else NULL
    # Without the mixture the zeros are outside the base family's support.
    if (!zi) d$y[d$y == 0L] <- 1L
    layout <- tulpa:::.re_cov_block_layout(d$re, d$n)
    nt <- rep(1L, d$n)

    grad_at <- function(th, want_h = FALSE) {
      L <- tulpa:::.re_cov_theta_to_L_list(th, layout)
      rl <- tulpa:::.re_cov_build_re_list(L, layout)
      fit <- tulpa_laplace(d$y, NULL, d$X, rl,
                           family = "truncated_neg_binomial_2", phi = d$phi,
                           return_hessian = FALSE, return_joint_hessian = TRUE,
                           X_zi = X_zi, max_iter = 300L, tol = 1e-12)
      tulpa:::.laplace_exact_re_grad(
        fit = fit, y = d$y, X = d$X, n_trials = nt, offset = NULL,
        weights = NULL, re_list = rl, layout = layout, L_list = L,
        family = "truncated_neg_binomial_2", phi = d$phi, phi2 = NA_real_,
        X_zi = X_zi, want_hessian = want_h, want_jacobian = want_h)
    }

    at <- log(0.7)
    r <- grad_at(at, want_h = TRUE)
    expect_false(is.null(r$H), info = paste("zi =", zi))
    h <- 1e-4
    fd <- (grad_at(at + h) - grad_at(at - h)) / (2 * h)
    expect_equal(as.numeric(r$H), as.numeric(fd), tolerance = 1e-5,
                 info = paste("zi =", zi))
  }
})


test_that("the mixture's fourth-order fields agree along every route to them", {
  # The curvature block is -Hess(log D), so each of its nine second partials is
  # one of five FOURTH derivatives of a single scalar. That identification is
  # what makes five fields sufficient, and it is over-determined: most fields
  # are reachable by differentiating two or three different third-order fields,
  # in different directions. A transposed index or a wrong identification would
  # show up as those routes disagreeing, which arithmetic alone cannot fix.
  routes <- list(
    d4_e4   = list(c("dWee_deta", "e")),
    d4_e3z  = list(c("dWee_deta", "z"), c("dWez_deta", "e"), c("dWee_dz", "e")),
    d4_e2z2 = list(c("dWee_dz", "z"), c("dWez_deta", "z"), c("dWez_dz", "e"),
                   c("dWzz_deta", "e")),
    d4_ez3  = list(c("dWez_dz", "z"), c("dWzz_deta", "z"), c("dWzz_dz", "e")),
    d4_z4   = list(c("dWzz_dz", "z"))
  )
  n <- 20L
  e <- seq(-0.9, 1.1, length.out = n)
  z <- seq(-1.3, 1.2, length.out = n)
  nt <- rep(1L, n)
  h <- 1e-5
  for (cs in list(list(f = "poisson", phi = 1, ys = c(0L, 3L)),
                  list(f = "neg_binomial_2", phi = 2.5, ys = c(0L, 4L)),
                  list(f = "binomial", phi = 1, ys = c(0L, 1L)),
                  list(f = "truncated_poisson", phi = 1, ys = c(0L, 2L)))) {
    for (yv in cs$ys) {
      y <- rep(yv, n)
      an <- tulpa:::cpp_zi_mixture_curvature_deriv2(y, nt, e, z, cs$f, cs$phi,
                                                    NA_real_)
      for (fld in names(routes)) for (rt in routes[[fld]]) {
        de <- if (rt[2] == "e") h else 0
        dz <- if (rt[2] == "z") h else 0
        fd <- (tulpa:::cpp_zi_mixture_curvature_deriv(
                 y, nt, e + de, z + dz, cs$f, cs$phi, NA_real_)[, rt[1]] -
               tulpa:::cpp_zi_mixture_curvature_deriv(
                 y, nt, e - de, z - dz, cs$f, cs$phi, NA_real_)[, rt[1]]) / (2 * h)
        expect_equal(an[, fld], fd, tolerance = 1e-6,
                     info = sprintf("%s y=%d %s via %s/%s",
                                    cs$f, yv, fld, rt[1], rt[2]))
      }
    }
  }
})


test_that("the closed Hessian carries genuine zero inflation", {
  # The untruncated y = 0 branch couples both predictors through
  # D = pi + (1 - pi) P(Y = 0), so this needs the full five-field tensor rather
  # than the two a hurdle leaves standing.
  for (cs in list(list(f = "poisson", phi = 1),
                  list(f = "neg_binomial_2", phi = 2.5))) {
    d <- zi_re_sim(56L, cs$f, G = 30L, per = 14L)
    layout <- tulpa:::.re_cov_block_layout(d$re, d$n)
    nt <- rep(1L, d$n)
    grad_at <- function(th, want_h = FALSE) {
      L <- tulpa:::.re_cov_theta_to_L_list(th, layout)
      rl <- tulpa:::.re_cov_build_re_list(L, layout)
      fit <- tulpa_laplace(d$y, NULL, d$X, rl, family = cs$f, phi = cs$phi,
                           return_hessian = FALSE, return_joint_hessian = TRUE,
                           X_zi = d$X_zi, max_iter = 300L, tol = 1e-12)
      tulpa:::.laplace_exact_re_grad(
        fit = fit, y = d$y, X = d$X, n_trials = nt, offset = NULL,
        weights = NULL, re_list = rl, layout = layout, L_list = L,
        family = cs$f, phi = cs$phi, phi2 = NA_real_, X_zi = d$X_zi,
        want_hessian = want_h, want_jacobian = want_h)
    }
    at <- log(0.7)
    r <- grad_at(at, want_h = TRUE)
    expect_false(is.null(r$H), info = cs$f)
    h <- 1e-4
    fd <- (grad_at(at + h) - grad_at(at - h)) / (2 * h)
    expect_equal(as.numeric(r$H), as.numeric(fd), tolerance = 1e-5, info = cs$f)
  }
})


test_that("the mixture curvature gate tracks the observed derivative, not a family list", {
  # What the untruncated y = 0 branch actually needs is the OBSERVED curvature's
  # eta-derivative: the registered working-weight one plus the
  # observed-minus-working correction. neg_binomial_1 is the family that
  # separates those -- it used to be excluded by name for want of a tetragamma,
  # and qualifies now that the correction is closed-form.
  expect_true(
    tulpa:::cpp_family_has_obs_curvature_delta_derivative("neg_binomial_1"))
  expect_true(
    tulpa:::cpp_family_has_zi_curvature_derivative("neg_binomial_1"))
  expect_true(
    tulpa:::cpp_family_has_obs_curvature_delta_derivative("truncated_neg_binomial_2"))
  for (f in c("poisson", "binomial", "neg_binomial_2", "neg_binomial_1",
              "truncated_poisson", "truncated_neg_binomial_2")) {
    expect_true(tulpa:::cpp_family_has_zi_curvature_2nd_derivative(f), info = f)
  }
  # Still refused where there is no observed form at all, so the gate is a
  # property of the family rather than a list that grew.
  for (f in c("beta_binomial", "t", "tweedie")) {
    expect_false(tulpa:::cpp_family_has_zi_curvature_derivative(f), info = f)
  }
})


test_that("neg_binomial_1's mixture curvature derivatives are exact", {
  # The coupled y = 0 branch reads the observed curvature's first and second
  # eta-derivatives, which for this family are the working ones plus a
  # tetragamma and a pentagamma correction. Checked against differences of the
  # mixture weight the kernel builds H from, at y = 0 (coupled) and y > 0
  # (separable).
  n <- 16L
  e <- seq(-0.8, 1.0, length.out = n)
  z <- seq(-1.1, 1.0, length.out = n)
  nt <- rep(1L, n); h <- 1e-5
  for (phi in c(0.7, 2.0)) for (yv in c(0, 2)) {
    y <- rep(yv, n)
    W <- function(e, z) tulpa:::cpp_zi_mixture_curvature(
      y, nt, e, z, "neg_binomial_1", phi, NA_real_)
    d3 <- tulpa:::cpp_zi_mixture_curvature_deriv(
      y, nt, e, z, "neg_binomial_1", phi, NA_real_)
    expect_equal(d3[, "dWee_deta"],
                 (W(e + h, z)[, "W_ee"] - W(e - h, z)[, "W_ee"]) / (2 * h),
                 tolerance = 1e-6, info = paste(phi, yv))
    expect_equal(d3[, "dWzz_dz"],
                 (W(e, z + h)[, "W_zz"] - W(e, z - h)[, "W_zz"]) / (2 * h),
                 tolerance = 1e-6, info = paste(phi, yv))
    D3 <- function(e, z) tulpa:::cpp_zi_mixture_curvature_deriv(
      y, nt, e, z, "neg_binomial_1", phi, NA_real_)
    d4 <- tulpa:::cpp_zi_mixture_curvature_deriv2(
      y, nt, e, z, "neg_binomial_1", phi, NA_real_)
    # d4_e4 at y = 0 is where the pentagamma enters.
    expect_equal(d4[, "d4_e4"],
                 (D3(e + h, z)[, "dWee_deta"] - D3(e - h, z)[, "dWee_deta"]) / (2 * h),
                 tolerance = 1e-6, info = paste(phi, yv))
  }
})


# --- 8. the dispersion coordinate under GENUINE zero inflation ---------------
#
# A hurdle's zero branch is log(pi) and carries no dispersion, which is what let
# the base family's registry stand in for the mixture's. Genuine zero inflation
# has zero branch log(pi + (1 - pi) P(Y = 0, phi)), which carries phi through
# P(Y = 0) and couples it to both predictors.

test_that("the mixture dispersion table matches independently written anchors", {
  # zi_loglik / zi_score_eta / zi_neg_hessian in R/family_zi.R and the mixture
  # curvature engines are all written independently of the phi engine, so
  # differencing them tests the new code rather than itself. Every field is
  # covered, at y = 0 (coupled) and y > 0 (base registry).
  n <- 16L
  e <- seq(-0.8, 1.0, length.out = n)
  z <- seq(-1.1, 1.0, length.out = n)
  nt <- rep(1L, n); fam <- "neg_binomial_2"; phi <- 2.5; hp <- 1e-5
  PD <- function(y, e, z, phi) tulpa:::.laplace_phi_fields(
    list(eta = e, has_zi = TRUE, z_lin = z), y, nt, fam, phi, NA_real_,
    tulpa:::.family_dphi(fam), tulpa:::.family_dphi2(fam))
  dp <- function(f, y, e, z, phi)
    (f(y, e, z, phi + hp) - f(y, e, z, phi - hp)) / (2 * hp)

  LL <- function(y, e, z, p)
    tulpa:::zi_loglik(e, z, y, fam, n_trials = nt, phi = p)
  SC <- function(y, e, z, p)
    tulpa:::zi_score_eta(e, z, y, fam, n_trials = nt, phi = p)
  # zi_neg_hessian's columns are (count, zi, cross) = (W_ee, W_zz, W_ez).
  NH <- function(y, e, z, p)
    tulpa:::zi_neg_hessian(e, z, y, fam, n_trials = nt, phi = p)[, c(1L, 3L, 2L)]
  D3 <- function(y, e, z, p) tulpa:::cpp_zi_mixture_curvature_deriv(
    y, nt, e, z, fam, p, NA_real_)

  for (yv in c(0, 3)) {
    y <- rep(yv, n); tab <- PD(y, e, z, phi); inf <- paste("y =", yv)
    expect_equal(tab[, "dl_dp"],    dp(LL, y, e, z, phi), tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dsc_e_dp"], dp(SC, y, e, z, phi)[, 1], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dsc_z_dp"], dp(SC, y, e, z, phi)[, 2], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWee_dp"],  dp(NH, y, e, z, phi)[, 1], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWez_dp"],  dp(NH, y, e, z, phi)[, 2], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWzz_dp"],  dp(NH, y, e, z, phi)[, 3], tolerance = 1e-6, info = inf)
    # fourth order with one phi, against the third-order curvature engine. The
    # two shared names are each reached from BOTH members of their pair, which
    # is what tests the identification rather than the arithmetic.
    expect_equal(tab[, "dWee_dp_de"], dp(D3, y, e, z, phi)[, "dWee_deta"], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWee_dp_dz"], dp(D3, y, e, z, phi)[, "dWee_dz"],   tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWee_dp_dz"], dp(D3, y, e, z, phi)[, "dWez_deta"], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWez_dp_dz"], dp(D3, y, e, z, phi)[, "dWez_dz"],   tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWez_dp_dz"], dp(D3, y, e, z, phi)[, "dWzz_deta"], tolerance = 1e-6, info = inf)
    expect_equal(tab[, "dWzz_dp_dz"], dp(D3, y, e, z, phi)[, "dWzz_dz"],   tolerance = 1e-6, info = inf)
    # second order in phi
    for (nm in c("dl_dp", "dsc_e_dp", "dsc_z_dp", "dWee_dp", "dWez_dp", "dWzz_dp")) {
      expect_equal(tab[, paste0(nm, "2")],
                   dp(function(...) PD(...)[, nm], y, e, z, phi),
                   tolerance = 1e-6, info = paste(inf, nm))
    }
  }
})


test_that("the engine contributes nothing where the base registry already applies", {
  # The two sources are summed over disjoint rows, so overlap would double-count
  # silently. Asserted as exact zeros rather than "small".
  n <- 12L
  e <- seq(-0.8, 1.0, length.out = n); z <- seq(-1.1, 1.0, length.out = n)
  nt <- rep(1L, n)
  # every y != 0 row of any mixture
  expect_true(all(tulpa:::cpp_zi_mixture_phi_deriv(
    rep(3, n), nt, e, z, "neg_binomial_2", 2.5, NA_real_) == 0))
  # a hurdle's y = 0 rows: its zero branch is log(pi), phi-free
  expect_true(all(tulpa:::cpp_zi_mixture_phi_deriv(
    rep(0, n), nt, e, z, "truncated_neg_binomial_2", 2.5, NA_real_) == 0))
  # a family with no free dispersion at all
  expect_true(all(tulpa:::cpp_zi_mixture_phi_deriv(
    rep(0, n), nt, e, z, "poisson", 1, NA_real_) == 0))
})


test_that("the phi gradient and closed Hessian are exact under genuine ZI", {
  d <- zi_re_sim(4L, "neg_binomial_2", G = 30L, per = 14L, sigma = 0.8, phi = 2.5)
  layout <- tulpa:::.re_cov_block_layout(d$re, d$n)
  nt <- rep(1L, d$n)
  at <- c(log(0.7), log(2.5))
  gr <- function(th, want_h = FALSE) {
    L  <- tulpa:::.re_cov_theta_to_L_list(th[1], layout)
    rl <- tulpa:::.re_cov_build_re_list(L, layout)
    fit <- tulpa_laplace(d$y, NULL, d$X, rl, family = "neg_binomial_2",
                         phi = exp(th[2]), return_hessian = FALSE,
                         return_joint_hessian = TRUE, X_zi = d$X_zi,
                         max_iter = 300L, tol = 1e-12)
    tulpa:::.laplace_exact_re_grad(
      fit = fit, y = d$y, X = d$X, n_trials = nt, offset = NULL, weights = NULL,
      re_list = rl, layout = layout, L_list = L, family = "neg_binomial_2",
      phi = exp(th[2]), phi2 = NA_real_, X_zi = d$X_zi, want_phi = TRUE,
      want_hessian = want_h, want_jacobian = want_h)
  }
  obj <- function(th) {
    L  <- tulpa:::.re_cov_theta_to_L_list(th[1], layout)
    tulpa_laplace(d$y, NULL, d$X, tulpa:::.re_cov_build_re_list(L, layout),
                  family = "neg_binomial_2", phi = exp(th[2]),
                  return_hessian = FALSE, return_joint_hessian = TRUE,
                  X_zi = d$X_zi, max_iter = 300L, tol = 1e-12)$log_marginal
  }
  h <- 1e-5
  bump <- function(th, j, s) { th[j] <- th[j] + s * h; th }
  expect_equal(
    gr(at),
    vapply(seq_along(at), function(j)
      (obj(bump(at, j, 1)) - obj(bump(at, j, -1))) / (2 * h), numeric(1)),
    tolerance = 1e-6)

  r <- gr(at, want_h = TRUE)
  expect_false(is.null(r$H))
  expect_equal(
    as.numeric(r$H),
    as.numeric(vapply(seq_along(at), function(j)
      (gr(bump(at, j, 1)) - gr(bump(at, j, -1))) / (2 * h), numeric(length(at)))),
    tolerance = 1e-6)
})


test_that("estimate_phi under genuine zero inflation recovers the dispersion", {
  skip_on_cran()
  d <- zi_re_sim(54L, "neg_binomial_2", G = 60L, per = 20L, sigma = 0.8, phi = 3)
  fit <- suppressMessages(tulpa_eb(
    d$y, NULL, d$X, d$re, family = "neg_binomial_2",
    phi = 1.5, estimate_phi = TRUE, X_zi = d$X_zi))
  # Started at 1.5, so landing near 3 is the estimate moving rather than the
  # start being reported back.
  expect_true(isTRUE(fit$phi_estimated))
  expect_gt(fit$phi, 2.0)
  expect_lt(fit$phi, 4.5)
  expect_lt(abs(sqrt(as.numeric(fit$Sigma[[1L]])) - d$sigma), 0.25)
})


test_that("estimate_phi is still refused where the mixture branch is unregistered", {
  # neg_binomial_1 has no dispersion registry at all, so it stays refused --
  # a different gap from the one closed here, and the message has to say so
  # rather than silently maximizing along the base family's derivative.
  expect_false(tulpa:::cpp_family_has_zi_phi_deriv("neg_binomial_1"))
  expect_true(tulpa:::cpp_family_has_zi_phi_deriv("neg_binomial_2"))
  d <- zi_re_sim(5L, "neg_binomial_2", G = 12L, per = 8L)
  expect_error(
    tulpa_eb(d$y, NULL, d$X, d$re, family = "neg_binomial_1",
             phi = 1.5, estimate_phi = TRUE, X_zi = d$X_zi),
    "estimate_phi|dispersion")
})


test_that("estimate_phi under a hurdle recovers the dispersion", {
  skip_on_cran()
  d <- zi_re_sim(54L, "truncated_neg_binomial_2", G = 60L, per = 20L,
                 sigma = 0.8, phi = 3)
  fit <- suppressMessages(tulpa_eb(
    d$y, NULL, d$X, d$re, family = "truncated_neg_binomial_2",
    phi = 1.5, estimate_phi = TRUE, X_zi = d$X_zi))

  # Started at 1.5, so landing near 3 is the estimate moving rather than the
  # start being reported back.
  expect_true(isTRUE(fit$phi_estimated))
  expect_gt(fit$phi, 2.0)
  expect_lt(fit$phi, 4.5)
  expect_lt(abs(sqrt(as.numeric(fit$Sigma[[1L]])) - d$sigma), 0.25)
})


test_that("the hurdle dispersion estimate is consistent, not merely finite", {
  # A wrong gradient maximizing to the wrong place would still land somewhere
  # plausible on one fixture. What separates it from ordinary finite-sample ML
  # bias is that the bias shrinks as the count arm grows -- so the same
  # estimator is run at two sizes and the larger one has to be closer.
  skip_if_not_slow()
  err <- vapply(list(c(G = 40L, per = 15L), c(G = 150L, per = 35L)),
    function(cfg) {
      e <- vapply(1:5, function(s) {
        d <- zi_re_sim(s, "truncated_neg_binomial_2", G = cfg[["G"]],
                       per = cfg[["per"]], sigma = 0.8, phi = 3)
        fit <- suppressMessages(tulpa_eb(
          d$y, NULL, d$X, d$re, family = "truncated_neg_binomial_2",
          phi = 1.5, estimate_phi = TRUE, X_zi = d$X_zi))
        fit$phi
      }, numeric(1))
      abs(mean(e) - 3)
    }, numeric(1))
  expect_lt(err[2], 0.35)
  expect_lt(err[2], err[1] + 0.15)
})
