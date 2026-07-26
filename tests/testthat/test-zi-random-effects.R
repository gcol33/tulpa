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
  # needs d/deta of the OBSERVED curvature. neg_binomial_1 is ZI-fittable but
  # has only the working-weight derivative registered, so it is refused rather
  # than served a plausible wrong number.
  expect_false(cpp_family_has_zi_curvature_derivative("neg_binomial_1"))
  expect_false(tulpa:::.laplace_exact_supports_zi("neg_binomial_1"))
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
  expect_error(
    tulpa_eb(d$y, NULL, d$X, d$re, family = "neg_binomial_2", X_zi = d$X_zi,
             estimate_phi = TRUE),
    "not available alongside a zero-inflation process")
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
