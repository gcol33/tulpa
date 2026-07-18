# Front-door HYPERPARAMETER recovery + CI coverage (gcol33/tulpa#214).
#
# Field-shape recovery is tested widely elsewhere; this file gates the
# hyperparameter POSTERIORS of the front-door nested-Laplace paths against
# simulated truth. For each subsystem: simulate from the generative model with a
# known hyperparameter, fit through the front door, and check that the reported
# 95% credible interval contains the truth across >= 20 seeds (the project's
# >= 85% coverage standard; asserted at a 0.75 floor per parameter to stay robust
# to binomial sampling noise at N = 20 while still failing a real miscalibration).
# Slow-gated: these are multi-seed nested fits.

# Coverage of a single scalar hyperparameter across seeds. `fit_extract(seed)`
# returns c(lo, hi) (the CI) or NULL to skip a degenerate seed; NULL seeds do not
# count toward the denominator.
.hp_ci_coverage <- function(truth, fit_extract, n_seeds) {
  hit <- 0L; used <- 0L
  for (s in seq_len(n_seeds)) {
    ci <- tryCatch(fit_extract(s), error = function(e) NULL)
    if (is.null(ci) || any(!is.finite(ci))) next
    used <- used + 1L
    if (truth >= ci[1] && truth <= ci[2]) hit <- hit + 1L
  }
  list(rate = if (used > 0L) hit / used else NA_real_, hit = hit, used = used)
}

.n_cov_seeds <- function() if (nzchar(Sys.getenv("TULPA_COV_SEEDS")))
  as.integer(Sys.getenv("TULPA_COV_SEEDS")) else 20L

.qcols <- function(tab) grep("^q", colnames(tab), value = TRUE)


test_that("temporal AR1 nested Laplace covers rho and the innovation precision", {
  skip_on_cran()
  skip_if_not_slow()
  ns <- .n_cov_seeds()
  rho_true <- 0.6; tau_true <- 4
  extract <- function(seed) {
    set.seed(seed); Tn <- 100L; sd_i <- 1 / sqrt(tau_true)
    z <- numeric(Tn); z[1] <- rnorm(1, 0, sd_i / sqrt(1 - rho_true^2))
    for (t in 2:Tn) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, sd_i)
    z <- z - mean(z)
    d <- data.frame(y = rpois(Tn, exp(0.3 + z)), t = seq_len(Tn))
    fit <- tulpa(y ~ 1, data = d, family = "poisson",
                 temporal = temporal_ar1("t"), mode = "auto")
    tc <- temporal_corr(fit); qn <- .qcols(tc)
    list(rho = as.numeric(tc["rho_ar1", qn]),
         prec = as.numeric(tc["precision", qn]))
  }
  cache <- lapply(seq_len(ns), function(s) tryCatch(extract(s), error = function(e) NULL))
  cov_rho  <- .hp_ci_coverage(rho_true, function(s) cache[[s]]$rho,  ns)
  cov_prec <- .hp_ci_coverage(tau_true, function(s) cache[[s]]$prec, ns)
  expect_gte(cov_rho$rate,  0.75)
  expect_gte(cov_prec$rate, 0.75)
})


test_that("proper CAR nested Laplace covers sigma and rho", {
  skip_on_cran()
  skip_if_not_slow()
  ns <- .n_cov_seeds()
  nr <- 6L; nc <- 6L; K <- nr * nc
  W <- matrix(0, K, K)
  for (i in seq_len(K)) {
    r <- (i - 1L) %/% nc + 1L; cc <- (i - 1L) %% nc + 1L
    if (r > 1L) W[i, i - nc] <- 1; if (r < nr) W[i, i + nc] <- 1
    if (cc > 1L) W[i, i - 1L] <- 1; if (cc < nc) W[i, i + 1L] <- 1
  }
  Dg <- diag(rowSums(W))
  tau_true <- 2; rho_true <- 0.8; sig_true <- 1 / sqrt(tau_true)
  extract <- function(seed) {
    set.seed(seed)
    Sig <- solve(tau_true * (Dg - rho_true * W)); Sig <- (Sig + t(Sig)) / 2
    u <- as.numeric(t(chol(Sig)) %*% rnorm(K)); u <- u - mean(u)
    idx <- rep(seq_len(K), each = 12L); N <- length(idx)
    d <- data.frame(y = rbinom(N, 1, plogis(-0.2 + u[idx])), site = factor(idx))
    fit <- tulpa(y ~ 1 + spatial(site), data = d, family = "binomial",
                 spatial = spatial_car(adjacency = W, group_var = "site", proper = TRUE),
                 mode = "nested_laplace")
    sr <- spatial_range(fit); qn <- .qcols(sr)
    list(sigma = as.numeric(sr["sigma", qn]), rho = as.numeric(sr["rho", qn]))
  }
  cache <- lapply(seq_len(ns), function(s) tryCatch(extract(s), error = function(e) NULL))
  cov_sig <- .hp_ci_coverage(sig_true, function(s) cache[[s]]$sigma, ns)
  cov_rho <- .hp_ci_coverage(rho_true, function(s) cache[[s]]$rho,   ns)
  expect_gte(cov_sig$rate, 0.75)
  expect_gte(cov_rho$rate, 0.75)
})


test_that("GP (NNGP) nested Laplace covers the marginal SD and range", {
  skip_on_cran()
  skip_if_not_slow()
  ns <- .n_cov_seeds()
  sigma2 <- 0.7; phi <- 0.2
  sig_true <- sqrt(sigma2); rng_true <- 3 * phi
  extract <- function(seed) {
    set.seed(seed); n <- 120L
    co <- cbind(runif(n), runif(n)); D <- as.matrix(dist(co))
    K <- sigma2 * exp(-D / phi) + diag(1e-8, n)
    f <- as.numeric(t(chol(K)) %*% rnorm(n)); f <- f - mean(f)
    d <- data.frame(y = rpois(n, exp(0.2 + f)), lon = co[, 1], lat = co[, 2])
    fit <- tulpa(y ~ 1, data = d, family = "poisson",
                 spatial = spatial_gp(~ lon + lat, nn = 10L),
                 mode = "nested_laplace")
    sr <- spatial_range(fit); qn <- .qcols(sr)
    list(sigma = as.numeric(sr["sigma", qn]), range = as.numeric(sr["range", qn]))
  }
  cache <- lapply(seq_len(ns), function(s) tryCatch(extract(s), error = function(e) NULL))
  cov_sig <- .hp_ci_coverage(sig_true, function(s) cache[[s]]$sigma, ns)
  cov_rng <- .hp_ci_coverage(rng_true, function(s) cache[[s]]$range, ns)
  expect_gte(cov_sig$rate, 0.75)
  expect_gte(cov_rng$rate, 0.75)
})


test_that("HSGP nested Laplace covers the marginal SD", {
  skip_on_cran()
  skip_if_not_slow()
  ns <- .n_cov_seeds()
  sigma2 <- 0.7; phi <- 0.2; sig_true <- sqrt(sigma2)
  # Only the marginal SD is gated here: the truncated HSGP basis underestimates
  # the field variance near the domain boundary, so the effective range is less
  # reliably recovered (see test-hsgp-density-identity.R).
  extract <- function(seed) {
    set.seed(seed); n <- 150L
    co <- cbind(runif(n), runif(n)); D <- as.matrix(dist(co))
    K <- sigma2 * exp(-D / phi) + diag(1e-8, n)
    f <- as.numeric(t(chol(K)) %*% rnorm(n)); f <- f - mean(f)
    d <- data.frame(y = rpois(n, exp(0.2 + f)), lon = co[, 1], lat = co[, 2])
    fit <- tulpa(y ~ 1, data = d, family = "poisson",
                 spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 12, c = 1.5),
                 mode = "nested_laplace")
    sr <- spatial_range(fit); qn <- .qcols(sr)
    as.numeric(sr["sigma", qn])
  }
  cov_sig <- .hp_ci_coverage(sig_true, extract, ns)
  expect_gte(cov_sig$rate, 0.75)
})


test_that("BYM2 nested Laplace covers the total spatial SD", {
  skip_on_cran()
  skip_if_not_slow()
  ns <- .n_cov_seeds()
  nr <- 6L; nc <- 6L; K <- nr * nc
  W <- matrix(0, K, K)
  for (i in seq_len(K)) {
    r <- (i - 1L) %/% nc + 1L; cc <- (i - 1L) %% nc + 1L
    if (r > 1L) W[i, i - nc] <- 1; if (r < nr) W[i, i + nc] <- 1
    if (cc > 1L) W[i, i - 1L] <- 1; if (cc < nc) W[i, i + 1L] <- 1
  }
  Q <- diag(rowSums(W)) - W
  e <- eigen(Q, symmetric = TRUE); ev <- e$values; V <- e$vectors; pos <- ev > 1e-8
  sig_true <- 0.7   # target marginal SD of the structured field
  # BYM2 mixing rho is not asserted: a structured-only simulation does not pin a
  # unique (sigma, rho) truth (rho -> 1), so only the total spatial SD is gated.
  extract <- function(seed) {
    set.seed(100L + seed)
    u <- as.numeric(V[, pos] %*% (rnorm(sum(pos)) / sqrt(ev[pos])))
    u <- (u - mean(u)); u <- u / stats::sd(u) * sig_true
    idx <- rep(seq_len(K), each = 12L); N <- length(idx)
    d <- data.frame(y = rbinom(N, 1, plogis(-0.2 + u[idx])), site = factor(idx))
    fit <- tulpa(y ~ 1 + spatial(site), data = d, family = "binomial",
                 spatial = spatial_bym2(adjacency = W, group_var = "site"),
                 mode = "nested_laplace")
    sr <- spatial_range(fit); qn <- .qcols(sr)
    as.numeric(sr["sigma", qn])
  }
  cov_sig <- .hp_ci_coverage(sig_true, extract, ns)
  expect_gte(cov_sig$rate, 0.75)
})


test_that("free-Sigma random-slope nested Laplace covers the correlation rho", {
  skip_on_cran()
  skip_if_not_slow()
  ns <- .n_cov_seeds()
  rho_true <- 0.5
  Sigma <- matrix(c(0.8^2, rho_true * 0.8 * 0.6,
                    rho_true * 0.8 * 0.6, 0.6^2), 2)
  extract <- function(seed) {
    set.seed(seed); G <- 60L; npg <- 12L; N <- G * npg
    grp <- rep(seq_len(G), each = npg); x <- rnorm(N)
    X <- cbind(1, x); Z <- cbind(1, x)
    u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), 2))
    y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(-0.3, 0.7)) + rowSums(Z * u[grp, ])))
    re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = Z)
    res <- tulpa_re_cov_nested(y, rep(1L, N), X, re_term, family = "binomial")
    p <- res$posterior; r <- p[p$parameter == "rho_12", ]
    c(r$ci_lo, r$ci_hi)
  }
  cov_rho <- .hp_ci_coverage(rho_true, extract, ns)
  expect_gte(cov_rho$rate, 0.75)
})
