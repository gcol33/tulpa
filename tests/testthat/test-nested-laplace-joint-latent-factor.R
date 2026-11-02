# Latent-factor block (`type = "lf"`, BILINEAR_FACTOR).
#
# Each obs i of arm k contributes eta_i += sum_f u_{l(i), f} lambda_{k, f}, the
# scores u_l ~ N(0, I) and the loadings in the inner solve's latent vector.
# The inner solve integrates the scores out unit by unit (Laplace), so the
# loadings and the fixed effects sit at the mode of their score-marginal
# posterior. The reference below is that posterior computed directly: an R
# optimisation of the per-unit Laplace marginal over (beta, lambda).

skip_on_cran()

.lf_sim_pois <- function(seed, S, lam, b0, b1) {
    set.seed(seed)
    K <- length(lam)
    u <- rnorm(S)
    x <- rnorm(S)
    Y <- sapply(seq_len(K), function(k) rpois(S, exp(b0[k] + b1[k] * x + lam[k] * u)))
    resp <- lapply(seq_len(K), function(k) list(
        y = Y[, k], n_trials = rep(1L, S), X = cbind(1, x = x), family = "poisson"))
    names(resp) <- paste0("sp", seq_len(K))
    list(S = S, K = K, u = u, x = x, Y = Y, lam = lam, responses = resp,
         obs_idx = rep(list(seq_len(S)), K))
}

# Laplace marginal of a one-factor Poisson model with u_l integrated per unit:
# theta = (b0, b1, lambda), priors N(0, s^2) on b (the joint driver's default
# fixed-effect prior, DEFAULT_TAU_BETA = 1 / s^2) and N(0, 1) on lambda.
.lf_ref_log_marg <- function(th, d) {
    K <- d$K
    b0 <- th[1:K]; b1 <- th[K + 1:K]; lam <- th[2 * K + 1:K]
    E0 <- outer(d$x, b1) + matrix(b0, d$S, K, byrow = TRUE)
    u <- rep(0, d$S)
    for (it in 1:50) {
        mu <- exp(E0 + outer(u, lam))
        u <- u + (as.numeric((d$Y - mu) %*% lam) - u) / (as.numeric(mu %*% lam^2) + 1)
    }
    E <- E0 + outer(u, lam)
    h <- as.numeric(exp(E) %*% lam^2) + 1
    sum(rowSums(d$Y * E - exp(E) - lgamma(d$Y + 1)) - u^2 / 2 - 0.5 * log(h)) +
        sum(dnorm(lam, 0, 1, log = TRUE)) +
        sum(dnorm(c(b0, b1), 0, default_sigma_beta(), log = TRUE))
}

.lf_ref_fit <- function(d) {
    th0 <- c(log(colMeans(d$Y) + 0.1), rep(0, d$K), rep(0.5, d$K))
    o <- optim(th0, .lf_ref_log_marg, d = d, method = "BFGS",
               control = list(fnscale = -1, maxit = 1000, reltol = 1e-14))
    li <- 2 * d$K + seq_len(d$K)
    if (o$par[li[1]] < 0) o$par[li] <- -o$par[li]
    H <- optimHess(o$par, .lf_ref_log_marg, d = d)
    list(par = o$par, se = sqrt(diag(solve(-H))))
}

.lf_fit <- function(d, n_factors = 1L, ...) {
    blk <- list(type = "lf", n_latent = d$S, obs_idx = d$obs_idx, ...)
    tulpa_nested_laplace_joint(
        responses = d$responses, prior = rep(list(blk), n_factors),
        control = list(n_threads = 1L, store_Q = TRUE, progress = FALSE,
                       diagnose_k = FALSE))
}

.lf_lam_true <- c(0.9, 0.7, -0.6, 0.3, -0.8)
.lf_b0 <- c(1.0, 0.5, 1.2, 0.2, 0.8)
.lf_b1 <- c(0.3, -0.2, 0.5, 0.0, 0.4)

test_that("lf block: loadings and fixed effects are the score-marginal mode", {
    skip_if_fast()
    d <- .lf_sim_pois(3L, S = 60L, .lf_lam_true, .lf_b0, .lf_b1)
    fit <- .lf_fit(d)
    expect_true(isTRUE(fit$converged))
    expect_equal(fit$arm_layout$block_size[1], d$S + d$K)

    ref <- .lf_ref_fit(d)
    m   <- fit$modes[1L, ]
    bs  <- fit$arm_layout$block_start[1] + 1L
    lc  <- bs + d$S + seq_len(d$K) - 1L
    beta <- unlist(lapply(seq_len(d$K), function(k)
        m[fit$arm_layout$beta_start[k] + 1:2]))
    li <- 2 * d$K + seq_len(d$K)
    expect_equal(unname(m[lc]), ref$par[li], tolerance = 1e-4)
    expect_equal(unname(beta[c(TRUE, FALSE)]), ref$par[1:d$K], tolerance = 1e-4)
    expect_gt(m[lc[1]], 0)

    # The loadings' posterior spread is the curvature of the same marginal.
    dr <- tulpa_posterior_draws(fit, n = 4000)
    sd_fit <- apply(dr[, lc], 2, sd)
    expect_lt(max(abs(sd_fit / ref$se[li] - 1)), 0.1)
})

test_that("lf block: scores keep their unit spread and loadings are not inflated", {
    skip_if_fast()
    d <- .lf_sim_pois(11L, S = 200L, .lf_lam_true, .lf_b0, .lf_b1)
    fit <- .lf_fit(d)
    m  <- fit$modes[1L, ]
    bs <- fit$arm_layout$block_start[1] + 1L
    uc <- bs + seq_len(d$S) - 1L
    lc <- bs + d$S + seq_len(d$K) - 1L
    # The joint mode over (u, lambda) read 1.2-1.3 here with modal scores of
    # SD 0.7-0.8.
    slope <- unname(coef(lm(m[lc] ~ 0 + d$lam)))
    expect_gt(slope, 0.85)
    expect_lt(slope, 1.10)
    # Score modes plus their conditional variances add up to the unit prior.
    v_u <- apply(tulpa_posterior_draws(fit, n = 2000)[, uc], 2, var)
    expect_equal(mean(m[uc]^2 + v_u), 1, tolerance = 0.15)
})

test_that("lf block: mixed-family arms recover the loading ratio", {
    skip_if_fast()
    set.seed(42L)
    n_s <- 30L; N <- 400L
    u <- rnorm(n_s)
    idx1 <- sample.int(n_s, N, replace = TRUE); idx2 <- sample.int(n_s, N, replace = TRUE)
    X1 <- cbind(1, rnorm(N)); X2 <- cbind(1, rnorm(N))
    lam <- c(0.8, 1.2)
    y1 <- rbinom(N, 1L, plogis(X1 %*% c(-0.2, 0.4) + lam[1] * u[idx1]))
    y2 <- rnorm(N, X2 %*% c(0.1, -0.3) + lam[2] * u[idx2], 0.4)
    resp <- list(
        occ = list(y = y1, n_trials = rep(1L, N), X = X1, family = "binomial"),
        pos = list(y = y2, n_trials = rep(1L, N), X = X2, family = "gaussian",
                   phi = 0.4^2))
    fit <- tulpa_nested_laplace_joint(
        responses = resp,
        prior = list(list(type = "lf", n_latent = n_s, obs_idx = list(idx1, idx2),
                          sigma_lambda = 2)),
        control = list(n_threads = 1L, progress = FALSE, diagnose_k = FALSE))
    expect_true(isTRUE(fit$converged))
    m  <- fit$modes[1L, ]
    bs <- fit$arm_layout$block_start[1] + 1L
    lam_hat <- m[bs + n_s + 0:1]
    expect_gt(lam_hat[1], 0)
    expect_lt(abs(lam_hat[2] / lam_hat[1] - lam[2] / lam[1]), 0.3)
    expect_gt(cor(m[bs + seq_len(n_s) - 1L], u), 0.8)
})

test_that("two lf blocks: lower-triangular loadings, converged, stationary for the marginal", {
    skip_if_fast()
    set.seed(5L)
    S <- 80L; K <- 5L
    U <- cbind(rnorm(S), rnorm(S))
    Lam <- cbind(c(1, 0.8, -0.6, 0.4, 0.7), c(0, 0.9, 0.6, -0.7, 0.3))
    x <- rnorm(S)
    Y <- sapply(seq_len(K), function(k) rpois(S, exp(0.5 + 0.2 * x + U %*% Lam[k, ])))
    resp <- lapply(seq_len(K), function(k) list(
        y = Y[, k], n_trials = rep(1L, S), X = cbind(1, x = x), family = "poisson"))
    names(resp) <- paste0("sp", seq_len(K))
    blk <- list(type = "lf", n_latent = S, obs_idx = rep(list(seq_len(S)), K))
    fit <- tulpa_nested_laplace_joint(
        responses = resp, prior = list(blk, blk),
        control = list(n_threads = 1L, progress = FALSE, diagnose_k = FALSE))
    expect_true(isTRUE(fit$converged))
    expect_lt(fit$score_max, 1e-3)
    # Factor 2 carries no loading on arm 1.
    expect_equal(fit$arm_layout$block_size, c(S + K, S + K - 1L))

    m  <- fit$modes[1L, ]
    bs <- fit$arm_layout$block_start + 1L
    L  <- matrix(0, K, 2)
    L[, 1] <- m[bs[1] + S + 0:(K - 1)]
    L[2:K, 2] <- m[bs[2] + S + 0:(K - 2)]
    expect_gt(L[1, 1], 0)
    expect_gt(L[2, 2], 0)

    # Score-marginal log density over (b, lambda), per-unit 2-D Laplace.
    beta <- unlist(lapply(seq_len(K), function(k) m[fit$arm_layout$beta_start[k] + 1:2]))
    lmarg <- function(th) {
        b0 <- th[1:K]; b1 <- th[K + 1:K]
        Lm <- matrix(0, K, 2); Lm[, 1] <- th[2 * K + 1:K]; Lm[2:K, 2] <- th[3 * K + 1:(K - 1)]
        tot <- 0
        for (i in seq_len(S)) {
            e0 <- b0 + b1 * x[i]; u <- c(0, 0)
            for (it in 1:40) {
                mu <- exp(e0 + as.numeric(Lm %*% u))
                H <- crossprod(Lm, mu * Lm) + diag(2)
                u <- u + solve(H, as.numeric(crossprod(Lm, Y[i, ] - mu)) - u)
            }
            eta <- e0 + as.numeric(Lm %*% u); mu <- exp(eta)
            H <- crossprod(Lm, mu * Lm) + diag(2)
            tot <- tot + sum(Y[i, ] * eta - mu) - sum(u^2) / 2 -
                0.5 * as.numeric(determinant(H)$modulus)
        }
        lv <- th[-(1:(2 * K))]
        tot - sum(lv^2) / 2 -
            sum(th[1:(2 * K)]^2) / (2 * default_sigma_beta()^2)
    }
    th <- c(beta[c(TRUE, FALSE)], beta[c(FALSE, TRUE)], L[, 1], L[2:K, 2])
    g <- vapply(seq_along(th), function(j) {
        e <- 1e-5 * max(1, abs(th[j])); tp <- th; tm <- th
        tp[j] <- tp[j] + e; tm[j] <- tm[j] - e
        (lmarg(tp) - lmarg(tm)) / (2 * e)
    }, numeric(1))
    expect_lt(max(abs(g)), 1e-3)
})

test_that("lf block: recovery and loading coverage over seeds", {
    skip_if_not_slow()
    lam <- c(1, 0.9, 0.7, -0.6, 0.3, 0, -0.8, 0.5)
    b0 <- c(1, 0.5, 1.2, 0.2, 0.8, 1.5, 0.3, 0.6)
    b1 <- c(0.3, -0.2, 0.5, 0, 0.4, -0.3, 0.2, 0.1)
    out <- t(vapply(1:10, function(seed) {
        d <- .lf_sim_pois(seed, S = 200L, lam, b0, b1)
        fit <- .lf_fit(d)
        bs <- fit$arm_layout$block_start[1] + 1L
        lc <- bs + d$S + seq_len(d$K) - 1L
        q <- apply(tulpa_posterior_draws(fit, n = 2000)[, lc], 2, quantile,
                   c(0.025, 0.975))
        c(slope = unname(coef(lm(fit$modes[1L, lc] ~ 0 + lam))),
          cover = mean(q[1, ] <= lam & lam <= q[2, ]))
    }, numeric(2)))
    expect_gt(mean(out[, "slope"]), 0.92)
    expect_lt(mean(out[, "slope"]), 1.05)
    expect_gt(mean(out[, "cover"]), 0.85)
})

test_that("lf block raises clean errors on bad spec", {
    d <- .lf_sim_pois(43L, S = 20L, .lf_lam_true[1:2], .lf_b0[1:2], .lf_b1[1:2])
    run <- function(prior) tulpa_nested_laplace_joint(
        responses = d$responses, prior = prior,
        control = list(max_iter = 5L, n_threads = 1L, progress = FALSE))
    blk <- list(type = "lf", n_latent = d$S, obs_idx = d$obs_idx)
    expect_error(run(list(utils::modifyList(blk, list(n_latent = 1L)))),
                 "n_latent.*at least 2")
    expect_error(run(list(blk[c("type", "obs_idx")])), "n_latent")
    expect_error(run(list(c(blk, sigma_u = 2))), "sigma_u")
    expect_error(run(list(c(blk, anchor_eps = 1e-3))), "anchor_eps")
    expect_error(run(list(c(blk, first_arm = 1L))), "first_arm")
    other <- blk; other$obs_idx <- list(rev(seq_len(d$S)), seq_len(d$S))
    expect_error(run(list(blk, other)), "different units")
    expect_error(run(list(blk, blk, blk)), "at most as many factors as arms")
})
