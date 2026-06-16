# Latent-factor block (Stage 1.6a).
#
# The `lf` block introduces BILINEAR_FACTOR contrib_kind: each obs i in arm
# k_arm contributes eta_i += u[obs_idx(i)] * lambda[k_arm], with both u
# (length n_latent) and lambda (length n_arms) jointly optimized in the
# inner Newton. Identifiability handled by tight Gaussian anchors on
# (u_1, lambda_1) inside the C++ factory.
#
# Coverage:
#   (1) Smoke: 2-arm fit runs to convergence, log_marginal finite, latent
#       slots returned for both u and lambda.
#   (2) Recovery: simulate with known u_true and lambda_true, fit, check
#       that lambda_2 / lambda_1 ratio recovers the simulated relative
#       loading and that fitted u correlates with the truth.
#   (3) Validation errors: n_latent < 2 raises a clean error.

skip_on_cran()

.sim_lf <- function(seed = 41L, n_s = 30L, N1 = 200L, N2 = 200L,
                     lambda_true = c(1.0, 1.5),
                     sigma_u_true = 0.8,
                     gaussian_sd = 0.4) {
    set.seed(seed)
    u_true <- rnorm(n_s, 0, sigma_u_true)
    u_true[1] <- 0  # match the model's first-zero anchor
    s_idx_1 <- sample.int(n_s, N1, replace = TRUE)
    s_idx_2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))
    beta1 <- c(-0.2, 0.4)
    beta2 <- c( 0.1, -0.3)
    eta1 <- X1 %*% beta1 + lambda_true[1] * u_true[s_idx_1]
    eta2 <- X2 %*% beta2 + lambda_true[2] * u_true[s_idx_2]
    y1 <- rbinom(N1, 1L, plogis(eta1))
    y2 <- rnorm(N2, eta2, gaussian_sd)
    list(
        n_s = n_s, N1 = N1, N2 = N2,
        u_true = u_true, lambda_true = lambda_true,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N1),
                       X = X1, family = "binomial", phi = 1.0,
                       spatial_idx = s_idx_1,
                       re_idx = rep(0L, N1), n_re_groups = 0L,
                       sigma_re = 1.0),
            pos = list(y = as.numeric(y2), n_trials = rep(1L, N2),
                       X = X2, family = "gaussian", phi = gaussian_sd,
                       spatial_idx = s_idx_2,
                       re_idx = rep(0L, N2), n_re_groups = 0L,
                       sigma_re = 1.0)
        ),
        s_idx = list(s_idx_1, s_idx_2)
    )
}

# --------------------------------------------------------------------------- #
# (1) Smoke                                                                   #
# --------------------------------------------------------------------------- #

test_that("joint dispatch routes `type = 'lf'` and converges on a 2-arm fit", {
    sim <- .sim_lf(seed = 41L)
    prior <- list(
        list(
            type     = "lf",
            n_latent = sim$n_s,
            obs_idx  = sim$s_idx
        )
    )
    fit <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 60L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))
    # No outer-grid axes -> single cell.
    expect_equal(length(fit$log_marginal), 1L)
    # `modes` is [n_cells x n_x]. The latent vector holds per-arm beta
    # + per-arm RE + (u, lambda) — at least n_latent + n_arms slots.
    expect_true(!is.null(fit$modes) && ncol(fit$modes) >= sim$n_s + 2L)
    expect_equal(fit$arm_layout$block_size[1], sim$n_s + 2L)
})

# --------------------------------------------------------------------------- #
# (2) Recovery: lambda ratio + u correlation                                  #
# --------------------------------------------------------------------------- #

test_that("lf block recovers loading ratio and factor field up to sign", {
    sim <- .sim_lf(seed = 42L, n_s = 30L, N1 = 400L, N2 = 400L,
                    lambda_true = c(1.0, 1.5))
    prior <- list(
        list(
            type         = "lf",
            n_latent     = sim$n_s,
            obs_idx      = sim$s_idx,
            sigma_u      = 2.0,    # vague-ish on the factor field
            sigma_lambda = 2.0,    # vague-ish on the loadings
            anchor_eps   = 1e-3    # tight anchor on (u_1, lambda_1)
        )
    )
    fit <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L, verbose = FALSE)
    )
    expect_true(all(is.finite(fit$log_marginal)))

    # `modes` is [n_cells x n_x] (1 row in the lf-only case). Read off the
    # lf block from arm_layout$block_start (0-based offset into the row).
    mode_vec   <- as.numeric(fit$modes[1L, ])
    blk_start  <- fit$arm_layout$block_start[1] + 1L  # to 1-based
    u_hat      <- mode_vec[blk_start:(blk_start + sim$n_s - 1L)]
    lambda_hat <- mode_vec[(blk_start + sim$n_s):(blk_start + sim$n_s + 1L)]
    expect_equal(length(u_hat), sim$n_s)
    expect_equal(length(lambda_hat), 2L)

    # lambda_1 anchored at ~1 by the tight prior.
    expect_lt(abs(lambda_hat[1] - 1.0), 0.05)
    # lambda_2 / lambda_1 should track the simulated relative loading.
    ratio_hat  <- lambda_hat[2] / lambda_hat[1]
    ratio_true <- sim$lambda_true[2] / sim$lambda_true[1]
    expect_lt(abs(ratio_hat - ratio_true), 0.30)

    # u_hat correlates strongly with the true factor field on its
    # non-anchored entries (entries 2..n_s).
    rho <- cor(u_hat[-1], sim$u_true[-1])
    expect_gt(rho, 0.6)
})

# --------------------------------------------------------------------------- #
# (3) Validation                                                              #
# --------------------------------------------------------------------------- #

test_that("lf block raises clean errors on bad spec", {
    sim <- .sim_lf(seed = 43L)
    expect_error(
        tulpa_nested_laplace_joint(
            responses = sim$responses,
            prior = list(list(type = "lf", n_latent = 1L,
                              obs_idx = sim$s_idx)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "n_latent.*at least 2"
    )
    expect_error(
        tulpa_nested_laplace_joint(
            responses = sim$responses,
            prior = list(list(type = "lf",
                              obs_idx = sim$s_idx)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "n_latent"
    )
})
