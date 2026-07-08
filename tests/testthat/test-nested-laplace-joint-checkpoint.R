# Grid-cell checkpoint/resume for tulpa_nested_laplace_joint (gcol33/tulpa#50).
#
# The outer grid is the checkpoint boundary: each completed cell is appended to
# a file, and a resume run loads finished cells and solves only the rest. The
# loaded path must be exactly equivalent to a from-scratch fit (a resumed cell
# is the same self-contained LaplaceResult), a torn tail from a killed run must
# be discarded and re-solved, and a file written for different data must be
# rejected rather than resumed onto a stale result.

skip_on_cran()

.ck_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(
        adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
        adj_col_idx = as.integer(unlist(nbr)) - 1L,
        n_neighbors = as.integer(n_neighbors),
        n_spatial_units = n_s
    )
}

.ck_sim_joint <- function(seed = 42L, N = 160L, n_s = 20L,
                          sigma = 0.6, alpha_true = 1.2) {
    set.seed(seed)
    adj <- .ck_chain_adj(n_s)
    w <- as.numeric(arima.sim(n = n_s, list(ar = 0.6))) * sigma
    w <- w - mean(w)

    s_idx_1 <- sample(seq_len(n_s), N, replace = TRUE)
    s_idx_2 <- sample(seq_len(n_s), N, replace = TRUE)
    x1 <- rnorm(N); x2 <- rnorm(N)

    eta1 <- 0.3 + 0.5 * x1 + w[s_idx_1]
    eta2 <- -0.2 + 0.4 * x2 + alpha_true * w[s_idx_2]

    y1 <- rbinom(N, 1L, plogis(eta1))
    y2 <- rnorm(N, mean = eta2, sd = 0.4)

    list(
        adj = adj,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x1),
                       spatial_idx = s_idx_1,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0, family = "binomial", phi = 1.0),
            pos = list(y = as.numeric(y2), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x2),
                       spatial_idx = s_idx_2,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0, family = "gaussian", phi = 0.4)
        )
    )
}

.ck_fit <- function(sim, prior, copy = NULL, control = list()) {
    base <- list(max_iter = 50L, tol = 1e-8, n_threads = 1L,
                 verbose = FALSE, diagnose_k = FALSE)
    responses <- sim$responses
    if (!is.null(copy))
        responses[[copy$arm]]$field_coef <- list(name = "alpha", grid = copy$alpha_grid)
    tulpa_nested_laplace_joint(
        responses = responses, prior = prior,
        control = utils::modifyList(base, control)
    )
}

.ck_expect_equiv <- function(fit_a, fit_b, lm_tol = 1e-9, mode_tol = 1e-9) {
    expect_equal(as.numeric(fit_b$log_marginal),
                 as.numeric(fit_a$log_marginal),
                 tolerance = lm_tol,
                 info = "log_marginal differs")
    if (is.matrix(fit_a$modes) && is.matrix(fit_b$modes)) {
        expect_equal(dim(fit_b$modes), dim(fit_a$modes))
        expect_equal(as.numeric(fit_b$modes), as.numeric(fit_a$modes),
                     tolerance = mode_tol, info = "modes differ")
    }
    expect_equal(as.numeric(fit_b$theta_mean),
                 as.numeric(fit_a$theta_mean),
                 tolerance = 1e-9, info = "theta_mean differs")
}

test_that("checkpoint fit equals an un-checkpointed fit (ICAR, no copy)", {
    sim <- .ck_sim_joint(seed = 11L)
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.6, 0.8, 1.0)),
               sim$adj)
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    fit_plain <- .ck_fit(sim, prior)
    fit_ckpt  <- .ck_fit(sim, prior,
                         control = list(checkpoint = list(path = path,
                                                          resume = FALSE)))
    .ck_expect_equiv(fit_plain, fit_ckpt)
    expect_true(file.exists(path))
    expect_gt(file.size(path), 16)  # header (magic + fingerprint) + >= 1 record
})

test_that("resume loads completed cells and re-solves nothing", {
    sim <- .ck_sim_joint(seed = 12L)
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.6, 0.8, 1.0)),
               sim$adj)
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    fit1 <- .ck_fit(sim, prior,
                    control = list(checkpoint = list(path = path,
                                                     resume = FALSE)))
    size_after_1 <- file.size(path)

    # Resume: every cell coordinate is already on disk, so the driver loads all
    # of them and appends nothing. Identical result, unchanged file size.
    fit2 <- .ck_fit(sim, prior,
                    control = list(checkpoint = list(path = path,
                                                     resume = TRUE)))
    .ck_expect_equiv(fit1, fit2)
    expect_equal(file.size(path), size_after_1,
                 info = "resume re-appended cells it should have loaded")
})

test_that("a torn tail (killed mid-write) is discarded and re-solved", {
    sim <- .ck_sim_joint(seed = 13L)
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.6, 0.8, 1.0)),
               sim$adj)
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    fit_full <- .ck_fit(sim, prior,
                        control = list(checkpoint = list(path = path,
                                                         resume = FALSE)))
    full_bytes <- readBin(path, "raw", n = file.size(path))

    # Simulate a process killed partway: keep the header plus part of the
    # records, chopping the final record mid-payload. The loader must stop at
    # the torn boundary (checksum / short read), keep the intact records, and
    # the resume must re-solve the rest to the same answer.
    keep <- as.integer(length(full_bytes) * 0.6)
    torn <- full_bytes[seq_len(keep)]
    writeBin(torn, path)

    fit_resumed <- .ck_fit(sim, prior,
                           control = list(checkpoint = list(path = path,
                                                            resume = TRUE)))
    .ck_expect_equiv(fit_full, fit_resumed)
})

test_that("resume onto a different dataset is rejected (fingerprint)", {
    sim_a <- .ck_sim_joint(seed = 14L)
    sim_b <- .ck_sim_joint(seed = 99L)  # different responses
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.6, 0.8)), sim_a$adj)
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    .ck_fit(sim_a, prior,
            control = list(checkpoint = list(path = path, resume = FALSE)))
    expect_error(
        .ck_fit(sim_b, prior,
                control = list(checkpoint = list(path = path, resume = TRUE))),
        "fingerprint"
    )
})

test_that("resume = FALSE starts over rather than resuming stale data", {
    sim_a <- .ck_sim_joint(seed = 15L)
    sim_b <- .ck_sim_joint(seed = 77L)
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.6, 0.8)), sim_a$adj)
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    .ck_fit(sim_a, prior,
            control = list(checkpoint = list(path = path, resume = FALSE)))
    # A fresh (resume = FALSE) run on different data removes the old file first,
    # so no fingerprint error and the result matches a plain fit of sim_b.
    fit_b_plain <- .ck_fit(sim_b, prior)
    fit_b_fresh <- .ck_fit(sim_b, prior,
                           control = list(checkpoint = list(path = path,
                                                            resume = FALSE)))
    .ck_expect_equiv(fit_b_plain, fit_b_fresh)
})

test_that("checkpoint is equivalent on the parallel outer-grid path", {
    sim <- .ck_sim_joint(seed = 16L)
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.6, 0.8, 1.0, 1.2)),
               sim$adj)
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    fit_plain <- .ck_fit(sim, prior, control = list(n_threads_outer = 2L))
    fit_ckpt  <- .ck_fit(sim, prior,
                         control = list(n_threads_outer = 2L,
                                        checkpoint = list(path = path,
                                                          resume = FALSE)))
    .ck_expect_equiv(fit_plain, fit_ckpt)

    # Half the records survive a kill; the parallel resume re-solves the rest.
    full_bytes <- readBin(path, "raw", n = file.size(path))
    keep <- as.integer(length(full_bytes) * 0.5)
    writeBin(full_bytes[seq_len(keep)], path)
    fit_resumed <- .ck_fit(sim, prior,
                           control = list(n_threads_outer = 2L,
                                          checkpoint = list(path = path,
                                                            resume = TRUE)))
    .ck_expect_equiv(fit_plain, fit_resumed)
})

test_that("checkpoint resume is equivalent on the sparse grid path", {
    sim <- .ck_sim_joint(seed = 17L, alpha_true = 1.1)
    prior <- c(list(type = "bym2",
                    sigma_grid = c(0.5, 0.7),
                    rho_grid   = c(0.5, 0.8)),
               sim$adj)
    copy_spec <- list(arm = "pos", alpha_grid = c(1.0, 1.2))
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)

    fit_plain <- .ck_fit(sim, prior, copy = copy_spec,
                         control = list(force_sparse = TRUE))
    fit_full  <- .ck_fit(sim, prior, copy = copy_spec,
                         control = list(force_sparse = TRUE,
                                        checkpoint = list(path = path,
                                                          resume = FALSE)))
    .ck_expect_equiv(fit_plain, fit_full)

    full_bytes <- readBin(path, "raw", n = file.size(path))
    keep <- as.integer(length(full_bytes) * 0.55)
    writeBin(full_bytes[seq_len(keep)], path)
    fit_resumed <- .ck_fit(sim, prior, copy = copy_spec,
                           control = list(force_sparse = TRUE,
                                          checkpoint = list(path = path,
                                                            resume = TRUE)))
    .ck_expect_equiv(fit_plain, fit_resumed)
})

test_that("control$checkpoint validates its argument", {
    sim <- .ck_sim_joint(seed = 18L)
    prior <- c(list(type = "icar", sigma_grid = c(0.5, 0.7)), sim$adj)
    expect_error(
        .ck_fit(sim, prior, control = list(checkpoint = list(resume = TRUE))),
        "path"
    )
    expect_error(
        .ck_fit(sim, prior,
                control = list(checkpoint = list(path = tempfile(),
                                                 resume = "yes"))),
        "resume"
    )
})
