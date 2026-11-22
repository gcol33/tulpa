# A refinement round's cells are screened against the grid they join
# (gcol33/tulpa#948): the cheap pass that drops cells of the declared grid drops
# the cells a consistency or boundary pass lays where the posterior is not, with
# the cut read as a share of the merged grid's mass rather than of the round.

.rs_chain <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(type = "icar", n_spatial_units = n_s,
         adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn))
}

.rs_sim <- function(seed = 948L, N = 600L, n_s = 40L) {
    set.seed(seed)
    s_idx <- sample.int(n_s, N, replace = TRUE)
    w <- cumsum(rnorm(n_s, 0, 0.3)); w <- w - mean(w)
    x <- rnorm(N); X <- cbind(1, x)
    occ <- rbinom(N, 1, plogis(-0.2 + 0.5 * x + w[s_idx]))
    pos <- occ == 1L
    y_pos <- rnorm(sum(pos), 0.3 - 0.4 * x[pos] + 0.8 * w[s_idx[pos]], 0.5)
    list(prior = c(.rs_chain(n_s), list(sigma_grid = c(0.3, 0.6, 1.0, 1.6))),
         occ = list(y = as.numeric(occ), n_trials = rep(1L, N), X = X,
                    spatial_idx = s_idx, family = "binomial", phi = 1.0),
         pos = list(y = y_pos, n_trials = rep(1L, sum(pos)),
                    X = X[pos, , drop = FALSE], spatial_idx = s_idx[pos],
                    family = "gaussian", phi = 0.25,
                    field_coef = list(name = "alpha",
                                      grid = c(0.4, 0.8, 1.2, 1.6))))
}

# The residual variance is stated on three nodes far apart, so the consistency
# pass bisects it in every sigma x alpha row that holds the posterior.
.rs_fit <- function(sim, prune) {
    suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = sim$occ, pos = sim$pos), prior = sim$prior,
        phi_grid = list(pos = c(0.05, 0.4, 3)),
        control = list(n_threads = 1L, prune = prune, progress = FALSE,
                       diagnose_k = FALSE)))
}

test_that("the dropped-mass bound counts every kept cell and the joined grid", {
    res <- list(log_marginal = c(0, -1, -Inf, -2),
                prune_mask = c(FALSE, FALSE, TRUE, FALSE),
                prune_cheap_log_marginal = c(-0.5, -1.5, -8, NA))
    # The cell no screen saw (NA cheap) is kept mass with no error to read.
    e <- 0.5
    expect_equal(tulpa:::.nl_prune_dropped_mass(res),
                 exp(-8 + e) / (exp(0) + exp(-1) + exp(-2)))
    res$prune_screen_log_ref <- 1
    expect_equal(tulpa:::.nl_prune_dropped_mass(res),
                 exp(-8 + e) / (exp(1) + exp(0) + exp(-1) + exp(-2)))
})

skip_on_cran_budget()

test_that("a round's screen record measures the cells on the merged grid", {
    skip_on_cran_budget()
    specs <- list(list(name = "a", log_scale = FALSE, refinable = TRUE,
                       grid = c(0, 1, 2)))
    tg <- matrix(c(0, 1, 2), ncol = 1, dimnames = list(NULL, "a"))
    lm <- c(-1, 0, -1)
    cells <- matrix(c(0.5, 1.5), ncol = 1, dimnames = list(NULL, "a"))
    scr <- tulpa:::.hyper_refine_screen(tg, lm, rep("", 3), cells, "a", specs)
    lq <- tulpa:::.hyper_log_quad_weights(rbind(tg, cells), specs,
                                          refining = c("", "", "", "a", "a"))
    expect_equal(scr$log_measure, lq[4:5])
    expect_equal(scr$log_ref, log(sum(exp(lm + lq[1:3]))))
    off <- tulpa:::.nl_screen_log_offset(cells, list(list(lp = c(0.1, 0.2))),
                                         screen = scr)
    expect_equal(as.numeric(off), c(0.1, 0.2) + lq[4:5])
    expect_equal(attr(off, "log_ref"), scr$log_ref)
    expect_error(tulpa:::.nl_screen_log_offset(cells[1, , drop = FALSE],
                                               list(), screen = scr),
                 "refinement screen")
})

skip_on_cran()

test_that("refinement cells are screened against the grid they join", {
    skip_on_cran()
    sim <- .rs_sim()
    off <- .rs_fit(sim, prune = FALSE)
    on  <- .rs_fit(sim, prune = TRUE)
    n <- length(on$log_marginal)

    added <- nzchar(on$refining_axis)
    expect_true(any(added))
    expect_length(on$prune_mask, n)
    expect_length(on$prune_cheap_log_marginal, n)
    expect_identical(on$prune_n_pruned, sum(on$prune_mask))
    expect_true(any(on$prune_mask[added]))
    expect_true(all(!is.finite(on$log_marginal[on$prune_mask])))
    expect_true(all(is.finite(on$prune_cheap_log_marginal[added & on$prune_mask])))

    # The bound is read over the merged grid under its final measure.
    expect_true(is.finite(on$prune_dropped_mass_bound))
    expect_lte(on$prune_dropped_mass_bound, tulpa:::.nl_screen("gate_mass"))

    # Fewer full solves, the same posterior.
    expect_lt(sum(is.finite(on$log_marginal)), sum(is.finite(off$log_marginal)))
    sd_ref <- pmax(off$theta_sd, 1e-8)
    expect_true(all(abs(on$theta_mean - off$theta_mean) / sd_ref < 0.1))
    expect_equal(coef(on), coef(off), tolerance = 1e-2)
})

test_that("the cells screened across refinement rounds are held to the gate as a whole", {
    # Five rounds' worth of dropped cells, each a small share by itself; the
    # mock kernel's full log marginal sits at the screened one plus a 0.5-nat
    # error, the worst error a kept cell shows.
    tg <- cbind(a = seq_len(40), b = 1)
    full <- -0.05 * (tg[, "a"] - 10)^2 / 10
    pruned <- tg[, "a"] > 12
    cheap <- full - 0.5
    lm <- ifelse(pruned, -Inf, full)
    ex <- lapply(seq_len(40), function(i)
        list(pruned = pruned[i], cheap_lm = cheap[i], mode = c(0, 0)))
    gate <- 0.01
    pre <- tulpa:::.nl_prune_lifted(ifelse(pruned, -Inf, full),
                                    cheap, pruned)
    expect_gt(pre$bound, gate)
    calls <- 0L
    kfn <- function(cells, warm_start = NULL, store_extras = FALSE, screen = NULL) {
        calls <<- calls + 1L
        expect_null(screen)
        i <- cells[, "a"]
        list(log_marginal = full[i],
             extras = lapply(i, function(k) list(pruned = FALSE, mode = c(0, 0))))
    }
    out <- tulpa:::.hyper_screen_repair(tg, lm, ex, rep("a", 40), list(), kfn,
                                        gate = gate)
    expect_gt(out$info$n_repaired, 0L)
    expect_lte(out$info$bound_after, gate)
    expect_equal(out$info$bound_before, pre$bound)
    left <- vapply(out$extras, function(e) isTRUE(e$pruned), logical(1))
    # Every repaired cell carries its full solve, the rest stay dropped.
    expect_equal(out$log_marginal[!left], full[!left])
    expect_true(all(!is.finite(out$log_marginal[left])))
    # The heaviest dropped cells are the ones solved.
    expect_true(max(tg[!left & pruned, "a"]) < min(tg[left, "a"]))
})

test_that("a refined HSGP x AR(1) fit holds the dropped-mass bound under the gate", {
    skip_on_cran()
    # The 12-observation fixture's consistency pass extends all four axes over
    # about twenty rounds; their screened cells together carried 5.1% of the
    # posterior before the merged grid was held to the gate.
    set.seed(43)
    n <- 12L
    xy <- cbind(runif(n), runif(n))
    tt <- rep(1:3, 4)
    fit <- suppressWarnings(fit_st_nested(rnorm(n), matrix(1, n, 1), NULL, NULL,
        tt, 3L, spatial_type = "hsgp", temporal_type = "ar1",
        family = "gaussian", phi = .4, coords = xy, hsgp_m = 5L, hsgp_c = 1.8,
        control = list(progress = FALSE)))
    expect_true(any(nzchar(fit$refining_axis)))
    expect_lte(fit$prune_dropped_mass_bound, tulpa:::.nl_screen("gate_mass"))
})
