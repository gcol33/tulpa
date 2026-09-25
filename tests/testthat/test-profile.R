# Phase profiler: the per-phase accumulator (scatter / factorize / eta / ...),
# its cross-thread aggregation, its on/off gate, and the tulpa_profile()
# reader. The joint fits below force a field above SPARSE_THRESHOLD (200) to
# exercise the sparse joint solver; the single-response Laplace, the
# nested-Laplace outer grid and the NUTS sampler are timed too (#887).

# Run `expr` with timing on, from a reset accumulator, and return the raw read.
.profile_raw <- function(expr) {
  cpp_profile_reset()
  was_on <- cpp_profile_enable(TRUE)
  on.exit(cpp_profile_enable(was_on), add = TRUE)
  force(expr)
  cpp_profile_read()
}

.profile_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_neighbors <- vapply(nbr, length, integer(1))
  list(
    adj_row_ptr     = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx     = as.integer(unlist(nbr)) - 1L,
    n_neighbors     = as.integer(n_neighbors),
    n_spatial_units = n_s
  )
}

# A joint binomial+gaussian ICAR fit over a field large enough to take the
# sparse Newton path. Returns the responses/prior/copy list for reuse.
.profile_joint_inputs <- function(N = 1500L, n_s = 250L, seed = 31) {
  set.seed(seed)
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  rw    <- cumsum(rnorm(n_s, 0, 0.6 / sqrt(n_s)))
  phi_s <- rw - mean(rw)
  x     <- rnorm(N)
  Xocc  <- cbind(1, x)
  eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]
  occur   <- rbinom(N, 1L, plogis(eta_occ))
  is_pos  <- occur == 1L
  Xpos    <- Xocc[is_pos, , drop = FALSE]
  spi_pos <- spatial_idx[is_pos]
  eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + phi_s[spi_pos]
  y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)

  adj <- .profile_chain_adj(n_s)
  arm_occ <- list(
    y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
    spatial_idx = spatial_idx, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1.0, family = "binomial", phi = 1.0
  )
  arm_pos <- list(
    y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
    spatial_idx = spi_pos, re_idx = rep(0, length(y_pos)), n_re_groups = 0L,
    sigma_re = 1.0, family = "gaussian", phi = 1.0
  )
  prior <- list(
    type = "icar", n_spatial_units = adj$n_spatial_units,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, sigma_grid = c(0.5, 1.0)
  )
  list(
    responses = list(
      occ = arm_occ,
      pos = modifyList(arm_pos, list(
        field_coef = list(name = "alpha", grid = c(0.5, 1.0))))
    ),
    prior     = prior
  )
}

test_that("cpp_profile_reset zeroes every phase", {
  cpp_profile_reset()
  r <- cpp_profile_read()
  expect_identical(
    as.character(r$names),
    c("pattern_build", "prep", "eta", "scatter", "analyze", "factorize",
      "solve", "line_search", "log_det", "log_lik_prior", "hessian_extract",
      "inner_diagnostics", "gradient", "outer_grid_cell", "nuts_warmup",
      "nuts_sampling")
  )
  expect_identical(
    as.character(r$names)[r$enclosing],
    c("outer_grid_cell", "nuts_warmup", "nuts_sampling")
  )
  expect_true(all(r$us == 0))
  expect_true(all(r$calls == 0L))
})

test_that("timing is off outside tulpa_profile() and restored after it", {
  skip_on_cran()
  set.seed(1)
  n <- 200L; X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, plogis(X %*% c(0, 0.5)))
  expect_false(cpp_profile_enable(FALSE))  # off by default
  cpp_profile_reset()
  tulpa_laplace(y, rep(1L, n), X, family = "binomial")
  expect_true(all(cpp_profile_read()$calls == 0L))
  # tulpa_profile() switches it on for its expression only, even on error.
  expect_error(tulpa_profile(stop("boom")), "boom")
  expect_false(cpp_profile_enable(FALSE))
})

test_that("the sparse joint solver records scatter and factorize separately", {
  skip_on_cran()
  inp <- .profile_joint_inputs()

  r <- .profile_raw(
    fit <- tulpa_nested_laplace_joint(responses = inp$responses,
                                      prior = inp$prior, copy = inp$copy))

  expect_s3_class(fit, "tulpa_nested_laplace_joint")

  named <- function(nm) r$us[[which(r$names == nm)]]
  calls <- function(nm) r$calls[[which(r$names == nm)]]

  # The instrumented sparse path ran: both the assembly scatter and the
  # numeric factorize accumulated time and fired the same number of times
  # (one scatter + one factor per inner Newton iter, plus the final pass).
  expect_gt(named("scatter"), 0)
  expect_gt(named("factorize"), 0)
  expect_gt(calls("scatter"), 0L)
  expect_equal(calls("factorize"), calls("scatter"))
})

test_that("tulpa_profile returns the phase split and carries the fit", {
  skip_on_cran()
  inp <- .profile_joint_inputs()

  p <- tulpa_profile(
    tulpa_nested_laplace_joint(responses = inp$responses,
                               prior = inp$prior, copy = inp$copy)
  )

  expect_s3_class(p, "data.frame")
  expect_identical(names(p),
                   c("phase", "seconds", "calls", "ms_per_call", "share"))
  expect_true(all(c("scatter", "factorize") %in% p$phase))
  # Rows are ordered by descending time.
  expect_false(is.unsorted(rev(p$seconds)))
  # Shares of the timed leaf phases sum to 1; the enclosing outer-grid cell
  # overlaps them and carries no share.
  leaf <- !is.na(p$share)
  expect_equal(sum(p$share[leaf & p$seconds > 0]), 1, tolerance = 1e-8)
  expect_true(is.na(p$share[p$phase == "outer_grid_cell"]))
  expect_gt(p$calls[p$phase == "outer_grid_cell"], 0L)
  # ms_per_call is consistent with seconds / calls where calls > 0.
  pos <- p$calls > 0
  expect_equal(p$ms_per_call[pos], (p$seconds[pos] * 1e3) / p$calls[pos],
               tolerance = 1e-6)

  expect_s3_class(attr(p, "value"), "tulpa_nested_laplace_joint")
})

test_that("tulpa_profile resets between runs (no carry-over)", {
  skip_on_cran()
  inp <- .profile_joint_inputs()

  p1 <- tulpa_profile(
    tulpa_nested_laplace_joint(responses = inp$responses,
                               prior = inp$prior, copy = inp$copy)
  )
  # A second profiled run reports only its own work, not the sum of both.
  p2 <- tulpa_profile(
    tulpa_nested_laplace_joint(responses = inp$responses,
                               prior = inp$prior, copy = inp$copy)
  )
  s1 <- p1$calls[p1$phase == "scatter"]
  s2 <- p2$calls[p2$phase == "scatter"]
  expect_gt(s2, 0L)
  # Deterministic inputs -> same number of scatter calls, not double.
  expect_equal(s2, s1)
})

test_that("tulpa_profile warns when nothing instrumented ran (#887)", {
  # An all-zero table is not a measurement: the expression never reached a
  # timed solver.
  expect_warning(p <- tulpa_profile(1 + 1), "no instrumented phase")
  expect_true(all(p$calls == 0L))
  expect_identical(attr(p, "value"), 2)
})

# The phases a Laplace inner solve records: the Newton loop's eta / scatter /
# factorize / line_search and the final pass's log_det / log_lik_prior.
.profile_newton_phases <- c("eta", "scatter", "factorize", "line_search",
                            "log_det", "log_lik_prior")

test_that("the single-response Laplace is timed by phase (#887)", {
  skip_on_cran()
  set.seed(1)
  n <- 200L; X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, plogis(X %*% c(0, 0.5)))
  expect_no_warning(
    p <- tulpa_profile(tulpa_laplace(y, rep(1L, n), X, family = "binomial")))
  timed <- p$phase[p$calls > 0L]
  expect_true(all(.profile_newton_phases %in% timed))
  expect_true(all(p$seconds[p$phase %in% .profile_newton_phases] > 0))
  # One scatter per Newton iteration plus the final pass, and one factorize
  # per iteration: so exactly one more scatter than factorize.
  n_sc <- p$calls[p$phase == "scatter"]
  expect_equal(p$calls[p$phase == "factorize"], n_sc - 1L)
  expect_equal(p$calls[p$phase == "log_det"], 1L)
  # No outer grid, no sampler.
  expect_false(any(c("outer_grid_cell", "gradient") %in% timed))
  expect_equal(sum(p$share, na.rm = TRUE), 1, tolerance = 1e-8)
  expect_true(is.list(attr(p, "value")))
})

test_that("tulpa(mode = 'laplace') reaches the same timed solve (#887)", {
  skip_on_cran()
  set.seed(2)
  d <- data.frame(x = rnorm(150L))
  d$y <- rpois(150L, exp(0.2 + 0.4 * d$x))
  expect_no_warning(
    p <- tulpa_profile(tulpa(y ~ x, data = d, family = "poisson",
                             mode = "laplace")))
  expect_true(all(p$calls[p$phase %in% .profile_newton_phases] > 0L))
})

test_that("the nested-Laplace outer grid times each cell and its inner solve (#887)", {
  skip_on_cran()
  set.seed(11L)
  n_s <- 20L; N <- 160L
  adj <- .profile_chain_adj(n_s)
  w <- sin(seq_len(n_s) / 3); w <- w - mean(w)
  sidx <- sample(n_s, N, replace = TRUE); x <- rnorm(N)
  y <- rbinom(N, 1, plogis(0.3 + 0.5 * x + w[sidx]))
  sig <- c(0.5, 0.8, 1.2, 1.8)
  prior <- c(list(type = "icar", tau_grid = 1 / sig^2, spatial_idx = sidx), adj)
  expect_no_warning(p <- tulpa_profile(
    tulpa_nested_laplace(y = y, n_trials = rep(1L, N), X = cbind(1, x),
                         prior = prior, family = "binomial",
                         control = list(diagnose_k = FALSE))))
  n_cell <- p$calls[p$phase == "outer_grid_cell"]
  expect_gte(n_cell, length(sig))
  expect_gt(p$seconds[p$phase == "outer_grid_cell"], 0)
  # Every cell ran a full inner solve: at least one final-pass log-det each.
  expect_gte(p$calls[p$phase == "log_det"], n_cell)
  expect_true(all(p$seconds[p$phase %in% .profile_newton_phases] > 0))
  # The enclosing cell phase overlaps the leaves and takes no share.
  expect_true(is.na(p$share[p$phase == "outer_grid_cell"]))
  expect_equal(sum(p$share, na.rm = TRUE), 1, tolerance = 1e-8)
})

test_that("the NUTS sampler times warmup, sampling and gradients (#887)", {
  skip_on_cran()
  set.seed(8L)
  d <- data.frame(x = rnorm(100L))
  d$y <- rpois(100L, exp(0.3 + 0.5 * d$x))
  expect_no_warning(p <- tulpa_profile(suppressWarnings(
    tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
          control = list(n_iter = 60L, warmup = 30L, n_chains = 1L,
                         seed = 1L)))))
  calls <- setNames(p$calls, p$phase)
  expect_equal(calls[["nuts_warmup"]], 30L)
  expect_equal(calls[["nuts_sampling"]], 30L)
  # At least one gradient per iteration, timed as a leaf inside them.
  expect_gte(calls[["gradient"]], 60L)
  expect_gt(p$seconds[p$phase == "gradient"], 0)
  expect_false(is.na(p$share[p$phase == "gradient"]))
  expect_true(all(is.na(p$share[p$phase %in% c("nuts_warmup", "nuts_sampling")])))
})
