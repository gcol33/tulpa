# Inline correlated areal fields (separable MCAR) via a single bar:
# spatial(graph, ~ 1 + x | cell) (gcol33/tulpa#89). The per-cell coefficient
# vector (u_c, s_c) shares a cross-covariance Sigma; covariance Sigma (x) Q^-1.

.mcar_grid_adj <- function(nx, ny) {
  n <- nx * ny
  W <- matrix(0L, n, n)
  id <- function(i, j) (j - 1L) * nx + i
  for (i in seq_len(nx)) for (j in seq_len(ny)) {
    if (i < nx) { a <- id(i, j); b <- id(i + 1L, j); W[a, b] <- W[b, a] <- 1L }
    if (j < ny) { a <- id(i, j); b <- id(i, j + 1L); W[a, b] <- W[b, a] <- 1L }
  }
  W
}

# Draw correlated fields (u, s) with cov Sigma (x) Q^-1 via the LMC factorization
# u = L11 z1, s = L21 z1 + L22 z2 with z1, z2 iid smooth CAR (rho = 0.99, proper
# so invertible, near-intrinsic), L = chol(Sigma) lower. Centre each field.
.mcar_sim <- function(adj, Sigma) {
  n <- nrow(adj)
  Qp <- diag(rowSums(adj)) - 0.99 * adj
  U  <- chol(Qp)
  z1 <- backsolve(U, stats::rnorm(n)); z1 <- z1 - mean(z1)
  z2 <- backsolve(U, stats::rnorm(n)); z2 <- z2 - mean(z2)
  L  <- t(chol(Sigma))
  u <- L[1, 1] * z1
  s <- L[2, 1] * z1 + L[2, 2] * z2
  list(u = u - mean(u), s = s - mean(s))
}

# The p = 2 recovery fixture: a 10 x 10 lattice with Sigma = (1, 0.8; rho 0.6)
# and 10 Gaussian observations per cell. rho is read off the 100 cells' fields,
# whose SDs are 0.5-0.8 on these seeds, and 10 observations at noise SD 0.4 put
# each cell's (u, s) at a standard error of about 0.13, so the cell count is
# what limits the cross-correlation, and it is the lattice that is kept.
.mcar_recovery_rho <- 0.6
.mcar_recovery_fit <- fixture_once(function(seed) {
  adj <- .mcar_grid_adj(10L, 10L)
  sig1 <- 1.0; sig2 <- 0.8; rho <- .mcar_recovery_rho
  Sigma <- matrix(c(sig1^2, rho * sig1 * sig2,
                    rho * sig1 * sig2, sig2^2), 2, 2)
  set.seed(seed)
  fld <- .mcar_sim(adj, Sigma)
  cell <- rep(seq_len(nrow(adj)), each = 10L)
  N <- length(cell)
  x <- rnorm(N)
  eta <- 0.3 + fld$u[cell] + x * fld$s[cell]
  y <- rnorm(N, eta, 0.4)
  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 + x | cell),
    data = data.frame(y = y, x = x, cell = cell),
    family = "gaussian", mode = "laplace",
    control = list(n_draws = 200L)))
  list(fit = fit, fld = fld)
})

skip_on_cran_budget()

test_that("spatial(~ ... | cell) builds one coupled MCAR block", {
  skip_on_cran_budget()
  adj <- .mcar_grid_adj(4L, 4L)
  f <- spatial(graph = adj, formula = ~ 1 + x | cell)
  expect_true(f$correlated)
  d <- data.frame(cell = rep(seq_len(16L), each = 2L), x = rnorm(32L))
  blocks <- tulpa:::.spatial_field_blocks(f, d)
  expect_length(blocks, 1L)
  expect_identical(blocks[[1]]$type, "mcar")
  expect_identical(blocks[[1]]$n_fields, 2L)
  expect_length(blocks[[1]]$field_weight, 2L)
  expect_setequal(blocks[[1]]$field_names, c("cell.Intercept", "cell.x"))

  # Correlated + proper is out of scope.
  expect_error(spatial(graph = adj, formula = ~ 1 | cell, proper = TRUE),
               "not in scope")
})

skip_on_cran()

test_that("MCAR recovers the correlated fields and Sigma vs simulated truth", {
  skip_on_cran()

  rho_true <- .mcar_recovery_rho
  cor_u <- cor_s <- rho_med <- numeric(0)
  rho_cover <- logical(0)
  for (seed in 1:3) {
    rec <- .mcar_recovery_fit(seed)
    fit <- rec$fit
    fld <- rec$fld

    expect_s3_class(fit, "tulpa_spatial_field_fit")
    expect_true(isTRUE(fit$correlated))
    # Strong, well-identified data: the outer Sigma grid is the mode-centred CCD
    # over the log-Cholesky coordinates (1 + 2*3 + 2^3 = 15 nodes for p = 2), not
    # the fixed log-Cholesky tensor -- the documented follow-up. The outer
    # Pareto-k is then scored (was NA when the fit declined the CCD).
    expect_identical(fit$integration, "ccd")
    expect_false(is.na(fit$pareto_k))
    u_hat <- fit$spatial_fields[["cell.Intercept"]]$mean
    s_hat <- fit$spatial_fields[["cell.x"]]$mean
    cor_u <- c(cor_u, abs(cor(u_hat, fld$u)))
    cor_s <- c(cor_s, abs(cor(s_hat, fld$s)))

    rq <- fit$mcar_summary[["rho_12"]]$q
    rho_med <- c(rho_med, rq[2L])
    rho_cover <- c(rho_cover, rq[1L] <= rho_true && rho_true <= rq[3L])
  }

  # Both fields recover every seed.
  expect_true(all(cor_u > 0.7))
  expect_true(all(cor_s > 0.7))
  # rho recovers a clearly positive cross-correlation (true 0.6), and the
  # marginalized CI covers the truth in the majority of seeds.
  expect_gt(median(rho_med), 0.3)
  expect_gte(sum(rho_cover), 2L)
})

test_that("MCAR mode-centred CCD scales to p = 3 (general p > 2)", {
  skip_on_cran()
  set.seed(3)
  nx <- ny <- 8L
  adj <- .mcar_grid_adj(nx, ny)
  n_s <- nx * ny
  # Lower Cholesky factor of the 3-field cross-covariance; Sigma = L L'.
  L3 <- matrix(c(1.0,  0.0, 0.0,
                 0.4,  0.8, 0.0,
                 -0.2, 0.3, 0.7), 3, 3, byrow = TRUE)
  Sigma3 <- L3 %*% t(L3)
  R3 <- cov2cor(Sigma3)

  # Three correlated near-intrinsic CAR fields via the LMC factorization.
  Qp <- diag(rowSums(adj)) - 0.99 * adj
  U  <- chol(Qp)
  Z  <- sapply(1:3, function(k) { z <- backsolve(U, rnorm(n_s)); z - mean(z) })
  Fl <- Z %*% t(L3)
  Fl <- apply(Fl, 2L, function(col) col - mean(col))

  cell <- rep(seq_len(n_s), each = 30L)
  N <- length(cell)
  x1 <- rnorm(N); x2 <- rnorm(N)
  eta <- 0.2 + Fl[cell, 1] + x1 * Fl[cell, 2] + x2 * Fl[cell, 3]
  y <- rnorm(N, eta, 0.4)

  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 + x1 + x2 | cell),
    data = data.frame(y = y, x1 = x1, x2 = x2, cell = cell),
    family = "gaussian", mode = "laplace", control = list(n_draws = 200L)))

  expect_s3_class(fit, "tulpa_spatial_field_fit")
  # CCD over the 6 log-Cholesky axes and the residual variance, which `phi`
  # left out makes an axis too -- a polynomial node count where the fixed
  # tensor would be thousands of cells.
  expect_identical(fit$integration, "ccd")
  expect_identical(ncol(fit$theta_grid), 7L)
  expect_true("phi_y" %in% colnames(fit$theta_grid))
  # All three cross-correlations recover with a CI covering the truth.
  for (ij in list(c(1L, 2L), c(1L, 3L), c(2L, 3L))) {
    nm <- sprintf("rho_%d%d", ij[1L], ij[2L])
    q  <- fit$mcar_summary[[nm]]$q
    rho_true <- R3[ij[1L], ij[2L]]
    expect_true(q[1L] <= rho_true && rho_true <= q[3L],
                info = sprintf("%s CI (%.3f, %.3f) should cover %.3f",
                               nm, q[1L], q[3L], rho_true))
  }
})

test_that("MCAR fit print reports the Sigma cross-covariance (rho)", {
  skip_on_cran()
  # Any p = 2 MCAR fit carries the Sigma summary the print reads.
  fit <- .mcar_recovery_fit(1L)$fit
  expect_false(is.null(fit$mcar_summary[["rho_12"]]))
  out <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(out, "MCAR")
  expect_match(out, "rho_12")
})

test_that("MCAR fixed-effect SEs match the uncorrelated (||) fit, not the singular one-group constraint (gcol33/tulpa#797)", {
  skip_on_cran()
  set.seed(1)
  W <- adjacency(expand.grid(x = 1:6, y = 1:6), x_coord = "x", y_coord = "y",
                 type = "rook")$adjacency
  d <- data.frame(region = rep(1:36, each = 4), x = rnorm(144))
  u <- rnorm(36, 0, .7)
  d$y <- rpois(144, exp(0.3 + 0.5 * d$x + u[d$region]))

  fm_mcar <- y ~ x + spatial(graph = W, formula = ~ 1 + x | region)
  fm_iid  <- y ~ x + spatial(graph = W, formula = ~ 1 + x || region)
  f_mcar  <- suppressWarnings(tulpa(fm_mcar, data = d, family = "poisson"))
  f_iid   <- suppressWarnings(tulpa(fm_iid, data = d, family = "poisson"))

  se_mcar <- coef(summary(f_mcar))$std.error
  se_iid  <- coef(summary(f_iid))$std.error
  # A one-group-over-both-fields constraint left ~75; the correct per-field
  # constraint should read within a small multiple of the uncorrelated fit's SE.
  expect_true(all(se_mcar < 5 * se_iid))
  expect_true(all(se_mcar < 1))
})

test_that("an MCAR block's logchol_grid without column names takes the layout's", {
  skip_on_cran()
  expect_identical(tulpa:::.mcar_logchol_names(3L),
                   c("L11", "L21", "L31", "L22", "L32", "L33"))
  expect_identical(colnames(tulpa:::.mcar_default_logchol_grid(3L)),
                   tulpa:::.mcar_logchol_names(3L))
  set.seed(10534)
  W <- .mcar_grid_adj(2L, 4L); n <- nrow(W); N <- 160L
  cell <- sample.int(n, N, TRUE); x <- rnorm(N)
  y <- as.numeric(rbinom(N, 1, plogis(0.2 + 0.4 * x + rnorm(n, 0, 0.6)[cell])))
  adj <- tulpa:::adjacency_to_csr_tulpa(W)
  th <- c(log(0.9), 0.3, log(0.6))
  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              family = "binomial", phi = 1)
  blk <- function(g) list(type = "mcar", spatial_idx = list(cell),
                          n_spatial_units = n, n_fields = 2L,
                          adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
                          n_neighbors = adj$n_neighbors,
                          field_weight = list(list(rep(1, N)), list(x)),
                          logchol_grid = g)
  fit <- function(g) suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = list(blk(g)),
    control = list(progress = FALSE, axis_refine = "none")))
  unnamed <- fit(matrix(th, 1))
  named <- fit(matrix(th, 1, dimnames = list(NULL, c("L11", "L21", "L22"))))
  expect_identical(colnames(unnamed$theta_grid), colnames(named$theta_grid))
  expect_equal(unnamed$log_marginal, named$log_marginal)
  expect_error(fit(matrix(th[1:2], 1)), "3 log-Cholesky coordinates")
})

# A two-arm MCAR fit on the joint door: a 7 x 7 lattice, a Poisson count arm
# carrying field 1 and a binomial arm carrying field 2, Sigma with sigma =
# (1, 0.8) and rho = 0.6. The block states no `logchol_grid`, so the engine
# starts from its default (sigma_1, sigma_2, rho) grid.
.mcar_joint_case <- fixture_once(function(seed) {
  set.seed(seed)
  W <- .mcar_grid_adj(7L, 7L); S <- nrow(W)
  adj <- tulpa:::adjacency_to_csr_tulpa(W)
  Sigma <- matrix(c(1, 0.48, 0.48, 0.64), 2)
  R <- chol(diag(rowSums(W)) - 0.99 * W)
  f <- sapply(1:2, function(k) backsolve(R, rnorm(S))) %*% chol(Sigma)
  f <- sweep(f, 2, colMeans(f))
  s1 <- rep(seq_len(S), each = 4L); x1 <- rnorm(length(s1))
  s2 <- rep(seq_len(S), each = 4L); x2 <- rnorm(length(s2))
  n1 <- length(s1); n2 <- length(s2)
  arms <- list(
    a = list(y = rpois(n1, exp(0.5 + 0.4 * x1 + f[s1, 1])),
             n_trials = rep(1L, n1), X = cbind(1, x = x1), family = "poisson"),
    b = list(y = rbinom(n2, 10L, plogis(-0.3 + 0.8 * x2 + f[s2, 2])),
             n_trials = rep(10L, n2), X = cbind(1, x = x2), family = "binomial"))
  blk <- list(type = "mcar", n_spatial_units = S, n_fields = 2L,
              adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
              n_neighbors = adj$n_neighbors, spatial_idx = list(s1, s2),
              field_weight = list(list(rep(1, n1), rep(0, n2)),
                                  list(rep(0, n1), rep(1, n2))))
  fit <- tulpa_nested_laplace_joint(arms, prior = list(blk),
                                    control = list(progress = FALSE,
                                                   store_Q = TRUE))
  list(arms = arms, blk = blk, fit = fit)
})

.mcar_cell_sigma_rho <- function(nodes) {
  s2 <- sqrt(nodes[, "b1.L21"]^2 + exp(2 * nodes[, "b1.L22"]))
  cbind(sigma_1 = exp(nodes[, "b1.L11"]), sigma_2 = s2,
        rho_12 = nodes[, "b1.L21"] / s2,
        Sigma_12 = exp(nodes[, "b1.L11"]) * nodes[, "b1.L21"])
}

.mcar_refine_modes <- function(fit, axes) {
  vapply(fit$axis_span[axes], function(a) a$refine, character(1))
}

test_that("an MCAR block's sigma_i / rho_ij / Sigma_ij are marginalized across the cells", {
  skip_on_cran()
  fit <- .mcar_joint_case(1062L)$fit
  h <- summary(fit)$hyperparameters
  expect_true(all(c("b1.sigma_1", "b1.sigma_2", "b1.rho_12", "b1.Sigma_11",
                    "b1.Sigma_12", "b1.Sigma_22") %in% h$parameter))
  # Each quantity is read across the cells, off the within-cell distribution
  # the fit's hyperparameter draws sample: the summary and the derived values
  # of the draws describe one distribution.
  set.seed(7)
  th <- tulpa_hyper_draws(fit, n = 2e5)
  D <- .mcar_cell_sigma_rho(th)
  for (nm in colnames(D)) {
    row <- h[h$parameter == paste0("b1.", nm), ]
    s <- stats::sd(D[, nm])
    expect_lt(abs(row$mean - mean(D[, nm])), 5 * s / sqrt(nrow(D)))
    expect_lt(abs(row$sd / s - 1), 0.02)
    q <- stats::quantile(D[, nm], c(0.025, 0.975), names = FALSE)
    expect_lt(abs(row$ci_lo - q[1L]), 0.05 * s)
    expect_lt(abs(row$ci_hi - q[2L]), 0.05 * s)
    expect_true(row$ci_lo < row$median && row$median < row$ci_hi)
  }
  rho <- h[h$parameter == "b1.rho_12", ]
  expect_true(rho$ci_lo > -1 && rho$ci_hi < 1)
})

test_that("the default MCAR grid is placed at the outer mode, as a log-Cholesky tensor, against a dense reference", {
  skip_on_cran()
  case <- .mcar_joint_case(1062L)
  fit <- case$fit
  axes <- c("b1.L11", "b1.L21", "b1.L22")
  expect_identical(fit$outer_grid_placement, "auto_recentered")
  expect_setequal(fit$outer_grid_recenter_axes, axes)
  lc <- fit$theta_grid[, axes]
  expect_identical(tulpa:::.hp_logchol_design(lc)$design, "logchol")
  # Placed by the engine, the axes may take levels past their nodes.
  expect_true(all(.mcar_refine_modes(fit, axes) == "extend"))

  # Dense reference: a 25^3 log-Cholesky tensor over +/- 5 posterior SDs,
  # 0.42 SD apart, integrated as given.
  m <- colSums(fit$weights * lc)
  s <- sqrt(colSums(fit$weights * sweep(lc, 2, m)^2))
  lev <- lapply(1:3, function(j) seq(m[j] - 5 * s[j], m[j] + 5 * s[j],
                                     length.out = 25L))
  G <- as.matrix(expand.grid(L11 = lev[[1]], L21 = lev[[2]], L22 = lev[[3]],
                             KEEP.OUT.ATTRS = FALSE))
  ref <- suppressWarnings(tulpa_nested_laplace_joint(
    case$arms, prior = list(c(case$blk, list(logchol_grid = G))),
    control = list(progress = FALSE, auto_recenter = FALSE,
                   axis_refine = "none", prune = FALSE, diagnose_k = FALSE,
                   diagnose_skew = FALSE)))
  hf <- summary(fit)$hyperparameters
  hr <- summary(ref)$hyperparameters
  for (nm in c("b1.sigma_1", "b1.sigma_2", "b1.rho_12", "b1.Sigma_11",
               "b1.Sigma_12", "b1.Sigma_22")) {
    a <- hf[hf$parameter == nm, ]; b <- hr[hr$parameter == nm, ]
    expect_lt(abs(a$mean - b$mean), 0.1 * b$sd)
    expect_lt(abs(a$sd / b$sd - 1), 0.1)
    # The interval ends too: read at the cell centres, the correlation's lower
    # end sat 0.58 reference SDs above the dense grid's (0.098 against -0.013).
    expect_lt(abs(a$ci_lo - b$ci_lo), 0.15 * b$sd)
    expect_lt(abs(a$ci_hi - b$ci_hi), 0.15 * b$sd)
  }
})

test_that("a stated log-Cholesky tensor is refined by levels; a (sigma, rho) grid is not", {
  skip_on_cran()
  case <- .mcar_joint_case(1062L)
  axes <- c("b1.L11", "b1.L21", "b1.L22")
  lc <- case$fit$theta_grid[, axes]
  m <- colSums(case$fit$weights * lc)
  s <- sqrt(colSums(case$fit$weights * sweep(lc, 2, m)^2))
  lev <- lapply(1:3, function(j) m[j] + c(-3, 0, 3) * s[j])
  G <- as.matrix(expand.grid(L11 = lev[[1]], L21 = lev[[2]], L22 = lev[[3]],
                             KEEP.OUT.ATTRS = FALSE))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    case$arms, prior = list(c(case$blk, list(logchol_grid = G))),
    control = list(progress = FALSE)))
  expect_identical(fit$outer_grid_placement, "fixed")
  expect_true(any(fit$refining_axis %in% axes))
  expect_gt(nrow(fit$theta_grid), nrow(G))
  expect_true(all(.mcar_refine_modes(fit, axes) == "densify"))

  sds <- c(0.5, 0.8, 1.2)
  g <- expand.grid(s1 = sds, s2 = sds, rho = c(0, 0.3, 0.6))
  G2 <- cbind(L11 = log(g$s1), L21 = g$rho * g$s2,
              L22 = log(g$s2 * sqrt(1 - g$rho^2)))
  fit2 <- suppressWarnings(tulpa_nested_laplace_joint(
    case$arms, prior = list(c(case$blk, list(logchol_grid = G2))),
    control = list(progress = FALSE)))
  expect_false(any(fit2$refining_axis %in% axes))
  expect_identical(nrow(fit2$theta_grid), nrow(G2))
})

test_that("diagnostics() grades the inner Laplace layer of an MCAR block, solving against the pinned Hessian", {
  skip_on_cran()
  case <- .mcar_joint_case(1062L)
  fit <- case$fit
  expect_true(is.na(fit$inner_skew_declined))
  expect_true(all(is.finite(fit$inner_skew)))
  expect_true(all(is.finite(fit$inner_pareto_k)))

  # The probes' conditional SD at each probed coefficient is the one of the
  # stored (unpinned) precision of the probed cell plus the field's
  # sum-to-zero augmentation, Sigma^-1 (x) 1 1' / n.
  probe <- NULL
  env <- environment()
  trace(".inner_skew_attach", where = asNamespace("tulpa"), print = FALSE,
        tracer = bquote(assign("probe", out, envir = .(env))))
  on.exit(untrace(".inner_skew_attach", where = asNamespace("tulpa")),
          add = TRUE)
  fit2 <- suppressWarnings(tulpa_nested_laplace_joint(
    case$arms, prior = list(case$blk),
    control = list(progress = FALSE, store_Q = TRUE)))
  expect_false(is.null(probe$inner_is_sigma))
  k <- which.max(fit2$weights)
  n <- length(fit2$modes[k, ])
  Hl <- Matrix::sparseMatrix(i = fit2$Q_csc_i_per_grid[[k]] + 1L,
                             p = fit2$Q_csc_p_per_grid[[k]],
                             x = fit2$Q_csc_x_per_grid[[k]], dims = c(n, n))
  H <- as.matrix(Hl)
  H <- H + t(H) - diag(diag(H))
  th <- fit2$theta_grid[k, ]
  L <- matrix(c(exp(th[["b1.L11"]]), th[["b1.L21"]], 0,
                exp(th[["b1.L22"]])), 2)
  S <- case$blk$n_spatial_units
  fs <- fit2$arm_layout$field_starts
  U <- matrix(0, n, 2)
  U[fs + seq_len(S), 1] <- 1
  U[fs + S + seq_len(S), 2] <- 1
  Haug <- H + U %*% (solve(tcrossprod(L)) / S) %*% t(U)
  expect_equal(as.numeric(probe$inner_is_sigma),
               sqrt(diag(solve(Haug))[probe$inner_skew_idx]),
               tolerance = 1e-8)
})
