# An intrinsic field on a DISCONNECTED graph: each connected component's level
# carries the field's own precision through the augmentation (Q_aug = Q +
# sum_c 1_c 1_c' / J_c), and the whole field's level goes to the intercept.
# The reads are checked against the exact Gaussian posterior of that model,
# conditioned on the level the inner solve centres.
#
# For a per-component-scaled BYM2 field the prior is diag(w) Q_aug, and
# diag(w) Q_aug 1 = w: the constant direction is a priori independent of the
# contrasts with w'x = 0, not of those with 1'x = 0. The level is therefore the
# w-weighted mean, and the field's contrasts carry the Schur complement
# diag(w) Q_aug - w w' / W, which is also the prior the Polya-Gamma kernels
# sample (PgIcarPrior). Conditioning on the plain sum instead read a
# covariance 3.5% of an SD away from the one of the mode the solve returned.

.dc_ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
                prune = FALSE, var_of_means_consistency = FALSE,
                skew_correct = FALSE, progress = FALSE)

# A path of 8, a cycle of 6, a path of 3 and an island, nodes shuffled so no
# component is contiguous.
.dc_graph <- function() {
  path <- function(n) {
    A <- matrix(0, n, n)
    for (i in seq_len(n - 1)) A[i, i + 1] <- A[i + 1, i] <- 1
    A
  }
  cyc <- path(6); cyc[1, 6] <- cyc[6, 1] <- 1
  A <- as.matrix(Matrix::bdiag(path(8), cyc, path(3), matrix(0, 1, 1)))
  comp <- c(rep(1, 8), rep(2, 6), rep(3, 3), 4)
  set.seed(7)
  perm <- sample.int(nrow(A))
  list(A = A[perm, perm], comp = comp[perm])
}

.dc_aug <- function(A, comp) {
  Q <- diag(rowSums(A)) - A
  for (c in unique(comp)) {
    m <- comp == c
    Q[m, m] <- Q[m, m] + 1 / sum(m)
  }
  Q
}

# Exact posterior of the field loading under a Gaussian response, the latent
# x = (beta, field coordinates) with prior precision Pprior, conditioned on
# a' x = 0.
.dc_exact <- function(y, D, Pprior, phi, a, L) {
  P <- Pprior + crossprod(D) / phi
  Cx <- solve(P)
  m <- Cx %*% crossprod(D, y) / phi
  Ca <- Cx %*% a
  m <- m - Ca * sum(a * m) / sum(a * Ca)
  Cx <- Cx - Ca %*% t(Ca) / sum(a * Ca)
  list(mean = as.numeric(t(L) %*% m), sd = sqrt(diag(t(L) %*% Cx %*% L)))
}

test_that("an ICAR field on a disconnected graph reads each component exactly", {
  skip_on_cran()
  g <- .dc_graph(); A <- g$A; K <- nrow(A)
  set.seed(11)
  idx <- rep(seq_len(K), each = 3L); n <- length(idx); x <- rnorm(n)
  y <- 0.3 + 0.5 * x + c(0, 1.2, -1, 2)[g$comp][idx] + rnorm(n, sd = 0.4)
  d <- data.frame(y = y, x = x, region = idx)
  phi <- 0.2; tau <- 2
  f <- suppressMessages(tulpa(y ~ x + spatial(region), d, family = "gaussian",
             phi = phi, spatial = list(type = "icar", adjacency = A, tau_grid = tau),
             mode = "nested_laplace", control = .dc_ctl))
  fe <- field_effects(f)
  sb <- tulpa:::.tulpa_prior_sd()
  Z <- outer(idx, seq_len(K), "==") * 1
  ex <- .dc_exact(y, cbind(1, x, Z),
                  as.matrix(Matrix::bdiag(diag(1 / sb^2, 2), tau * .dc_aug(A, g$comp))),
                  phi, c(0, 0, rep(1, K)), rbind(matrix(0, 2, K), diag(K)))
  expect_equal(fe$estimate, ex$mean, tolerance = 1e-8)
  expect_equal(fe$sd, ex$sd, tolerance = 1e-8)
  # The component levels, the quantity the augmentation identifies softly.
  expect_equal(as.numeric(tapply(fe$estimate, g$comp, mean)),
               as.numeric(tapply(ex$mean, g$comp, mean)), tolerance = 1e-8)
})

test_that("a per-component BYM2 field is conditioned on its weighted level", {
  skip_on_cran()
  g <- .dc_graph(); A <- g$A; K <- nrow(A)
  bs <- tulpa:::.bym2_component_scaling(A)
  w <- bs$node_prec; sf <- bs$scale_factor
  expect_gt(length(unique(w)), 1L)
  set.seed(5)
  x <- rnorm(K)
  y <- 0.3 + 0.5 * x + c(0, 1.2, -1, 2)[g$comp] + rnorm(K)
  d <- data.frame(y = y, x = x, region = seq_len(K))
  s <- 0.9; r <- 0.6; phi <- 1
  f <- suppressMessages(tulpa(y ~ x + spatial(region), d, family = "gaussian",
             phi = phi, spatial = list(type = "bym2", adjacency = A,
                                       sigma_grid = s, rho_grid = r),
             mode = "nested_laplace", control = .dc_ctl))
  fe <- field_effects(f)
  a <- s * sqrt(r) * sf; b <- s * sqrt(1 - r); I_K <- diag(K)
  sb <- tulpa:::.tulpa_prior_sd()
  Pprior <- as.matrix(Matrix::bdiag(diag(1 / sb^2, 2),
                                    diag(w) %*% .dc_aug(A, g$comp), I_K))
  ex <- .dc_exact(y, cbind(1, x, a * I_K, b * I_K), Pprior, phi,
                  c(0, 0, w, rep(0, K)),
                  rbind(matrix(0, 2, K), a * I_K, b * I_K))
  # The level is a hard constraint on the inner solve (gcol33/tulpa#1053), so
  # the mode is the conditional mean itself; folding the augmented mode's
  # level into the intercept left 1.4e-5 between them.
  expect_equal(fe$estimate, ex$mean, tolerance = 1e-8)
  expect_equal(fe$sd, ex$sd, tolerance = 1e-8)
  # The structured component's reported level is the weighted one.
  phi_mode <- f$modes[which.max(f$weights), 2L + seq_len(K)]
  expect_lt(abs(sum(w * phi_mode)), 1e-8)
})

test_that("the disconnected-graph evidence is exact on the single and multi-block drivers", {
  skip_on_cran()
  g <- .dc_graph(); A <- g$A; K <- nrow(A)
  csr <- tulpa:::adjacency_to_csr_tulpa(A)
  sb <- default_sigma_beta()
  set.seed(11)
  idx <- rep(seq_len(K), each = 3L); n <- length(idx); x <- rnorm(n)
  y <- 0.3 + 0.5 * x + c(0, 1.2, -1, 2)[g$comp][idx] + rnorm(n, sd = 0.4)
  X <- cbind(1, x); Z <- outer(idx, seq_len(K), "==") * 1; phi <- 0.2
  blk <- function(...) c(list(spatial_idx = idx, n_spatial_units = K,
                              adj_row_ptr = csr$row_ptr,
                              adj_col_idx = csr$col_idx,
                              n_neighbors = csr$n_neighbors), list(...))
  evid <- function(C) {
    Sg <- phi * diag(n) + sb^2 * X %*% t(X) + C
    -0.5 * (n * log(2 * pi) + as.numeric(determinant(Sg)$modulus) +
              sum(y * solve(Sg, y)))
  }
  # The augmented field covariance conditioned on the block's level row
  # a'u = 0: the level is a hard constraint and sits in the intercept
  # (gcol33/tulpa#1053), the per-component contrasts stay in the field.
  cond <- function(C, a) { Ca <- C %*% a; C - tcrossprod(Ca) / sum(a * Ca) }
  Qa <- .dc_aug(A, g$comp)
  f <- tulpa_nested_laplace(y, rep(1L, n), X, prior = blk(type = "icar", tau_grid = 2),
                            family = "gaussian", phi = phi, hyperprior = "flat",
                            control = .dc_ctl)
  expect_equal(f$log_marginal, evid(Z %*% cond(solve(2 * Qa), rep(1, K)) %*% t(Z)),
               tolerance = 1e-8)
  bs <- tulpa:::.bym2_component_scaling(A)
  Cb <- 0.81 * (0.6 * bs$scale_factor^2 *
                  cond(solve(diag(bs$node_prec) %*% Qa), bs$node_prec) +
                  0.4 * diag(K))
  f <- tulpa_nested_laplace(y, rep(1L, n), X,
    prior = blk(type = "bym2", scale_factor = bs$scale_factor,
                node_prec = bs$node_prec, sigma_grid = 0.9, rho_grid = 0.6),
    family = "gaussian", phi = phi, hyperprior = "flat", control = .dc_ctl)
  expect_equal(f$log_marginal, evid(Z %*% Cb %*% t(Z)), tolerance = 1e-8)
  set.seed(3); re <- sample(1:5, n, TRUE); Zr <- outer(re, 1:5, "==") * 1
  iid <- list(type = "iid", obs_idx = re, n_units = 5L, sigma_grid = 0.7)
  f <- tulpa_nested_laplace(y, rep(1L, n), X,
    prior = list(blk(type = "bym2", scale_factor = bs$scale_factor,
                     node_prec = bs$node_prec, sigma_grid = 0.9,
                     rho_grid = 0.6), iid),
    family = "gaussian", phi = phi, hyperprior = "flat", control = .dc_ctl)
  expect_equal(f$log_marginal,
               evid(Z %*% Cb %*% t(Z) + 0.49 * Zr %*% t(Zr)), tolerance = 1e-8)
})

test_that("the joint door centres and conditions a per-component BYM2 on its weighted level", {
  skip_on_cran()
  g <- .dc_graph(); A <- g$A; K <- nrow(A)
  csr <- tulpa:::adjacency_to_csr_tulpa(A)
  bs <- tulpa:::.bym2_component_scaling(A); w <- bs$node_prec
  set.seed(5); x <- rnorm(K)
  y <- 0.3 + 0.5 * x + c(0, 1.2, -1, 2)[g$comp] + rnorm(K)
  arm <- list(y = y, n_trials = rep(1L, K), X = cbind(1, x),
              spatial_idx = seq_len(K), re_idx = rep(0, K), n_re_groups = 0L,
              sigma_re = 1, family = "gaussian", phi = 1)
  prior <- list(type = "bym2", n_spatial_units = K, adj_row_ptr = csr$row_ptr,
                adj_col_idx = csr$col_idx, n_neighbors = csr$n_neighbors,
                scale_factor = bs$scale_factor, node_prec = w,
                sigma_grid = 0.9, rho_grid = 0.6)
  f <- suppressMessages(tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = prior, control = list(progress = FALSE)))
  phi <- f$modes[which.max(f$weights), f$arm_layout$phi_start + seq_len(K)]
  expect_lt(abs(sum(w * phi)), 1e-8)
  cc <- f$constraint_cols
  expect_identical(length(cc), 1L)
  expect_equal(cc[[1L]]$w, w)
})
