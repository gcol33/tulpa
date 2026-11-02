# Knorr-Held space-time interactions (spatiotemporal(), types I-IV) and the
# hard linear-constraint layer of the sparse joint Newton they ride on.
#
# The interaction's prior is N(0, (tau R)^-) conditioned on A delta = 0, with
# the rows of A spanning null(R). Against a Gaussian response at one outer cell
# the model has a closed form: delta = B z, B an orthonormal basis of null(A),
# z ~ N(0, (tau B' R B)^-1), so the evidence and the cell posterior are those of
# a Gaussian linear model with the covariance B (tau B' R B)^-1 B'. The main
# effects carry the engine's augmented intrinsic priors, whose own closed form
# test-areal-evidence-exact.R pins.

.kh_ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
                prune = FALSE, var_of_means_consistency = FALSE,
                skew_correct = FALSE, progress = FALSE)

.kh_graph <- function(island = FALSE) {
  S <- if (island) 6L else 5L
  A <- matrix(0, S, S)
  for (i in 1:4) A[i, i + 1] <- A[i + 1, i] <- 1
  A[1, 3] <- A[3, 1] <- 1
  A
}

.kh_null_basis <- function(st) {
  nd <- st$n_latent
  if (!length(st$constr)) return(diag(nd))
  Am <- t(vapply(st$constr, tulpa:::.constraint_row_dense, numeric(nd),
                 n_x = nd))
  MASS::Null(t(Am))
}

# A covariance conditioned on the sum of its coordinates being zero.
.kh_cond1 <- function(C) { c1 <- rowSums(C); C - tcrossprod(c1) / sum(c1) }

.kh_cov <- function(st, tau) {
  B <- .kh_null_basis(st)
  B %*% solve(tau * crossprod(B, as.matrix(st$R) %*% B)) %*% t(B)
}

test_that("the constraint rows span the structure's null space, independently", {
  skip_on_cran_budget()
  for (island in c(FALSE, TRUE)) {
    A <- .kh_graph(island)
    S <- nrow(A)
    for (walk in c("rw1", "rw2")) for (type in c("I", "II", "III", "IV")) {
      st <- tulpa:::.st_interaction_structure(type, A, 4L, walk, FALSE)
      R <- as.matrix(st$R)
      ev <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
      nul <- sum(ev < 1e-9)
      expect_identical(length(st$constr), nul)
      if (nul) {
        Am <- t(vapply(st$constr, tulpa:::.constraint_row_dense,
                       numeric(st$n_latent), n_x = st$n_latent))
        expect_lt(max(abs(Am %*% R)), 1e-10)
        expect_identical(qr(Am)$rank, nul)
      }
      expect_equal(st$log_pdet, sum(log(ev[ev > 1e-9])), tolerance = 1e-8)
      # The kernel's matrix R + A'A is positive definite.
      Sa <- Matrix::sparseMatrix(i = st$S_i, p = st$S_p, x = st$S_x,
                                 index1 = FALSE)
      expect_gt(min(eigen(as.matrix(Sa), symmetric = TRUE,
                          only.values = TRUE)$values), 1e-8)
    }
  }
})

test_that("an interaction's log marginal and mode are the exact Gaussian ones", {
  skip_on_cran()
  sb <- default_sigma_beta()
  A <- .kh_graph(); S <- nrow(A); n_t <- 4L
  csr <- tulpa:::adjacency_to_csr_tulpa(A)
  set.seed(2)
  d <- expand.grid(s = 1:S, t = 1:n_t)
  d <- d[rep(seq_len(nrow(d)), 2), ]
  n <- nrow(d); x <- rnorm(n); X <- cbind(1, x); phi <- 0.3
  y <- 0.2 + 0.5 * x + 0.5 * sin(d$s) + 0.2 * d$t + rnorm(n, sd = 0.4) +
    rnorm(S * n_t, sd = 0.5)[(d$t - 1) * S + d$s]
  Zs <- outer(d$s, 1:S, "==") * 1; Zt <- outer(d$t, 1:n_t, "==") * 1
  Zd <- outer((d$t - 1) * S + d$s, 1:(S * n_t), "==") * 1
  tau_s <- 2; tau_t <- 3; tau_d <- 4
  # The main effects' augmented covariances conditioned on their level rows:
  # each level is a hard constraint and sits in the intercept
  # (gcol33/tulpa#1053).
  Cs <- .kh_cond1(solve(tau_s * (diag(rowSums(A)) - A + 1 / S)))
  Ct <- .kh_cond1(solve(tau_t * (crossprod(diff(diag(n_t))) + 1 / n_t)))
  icar <- list(type = "icar", spatial_idx = d$s, n_spatial_units = S,
               adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
               n_neighbors = csr$n_neighbors, tau_grid = tau_s)
  rw <- list(type = "rw1", temporal_idx = d$t, n_times = n_t, tau_grid = tau_t)
  for (walk in c("rw1", "rw2")) for (type in c("I", "II", "III", "IV")) {
    st <- tulpa:::.st_interaction_structure(type, A, n_t, walk, FALSE)
    blk <- c(list(type = "st_interaction", obs_idx = (d$t - 1L) * S + d$s,
                  tau_grid = tau_d),
             st[c("n_latent", "S_p", "S_i", "S_x", "log_pdet", "constr")])
    f <- tulpa_nested_laplace(y, rep(1L, n), X, prior = list(icar, rw, blk),
                              family = "gaussian", phi = phi,
                              hyperprior = "flat", control = .kh_ctl)
    Cd <- .kh_cov(st, tau_d)
    Sig <- phi * diag(n) + sb^2 * X %*% t(X) + Zs %*% Cs %*% t(Zs) +
      Zt %*% Ct %*% t(Zt) + Zd %*% Cd %*% t(Zd)
    ev <- -0.5 * (n * log(2 * pi) + as.numeric(determinant(Sig)$modulus) +
                    sum(y * solve(Sig, y)))
    expect_equal(f$log_marginal, ev, tolerance = 1e-8)
    off <- f$block_latent_offsets
    md <- as.numeric(Cd %*% t(Zd) %*% solve(Sig, y))
    expect_equal(as.numeric(f$modes[1, (off[3] + 1):off[4]]), md,
                 tolerance = 1e-8)
  }
})

.kh_front_data <- function(seed = 3) {
  A <- .kh_graph(island = TRUE); S <- nrow(A); n_t <- 5L
  set.seed(seed)
  d <- expand.grid(region = 1:S, year = 1:n_t)
  d <- d[rep(seq_len(nrow(d)), 2), ]
  d$x <- rnorm(nrow(d))
  d$y <- 0.2 + 0.5 * d$x +
    rnorm(S * n_t, sd = 0.6)[(d$year - 1) * S + d$region] +
    rnorm(nrow(d), sd = 0.5)
  list(A = A, S = S, n_t = n_t, d = d)
}

test_that("spatiotemporal_effects() on a tulpa() fit is the exact cell posterior", {
  skip_on_cran()
  fx <- .kh_front_data(); d <- fx$d; A <- fx$A; S <- fx$S; n_t <- fx$n_t
  phi <- 0.25; tau_s <- 2; tau_t <- 3; tau_d <- 1.5
  tm <- temporal_rw1("year"); tm$tau_grid <- tau_t
  f <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = phi,
             spatial = list(type = "icar", adjacency = A, tau_grid = tau_s),
             temporal = tm,
             spatiotemporal = spatiotemporal(type = "IV", tau_grid = tau_d),
             mode = "nested_laplace", control = .kh_ctl)
  se <- spatiotemporal_effects(f, format = "summary")
  expect_s3_class(se, "tulpa_st_summary")
  expect_identical(nrow(se), S * n_t)
  sb <- tulpa:::.tulpa_prior_sd()
  st <- tulpa:::.st_interaction_structure("IV", A, n_t, "rw1", FALSE)
  Cd <- .kh_cov(st, tau_d)
  aug <- diag(rowSums(A)) - A
  for (cc in tulpa:::.graph_components(A))
    aug[cc, cc] <- aug[cc, cc] + 1 / length(cc)
  Cs <- .kh_cond1(solve(tau_s * aug))
  Ct <- .kh_cond1(solve(tau_t * (crossprod(diff(diag(n_t))) + 1 / n_t)))
  Zs <- outer(d$region, 1:S, "==") * 1; Zt <- outer(d$year, 1:n_t, "==") * 1
  Zd <- outer((d$year - 1) * S + d$region, 1:(S * n_t), "==") * 1
  X <- cbind(1, d$x)
  Sig <- phi * diag(nrow(d)) + sb^2 * X %*% t(X) + Zs %*% Cs %*% t(Zs) +
    Zt %*% Ct %*% t(Zt) + Zd %*% Cd %*% t(Zd)
  md <- as.numeric(Cd %*% t(Zd) %*% solve(Sig, d$y))
  vd <- diag(Cd - Cd %*% t(Zd) %*% solve(Sig, Zd %*% Cd))
  expect_equal(se$mean, md, tolerance = 1e-8)
  expect_equal(se$sd, sqrt(pmax(vd, 0)), tolerance = 1e-6)

  # Joint draws: every draw satisfies the constraints, and their moments
  # reproduce the exact summary to Monte Carlo error.
  arr <- spatiotemporal_effects(f, format = "array", ndraws = 4000L, seed = 1)
  expect_identical(dim(arr), c(S, n_t, 4000L))
  comps <- tulpa:::.graph_components(A)
  for (cc in comps) {
    expect_lt(max(abs(apply(arr[cc, , , drop = FALSE], c(2, 3), sum))), 1e-10)
  }
  expect_lt(max(abs(apply(arr, c(1, 3), sum))), 1e-10)
  ok <- se$sd > 0.05
  z <- (c(apply(arr, c(1, 2), mean)) - se$mean)[ok] / (se$sd[ok] / sqrt(4000))
  expect_lt(max(abs(z)), 4.5)
  expect_lt(max(abs(c(apply(arr, c(1, 2), sd))[ok] / se$sd[ok] - 1)), 0.08)
})

test_that("a type-IV fit's fixed-effect covariance is the exact Gaussian one", {
  # The stored precision carries no curvature on the opposite shift of the
  # ICAR and RW1 levels (+c on one, -c on the other); the reported covariance
  # is conditioned on the constraint rows all the same.
  skip_on_cran()
  fx <- .kh_front_data(); d <- fx$d; A <- fx$A; S <- fx$S; n_t <- fx$n_t
  phi <- 0.25; tau_s <- 2; tau_t <- 3; tau_d <- 1.5
  tm <- temporal_rw1("year"); tm$tau_grid <- tau_t
  f <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = phi,
             spatial = list(type = "icar", adjacency = A, tau_grid = tau_s),
             temporal = tm,
             spatiotemporal = spatiotemporal(type = "IV", tau_grid = tau_d),
             mode = "nested_laplace", control = .kh_ctl)
  sb <- tulpa:::.tulpa_prior_sd()
  st <- tulpa:::.st_interaction_structure("IV", A, n_t, "rw1", FALSE)
  Cd <- .kh_cov(st, tau_d)
  aug <- diag(rowSums(A)) - A
  for (cc in tulpa:::.graph_components(A))
    aug[cc, cc] <- aug[cc, cc] + 1 / length(cc)
  Cs <- .kh_cond1(solve(tau_s * aug))
  Ct <- .kh_cond1(solve(tau_t * (crossprod(diff(diag(n_t))) + 1 / n_t)))
  Zs <- outer(d$region, 1:S, "==") * 1; Zt <- outer(d$year, 1:n_t, "==") * 1
  Zd <- outer((d$year - 1) * S + d$region, 1:(S * n_t), "==") * 1
  X <- cbind(1, d$x)
  Sig0 <- phi * diag(nrow(d)) + Zs %*% Cs %*% t(Zs) + Zt %*% Ct %*% t(Zt) +
    Zd %*% Cd %*% t(Zd)
  V <- solve(crossprod(X, solve(Sig0, X)) + diag(1 / sb^2, 2))
  expect_equal(unname(solve(f$grid_hessians[[1L]])), V, tolerance = 1e-8)
  expect_equal(unname(vcov(f)), V, tolerance = 1e-8)

  # The per-row predictive variance, read off the cell's selected inverse.
  K <- sb^2 * X %*% t(X) + Sig0 - phi * diag(nrow(d))
  ve <- diag(K - K %*% solve(Sig0 + sb^2 * X %*% t(X), K))
  expect_equal(as.numeric(f$fitted_eta_var), ve, tolerance = 1e-8)
})

test_that("a simulated type-IV interaction is recovered", {
  skip_on_cran()
  nr <- 3L; nc <- 4L; S <- nr * nc; n_t <- 8L
  A <- matrix(0, S, S); id <- function(i, j) (j - 1L) * nr + i
  for (i in 1:nr) for (j in 1:nc) {
    if (i < nr) A[id(i, j), id(i + 1, j)] <- A[id(i + 1, j), id(i, j)] <- 1
    if (j < nc) A[id(i, j), id(i, j + 1)] <- A[id(i, j + 1), id(i, j)] <- 1
  }
  st <- tulpa:::.st_interaction_structure("IV", A, n_t, "rw1", FALSE)
  B <- .kh_null_basis(st)
  set.seed(1)
  delta <- as.numeric(B %*% backsolve(chol(2 * crossprod(B, as.matrix(st$R) %*% B)),
                                      rnorm(ncol(B))))
  set.seed(101)
  u <- rnorm(S, sd = 0.3); u <- u - mean(u)
  v <- cumsum(rnorm(n_t, sd = 0.15)); v <- v - mean(v)
  d <- expand.grid(region = 1:S, year = 1:n_t)
  d$E <- 40
  d$y <- rpois(nrow(d), exp(log(d$E) - 1 + u[d$region] + v[d$year] +
                              delta[(d$year - 1) * S + d$region]))
  f <- tulpa(y ~ 1 + offset(log(E)) + spatial(region), d, family = "poisson",
             spatial = list(type = "icar", adjacency = A),
             temporal = temporal_rw1("year"),
             spatiotemporal = spatiotemporal(type = "IV"),
             mode = "nested_laplace", control = list(progress = FALSE))
  se <- spatiotemporal_effects(f, format = "summary")
  # Measured at this seed: cor 0.915, coverage 90 / 96 cells.
  expect_gt(cor(se$mean, delta), 0.85)
  expect_gt(mean(delta >= se$q2.5 & delta <= se$q97.5), 0.85)
  expect_lt(max(abs(tapply(se$mean, se$t, sum))), 1e-8)
  expect_lt(max(abs(tapply(se$mean, se$s, sum))), 1e-8)
})

test_that("spatiotemporal() refuses what its structure does not carry", {
  fx <- .kh_front_data()
  expect_error(
    tulpa(y ~ x + spatial(region), fx$d, family = "gaussian",
          spatial = list(type = "icar", adjacency = fx$A),
          temporal = temporal_ar1("year"),
          spatiotemporal = spatiotemporal(type = "IV"),
          mode = "nested_laplace", control = .kh_ctl),
    "intrinsic walk")
  expect_error(spatiotemporal(type = "V"))
  expect_error(spatiotemporal(tau_grid = -1), "positive")
  expect_identical(spatiotemporal(type = "separable")$type, "IV")
  expect_identical(spatiotemporal(type = "iid")$type, "I")
})
