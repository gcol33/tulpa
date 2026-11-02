# A nested fit retains each cell's per-unit spatial field and random effects as
# loadings of its latent Gaussian: the variance is the loading's quadratic form
# in the constrained inverse precision, read off one selected inversion per
# cell. field_effects() and ranef() mix those cells over the outer weights. A
# multi-block iid block is non-centred, so its random effect is the cell's
# coordinate times the cell's sigma.

.mk_cells <- function() {
  set.seed(11)
  n_x <- 2L + 10L
  one <- function(tau) {
    Q <- matrix(0, n_x, n_x)
    Q[1:2, 1:2] <- diag(2, 2) + 0.1
    Q[3:n_x, 3:n_x] <- tau * crossprod(diff(diag(10))) + diag(0.3, 10) + 1 / 10
    Q[1, 3:n_x] <- Q[3:n_x, 1] <- 0.1
    # Coordinates 3..7 and 8..12 load the same observations pairwise.
    for (s in 1:5) Q[2 + s, 7 + s] <- Q[7 + s, 2 + s] <- 0.2
    L <- Matrix::tril(methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix"))
    list(Q = Q, p = L@p, i = L@i, x = L@x)
  }
  lapply(c(1, 4, 9), one)
}

test_that("the selected-inversion loading variance equals the dense constrained one", {
  cells <- .mk_cells()
  A <- list(3:12)
  w <- rbind(c(0.7, 0.3), c(1.1, 0.2), c(0.4, 0.9))
  idx <- as.integer(rbind(3:7, 8:12))
  W <- t(apply(w, 1L, function(r) rep(r, 5)))
  v <- cpp_loading_marginal_var(
    lapply(cells, `[[`, "p"), lapply(cells, `[[`, "i"), lapply(cells, `[[`, "x"),
    12L, idx, as.integer(seq(0, 10, by = 2)), W, A)
  for (k in 1:3) {
    S <- solve(cells[[k]]$Q)
    a <- numeric(12); a[3:12] <- 1
    Sc <- S - S %*% a %*% solve(crossprod(a, S %*% a)) %*% crossprod(a, S)
    ref <- vapply(1:5, function(s) {
      l <- numeric(12); l[2 + s] <- w[k, 1]; l[7 + s] <- w[k, 2]
      drop(crossprod(l, Sc %*% l))
    }, numeric(1))
    expect_equal(v[k, ], ref, tolerance = 1e-8)
  }
})

test_that("field_effects() and ranef() report the mixture on space-time fits", {
  skip_if_fast()
  set.seed(1)
  nr <- 5L; nc <- 5L; S <- nr * nc; n_t <- 8L
  xy <- expand.grid(col = seq_len(nc), row = seq_len(nr))
  W <- 1 * (as.matrix(stats::dist(xy, method = "manhattan")) == 1)
  u <- as.numeric(scale(xy$col + 0.5 * xy$row)) * 0.6
  v <- as.numeric(scale(sin(2 * pi * seq_len(n_t) / n_t))) * 0.5
  d <- expand.grid(region = seq_len(S), time = seq_len(n_t))
  d$x <- stats::rnorm(nrow(d))
  d$y <- stats::rpois(nrow(d), exp(1.5 + 0.4 * d$x + u[d$region] + v[d$time]))
  d$cell <- factor((d$time - 1L) * S + d$region)
  ctl <- list(n_threads = 1L, progress = FALSE)

  fa <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d,
    family = "poisson", spatial = list(type = "icar", adjacency = W),
    temporal = temporal_rw1("time"), mode = "auto", control = ctl))
  fe <- field_effects(fa)
  expect_true(all(fe$source == "mixture"))
  expect_true(all(is.finite(fe$sd) & fe$sd > 0))

  # The same additive model through fit_st_nested()'s own kernel.
  for (st in c("icar", "bym2")) {
    f3 <- fit_st_nested(d$y, cbind(1, d$x), d$region, W, d$time, n_t,
                        spatial_type = st, temporal_type = "rw1",
                        family = "poisson", control = list(n_threads = 1L))
    fe3 <- field_effects(f3)
    expect_true(all(fe3$source == "mixture"))
    expect_true(all(is.finite(fe3$sd) & fe3$sd > 0))
    if (st == "icar") {
      expect_equal(fe3$sd, fe$sd, tolerance = 0.05)
      expect_equal(fe3$estimate, fe$estimate, tolerance = 0.05)
    }
  }

  fI <- suppressWarnings(tulpa(y ~ x + spatial(region) + (1 | cell), data = d,
    family = "poisson", spatial = list(type = "icar", adjacency = W),
    temporal = temporal_rw1("time"), mode = "auto", control = ctl))
  r <- ranef(fI)
  expect_true(all(r$source == "mixture"))
  # On the effect's own scale: bounded by the posterior of its scale, far
  # below the N(0, 1) of the non-centred coordinate.
  hp <- summary(fI)$hyperparameters
  expect_true(all(r$sd < hp$ci_hi[hp$parameter == "b3.sigma"]))
})
