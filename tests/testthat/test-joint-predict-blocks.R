# predict() on a direct tulpa_nested_laplace_joint() fit at new rows of one
# arm: each row's loadings come from the kernel's own eta walk over the arm
# (cpp_joint_arm_loadings()), each cell's covariance from its stored precision.
# The reference is written out densely: per cell, the loading row a built by
# hand (design row, the block's amplitude at the row's unit), the constrained
# covariance inverted from the precision, and the Gaussian mixture over the
# outer weights.

skip_on_cran()

.jpb_lattice <- function(side) {
  K <- side^2
  W <- matrix(0, K, K)
  for (i in 1:K) for (j in 1:K) {
    if (abs((i - 1) %/% side - (j - 1) %/% side) +
        abs((i - 1) %% side - (j - 1) %% side) == 1) W[i, j] <- 1
  }
  W
}

.jpb_fixture <- function(seed = 5L, side = 5L, sd_z = 0.6) {
  set.seed(seed)
  nu <- side^2
  csr <- tulpa:::adjacency_to_csr_tulpa(.jpb_lattice(side))
  n1 <- 200L; n2 <- 150L
  u1 <- sample(nu, n1, TRUE); u2 <- sample(nu, n2, TRUE)
  z <- stats::rnorm(nu, 0, sd_z); z <- z - mean(z)
  x1 <- stats::rnorm(n1); x2 <- stats::rnorm(n2)
  list(
    nu = nu, z = z, u1 = u1, u2 = u2,
    resp = list(
      a = list(y = stats::rpois(n1, exp(0.3 + 0.5 * x1 + z[u1])),
               X = cbind(1, x1), family = "poisson"),
      b = list(y = stats::rbinom(n2, 1, stats::plogis(-0.2 + 0.8 * z[u2])),
               X = cbind(1, x2), family = "binomial", n_trials = rep(1L, n2))),
    icar = list(type = "icar", n_spatial_units = nu,
                adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
                n_neighbors = csr$n_neighbors, spatial_idx = list(u1, u2)))
}

# The dense mixture of a_k' x over the cells: `cols(Xn, un)` gives each row's
# latent loadings as an n x n_x matrix at cell j via `load(j, A)`.
.jpb_dense <- function(fit, load) {
  w <- fit$weights
  on <- which(w > 0)
  w <- w[on] / sum(w[on])
  n_x <- ncol(fit$modes)
  mus <- vars <- NULL
  for (c in seq_along(on)) {
    j <- on[c]
    A <- load(j)
    # The cell's inner Gaussian: the stored precision plus the pins the solve
    # folded (soft rows, precision D), conditioned on the hard rows.
    cons <- tulpa:::.inner_constraint(fit, j)
    At <- if (length(cons$hard))
      as.matrix(tulpa:::.constraint_rows_matrix(cons$hard, n_x))
    Q <- as.matrix(Matrix::sparseMatrix(i = fit$Q_csc_i_per_grid[[j]] + 1L,
                                        p = fit$Q_csc_p_per_grid[[j]],
                                        x = fit$Q_csc_x_per_grid[[j]],
                                        dims = c(n_x, n_x), symmetric = TRUE))
    if (!is.null(cons$soft_D)) {
      Sm <- as.matrix(tulpa:::.constraint_rows_matrix(cons$soft, n_x))
      Q <- Q + Sm %*% as.matrix(cons$soft_D[[1L]]) %*% t(Sm)
    }
    S <- solve(Q)
    if (!is.null(At)) {
      SA <- S %*% At
      S <- S - SA %*% solve(t(At) %*% SA, t(SA))
    }
    mus <- rbind(mus, as.numeric(A %*% fit$modes[j, ]))
    vars <- rbind(vars, rowSums((A %*% S) * A))
  }
  m <- colSums(w * mus)
  list(mean = m, sd = sqrt(colSums(w * (vars + mus^2)) - m^2))
}

.jpb_loader <- function(fit, k, Xn, un, nu, amp = function(j) 1) {
  n_x <- ncol(fit$modes)
  bcols <- fit$arm_layout$beta_start[k] + seq_len(ncol(Xn))
  fcols <- n_x - nu + seq_len(nu)
  function(j) {
    A <- matrix(0, nrow(Xn), n_x)
    A[, bcols] <- Xn
    A[cbind(seq_len(nrow(Xn)), fcols[un])] <- amp(j)
    A
  }
}

test_that("an areal block is read at the new rows' units on both arms and both doors (#1102)", {
  fx <- .jpb_fixture()
  Xn <- cbind(1, c(-1, 0, 0.5, 2))
  un <- c(1L, 7L, 13L, 25L)
  fm <- tulpa_nested_laplace_joint(responses = fx$resp, prior = list(fx$icar),
                                   control = list(diagnose_k = FALSE))
  # The precision is kept by default when a block is present.
  expect_true(is.list(fm$Q_csc_x_per_grid))
  for (k in 1:2) {
    pr <- predict(fm, newdata = list(X = Xn, index = list(un)),
                  arm = names(fx$resp)[k], se.fit = TRUE)
    ref <- .jpb_dense(fm, .jpb_loader(fm, k, Xn, un, fx$nu))
    expect_equal(pr$fit, ref$mean, tolerance = 1e-10)
    expect_equal(pr$se.fit, ref$sd, tolerance = 1e-8)
  }
  # The point prediction is the same mixture mean, and the training rows
  # carry the fit's own linear predictor.
  expect_equal(predict(fm, newdata = list(X = Xn, index = list(b1 = un)),
                       arm = "a"),
               .jpb_dense(fm, .jpb_loader(fm, 1L, Xn, un, fx$nu))$mean,
               tolerance = 1e-10)
  # A row without the intercept reads the field alone; the centering fold the
  # fit checks its design against does not apply to rows only predicted at.
  Xz <- matrix(0, length(un), 2L)
  pz <- predict(fm, newdata = list(X = Xz, index = list(un)), arm = "a",
                se.fit = TRUE)
  ref <- .jpb_dense(fm, .jpb_loader(fm, 1L, Xz, un, fx$nu))
  expect_equal(pz$fit, ref$mean, tolerance = 1e-10)
  expect_equal(pz$se.fit, ref$sd, tolerance = 1e-8)
  # The single-block door: the same field given by the arms' spatial_idx.
  resp1 <- fx$resp
  resp1$a$spatial_idx <- fx$u1
  resp1$b$spatial_idx <- fx$u2
  p1 <- fx$icar
  p1$spatial_idx <- NULL
  fs <- tulpa_nested_laplace_joint(responses = resp1, prior = p1,
                                   control = list(diagnose_k = FALSE))
  ps <- predict(fs, newdata = list(X = Xn, index = list(un)), arm = "a",
                se.fit = TRUE)
  ref <- .jpb_dense(fs, .jpb_loader(fs, 1L, Xn, un, fx$nu))
  expect_equal(ps$fit, ref$mean, tolerance = 1e-10)
  expect_equal(ps$se.fit, ref$sd, tolerance = 1e-8)
  # A row's unit has to be given.
  expect_error(predict(fm, newdata = list(X = Xn), arm = "a"),
               "newdata\\$index")
})

test_that("a copy arm reads the field at alpha x sigma (#1102)", {
  fx <- .jpb_fixture()
  Xn <- cbind(1, c(-1, 0.5, 2))
  un <- c(3L, 12L, 20L)
  fc <- tulpa_nested_laplace_joint(
    responses = fx$resp, prior = list(fx$icar),
    copy = list(arm = "b", block = 1L, alpha_grid = c(0, 0.5, 1, 1.5)),
    control = list(diagnose_k = FALSE, axis_refine = "none"))
  tg <- fc$theta_grid
  for (k in 1:2) {
    amp <- if (k == 1L) function(j) tg[j, "b1.sigma"]
           else function(j) tg[j, "b1.alpha"] * tg[j, "b1.sigma"]
    pr <- predict(fc, newdata = list(X = Xn, index = list(un)),
                  arm = names(fx$resp)[k], se.fit = TRUE)
    ref <- .jpb_dense(fc, .jpb_loader(fc, k, Xn, un, fx$nu, amp))
    expect_equal(pr$fit, ref$mean, tolerance = 1e-10)
    expect_equal(pr$se.fit, ref$sd, tolerance = 1e-8)
  }
})

test_that("a non-centred iid block and an arm's random intercepts are read at the new rows (#1102)", {
  set.seed(21)
  G <- 15L; N <- 600L
  grp <- sample(G, N, TRUE)
  x <- stats::rnorm(N)
  b <- stats::rnorm(G, 0, 0.6)
  h <- sample(8L, N, TRUE)
  arm <- list(y = stats::rpois(N, exp(0.3 + 0.5 * x + b[grp])),
              X = cbind(1, x), family = "poisson", re_idx = h, n_re_groups = 8L,
              sigma_re = 0.5)
  fi <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = arm),
    prior = list(list(type = "iid", n_units = G, obs_idx = list(grp))),
    control = list(diagnose_k = FALSE)))
  Xn <- cbind(1, c(0, 1, -1))
  gn <- c(2L, 9L, 15L)
  hn <- c(1L, NA, 8L)
  pr <- predict(fi, newdata = list(X = Xn, index = list(gn), re_idx = hn),
                se.fit = TRUE)
  n_x <- ncol(fi$modes)
  lay <- fi$arm_layout
  st <- lay$block_start[1L]
  ref <- .jpb_dense(fi, function(j) {
    A <- matrix(0, 3, n_x)
    A[, lay$beta_start[1L] + 1:2] <- Xn
    # The coordinate enters times the cell's sigma; a fitted group's random
    # intercept is its own coordinate.
    A[cbind(1:3, st + gn)] <- fi$theta_grid[j, "b1.sigma"]
    A[cbind(c(1L, 3L), lay$re_start[1L] + hn[c(1L, 3L)])] <- 1
    A
  })
  expect_equal(pr$fit, ref$mean, tolerance = 1e-10)
  # A group the fit did not see adds its N(0, sigma_re^2) draw.
  expect_equal(pr$se.fit^2, ref$sd^2 + c(0, 0.5^2, 0), tolerance = 1e-8)
  expect_error(predict(fi, newdata = list(X = Xn, index = list(gn))),
               "re_idx")
})

test_that("a joint fit's predicted field covers the simulated one (#1102)", {
  skip_if_not_slow()
  # The field at each unit, read through a row whose design row is zero,
  # against the field that simulated the data: 90% intervals.
  hit <- unlist(lapply(1:10, function(s) {
    fx <- .jpb_fixture(seed = 100L + s)
    fm <- tulpa_nested_laplace_joint(responses = fx$resp,
                                     prior = list(fx$icar),
                                     control = list(diagnose_k = FALSE))
    pr <- predict(fm, newdata = list(X = matrix(0, fx$nu, 2L),
                                     index = list(seq_len(fx$nu))),
                  arm = "a", se.fit = TRUE, level = 0.9)
    pr$lower <= fx$z & fx$z <= pr$upper
  }))
  expect_gt(mean(hit), 0.8)
})
