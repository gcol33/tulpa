.cpc_zero_grid <- function() {
  g <- expand.grid(l1 = seq(-1.6, 1.3, by = .1),
                   l2 = seq(-8, 1.6, by = .2),
                   y = seq(-2.6, 2.6, by = .13))
  rho <- tanh(g$y)
  list(coords = cbind(L11 = g$l1, L21 = g$y, L22 = g$l2),
       grid = cbind(L11 = g$l1, L21 = rho * exp(g$l2),
                    L22 = g$l2 + .5 * log1p(-rho^2)))
}

test_that("coordinate round trips preserve a tensor level at zero", {
  C <- as.matrix(expand.grid(L11 = c(-1, 0, 1), L21 = c(-1.3, 0, 1.3),
                            L22 = c(-8, 0, 1.6)))
  M <- tulpa:::.hp_logchol_from_coords(C, list(design = "cpc", p = 2L))
  expect_identical(tulpa:::.hp_logchol_design(M)$design, "cpc")
  snapped <- tulpa:::.hp_grid_design_coords(M, list("<>" = list(design = "cpc", p = 2L)))
  expect_equal(snapped, C, tolerance = 1e-11, ignore_attr = TRUE)
  expect_equal(vapply(seq_len(3L), function(j) length(unique(snapped[, j])), integer(1)),
               rep(3L, 3L))
})

test_that("batched covariance quantities equal the matrix definition", {
  set.seed(7L)
  p <- 3L
  M <- matrix(rnorm(600, sd = .4), 100L)
  colnames(M) <- c("L11", "L21", "L31", "L22", "L32", "L33")
  Sig <- lapply(seq_len(nrow(M)), function(r) {
    L <- matrix(0, p, p)
    L[lower.tri(L, diag = TRUE)] <- M[r, ]
    diag(L) <- exp(diag(L))
    tcrossprod(L)
  })
  w <- seq_len(nrow(M)); w <- w / sum(w)
  want <- tulpa:::.re_cov_derived_summary(lapply(Sig, list), w,
    list(list(nc = p, full = TRUE)), support = "sample")$posterior
  got <- tulpa:::.logchol_grid_derived_summary(M, w, "sample")
  expect_equal(got, want, tolerance = 1e-12)
})

test_that("covariance correlations retain their scale and support", {
  variance <- c(1e200, 1e-200, 1)
  D <- tulpa:::.re_cov_derived_columns(2L, TRUE, function(i, j)
    variance * if (i == j) 1 else .5)
  expect_equal(D[, "rho_12"], rep(.5, 3L), tolerance = 1e-14)
  Sig <- lapply(seq_len(32L), function(i) {
    a <- .1 + i / 25
    k <- if (i %% 2L) 1.3 else -1.3
    L <- rbind(c(.3, 0, 0), c(cos(a), sin(a), 0),
               c(k * cos(a), k * sin(a), 1e-20))
    tcrossprod(L)
  })
  D <- tulpa:::.re_cov_derived_columns(3L, TRUE, function(i, j)
    vapply(Sig, function(S) S[i, j], numeric(1)))
  expect_true(all(abs(D[, c("rho_12", "rho_13", "rho_23")]) <= 1))
  expect_equal(abs(D[, "rho_23"]), rep(1, 32L), tolerance = 1e-14)
})

test_that("a large stated CPC tensor keeps its normalized prior and cell measure", {
  skip_on_cran()
  g <- .cpc_zero_grid()
  expect_equal(nrow(g$grid), 60270L)
  d <- tulpa:::.hp_logchol_design(g$grid)
  expect_identical(d$design, "cpc")
  got <- tulpa:::.hp_logchol_log_density(g$grid, d)$lp
  a <- tulpa:::.nl_scale_anchor()
  lambda <- -log(a[2L]) / a[1L]
  eta <- tulpa:::.nl_hyperprior("lkj_eta")
  pc <- function(v) log(lambda) + v - lambda * exp(v)
  rho <- tanh(g$coords[, 2L])
  want <- pc(g$coords[, 1L]) + pc(g$coords[, 3L]) + eta * log1p(-rho^2) -
    tulpa:::.lkj_log_normaliser(2L, eta)
  expect_equal(got, want, tolerance = 1e-10)
  lm <- tulpa:::.hyper_logchol_log_measure(g$grid, d, absolute = TRUE)
  expect_true(all(is.finite(lm)))
  expect_lt(diff(range(lm)), 1e-10)
})
test_that("row warm starts measure signed covariance coordinates within their domain", {
  grid <- matrix(c(-3, -1, 2, 10), ncol = 1L, dimnames = list(NULL, "L11"))
  cells <- matrix(c(5, -2.8, -.9, 0), ncol = 1L, dimnames = list(NULL, "L11"))
  extras <- lapply(1:4, function(i) list(mode = as.numeric(i)))
  expect_no_warning(got <- tulpa:::.hyper_row_warm_starts(
    cells, grid, rep(0, 4L), extras, "L11", list(mode = 0)))
  expect_equal(drop(got$modes), c(4, 1, 2, 2))
})
