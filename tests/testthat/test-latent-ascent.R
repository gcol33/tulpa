# tulpa_latent_ascent(): the block-coordinate ascent over a grouped model with
# a shared GMRF field and / or latent factors, and the pieces it is built from
# (the Gauss-Hermite rule, the joint site marginal, the factor magnitude, the
# field solves, the ridge ladder), each checked against an independent
# reference. The family enters only through an oracle callback; these tests
# write their own Poisson / Bernoulli oracles.

.la_oracle <- function(y, link = "log") {
  if (identical(link, "logit")) {
    ll_cell <- function(eta, idx = NULL) {
      e  <- if (is.null(idx)) eta else eta[idx, , drop = FALSE]
      ym <- if (is.null(idx)) y else y[idx, , drop = FALSE]
      ifelse(ym > 0, stats::plogis(e, log.p = TRUE), stats::plogis(-e, log.p = TRUE))
    }
    return(list(
      working = function(eta) {
        p <- stats::plogis(eta); list(score = y - p, curv = p * (1 - p))
      },
      ll_cell = ll_cell, data_ll = function(eta) sum(ll_cell(eta))))
  }
  ll_cell <- function(eta, idx = NULL) {
    e  <- if (is.null(idx)) eta else eta[idx, , drop = FALSE]
    ym <- if (is.null(idx)) y else y[idx, , drop = FALSE]
    e  <- pmin(e, 700)
    ym * e - exp(e)
  }
  list(working = function(eta) {
         mu <- exp(pmin(eta, 700)); list(score = y - mu, curv = mu)
       },
       ll_cell = ll_cell, data_ll = function(eta) sum(ll_cell(eta)))
}

.la_chain <- function(n) {
  A <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- 1
  A
}

# A per-species intercept "community EM": Newton on each intercept given the
# offsets, enough to drive the ascent end to end.
.la_em <- function(y) {
  Ns <- nrow(y); S <- ncol(y)
  list(
    em_fit = function(site_off, fac_off, em_prev) {
      b <- if (is.null(em_prev)) rep(0, S) else em_prev
      off <- matrix(site_off, Ns, S) + fac_off
      for (it in 1:30) {
        mu <- exp(outer(rep(1, Ns), b) + off)
        b <- b + colSums(y - mu) / colSums(mu)
      }
      b
    },
    offset_of = function(b) outer(rep(1, Ns), b))
}


test_that("tulpa_gauss_hermite() integrates the standard normal", {
  for (n in c(1L, 3L, 5L, 7L, 15L)) {
    gh <- tulpa_gauss_hermite(n)
    expect_length(gh$nodes, n)
    expect_equal(sum(gh$weights), 1, tolerance = 1e-12)
    expect_false(is.unsorted(gh$nodes))
    expect_equal(sum(gh$weights * gh$nodes), 0, tolerance = 1e-10)
    if (n >= 2L) expect_equal(sum(gh$weights * gh$nodes^2), 1, tolerance = 1e-10)
    if (n >= 3L) expect_equal(sum(gh$weights * gh$nodes^4), 3, tolerance = 1e-10)
    if (n >= 5L) expect_equal(sum(gh$weights * gh$nodes^6), 15, tolerance = 1e-9)
  }
  expect_identical(tulpa_gauss_hermite(9L), tulpa:::gauss_hermite_prob(9L))
})


test_that("tulpa_car_precision_Q() is D - rho W", {
  A <- .la_chain(5L)
  expect_equal(as.matrix(tulpa_car_precision_Q(A)), diag(rowSums(A)) - A,
               ignore_attr = TRUE)
  expect_equal(as.matrix(tulpa_car_precision_Q(A, rho = 0.7, tau = 2)),
               2 * (diag(rowSums(A)) - 0.7 * A), ignore_attr = TRUE)
  # The spec-based builder is the same matrix.
  expect_identical(tulpa:::.car_proper_precision_Q(list(adjacency = A), 2, 0.7),
                   tulpa_car_precision_Q(A, rho = 0.7, tau = 2))
})


test_that("the joint site marginal matches direct numerical integration", {
  set.seed(4)
  Ns <- 6L; S <- 4L
  y   <- matrix(rbinom(Ns * S, 1, 0.5), Ns, S)
  eta <- matrix(rnorm(Ns * S, 0, 0.7), Ns, S)
  lam <- matrix(rnorm(S, 0, 0.9), S, 1L)
  or  <- .la_oracle(y, "logit")
  ref <- sum(vapply(seq_len(Ns), function(i) {
    f <- function(z) vapply(z, function(zz) {
      p <- stats::plogis(eta[i, ] + as.numeric(lam) * zz)
      prod(ifelse(y[i, ] > 0, p, 1 - p)) * stats::dnorm(zz)
    }, numeric(1))
    log(stats::integrate(f, -12, 12, rel.tol = 1e-10)$value)
  }, numeric(1)))
  expect_equal(tulpa:::.latent_joint_marginal(or, eta, lam,
                                              tulpa_gauss_hermite(25L)),
               ref, tolerance = 1e-7)

  # Q = 2 against a dense tensor rule.
  lam2 <- matrix(rnorm(S * 2L, 0, 0.8), S, 2L)
  dense <- tulpa:::.latent_joint_marginal(or, eta, lam2, tulpa_gauss_hermite(35L))
  expect_equal(tulpa:::.latent_joint_marginal(or, eta, lam2,
                                              tulpa_gauss_hermite(9L)),
               dense, tolerance = 1e-4)
})


test_that("the factor scale discounts a known inflation of the loadings", {
  skip_on_cran()
  set.seed(11)
  Ns <- 400L; S <- 14L; Q <- 2L
  lam  <- matrix(rnorm(S * Q, 0, 0.8), S, Q)
  zeta <- matrix(rnorm(Ns * Q), Ns, Q)
  eta  <- matrix(rnorm(Ns * S, 0, 0.5), Ns, S)
  y    <- matrix(rbinom(Ns * S, 1, stats::plogis(eta + tcrossprod(zeta, lam))),
                 Ns, S)
  or <- .la_oracle(y, "logit")
  gh <- tulpa_gauss_hermite(5L)
  for (infl in c(1.5, 2)) {
    cc <- tulpa:::.latent_factor_scale(or, eta, lam * infl, gh)
    expect_false(attr(cc, "saturated"))
    expect_equal(as.numeric(cc), 1 / infl, tolerance = 0.35)
  }
  # an all-zero loading matrix has no magnitude to set
  expect_identical(tulpa:::.latent_factor_scale(or, eta, matrix(0, S, Q), gh), 1)
  # the adaptive 5-node rule reaches the converged argmax
  ref <- tulpa:::.latent_factor_scale(or, eta, lam * 1.5, tulpa_gauss_hermite(21L))
  for (n in c(5L, 9L, 15L)) {
    expect_equal(tulpa:::.latent_factor_scale(or, eta, lam * 1.5,
                                              tulpa_gauss_hermite(n)),
                 ref, tolerance = 0.01)
  }
})


test_that("the factor scale recovers the magnitude on a Poisson oracle", {
  skip_on_cran()
  # A Poisson site carries a sharply peaked integrand that prior-scale nodes do
  # not resolve; the adaptive rule reaches the converged argmax at 5 nodes.
  set.seed(11)
  Ns <- 400L; S <- 14L; Q <- 1L
  lam  <- matrix(rnorm(S * Q, 0, 0.8), S, Q)
  zeta <- matrix(rnorm(Ns * Q), Ns, Q)
  eta  <- matrix(rnorm(Ns * S, log(4), 0.4), Ns, S)
  y    <- matrix(rpois(Ns * S, exp(eta + tcrossprod(zeta, lam))), Ns, S)
  or   <- .la_oracle(y, "log")
  cc <- tulpa:::.latent_factor_scale(or, eta, lam, tulpa_gauss_hermite(5L))
  expect_false(attr(cc, "saturated"))
  expect_gt(as.numeric(cc), 0.75)
  expect_lt(as.numeric(cc), 1.35)
  for (infl in c(1.5, 2)) {
    ci <- tulpa:::.latent_factor_scale(or, eta, lam * infl, tulpa_gauss_hermite(5L))
    expect_equal(as.numeric(ci) * infl, as.numeric(cc), tolerance = 0.2)
  }
  for (n in c(9L, 15L)) {
    expect_equal(as.numeric(tulpa:::.latent_factor_scale(
      or, eta, lam, tulpa_gauss_hermite(n))), as.numeric(cc), tolerance = 0.05)
  }
})


test_that("the factor scale expands its bracket instead of clamping", {
  skip_on_cran()
  set.seed(5)
  Ns <- 200L; S <- 8L
  lam  <- matrix(rnorm(S, 0, 0.9), S, 1L)
  zeta <- matrix(rnorm(Ns), Ns, 1L)
  eta  <- matrix(rnorm(Ns * S, 0, 0.5), Ns, S)
  y    <- matrix(rbinom(Ns * S, 1, plogis(eta + tcrossprod(zeta, lam))), Ns, S)
  or   <- .la_oracle(y, "logit")
  # loadings 10x too large: the corrective scale ~0.1 sits below the initial
  # bracket's floor of 0.2 and is found by expanding it
  cc <- tulpa:::.latent_factor_scale(or, eta, lam * 10, tulpa_gauss_hermite(5L))
  expect_lt(as.numeric(cc), 0.2)
  expect_false(attr(cc, "saturated"))
  expect_equal(as.numeric(cc) * 10, 1, tolerance = 0.4)
})


test_that("the ridge ladder solves, retries and never builds a NaN ridge", {
  H <- Matrix::Diagonal(4, c(2, 3, 4, 5))
  g <- c(1, 1, 1, 1)
  expect_equal(tulpa:::.latent_ridge_solve(H, g), as.numeric(solve(as.matrix(H), g)))
  expect_equal(as.matrix(tulpa:::.latent_ridge_solve(H)), solve(as.matrix(H)))
  # A singular H still returns a finite step rather than throwing.
  Hs <- Matrix::Matrix(matrix(1, 2, 2), sparse = TRUE)
  expect_true(all(is.finite(tulpa:::.latent_ridge_solve(Hs, c(1, 1)))))
  expect_true(all(is.finite(as.matrix(tulpa:::.latent_ridge_solve(Hs)))))
  # A non-finite mean diagonal falls back to a unit ridge scale.
  src <- paste(deparse(body(tulpa:::.latent_ridge_solve)), collapse = " ")
  expect_true(grepl("is.finite(d) && d > 0", src, fixed = TRUE))
  # The field Newtons and the factor Newton all retry through it (the field
  # solves via the constrained step), and the field Newton holds a non-finite
  # step before it reaches F.
  for (fn in c(".latent_constrained_step", ".latent_factor_update")) {
    fsrc <- paste(deparse(get(fn, envir = asNamespace("tulpa"))), collapse = " ")
    expect_true(grepl(".latent_ridge_solve", fsrc, fixed = TRUE), info = fn)
  }
  for (fn in c(".latent_field_solve", ".latent_bym2_solve")) {
    fsrc <- paste(deparse(get(fn, envir = asNamespace("tulpa"))), collapse = " ")
    expect_true(grepl(".latent_constrained_step", fsrc, fixed = TRUE), info = fn)
  }
  fsrc <- paste(deparse(tulpa:::.latent_field_solve), collapse = " ")
  expect_gt(regexpr("Fm + t * S", fsrc, fixed = TRUE),
            regexpr("all(is.finite(step))", fsrc, fixed = TRUE))
})


test_that("the constrained step is the Newton step on the constrained subspace", {
  # On a concave quadratic f(x) = g'x - x'Hx/2 one step from 0 has to land on
  # the maximiser over {C x = 0}, the solution of the KKT system.
  set.seed(3)
  n <- 6L
  B <- matrix(stats::rnorm(n * n), n); H <- crossprod(B) + diag(n)
  g <- stats::rnorm(n)
  Cm <- tulpa:::.latent_constraint_matrix(list(1:3, 4:6), n)
  s <- tulpa:::.latent_constrained_step(Matrix::Matrix(H, sparse = TRUE), g, Cm)
  kkt <- solve(rbind(cbind(H, t(Cm)), cbind(Cm, matrix(0, 2, 2))), c(g, 0, 0))
  expect_equal(s, kkt[seq_len(n)], tolerance = 1e-10)
  expect_equal(as.numeric(Cm %*% s), c(0, 0), tolerance = 1e-12)
  expect_equal(tulpa:::.latent_constrained_step(Matrix::Matrix(H, sparse = TRUE), g),
               as.numeric(solve(H, g)))
})


test_that("the field solves reach the constrained mode on a two-component graph", {
  # Two components of unequal size whose levels differ: the per-component
  # constraint leaves each level to the data, so the iteration has to find the
  # constrained stationary point (gradient constant within each component)
  # rather than cycle.
  A <- matrix(0, 36, 36)
  A[1:6, 1:6] <- .la_chain(6); A[7:36, 7:36] <- .la_chain(30)
  comps <- list(1:6, 7:36)
  set.seed(2)
  S <- 6L
  f_true <- c(rep(1.2, 6), sin(seq(0, 3, length.out = 30)) - 0.3)
  y <- matrix(stats::rpois(36 * S, exp(2 + f_true)), 36, S)
  oracle <- .la_oracle(y, "log")
  Q <- tulpa_car_precision_Q(A)
  eta0 <- matrix(2, 36, S)
  br <- tulpa:::.latent_bym2_solve(oracle, eta0, Q, numeric(36), numeric(36),
                                   a = 0.58, b = 0.42, max_iter = 200L,
                                   components = comps)
  sc <- rowSums(oracle$working(eta0 + 0.58 * br$v + 0.42 * br$u)$score)
  gv <- 0.58 * sc - as.numeric(Q %*% br$v)
  gu <- 0.42 * sc - br$u
  # u is held at overall sum zero, so its gradient is constant, not zero.
  expect_lt(diff(range(gu)), 1e-6)
  expect_equal(sum(br$u), 0, tolerance = 1e-10)
  for (cc in comps) expect_lt(diff(range(gv[cc])), 1e-6)
  expect_true(is.finite(br$m))
  fu <- tulpa:::.latent_field_solve(oracle, eta0, Q, matrix(0, 36, 1), 1,
                                    matrix(1, 36, 1), max_iter = 200L,
                                    components = comps)
  scf <- rowSums(oracle$working(eta0 + fu$Fm[, 1])$score)
  gf  <- scf - as.numeric(Q %*% fu$Fm[, 1])
  for (cc in comps) expect_lt(diff(range(gf[cc])), 1e-6)
})


test_that("the field solves centre over each connected component", {
  A <- matrix(0, 10, 10)
  A[1:5, 1:5] <- .la_chain(5); A[6:10, 6:10] <- .la_chain(5)
  comps <- list(1:5, 6:10)
  set.seed(1)
  S <- 4L
  f_true <- c(seq(-1, 1, length.out = 5), seq(1, -1, length.out = 5))
  y <- matrix(stats::rpois(10 * S, exp(1 + f_true)), 10, S)
  oracle <- .la_oracle(y, "log")
  Q <- tulpa_car_precision_Q(A)
  fu <- tulpa:::.latent_field_solve(oracle, matrix(1, 10, S), Q,
                                    matrix(0, 10, 1), 1, matrix(1, 10, 1),
                                    components = comps)
  expect_equal(sum(fu$Fm[1:5, 1]), 0, tolerance = 1e-10)
  expect_equal(sum(fu$Fm[6:10, 1]), 0, tolerance = 1e-10)
  expect_gt(stats::cor(fu$Fm[, 1], f_true), 0.9)
  br <- tulpa:::.latent_bym2_solve(oracle, matrix(1, 10, S), Q, numeric(10),
                                   numeric(10), a = 0.8, b = 0.3,
                                   components = comps)
  expect_equal(sum(br$v[1:5]), 0, tolerance = 1e-10)
  expect_equal(sum(br$v[6:10]), 0, tolerance = 1e-10)
})


test_that("the constrained log-determinant drops one eigenvalue per constraint", {
  A <- .la_chain(6L)
  L <- tulpa_car_precision_Q(A)
  ev <- sort(eigen(as.matrix(L), symmetric = TRUE)$values, decreasing = TRUE)
  expect_equal(tulpa:::.latent_logdet(L, 1L), sum(log(ev[1:5])))
  P <- tulpa_car_precision_Q(A, rho = 0.5)
  expect_equal(tulpa:::.latent_logdet(P, 0L),
               as.numeric(determinant(as.matrix(P))$modulus))
})


test_that("the hyperparameter profile keeps the best cell, first axis fastest", {
  seen <- list()
  best <- tulpa:::.latent_profile_hyper(
    list(a = c(1, 2, 3), b = c(10, 20)),
    function(h) {
      seen[[length(seen) + 1L]] <<- h
      if (h[["a"]] == 3 && h[["b"]] == 20) stop("solver failure")
      list(m = -(h[["a"]] - 2)^2 - (h[["b"]] - 20)^2 / 100, tag = h[["a"]])
    })
  expect_identical(best$hyper, c(a = 2, b = 20))
  expect_identical(best$tag, 2)
  expect_identical(vapply(seen, function(h) h[["a"]], 0), c(1, 2, 3, 1, 2, 3))
  expect_null(tulpa:::.latent_profile_hyper(list(a = 1:2),
                                            function(h) list(m = -Inf)))
})


test_that("tulpa_latent_ascent() fits a shared field with a profiled CAR rho", {
  set.seed(3)
  side <- 5L; n <- side^2
  A <- matrix(0, n, n); id <- matrix(seq_len(n), side)
  for (i in 1:side) for (j in 1:side) {
    if (i < side) A[id[i, j], id[i + 1, j]] <- A[id[i + 1, j], id[i, j]] <- 1
    if (j < side) A[id[i, j], id[i, j + 1]] <- A[id[i, j + 1], id[i, j]] <- 1
  }
  co <- expand.grid(r = 1:side, c = 1:side)
  f <- 0.7 * scale(sin(co$r / side * pi) + cos(co$c / side * pi))[, 1]
  f <- f - mean(f)
  S <- 6L
  b <- rnorm(S, 1, 0.3)
  y <- matrix(rpois(n * S, exp(outer(f, rep(1, S)) + outer(rep(1, n), b))), n, S)
  em <- .la_em(y)
  geom <- list(type = "car_proper", n_nodes = n, M = NULL, W = matrix(1, n, 1L),
               components = list(seq_len(n)),
               Q = function(h) tulpa_car_precision_Q(A, rho = h[["rho"]]),
               hyper = c(rho = 0.95), hyper_grid = list(rho = c(0.5, 0.8, 0.95)))
  seed_before <- .Random.seed
  res <- tulpa_latent_ascent(function(e) .la_oracle(y), em$em_fit, em$offset_of,
                             n_sites = n, n_species = S, geometry = geom)
  expect_identical(.Random.seed, seed_before)
  expect_true(res$field$hyper[["rho"]] %in% c(0.5, 0.8, 0.95))
  expect_equal(sum(res$field$Fm), 0, tolerance = 1e-8)
  expect_gt(stats::cor(res$field$site_off, f), 0.8)
  expect_null(res$factor)
  expect_identical(res$settings$max_outer, 400L)
  expect_true(is.na(res$settings$factor_starts))
})


test_that("tulpa_latent_ascent() recovers a factor co-occurrence and is reproducible", {
  skip_on_cran()
  set.seed(7)
  Ns <- 80L; S <- 6L
  lam <- matrix(rnorm(S, 0, 0.8), S, 1L)
  zeta <- rnorm(Ns)
  b <- rnorm(S, 1, 0.3)
  y <- matrix(rpois(Ns * S, exp(outer(rep(1, Ns), b) + tcrossprod(zeta, lam))),
              Ns, S)
  em <- .la_em(y)
  start <- list(zeta = matrix(scale(cos(seq_len(Ns)))[, 1], Ns, 1L),
                lambda = matrix(0.1, S, 1L))
  fit <- function() tulpa_latent_ascent(
    function(e) .la_oracle(y), em$em_fit, em$offset_of, n_sites = Ns,
    n_species = S, factors = start,
    control = list(max_outer = 4L, factor_starts = 3L))
  r1 <- fit(); r2 <- fit()
  expect_identical(r1$factor$lambda, r2$factor$lambda)
  expect_identical(r1$factor$n_factors, 1L)
  expect_identical(dim(r1$factor$offset), c(Ns, S))
  expect_gt(abs(stats::cor(r1$factor$lambda[, 1], lam[, 1])), 0.8)
  expect_equal(sqrt(sum(r1$factor$lambda^2)) / sqrt(sum(lam^2)), 1,
               tolerance = 0.5)
  expect_identical(r1$settings$factor_starts, 3L)
  expect_identical(r1$settings$n_quad, 5L)
  expect_true(is.finite(r1$factor$marginal_loglik))
})


test_that("tulpa_latent_ascent() validates its inputs", {
  em <- .la_em(matrix(1, 3, 2))
  expect_error(tulpa_latent_ascent(function(e) NULL, em$em_fit, em$offset_of,
                                   n_sites = 3, n_species = 2),
               "give a field")
  expect_error(tulpa_latent_ascent(function(e) NULL, em$em_fit, em$offset_of,
                                   n_sites = 3, n_species = 2,
                                   factors = list(zeta = matrix(0, 3, 1),
                                                  lambda = matrix(0, 2, 1)),
                                   control = list(max.outer = 2)),
               "unknown control key")
})
