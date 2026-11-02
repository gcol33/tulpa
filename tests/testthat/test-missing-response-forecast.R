# A row whose response is NA has no likelihood term on the nested-Laplace
# kernels (builtin_row_missing), while its linear predictor is still formed:
# a fit with NA rows is the fit of the observed rows, and the NA rows' eta is
# the posterior predictive of the latent -- a forecast.

.na_ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
                prune = FALSE, var_of_means_consistency = FALSE,
                skew_correct = FALSE, progress = FALSE)

.na_fixture <- function() {
  S <- 8L; n_t <- 6L
  A <- matrix(0, S, S)
  for (i in 1:(S - 1)) A[i, i + 1] <- A[i + 1, i] <- 1
  set.seed(4)
  d <- expand.grid(region = 1:S, year = 1:n_t)
  d$x <- rnorm(nrow(d))
  d$y <- rpois(nrow(d), exp(0.5 + 0.3 * d$x + 0.2 * sin(d$region) +
                              0.1 * d$year))
  list(A = A, S = S, n_t = n_t, d = d, na = d$year == n_t)
}

test_that("an NA response is masked on every nested kernel", {
  skip_on_cran()
  fx <- .na_fixture()
  dn <- fx$d; dn$y[fx$na] <- NA
  dd <- fx$d[!fx$na, ]
  A <- fx$A
  # The single-block areal kernel: the fit with NA rows IS the fit of the
  # observed rows, cell for cell.
  icar <- function(dat) tulpa(y ~ x + spatial(region), dat, family = "poisson",
                              spatial = list(type = "icar", adjacency = A),
                              mode = "nested_laplace", control = .na_ctl)
  f1 <- icar(dn); f2 <- icar(dd)
  expect_equal(f1$log_marginal, f2$log_marginal, tolerance = 1e-10)
  expect_equal(coef(f1), coef(f2), tolerance = 1e-8)
  # The registry door, directly.
  csr <- tulpa:::adjacency_to_csr_tulpa(A)
  blk <- function(dat) list(type = "icar", spatial_idx = dat$region,
                            n_spatial_units = fx$S, adj_row_ptr = csr$row_ptr,
                            adj_col_idx = csr$col_idx,
                            n_neighbors = csr$n_neighbors)
  g1 <- tulpa_nested_laplace(dn$y, rep(1L, nrow(dn)), cbind(1, dn$x),
                             prior = blk(dn), family = "poisson",
                             control = .na_ctl)
  g2 <- tulpa_nested_laplace(dd$y, rep(1L, nrow(dd)), cbind(1, dd$x),
                             prior = blk(dd), family = "poisson",
                             control = .na_ctl)
  expect_equal(g1$log_marginal, g2$log_marginal, tolerance = 1e-10)
  # The single temporal kernel, the dense multi-block driver and the sparse
  # joint Newton (a Knorr-Held interaction) each fit, and every NA row reads a
  # finite predictor with a positive variance.
  doors <- list(
    temporal = function(dat) tulpa(y ~ x, dat, family = "poisson",
      temporal = temporal_rw1("year"), mode = "nested_laplace",
      control = .na_ctl),
    multi = function(dat) tulpa(y ~ x + spatial(region), dat,
      family = "poisson", spatial = list(type = "icar", adjacency = A),
      temporal = temporal_rw1("year"), mode = "nested_laplace",
      control = .na_ctl),
    joint = function(dat) tulpa(y ~ x + spatial(region), dat,
      family = "poisson", spatial = list(type = "icar", adjacency = A),
      temporal = temporal_rw1("year"),
      spatiotemporal = spatiotemporal(type = "IV"),
      mode = "nested_laplace", control = .na_ctl))
  for (nm in names(doors)) {
    pl <- posterior_linpred(doors[[nm]](dn), summary = TRUE)
    expect_true(all(is.finite(pl$estimate[fx$na])), info = nm)
    expect_true(all(pl$sd[fx$na] > 0), info = nm)
  }
})

test_that("a forecast row's linear predictor is the exact Gaussian predictive", {
  skip_on_cran()
  fx <- .na_fixture(); A <- fx$A; S <- fx$S; n_t <- fx$n_t
  d <- fx$d
  set.seed(9)
  d$y <- 0.3 + 0.4 * d$x + rnorm(S * n_t, sd = 0.5)[(d$year - 1) * S + d$region] +
    rnorm(nrow(d), sd = 0.4)
  d$y[fx$na] <- NA
  phi <- 0.2; tau_s <- 2; tau_t <- 3; tau_d <- 1.5
  tm <- temporal_rw1("year"); tm$tau_grid <- tau_t
  f <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = phi,
             spatial = list(type = "icar", adjacency = A, tau_grid = tau_s),
             temporal = tm,
             spatiotemporal = spatiotemporal(type = "IV", tau_grid = tau_d),
             mode = "nested_laplace", control = .na_ctl)
  pl <- posterior_linpred(f, summary = TRUE)
  sb <- tulpa:::.tulpa_prior_sd()
  st <- tulpa:::.st_interaction_structure("IV", A, n_t, "rw1", FALSE)
  nd <- st$n_latent
  Am <- t(vapply(st$constr, tulpa:::.constraint_row_dense, numeric(nd), n_x = nd))
  B <- MASS::Null(t(Am))
  Cd <- B %*% solve(tau_d * crossprod(B, as.matrix(st$R) %*% B)) %*% t(B)
  Cs <- solve(tau_s * (diag(rowSums(A)) - A + 1 / S))
  Ct <- solve(tau_t * (crossprod(diff(diag(n_t))) + 1 / n_t))
  Zs <- outer(d$region, 1:S, "==") * 1; Zt <- outer(d$year, 1:n_t, "==") * 1
  Zd <- outer((d$year - 1) * S + d$region, 1:nd, "==") * 1
  X <- cbind(1, d$x)
  # Prior covariance of eta over every row; condition on the observed ones.
  K <- sb^2 * X %*% t(X) + Zs %*% Cs %*% t(Zs) + Zt %*% Ct %*% t(Zt) +
    Zd %*% Cd %*% t(Zd)
  o <- !fx$na
  G <- K[, o] %*% solve(K[o, o] + phi * diag(sum(o)))
  m <- as.numeric(G %*% d$y[o])
  v <- diag(K - G %*% K[o, ])
  expect_equal(pl$estimate, m, tolerance = 1e-6)
  expect_equal(pl$sd, sqrt(v), tolerance = 1e-6)
})

test_that("a backend without the mask refuses an NA response by name", {
  skip_on_cran_budget()
  fx <- .na_fixture()
  dn <- fx$d; dn$y[fx$na] <- NA
  expect_error(tulpa(y ~ x, dn, family = "poisson", mode = "laplace"),
               "nested-Laplace path")
  dn$y[1] <- NaN
  expect_error(tulpa(y ~ x, dn, family = "poisson", mode = "nested_laplace",
                     temporal = temporal_rw1("year"), control = .na_ctl),
               "Non-finite")
})
