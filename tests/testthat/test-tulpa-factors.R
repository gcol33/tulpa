# tulpa(y ~ x, data = long, factors = latent_factor(n, unit =, response =)):
# one joint arm per response level, the factors as lf blocks over the units.
# Recovery is read on Lambda Lambda', which a rotation, a sign flip or a
# rescaling of the factors leaves unchanged.

.fac_sim <- function(seed, S, Lam, b0, b1, miss = 0) {
  set.seed(seed)
  K <- nrow(Lam); F_ <- ncol(Lam)
  U <- matrix(rnorm(S * F_), S, F_)
  x <- rnorm(S)
  d <- expand.grid(site = factor(seq_len(S)),
                   species = factor(paste0("sp", seq_len(K))))
  s <- as.integer(d$site); k <- as.integer(d$species)
  d$x <- x[s]
  eta <- b0[k] + b1[k] * d$x +
    rowSums(U[s, , drop = FALSE] * Lam[k, , drop = FALSE])
  d$y <- rpois(nrow(d), exp(eta))
  if (miss > 0) d <- d[-sample.int(nrow(d), round(miss * nrow(d))), ]
  d[sample.int(nrow(d)), ]
}

.fac_Lam <- cbind(c(0.9, 0.6, -0.5, 0.4, 0.7, -0.3),
                  c(0, 0.8, 0.5, -0.6, 0.3, 0.5))
.fac_b0 <- c(1, 0.5, 1.2, 0.2, 0.8, 0.6)
.fac_b1 <- c(0.3, -0.2, 0.5, 0, 0.4, 0.1)
.fac_spec <- function(...) latent_factor(2, unit = "site", response = "species",
                                         ...)

test_that("latent_factor() builds and checks the shared specification", {
  sp <- latent_factor(2, unit = "site", response = "species")
  expect_s3_class(sp, "tulpa_latent")
  expect_identical(sp$n_factors, 2L)
  expect_identical(sp[c("unit", "response")], list(unit = "site",
                                                   response = "species"))
  expect_null(sp$prior)
  expect_null(sp$constraint)
  expect_identical(latent_factor(constraint = "first_zero")$constraint,
                   "first_zero")
  expect_output(print(sp), "Unit column: site")
  expect_error(latent_factor(0), "positive integer")
  expect_error(latent_factor(unit = 3), "`unit`")
  expect_error(latent_factor(prior = list(sd = 1)), "prior object")
  expect_error(latent_factor(constraint = "none"), "should be one of")
})

test_that("tulpa(factors =) refuses what its arms do not carry", {
  d <- .fac_sim(1L, S = 10L, .fac_Lam, .fac_b0, .fac_b1)
  d$g <- factor(rep(1:2, length.out = nrow(d)))
  fit_with <- function(..., formula = y ~ x, spec = .fac_spec())
    tulpa(formula, data = d, family = "poisson", factors = spec, ...)
  expect_error(fit_with(mode = "hmc"), "mode")
  expect_error(fit_with(formula = y ~ x + (1 | g)), "random-effect terms")
  expect_error(fit_with(formula = y ~ x + offset(log(x^2))), "an offset")
  expect_error(fit_with(weights = rep(1, nrow(d))), "`weights`")
  expect_error(fit_with(ziformula = ~ 1), "`ziformula`")
  expect_error(fit_with(spec = .fac_spec(shared = FALSE)), "shared")
  expect_error(fit_with(spec = .fac_spec(constraint = "sum_to_zero")),
               "constraint")
  expect_error(fit_with(spec = .fac_spec(prior = prior_pc(1, 0.01))),
               "prior_normal")
  expect_error(fit_with(spec = .fac_spec(prior = prior_normal(1, 1))),
               "prior_normal")
  expect_error(fit_with(spec = latent_factor(2, unit = "plot",
                                             response = "species")),
               "column 'plot'")
  expect_error(fit_with(spec = latent_factor(7, unit = "site",
                                             response = "species")),
               "at most as many factors")
  expect_error(tulpa(y ~ x, data = d, family = "multinomial",
                     factors = .fac_spec()), "fixed-effect models only")
})

test_that("tulpa(factors =) is the joint lf fit of its arms, row order kept", {
  skip_if_fast()
  d <- .fac_sim(2L, S = 60L, .fac_Lam, .fac_b0, .fac_b1, miss = 0.15)
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               factors = .fac_spec(prior = prior_normal(0, 1.5)),
               control = list(n_threads = 1L))
  expect_s3_class(fit, "tulpa_factor_fit")

  # The same model through the joint door, built from the data by hand.
  lev <- levels(d$species)
  arms <- lapply(lev, function(l) {
    r <- d$species == l
    list(y = d$y[r], n_trials = rep(1L, sum(r)),
         X = cbind("(Intercept)" = 1, x = d$x[r]), family = "poisson",
         phi = 1)
  })
  names(arms) <- lev
  obs_idx <- lapply(lev, function(l) as.integer(d$site[d$species == l]))
  blk <- list(type = "lf", n_latent = 60L, obs_idx = obs_idx,
              sigma_lambda = 1.5)
  ref <- tulpa_nested_laplace_joint(
    responses = arms, prior = list(blk, blk),
    control = list(n_threads = 1L, progress = FALSE, store_Q = TRUE))
  expect_equal(unname(fit$modes), unname(ref$modes), tolerance = 1e-10)

  # Conditional fitted values: each row's own level and unit, in data order.
  m <- ref$modes[1L, ]
  al <- ref$arm_layout
  K <- length(lev)
  L <- matrix(0, K, 2)
  L[, 1] <- m[al$block_start[1] + 60L + seq_len(K)]
  L[2:K, 2] <- m[al$block_start[2] + 60L + seq_len(K - 1L)]
  U <- cbind(m[al$block_start[1] + seq_len(60L)],
             m[al$block_start[2] + seq_len(60L)])
  k <- as.integer(d$species); s <- as.integer(d$site)
  B <- matrix(m[seq_len(2L * K)], K, 2, byrow = TRUE)
  eta <- B[k, 1] + B[k, 2] * d$x + rowSums(L[k, ] * U[s, ])
  expect_equal(fitted(fit), exp(eta), tolerance = 1e-8)

  # Fixed-effect names, the stacked design, and predict() at newdata.
  expect_identical(names(coef(fit))[1:2], c("sp1:(Intercept)", "sp1:x"))
  nd <- d[1:5, ]
  b <- coef(fit)
  kk <- as.integer(nd$species)
  expect_equal(unname(predict(fit, newdata = nd)),
               unname(b[2 * kk - 1] + b[2 * kk] * nd$x), tolerance = 1e-10)
  nd$species <- "spX"
  expect_error(predict(fit, newdata = nd), "spX")

  # Loadings: lower-triangular, positive diagonal, read by stats::loadings().
  expect_identical(dim(loadings(fit)), c(K, 2L))
  expect_identical(loadings(fit)[1, 2], 0)
  expect_true(all(diag(loadings(fit)) > 0))
  sc <- latent_factors(fit)
  expect_identical(nrow(sc), 120L)
  expect_identical(levels(d$site)[sc$observation], sc$unit)
})

test_that("tulpa(factors =) recovers Lambda Lambda' of a two-factor model", {
  skip_if_fast()
  d <- .fac_sim(1L, S = 150L, .fac_Lam, .fac_b0, .fac_b1, miss = 0.1)
  fit <- tulpa(y ~ x, data = d, family = "poisson", factors = .fac_spec(),
               control = list(n_threads = 1L))
  C_true <- tcrossprod(.fac_Lam)
  err <- norm(fit$factor_cov - C_true, "F") / norm(C_true, "F")
  expect_lt(err, 0.25)
  low <- lower.tri(C_true, diag = TRUE)
  expect_gt(cor(fit$factor_cov[low], C_true[low]), 0.95)
})

test_that("tulpa(factors =): Lambda Lambda' recovery and coverage over seeds", {
  skip_if_not_slow()
  C_true <- tcrossprod(.fac_Lam)
  low <- lower.tri(C_true, diag = TRUE)
  out <- t(vapply(1:10, function(seed) {
    d <- .fac_sim(seed, S = 150L, .fac_Lam, .fac_b0, .fac_b1, miss = 0.1)
    fit <- tulpa(y ~ x, data = d, family = "poisson", factors = .fac_spec(),
                 control = list(n_threads = 1L))
    cd <- apply(fit$loadings_draws, 1, function(L) tcrossprod(L)[low])
    q <- apply(cd, 1, stats::quantile, c(0.025, 0.975))
    ct <- C_true[low]
    c(err = norm(fit$factor_cov - C_true, "F") / norm(C_true, "F"),
      slope = unname(stats::coef(stats::lm(fit$factor_cov[low] ~ 0 + ct))),
      cover = mean(q[1, ] <= ct & ct <= q[2, ]))
  }, numeric(3)))
  expect_lt(mean(out[, "err"]), 0.27)
  expect_gt(mean(out[, "slope"]), 0.88)
  expect_lt(mean(out[, "slope"]), 1.05)
  expect_gt(mean(out[, "cover"]), 0.88)
})
