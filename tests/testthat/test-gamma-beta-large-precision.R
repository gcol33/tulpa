precision_fixture <- function(name) {
  read.csv(testthat::test_path("fixtures", paste0("precision-", name, ".csv")))
}

test_that("Stirling remainders and relative deviances match exact references", {
  refs <- precision_fixture("primitives")
  for (i in seq_len(nrow(refs))) {
    r <- refs[i, ]
    actual <- if (r$kind == "s") tulpa:::.stirling_remainder(r$x, r$order) else
      tulpa:::.log1pmx(r$x)
    if (is.infinite(r$expected)) expect_identical(actual, r$expected) else
      expect_lte(abs(actual - r$expected), 4 * .Machine$double.eps * abs(r$expected),
                 label = paste(r$kind, r$order, r$x))
  }
  x <- matrix(c(.3, 2, 1e8, NA_real_), 2, dimnames = list(c("a", "b"), c("c", "d")))
  expect_identical(dimnames(tulpa:::.stirling_remainder(x)), dimnames(x))
  expect_identical(tulpa:::.log1pmx(numeric()), numeric())
  expect_true(is.na(tulpa:::.stirling_remainder(NA_real_)))
})

test_that("every gamma and beta density matches the exact density at every precision", {
  refs <- precision_fixture("densities")
  for (i in seq_len(nrow(refs))) {
    r <- refs[i, ]; family <- r$family
    ops <- tulpa:::.FAMILY_OPS[[family]]
    paths <- c(R_registry = ops$loglik(r$eta, r$y, 1, r$phi),
      R_mu = tulpa:::.FAMILY_MU[[family]]$loglik_mu(r$y, r$mu, r$phi, 1),
      string = tulpa:::cpp_family_terms(r$y, 1L, r$eta, family, r$phi)[1L],
      tulpa:::cpp_precision_density_paths(r$y, r$eta, family, r$phi),
      builtin_AD = tulpa:::cpp_family_ad_terms(r$y, 1L, r$eta, family, r$phi)[1L],
      SPDE = tulpa:::cpp_spde_density_terms(r$y, r$eta, log(r$phi), family)[1L])
    if (family == "beta") paths <- c(paths,
      beta_sampler = tulpa:::cpp_beta_density_terms(r$y, r$eta, log(r$phi))[1L])
    for (j in seq_along(paths)) {
      ref <- if (j <= 2L) r$expected else r$cppexpected
      expect_lt(abs(paths[j] - ref), 2e-13,
        label = paste(family, r$phi, names(paths)[j]))
    }
    probe <- tulpa:::cpp_precision_density_paths(r$y,r$eta,family,r$phi)
    expect_identical(attr(probe,"mu"),r$cppmu)
    dphi <- tulpa:::.FAMILY_DPHI[[family]]$dloglik(r$eta, r$y, 1, r$phi)
    expect_lt(abs((dphi-r$dphi)/r$dphi), 1e-12, label = paste(family, r$phi))
    dphi2 <- tulpa:::.precision_density(r$y,r$mu,r$phi,beta=family=="beta",order=2L)
    expect_lt(abs((dphi2-r$dphi2)/r$dphi2), 1e-12, label = paste(family,r$phi))
    if (family == "gamma") expect_equal(
      tulpa:::.FAMILY_DPHI2$gamma$dloglik2(r$eta,r$y,1,r$phi), dphi2, tolerance=1e-14)
    sp <- tulpa:::cpp_spde_density_terms(r$y,r$eta,log(r$phi),family)
    expect_lt(abs(sp[3L] - r$phi*r$cppdphi), 2e-13, label = paste(family,r$phi))
    for (start in c(1L,4L,7L)) {
      expect_lt(abs(sp[start] - r$cppexpected), 2e-13, label=paste(family,r$phi,start))
      expect_lt(abs(sp[start+1L]-r$cppgrad)/max(1,abs(r$cppgrad)),2e-13,
        label=paste(family,r$phi,start,"eta"))
      expect_lt(abs(sp[start+2L]-r$phi*r$cppdphi),2e-13,
        label=paste(family,r$phi,start,"log precision"))
    }
    expect_lt(abs(ops$score(r$eta,r$y,1,r$phi)-r$grad)/max(1,abs(r$grad)),2e-13)
    expect_lt(abs(tulpa:::.FAMILY_DPHI[[family]]$dscore(r$eta,r$y,1,r$phi)-r$mixed)/
      max(abs(r$mixed),1e-300),1e-12)
    if (family=="beta") {
      group <- tulpa:::cpp_beta_grouped_terms(c(r$low,r$mu,r$y),r$eta,r$phi)
      expect_lt(abs(group[1]-r$cppgroup), 5e-13)
      expect_lt(abs(group[4]-r$cppgroup), 5e-13)
    }
  }
})

test_that("autodiff primitives carry their exact unary chain rule", {
  refs <- precision_fixture("primitives")
  for (i in which(refs$order == 0L & refs$x > -1 & is.finite(refs$expected))) {
    r <- refs[i, ]
    val <- tulpa:::cpp_precision_ad_math(r$x, r$kind=="s")
    deriv <- if (r$kind=="s") tulpa:::.stirling_remainder(r$x,1L) else r$x/(1+r$x)
    for (start in c(1L,4L,7L)) {
      expect_equal(val[start],r$expected,tolerance=1e-14)
      expect_equal(val[start+1L],deriv,tolerance=1e-14)
    }
  }
  endpoint <- tulpa:::cpp_precision_ad_math(-1, FALSE)
  expect_identical(endpoint[c(1,4,7)],rep(Inf,3))
  expect_identical(endpoint[c(2,5,8)],rep(-Inf,3))
})

test_that("joint beta arms carry raw grouped responses", {
  arm <- list(y=c(.3,.4), X=matrix(1,2,1), family="beta",phi=100,
    n_trials=c(2L,3L), beta_y=list(c(.29,.31), c(.39,.4,.41)))
  normalise <- function(a) tulpa:::.normalise_joint_arm_core(a,1L,spatial_idx="optional")
  out <- normalise(arm)
  expect_identical(out$beta_y,arm$beta_y)
  expect_error(normalise(modifyList(arm,list(n_trials=c(1L,1L)))),"group sizes")
  invalid <- arm
  invalid$beta_y <- list(c(0,.3), c(.3,.4,.5))
  expect_error(normalise(invalid),"responses")
  expect_error(normalise(modifyList(arm,list(slog_y=c(-1,-1)))),"raw responses")
})

test_that("relative deviances retain extreme positive responses", {
  for (phi in c(.3, 2, 100)) {
    y <- c(1e-250, 1e-30, .01, .99, 1-1e-12)
    expect_equal(tulpa:::.precision_density(y, .3, phi, beta=TRUE),
      dbeta(y,.3*phi,.7*phi,log=TRUE), tolerance=1e-12)
    expect_equal(tulpa:::.precision_density(y, 1, phi),
      dgamma(y,shape=phi,rate=phi,log=TRUE), tolerance=1e-12)
  }
})
