collapsed_target_fixtures <- function() {
  S <- 4L; group <- as.integer(rep_len(seq_len(S), 16L))
  sp <- list(type = "icar", parameterization = "collapsed", spatial_idx = group,
    n_spatial_units = S, adj_row_ptr = c(0L,1L,3L,5L,6L),
    adj_col_idx = c(1L,0L,2L,1L,3L,2L), n_neighbors = c(1L,2L,2L,1L))
  d <- data.frame(lon = rep(c(0,.2,.7,1),4), lat = rep(c(.1,.8,.3,.9),4))
  gp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(spatial_gp(~lon+lat, nn=3L),d))
  gp$parameterization <- "collapsed"
  bym2 <- sp; bym2$type <- "bym2"
  list(icar = sp, bym2 = bym2, gp = gp)
}

test_that("non-Gaussian collapsing gives an explicit full-field route", {
  X <- matrix(1,16,1)
  for (sp in collapsed_target_fixtures()) for (family in c("poisson","binomial")) {
    expect_error(tulpa:::tulpa_dispatch("hmc", fitter_args = list(
      y = rep(c(0,1),8), n_trials = rep(1L,16), X = X, family = family,
      backend = "hmc", spatial_spec = sp,
      control = list(n_iter=20L,warmup=10L,n_chains=1L,seed=1177L))),
      "uncorrected Laplace marginal \\(Tier 2\\).*full-field NUTS")
  }
  A <- matrix(0,4,4)
  A[cbind(1:3,2:4)] <- 1; A <- A+t(A)
  d <- data.frame(y=rep(c(0,1),8),region=factor(rep(1:4,4)))
  expect_error(tulpa(y~spatial(region), data=d, family="poisson",mode="hmc",
    spatial=spatial_car(A,group_var="region",parameterization="collapsed"),
    control=list(n_iter=20L,warmup=10L,n_chains=1L,seed=1177L)),
    "uncorrected Laplace marginal \\(Tier 2\\).*full-field NUTS")
})

test_that("built-in Gaussian collapsing matches the dense marginal at fixed dispersion", {
  sp <- collapsed_target_fixtures()$icar
  X <- cbind(1,seq(-1,1,length.out=16)); y <- sin(seq_len(16))
  S <- sp$n_spatial_units
  A <- matrix(0,S,S); A[cbind(1:3,2:4)] <- 1; A <- A+t(A)
  Q <- diag(rowSums(A))-A+matrix(1/S,S,S)
  B <- (diag(S)-matrix(1/S,S,S))[sp$spatial_idx,]
  points <- rbind(c(.2,-.3,.4),c(0,0,0),c(-.4,.1,-.2))
  rate <- -log(.01)/3
  for (sd in c(.4,1.7)) {
    actual <- tulpa:::cpp_test_sampler_log_post_paths(points,y,rep(1L,16),X,
      "gaussian",phi=sd,spatial_spec=sp)
    expected <- apply(points,1,function(q) {
      V <- diag(sd^2,16)+B%*%solve(exp(q[3])*Q,t(B))
      r <- y-drop(X%*%q[1:2])
      -.5*sum(r*solve(V,r))-.5*as.numeric(determinant(V)$modulus)-
        .5*sum(q[1:2]^2)/2.5^2-.5*q[3]-rate*exp(-.5*q[3])
    })
    expect_equal(actual[,1],actual[,2],tolerance=1e-9)
    expect_equal(actual[,2]-actual[2,2],expected-expected[2],tolerance=1e-9)
  }
})

test_that("Gaussian collapsed R fits carry coupled joint fields and their target", {
  skip_if_not_slow()
  X <- cbind("(Intercept)"=1,x=seq(-1,1,length.out=16))
  y <- sin(seq_len(16))
  for (sp in collapsed_target_fixtures()) {
    fit <- tulpa:::tulpa_dispatch("hmc", fitter_args=list(y=y,n_trials=rep(1L,16),
      X=X,family="gaussian",backend="hmc",spatial_spec=sp,
      control=list(n_iter=150L,warmup=75L,n_chains=2L,seed=1177L)))
    expect_identical(fit$inference_tier,1L)
    expect_identical(fit$inference_mode,"exact")
    expect_match(fit$target,"Analytic Gaussian spatial marginal",fixed=TRUE)
    expect_equal(nrow(fit$draws),150L)
    expect_equal(ncol(fit$draws)-ncol(fit$sampler_draws),if(sp$type=="bym2") 8L else 4L)
    expect_equal(fit$draws[,colnames(fit$sampler_draws),drop=FALSE],fit$sampler_draws)
    expect_identical(fit$model_inputs$spatial_spec$parameterization,"standard")
    eta <- tulpa:::.sampler_engine_eta(fit,fit$draws,NULL)
    expect_equal(dim(eta),c(150L,16L))
    expect_true(all(is.finite(eta)))
    expect_error(tulpa:::tulpa_sample_glmm(y,rep(1L,16),X,"gaussian","vi",
      spatial_spec=sp,control=list(vi_max_iter=1L)),"supported by NUTS/HMC")
  }
  A <- matrix(0,4,4); A[cbind(1:3,2:4)] <- 1; A <- A+t(A)
  d <- data.frame(y=y,x=X[,2],region=factor(rep(1:4,4)))
  fit <- tulpa(y~x+spatial(region),data=d,family="gaussian",mode="hmc",
    spatial=spatial_car(A,group_var="region",parameterization="collapsed"),
    control=list(n_iter=150L,warmup=75L,n_chains=1L,seed=1177L))
  expect_identical(fit$inference_tier,1L)
  expect_match(fit$target,"Analytic Gaussian spatial marginal",fixed=TRUE)
  expect_equal(ncol(fit$draws),7L)
  expect_true(all(is.finite(fitted(fit))))
})

test_that("C-ABI collapsed targets require a Gaussian declaration on both entries", {
  skip_on_cran_budget()
  skip_if_not(.cpp_fixture_has_toolchain(),"no C++ toolchain")
  root <- .tulpa_source_root()
  skip_if(is.null(root),"tulpa source tree not found")
  env <- new.env(parent=globalenv())
  include <- normalizePath(file.path(root,"inst/include"),winslash="/")
  withr::with_envvar(c(PKG_CPPFLAGS=paste0('-I"',include,'"')),
    Rcpp::sourceCpp(test_path("cpp","collapsed_target.cpp"),env=env,
      cacheDir=file.path(tempdir(),"tulpa-collapsed-target")))
  for (n in 1:2) {
    expect_error(env$cpp_collapsed_target(FALSE,n),"uncorrected Laplace marginal \\(Tier 2\\).*full-field NUTS")
    fits <- env$cpp_collapsed_target(TRUE,n)
    for (fit in fits) {
      expect_identical(fit$inference_tier,1L)
      expect_match(fit$target,"Analytic Gaussian spatial marginal",fixed=TRUE)
      expect_identical(fit$n_fields,2L)
      expect_identical(fit$n_samples,15L)
    }
  }
  for (attempt in 1:2) {
    expect_error(env$cpp_collapsed_target(TRUE,weights=FALSE),
      "requires double and arena likelihood weights and priors")
    expect_identical(env$cpp_collapsed_target(TRUE)[[1]]$inference_tier,1L)
  }
  joint <- env$cpp_collapsed_target(FALSE,collapsed=FALSE)[[1]]
  expect_identical(joint$target,"Joint posterior")
  expect_identical(joint$inference_tier,1L)
  laplace <- env$cpp_laplace_target_metadata()
  expect_identical(laplace$inference_tier,2L)
  expect_match(laplace$target,"Laplace spatial marginal",fixed=TRUE)
})
