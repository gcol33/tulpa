test_that("GP lengthscale logit tails retain their Jacobian and gradient", {
  d <- data.frame(day=rep(c(0,.1,.4,.8,1.4),each=2),x=sin(seq_len(10)))
  X <- model.matrix(~x,d)
  t <- tulpa:::validate_tvc(temporal_tvc("day",terms=~x-1,structure="gp",
    parameterization="noncentered",scale_coords=FALSE),d,X)
  coefficient <- tulpa:::.tvc_sampler_spec(t,X)
  additive <- list(type="gp",parameterization="noncentered",n_times=5L,
    n_groups=1L,time_idx=rep(1:5,each=2),time_values=t$time_values,
    cov="exponential",phi_prior_lower=t$phi_prior_lower,phi_prior_upper=t$phi_prior_upper)
  for(door in c("coefficient","additive")) for(sign in c(-1,1)) {
    evaluate <- function(raw) {
      args <- list(y=rep(0,10),X=X,point=c(.1,.2,log(.4),raw,cos(1:5)/2))
      if(door=="coefficient") args$tvc_spec <- coefficient else args$temporal_spec <- additive
      do.call(tulpa:::cpp_test_field_coordinates,args)
    }
    reference <- evaluate(sign*30)
    for(raw in sign*c(40,100,1000)) {
      r <- evaluate(raw)
      expect_equal(r$raw_lp-reference$raw_lp,-(abs(raw)-30),tolerance=1e-8)
      expect_equal(r$prior_gradient[4],-sign,tolerance=1e-10)
      expect_equal(r$gradient,r$numerical,tolerance=1e-5)
      expect_true(r$verified)
    }
  }
})

test_that("GP innovations keep the scale and lengthscale prior gradients", {
  times <- c(0, .07, .31, .8, 1.4)
  d <- data.frame(day=rep(times,each=2),x=sin(seq_len(10)))
  X <- model.matrix(~x,d)
  z <- cos(seq_along(times))/2
  for (covariance in c("exponential","matern","gaussian","periodic")) {
    temporal <- temporal_tvc("day",terms=~x-1,structure="gp",cov=covariance,
      period=if(covariance=="periodic") 2 else NULL,scale_coords=FALSE,
      parameterization="noncentered")
    spec <- tulpa:::.tvc_sampler_spec(tulpa:::validate_tvc(temporal,d,X),X)
    lo <- spec$phi_prior_lower; hi <- spec$phi_prior_upper
    phi <- .2; s <- (phi-lo)/(hi-lo)
    q <- c(.1,.2,log(.4),qlogis(s),z)
    lag <- abs(outer(times,times,"-"))
    correlation <- switch(covariance,
      exponential=exp(-lag/phi),
      matern=(1+sqrt(3)*lag/phi)*exp(-sqrt(3)*lag/phi),
      gaussian=exp(-(lag/phi)^2),periodic=exp(-2*sin(pi*lag/2)^2/phi^2))
    K <- .4*correlation
    if(covariance!="exponential") diag(K) <- diag(K)+1e-8
    L <- t(chol(K))
    H <- temporal_gp_reference_rotation(L)
    expect_equal(crossprod(H),diag(5),tolerance=1e-12)
    level <- drop(L%*%H[,1])
    expect_equal(level,rep(level[1],5),tolerance=1e-10)
    field <- drop(L%*%H%*%z)
    additive <- list(type="gp",parameterization="noncentered",n_times=5L,
      n_groups=1L,time_idx=rep(1:5,each=2),time_values=times,cov=covariance,
      nu=1.5,period=2,phi_prior_lower=lo,phi_prior_upper=hi)
    for(door in c("coefficient","additive")) {
      args <- list(y=rep(0,10),X=X,point=q)
      if(door=="coefficient") args$tvc_spec <- spec else args$temporal_spec <- additive
      r <- do.call(tulpa:::cpp_test_field_coordinates,args)
      expect_equal(tail(r$prior_gradient,5),-z,tolerance=1e-10)
      expect_equal(r$prior_gradient[3:4],c(.5-.5*(-log(.01))*sqrt(.4),1-2*s),tolerance=1e-10)
      expect_equal(r$prior_gradient,r$numerical_prior,tolerance=1e-5)
      expect_equal(r$gradient,r$numerical,tolerance=1e-5)
      expect_equal(tail(r$stored,5),if(door=="coefficient")field-mean(field)else drop(L%*%z),tolerance=1e-10)
      expect_true(r$verified)
    }
  }
})

test_that("GP field coordinates share the density and derivatives at both doors", {
  load_cpp_fixture("temporal_gp")
  times <- c(0, .07, .31, .8, 1.4)
  z <- sin(seq_len(10)) / 2
  q <- c(z, log(.4), qlogis((.2-.01)/9.99))
  for (cov in 0:3) {
    call <- function(q, nc, tvc = FALSE) cpp_test_temporal_gp_density(
      times, 2L, q[seq_along(z)], q[11], q[12], nc,
      tvc = tvc, covariance = cov, nu = 1.5, period = 2)
    nc <- call(q, 1L)
    coefficient <- call(q, 1L, TRUE)
    expect_equal(coefficient$log_post, nc$log_post, tolerance = 1e-12)
    expect_equal(coefficient$gradient, nc$gradient, tolerance = 1e-10)
    phi <- .01 + 9.99 * plogis(q[12])
    distance <- abs(outer(times, times, "-"))
    correlation <- switch(cov + 1L,
      exp(-distance/phi),
      (1 + sqrt(3)*distance/phi)*exp(-sqrt(3)*distance/phi),
      exp(-(distance/phi)^2),
      exp(-2*sin(pi*distance/2)^2/phi^2))
    covariance <- exp(q[11]) * correlation
    if (cov != 0L) diag(covariance) <- diag(covariance) + 1e-8
    L <- t(chol(covariance))
    H <- temporal_gp_reference_rotation(L)
    expect_equal(nc$field, as.numeric(L %*% matrix(z, 5)), tolerance = 1e-7)
    expected <- L %*% H %*% matrix(z,5)
    expected <- sweep(expected,2,colMeans(expected))
    expect_equal(coefficient$field,as.numeric(expected),tolerance=1e-7)
    centered <- call(c(nc$field, q[11:12]), 0L)
    shift <- 2 * (sum(log(diag(L))) + 2.5 * log(2*pi))
    expect_equal(nc$log_post - centered$log_post, shift, tolerance = 1e-6)
    for (parameterization in 0:1) {
      point <- if (parameterization == 1L) q else c(nc$field, q[11:12])
      for (door in c(FALSE, TRUE)) {
        numerical <- vapply(seq_along(point), function(j) {
          plus <- minus <- point
          h <- 1e-5 * max(1, abs(point[j]))
          plus[j] <- plus[j] + h; minus[j] <- minus[j] - h
          (call(plus, parameterization, door)$log_post - call(minus, parameterization, door)$log_post)/(2*h)
        }, numeric(1))
        expect_equal(call(point, parameterization, door)$gradient, numerical, tolerance = 2e-5)
      }
    }
  }
})

test_that("automatic temporal GP transport whitens a Gaussian conditional", {
  times <- c(0, .07, .31, .8, 1.4)
  index <- rep(1:5, each=2)
  d <- data.frame(day=times[index], x=sin(seq_len(10)))
  X <- model.matrix(~x,d)
  y <- .3 + .4*d$x + cos(seq_len(10))/3
  z <- sin(seq_len(5))/2
  phi <- .2
  for(covariance in c("exponential","matern","gaussian","periodic")) {
    temporal <- temporal_tvc("day",terms=~x-1,structure="gp",cov=covariance,
      period=if(covariance=="periodic") 2 else NULL,scale_coords=FALSE)
    coefficient <- tulpa:::.tvc_sampler_spec(tulpa:::validate_tvc(temporal,d,X),X)
    additive <- list(type="gp",parameterization="auto",n_times=5L,
      n_groups=1L,time_idx=index,time_values=times,cov=covariance,
      nu=1.5,period=2,phi_prior_lower=.01,phi_prior_upper=10)
    lag <- abs(outer(times,times,"-"))
    correlation <- switch(covariance,
      exponential=exp(-lag/phi),
      matern=(1+sqrt(3)*lag/phi)*exp(-sqrt(3)*lag/phi),
      gaussian=exp(-(lag/phi)^2),periodic=exp(-2*sin(pi*lag/2)^2/phi^2))
    K <- .4*correlation
    if(covariance!="exponential") diag(K) <- diag(K)+1e-8
    L <- t(chol(K))
    F <- L%*%temporal_gp_reference_rotation(L)
    for(door in c("coefficient","additive")) {
      specification <- if(door=="coefficient") coefficient else additive
      lower <- specification$phi_prior_lower; upper <- specification$phi_prior_upper
      point <- c(.1,.2,log(.4),qlogis((phi-lower)/(upper-lower)),z)
      basis <- if(door=="coefficient") sweep(F,2,colMeans(F)) else L
      design <- basis[index,,drop=FALSE]
      if(door=="coefficient") design <- design*d$x
      precision <- diag(5)+crossprod(design)
      B <- chol(precision)
      mean <- solve(precision,crossprod(design,y-drop(X%*%point[1:2])))
      native <- drop(mean+backsolve(B,z))
      args <- list(y=y,X=X,point=point,native_point=FALSE)
      if(door=="coefficient") args$tvc_spec <- coefficient else args$temporal_spec <- additive
      r <- do.call(tulpa:::cpp_test_field_coordinates,args)
      expect_true(r$conditional)
      expect_equal(tail(r$x,5),native,tolerance=2e-6)
      expect_equal(r$jacobian,-sum(log(diag(B))),tolerance=2e-6)
      expect_equal(tail(r$gradient,5),-z,tolerance=2e-6)
      expect_equal(r$inverse,point,tolerance=1e-10)
      expect_equal(tail(r$stored,5),drop(basis%*%native),tolerance=2e-6)
      expect_equal(r$gradient,r$numerical,tolerance=1e-5)
      expect_equal(r$prior_gradient,r$numerical_prior,tolerance=1e-5)
      expect_true(r$verified)
    }
  }
})

test_that("Gaussian GP transports compose across groups and coefficient terms", {
  d <- data.frame(day=rep(c(0,.1,.4,.8,1.4),each=8),
    group=rep(c("a","b"),20),x=sin(seq_len(40)),z=cos(seq_len(40)))
  X <- model.matrix(~x+z,d)
  temporal <- temporal_tvc("day",terms=~x+z-1,group_var="group",structure="gp")
  spec <- tulpa:::.tvc_sampler_spec(tulpa:::validate_tvc(temporal,d,X),X)
  r <- tulpa:::cpp_test_field_coordinates(y=.3+.5*d$x,X=X,tvc_spec=spec,shift=.4)
  expect_true(r$conditional)
  expect_equal(r$inverse,r$q,tolerance=1e-10)
  expect_equal(r$gradient,r$numerical,tolerance=2e-5)
  expect_equal(r$prior_gradient,r$numerical_prior,tolerance=2e-5)
  expect_equal(colMeans(matrix(tail(r$stored,20),5)),rep(0,4),tolerance=1e-12)
  expect_true(r$verified)
})

test_that("GP coefficient constructors select coordinates and retain the prior anchors", {
  for (p in c("auto", "noncentered", "centered")) {
    spec <- temporal_tvc("day", structure="gp", parameterization=p)
    expect_identical(spec$parameterization, p)
    expect_equal(c(spec$sigma_prior_U, spec$sigma_prior_alpha), c(1, .01))
  }
  expect_error(temporal_tvc("day", structure="gp", parameterization="unknown"), "arg")

})

test_that("GP coefficient fits recover the planted slope over sampler seeds", {
  skip_if_not_slow()
  set.seed(4)
  times <- sort(cumsum(rexp(20, rate=1/4)))
  times <- (times-min(times))/diff(range(times))*100
  deviation <- .6*sin(2*pi*times/70)
  deviation <- deviation - mean(deviation)
  truth <- .6 + deviation
  d <- data.frame(day=rep(times, each=10), x=rnorm(200))
  d$y <- rpois(200, exp(.3 + truth[match(d$day,times)]*d$x))
  for (seed in 1:5) {
    fit <- suppressWarnings(tulpa(y~x, data=d, family="poisson",
      temporal=temporal_tvc("day", terms=~x-1, structure="gp"), mode="exact",
      control=list(n_iter=2000L, warmup=1000L, n_chains=2L, seed=seed)))
    diagnostic <- diagnostics(fit)
    expect_equal(n_divergent(fit), 0L)
    expect_lt(max(diagnostic$rhat, na.rm=TRUE), 1.05)
    expect_gt(min(diagnostic$ess_bulk, na.rm=TRUE), 100)
    post <- tvc(fit)
    slope <- fit$draws[,"x"] + post$draws[,,1]
    levels <- as.numeric(post$time_levels)
    key <- vapply(levels, function(t) {
      hit <- which(abs(times-t) <= sqrt(.Machine$double.eps)*max(1,abs(times)))
      stopifnot(length(hit)==1L); hit
    }, integer(1))
    expect_false(anyNA(key))
    expect_gt(cor(colMeans(slope), truth[key]), .8)
    expect_lt(sqrt(mean((colMeans(slope)-truth[key])^2)), .3)
    mi <- fit$model_inputs
    expected_eta <- outer(fit$draws[,"(Intercept)"], rep(1,200)) +
      sweep(slope[,match(d$day,times[key]),drop=FALSE], 2, d$x, "*")
    expect_equal(tulpa:::.sampler_engine_eta(fit, fit$draws, mi$offset),
      expected_eta, tolerance=1e-10)
  }
})
