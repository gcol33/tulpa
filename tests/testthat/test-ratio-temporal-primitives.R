test_that("temporal quadratic forms retain their stencil references", {
  q1 <- function(x,cyclic=FALSE) tulpa:::cpp_test_rw1_quadratic_form(x,cyclic)
  q2 <- function(x,cyclic=FALSE) tulpa:::cpp_test_rw2_quadratic_form(x,cyclic)
  expect_equal(q1(rep(5,5)),0)
  expect_equal(q1(0:4),4)
  expect_equal(q1(c(0,3)),9)
  expect_equal(q1(0:2),2)
  expect_equal(q1(0:2,TRUE),6)
  expect_equal(q1(rep(3,4),TRUE),0)
  expect_equal(q2(rep(2,5)),0)
  expect_equal(q2(0:4),0)
  expect_equal(q2((0:4)^2),12)
  expect_equal(q2(c(1,2)),0)
  expect_equal(q2(c(1,2,4)),1)
  expect_equal(q2(0:3,TRUE),32)
})

test_that("AR1 densities agree with stationary Gaussian densities", {
  x <- c(1,.5,-.3)
  expect_equal(tulpa:::cpp_test_ar1_log_density(x,0,2),
    sum(dnorm(x,0,1/sqrt(2),log=TRUE)),tolerance=1e-10)
  x <- c(1,.9,.8,.7)
  expect_gt(tulpa:::cpp_test_ar1_log_density(x,.9,1),
            tulpa:::cpp_test_ar1_log_density(x,.1,1))
  expect_equal(tulpa:::cpp_test_ar1_log_density(1,.5,1),
    dnorm(1,0,1/sqrt(1-.5^2),log=TRUE))
  x <- c(1,.5);rho<-.8;tau<-2
  reference<-dnorm(x[1],0,1/sqrt(tau*(1-rho^2)),log=TRUE)+
    dnorm(x[2],rho*x[1],1/sqrt(tau),log=TRUE)
  expect_equal(tulpa:::cpp_test_ar1_log_density(x,rho,tau),reference)
})

test_that("temporal prior dispatch agrees with normalized Gaussian references", {
  x<-c(0,1,1.5,1.2,.8);tau<-2;rho<-.7
  prior<-function(type,cyclic=FALSE)
    tulpa:::cpp_test_temporal_log_prior(x,type,tau,rho,cyclic)
  expect_equal(prior("none"),0)
  for(cyclic in c(FALSE,TRUE)) for(order in 1:2) {
    extended<-if(cyclic) c(x,x[seq_len(order)]) else x
    quad<-sum(diff(extended,differences=order)^2)
    rank<-length(x)-if(cyclic) 1L else order
    expect_equal(prior(paste0("rw",order),cyclic),
      rank/2*(log(tau)-log(2*pi))-.5*tau*quad)
  }
  expect_equal(prior("ar1"),tulpa:::cpp_test_ar1_log_density(x,rho,tau))
  expect_equal(prior("iid"),sum(dnorm(x,0,1/sqrt(tau),log=TRUE)))
})
