test_that("covariance kernels agree with independent scalar references", {
  kernel<-function(d,sigma2=1,range=1,type=0L)
    tulpa:::cpp_gp_cov_value(d,sigma2,range,as.integer(type))
  expect_equal(kernel(0),1)
  expect_gt(kernel(1),kernel(2))
  expect_equal(kernel(2)/kernel(1),exp(-1))
  expect_gt(kernel(2,range=2),kernel(2,range=1))
  expect_equal(kernel(1,sigma2=4),4*kernel(1))
  expect_equal(kernel(0,type=1),1)
  expect_true(all(diff(kernel(c(.5,1,2),type=1))<0))
  expect_equal(kernel(1,type=1),(1+sqrt(3))*exp(-sqrt(3)))
  expect_equal(kernel(0,range=sqrt(2),type=2),1)
  expect_lt(kernel(3,range=sqrt(2),type=2),kernel(3))
  expect_equal(kernel(2,range=sqrt(2),type=2),kernel(2))
  expect_equal(kernel(1,range=sqrt(2),type=2),exp(-.5))
  expect_equal(kernel(0,range=2,type=3),1)
  expect_equal(kernel(c(2,3,5),range=2,type=3),rep(0,3))
  expect_equal(kernel(1,range=2,type=3),1-1.5*.5+.5*.5^3)
  for(type in 0:3) {
    d<-1;phi<-1.5;u<-d/phi
    reference<-switch(type+1,exp(-u),(1+sqrt(3)*u)*exp(-sqrt(3)*u),
      exp(-u^2),1-1.5*u+.5*u^3)
    expect_equal(kernel(d,2,phi,type),2*reference)
  }
})
