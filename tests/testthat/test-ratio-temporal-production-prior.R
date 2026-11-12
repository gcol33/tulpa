test_that("production temporal priors match dense quadratic forms", {
  T <- 6L; G <- 2L; tau <- 1.7; rho <- -0.35
  w <- seq(-0.7,1.2,length.out=T*G)
  for (type in c('rw1','rw2','ar1','iid')) for (cyclic in c(FALSE,TRUE)) {
    for (augment in if(type %in% c('rw1','rw2')) c(FALSE,TRUE) else FALSE) {
      reference <- 0
      for (g in seq_len(G)) {
        v <- w[(g-1L)*T+seq_len(T)]
        if (type %in% c('rw1','rw2')) {
          order <- if(type=='rw1') 1L else 2L
          D <- diff(diag(T),differences=order)
          if(cyclic) {
            if(order==1L) D <- rbind(D,c(1,rep(0,T-2L),-1))
            else D <- rbind(D,c(1,rep(0,T-3L),1,-2),c(-2,1,rep(0,T-3L),1))
          }
          Q <- crossprod(D)
          rank <- qr(D)$rank
          if(augment) { Q <- Q + matrix(1/T,T,T); rank <- rank+1L }
          reference <- reference + rank*(log(tau)-log(2*pi))/2 - tau*drop(crossprod(v,Q %*% v))/2
        } else if (type=='ar1') {
          reference <- reference + dnorm(v[1],0,1/sqrt(tau*(1-rho*rho)),log=TRUE) +
            sum(dnorm(v[-1],rho*v[-T],1/sqrt(tau),log=TRUE))
        } else reference <- reference + sum(dnorm(v,0,1/sqrt(tau),log=TRUE))
      }
      actual <- tulpa:::cpp_test_temporal_log_prior(w,type,tau,rho,cyclic,G,augment)
      expect_equal(actual,reference,tolerance=1e-12,info=paste(type,cyclic,augment))
      if (augment == (type %in% c('rw1','rw2'))) expect_equal(
        tulpa:::cpp_test_temporal_log_prior(w,type,tau,rho,cyclic,G,FALSE,TRUE),
        reference,tolerance=1e-12,info=paste('TVC',type,cyclic))
    }
  }
})
