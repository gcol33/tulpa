test_that('weighted dense scatter validates its matrix shapes', {
  f <- device_scatter_fixture(7L,3L,5L)
  f$weights <- f$weights[-1L,,drop=FALSE]
  expect_error(device_scatter_run(f,'cpu'),'invalid dense scatter dimensions')
})

test_that('dense observation contributions agree with weighted cross products', {
  for (signed in c(FALSE,TRUE)) {
    f <- device_scatter_fixture(37L,17L,13L,signed)
    cpu <- device_scatter_run(f,'cpu')
    expect_identical(cpu$errors,rep('',13L))
    for(k in seq_len(ncol(f$scores))) {
      d <- sweep(f$design,2L,f$scales[,k],'*')
      h <- crossprod(d,d*f$weights[,k]);h[upper.tri(h)] <- 0
      expect_device_equal(cpu$gradient[,k],as.vector(crossprod(d,f$scores[,k])),1e-10,'CPU dense score')
      expect_device_equal(cpu$hessian[k,],as.vector(t(h)),1e-10,'CPU signed dense Hessian')
    }
  }
})

test_that('device scatter streams and batches preserve signed curvature cell by cell', {
  skip_on_cran()
  skip_if_no_cuda('nvrtc')
  for(signed in c(FALSE,TRUE)) {
    f <- device_scatter_fixture(37L,17L,13L,signed)
    cpu <- device_scatter_run(f,'cpu')
    for(batched in c(FALSE,TRUE)) {
      cuda <- device_scatter_run(f,'cuda',4L,batched)
      expect_identical(cuda$errors,rep('',13L))
      expect_false(any(cuda$recovered))
      for(q in c('gradient','hessian'))expect_device_equal(cuda[[q]],cpu[[q]],1e-9,paste('CUDA scatter',q))
    }
  }
})

test_that('a nonfinite scatter result replays only its cell on CPU', {
  skip_on_cran()
  skip_if_no_cuda('nvrtc')
  f <- device_scatter_fixture(37L,17L,13L,TRUE)
  f$scores[1L,4L] <- Inf
  cpu <- device_scatter_run(f,'cpu')
  cuda <- device_scatter_run(f,'cuda')
  expect_identical(which(cuda$recovered),4L)
  expect_identical(cuda$errors,rep('',13L))
  expect_true(all(is.finite(cuda$gradient[,-4L])))
  for(q in c('gradient','hessian'))expect_device_equal(cuda[[q]],cpu[[q]],1e-9,paste('scatter replay',q))
})

test_that('weighted assembly selects the measured route on both sides of each gate', {
  skip_on_cran()
  skip_if_no_cuda('nvrtc')
  table <- tulpa:::.device_thresholds(read.csv(system.file('device','benchmarks.csv',package='tulpa')))
  gates <- table[table$operation=='laplace_scatter' & is.finite(table$threshold),]
  for(row in seq_len(nrow(gates))) {
    gate <- gates$threshold[row];width <- gates$variant[row]
    p <- as.integer(min(512L,max(1L,floor(sqrt(gate/120)))))
    sizes <- unique(as.integer(c(max(1,floor((gate-1)/(p*p))),ceiling((gate+1)/(p*p)))))
    for(n in sizes) {
      f <- device_scatter_fixture(n,p,13L)
      cpu <- device_scatter_run(f,'cpu',width)
      auto <- device_scatter_run(f,'auto',width)
      expected <- as.double(n)*p*p >= gate
      used <- auto$stages$cuda_calls[auto$stages$stage=='laplace_scatter'] > 0
      expect_identical(used,expected)
      batched <- sum(auto$stages$cuda_calls[auto$stages$stage=='laplace_scatter_batched_launch']) > 0
      expect_identical(batched,expected && as.double(n)*p*p >= gates$batch_threshold[row])
      expect_false(any(auto$recovered))
      for(q in c('gradient','hessian')) {
        cat(sprintf('scatter n=%d p=%d width=%d %s scale=%.17g tolerance=1e-9\n',n,p,width,q,max(1,abs(cpu[[q]]))))
        expect_device_equal(auto[[q]],cpu[[q]],1e-9,paste('scatter threshold',q))
      }
    }
  }
})
