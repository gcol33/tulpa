skip_on_cran()

test_that('compact constrained covariance readback equals full blocks on CPU and CUDA', {
  n <- 37L; p <- 2L
  x <- outer(seq_len(n),seq_len(n),function(i,j)sin(i*.13+j*.31))
  q <- crossprod(x)/n + diag(2,n)
  lower <- methods::as(Matrix::tril(Matrix::Matrix(q,sparse=TRUE)),'CsparseMatrix')
  input <- list(Q_p_per_grid=list(lower@p),Q_i_per_grid=list(lower@i),
    Q_x_per_grid=list(lower@x),n_x=n,idx=seq_len(n),n_dense=p,
    A_cols_list=list(3:19,20:37),soft_cols_list=list(c(3L,7L,20L)),
    soft_D_per_grid=list(matrix(.4,1L,1L)))
  run <- function(choice,compact) {
    session <- tulpa:::cpp_device_begin(choice,inherit=FALSE)
    on.exit(tulpa:::cpp_device_end(session))
    do.call(tulpa:::cpp_joint_inner_vcov_blocks,c(input,
      list(field_marginal=compact,compact=compact)))[[1L]]
  }
  full <- run('cpu',FALSE)
  check <- function(choice) {
    compact <- run(choice,TRUE)
    expect_device_equal(compact$fixed,full[1:2,1:2],1e-8,paste('compact',choice,'fixed covariance'))
    expect_device_equal(compact$field_diag,diag(full)[-(1:2)],1e-8,paste('compact',choice,'field diagonal'))
    expect_lt(as.numeric(object.size(compact)),as.numeric(object.size(full)))
  }
  check('cpu')
  expect_error(do.call(tulpa:::cpp_joint_inner_vcov_blocks,c(input,
    list(field_marginal=FALSE,compact=TRUE))),'requires field_marginal')
  skip_if_no_cuda(c('cudss','cusolver','nvrtc','cublas'))
  check('cuda')
})

test_that('large temporal readback agrees with tridiagonal and Schur identities in linear storage', {
  skip_if_not_slow()
  n <- 100000L; i <- seq_len(n)
  tri <- Matrix::bandSparse(n,k=c(-1L,0L,1L),diagonals=list(rep(-.3,n-1L),rep(2,n),rep(-.3,n-1L)))
  coupling <- cbind(sin(i*.13),cos(i*.21))*.01/sqrt(n)
  solved <- as.matrix(Matrix::solve(tri,coupling))
  vb <- solve(diag(3,2L)-crossprod(coupling,solved))
  d <- rep(2,n)
  for(j in 2:n) d[j] <- 2-.09/d[j-1L]
  diagonal <- numeric(n); diagonal[n] <- 1/d[n]
  for(j in seq.int(n-1L,1L)) diagonal[j] <- 1/d[j]+(.3/d[j])^2*diagonal[j+1L]
  expected <- diagonal+rowSums((solved%*%vb)*solved)
  base <- Matrix::bdiag(Matrix::Diagonal(2L,3),tri)
  cross <- Matrix::sparseMatrix(i=rep(i+2L,2L),j=rep(1:2,each=n),
    x=as.numeric(coupling),dims=c(n+2L,n+2L))
  q <- Matrix::forceSymmetric(base+cross,uplo='L')
  res <- list(Q_csc_p_per_grid=list(q@p),Q_csc_i_per_grid=list(q@i),
    Q_csc_x_per_grid=list(q@x),Q_csc_n=n+2L,modes=matrix(0,1L,n+2L),
    log_marginal=0,constraint_cols=list())
  session <- tulpa:::cpp_device_begin('cpu',inherit=FALSE)
  on.exit(tulpa:::cpp_device_end(session))
  result <- tulpa:::.nl_attach_grid_hessians(res,2L,var_cols=i+2L,units=list())
  expect_device_equal(solve(result$grid_hessians[[1L]]),vb,1e-10,'large readback fixed covariance')
  expect_device_equal(as.numeric(result$grid_field_var),expected,1e-10,'large readback field diagonal')
  expect_null(result$grid_field_cov)
  expect_lt(as.numeric(object.size(result$grid_field_var)),16*n)
})
