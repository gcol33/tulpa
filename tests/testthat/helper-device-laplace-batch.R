device_laplace_fixture <- function(n, cells = 64L) {
  stopifnot(n >= 1L, cells >= 1L)
  i <- seq_len(n)
  base <- outer(i, i, function(x, y) exp(-abs(x-y)/7))
  matrices <- t(vapply(seq_len(cells), function(k)
    as.vector(t(base + diag(0.5 + i/n + k/cells, n))), numeric(n*n)))
  rhs <- outer(seq_len(cells), i, function(k, j) sin(k*0.3 + j*0.17))
  stopifnot(nrow(matrices) == cells, ncol(matrices) == n*n,
    identical(dim(rhs), c(as.integer(cells), as.integer(n))))
  list(matrices = matrices, rhs = rhs, n = as.integer(n))
}

device_laplace_run <- function(fixture, choice, workers = 4L, batched = TRUE, fail_cell = -1L) {
  do.call(tulpa:::cpp_device_laplace_batch, c(fixture,
    list(choice = choice, workers = workers, batched = batched, fail_cell = fail_cell)))
}

device_full_csc_fixture <- function(fixture) {
  n <- fixture$n
  p <- as.integer(c(0,cumsum(rev(seq_len(n)))))
  i <- as.integer(unlist(lapply(seq_len(n)-1L,function(j)seq.int(j,n-1L))))
  columns <- rep(seq_len(n)-1L,rev(seq_len(n)))
  list(p=p,i=i,values=t(fixture$matrices[,as.double(i)*n+columns+1L,drop=FALSE]),
       b=t(fixture$rhs),per_cell_rhs=TRUE)
}

device_scatter_fixture <- function(rows, columns, cells = 128L, signed = FALSE) {
  i <- seq_len(rows); j <- seq_len(columns); k <- seq_len(cells)
  design <- outer(i,j,function(x,y)sin(x*0.13+y*0.17)+cos(x*y*0.011))
  scores <- outer(i,k,function(x,y)sin(x*0.31+y*0.7))
  weights <- outer(i,k,function(x,y)0.3+abs(cos(x*0.21+y*0.09)))
  if (signed) weights[seq.int(1L,rows,by=3L),] <- -weights[seq.int(1L,rows,by=3L),]
  scales <- outer(j,k,function(x,y)0.8+0.2*cos(x*0.19+y*0.03))
  list(design=design,scores=scores,weights=weights,scales=scales)
}

device_scatter_run <- function(fixture, choice, workers = 4L, batched = TRUE) {
  do.call(tulpa:::cpp_device_scatter_batch,c(fixture,list(choice=choice,workers=workers,batched=batched)))
}
