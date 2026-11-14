args <- commandArgs(trailingOnly = TRUE)
if (length(args)) .libPaths(c(args[[1L]], .libPaths()))
library(tulpa)
n <- 257L
p <- as.integer(c(0L, cumsum(c(rep(2L, n - 1L), 1L))))
i <- as.integer(unlist(lapply(seq.int(0L, n - 1L), function(j) {
  c(j, if (j + 1L < n) j + 1L)
})))
x <- c(rep(c(4, -1), n - 1L), 4)
stopifnot(length(p) == n + 1L, length(i) == 2L * n - 1L,
          length(x) == length(i), p[n + 1L] == length(i))
result <- tulpa:::cpp_device_sparse(p, i, matrix(x, ncol = 1L),
                                   matrix(1, n, 1L), 'cuda')
stopifnot(result$ok, result$used_cuda)
print(result[c('ok','used_cuda','logdet','analyses')])
