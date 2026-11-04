args <- commandArgs(TRUE)
root <- normalizePath(".", winslash = "/", mustWork = TRUE)
stopifnot(file.exists(file.path(root, "DESCRIPTION")))
out <- if (length(args)) args[1L] else file.path(root, "dev_notes/integ9/output")
steps <- if (length(args) > 1L) as.numeric(args[-1L]) else c(.25, .2)
stopifnot(all(is.finite(steps)), all(steps > 0))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
Rcpp::sourceCpp(file.path(root, "dev_notes/integ9/exact_multiscale.cpp"),
                cacheDir = file.path(out, "cpp_cache"))
set.seed(1L)
n <- 60L
lon <- runif(n, 0, 10)
lat <- runif(n, 0, 10)
D <- as.matrix(dist(cbind(lon, lat)))
draw <- function(s, r) as.numeric(t(chol(s^2 * exp(-D / r) + diag(1e-9, n))) %*% rnorm(n))
y <- .3 + draw(.63, .5) + draw(1, 3) + rnorm(n, sd = sqrt(.1))
stopifnot(length(y) == 60L, all(dim(D) == 60L))
cat("Observed locations:", n, "\n")
for (h in steps) {
  axes <- list(seq(-18, 3, by = h), seq(log(.02), log(1500), by = h),
               seq(-18, 3, by = h), seq(log(.1), log(30000), by = h))
  physical <- axes
  physical[c(2L, 4L)] <- lapply(axes[c(2L, 4L)], function(x) x + log(2))
  start <- proc.time()[["elapsed"]]
  # rho = 2 ell and D_physical = 2 D preserve exp(-D / ell).
  result <- exact_ms_grid(physical, 2 * D, y, .1)
  result$seconds <- proc.time()[["elapsed"]] - start
  result$axes <- axes
  result$cells <- prod(lengths(axes))
  result$q <- t(vapply(seq_len(4L), function(j) {
    m <- result$marg[[j]]
    a <- axes[[j]]
    approx(c(0, cumsum(m)), c(a - h / 2, max(a) + h / 2),
           c(.025, .5, .975), ties = "ordered")$y
  }, numeric(3L)))
  result$sd <- vapply(seq_len(4L), function(j) {
    m <- result$marg[[j]]
    a <- axes[[j]]
    sqrt(sum(m * (a - sum(m * a))^2))
  }, numeric(1L))
  result$boundary <- max(vapply(result$marg, function(m) max(m[c(1L, length(m))]), numeric(1L)))
  stopifnot(all(is.finite(result$q)), all(is.finite(result$sd)),
            all(result$sd > 0), max(abs(vapply(result$marg, sum, numeric(1L)) - 1)) < 1e-10)
  print(result[c("cells", "q", "sd", "log_evidence", "boundary", "seconds")], digits = 12)
  saveRDS(result, file.path(out, paste0("physical_range_oracle_", h, ".rds")))
}
