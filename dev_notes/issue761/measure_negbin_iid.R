# gcol33/tulpa#761: does the negative-binomial iid Gibbs kernel sample its stated
# posterior? Each arm is scored against an adaptive random-walk Metropolis chain on
# the same log density, written here independently of the kernel.
#
#   Rscript measure_negbin_iid.R <arm> <out.rds> [lib]
#   arm = "gibbs" (the package loaded from [lib], or load_all() of the tree when
#   [lib] is absent) or "reference".
args <- commandArgs(trailingOnly = TRUE)
arm <- args[1]; out <- args[2]; lib <- if (length(args) >= 3) args[3] else NA

set.seed(761)
J <- 6L; per <- 10L
g <- rep(seq_len(J), each = per)
N <- length(g)
X <- cbind(1, rnorm(N))
b_true <- rnorm(J, 0, 0.6)
y <- as.integer(rnbinom(N, size = 5, mu = exp(1.0 + 0.4 * X[, 2] + b_true[g])))
PRI <- list(beta_sd = 2.5, sigma_scale = 1, r_shape = 2, r_rate = 0.2)

if (arm == "gibbs") {
  if (!is.na(lib)) {
    library(tulpa, lib.loc = lib)
    kern <- getFromNamespace("cpp_pg_negbin_gibbs", "tulpa")
  } else {
    setwd("C:/GillesC/Documents/dev/tulpa")
    options(pkg.build_extra_flags = FALSE)
    suppressMessages(devtools::load_all(quiet = TRUE))
    kern <- cpp_pg_negbin_gibbs
  }
  res <- lapply(1:4, function(ch) {
    set.seed(100 + ch)
    kern(y, X, g, J, 30000L, 3000L, 1L, PRI$beta_sd, PRI$sigma_scale,
         PRI$r_shape, PRI$r_rate, 5, FALSE, FALSE, 1L)
  })
  draws <- do.call(rbind, lapply(res, function(r)
    cbind(b0 = r$beta[, 1], b1 = r$beta[, 2], sigma = r$sigma_re, r = r$r,
          re_mean = rowMeans(r$re))))
  saveRDS(draws, out)
  quit(save = "no")
}

# Reference: the density the kernel states, on (beta_zhou, re, log sigma, log r).
lpost <- function(th) {
  bz <- th[1:2]; re <- th[3:(2 + J)]; s <- exp(th[3 + J]); r <- exp(th[4 + J])
  if (r < 0.1 || r > 500) return(-Inf)
  bnb <- bz + c(log(r), 0)
  sum(dnbinom(y, size = r, mu = exp(drop(X %*% bnb) + re[g]), log = TRUE)) +
    sum(dnorm(bz, 0, PRI$beta_sd, log = TRUE)) + sum(dnorm(re, 0, s, log = TRUE)) +
    log(2) + dcauchy(s, 0, PRI$sigma_scale, log = TRUE) +
    dgamma(r, PRI$r_shape, PRI$r_rate, log = TRUE) + th[3 + J] + th[4 + J]
}
d <- J + 4L
run_am <- function(seed, n_iter, burn) {
  set.seed(seed)
  th <- c(log(mean(y)) - log(5), 0, rep(0, J), log(0.5), log(5))
  lp <- lpost(th)
  S <- diag(0.05, d); mu <- th; C <- diag(0.01, d)
  L <- chol(S)
  keep <- matrix(NA_real_, n_iter - burn, d)
  for (it in seq_len(n_iter)) {
    prop <- th + drop(crossprod(L, rnorm(d)))
    lpp <- lpost(prop)
    if (is.finite(lpp) && log(runif(1)) < lpp - lp) { th <- prop; lp <- lpp }
    if (it <= burn) {
      w <- 1 / it
      dm <- th - mu; mu <- mu + w * dm
      C <- (1 - w) * C + w * tcrossprod(dm)
      if (it %% 500 == 0 && it > 2000) L <- chol((2.38^2 / d) * C + diag(1e-8, d))
    } else {
      keep[it - burn, ] <- th
    }
  }
  keep
}
ch <- lapply(1:4, function(k) run_am(900 + k, 260000L, 60000L))
th <- do.call(rbind, ch)
r <- exp(th[, 4 + J])
draws <- cbind(b0 = th[, 1] + log(r), b1 = th[, 2], sigma = exp(th[, 3 + J]), r = r,
               re_mean = rowMeans(th[, 3:(2 + J)]))
saveRDS(draws, out)
