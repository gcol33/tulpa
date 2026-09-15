# gcol33/tulpa#761: does the binomial NNGP Gibbs kernel sample its stated
# posterior? Each arm is scored against an adaptive random-walk Metropolis chain
# on the same log density, written here independently of the kernel.
#
#   Rscript measure_binomial_nngp.R <arm> <out.rds> [lib]
args <- commandArgs(trailingOnly = TRUE)
arm <- args[1]; out <- args[2]; lib <- if (length(args) >= 3) args[3] else NA

set.seed(7612)
n <- 10L
coords <- cbind(runif(n), runif(n))
ord <- order(coords[, 1])
nn_idx <- matrix(c(0L, seq_len(n - 1L)), ncol = 1L)
d1 <- c(0, sqrt(rowSums((coords[ord[-1], , drop = FALSE] -
                         coords[ord[-n], , drop = FALSE])^2)))
nn_dist <- matrix(d1, ncol = 1L)
X <- cbind(1, rnorm(n))
ntr <- rep(8L, n)
w_true <- as.numeric(t(chol(exp(-as.matrix(dist(coords)) / 0.4))) %*% rnorm(n)) * 0.8
y <- as.integer(rbinom(n, ntr, plogis(0.8 + 0.5 * X[, 2] + w_true)))
PRI <- list(beta_sd = 2.5, U = 1, alpha = 0.05, lo = 0.05, hi = 2)

if (arm == "gibbs") {
  if (!is.na(lib)) {
    library(tulpa, lib.loc = lib)
    kern <- getFromNamespace("cpp_pg_binomial_gibbs_gp", "tulpa")
  } else {
    setwd("C:/GillesC/Documents/dev/tulpa")
    options(pkg.build_extra_flags = FALSE)
    suppressMessages(devtools::load_all(quiet = TRUE))
    kern <- cpp_pg_binomial_gibbs_gp
  }
  res <- lapply(1:4, function(ch) {
    set.seed(200 + ch)
    kern(y, ntr, X, rep(1L, n), 0L, coords, nn_idx, nn_dist, as.integer(ord - 1L),
         n, 1L, 0.5, 0.5, 0L, 60000L, 5000L, 1L, prior_beta_sd = PRI$beta_sd,
         prior_sigma_gp_U = PRI$U, prior_sigma_gp_alpha = PRI$alpha,
         prior_phi_lower = PRI$lo, prior_phi_upper = PRI$hi, verbose = FALSE)
  })
  draws <- do.call(rbind, lapply(res, function(r)
    cbind(b0 = r$beta[, 1], b1 = r$beta[, 2], sigma2 = r$sigma2_gp,
          phi = r$phi_gp, w_mean = rowMeans(r$gp))))
  saveRDS(draws, out)
  quit(save = "no")
}

nug <- 1e-8
lpost <- function(th) {
  b <- th[1:2]; w <- th[3:(2 + n)]
  s2 <- exp(th[3 + n]); u <- plogis(th[4 + n]); phi <- PRI$lo + (PRI$hi - PRI$lo) * u
  lp <- sum(dbinom(y, ntr, plogis(drop(X %*% b) + w), log = TRUE)) +
    sum(dnorm(b, 0, PRI$beta_sd, log = TRUE))
  o <- ord
  lp <- lp + dnorm(w[o[1]], 0, sqrt(s2), log = TRUE)
  rho <- exp(-d1[-1] / phi)
  B <- rho / (1 + nug); F <- pmax(1 - rho^2 / (1 + nug), 1e-10)
  lp <- lp + sum(dnorm(w[o[-1]], B * w[o[-n]], sqrt(s2 * F), log = TRUE))
  lam <- -log(PRI$alpha) / PRI$U; s <- sqrt(s2)
  lp + log(lam) - lam * s - log(2 * s) - log(PRI$hi - PRI$lo) +
    th[3 + n] + log(PRI$hi - PRI$lo) + log(u) + log1p(-u)
}
d <- n + 4L
run_am <- function(seed, n_iter, burn) {
  set.seed(seed)
  th <- c(0, 0, rep(0, n), log(0.5), 0)
  lp <- lpost(th)
  mu <- th; C <- diag(0.01, d); L <- chol(diag(0.02, d))
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
ch <- lapply(1:4, function(k) run_am(700 + k, 400000L, 80000L))
th <- do.call(rbind, ch)
draws <- cbind(b0 = th[, 1], b1 = th[, 2], sigma2 = exp(th[, 3 + n]),
               phi = PRI$lo + (PRI$hi - PRI$lo) * plogis(th[, 4 + n]),
               w_mean = rowMeans(th[, 3:(2 + n)]))
saveRDS(draws, out)
