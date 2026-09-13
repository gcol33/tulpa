# Does tulpa()'s nested-Laplace route carry an offset() term?
#
# Run from the tulpa repo root:  Rscript dev_notes/issue_offset_nested/nested_drops_offset.R
#
# If the offset reaches the inner solve, the coefficients move when it is added
# (here the intercept should fall by about mean(off) = 0.6). If it is dropped,
# the two fits are identical and the intercept absorbs the offset's mean.

suppressMessages(library(tulpa))
set.seed(1)
nT <- 20; n <- 400
time <- sample.int(nT, n, replace = TRUE)
trend <- cumsum(rnorm(nT, 0, 0.3)); trend <- trend - mean(trend)
x <- rnorm(n)
off <- log(runif(n, 1, 3))
df <- data.frame(y = rpois(n, exp(0.2 + 0.5 * x + trend[time] + off)),
                 x = x, time = time, off = off)

f1 <- tulpa(y ~ x + offset(off), data = df, family = "poisson",
            temporal = temporal_rw1("time"))
f0 <- tulpa(y ~ x, data = df, family = "poisson",
            temporal = temporal_rw1("time"))
cat("backend:", f1$backend, "\n")
cat("coef, offset(off) in formula:", format(coef(f1), digits = 6), "\n")
cat("coef, no offset             :", format(coef(f0), digits = 6), "\n")
cat("identical:", identical(coef(f1), coef(f0)), "\n")
