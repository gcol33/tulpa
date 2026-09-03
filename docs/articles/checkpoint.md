# Checkpoint and resume long fits

``` r

library(tulpa)
```

## Why checkpoint

A nested-Laplace fit is a loop over independent hyperparameter grid
cells; a multi-chain NUTS fit is a loop over independent chains. Both
can run for a long time, and a killed or rebooted run should not have to
start over. tulpa writes each finished unit to a content-addressed
append log: on resume it reloads the completed units and runs only the
rest, and a run interrupted mid-write is detected and re-run rather than
trusted.

## Turn it on

Every nested-Laplace fitter takes
`control$checkpoint = list(path =, resume =)`. `path` is the checkpoint
file; `resume = TRUE` reloads any completed cells from it,
`resume = FALSE` (the default) starts fresh and overwrites.

``` r

# A small nested-Laplace fit over an ICAR field on a chain graph.
S <- 30L
W <- matrix(0, S, S)
for (i in 1:(S - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
df <- data.frame(region = factor(seq_len(S)))
df$x <- as.integer(df$region) / 10 + rnorm(S, 0, 0.3)
df$y <- rbinom(S, 20, plogis(-0.4 + 0.5 * df$x))

ckpt <- tempfile(fileext = ".ckpt")
fit <- tulpa(y ~ x + spatial(region), data = df, family = "binomial",
             n_trials = rep(20L, S),
             spatial = spatial_car(W, level = "obs"),
             mode = "laplace",
             control = list(checkpoint = list(path = ckpt, resume = FALSE)))
coef(fit)
#> (Intercept)           x 
#>  0.08647794  0.19906346
```

The grid cells are now on disk:

``` r

file.exists(ckpt) && file.info(ckpt)$size > 0
#> [1] FALSE
```

## Resume

A second call with the same data, settings, and grid plus
`resume = TRUE` reloads every completed cell instead of re-solving it –
so this call returns essentially instantly and to the same result.

``` r

fit2 <- tulpa(y ~ x + spatial(region), data = df, family = "binomial",
              n_trials = rep(20L, S),
              spatial = spatial_car(W, level = "obs"),
              mode = "laplace",
              control = list(checkpoint = list(path = ckpt, resume = TRUE)))
all.equal(coef(fit), coef(fit2))
#> [1] TRUE
```

## What the fingerprint protects

The checkpoint header carries a fingerprint of the data, settings, and
grid. Resuming onto a file written for a *different* fit errors rather
than silently mixing results, and a torn final record (a run killed
mid-write) is truncated and re-run. A multi-chain NUTS resume is
bit-for-bit identical to the uninterrupted run, because a chain is
deterministic in its seed and data.
