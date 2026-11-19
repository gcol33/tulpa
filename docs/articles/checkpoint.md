# Checkpointing

A nested Laplace fit solves one inner problem per cell of its
hyperparameter grid, and a NUTS fit runs several independent chains.
Either can run for hours, and a reboot or a killed job near the end
would otherwise lose all of it. With `control$checkpoint` set, tulpa
appends every finished cell or chain to a file; a later call with the
same inputs loads what is finished and computes only the rest. This
article turns checkpointing on, cuts the file the way a kill during a
write would, resumes from it, and shows the file being refused once the
data change.

Installation is covered in [Get
started](https://gillescolling.com/tulpa/articles/tulpa.md).

``` r

library(tulpa)
```

## Turn it on

`control$checkpoint = list(path =, resume =)` takes the file and whether
to continue from it. With a `path` given, `resume` defaults to `TRUE`;
`resume = FALSE` removes any existing file and starts over. The example
is a binomial model with an intrinsic CAR field on a chain of 30
regions, fitted by nested Laplace through `mode = "structured"`.
`fit_ckpt()` wraps the call so the later sections can repeat it with
other data or another `resume`.

``` r

set.seed(20260716)
S <- 30L
W <- matrix(0, S, S)
for (i in 1:(S - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
df <- data.frame(region = factor(seq_len(S)))
df$x <- as.integer(df$region) / 10 + rnorm(S, 0, 0.3)
df$y <- rbinom(S, 20, plogis(-0.4 + 0.5 * df$x))

ckpt <- tempfile(fileext = ".ckpt")
fit_ckpt <- function(data, resume) {
  tulpa(y ~ x + spatial(region), data = data, family = "binomial",
        n_trials = rep(20L, S), spatial = spatial_car(W, level = "obs"),
        mode = "structured",
        control = list(n_threads = 2,
                       checkpoint = list(path = ckpt, resume = resume)))
}
fit <- fit_ckpt(df, resume = FALSE)
#> [nested-laplace] 1/9 cells (11%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 6/6 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
#> [nested-laplace] 1/5 cells (20%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 5/5 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
coef(fit)
#> (Intercept)           x 
#>  -0.2760639   0.4225753
full_size <- file.size(ckpt)
full_size
#> [1] 80536
```

The file starts with a header holding a fingerprint of the inputs: the
response, trial counts, design matrix, offsets, the field’s structure
and the solver settings. Each solved grid cell follows as one record,
keyed by its hyperparameter coordinate and closed by a checksum over key
and payload.

## Resume after an interruption

A job killed while it writes leaves a file that ends in the middle of a
record. Cutting the file to 45% of its bytes reproduces that: the
records before the cut are intact, the last one is torn, and the rest
are missing.

``` r

bytes <- readBin(ckpt, "raw", full_size)
writeBin(bytes[seq_len(floor(0.45 * full_size))], ckpt)
file.size(ckpt)
#> [1] 36241
```

The same call with `resume = TRUE` reads the intact records, detects the
torn one by its short length or failed checksum, truncates the file to
the last good record, and solves the cells that are not on disk,
appending each as it finishes.

``` r

fit2 <- fit_ckpt(df, resume = TRUE)
#> [nested-laplace] 1/2 cells (50%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 2/2 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
#> [nested-laplace] 1/5 cells (20%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 5/5 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
c(size_after_resume = file.size(ckpt), size_uninterrupted = full_size)
#>  size_after_resume size_uninterrupted 
#>              80536              80536
all.equal(coef(fit), coef(fit2))
#> [1] TRUE
```

The file is complete again, and the resumed fit returns the coefficients
of the uninterrupted one.

## A changed input is refused

Resuming onto a file written for other inputs would mix cells from two
models. Changing a single count changes the fingerprint, and the resume
stops before solving anything:

``` r

df_changed <- df
df_changed$y[1] <- df_changed$y[1] + 1L
fit_ckpt(df_changed, resume = TRUE)
#> Error:
#> ! checkpoint: 'C:\Users\GILLES~1\AppData\Local\Temp\RtmpcJYJRD\filec9bc2daf6789.ckpt' was written for different data or solver settings (fingerprint mismatch). Resume needs the same inputs + grid + control; use a fresh path or set checkpoint$resume = FALSE to start over.
```

`resume = FALSE` on the same path discards the old file and fits the new
data from the start:

``` r

fit3 <- fit_ckpt(df_changed, resume = FALSE)
#> [nested-laplace] 1/9 cells (11%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 6/6 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
#> [nested-laplace] 1/5 cells (20%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 5/5 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
coef(fit3)
#> (Intercept)           x 
#>  -0.2341845   0.4000448
```

## Sampler chains

The sampler kernel of `mode = "hmc"` takes the same
`control$checkpoint`, with a chain as the unit. A chain is deterministic
in its seed, its chain number, the data and the settings, so a resumed
run returns exactly the draws of the uninterrupted one. The example is a
Poisson model with a random intercept over 10 groups, four chains, cut
at 60% of the file:

``` r

set.seed(1)
g <- factor(rep(1:10, each = 10))
d <- data.frame(g = g, x = rnorm(100))
d$y <- rpois(100, exp(0.3 + 0.4 * d$x + rnorm(10, 0, 0.5)[g]))

ckpt_chains <- tempfile(fileext = ".ckpt")
fit_chains <- function(resume) {
  tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "hmc",
        control = list(n_chains = 4, n_iter = 1000, warmup = 500, seed = 11,
                       checkpoint = list(path = ckpt_chains, resume = resume)))
}
run1 <- fit_chains(resume = FALSE)
size <- file.size(ckpt_chains)
writeBin(readBin(ckpt_chains, "raw", size)[seq_len(floor(0.6 * size))],
         ckpt_chains)
run2 <- fit_chains(resume = TRUE)
identical(run1$draws, run2$draws)
#> [1] TRUE
```

The records are stored in the byte order of the machine that wrote them,
so a checkpoint is for resuming on that machine, not for moving a fit
between platforms. The fitters that take `control$checkpoint` are the
nested Laplace doors
([`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md),
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
[`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
and [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) when
it routes to them) and the NUTS/HMC sampler; other sampler backends
refuse the key.

## See also

- [Inference
  modes](https://gillescolling.com/tulpa/articles/inference-modes.md):
  which mode routes to nested Laplace and which to a sampler.
- [`?tulpa_nested_laplace`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md),
  section on `control`, for the checkpoint entry alongside the other
  grid knobs.
