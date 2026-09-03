# Inference modes: the three tiers and how to choose one

``` r

library(tulpa)
```

## Why the inference choice matters

A fitted model is two things at once: a set of parameter estimates and a
claim about how certain those estimates are. The estimates often agree
across methods. The uncertainty does not. A Laplace approximation, a
gradient sampler, and a variational fit can return the same posterior
mean for a slope and still disagree about the width of its credible
interval, the shape of its tail, and whether two parameters are
correlated. The number you report as a 95% interval depends on which
method produced it, and the methods do not all mean the same thing by
“95%”.

Most fitting interfaces hide this. You call a fit function, a default
runs, and the output looks identical no matter which numerical engine
did the work. The default is usually a reasonable one. The problem is
that it is silent: when the default is wrong for your model, nothing in
the output tells you, and the failure shows up downstream as an interval
that is too narrow or a correlation that was never there.

tulpa makes the choice explicit. Every fit carries the mode it ran, the
tier that mode belongs to, and a recorded reason for the selection. The
tier carries the weight, because it pins down what the credible
intervals mean. Two backends sharing a tier offer the same epistemic
warranty; a backend one rung higher offers a weaker one. That warranty
reads straight off the fitted object, and `mode = "auto"` picks among
tiers by a rule you can inspect rather than by a buried heuristic.

This vignette walks through the tier system, the automatic selection
rule, and the backends you can drive from R. It fits one model three
ways and compares the estimates and the timing, so the
cost-versus-guarantee trade is concrete rather than abstract. It closes
with practical guidance: which mode to reach for, and at what point to
move up a tier.

## The three tiers

tulpa sorts every inference backend into one of three tiers. The tier is
not a speed label. The tier states how correct the uncertainty a backend
reports actually is.
[`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)
prints the full map, tier by tier, with the guarantee attached to each.

``` r

inference_mode_info()
#> tulpa Inference Modes
#> ========================
#> 
#> Modes encode EPISTEMIC GUARANTEES, not just runtime characteristics.
#> 
#> TIER 1: EXACT
#>   Asymptotically correct posterior inference
#>   Guarantee: Credible intervals interpretable as posterior uncertainty
#>   Backends: hmc, ess, pg, gibbs, re_cov_gibbs, smc, imh_laplace, mala
#>   Note: Reference standard
#> 
#> TIER 2: STRUCTURED
#>   Accurate inference conditional on structural assumptions
#>   Guarantee: Correct if model meets structural assumptions
#>   Backends: laplace, re_cov_nested, pathfinder, agq, ep, eb, nested_laplace, nested_laplace_joint, spde
#>   Note: Controlled approximation, not heuristics
#> 
#> TIER 3: OPTIMIZED
#>   Optimization-based approximate inference
#>   Guarantee: No general correctness guarantee
#>   Backends: sghmc, sgld, mclmc, vi
#>   Note: Uncertainty often underestimated; requires explicit opt-in
#> 
#> AUTO MODE
#>   Selects between Tier 1 and Tier 2 only.
#>   NEVER silently chooses Tier 3 (Optimized).
#>   Contract: 'Use the most reliable method expected to finish.'
#> 
#> Usage (by tier):
#>   tulpa(..., mode = 'auto')       # Tier 1 or 2 (default)
#>   tulpa(..., mode = 'exact')      # Tier 1
#>   tulpa(..., mode = 'structured') # Tier 2
#>   tulpa(..., mode = 'optimized')  # Tier 3 (explicit opt-in)
#> 
#> Backends ([R] = callable from R, [C-ABI] = model-package kernel only):
#>   [R]     mode = 'hmc'        # Tier 1 (Exact)
#>   [R]     mode = 'ess'        # Tier 1 (Exact)
#>   [C-ABI] mode = 'pg'         # Tier 1 (Exact)
#>   [R]     mode = 'gibbs'      # Tier 1 (Exact)
#>   [R]     mode = 're_cov_gibbs' # Tier 1 (Exact)
#>   [R]     mode = 'smc'        # Tier 1 (Exact)
#>   [R]     mode = 'imh_laplace' # Tier 1 (Exact)
#>   [R]     mode = 'mala'       # Tier 1 (Exact)
#>   [R]     mode = 'laplace'    # Tier 2 (Structured)
#>   [R]     mode = 're_cov_nested' # Tier 2 (Structured)
#>   [R]     mode = 'pathfinder' # Tier 2 (Structured)
#>   [R]     mode = 'agq'        # Tier 2 (Structured)
#>   [R]     mode = 'ep'         # Tier 2 (Structured)
#>   [R]     mode = 'eb'         # Tier 2 (Structured)
#>   [R]     mode = 'nested_laplace' # Tier 2 (Structured)
#>   [R]     mode = 'nested_laplace_joint' # Tier 2 (Structured)
#>   [R]     mode = 'spde'       # Tier 2 (Structured)
#>   [R]     mode = 'sghmc'      # Tier 3 (Optimized)
#>   [R]     mode = 'sgld'       # Tier 3 (Optimized)
#>   [R]     mode = 'mclmc'      # Tier 3 (Optimized)
#>   [R]     mode = 'vi'         # Tier 3 (Optimized)
```

The output names three tiers. Read them as a ladder of promises.

**Tier 1, Exact.** A Tier 1 backend draws from the posterior and its
credible intervals are interpretable as posterior uncertainty, up to
Monte Carlo error that shrinks as you draw more samples. This is the
reference standard: if you want an interval you can quote without an
asterisk, a Tier 1 fit gives it to you. The cost is iteration. Every
draw is a likelihood evaluation, sometimes a gradient evaluation, and
you need many of them. MALA and the independence-MH samplers live here,
along with the Polya-Gamma Gibbs sampler for the families that admit it.

**Tier 2, Structured.** A Tier 2 backend is accurate conditional on an
explicit structural assumption. The Laplace approximation assumes the
posterior is Gaussian near its mode. Pathfinder fits a Gaussian along an
optimisation path. When the assumption holds, and for a latent Gaussian
model with enough data it usually does, the fit is fast and its
intervals are close to the exact ones. When the assumption fails, the
failure is predictable: a skewed posterior gets symmetrised, a heavy
tail gets clipped. You can see it coming from the model, and you can
check it by comparing against a Tier 1 fit. The promise is conditional,
and the condition is stated.

**Tier 3, Optimized.** A Tier 3 backend gives a point estimate and a
covariance from an optimisation objective with no general correctness
guarantee on the uncertainty. The mode is usually good. The spread is
often too small, the tails unreliable, the correlations approximate, and
the failure is typically silent. Generic variational inference sits
here. Tier 3 is optimisation, not sampling, and tulpa treats it as a
deliberate choice rather than a fallback, reachable only by asking for
it by name.

The ladder runs from a promise you can quote, through a promise with a
stated condition, to a tool with no promise on its uncertainty. Moving
up a tier costs computation and buys a stronger guarantee. The point of
the system is that the guarantee is never traded away without a record
of the trade.

The design follows four rules, and they are worth stating because they
explain why the tier shows up on every fit. First, a mode changes the
meaning of the output, not just its runtime: an interval from the
optimised tier does not mean what an interval from the exact tier means,
so the two must be distinguishable. Second, the mode is always visible
in the result. Third, there is no silent upgrading or downgrading
between tiers; if an exact fit fails, the call errors rather than
quietly substituting a structured approximation and handing back numbers
that look the same but promise less. Fourth, adding a backend slots it
into an existing tier and never invents a new kind of promise, so the
three guarantees above are the whole vocabulary. Tier membership for
each backend is derived from a single registry, which is why
[`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)
and the field on a fit can never drift apart.

## How `mode = "auto"` chooses

The default mode is `"auto"`. Its contract is one sentence: use the most
reliable method that is expected to finish for this model. Two parts of
that sentence carry weight. “Most reliable” means it prefers Tier 1 over
Tier 2 when both are feasible. “Expected to finish” means it steps down
to Tier 2 when a Tier 1 sampler would be too slow to be practical. The
rule never reaches Tier 3. The optimisation tier is never selected on
your behalf, because its silent under-coverage is exactly the failure an
automatic default should not introduce.

The decision is deterministic and depends on the model, not on chance.
Two properties drive it: structure and size. A latent prior block goes
to nested Laplace, the designed Tier 2 hot path for latent Gaussian
structure. A spatial field branches by its kind, with a binomial areal
field handed to the exact component-wise Polya-Gamma Gibbs sampler
(Tier 1) and most other fields sent to nested-Laplace integration of the
spatial hyperparameter (Tier 2). An ordinary model carrying neither
spatial nor latent structure heads for a full sampler, unless the
dataset grows past tens of thousands of rows, where the Laplace path
takes over because a sampler would no longer finish quickly enough to
serve as a default.

Every fit records the reason. The `selection_reason` field is a short
string explaining the branch that was taken, and `backend` and
`inference_tier` report the result. Take a binomial areal model: twenty
regions on a ring, each adjacent to its two neighbours, with a latent
field that varies smoothly around the loop.

``` r

n <- 400L; K <- 20L
W <- matrix(0, K, K)
for (i in 1:K) { j <- if (i < K) i + 1L else 1L; W[i, j] <- W[j, i] <- 1 }
x      <- rnorm(n)
region <- factor(sample(1:K, n, replace = TRUE))
field  <- as.numeric(scale(sin(2 * pi * (1:K) / K)))[region]
ds <- data.frame(y = rbinom(n, 1, plogis(-0.2 + 0.7 * x + field)),
                 x = x, region = region)
```

Fitting with `mode = "auto"` lets the rule pick. For a binomial ICAR
field it selects the exact Polya-Gamma Gibbs sampler, a Tier 1 backend,
and says so.

``` r

fit_auto <- tulpa(y ~ x + spatial(region), data = ds, family = "binomial",
                  spatial = list(type = "icar", adjacency = W), mode = "auto")
c(backend = fit_auto$backend, tier = fit_auto$inference_tier)
#> backend    tier 
#> "gibbs"     "1"
fit_auto$selection_reason
#> [1] "icar spatial model (binomial Polya-Gamma Gibbs)"
```

Auto reached for the most reliable method, Tier 1, because for this
model class an exact sampler is both available and fast. A binomial
likelihood with an areal field admits the Polya-Gamma data augmentation,
which turns each conditional update into a Gaussian draw, and the field
components update one at a time. Those component-wise updates sidestep
the dimensionality that would slow a general gradient sampler on a field
of this size, so the exact route is also the practical one. The rule saw
a case where the strongest guarantee was affordable and took it. Now
change the situation to a large, plain Gaussian model where a full
sampler would be slow.

``` r

nL <- 60000L
xL <- rnorm(nL)
dL <- data.frame(y = 0.5 + 1.2 * xL + rnorm(nL, sd = 0.8), x = xL)
fit_big <- tulpa(y ~ x, data = dL, family = "gaussian", mode = "auto",
                 phi = 0.8^2)
c(backend = fit_big$backend, tier = fit_big$inference_tier)
#>   backend      tier 
#> "laplace"       "2"
fit_big$selection_reason
#> [1] "large dataset (n=60,000)"
```

Here auto stepped down to the Laplace path. The reason names the dataset
size: at sixty thousand rows the rule judges a full sampler too slow to
be the sensible default and takes the Tier 2 route instead, which for a
Gaussian likelihood is exact anyway. The estimates land on the truth.

``` r

coef(fit_big)
#> (Intercept)           x 
#>    0.506521    1.203624
```

Two models, two different branches, each recorded. The same rule that
picked an exact sampler for the small spatial model picked the
structured approximation for the large plain one, and in both cases the
choice traces to a property of the model you can name in advance. That
predictability is the whole point. You are never left guessing which
engine ran, and you can reason about what auto will do before you call
it. When you disagree with the choice, override it by passing a tier
(`"exact"` or `"structured"`) or a backend name directly, and the fit
records that you overrode it rather than papering over the change.

## The R-callable backends

A backend is the concrete algorithm a mode resolves to. tulpa registers
more backends than it exposes from R: several ship a C++ kernel
reachable by model packages that link against tulpa, with no R entry
point yet. The
[`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)
map tags each one, `[R]` for callable from R and `[C-ABI]` for the
kernel-only ones. Asking for a C-ABI backend by name from R errors with
a message that names the missing entry point rather than pretending to
dispatch. This section covers the backends you can drive from R today.

### laplace (Tier 2)

The workhorse. `mode = "laplace"` finds the posterior mode and the
curvature there, then reports a Gaussian centred at the mode with that
curvature as its precision. For a Gaussian likelihood this is exact and
returns immediately. For other families it is the Gaussian approximation
to the posterior, accurate when the posterior is close to Gaussian.

``` r

set.seed(101)
g  <- factor(sample(1:12, n, replace = TRUE))
u  <- rnorm(12, sd = 0.6)
db <- data.frame(y = rbinom(n, 1, plogis(-0.3 + 1.0 * x + u[g])),
                 x = x, g = g)
fit_lap <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                 mode = "laplace", sigma_re = 0.6)
coef(fit_lap)
#> (Intercept)           x 
#>  -0.2581721   0.8021130
```

The Laplace fit is deterministic, carries no Monte Carlo error, and its
[`logLik()`](https://rdrr.io/r/stats/logLik.html) is the approximate log
marginal likelihood, the model evidence that
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
uses. This is the first fit to run, every time, as the fast sanity check
and the comparison baseline.

### pathfinder (Tier 2)

Pathfinder runs a quasi-Newton optimiser toward the mode, fits a
Gaussian at the optimum from the inverse-Hessian estimate it accumulates
along the way, and returns draws from that Gaussian plus an ELBO score.
It sits in the same tier as Laplace: the output is a Gaussian
approximation, not exact samples. What it adds over a plain Laplace fit
is a set of draws and a diagnostic. The draws make it a drop-in where
downstream code expects samples, and the ELBO gauges how well the
Gaussian fits.

``` r

fit_pf <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                mode = "pathfinder", sigma_re = 0.6,
                control = list(n_draws = 450))
coef(fit_pf)
#> (Intercept)           x 
#>  -0.2313448   0.7959445
fit_pf$elbo
#> [1] -252.4946
```

Pathfinder fits three situations well: you want draws but not the cost
of full MCMC, you need a warm start for a sampler, or you want a quick
check on whether the Laplace Gaussian is reasonable.

### mala (Tier 1)

The Metropolis-adjusted Langevin algorithm is a gradient sampler. Each
step proposes a move along the gradient of the log posterior and accepts
or rejects it with a Metropolis step, which makes the chain
asymptotically correct. Each iteration runs cheaper than full
Hamiltonian Monte Carlo because no leapfrog trajectory needs
integrating, and mixing beats a random walk because the drift term
nudges proposals toward higher density. The step size adapts during
warmup toward an acceptance rate near 0.574.

``` r

fit_mala <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                  mode = "mala", sigma_re = 0.6,
                  control = list(n_iter = 450, warmup = 150))
coef(fit_mala)
#> (Intercept)           x 
#>   -0.330130    0.805812
fit_mala$mean_accept
#> [1] 0.5493333
```

MALA earns its keep when you want exact posterior moments at moderate
dimension and the posterior is not so badly scaled that a single step
size struggles across all directions. The acceptance rate is the
diagnostic: far below the target means the chain needs a smaller step or
more warmup.

### imh_laplace (Tier 1)

Independence Metropolis-Hastings with a Laplace proposal is the cheapest
route to exact-tier draws when Laplace is almost right. It builds the
Laplace mode and precision once, then proposes from that Gaussian and
corrects with a Metropolis step. The proposal ignores the current state,
so the iterations are nearly free and the acceptance rate doubles as a
verdict on the Laplace approximation: high acceptance means the
posterior is close to the Gaussian, low acceptance means it is far and
Laplace was biased.

``` r

fit_imh <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                 mode = "imh_laplace", sigma_re = 0.6,
                 control = list(n_iter = 450, warmup = 150))
coef(fit_imh)
#> (Intercept)           x 
#>  -0.2746751   0.8342437
fit_imh$mean_accept
#> [1] 0.8833333
```

The acceptance here is high, which says the Laplace fit was already
close. Two jobs suit `imh_laplace`: debiasing a Laplace fit at low
parameter dimension, and repeated-fit workflows such as cross-validation
where a full sampler’s startup cost would dominate.

### gibbs (Tier 1)

The Gibbs backend is the Polya-Gamma sampler for binomial and negative
binomial responses, including the binomial areal spatial models that
`mode = "auto"` selects it for. It samples the random-effect standard
deviation rather than conditioning on a fixed `sigma_re`, which is why
the spatial fit earlier reported a Gibbs backend. Its output carries the
fixed effects in `$beta` and the field and variance components
alongside.

``` r

round(fit_auto$beta, 3)
#>           [,1]  [,2]
#>    [1,] -0.139 0.612
#>    [2,] -0.060 0.653
#>    [3,] -0.013 0.637
#>    [4,] -0.201 0.680
#>    [5,] -0.177 0.611
#>    [6,] -0.239 0.737
#>    [7,] -0.007 0.532
#>    [8,] -0.194 0.803
#>    [9,] -0.261 0.591
#>   [10,] -0.112 0.592
#>   [11,] -0.093 0.529
#>   [12,] -0.200 0.524
#>   [13,] -0.136 0.509
#>   [14,] -0.307 0.628
#>   [15,] -0.285 0.666
#>   [16,] -0.200 0.668
#>   [17,] -0.076 0.771
#>   [18,] -0.374 0.575
#>   [19,] -0.229 0.529
#>   [20,] -0.246 0.944
#>   [21,] -0.168 0.733
#>   [22,] -0.295 0.666
#>   [23,] -0.083 0.617
#>   [24,] -0.068 0.776
#>   [25,] -0.343 0.498
#>   [26,] -0.096 0.529
#>   [27,] -0.356 0.623
#>   [28,] -0.245 0.586
#>   [29,] -0.155 0.724
#>   [30,] -0.068 0.635
#>   [31,] -0.208 0.603
#>   [32,] -0.168 0.634
#>   [33,] -0.216 0.689
#>   [34,] -0.112 0.652
#>   [35,] -0.251 0.667
#>   [36,] -0.179 0.467
#>   [37,] -0.281 0.595
#>   [38,] -0.286 0.621
#>   [39,] -0.341 0.269
#>   [40,] -0.118 0.635
#>   [41,] -0.086 0.719
#>   [42,] -0.126 0.686
#>   [43,]  0.079 0.560
#>   [44,] -0.076 0.594
#>   [45,] -0.043 0.633
#>   [46,] -0.186 0.643
#>   [47,] -0.295 0.506
#>   [48,] -0.252 0.540
#>   [49,] -0.182 0.534
#>   [50,] -0.089 0.776
#>   [51,] -0.322 0.684
#>   [52,] -0.290 0.730
#>   [53,] -0.325 0.704
#>   [54,] -0.221 0.603
#>   [55,] -0.246 0.811
#>   [56,] -0.424 0.518
#>   [57,] -0.211 0.805
#>   [58,] -0.076 0.802
#>   [59,] -0.094 0.743
#>   [60,] -0.035 0.513
#>   [61,] -0.091 0.513
#>   [62,] -0.064 0.573
#>   [63,] -0.124 0.638
#>   [64,] -0.209 0.548
#>   [65,] -0.194 0.436
#>   [66,] -0.091 0.633
#>   [67,] -0.196 0.476
#>   [68,] -0.192 0.542
#>   [69,] -0.162 0.627
#>   [70,] -0.340 0.625
#>   [71,] -0.248 0.697
#>   [72,] -0.198 0.612
#>   [73,] -0.237 0.436
#>   [74,] -0.315 0.561
#>   [75,]  0.008 0.643
#>   [76,] -0.247 0.717
#>   [77,] -0.319 0.569
#>   [78,] -0.123 0.485
#>   [79,] -0.318 0.410
#>   [80,] -0.296 0.698
#>   [81,] -0.050 0.425
#>   [82,] -0.267 0.494
#>   [83,] -0.144 0.688
#>   [84,] -0.259 0.541
#>   [85,] -0.017 0.558
#>   [86,] -0.001 0.654
#>   [87,] -0.380 0.548
#>   [88,] -0.068 0.556
#>   [89,] -0.453 0.631
#>   [90,] -0.380 0.643
#>   [91,] -0.295 0.649
#>   [92,] -0.303 0.516
#>   [93,] -0.088 0.565
#>   [94,] -0.166 0.446
#>   [95,] -0.294 0.658
#>   [96,] -0.111 0.736
#>   [97,] -0.183 0.586
#>   [98,] -0.089 0.879
#>   [99,]  0.035 0.616
#>  [100,] -0.206 0.766
#>  [101,]  0.033 0.685
#>  [102,] -0.088 0.853
#>  [103,] -0.060 0.582
#>  [104,] -0.170 0.622
#>  [105,]  0.087 0.600
#>  [106,] -0.219 0.670
#>  [107,]  0.007 0.559
#>  [108,] -0.290 0.688
#>  [109,] -0.096 0.509
#>  [110,] -0.221 0.318
#>  [111,] -0.244 0.476
#>  [112,]  0.051 0.634
#>  [113,] -0.046 0.700
#>  [114,]  0.086 0.536
#>  [115,] -0.065 0.431
#>  [116,] -0.249 0.575
#>  [117,] -0.254 0.586
#>  [118,] -0.147 0.633
#>  [119,] -0.371 0.422
#>  [120,] -0.103 0.587
#>  [121,] -0.182 0.556
#>  [122,]  0.014 0.545
#>  [123,] -0.024 0.507
#>  [124,] -0.254 0.428
#>  [125,] -0.171 0.545
#>  [126,] -0.109 0.407
#>  [127,] -0.264 0.475
#>  [128,] -0.091 0.530
#>  [129,]  0.018 0.783
#>  [130,] -0.173 0.542
#>  [131,]  0.023 0.677
#>  [132,] -0.211 0.698
#>  [133,] -0.267 0.503
#>  [134,] -0.272 0.676
#>  [135,] -0.243 0.521
#>  [136,] -0.176 0.829
#>  [137,] -0.194 0.751
#>  [138,] -0.372 0.724
#>  [139,] -0.389 0.647
#>  [140,] -0.344 0.727
#>  [141,] -0.287 0.557
#>  [142,] -0.201 0.843
#>  [143,] -0.248 0.631
#>  [144,] -0.175 0.379
#>  [145,] -0.116 0.469
#>  [146,] -0.172 0.476
#>  [147,] -0.075 0.611
#>  [148,] -0.273 0.619
#>  [149,] -0.061 0.764
#>  [150,] -0.063 0.791
#>  [151,] -0.262 0.681
#>  [152,] -0.145 0.834
#>  [153,] -0.208 0.479
#>  [154,] -0.168 0.631
#>  [155,] -0.005 0.592
#>  [156,] -0.005 0.802
#>  [157,]  0.010 0.546
#>  [158,] -0.150 0.699
#>  [159,] -0.063 0.626
#>  [160,] -0.144 0.501
#>  [161,] -0.123 0.507
#>  [162,] -0.442 0.677
#>  [163,] -0.080 0.569
#>  [164,] -0.008 0.626
#>  [165,] -0.131 0.699
#>  [166,] -0.069 0.762
#>  [167,] -0.076 0.940
#>  [168,] -0.001 0.588
#>  [169,] -0.380 0.769
#>  [170,] -0.148 0.606
#>  [171,] -0.203 0.686
#>  [172,] -0.231 0.717
#>  [173,] -0.405 0.935
#>  [174,] -0.310 0.869
#>  [175,] -0.068 0.915
#>  [176,] -0.160 0.918
#>  [177,] -0.069 0.530
#>  [178,]  0.118 0.604
#>  [179,]  0.018 0.516
#>  [180,] -0.188 0.640
#>  [181,] -0.061 0.420
#>  [182,] -0.046 0.586
#>  [183,] -0.191 0.449
#>  [184,]  0.031 0.605
#>  [185,] -0.218 0.590
#>  [186,] -0.226 0.560
#>  [187,] -0.292 0.611
#>  [188,] -0.069 0.558
#>  [189,] -0.301 0.637
#>  [190,] -0.118 0.451
#>  [191,] -0.204 0.464
#>  [192,] -0.092 0.534
#>  [193,] -0.044 0.627
#>  [194,] -0.060 0.479
#>  [195,] -0.192 0.608
#>  [196,] -0.257 0.891
#>  [197,] -0.136 0.658
#>  [198,] -0.196 0.629
#>  [199,] -0.074 0.629
#>  [200,] -0.215 0.466
#>  [201,] -0.307 0.431
#>  [202,] -0.410 0.546
#>  [203,] -0.208 0.686
#>  [204,] -0.073 0.637
#>  [205,] -0.292 0.455
#>  [206,] -0.089 0.406
#>  [207,] -0.215 0.507
#>  [208,] -0.315 0.645
#>  [209,] -0.175 0.531
#>  [210,] -0.244 0.477
#>  [211,] -0.126 0.458
#>  [212,] -0.162 0.673
#>  [213,] -0.193 0.583
#>  [214,] -0.315 0.541
#>  [215,] -0.049 0.620
#>  [216,] -0.182 0.835
#>  [217,] -0.180 0.845
#>  [218,] -0.114 0.537
#>  [219,] -0.365 0.494
#>  [220,] -0.215 0.677
#>  [221,] -0.190 0.627
#>  [222,] -0.209 0.838
#>  [223,] -0.214 0.867
#>  [224,] -0.196 0.640
#>  [225,] -0.396 0.706
#>  [226,]  0.054 0.572
#>  [227,] -0.076 0.618
#>  [228,] -0.288 0.882
#>  [229,] -0.218 0.896
#>  [230,] -0.209 0.720
#>  [231,] -0.341 0.667
#>  [232,] -0.214 0.693
#>  [233,] -0.200 0.522
#>  [234,] -0.110 0.606
#>  [235,] -0.152 0.751
#>  [236,] -0.087 0.935
#>  [237,] -0.059 0.962
#>  [238,] -0.206 0.597
#>  [239,] -0.134 0.346
#>  [240,] -0.263 0.664
#>  [241,] -0.173 0.509
#>  [242,] -0.327 0.631
#>  [243,] -0.174 0.642
#>  [244,] -0.162 0.669
#>  [245,] -0.232 0.484
#>  [246,] -0.356 0.494
#>  [247,] -0.241 0.678
#>  [248,] -0.262 0.544
#>  [249,] -0.254 0.332
#>  [250,] -0.085 0.675
#>  [251,] -0.153 0.442
#>  [252,] -0.090 0.664
#>  [253,] -0.181 0.653
#>  [254,]  0.096 0.463
#>  [255,] -0.044 0.683
#>  [256,] -0.255 0.614
#>  [257,] -0.327 0.562
#>  [258,] -0.302 0.718
#>  [259,] -0.219 0.756
#>  [260,] -0.277 0.351
#>  [261,] -0.217 0.719
#>  [262,] -0.121 0.589
#>  [263,] -0.102 0.532
#>  [264,] -0.210 0.548
#>  [265,] -0.268 0.602
#>  [266,] -0.228 0.543
#>  [267,] -0.193 0.613
#>  [268,] -0.161 0.542
#>  [269,] -0.230 0.876
#>  [270,] -0.083 0.542
#>  [271,] -0.154 0.712
#>  [272,] -0.241 0.493
#>  [273,] -0.049 0.454
#>  [274,] -0.110 0.598
#>  [275,] -0.162 0.755
#>  [276,] -0.340 0.653
#>  [277,]  0.018 0.838
#>  [278,]  0.136 0.692
#>  [279,] -0.112 0.595
#>  [280,] -0.159 0.566
#>  [281,] -0.086 0.434
#>  [282,] -0.327 0.643
#>  [283,]  0.022 0.700
#>  [284,] -0.248 0.657
#>  [285,] -0.105 0.792
#>  [286,] -0.130 0.515
#>  [287,] -0.126 0.478
#>  [288,] -0.199 0.690
#>  [289,] -0.125 0.598
#>  [290,] -0.204 0.622
#>  [291,] -0.237 0.599
#>  [292,]  0.007 0.696
#>  [293,] -0.016 0.561
#>  [294,] -0.011 0.587
#>  [295,]  0.112 0.550
#>  [296,] -0.083 0.778
#>  [297,] -0.201 0.559
#>  [298,] -0.100 0.301
#>  [299,]  0.005 0.514
#>  [300,] -0.073 0.608
#>  [301,] -0.103 0.780
#>  [302,] -0.164 0.650
#>  [303,] -0.128 0.840
#>  [304,] -0.084 0.697
#>  [305,] -0.151 0.778
#>  [306,] -0.123 0.831
#>  [307,] -0.176 0.617
#>  [308,] -0.138 0.586
#>  [309,] -0.289 0.439
#>  [310,]  0.012 0.509
#>  [311,] -0.038 0.528
#>  [312,]  0.127 0.438
#>  [313,] -0.237 0.685
#>  [314,] -0.381 0.568
#>  [315,] -0.132 0.750
#>  [316,] -0.177 0.612
#>  [317,] -0.270 0.613
#>  [318,] -0.282 0.689
#>  [319,] -0.094 0.561
#>  [320,] -0.194 0.616
#>  [321,] -0.163 0.602
#>  [322,] -0.364 0.469
#>  [323,] -0.264 0.641
#>  [324,] -0.016 0.798
#>  [325,] -0.136 0.552
#>  [326,] -0.271 0.596
#>  [327,] -0.156 0.608
#>  [328,]  0.062 0.702
#>  [329,] -0.227 0.487
#>  [330,]  0.011 0.759
#>  [331,] -0.207 0.474
#>  [332,] -0.315 0.645
#>  [333,] -0.021 0.766
#>  [334,] -0.168 0.645
#>  [335,] -0.220 0.860
#>  [336,] -0.258 0.716
#>  [337,] -0.110 0.608
#>  [338,] -0.057 0.548
#>  [339,] -0.073 0.520
#>  [340,] -0.178 0.523
#>  [341,] -0.082 0.543
#>  [342,] -0.220 0.462
#>  [343,] -0.072 0.528
#>  [344,] -0.012 0.489
#>  [345,]  0.034 0.627
#>  [346,]  0.049 0.700
#>  [347,] -0.083 0.653
#>  [348,] -0.039 0.646
#>  [349,] -0.063 0.581
#>  [350,]  0.030 0.680
#>  [351,] -0.148 0.612
#>  [352,] -0.147 0.668
#>  [353,] -0.188 0.864
#>  [354,] -0.179 0.983
#>  [355,] -0.196 0.704
#>  [356,] -0.298 0.799
#>  [357,] -0.327 0.640
#>  [358,] -0.105 0.526
#>  [359,] -0.104 0.693
#>  [360,] -0.165 0.573
#>  [361,] -0.191 0.627
#>  [362,] -0.088 0.457
#>  [363,] -0.299 0.695
#>  [364,] -0.313 0.596
#>  [365,] -0.109 0.623
#>  [366,] -0.075 0.648
#>  [367,] -0.523 0.789
#>  [368,] -0.255 0.701
#>  [369,] -0.185 0.762
#>  [370,] -0.010 0.655
#>  [371,] -0.274 0.467
#>  [372,] -0.171 0.635
#>  [373,] -0.115 0.663
#>  [374,] -0.219 0.697
#>  [375,] -0.277 0.740
#>  [376,] -0.255 0.759
#>  [377,] -0.254 0.407
#>  [378,] -0.084 0.511
#>  [379,] -0.279 0.438
#>  [380,] -0.206 0.502
#>  [381,] -0.331 0.502
#>  [382,] -0.044 0.396
#>  [383,] -0.076 0.579
#>  [384,] -0.170 0.466
#>  [385,] -0.035 0.512
#>  [386,] -0.070 0.752
#>  [387,] -0.352 0.604
#>  [388,] -0.289 0.360
#>  [389,] -0.275 0.609
#>  [390,] -0.151 0.650
#>  [391,] -0.052 0.558
#>  [392,] -0.216 0.701
#>  [393,]  0.033 0.706
#>  [394,]  0.032 0.579
#>  [395,] -0.230 0.732
#>  [396,] -0.221 0.666
#>  [397,] -0.093 0.712
#>  [398,] -0.105 0.762
#>  [399,] -0.164 0.880
#>  [400,] -0.155 0.819
#>  [401,] -0.294 0.692
#>  [402,] -0.083 0.647
#>  [403,] -0.129 0.680
#>  [404,]  0.018 0.980
#>  [405,] -0.146 0.603
#>  [406,] -0.232 0.642
#>  [407,] -0.274 0.702
#>  [408,] -0.215 0.508
#>  [409,] -0.218 0.563
#>  [410,] -0.413 0.483
#>  [411,]  0.032 0.377
#>  [412,] -0.058 0.446
#>  [413,] -0.255 0.518
#>  [414,] -0.175 0.525
#>  [415,] -0.140 0.664
#>  [416,] -0.143 0.534
#>  [417,] -0.330 0.497
#>  [418,] -0.309 0.502
#>  [419,] -0.255 0.718
#>  [420,] -0.159 0.569
#>  [421,] -0.216 0.550
#>  [422,] -0.123 0.689
#>  [423,] -0.032 0.714
#>  [424,] -0.329 0.705
#>  [425,] -0.300 0.702
#>  [426,] -0.165 0.618
#>  [427,] -0.104 0.840
#>  [428,] -0.154 0.599
#>  [429,] -0.249 0.632
#>  [430,] -0.061 0.598
#>  [431,] -0.072 0.522
#>  [432,] -0.233 0.501
#>  [433,]  0.002 0.674
#>  [434,] -0.135 0.687
#>  [435,] -0.342 0.621
#>  [436,] -0.050 0.586
#>  [437,] -0.035 0.620
#>  [438,] -0.168 0.669
#>  [439,]  0.055 0.672
#>  [440,] -0.298 0.624
#>  [441,] -0.204 0.402
#>  [442,] -0.224 0.839
#>  [443,] -0.073 0.493
#>  [444,] -0.129 0.615
#>  [445,] -0.136 0.434
#>  [446,] -0.225 0.514
#>  [447,] -0.075 0.695
#>  [448,] -0.003 0.700
#>  [449,] -0.044 0.803
#>  [450,]  0.049 0.571
#>  [451,] -0.451 0.601
#>  [452,] -0.248 0.541
#>  [453,] -0.154 0.491
#>  [454,] -0.151 0.607
#>  [455,] -0.126 0.532
#>  [456,] -0.143 0.470
#>  [457,] -0.407 0.592
#>  [458,] -0.290 0.633
#>  [459,]  0.114 0.618
#>  [460,] -0.066 0.628
#>  [461,] -0.096 0.508
#>  [462,] -0.133 0.610
#>  [463,] -0.295 0.572
#>  [464,] -0.142 0.446
#>  [465,] -0.244 0.494
#>  [466,] -0.076 0.413
#>  [467,] -0.175 0.465
#>  [468,] -0.028 0.464
#>  [469,]  0.066 0.514
#>  [470,] -0.203 0.522
#>  [471,] -0.040 0.356
#>  [472,] -0.090 0.486
#>  [473,] -0.210 0.687
#>  [474,] -0.176 0.710
#>  [475,] -0.100 0.876
#>  [476,] -0.184 0.843
#>  [477,] -0.084 0.866
#>  [478,] -0.154 0.795
#>  [479,] -0.165 0.501
#>  [480,] -0.051 0.650
#>  [481,] -0.176 0.714
#>  [482,] -0.214 0.436
#>  [483,] -0.341 0.667
#>  [484,] -0.132 0.503
#>  [485,]  0.101 0.430
#>  [486,] -0.217 0.429
#>  [487,] -0.251 0.862
#>  [488,] -0.238 0.693
#>  [489,] -0.066 0.462
#>  [490,] -0.226 0.391
#>  [491,] -0.195 0.365
#>  [492,] -0.117 0.693
#>  [493,] -0.046 0.559
#>  [494,] -0.104 0.666
#>  [495,]  0.002 0.686
#>  [496,] -0.094 0.521
#>  [497,] -0.234 0.594
#>  [498,] -0.141 0.799
#>  [499,] -0.170 0.811
#>  [500,] -0.102 0.758
#>  [501,] -0.216 0.619
#>  [502,]  0.037 0.695
#>  [503,] -0.183 0.775
#>  [504,] -0.317 0.809
#>  [505,] -0.316 0.635
#>  [506,] -0.239 0.652
#>  [507,] -0.292 0.667
#>  [508,] -0.212 0.689
#>  [509,] -0.168 0.665
#>  [510,]  0.047 0.632
#>  [511,]  0.005 0.838
#>  [512,] -0.198 0.872
#>  [513,] -0.017 0.835
#>  [514,] -0.249 0.565
#>  [515,] -0.185 0.554
#>  [516,] -0.192 0.475
#>  [517,] -0.096 0.576
#>  [518,] -0.155 0.736
#>  [519,] -0.050 0.731
#>  [520,] -0.242 0.734
#>  [521,] -0.055 0.579
#>  [522,] -0.149 0.639
#>  [523,] -0.060 0.689
#>  [524,] -0.085 0.570
#>  [525,] -0.092 0.603
#>  [526,] -0.337 0.666
#>  [527,] -0.057 0.459
#>  [528,] -0.108 0.627
#>  [529,] -0.176 0.567
#>  [530,] -0.279 0.513
#>  [531,] -0.204 0.461
#>  [532,] -0.088 0.479
#>  [533,] -0.292 0.482
#>  [534,]  0.085 0.705
#>  [535,] -0.126 0.576
#>  [536,] -0.065 0.486
#>  [537,] -0.109 0.665
#>  [538,] -0.056 0.656
#>  [539,] -0.174 0.805
#>  [540,] -0.347 0.724
#>  [541,] -0.174 0.703
#>  [542,] -0.189 0.441
#>  [543,] -0.260 0.635
#>  [544,] -0.097 0.548
#>  [545,] -0.309 0.522
#>  [546,] -0.265 0.544
#>  [547,] -0.098 0.521
#>  [548,] -0.234 0.651
#>  [549,] -0.047 0.408
#>  [550,] -0.136 0.506
#>  [551,] -0.102 0.600
#>  [552,] -0.308 0.467
#>  [553,] -0.087 0.698
#>  [554,] -0.280 0.493
#>  [555,] -0.218 0.708
#>  [556,]  0.025 0.663
#>  [557,] -0.184 0.616
#>  [558,] -0.263 0.706
#>  [559,] -0.385 0.641
#>  [560,] -0.328 1.096
#>  [561,] -0.231 0.412
#>  [562,] -0.036 0.523
#>  [563,] -0.052 0.699
#>  [564,]  0.080 0.569
#>  [565,]  0.076 0.829
#>  [566,] -0.004 0.594
#>  [567,] -0.081 0.658
#>  [568,] -0.401 0.610
#>  [569,] -0.209 0.708
#>  [570,] -0.203 0.546
#>  [571,] -0.051 0.535
#>  [572,] -0.101 0.574
#>  [573,]  0.061 0.783
#>  [574,] -0.248 0.794
#>  [575,] -0.269 0.600
#>  [576,] -0.283 0.710
#>  [577,] -0.055 0.549
#>  [578,] -0.121 0.540
#>  [579,] -0.278 0.564
#>  [580,] -0.158 0.619
#>  [581,] -0.005 0.523
#>  [582,] -0.024 0.695
#>  [583,] -0.187 0.640
#>  [584,] -0.059 0.779
#>  [585,] -0.244 0.536
#>  [586,] -0.103 0.776
#>  [587,] -0.055 0.599
#>  [588,] -0.133 0.681
#>  [589,] -0.101 0.899
#>  [590,] -0.258 0.539
#>  [591,] -0.185 0.659
#>  [592,] -0.235 0.569
#>  [593,] -0.233 0.462
#>  [594,] -0.237 0.545
#>  [595,] -0.165 0.674
#>  [596,] -0.091 0.650
#>  [597,] -0.010 0.735
#>  [598,]  0.007 0.531
#>  [599,] -0.303 0.590
#>  [600,] -0.216 0.544
#>  [601,] -0.101 0.620
#>  [602,] -0.069 0.650
#>  [603,] -0.149 0.820
#>  [604,] -0.103 0.845
#>  [605,] -0.258 0.762
#>  [606,] -0.165 0.668
#>  [607,] -0.314 0.524
#>  [608,] -0.195 0.414
#>  [609,] -0.317 0.405
#>  [610,] -0.194 0.716
#>  [611,] -0.073 0.589
#>  [612,] -0.058 0.561
#>  [613,] -0.063 0.632
#>  [614,] -0.064 0.615
#>  [615,] -0.169 0.630
#>  [616,] -0.177 0.507
#>  [617,] -0.248 0.591
#>  [618,] -0.167 0.438
#>  [619,]  0.155 0.589
#>  [620,] -0.155 0.634
#>  [621,] -0.336 0.598
#>  [622,] -0.128 0.543
#>  [623,] -0.405 0.659
#>  [624,] -0.139 0.544
#>  [625,] -0.180 0.656
#>  [626,] -0.145 0.726
#>  [627,] -0.470 0.523
#>  [628,] -0.314 0.558
#>  [629,] -0.225 0.732
#>  [630,] -0.271 0.560
#>  [631,] -0.132 0.610
#>  [632,] -0.081 0.483
#>  [633,] -0.146 0.688
#>  [634,] -0.186 0.501
#>  [635,] -0.287 0.534
#>  [636,] -0.248 0.699
#>  [637,] -0.106 0.535
#>  [638,]  0.013 0.745
#>  [639,] -0.190 0.580
#>  [640,] -0.241 0.653
#>  [641,] -0.271 0.495
#>  [642,] -0.592 0.827
#>  [643,] -0.453 0.547
#>  [644,] -0.259 0.566
#>  [645,] -0.224 0.723
#>  [646,] -0.179 0.894
#>  [647,] -0.346 0.847
#>  [648,] -0.299 0.577
#>  [649,] -0.227 0.750
#>  [650,] -0.073 0.740
#>  [651,] -0.053 0.631
#>  [652,] -0.193 0.596
#>  [653,] -0.163 0.564
#>  [654,] -0.094 0.559
#>  [655,] -0.086 0.890
#>  [656,] -0.389 0.657
#>  [657,] -0.185 0.635
#>  [658,] -0.179 0.715
#>  [659,] -0.134 0.428
#>  [660,] -0.104 0.505
#>  [661,] -0.276 0.638
#>  [662,] -0.189 0.531
#>  [663,] -0.184 0.544
#>  [664,] -0.172 0.727
#>  [665,] -0.301 0.662
#>  [666,] -0.263 0.462
#>  [667,] -0.191 0.719
#>  [668,] -0.039 0.674
#>  [669,] -0.059 0.712
#>  [670,] -0.404 0.598
#>  [671,] -0.327 0.816
#>  [672,] -0.298 0.512
#>  [673,] -0.184 0.677
#>  [674,] -0.215 0.548
#>  [675,] -0.177 0.458
#>  [676,] -0.301 0.889
#>  [677,] -0.338 0.627
#>  [678,] -0.347 0.557
#>  [679,] -0.026 0.723
#>  [680,] -0.060 0.611
#>  [681,] -0.176 0.691
#>  [682,] -0.101 0.732
#>  [683,] -0.122 0.772
#>  [684,] -0.437 0.834
#>  [685,] -0.329 0.738
#>  [686,] -0.263 0.844
#>  [687,] -0.023 0.641
#>  [688,] -0.334 0.475
#>  [689,] -0.302 0.503
#>  [690,] -0.272 0.453
#>  [691,] -0.085 0.504
#>  [692,] -0.202 0.558
#>  [693,] -0.218 0.571
#>  [694,] -0.030 0.566
#>  [695,] -0.131 0.646
#>  [696,] -0.134 0.720
#>  [697,] -0.066 0.658
#>  [698,] -0.221 0.613
#>  [699,] -0.213 0.634
#>  [700,] -0.231 0.791
#>  [701,] -0.142 0.515
#>  [702,]  0.031 0.490
#>  [703,] -0.119 0.814
#>  [704,] -0.173 0.658
#>  [705,] -0.325 0.603
#>  [706,] -0.275 0.657
#>  [707,] -0.090 0.588
#>  [708,] -0.103 0.484
#>  [709,] -0.091 0.320
#>  [710,] -0.266 0.896
#>  [711,] -0.195 0.510
#>  [712,] -0.109 0.772
#>  [713,] -0.267 0.715
#>  [714,] -0.071 0.416
#>  [715,] -0.110 0.486
#>  [716,] -0.246 0.556
#>  [717,] -0.151 0.849
#>  [718,] -0.085 0.689
#>  [719,] -0.155 0.968
#>  [720,] -0.098 0.813
#>  [721,] -0.207 0.690
#>  [722,] -0.121 0.553
#>  [723,]  0.007 0.637
#>  [724,] -0.092 0.590
#>  [725,] -0.174 0.528
#>  [726,] -0.341 0.655
#>  [727,] -0.042 0.560
#>  [728,] -0.117 0.623
#>  [729,] -0.252 0.543
#>  [730,] -0.168 0.584
#>  [731,] -0.015 0.606
#>  [732,] -0.168 0.607
#>  [733,] -0.048 0.571
#>  [734,] -0.169 0.664
#>  [735,] -0.084 0.717
#>  [736,] -0.319 0.686
#>  [737,] -0.198 0.599
#>  [738,] -0.125 0.488
#>  [739,] -0.279 0.554
#>  [740,] -0.147 0.559
#>  [741,] -0.162 0.372
#>  [742,] -0.104 0.615
#>  [743,] -0.151 0.627
#>  [744,] -0.427 0.534
#>  [745,] -0.310 0.762
#>  [746,] -0.190 0.746
#>  [747,] -0.193 0.580
#>  [748,] -0.272 0.672
#>  [749,] -0.181 0.586
#>  [750,] -0.177 0.923
#>  [751,] -0.162 0.684
#>  [752,]  0.011 0.587
#>  [753,] -0.085 0.702
#>  [754,] -0.276 0.603
#>  [755,] -0.364 0.598
#>  [756,] -0.246 0.596
#>  [757,]  0.060 0.568
#>  [758,] -0.140 0.629
#>  [759,] -0.061 0.620
#>  [760,]  0.008 0.553
#>  [761,] -0.021 0.758
#>  [762,] -0.367 0.689
#>  [763,] -0.206 0.586
#>  [764,] -0.190 0.381
#>  [765,] -0.180 0.608
#>  [766,] -0.229 0.704
#>  [767,] -0.039 0.384
#>  [768,]  0.053 0.867
#>  [769,] -0.034 0.566
#>  [770,] -0.140 0.691
#>  [771,] -0.276 0.638
#>  [772,] -0.320 0.554
#>  [773,] -0.496 0.656
#>  [774,] -0.193 0.689
#>  [775,] -0.147 0.770
#>  [776,] -0.247 0.806
#>  [777,] -0.215 0.545
#>  [778,] -0.291 0.712
#>  [779,] -0.067 0.569
#>  [780,] -0.198 0.501
#>  [781,] -0.207 0.459
#>  [782,] -0.067 0.508
#>  [783,]  0.055 0.478
#>  [784,] -0.270 0.614
#>  [785,] -0.110 0.526
#>  [786,]  0.071 0.575
#>  [787,] -0.079 0.601
#>  [788,] -0.048 0.696
#>  [789,] -0.244 0.520
#>  [790,] -0.299 0.437
#>  [791,] -0.227 0.660
#>  [792,] -0.187 0.719
#>  [793,] -0.235 0.743
#>  [794,] -0.282 0.714
#>  [795,] -0.134 0.800
#>  [796,] -0.236 0.614
#>  [797,] -0.269 0.612
#>  [798,] -0.052 0.843
#>  [799,] -0.175 0.672
#>  [800,] -0.002 0.511
#>  [801,] -0.135 0.645
#>  [802,] -0.141 0.579
#>  [803,] -0.163 0.737
#>  [804,] -0.229 0.806
#>  [805,] -0.414 0.512
#>  [806,]  0.003 0.329
#>  [807,] -0.061 0.536
#>  [808,] -0.272 0.544
#>  [809,] -0.271 0.349
#>  [810,] -0.162 0.602
#>  [811,] -0.140 0.630
#>  [812,]  0.047 0.556
#>  [813,] -0.235 0.602
#>  [814,] -0.201 0.701
#>  [815,] -0.117 0.522
#>  [816,] -0.109 0.709
#>  [817,] -0.181 1.015
#>  [818,] -0.053 0.659
#>  [819,] -0.023 0.639
#>  [820,] -0.224 0.525
#>  [821,] -0.212 0.489
#>  [822,] -0.187 0.701
#>  [823,] -0.104 0.677
#>  [824,] -0.210 0.505
#>  [825,] -0.233 0.498
#>  [826,] -0.278 0.724
#>  [827,] -0.103 0.763
#>  [828,] -0.143 0.599
#>  [829,] -0.043 0.639
#>  [830,] -0.334 0.616
#>  [831,] -0.154 0.573
#>  [832,] -0.322 0.664
#>  [833,] -0.150 0.644
#>  [834,] -0.065 0.605
#>  [835,] -0.213 0.515
#>  [836,] -0.218 0.802
#>  [837,] -0.207 0.640
#>  [838,] -0.173 0.854
#>  [839,] -0.234 0.667
#>  [840,] -0.167 0.764
#>  [841,] -0.279 0.522
#>  [842,] -0.083 0.631
#>  [843,] -0.242 0.559
#>  [844,] -0.057 0.576
#>  [845,] -0.359 0.640
#>  [846,] -0.291 0.965
#>  [847,] -0.417 0.773
#>  [848,] -0.147 0.900
#>  [849,]  0.181 0.655
#>  [850,] -0.089 0.612
#>  [851,] -0.017 0.429
#>  [852,] -0.255 0.750
#>  [853,] -0.067 0.593
#>  [854,]  0.072 0.555
#>  [855,] -0.246 0.587
#>  [856,] -0.386 0.502
#>  [857,] -0.148 0.778
#>  [858,] -0.122 0.702
#>  [859,] -0.227 0.685
#>  [860,] -0.213 0.647
#>  [861,] -0.215 0.499
#>  [862,] -0.170 0.745
#>  [863,]  0.067 0.472
#>  [864,] -0.226 0.505
#>  [865,] -0.169 0.636
#>  [866,] -0.053 0.636
#>  [867,] -0.015 0.493
#>  [868,] -0.178 0.677
#>  [869,] -0.069 0.417
#>  [870,] -0.209 0.722
#>  [871,] -0.271 0.684
#>  [872,] -0.192 0.655
#>  [873,] -0.417 0.718
#>  [874,] -0.167 0.685
#>  [875,] -0.201 0.707
#>  [876,] -0.063 0.888
#>  [877,] -0.155 0.540
#>  [878,] -0.278 0.700
#>  [879,] -0.154 0.479
#>  [880,] -0.204 0.653
#>  [881,]  0.083 0.633
#>  [882,] -0.103 0.562
#>  [883,] -0.105 0.641
#>  [884,] -0.067 0.476
#>  [885,]  0.053 0.738
#>  [886,] -0.300 0.558
#>  [887,] -0.125 0.584
#>  [888,] -0.133 0.627
#>  [889,] -0.292 0.751
#>  [890,] -0.158 0.655
#>  [891,] -0.365 0.472
#>  [892,] -0.202 0.638
#>  [893,] -0.227 0.649
#>  [894,] -0.127 0.492
#>  [895,] -0.037 0.548
#>  [896,] -0.140 0.527
#>  [897,] -0.201 0.631
#>  [898,] -0.082 0.720
#>  [899,] -0.154 0.920
#>  [900,] -0.233 0.620
#>  [901,] -0.048 0.650
#>  [902,]  0.014 0.475
#>  [903,] -0.194 0.583
#>  [904,] -0.264 0.773
#>  [905,] -0.055 0.588
#>  [906,] -0.037 0.946
#>  [907,] -0.145 0.877
#>  [908,] -0.169 0.703
#>  [909,] -0.398 0.531
#>  [910,] -0.229 0.519
#>  [911,] -0.149 0.709
#>  [912,] -0.263 0.541
#>  [913,] -0.010 0.634
#>  [914,] -0.107 0.668
#>  [915,] -0.073 0.802
#>  [916,] -0.183 0.549
#>  [917,] -0.198 0.687
#>  [918,] -0.020 0.612
#>  [919,] -0.344 0.733
#>  [920,] -0.258 0.726
#>  [921,] -0.178 0.588
#>  [922,] -0.194 0.644
#>  [923,] -0.197 0.552
#>  [924,] -0.157 0.702
#>  [925,] -0.186 0.456
#>  [926,] -0.301 0.793
#>  [927,] -0.293 0.793
#>  [928,] -0.071 0.605
#>  [929,] -0.177 0.758
#>  [930,] -0.276 0.803
#>  [931,]  0.004 0.607
#>  [932,]  0.128 0.737
#>  [933,]  0.031 0.695
#>  [934,] -0.243 0.647
#>  [935,] -0.057 0.735
#>  [936,] -0.100 0.849
#>  [937,] -0.258 0.595
#>  [938,] -0.337 0.575
#>  [939,] -0.218 0.669
#>  [940,] -0.403 0.624
#>  [941,] -0.240 0.629
#>  [942,] -0.009 0.583
#>  [943,] -0.052 0.702
#>  [944,] -0.081 0.389
#>  [945,] -0.236 0.481
#>  [946,] -0.328 0.654
#>  [947,] -0.268 0.739
#>  [948,] -0.129 0.845
#>  [949,] -0.258 0.627
#>  [950,] -0.193 0.337
#>  [951,] -0.182 0.339
#>  [952,] -0.401 0.359
#>  [953,] -0.205 0.471
#>  [954,] -0.019 0.565
#>  [955,] -0.096 0.568
#>  [956,] -0.131 0.572
#>  [957,] -0.048 0.633
#>  [958,] -0.164 0.547
#>  [959,] -0.098 0.693
#>  [960,] -0.186 0.586
#>  [961,] -0.040 0.755
#>  [962,] -0.040 0.692
#>  [963,] -0.159 0.574
#>  [964,] -0.317 0.539
#>  [965,] -0.289 0.960
#>  [966,] -0.122 0.721
#>  [967,] -0.089 0.727
#>  [968,] -0.061 0.629
#>  [969,]  0.080 0.506
#>  [970,] -0.222 0.825
#>  [971,] -0.132 0.481
#>  [972,] -0.329 0.740
#>  [973,] -0.448 0.418
#>  [974,] -0.038 0.388
#>  [975,] -0.298 0.189
#>  [976,] -0.166 0.622
#>  [977,] -0.202 0.672
#>  [978,] -0.085 0.651
#>  [979,]  0.041 0.919
#>  [980,] -0.104 0.908
#>  [981,] -0.352 0.661
#>  [982,] -0.047 0.640
#>  [983,] -0.138 0.650
#>  [984,] -0.231 0.788
#>  [985,] -0.079 0.994
#>  [986,] -0.219 0.721
#>  [987,] -0.080 0.540
#>  [988,] -0.116 0.540
#>  [989,] -0.188 0.585
#>  [990,] -0.332 0.661
#>  [991,] -0.102 0.675
#>  [992,] -0.019 0.670
#>  [993,] -0.192 0.431
#>  [994,]  0.039 0.568
#>  [995,] -0.183 0.863
#>  [996,] -0.277 0.797
#>  [997,] -0.007 0.796
#>  [998,] -0.110 0.479
#>  [999,] -0.241 0.551
#> [1000,] -0.114 0.528
```

Gibbs is the right call when the family is conjugate-friendly under the
Polya-Gamma scheme and you want an exact fit that samples the variance
rather than fixing it. For a binomial areal field, auto takes this path
on its own.

### re_cov_nested and agq (Tier 2)

Two further Tier 2 backends serve narrower roles. `re_cov_nested`
integrates a random-effect covariance matrix rather than conditioning on
a fixed standard deviation. A random-slope term such as `(1 + x | g)`
has no scalar `sigma_re` to condition on, so when a slope term is
present the Laplace path redirects to this backend automatically and
integrates the covariance with a CCD design and a PC or LKJ prior. You
reach it not by name but by writing a slope term and fitting at
`mode = "laplace"`.

``` r

set.seed(20260531)
ng <- 60; ni <- 15
g   <- rep(seq_len(ng), each = ni)
xg  <- rnorm(ng * ni)
Sig <- matrix(c(0.9^2,            0.5 * 0.9 * 0.6,
                0.5 * 0.9 * 0.6,  0.6^2), 2)
b   <- matrix(rnorm(ng * 2), ng) %*% chol(Sig)
eta <- -0.2 + 0.7 * xg + b[g, 1] + b[g, 2] * xg
dsl <- data.frame(y = rbinom(ng * ni, 1, plogis(eta)),
                  x = xg, g = factor(g))

fit_rc <- tulpa(y ~ x + (1 + x | g), data = dsl,
                family = "binomial", mode = "laplace")
fit_rc$backend
#> [1] "re_cov_nested"
```

The slope term routed the fit to `re_cov_nested` without being named.
What comes back is the integrated covariance, not a point estimate of
one: a 2x2 matrix carrying the intercept and slope variances on its
diagonal and their covariance off it.

``` r

round(fit_rc$Sigma_mean, 3)
#>       [,1]  [,2]
#> [1,] 1.052 0.339
#> [2,] 0.339 0.627
```

Integrating the covariance through a Gaussian grid over the
hyperparameters means the fit carries a Pareto-k-hat accuracy
diagnostic, the nested-approximation counterpart to the Rhat a sampler
reports.

``` r

round(fit_rc$pareto_k, 2)
#> [1] 0.55
```

Here k-hat sits below the 0.7 threshold, so the Gaussian grid fits the
covariance posterior well and the integrated matrix can be trusted. With
smaller groups or sparser binary data that posterior turns skewed and
k-hat climbs past 0.7 – not a defect but the signal to escalate to the
exact debias, `control = list(re_cov = "gibbs")`, which replaces the
deterministic integration with a Metropolis-within-Gibbs sweep and a
conjugate inverse-Wishart draw for the covariance.

The adaptive Gauss-Hermite quadrature backend, `agq`, integrates a
random-intercept variance by quadrature and is callable through its
fitter
[`agq_fit()`](https://gillescolling.com/tulpa/reference/agq_fit.md) for
the single-grouping case rather than through
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).

### ep (Tier 2)

Expectation Propagation approximates the posterior of a fixed-effect GLM
by a Gaussian whose per-observation sites match the moments of the
tilted distribution, rather than the mode curvature Laplace uses. It is
exact when the likelihood is Gaussian and is typically more accurate
than Laplace on a skewed GLM likelihood. EP fits fixed effects only (no
random-effect, spatial, or latent structure); reach it with
`mode = "ep"` or call
[`tulpa_ep()`](https://gillescolling.com/tulpa/reference/tulpa_ep.md)
directly.

``` r

fit_ep <- tulpa(y ~ x, data = d, family = "binomial", mode = "ep")
coef(fit_ep)
```

The R-callable backends span both exact tiers and the structured tier.
The next section puts three of them on the same model so the trade
between cost and guarantee is visible in numbers.

## Cost versus guarantee

The clearest way to see what a tier buys is to fit one model several
ways and lay the results side by side. Take the binomial
random-intercept model from above and fit it at Laplace, MALA, and
Pathfinder, timing each with
[`system.time()`](https://rdrr.io/r/base/system.time.html).

``` r

t_lap  <- system.time(
  f_lap  <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                  mode = "laplace", sigma_re = 0.6))[["elapsed"]]
t_mala <- system.time(
  f_mala <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                  mode = "mala", sigma_re = 0.6,
                  control = list(n_iter = 450, warmup = 150)))[["elapsed"]]
t_pf   <- system.time(
  f_pf   <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                  mode = "pathfinder", sigma_re = 0.6,
                  control = list(n_draws = 450)))[["elapsed"]]
```

Collect the slope estimate, its standard error, the tier, and the
elapsed time into one table.

``` r

slope_se <- function(f) summary(f)["x", "std.error"]
data.frame(
  backend  = c(f_lap$backend, f_mala$backend, f_pf$backend),
  tier     = c(f_lap$inference_tier, f_mala$inference_tier,
               f_pf$inference_tier),
  slope    = round(c(coef(f_lap)["x"], coef(f_mala)["x"],
                     coef(f_pf)["x"]), 3),
  slope_se = round(c(slope_se(f_lap), slope_se(f_mala),
                     slope_se(f_pf)), 3),
  seconds  = round(c(t_lap, t_mala, t_pf), 3)
)
#>      backend tier slope slope_se seconds
#> 1    laplace    2 0.802    0.128    0.00
#> 2       mala    1 0.823    0.138    0.07
#> 3 pathfinder    2 0.802    0.130    0.05
```

Three patterns sit in this table. The slope estimates agree across all
three backends to within their standard errors. That is expected: the
posterior mean is the easy quantity, and every method finds it. The
standard errors agree too, because for this model the posterior is
near-Gaussian and the Tier 2 approximation is accurate. That agreement
is the signal that Laplace was safe here, confirmed independently by the
high IMH acceptance earlier. Timing is where the methods part ways. The
deterministic Laplace fit returns fastest. Pathfinder costs an
optimisation plus a draw step. MALA pays for its full chain of gradient
evaluations, and that cost is the visible price of the exact guarantee.

Read the table as a guarantee you bought, not effort you wasted. The
sampler purchased a promise this model did not need and another model
would. The agreement you see is itself the product of a near-Gaussian
posterior. Push the model toward a skewed or heavy-tailed posterior, by
shrinking the data to a few dozen rows, moving to a sparse binomial with
most outcomes zero, or adding a poorly identified variance component,
and the Tier 2 standard error would start to drift away from the Tier 1
one. The samplers would track the true posterior spread through that
drift, while the Gaussian approximation would keep reporting the
symmetric interval its assumption forces on it. That is the moment the
price of the sampler turns into a reason to pay it.

The marginal likelihood ties the comparison back to model choice.
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
reads [`logLik()`](https://rdrr.io/r/stats/logLik.html) from each fit,
which on the Laplace tier is the approximate log marginal likelihood.

``` r

compare_models(laplace = f_lap, pathfinder = f_pf, criterion = "loglik")
#>        model n_params    logLik
#> 1    laplace        2 -259.8379
#> 2 pathfinder       14 -255.1685
```

The [`glance()`](https://generics.r-lib.org/reference/glance.html)
accessor surfaces the per-fit diagnostics that matter for the sampler
tiers: the number of post-warmup draws, the acceptance rate, and the
divergence count.

``` r

glance(f_mala)[c("n_samples", "mean_accept", "n_divergent")]
#>   n_samples mean_accept n_divergent
#> 1       375        0.72          NA
```

An acceptance rate inside the healthy band and no divergences say the
MALA chain mixed well, which is the prerequisite for trusting its
intervals. Without that check, a Tier 1 fit is only nominally exact: a
stuck chain reports a posterior it never explored.

## Choosing a mode

The tier system gives a default worth following and a small set of rules
for when to depart from it.

- **Start at `mode = "laplace"`.** The fit is deterministic, carries no
  Monte Carlo error, and returns in well under a second on the models in
  this vignette and in a fraction of a second per thousand rows on much
  larger ones. Use it for the first fit and for model comparison by
  marginal likelihood, and keep it as the baseline even when you intend
  to finish with a sampler.

- **Check Laplace before you trust its intervals.** The cheapest check
  is `mode = "imh_laplace"` on the same model: a high acceptance rate,
  above roughly 0.5, says the posterior is close to the Laplace Gaussian
  and the intervals are safe. An acceptance rate below about 0.1 says
  the posterior is far from Gaussian and the Laplace intervals are
  biased. The fit earlier accepted at a high rate, which is a clean
  pass.

- **Move to `mode = "mala"` for exact moments at moderate dimension.**
  When the posterior is visibly non-Gaussian, when imh_laplace
  acceptance is low, or when you simply need intervals you can quote
  without the Gaussian caveat, the gradient sampler is the next step.
  Budget a few thousand iterations with warmup at a third to a half of
  them, and watch the acceptance rate settle near 0.574 and the
  divergence count stay at zero.

- **Use `mode = "pathfinder"` when you want draws cheaply.** It costs
  more than Laplace and far less than MALA, returns a sample you can
  feed to downstream code, flags a poor Gaussian fit through its ELBO,
  and doubles as a warm start for a sampler.

- **Let `mode = "auto"` choose for structured models.** For a latent
  block or a spatial field, auto routes to the designed path: nested
  Laplace for latent and most spatial fields, exact Polya-Gamma Gibbs
  for binomial areal models, and the Laplace path for very large plain
  models past tens of thousands of rows. Read `selection_reason` to
  confirm the branch. On a small plain model with no spatial or latent
  structure auto defaults to the MALA gradient sampler (Tier 1); name
  `mode = "laplace"` instead when you want the fast deterministic fit.

- **Treat `mode = "optimized"` as a deliberate exception.** Tier 3 is
  opt-in because its uncertainty is unreliable and its failures are
  silent. It fits a narrow case: a fast point estimate when you have an
  independent handle on the uncertainty. Keep it off the default path,
  and off any interval you intend to report.

The thread through all of it is that the tier is visible and the choice
is recorded, so you are never reasoning about uncertainty you cannot
account for. Read `$backend` and `$inference_tier` off any fit to see
what ran and what its intervals promise, then read `$selection_reason`
to see why that backend was chosen; when the promise on a fit is not the
one your analysis needs, move up a tier and pay for the stronger
guarantee on purpose.

## See also

- [`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)
  prints the full tier and backend map, with the R-callable and
  C-ABI-only backends tagged.

- The quickstart vignette covers the fitting workflow.

- The [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md)
  vignette runs the same tier ladder on a user-defined latent block,
  from nested Laplace through exact sampling.

- [`?mala`](https://gillescolling.com/tulpa/reference/mala.md),
  [`?pathfinder`](https://gillescolling.com/tulpa/reference/pathfinder.md),
  [`?imh_laplace`](https://gillescolling.com/tulpa/reference/imh_laplace.md)
  for the low-level sampler interfaces that drive a `log_posterior`
  closure directly.
