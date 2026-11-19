# Extract temporal effects from a fitted model

Extract posterior distributions of temporal effects from a fitted tulpa
model with temporal specification.

## Usage

``` r
# S3 method for class 'tulpa_temporal_field_fit'
print(x, ...)

# S3 method for class 'tulpa_temporal_field'
print(x, ...)

temporal(
  object,
  component = "all",
  summary = FALSE,
  probs = c(0.025, 0.5, 0.975),
  ...
)

# S3 method for class 'tulpa_fit'
temporal(
  object,
  component = "all",
  summary = FALSE,
  probs = c(0.025, 0.5, 0.975),
  ...
)

# S3 method for class 'tulpa_temporal_posterior'
print(x, ...)

# S3 method for class 'tulpa_temporal_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_temporal_summary'
print(x, ...)
```

## Arguments

- x:

  A `temporal()` field specification, the fit
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  returns for one, a `tulpa_temporal_posterior` or a
  `tulpa_temporal_summary`.

- ...:

  Ignored

- object:

  A `tulpa_fit` object fitted with `temporal` argument

- component:

  Which component to extract for multi-scale models: `"all"` (default),
  `"trend"`, `"seasonal"`, or `"short_term"`.

- summary:

  Logical; if TRUE, return summary statistics instead of full posterior
  draws.

- probs:

  Quantiles to compute if `summary = TRUE`.

## Value

A `tulpa_temporal_posterior` object

[`summary()`](https://rdrr.io/r/base/summary.html) on a
`tulpa_temporal_posterior` returns a `tulpa_temporal_summary` data frame
with one row per time point (per component for multi-scale fits),
holding the posterior mean, SD and requested quantiles of the temporal
effect.

## Details

`temporal()` is overloaded. Given a fitted model it is the accessor
described here. Given a one-sided formula (or a named `formula =` /
`structure =` argument) it is instead the inline varying-coefficient
field constructor used in a
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) model
formula, the temporal mirror of
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md):
`temporal(formula = ~ 1 + x || time, structure = "rw1")` declares a
smooth temporal level (the intercept column) plus a temporally varying
slope on each covariate column. `structure` is one of `"rw1"` (default),
`"rw2"`, or `"ar1"`; only the double bar `||` (independent fields) is
supported.

On a sampler fit the posterior is the field's own draws. On a
nested-Laplace fit it is the Gaussian mixture the outer grid defines:
each hyperparameter cell contributes its conditional mode and marginal
variance, weighted by the cell's posterior weight, and
[`summary()`](https://rdrr.io/r/base/summary.html) reports that
mixture's exact mean, SD and quantiles. The `draws` of such a fit are
sampled from the same mixture, each from its cell's joint Gaussian, so
they carry the correlation between time points as well as each point's
marginal, and a quantity derived from them (a trend, a difference
between two times) has the right spread. A field so large that its
per-cell covariance was not retained (more than 4 million values over
the grid) is drawn with independent time points within each cell
instead. The `time` column of the summary holds the time values the
field was fitted on.

## See also

[`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md)

## Examples

``` r
# \donttest{
set.seed(131)
df <- data.frame(year = 1:40, x = rnorm(40))
df$count <- rpois(40, exp(1 + 0.2 * df$x))

fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  temporal = temporal_multiscale("year", trend = "rw2", seasonal = 12),
  mode = "exact",
  control = list(n_iter = 200L, n_warmup = 100L, seed = 1L)
)
#> Warning: tulpa(): the 'hmc' chain has not converged -- log_sigma2_trend: Rhat 1.19, bulk-ESS 15; log_sigma2_seasonal: Rhat 1.15, bulk-ESS 23; log_sigma2_short: Rhat 1.10, bulk-ESS 33; logit_rho_short: Rhat 1.05, bulk-ESS 84. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup); see check_diagnostics().

# Extract all temporal effects
temp_post <- temporal(fit)
summary(temp_post)
#> Temporal effect posterior summary, 400 posterior draws
#>                component time_idx time          mean         sd        q2.5
#> trend.1            trend        1    1  0.0159627760 0.27434818 -0.46178889
#> trend.2            trend        2    2  0.0188820609 0.23704484 -0.39491668
#> trend.3            trend        3    3  0.0180375136 0.20721704 -0.34455276
#> trend.4            trend        4    4  0.0162521912 0.18661155 -0.32034889
#> trend.5            trend        5    5  0.0164114715 0.17192060 -0.30966868
#> trend.6            trend        6    6  0.0165140026 0.16165645 -0.29621190
#> trend.7            trend        7    7  0.0144989355 0.15304249 -0.28862336
#> trend.8            trend        8    8  0.0130383777 0.14715506 -0.26988903
#> trend.9            trend        9    9  0.0107801790 0.13776461 -0.26818156
#> trend.10           trend       10   10  0.0107193913 0.12911779 -0.23512997
#> trend.11           trend       11   11  0.0080924118 0.11952996 -0.24577998
#> trend.12           trend       12   12  0.0034418652 0.11540987 -0.24226654
#> trend.13           trend       13   13 -0.0003072715 0.11571194 -0.24997106
#> trend.14           trend       14   14 -0.0010364416 0.11607365 -0.25804037
#> trend.15           trend       15   15 -0.0025447902 0.11931093 -0.28560641
#> trend.16           trend       16   16  0.0002533628 0.12166457 -0.28169292
#> trend.17           trend       17   17  0.0013253525 0.11968402 -0.25434951
#> trend.18           trend       18   18  0.0038059670 0.12017151 -0.23755186
#> trend.19           trend       19   19  0.0058393995 0.12265290 -0.23823442
#> trend.20           trend       20   20  0.0004153661 0.12263424 -0.22956585
#> trend.21           trend       21   21 -0.0071804029 0.12419538 -0.26702946
#> trend.22           trend       22   22 -0.0138160435 0.12715292 -0.29675469
#> trend.23           trend       23   23 -0.0205714509 0.13081925 -0.32194248
#> trend.24           trend       24   24 -0.0249018040 0.13098574 -0.30445746
#> trend.25           trend       25   25 -0.0245576903 0.13008508 -0.33446663
#> trend.26           trend       26   26 -0.0212941402 0.12545876 -0.32356586
#> trend.27           trend       27   27 -0.0184705276 0.12158297 -0.28884863
#> trend.28           trend       28   28 -0.0159771900 0.11895823 -0.28773151
#> trend.29           trend       29   29 -0.0135557074 0.11563978 -0.27660888
#> trend.30           trend       30   30 -0.0084860783 0.11549956 -0.25404382
#> trend.31           trend       31   31 -0.0072977617 0.12068147 -0.26805935
#> trend.32           trend       32   32 -0.0064005001 0.13012392 -0.26462981
#> trend.33           trend       33   33 -0.0013718636 0.13944231 -0.27175492
#> trend.34           trend       34   34  0.0024266599 0.15104627 -0.30538312
#> trend.35           trend       35   35  0.0038960603 0.16306098 -0.34283561
#> trend.36           trend       36   36  0.0038993590 0.17814175 -0.39632588
#> trend.37           trend       37   37  0.0033390334 0.19470714 -0.42656785
#> trend.38           trend       38   38  0.0010377126 0.21448635 -0.44812036
#> trend.39           trend       39   39  0.0001553765 0.23945545 -0.50863878
#> trend.40           trend       40   40 -0.0012551621 0.27224498 -0.58785289
#> seasonal.1      seasonal        1    1 -0.0611969195 0.11847859 -0.38817898
#> seasonal.2      seasonal        2    2 -0.0216286772 0.09873722 -0.23314404
#> seasonal.3      seasonal        3    3 -0.0113649823 0.09085242 -0.25003238
#> seasonal.4      seasonal        4    4 -0.0105990072 0.08906436 -0.19154698
#> seasonal.5      seasonal        5    5  0.0028172102 0.08509909 -0.17126719
#> seasonal.6      seasonal        6    6  0.0282167736 0.09948034 -0.13374704
#> seasonal.7      seasonal        7    7  0.0468105724 0.11107339 -0.12633733
#> seasonal.8      seasonal        8    8  0.0443829365 0.10443201 -0.09754301
#> seasonal.9      seasonal        9    9  0.0394698079 0.09940724 -0.10655310
#> seasonal.10     seasonal       10   10  0.0119841950 0.08781699 -0.16659175
#> seasonal.11     seasonal       11   11 -0.0240423649 0.08877533 -0.23908191
#> seasonal.12     seasonal       12   12 -0.0448495444 0.09899472 -0.30437832
#> short_term.1  short_term        1    1 -0.0158349558 0.12758346 -0.32144464
#> short_term.2  short_term        2    2  0.0099651948 0.11463826 -0.23732062
#> short_term.3  short_term        3    3 -0.0096536394 0.13692100 -0.29688955
#> short_term.4  short_term        4    4 -0.0245928144 0.13332509 -0.34667378
#> short_term.5  short_term        5    5 -0.0241361581 0.13883121 -0.42738624
#> short_term.6  short_term        6    6 -0.0009920726 0.11981862 -0.26124801
#> short_term.7  short_term        7    7  0.0151538674 0.12271578 -0.24621060
#> short_term.8  short_term        8    8 -0.0383878971 0.12497787 -0.35794161
#> short_term.9  short_term        9    9 -0.0234816152 0.12696620 -0.37085904
#> short_term.10 short_term       10   10  0.0149500203 0.11527204 -0.19291069
#> short_term.11 short_term       11   11  0.0065690957 0.13000791 -0.27763248
#> short_term.12 short_term       12   12  0.0062892604 0.11940855 -0.24395409
#> short_term.13 short_term       13   13 -0.0388392664 0.13860982 -0.43819475
#> short_term.14 short_term       14   14 -0.0166117340 0.12632082 -0.27567644
#> short_term.15 short_term       15   15 -0.0009049161 0.14709227 -0.35390484
#> short_term.16 short_term       16   16 -0.0466806191 0.13831608 -0.42237294
#> short_term.17 short_term       17   17 -0.0087816188 0.11703700 -0.28549654
#> short_term.18 short_term       18   18 -0.0135853806 0.12459428 -0.25215754
#> short_term.19 short_term       19   19 -0.0054859671 0.12650498 -0.30858557
#> short_term.20 short_term       20   20  0.0069260674 0.10709303 -0.19151473
#> short_term.21 short_term       21   21  0.0075976477 0.12135698 -0.25783097
#> short_term.22 short_term       22   22 -0.0093613582 0.12745539 -0.31872412
#> short_term.23 short_term       23   23 -0.0459483933 0.14267828 -0.45332974
#> short_term.24 short_term       24   24 -0.0255321936 0.14516205 -0.34844240
#> short_term.25 short_term       25   25 -0.0347436278 0.14951677 -0.49170746
#> short_term.26 short_term       26   26 -0.0254657931 0.16087109 -0.40318426
#> short_term.27 short_term       27   27  0.0179332484 0.15281020 -0.31699282
#> short_term.28 short_term       28   28 -0.0365254246 0.11676433 -0.29397900
#> short_term.29 short_term       29   29 -0.0092763729 0.15470062 -0.39473009
#> short_term.30 short_term       30   30 -0.0230142808 0.16561918 -0.45221178
#> short_term.31 short_term       31   31  0.0154993576 0.15059707 -0.25949721
#> short_term.32 short_term       32   32 -0.0140047370 0.13585177 -0.36990588
#> short_term.33 short_term       33   33 -0.0132839272 0.15055318 -0.33174061
#> short_term.34 short_term       34   34 -0.0108582699 0.13301021 -0.31261417
#> short_term.35 short_term       35   35  0.0171175638 0.12717882 -0.24890686
#> short_term.36 short_term       36   36 -0.0258209971 0.12145489 -0.31461056
#> short_term.37 short_term       37   37 -0.0087173958 0.13117757 -0.33444281
#> short_term.38 short_term       38   38  0.0013333209 0.16071018 -0.32930769
#> short_term.39 short_term       39   39 -0.0497830051 0.15533828 -0.40018140
#> short_term.40 short_term       40   40 -0.0180737855 0.16230605 -0.40398836
#>                         q50     q97.5
#> trend.1       -3.957051e-03 0.6820389
#> trend.2       -1.763548e-04 0.5732550
#> trend.3        1.122698e-02 0.4958335
#> trend.4        5.141121e-03 0.4582602
#> trend.5        1.538135e-02 0.4083654
#> trend.6        1.960149e-02 0.3416152
#> trend.7        1.018695e-02 0.3037412
#> trend.8        1.096395e-02 0.3000767
#> trend.9        1.226905e-02 0.2509539
#> trend.10       3.872798e-03 0.2534990
#> trend.11       1.054651e-02 0.2450926
#> trend.12       7.513038e-03 0.2252976
#> trend.13       7.338180e-03 0.2235518
#> trend.14       6.800436e-03 0.2159333
#> trend.15       1.521816e-03 0.2292656
#> trend.16      -2.185867e-03 0.2593656
#> trend.17      -4.580165e-03 0.2544960
#> trend.18      -4.231844e-03 0.2840449
#> trend.19      -2.121394e-03 0.2903559
#> trend.20      -2.785453e-03 0.2990071
#> trend.21      -4.034405e-04 0.3011592
#> trend.22      -5.075196e-03 0.2576425
#> trend.23      -1.163250e-02 0.2509019
#> trend.24      -1.028331e-02 0.2287782
#> trend.25      -8.774151e-03 0.2232186
#> trend.26      -9.505627e-03 0.2223657
#> trend.27      -5.136497e-03 0.2071523
#> trend.28      -5.612116e-03 0.1832719
#> trend.29      -5.791439e-03 0.1958728
#> trend.30      -6.183820e-03 0.2161065
#> trend.31      -4.232114e-03 0.2345217
#> trend.32       1.622893e-03 0.2447107
#> trend.33       8.194170e-03 0.2547443
#> trend.34       1.571078e-02 0.2675390
#> trend.35       1.612485e-02 0.3031091
#> trend.36       1.730328e-02 0.3241227
#> trend.37       1.189541e-02 0.3517547
#> trend.38       3.448017e-03 0.3905506
#> trend.39       3.639969e-03 0.4382208
#> trend.40       4.981223e-03 0.4809987
#> seasonal.1    -2.682112e-02 0.1102049
#> seasonal.2    -5.196575e-03 0.1752460
#> seasonal.3    -2.300376e-03 0.1678918
#> seasonal.4    -1.142914e-03 0.1579625
#> seasonal.5     2.969582e-03 0.2134862
#> seasonal.6     4.760443e-03 0.2896443
#> seasonal.7     2.054054e-02 0.3339483
#> seasonal.8     1.625966e-02 0.3194735
#> seasonal.9     1.315336e-02 0.3121303
#> seasonal.10    3.260550e-03 0.1933484
#> seasonal.11   -9.886300e-03 0.1328828
#> seasonal.12   -2.003594e-02 0.1038288
#> short_term.1  -3.181654e-03 0.2649560
#> short_term.2   4.336507e-03 0.3047280
#> short_term.3   4.618802e-03 0.2353785
#> short_term.4  -6.555862e-03 0.2134502
#> short_term.5  -2.105683e-04 0.2266274
#> short_term.6   6.239612e-05 0.2765914
#> short_term.7   7.801373e-03 0.3201465
#> short_term.8  -1.424871e-02 0.1729009
#> short_term.9  -7.158863e-03 0.1954952
#> short_term.10  2.969459e-03 0.2802415
#> short_term.11  7.026136e-05 0.3160165
#> short_term.12  1.235750e-03 0.2932518
#> short_term.13 -1.305733e-02 0.1657370
#> short_term.14 -5.330779e-03 0.2017704
#> short_term.15 -2.528205e-04 0.2484819
#> short_term.16 -1.547129e-02 0.1848427
#> short_term.17  2.767135e-04 0.2478064
#> short_term.18 -5.524648e-03 0.2256475
#> short_term.19 -1.090744e-03 0.2731455
#> short_term.20  1.156398e-03 0.2258226
#> short_term.21 -2.784386e-03 0.2689549
#> short_term.22 -4.376172e-03 0.2632767
#> short_term.23 -1.466670e-02 0.1873554
#> short_term.24 -5.671364e-03 0.2669409
#> short_term.25 -5.209901e-03 0.2049989
#> short_term.26 -6.016924e-03 0.2695065
#> short_term.27  6.326399e-03 0.3134377
#> short_term.28 -1.612428e-02 0.1355680
#> short_term.29  3.390484e-03 0.3034371
#> short_term.30 -1.996986e-03 0.2806912
#> short_term.31  5.291042e-03 0.3264470
#> short_term.32 -3.940729e-03 0.2694177
#> short_term.33 -4.398840e-03 0.3126359
#> short_term.34  1.535526e-03 0.2513994
#> short_term.35  6.887100e-03 0.2973612
#> short_term.36 -5.606628e-03 0.1974346
#> short_term.37 -1.801880e-03 0.2261580
#> short_term.38  3.319558e-03 0.3196978
#> short_term.39 -1.291469e-02 0.1767439
#> short_term.40 -4.828937e-03 0.2805653
# }
```
