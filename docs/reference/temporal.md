# Extract temporal effects from a fitted model

Extract posterior distributions of temporal effects from a fitted tulpa
model with temporal specification.

## Usage

``` r
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
```

## Arguments

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

- ...:

  Ignored

## Value

A `tulpa_temporal_posterior` object

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

# Extract all temporal effects
temp_post <- temporal(fit)
summary(temp_post)
#>                component time_idx time          mean        sd       q2.5
#> trend.1            trend        1    1  0.0946761266 0.4185446 -0.7478835
#> trend.2            trend        2    2  0.0833848927 0.3443812 -0.6502260
#> trend.3            trend        3    3  0.0768957414 0.2953237 -0.5497224
#> trend.4            trend        4    4  0.0777032890 0.2660524 -0.5615176
#> trend.5            trend        5    5  0.0764587110 0.2497619 -0.5595061
#> trend.6            trend        6    6  0.0755161844 0.2442556 -0.5283461
#> trend.7            trend        7    7  0.0738574598 0.2382864 -0.5689790
#> trend.8            trend        8    8  0.0696209379 0.2353344 -0.4943685
#> trend.9            trend        9    9  0.0606974870 0.2317397 -0.4608812
#> trend.10           trend       10   10  0.0474997275 0.2248787 -0.4367869
#> trend.11           trend       11   11  0.0291166926 0.2104319 -0.4207077
#> trend.12           trend       12   12  0.0020308484 0.1966243 -0.4489856
#> trend.13           trend       13   13 -0.0234564025 0.1881935 -0.4637084
#> trend.14           trend       14   14 -0.0333807570 0.1748805 -0.4260843
#> trend.15           trend       15   15 -0.0336798997 0.1707797 -0.4048317
#> trend.16           trend       16   16 -0.0281190750 0.1663724 -0.3568849
#> trend.17           trend       17   17 -0.0129358918 0.1662278 -0.3367689
#> trend.18           trend       18   18  0.0066561742 0.1761189 -0.3261972
#> trend.19           trend       19   19  0.0212149303 0.1893618 -0.3055977
#> trend.20           trend       20   20  0.0235837634 0.2014289 -0.2930683
#> trend.21           trend       21   21  0.0110394105 0.2058173 -0.3084689
#> trend.22           trend       22   22 -0.0163780519 0.2088789 -0.3722369
#> trend.23           trend       23   23 -0.0440497324 0.2204670 -0.4586116
#> trend.24           trend       24   24 -0.0613816949 0.2327388 -0.5101272
#> trend.25           trend       25   25 -0.0601693680 0.2408726 -0.4801051
#> trend.26           trend       26   26 -0.0407406010 0.2514404 -0.4474766
#> trend.27           trend       27   27 -0.0203597107 0.2581463 -0.3769447
#> trend.28           trend       28   28 -0.0125838897 0.2437141 -0.3631957
#> trend.29           trend       29   29 -0.0018908394 0.2340041 -0.3775607
#> trend.30           trend       30   30  0.0046887133 0.2220578 -0.3607330
#> trend.31           trend       31   31  0.0140273943 0.2151083 -0.3905934
#> trend.32           trend       32   32  0.0094926345 0.2055272 -0.4169802
#> trend.33           trend       33   33 -0.0049340209 0.1978934 -0.3966990
#> trend.34           trend       34   34 -0.0150948643 0.1955109 -0.4421149
#> trend.35           trend       35   35 -0.0266186212 0.1940773 -0.4230217
#> trend.36           trend       36   36 -0.0421183514 0.2022903 -0.4061837
#> trend.37           trend       37   37 -0.0617763217 0.2224021 -0.4935155
#> trend.38           trend       38   38 -0.0822113187 0.2681495 -0.5964643
#> trend.39           trend       39   39 -0.1032543970 0.3280922 -0.7928897
#> trend.40           trend       40   40 -0.1228626659 0.4136985 -1.0119876
#> seasonal.1      seasonal        1    1 -0.0893245760 0.1510766 -0.4259015
#> seasonal.2      seasonal        2    2 -0.0440444755 0.1214808 -0.2961277
#> seasonal.3      seasonal        3    3  0.0052276650 0.1140618 -0.2408689
#> seasonal.4      seasonal        4    4 -0.0131477009 0.1293311 -0.2925262
#> seasonal.5      seasonal        5    5  0.0005245145 0.1408271 -0.2745626
#> seasonal.6      seasonal        6    6  0.0229911174 0.1294869 -0.2212460
#> seasonal.7      seasonal        7    7  0.0361077172 0.1273813 -0.2245996
#> seasonal.8      seasonal        8    8  0.0452872177 0.1282364 -0.1561524
#> seasonal.9      seasonal        9    9  0.0514403986 0.1458285 -0.2019278
#> seasonal.10     seasonal       10   10  0.0294123181 0.1361817 -0.1933249
#> seasonal.11     seasonal       11   11 -0.0185967114 0.1385284 -0.3865186
#> seasonal.12     seasonal       12   12 -0.0493277378 0.1460900 -0.3955278
#> short_term.1  short_term        1    1 -0.0213109950 0.2079431 -0.4312568
#> short_term.2  short_term        2    2  0.0144109833 0.1800889 -0.3255675
#> short_term.3  short_term        3    3 -0.0131902609 0.1735466 -0.3773201
#> short_term.4  short_term        4    4 -0.0496655147 0.1946411 -0.5213481
#> short_term.5  short_term        5    5 -0.0342458591 0.2033329 -0.5657365
#> short_term.6  short_term        6    6  0.0662904432 0.1723927 -0.2227347
#> short_term.7  short_term        7    7  0.0769322374 0.2381257 -0.2387308
#> short_term.8  short_term        8    8 -0.0050546832 0.2275231 -0.4988237
#> short_term.9  short_term        9    9 -0.0068467818 0.2215155 -0.4559977
#> short_term.10 short_term       10   10 -0.0047221922 0.1818502 -0.4661501
#> short_term.11 short_term       11   11 -0.0091412778 0.1848720 -0.4303342
#> short_term.12 short_term       12   12  0.0124416040 0.2108773 -0.4726851
#> short_term.13 short_term       13   13 -0.0668062364 0.2496343 -0.6730752
#> short_term.14 short_term       14   14 -0.0858940622 0.2203126 -0.5220320
#> short_term.15 short_term       15   15 -0.0092901814 0.2436398 -0.5534208
#> short_term.16 short_term       16   16 -0.0947680011 0.2600132 -0.5739706
#> short_term.17 short_term       17   17  0.0085331725 0.2162953 -0.5540846
#> short_term.18 short_term       18   18  0.0206670247 0.2152174 -0.3681850
#> short_term.19 short_term       19   19 -0.0586893896 0.2075239 -0.5913858
#> short_term.20 short_term       20   20 -0.0175294098 0.2108082 -0.4834553
#> short_term.21 short_term       21   21  0.0329767863 0.2146170 -0.5250343
#> short_term.22 short_term       22   22 -0.0829206841 0.2835281 -0.9893086
#> short_term.23 short_term       23   23 -0.1831280890 0.3186058 -1.1759699
#> short_term.24 short_term       24   24 -0.1350556440 0.3501554 -1.3981581
#> short_term.25 short_term       25   25 -0.1313822436 0.3157924 -1.2118191
#> short_term.26 short_term       26   26 -0.0125680594 0.3064813 -1.0237132
#> short_term.27 short_term       27   27 -0.0153188982 0.2693669 -0.7243277
#> short_term.28 short_term       28   28 -0.0462464075 0.2707559 -0.9108535
#> short_term.29 short_term       29   29 -0.0083995876 0.2905118 -1.1270862
#> short_term.30 short_term       30   30 -0.0895002688 0.2923935 -1.0163276
#> short_term.31 short_term       31   31  0.0137289137 0.2513641 -0.7480548
#> short_term.32 short_term       32   32 -0.0385493131 0.2340114 -0.6842426
#> short_term.33 short_term       33   33 -0.0704012876 0.2579202 -0.7677413
#> short_term.34 short_term       34   34 -0.0206397960 0.2109022 -0.4188016
#> short_term.35 short_term       35   35  0.0063322826 0.1963994 -0.4801202
#> short_term.36 short_term       36   36 -0.0688443481 0.2234915 -0.6820227
#> short_term.37 short_term       37   37  0.0591974317 0.2627155 -0.4787139
#> short_term.38 short_term       38   38  0.0508673902 0.1744546 -0.2463924
#> short_term.39 short_term       39   39 -0.0342897090 0.2459611 -0.5817505
#> short_term.40 short_term       40   40  0.0401058927 0.2672929 -0.6442449
#>                         q50     q97.5
#> trend.1        0.1484922943 0.9350614
#> trend.2        0.1063653550 0.7865376
#> trend.3        0.0821104359 0.6212228
#> trend.4        0.1285426716 0.5617042
#> trend.5        0.1039685109 0.5121566
#> trend.6        0.0835263392 0.4771938
#> trend.7        0.0908855937 0.4738093
#> trend.8        0.0986968517 0.5154310
#> trend.9        0.0898350202 0.4577585
#> trend.10       0.0785930741 0.5040003
#> trend.11       0.0335469736 0.4860013
#> trend.12       0.0210461011 0.4273153
#> trend.13      -0.0069841377 0.4013438
#> trend.14       0.0033335106 0.2860818
#> trend.15      -0.0041867241 0.2865327
#> trend.16      -0.0017054459 0.3477318
#> trend.17      -0.0154438423 0.3939528
#> trend.18      -0.0149180450 0.4333311
#> trend.19      -0.0021708488 0.4720202
#> trend.20      -0.0007641698 0.4927101
#> trend.21      -0.0217676106 0.4411649
#> trend.22      -0.0411656079 0.4664737
#> trend.23      -0.0531667667 0.5351881
#> trend.24      -0.0579690815 0.5371778
#> trend.25      -0.0653014800 0.5606905
#> trend.26      -0.0732230763 0.5643914
#> trend.27      -0.0732430321 0.7220375
#> trend.28      -0.0662980119 0.6109542
#> trend.29      -0.0559725702 0.6727314
#> trend.30      -0.0315125572 0.7101787
#> trend.31      -0.0237282768 0.6258513
#> trend.32       0.0045233244 0.5310458
#> trend.33      -0.0067493573 0.4407062
#> trend.34      -0.0286575386 0.4039739
#> trend.35      -0.0570085912 0.3798879
#> trend.36      -0.0780877014 0.3242751
#> trend.37      -0.0843283978 0.3601565
#> trend.38      -0.1017583738 0.4050909
#> trend.39      -0.1058215814 0.4954348
#> trend.40      -0.1143974206 0.6070705
#> seasonal.1    -0.0218329804 0.1161879
#> seasonal.2    -0.0131968166 0.1921024
#> seasonal.3     0.0034435923 0.2405824
#> seasonal.4    -0.0086094313 0.3125501
#> seasonal.5    -0.0097733708 0.3963874
#> seasonal.6     0.0004656844 0.3065532
#> seasonal.7     0.0109256854 0.3304266
#> seasonal.8     0.0030827931 0.3636140
#> seasonal.9     0.0038518810 0.3972253
#> seasonal.10   -0.0003965129 0.3598855
#> seasonal.11    0.0001178167 0.3393793
#> seasonal.12   -0.0026796692 0.2416730
#> short_term.1  -0.0166454478 0.5088286
#> short_term.2   0.0131636481 0.3521444
#> short_term.3   0.0023298856 0.2911078
#> short_term.4  -0.0122672852 0.3457093
#> short_term.5   0.0066723230 0.3484269
#> short_term.6   0.0117406675 0.3971435
#> short_term.7   0.0102789593 0.5903279
#> short_term.8   0.0051305430 0.3345247
#> short_term.9  -0.0205776666 0.6058927
#> short_term.10 -0.0251265688 0.3544637
#> short_term.11 -0.0096328502 0.3103738
#> short_term.12 -0.0032343898 0.4248073
#> short_term.13 -0.0292912690 0.3856361
#> short_term.14 -0.0423127258 0.2553768
#> short_term.15 -0.0093814623 0.4288527
#> short_term.16 -0.0444007006 0.4134013
#> short_term.17  0.0155362283 0.3961754
#> short_term.18  0.0300812686 0.4905250
#> short_term.19 -0.0225385481 0.3267642
#> short_term.20 -0.0021054295 0.4168118
#> short_term.21  0.0242451330 0.4853611
#> short_term.22 -0.0493648913 0.3826325
#> short_term.23 -0.0970204545 0.1874598
#> short_term.24 -0.0528145052 0.3256541
#> short_term.25 -0.0306220514 0.1825838
#> short_term.26  0.0234424618 0.3362394
#> short_term.27 -0.0049898366 0.4002567
#> short_term.28 -0.0397915330 0.4289509
#> short_term.29  0.0086425009 0.3266677
#> short_term.30 -0.0703135755 0.3723799
#> short_term.31  0.0331855791 0.4050513
#> short_term.32 -0.0317821997 0.4011833
#> short_term.33 -0.0355144769 0.3466623
#> short_term.34 -0.0176839691 0.4699990
#> short_term.35  0.0332053034 0.2571143
#> short_term.36 -0.0284703391 0.2669030
#> short_term.37  0.0330401855 0.5472488
#> short_term.38  0.0453388092 0.4643314
#> short_term.39 -0.0264914119 0.4060525
#> short_term.40  0.0055116924 0.6238940
# }
```
