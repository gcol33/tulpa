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
#>                component time_idx time          mean         sd       q2.5
#> trend.1            trend        1    1  0.1315250833 0.36912178 -0.6253769
#> trend.2            trend        2    2  0.1367559885 0.30676160 -0.5034162
#> trend.3            trend        3    3  0.1405048919 0.26232826 -0.3578965
#> trend.4            trend        4    4  0.1401805074 0.24336981 -0.3045232
#> trend.5            trend        5    5  0.1418707252 0.23347180 -0.2893082
#> trend.6            trend        6    6  0.1480515082 0.23216564 -0.2269157
#> trend.7            trend        7    7  0.1496883474 0.22612918 -0.2499520
#> trend.8            trend        8    8  0.1456235848 0.22289475 -0.2507725
#> trend.9            trend        9    9  0.1360479994 0.21330804 -0.2424844
#> trend.10           trend       10   10  0.1124125895 0.20434191 -0.2536351
#> trend.11           trend       11   11  0.0832975218 0.19606087 -0.2492744
#> trend.12           trend       12   12  0.0515403073 0.19507183 -0.3866162
#> trend.13           trend       13   13  0.0258297429 0.19907755 -0.4131636
#> trend.14           trend       14   14  0.0101520268 0.20756972 -0.4217511
#> trend.15           trend       15   15  0.0044792399 0.20950047 -0.4219453
#> trend.16           trend       16   16  0.0024423875 0.20705403 -0.4177316
#> trend.17           trend       17   17  0.0029306889 0.20849929 -0.3982797
#> trend.18           trend       18   18 -0.0009766135 0.20278699 -0.3813533
#> trend.19           trend       19   19 -0.0102332414 0.18784881 -0.3627997
#> trend.20           trend       20   20 -0.0271579726 0.16161480 -0.3241738
#> trend.21           trend       21   21 -0.0541999787 0.14770667 -0.3388493
#> trend.22           trend       22   22 -0.0901969265 0.15594909 -0.4341414
#> trend.23           trend       23   23 -0.1193543156 0.16993189 -0.4998363
#> trend.24           trend       24   24 -0.1359528722 0.18223538 -0.6063157
#> trend.25           trend       25   25 -0.1333669777 0.17878541 -0.5390132
#> trend.26           trend       26   26 -0.1204754897 0.17896620 -0.5102293
#> trend.27           trend       27   27 -0.1033176359 0.18495144 -0.4831815
#> trend.28           trend       28   28 -0.0927707828 0.18871882 -0.4500081
#> trend.29           trend       29   29 -0.0834852167 0.19250026 -0.4133748
#> trend.30           trend       30   30 -0.0745611765 0.19451116 -0.4168467
#> trend.31           trend       31   31 -0.0594525218 0.20399649 -0.4028734
#> trend.32           trend       32   32 -0.0503802280 0.20239859 -0.3957232
#> trend.33           trend       33   33 -0.0437789076 0.20057373 -0.3868288
#> trend.34           trend       34   34 -0.0416238265 0.19702467 -0.3952232
#> trend.35           trend       35   35 -0.0416363560 0.19847654 -0.4282408
#> trend.36           trend       36   36 -0.0434408473 0.21238891 -0.4679533
#> trend.37           trend       37   37 -0.0445223409 0.23164062 -0.5200859
#> trend.38           trend       38   38 -0.0529334206 0.25932659 -0.5769705
#> trend.39           trend       39   39 -0.0605806035 0.30323370 -0.6457186
#> trend.40           trend       40   40 -0.0672816282 0.36836492 -0.7871853
#> seasonal.1      seasonal        1    1 -0.0692684837 0.14181173 -0.3842013
#> seasonal.2      seasonal        2    2 -0.0054242909 0.13467000 -0.2797523
#> seasonal.3      seasonal        3    3 -0.0134204751 0.14144859 -0.3419568
#> seasonal.4      seasonal        4    4 -0.0296986278 0.13738220 -0.4911207
#> seasonal.5      seasonal        5    5 -0.0090010395 0.13954669 -0.4096045
#> seasonal.6      seasonal        6    6  0.0159044163 0.13402794 -0.2833790
#> seasonal.7      seasonal        7    7  0.0355339763 0.13172347 -0.2066154
#> seasonal.8      seasonal        8    8  0.0412370699 0.14800599 -0.2764745
#> seasonal.9      seasonal        9    9  0.0622905356 0.13138128 -0.1536269
#> seasonal.10     seasonal       10   10  0.0401882386 0.12740791 -0.1545659
#> seasonal.11     seasonal       11   11 -0.0193782195 0.12719348 -0.2605118
#> seasonal.12     seasonal       12   12 -0.0485023548 0.15781783 -0.5080513
#> short_term.1  short_term        1    1 -0.0178611239 0.12265664 -0.2766201
#> short_term.2  short_term        2    2 -0.0048306966 0.11533045 -0.2727345
#> short_term.3  short_term        3    3 -0.0018118253 0.11068560 -0.2512452
#> short_term.4  short_term        4    4 -0.0053063652 0.12666033 -0.2573274
#> short_term.5  short_term        5    5 -0.0059150454 0.11855249 -0.2954149
#> short_term.6  short_term        6    6  0.0069929294 0.08461974 -0.1750743
#> short_term.7  short_term        7    7  0.0004047846 0.09081725 -0.1873992
#> short_term.8  short_term        8    8 -0.0189028686 0.10342680 -0.2384530
#> short_term.9  short_term        9    9 -0.0076987695 0.08209326 -0.1936042
#> short_term.10 short_term       10   10  0.0097525338 0.10050762 -0.2308859
#> short_term.11 short_term       11   11  0.0003433959 0.12443233 -0.2561923
#> short_term.12 short_term       12   12  0.0039461508 0.10149179 -0.2056944
#> short_term.13 short_term       13   13 -0.0336763850 0.12324033 -0.3262242
#> short_term.14 short_term       14   14 -0.0183655073 0.13005229 -0.3421551
#> short_term.15 short_term       15   15  0.0019874558 0.10679801 -0.2273036
#> short_term.16 short_term       16   16 -0.0325305331 0.09805160 -0.2748021
#> short_term.17 short_term       17   17  0.0211179182 0.12850227 -0.2284328
#> short_term.18 short_term       18   18  0.0018242815 0.12148225 -0.2506478
#> short_term.19 short_term       19   19 -0.0038680223 0.12180855 -0.3218901
#> short_term.20 short_term       20   20  0.0273140410 0.11932144 -0.1489153
#> short_term.21 short_term       21   21  0.0196258335 0.10391246 -0.1834510
#> short_term.22 short_term       22   22  0.0132605580 0.12363200 -0.2175994
#> short_term.23 short_term       23   23 -0.0210283526 0.12524507 -0.2795702
#> short_term.24 short_term       24   24  0.0042118019 0.10502216 -0.2246769
#> short_term.25 short_term       25   25  0.0068570144 0.09800001 -0.2456611
#> short_term.26 short_term       26   26  0.0281189509 0.10318411 -0.1546435
#> short_term.27 short_term       27   27  0.0365683974 0.12383560 -0.1790908
#> short_term.28 short_term       28   28  0.0113643322 0.10694800 -0.2010556
#> short_term.29 short_term       29   29  0.0152338656 0.11100303 -0.2003467
#> short_term.30 short_term       30   30 -0.0112763957 0.13978751 -0.3569973
#> short_term.31 short_term       31   31  0.0294432405 0.12138390 -0.1371820
#> short_term.32 short_term       32   32 -0.0069525703 0.10483273 -0.2164303
#> short_term.33 short_term       33   33  0.0058608206 0.10429880 -0.2081015
#> short_term.34 short_term       34   34  0.0079754931 0.09943901 -0.1840873
#> short_term.35 short_term       35   35  0.0104082924 0.09071412 -0.1640140
#> short_term.36 short_term       36   36 -0.0141383491 0.10727761 -0.2010742
#> short_term.37 short_term       37   37 -0.0045208139 0.10917214 -0.3006743
#> short_term.38 short_term       38   38  0.0273927256 0.10168007 -0.1621104
#> short_term.39 short_term       39   39  0.0004488582 0.09470251 -0.1540653
#> short_term.40 short_term       40   40  0.0183731354 0.12050576 -0.2036794
#>                         q50     q97.5
#> trend.1        0.1508967735 0.7001982
#> trend.2        0.1595124037 0.6002750
#> trend.3        0.1554359641 0.5689047
#> trend.4        0.1309831504 0.5804315
#> trend.5        0.1150570340 0.5808469
#> trend.6        0.1300276707 0.5705494
#> trend.7        0.1224668096 0.5890060
#> trend.8        0.1231138349 0.5841503
#> trend.9        0.1294237328 0.5304859
#> trend.10       0.0959037105 0.4855918
#> trend.11       0.0586961995 0.4416225
#> trend.12       0.0589034182 0.3803301
#> trend.13       0.0568539894 0.3514518
#> trend.14       0.0391365966 0.3402618
#> trend.15       0.0265895623 0.3609055
#> trend.16       0.0197697729 0.4181711
#> trend.17       0.0043080015 0.4385041
#> trend.18       0.0103176588 0.3632333
#> trend.19       0.0099777182 0.3224554
#> trend.20      -0.0092009310 0.2495218
#> trend.21      -0.0456760267 0.2319309
#> trend.22      -0.0809726702 0.1927367
#> trend.23      -0.1003471371 0.1491730
#> trend.24      -0.1182133481 0.1629943
#> trend.25      -0.1140606797 0.1696075
#> trend.26      -0.1039785778 0.1783066
#> trend.27      -0.0850027517 0.2321738
#> trend.28      -0.0854800260 0.2548827
#> trend.29      -0.0961990075 0.3224378
#> trend.30      -0.0742119894 0.3216213
#> trend.31      -0.0582236821 0.3700438
#> trend.32      -0.0570335916 0.3265971
#> trend.33      -0.0405349290 0.3660819
#> trend.34      -0.0369922907 0.3255255
#> trend.35      -0.0305408465 0.3165731
#> trend.36      -0.0204010356 0.3202590
#> trend.37      -0.0422441303 0.3655846
#> trend.38      -0.0431554771 0.3895360
#> trend.39      -0.0492509416 0.5386227
#> trend.40      -0.0634784941 0.6132115
#> seasonal.1    -0.0499171489 0.1862380
#> seasonal.2    -0.0097649147 0.3345025
#> seasonal.3    -0.0014721583 0.2477063
#> seasonal.4    -0.0022490268 0.1568992
#> seasonal.5     0.0083912058 0.2028665
#> seasonal.6     0.0227194051 0.2606615
#> seasonal.7     0.0185038073 0.3564788
#> seasonal.8     0.0201642475 0.3851814
#> seasonal.9     0.0399021633 0.4097533
#> seasonal.10    0.0171470454 0.3552288
#> seasonal.11   -0.0245111143 0.2997327
#> seasonal.12   -0.0356427192 0.2646616
#> short_term.1  -0.0109589547 0.2450598
#> short_term.2   0.0014788696 0.2314283
#> short_term.3  -0.0065433799 0.2085323
#> short_term.4  -0.0040548127 0.2341165
#> short_term.5  -0.0001699464 0.2288440
#> short_term.6   0.0062283399 0.1648837
#> short_term.7  -0.0023305206 0.2159489
#> short_term.8  -0.0054970837 0.1782616
#> short_term.9  -0.0016444934 0.1379902
#> short_term.10  0.0043882979 0.2774947
#> short_term.11 -0.0023888434 0.3233048
#> short_term.12  0.0041030819 0.2117443
#> short_term.13 -0.0126688045 0.2083748
#> short_term.14 -0.0065907609 0.3200452
#> short_term.15 -0.0010226230 0.2143636
#> short_term.16 -0.0190564571 0.1454632
#> short_term.17  0.0138475659 0.2470589
#> short_term.18  0.0018994721 0.2163323
#> short_term.19  0.0030164505 0.2716315
#> short_term.20  0.0086747314 0.3875990
#> short_term.21  0.0065121910 0.2989246
#> short_term.22  0.0037624371 0.3650234
#> short_term.23 -0.0083115111 0.1671735
#> short_term.24  0.0024997762 0.2421644
#> short_term.25  0.0115150388 0.2149884
#> short_term.26  0.0148674571 0.2845745
#> short_term.27  0.0130872082 0.3898667
#> short_term.28  0.0017208214 0.2403034
#> short_term.29  0.0052713416 0.2951364
#> short_term.30 -0.0024573238 0.2384549
#> short_term.31  0.0149126579 0.3238060
#> short_term.32 -0.0043722622 0.2496742
#> short_term.33  0.0021567956 0.2252785
#> short_term.34 -0.0009832965 0.2555872
#> short_term.35  0.0046054386 0.2249410
#> short_term.36 -0.0089596644 0.2080296
#> short_term.37 -0.0006215643 0.2569414
#> short_term.38  0.0166631429 0.2529769
#> short_term.39 -0.0054464506 0.2444554
#> short_term.40  0.0091148724 0.3809265
# }
```
