# The data a tulpa model expects

``` r

library(tulpa)
```

## Expected structure

Every tulpa fit starts from two things: a formula and a data frame. The
formula names the response, the fixed effects, the grouping factors for
random effects, and any special structure (an offset, a spatial unit).
The data frame holds one row per observation, with one column for each
name the formula mentions.
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) walks
the formula, pulls the matching columns out of the data, and turns them
into the numeric pieces an inference backend consumes.

Two functions do that translation, and both are exported so you can call
them on their own.
[`tulpa_parse_formula()`](https://gillescolling.com/tulpa/reference/tulpa_parse_formula.md)
reads the formula and reports its parts.
[`tulpa_build_model_data()`](https://gillescolling.com/tulpa/reference/tulpa_build_model_data.md)
takes that parsed object plus the data and returns the design. Looking
at their output is the fastest way to see what tulpa thinks your model
is before you commit to a fit.

The formula grammar follows lme4 and glmmTMB, so most mixed-model
formulas you already write carry over. Fixed effects sit on the right of
`~` as an additive chain. A random intercept is `(1 | g)`. A random
intercept and slope sharing a correlation is `(1 + x | g)`; the double
bar `(1 + x || g)` keeps the same two effects but drops the correlation
between them. An `offset(...)` term enters a known per-row shift on the
linear-predictor scale. A `spatial(col)` term names the column that maps
each row to a spatial unit, with the field structure itself arriving
through the `spatial=` argument.

Here is a formula that uses several of those at once. Print the parsed
object to read it back.

``` r

pf <- tulpa_parse_formula(y ~ x + z + (1 | g) + (1 + x || site))
pf
#> tulpa parsed formula
#>   Response: y 
#>   Fixed: y ~ x + z 
#>   Random effects: 2 term(s)
#>     ( 1 | g )
#>     ( 1 + x || site )
```

The print method separates the response from the fixed formula and lists
each random-effect term in its canonical form. The bar operator is shown
as `|` or `||`, so a glance confirms whether a slope carries a
correlation. The fixed formula has had every random, latent, and special
term stripped out, which is exactly the formula used to build the
fixed-effects design matrix.

## Building from a data frame

A tidy data frame is the natural input. One row per observation, columns
typed as you intend them: numeric for continuous predictors, `factor`
for unordered categories, `ordered` for ranked ones. tulpa reads the
type off each column and builds the matching design piece, so a factor
expands into contrasts while a numeric column stays a single slope.
Simulate a small long-format frame with a numeric predictor, a numeric
covariate, and two grouping factors.

``` r

n  <- 200
df <- data.frame(
  y    = rpois(n, 3),
  x    = rnorm(n),
  z    = rnorm(n),
  g    = factor(sample(letters[1:6], n, replace = TRUE)),
  site = factor(sample(paste0("s", 1:4), n, replace = TRUE))
)
str(df)
#> 'data.frame':    200 obs. of  5 variables:
#>  $ y   : int  3 1 3 3 4 2 2 1 4 3 ...
#>  $ x   : num  -0.658 -0.523 0.191 0.401 0.667 ...
#>  $ z   : num  1.443 -1.113 1.123 0.757 -1.058 ...
#>  $ g   : Factor w/ 6 levels "a","b","c","d",..: 1 1 3 4 5 2 4 1 2 6 ...
#>  $ site: Factor w/ 4 levels "s1","s2","s3",..: 2 3 1 2 3 2 2 1 3 1 ...
```

Pass the parsed formula and the data to
[`tulpa_build_model_data()`](https://gillescolling.com/tulpa/reference/tulpa_build_model_data.md).
The result is a list with the response vector, the fixed-effects design
matrix, an offset slot, the parsed random-effect terms, and a few count
fields the backends read.

``` r

bundle <- tulpa_build_model_data(pf, df)
names(bundle)
#> [1] "y"           "n_trials"    "X"           "offset"      "re_terms"   
#> [6] "n_obs"       "n_fixed"     "n_re_terms"  "fixed_names"
```

The design matrix `X` is the ordinary `model.matrix` output for the
fixed formula. It carries the intercept column unless you drop it,
expands factors into contrasts, and labels its columns.

``` r

dim(bundle$X)
#> [1] 200   3
colnames(bundle$X)
#> [1] "(Intercept)" "x"           "z"
head(bundle$X, 3)
#>   (Intercept)          x         z
#> 1           1 -0.6576350  1.443229
#> 2           1 -0.5229726 -1.112515
#> 3           1  0.1911304  1.123357
```

The response `y` comes from evaluating the left side of `~` against the
data, so a bare name, a transformation like `log(y)`, or a `cbind(...)`
pair all resolve through the same path. The count fields `n_obs`,
`n_fixed`, and `n_re_terms` are the sizes the backends need up front,
and `fixed_names` echoes the design columns.

``` r

c(n_obs = bundle$n_obs, n_fixed = bundle$n_fixed,
  n_re_terms = bundle$n_re_terms)
#>      n_obs    n_fixed n_re_terms 
#>        200          3          2
```

When [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
runs a full fit it calls these two functions for you, then attaches the
design back onto the returned object as `$model_matrix` and records
`$fixed_names` and the parameter layout. Calling the builders yourself
is useful for checking that a factor has the levels you expect or that
an interaction expanded the way you wanted, before any inference runs.
The cost is a single function call, and the payoff is that a design
problem shows up as a printed matrix instead of a confusing fit failure
several steps later.

## Random-effect terms

Each random-effect term becomes one entry in `bundle$re_terms`, and the
entry carries everything a backend needs to place group-level effects:
an integer index mapping every row to its group, the number of groups,
the level labels, the coefficient count, and a slope matrix when the
term has slopes. Read the first term, the random intercept `(1 | g)`.

``` r

re1 <- bundle$re_terms[[1]]
re1[c("group_var", "n_groups", "n_coefs", "has_intercept")]
#> $group_var
#> [1] "g"
#> 
#> $n_groups
#> [1] 6
#> 
#> $n_coefs
#> [1] 1
#> 
#> $has_intercept
#> [1] TRUE
re1$levels
#> [1] "a" "b" "c" "d" "e" "f"
```

`group_idx` is the heart of it. Each observation gets an integer
pointing at one of the `n_groups` levels, and the order of `levels`
fixes which integer means which group. A random intercept has
`n_coefs = 1` and no slope matrix, since the design column is an
implicit vector of ones.

``` r

head(re1$group_idx, 12)
#>  [1] 1 1 3 4 5 2 4 1 2 6 5 6
table(re1$group_idx)
#> 
#>  1  2  3  4  5  6 
#> 27 31 40 40 31 31
```

The second term, `(1 + x || site)`, adds a random slope on `x`. Its
coefficient count rises to two (an intercept and the slope), the slope
column shows up in `slope_matrix`, and `correlated` is `FALSE` because
the double bar asked for independent effects.

``` r

re2 <- bundle$re_terms[[2]]
re2[c("group_var", "n_groups", "n_coefs", "correlated")]
#> $group_var
#> [1] "site"
#> 
#> $n_groups
#> [1] 4
#> 
#> $n_coefs
#> [1] 2
#> 
#> $correlated
#> [1] FALSE
head(re2$slope_matrix)
#>            x
#> 1 -0.6576350
#> 2 -0.5229726
#> 3  0.1911304
#> 4  0.4008339
#> 5  0.6667822
#> 6 -0.6768624
```

Group levels come from factor levels, so the way you build a factor
controls the group ordering and which groups exist at all. If a level is
present in the factor but never observed, it still counts toward
`n_groups`, which inflates the number of group-level effects the model
carries with no data to inform them. To keep only the observed levels,
drop the unused ones with
[`droplevels()`](https://rdrr.io/r/base/droplevels.html) before fitting.
A character column is coerced to a factor on the fly. That works, and it
leaves the level order up to
[`as.factor()`](https://rdrr.io/r/base/factor.html), which sorts
alphabetically; setting the factor yourself keeps that order under your
control and makes the `levels` slot read the way you expect.

The link between `group_idx` and `levels` is worth holding onto, because
it is how a fit reports group-level effects back to you.
[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md) returns
one row per level in the order of `levels`, and a row of the design
points into that same vector through `group_idx`. When two factors share
level labels, keeping them as distinct columns rather than merging them
avoids a silent collision in the index.

The grouping side of a bar also accepts an expression rather than a bare
name, which covers crossed and nested designs without a reshape. Writing
`(1 | g:site)` crosses the two factors into one grouping, and tulpa
evaluates the interaction at build time instead of guessing column
names. Nested grouping `(1 | g/site)` expands into two terms, one for
`g` and one for the `g:site` combination, matching the lme4 convention.
The parsed object shows both terms, so you can confirm the expansion
before fitting rather than after.

``` r

pn <- tulpa_parse_formula(y ~ x + (1 | g/site))
pn$n_re_terms
#> [1] 2
vapply(pn$random_effects, `[[`, character(1), "group_var")
#> [1] "g"      "g:site"
```

## Binomial denominators and offsets

A binomial model needs to know the number of trials behind each success
count. tulpa offers two ways to supply that, and they suit different
data shapes. When your data frame has a column of successes and a column
of failures, pass both through
[`cbind()`](https://rdrr.io/r/base/cbind.html) on the response side. The
builder evaluates the call and returns a two-column matrix.

``` r

db <- data.frame(x = rnorm(n))
prob   <- plogis(-0.2 + 0.8 * db$x)
db$succ <- rbinom(n, 15, prob)
db$fail <- 15 - db$succ

pf_b   <- tulpa_parse_formula(cbind(succ, fail) ~ x)
bun_b  <- tulpa_build_model_data(pf_b, db)
head(bun_b$y, 3)
#> [1] 3 8 7
```

When the denominators live in their own vector, keep the response as the
success count and hand the trials to
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) through
`n_trials`, a vector with one entry per row. This is the shape to reach
for when every unit was observed a known number of times and you would
rather not carry a redundant failure column. The two routes are
equivalent: a single-column response of successes paired with
`n_trials`, or a two-column `cbind(succ, fail)` response, describe the
same model. Binary data is the special case where every denominator is
one, so a plain `0/1` response needs neither route.

``` r

db$tot <- 15L
fit_b <- tulpa(succ ~ x, data = db, family = "binomial",
               n_trials = db$tot, mode = "laplace")
coef(fit_b)
#> (Intercept)           x 
#>  -0.2868062   0.8230279

fit_cbind <- tulpa(cbind(succ, fail) ~ x, data = db, family = "binomial",
                   mode = "laplace")
max(abs(coef(fit_cbind) - coef(fit_b)))
#> [1] 0
```

An offset is a known additive term on the linear-predictor scale, fixed
at coefficient one rather than estimated, and exposure in a rate model
is the common case: a Poisson count observed over a varying window of
time or area enters as `offset(log(exposure))`, which turns the modelled
mean into a rate times exposure. The builder pulls the offset out of the
design and returns it in its own slot, so it never appears as a fitted
column.

``` r

dp <- data.frame(x = rnorm(n), expo = runif(n, 1, 8))
dp$y <- rpois(n, dp$expo * exp(0.1 + 0.5 * dp$x))

pf_o  <- tulpa_parse_formula(y ~ x + offset(log(expo)))
bun_o <- tulpa_build_model_data(pf_o, dp)
head(bun_o$offset, 4)
#> [1] 1.97118068 1.59092806 0.09973003 1.87423420
colnames(bun_o$X)
#> [1] "(Intercept)" "x"
```

The offset column is `log(expo)` evaluated row by row, and `X` holds
only the intercept and `x`. The transformation inside `offset(...)` runs
as written, so the log is yours to apply, not something tulpa adds. Fit
it as a Poisson model and the slope recovers near its true value of 0.5,
with the exposure absorbed through the offset rather than estimated as a
free coefficient.

``` r

fit_o <- tulpa(y ~ x + offset(log(expo)), data = dp,
               family = "poisson", mode = "laplace")
coef(fit_o)
#> (Intercept)           x 
#>   0.1037010   0.4749073
```

## Validation and errors

The builders fail early and name the problem, rather than passing a
half-formed design to a backend. The most common slip is a name in the
formula that has no matching column. Suppose the grouping factor is
called `g` in the data but the formula asks for `region`: the build
stops and quotes the offending name.

``` r

bad <- tulpa_parse_formula(y ~ x + (1 | region))
tulpa_build_model_data(bad, df)
#> Error:
#> ! Grouping variable 'region' not found in data
```

A response that cannot be found is reported the same way, against the
label that appeared on the left of `~`.

``` r

tulpa_build_model_data(tulpa_parse_formula(count ~ x), df)
#> Error:
#> ! Response 'count' not found in data
```

The special terms check their own shape at parse time. A `spatial(...)`
term wraps exactly one bare column name, so a call with the wrong arity
is caught before any data is touched, with a message that shows the
intended form.

``` r

tulpa_parse_formula(y ~ spatial(x, g))
#> Error:
#> ! spatial(...) takes exactly one bare column name, e.g. spatial(region).
```

A spatial term also has a partner requirement that surfaces at fit time.
Naming a unit column with `spatial(col)` tells tulpa where each row
sits, but the field structure (the adjacency for an areal model,
coordinates for a continuous one) arrives through the `spatial=`
argument. Leaving it out stops the fit with guidance on what to pass.

``` r

tulpa(y ~ x + spatial(g), data = df, family = "poisson", mode = "laplace")
#> Error:
#> ! Formula has a spatial(g) term but the structure spec `spatial=` was not supplied (e.g. list(type = 'icar', adjacency = W) or spatial_gp(~ lon + lat)).
```

These checks share a habit worth leaning on. The formula declares
intent, the data supplies the columns, and the builder reconciles the
two before anything numeric happens. When a fit fails to start, parsing
the formula on its own and then building the model data by hand isolates
whether the trouble is in the formula grammar or in the columns the data
does or does not carry. Reading `bundle$re_terms` confirms the group
counts; reading `colnames(bundle$X)` confirms how factors and
interactions expanded. Most data-shape questions answer themselves once
those two pieces are on screen.

## See also

- [`?tulpa_parse_formula`](https://gillescolling.com/tulpa/reference/tulpa_parse_formula.md),
  [`?tulpa_build_model_data`](https://gillescolling.com/tulpa/reference/tulpa_build_model_data.md)
  for the function references.
- The getting-started vignette for the fit, extract, predict, and
  compare workflow once the data is in shape.
- The spatial and priors vignettes for the structures that ride
  alongside the formula through `spatial=` and the prior arguments.
