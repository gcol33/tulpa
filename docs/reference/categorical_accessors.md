# Observation-level accessors for categorical fits

[`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
[`residuals()`](https://rdrr.io/r/stats/residuals.html),
[`predict()`](https://rdrr.io/r/stats/predict.html) and
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
for a categorical response: a
[`tulpa_multinomial()`](https://gillescolling.com/tulpa/reference/tulpa_multinomial.md)
fit (`family = "multinomial"`) or a
[`tulpa_ordinal()`](https://gillescolling.com/tulpa/reference/tulpa_ordinal.md)
fit (`family = "ordinal"`). The response is one of K classes, so the
response-scale quantities are N x K matrices with one column per class.

- [`fitted()`](https://rdrr.io/r/stats/fitted.values.html) returns the
  class probabilities at the posterior mode.

- [`residuals()`](https://rdrr.io/r/stats/residuals.html) returns, per
  class, the indicator of the observed class minus its fitted
  probability (`"response"`), or that difference divided by the
  indicator's standard deviation `sqrt(p (1 - p))` (`"pearson"`). A
  categorical response has no single scalar residual; each class column
  is the Bernoulli residual of that class's indicator.

- [`predict()`](https://rdrr.io/r/stats/predict.html) returns the link
  scale – the K - 1 baseline-category logits of the multinomial model (N
  x (K - 1)), or the ordinal location predictor `x' beta` (length N) –
  or the class probabilities (`type = "response"`).

- [`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
  draws one class per observation for each posterior draw and returns
  the class indices, with the response levels attached;
  [`simulate.tulpa_fit()`](https://gillescolling.com/tulpa/reference/simulate.tulpa_fit.md)
  turns them into factor columns.

The pointwise log-likelihood behind
[`cpo()`](https://gillescolling.com/tulpa/reference/criteria_doors.md),
[`dic()`](https://gillescolling.com/tulpa/reference/criteria_doors.md),
[`loo::waic()`](https://mc-stan.org/loo/reference/waic.html) and
[`loo::loo()`](https://mc-stan.org/loo/reference/loo.html) is the log
probability of each observed class at each stored posterior draw;
[`dic()`](https://gillescolling.com/tulpa/reference/criteria_doors.md)
evaluates it at the posterior-mean parameters.

## Usage

``` r
# S3 method for class 'tulpa_categorical'
fitted(object, ...)

# S3 method for class 'tulpa_categorical'
residuals(object, type = c("pearson", "response"), ...)

# S3 method for class 'tulpa_categorical'
predict(
  object,
  newdata = NULL,
  type = c("link", "response"),
  se.fit = FALSE,
  level = 0.95,
  ...
)

# S3 method for class 'tulpa_categorical'
posterior_predict(object, newdata = NULL, ndraws = NULL, seed = NULL, ...)

# S3 method for class 'tulpa_categorical'
pp_check(object, ndraws = 50, ...)
```

## Arguments

- object:

  A `tulpa_multinomial` or `tulpa_ordinal` fit.

- ...:

  Ignored.

- type:

  For [`residuals()`](https://rdrr.io/r/stats/residuals.html),
  `"pearson"` (default) or `"response"`. For
  [`predict()`](https://rdrr.io/r/stats/predict.html), `"link"`
  (default) or `"response"`.

- newdata:

  Optional data frame of covariates; `NULL` uses the training design.

- se.fit:

  For [`predict()`](https://rdrr.io/r/stats/predict.html), also return
  standard errors and credible bounds. On the link scale the standard
  error is exact for the Gaussian (Laplace) posterior, `sqrt(a' V a)`,
  with Gaussian bounds; on the response scale the standard error and the
  bounds are the posterior SD and quantiles of each class probability
  over the fit's posterior draws.

- level:

  Credible-interval level (default 0.95).

- ndraws:

  Number of posterior draws for
  [`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md);
  defaults to all stored draws.

- seed:

  Optional integer seed (RNG state is restored on exit).

## Value

[`fitted()`](https://rdrr.io/r/stats/fitted.values.html): an N x K
probability matrix.
[`residuals()`](https://rdrr.io/r/stats/residuals.html): an N x K
matrix. [`predict()`](https://rdrr.io/r/stats/predict.html): a matrix
(or, for the ordinal link scale, a vector); with `se.fit = TRUE` a list
of `fit`, `se.fit`, `lower`, `upper` of that shape.
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md):
an `ndraws x N` integer matrix of class indices with attributes `levels`
and `ordered`.

## See also

[`tulpa_multinomial()`](https://gillescolling.com/tulpa/reference/tulpa_multinomial.md),
[`tulpa_ordinal()`](https://gillescolling.com/tulpa/reference/tulpa_ordinal.md)
