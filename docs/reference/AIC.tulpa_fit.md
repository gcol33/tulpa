# Information criteria on a tulpa fit

AIC and BIC penalise a maximised log-likelihood.
[`logLik.tulpa_fit()`](https://gillescolling.com/tulpa/reference/logLik.tulpa_fit.md)
reports one only for a fit that maximised over every parameter it
reports (`quantity = "log_likelihood"`, as
[`agq_fit()`](https://gillescolling.com/tulpa/reference/agq_fit.md)
does); a mean log posterior, a log evidence and a conditional log
marginal likelihood are not, and both criteria refuse them rather than
return a number. Compare those fits by their evidence, or by
`compare_models(criterion = "waic")` / `"loo"`.

## Usage

``` r
# S3 method for class 'tulpa_fit'
AIC(object, ..., k = 2)

# S3 method for class 'tulpa_fit'
BIC(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Further fits.

- k:

  Penalty per parameter.

## Value

As [`stats::AIC()`](https://rdrr.io/r/stats/AIC.html) /
[`stats::BIC()`](https://rdrr.io/r/stats/AIC.html) for maximised
log-likelihoods; errors otherwise.
