# Fixed-effect coefficients

Fixed-effect coefficients

## Usage

``` r
# S3 method for class 'tulpa_fit'
coef(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Ignored.

## Value

Named numeric vector of fixed-effect posterior means (the Laplace mode
for the Laplace tier). Random effects come from
[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md).
