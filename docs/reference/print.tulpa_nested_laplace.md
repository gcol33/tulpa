# Print method for nested-Laplace fits

Compact one-screen summary of a
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
or
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
fit: the hyperparameters integrated over, the outer-grid size, the outer
Pareto-\\\hat{k}\\ accuracy diagnostic when present, and the wall-clock
timing line (`"fit in 5h 25m (grid 2h 09m)"`) when `fit$timing` is
attached. Inherited by the single-block joint and multi-block joint
subclasses.

## Usage

``` r
# S3 method for class 'tulpa_nested_laplace'
print(x, ...)
```

## Arguments

- x:

  A `tulpa_nested_laplace` fit (or a joint subclass).

- ...:

  Ignored.

## Value

`x`, invisibly.
