# Supported R-level family names

The response families `family = ` accepts on the engine's front doors,
in their canonical (default-link) spelling. A family's `*_<link>`
variants (e.g. `"gamma_inverse"`) are listed separately by
[`linked_family_names()`](https://gillescolling.com/tulpa/reference/linked_family_names.md).

## Usage

``` r
family_names()
```

## Value

A character vector of family names.

## Examples

``` r
family_names()
#>  [1] "binomial"                 "poisson"                 
#>  [3] "neg_binomial_2"           "neg_binomial_1"          
#>  [5] "truncated_poisson"        "truncated_neg_binomial_2"
#>  [7] "lognormal"                "gaussian"                
#>  [9] "beta"                     "gamma"                   
#> [11] "inverse_gaussian"         "beta_binomial"           
#> [13] "tweedie"                  "t"                       
```
