# Format a single prior for printing

Thin wrapper around [`format()`](https://rdrr.io/r/base/format.html) for
`tulpa_prior` objects. Each `prior_*()` constructor attaches a subclass
(e.g. `tulpa_prior_normal`) so adding a new distribution is one new
`format.tulpa_prior_<dist>()` method – no central if/else to extend.

## Usage

``` r
print_prior(prior, indent = "")
```

## Arguments

- prior:

  A tulpa_prior object

- indent:

  Indentation string
