# Select the "main" model parameters for diagnostic display

Drops per-element latent-field entries (names ending in a bracketed
index, e.g. `u[12]`, `w[3]`) so plots and summaries focus on scalar
coefficients and hyperparameters rather than thousands of latent values.

## Usage

``` r
select_main_params(param_names)
```

## Arguments

- param_names:

  Character vector of parameter names.

## Value

The subset of `param_names` that are not bracketed-index entries (or the
full vector if that would be empty).
