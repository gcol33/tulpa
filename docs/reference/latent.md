# Mark an expression as a latent block in a tulpa formula

Wraps a user-defined latent block (currently a
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) object)
so the formula parser can recognise and route it to the inference layer.
The call is structural – it is never executed at fit time. The parser
evaluates the inner expression in the formula's environment, removes the
`latent(...)` term from the fixed-effects formula, and attaches the
resulting object to `parsed$latent_blocks`.

## Usage

``` r
latent(block)
```

## Arguments

- block:

  A `tgmrf` (or, in the future, `tgeneric`) object.

## Value

Returns `block` invisibly. Outside a formula context the call is a no-op
pass-through.
