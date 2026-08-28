# Parse a mixed-model formula

Decomposes a formula into fixed effects and random effects by walking
the formula's abstract syntax tree. This is structural recursion with
pattern matching on bar terms – no string manipulation.

## Usage

``` r
tulpa_parse_formula(formula)
```

## Arguments

- formula:

  A formula object (e.g., `y ~ x + (1 | group)`)

## Value

A list with:

- `response`: character, the response variable label (`NULL` if none)

- `response_expr`: language object, the LHS of `~` (`NULL` if none)

- `fixed_formula`: formula, the fixed-effects-only formula

- `random_effects`: list of parsed RE specifications

- `n_re_terms`: integer, number of RE terms

- `latent_blocks`: list of evaluated user-defined latent block objects
  (typically `tgmrf` S3 objects), one per `latent(...)` term in the
  original formula. Empty list when no `latent(...)` terms are present.

- `n_latent_blocks`: integer, number of `latent(...)` terms

- `original`: the original formula

## Examples

``` r
pf <- tulpa_parse_formula(y ~ x1 + x2 + (1 | group) + (x1 || site))
pf$response        # "y"
#> [1] "y"
pf$fixed_formula   # y ~ x1 + x2
#> y ~ x1 + x2
#> <environment: 0x0000028324cb4008>
```
