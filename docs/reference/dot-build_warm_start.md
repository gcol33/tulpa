# Assemble NUTS initial positions and an inverse-mass diagonal

Assemble NUTS initial positions and an inverse-mass diagonal

## Usage

``` r
.build_warm_start(
  fit,
  layout,
  re_terms,
  n_chains,
  sigma_re = NULL,
  jitter = 1,
  metric = FALSE
)
```

## Arguments

- fit:

  Source fit (`mode = "eb"` or `"laplace"`).

- layout:

  The target sampler layout, from `cpp_tulpa_glmm_layout()`.

- re_terms:

  Random-effect terms of the model being sampled.

- n_chains:

  Number of chains to initialise.

- sigma_re:

  Random-effect SDs, when the source fit conditioned on them.

- jitter:

  Scale of the between-chain dispersion, as a multiple of each
  parameter's warm-start SD. `0` stacks every chain on the mode.

- metric:

  Hand the assembled inverse-mass diagonal to the kernel as well as the
  position. `FALSE` (default) because it can only be filled for some
  blocks today and a partial metric measurably slows the sampler; see
  the note at the end of this function.

## Value

`list(init = <n_chains x D matrix>, inv_metric_diag = <D vector or NULL>)`.
