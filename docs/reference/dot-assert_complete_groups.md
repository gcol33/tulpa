# Reject an incomplete random-effect design

The formula's grouping factors and random-slope covariates are built
with `na.action = na.pass` as the fixed design is, so an NA group or
slope value survives into `re_terms`. There it reached each backend in
its own words – an `re_idx_list` index message on the Laplace door, "NAs
are not allowed in subscripted assignments" on the samplers
(gcol33/tulpa#880). Refused here with the message
[`.assert_finite_model_inputs()`](https://gillescolling.com/tulpa/reference/dot-assert_finite_model_inputs.md)
gives the predictors.

## Usage

``` r
.assert_complete_groups(re_terms, where = NULL)
```

## Arguments

- re_terms:

  The `re_terms` of
  [`tulpa_build_model_data()`](https://gillescolling.com/tulpa/reference/tulpa_build_model_data.md).

- where:

  Caller name for the message prefix, or `NULL` for none.
