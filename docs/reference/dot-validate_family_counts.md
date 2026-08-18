# Validate the response `y` for a count family.

Errors when `family` is a count family and `y` carries negative or
non-integer values, which the integer-casting kernels would silently
floor into a biased likelihood. A no-op for continuous families.

## Usage

``` r
.validate_family_counts(family, y, zi = FALSE)
```

## Arguments

- zi:

  Whether a zero-inflation component is being fitted alongside the
  family. With a zero-truncated family this is the hurdle model, where
  the zeros belong to the zero component rather than to the truncated
  count density, so the `y >= 1` requirement does not apply to them.
