# The response as the numeric vector the kernels read

A logical response is the binary idiom
[`glm()`](https://rdrr.io/r/stats/glm.html) accepts for a binomial
family, `TRUE` a success; it is read as 0/1 there. Anywhere else, and
for any other non-numeric response (a factor, a character vector), the
family has no reading of it, so it is refused by name rather than left
to fail inside a backend in that backend's words –
`is.numeric(y) is not TRUE` on the Laplace door, a failed inner solve on
EB, non-finite integration nodes on the structured tier
(gcol33/tulpa#880).

## Usage

``` r
.numeric_response(family, y)
```

## Arguments

- family:

  Family identifier, canonical or aliased.

- y:

  The response, or `NULL`.
