# Gauss-Hermite rule for the standard normal

Nodes and weights of the `n`-point Gauss-Hermite rule for the
probabilists' weight, normalised to the standard normal density, so that
`sum(weights * f(nodes))` approximates `E[f(Z)]` for `Z ~ N(0, 1)` and
is exact for polynomials of degree up to `2n - 1`. Computed by
Golub-Welsch: the nodes are the eigenvalues of the symmetric tridiagonal
Jacobi matrix of the probabilists' Hermite recurrence (zero diagonal,
off-diagonal `sqrt(k)`), the weights the squared first components of its
eigenvectors. A normal `N(m, s^2)` is integrated at `m + s * nodes` with
the same weights.

## Usage

``` r
tulpa_gauss_hermite(n)
```

## Arguments

- n:

  Number of nodes (a positive integer).

## Value

A list with `nodes` (increasing) and `weights` (summing to one), each of
length `n`.

## See also

[`tulpa_re_aghq()`](https://gillescolling.com/tulpa/reference/tulpa_re_aghq.md),
[`tulpa_latent_ascent()`](https://gillescolling.com/tulpa/reference/tulpa_latent_ascent.md)

## Examples

``` r
gh <- tulpa_gauss_hermite(5)
sum(gh$weights * gh$nodes^4)   # E[Z^4] = 3
#> [1] 3
```
