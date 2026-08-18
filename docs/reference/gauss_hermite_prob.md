# Gauss-Hermite quadrature (probabilist's, exp(-z^2/2) weight)

Computes nodes and weights for `\int f(z) (2\pi)^{-1/2} exp(-z^2/2) dz`
via Golub-Welsch on the Hermite Jacobi matrix. Sums of `w_k f(z_k)`
approximate `E_{Z ~ N(0,1)}[f(Z)]`.

## Usage

``` r
gauss_hermite_prob(n)
```

## Arguments

- n:

  Number of quadrature nodes.

## Value

List with `nodes` and `weights` (each length `n`).
