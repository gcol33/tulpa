# Central composite design (CCD) grid for nested-Laplace integration

Produces a structured set of standardised hyperparameter points \\z \in
\mathbb{R}^k\\ for use as integration nodes in a nested Laplace
approximation when \\k \ge 3\\. CCD scales much better than the full
tensor [`expand.grid()`](https://rdrr.io/r/base/expand.grid.html) used
by the 1D and 2D backends: a CCD has \\1 + 2k + 2^{k - q}\\ points (1
centre, 2k axial, \\2^{k - q}\\ factorial), versus \\m^k\\ for an
`m`-per-axis tensor product.

Point layout (with centre+axial+factorial scaling \\f_0\\):

- 1 centre point at the origin;

- 2k axial points at \\\pm f_0\\ along each coordinate axis;

- \\2^{k - q}\\ factorial points at corners of the hypercube, scaled to
  lie on a sphere of radius \\f_0\\.

For \\k \le 6\\ the factorial portion is the full \\2^k\\ design. For
\\k \ge 7\\ a half-fraction (\\q = 1\\) using the defining word \\x_1
\cdots x_k\\ keeps the point count reasonable while preserving
Resolution V.

Used by
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
for higher-dimensional hyperparameter blocks. The standardised
z-coordinates are mapped to physical hyperparameters \\\theta\\ via
`ccd_to_theta()`.

`ccd_to_theta()` converts standardised CCD z-coordinates (in
\\\mathbb{R}^k\\) to physical hyperparameters \\\theta\\ via the affine
map \\\theta = \hat\theta + L \cdot z\\ with optional log-scale
transform per component. `L` is typically a Cholesky factor of the
negative Hessian inverse evaluated at the (working) mode \\\hat\theta\\
– i.e. it scales `z` to one posterior standard deviation per axis.

`ccd_weights()` gives the per-point design weights for a `ccd_grid()`
used as nested-Laplace integration nodes. It implements the corrected
R-INLA central-composite-design weights: the formula in Rue, Martino and
Chopin (2009) has a typo, corrected by the R-INLA team to give a
positive central weight. With `m` hyperparameters, `np` design points
and standardized scaling `f0`, every non-centre point gets \$\$w = 1 /
((np - 1)(1 + e^{-m f0^2 / 2}(f0^2 - 1)))\$\$ and the centre point gets
\\w_0 = 1 - (np - 1) w\\. Used as \\\Delta_k\\: the integration weight
of node \\k\\ is \\\Delta_k \exp(\log\\\mathrm{marginal}\_k)\\,
renormalized. On a standardized (whitened) hyperparameter posterior
these reproduce the Gaussian moments exactly.

## Usage

``` r
ccd_grid(k, f_0 = sqrt(k))

ccd_to_theta(z, theta_hat, L, log_scale = FALSE)

ccd_weights(ccd)
```

## Arguments

- k:

  Number of hyperparameters (length of theta). Must be \>= 1.

- f_0:

  Radius of the design sphere (default \\\sqrt{k}\\). Larger `f_0`
  spreads points further out; smaller concentrates near the centre.
  INLA's default scales like \\\sqrt{k}\\.

- z:

  Matrix `[n_points x k]` of standardised coordinates from `ccd_grid()`.

- theta_hat:

  Numeric vector of length `k`: centre of the design, in either physical
  or log-space (per `log_scale`).

- L:

  Numeric `[k x k]` matrix: scale/rotation applied to z. Pass `diag(sd)`
  for a diagonal axis-aligned grid where `sd` is the per-axis posterior
  SD; pass a Cholesky factor to capture correlations between
  hyperparameters.

- log_scale:

  Logical (or logical vector of length k). If `TRUE` for component j,
  `theta_hat[j]` and column j of the affine transform live on log-scale
  and are exponentiated afterward (useful for positive parameters like
  tau). Default `FALSE` everywhere.

- ccd:

  A `ccd_grid()` result. The standardized scaling is recovered as
  `f0 = ccd$f_0 / sqrt(m)`: a `ccd_grid(m, f_0 = sqrt(m) * f0)` design
  places the factorial corners at `+/- f0`, matching the INLA convention
  (`f0 = 1.1` is the INLA default).

## Value

`ccd_grid()`: a list with components:

- `z`: numeric matrix `[n_points x k]` of standardised hyperparameter
  coordinates.

- `n_points`: integer; total grid size.

- `kind`: character vector labelling each point as `"center"`,
  `"axial"`, or `"factorial"`.

- `f_0`: the sphere radius used.

`ccd_to_theta()`: a numeric matrix `[n_points x k]` of physical
theta-values.

`ccd_weights()`: a numeric vector of length `ccd$n_points`, the design
weight for each point (the centre weight `w0` for the central point, `w`
otherwise).

## Examples

``` r
g <- ccd_grid(3)
g$n_points
#> [1] 15
table(g$kind)
#> 
#>     axial    center factorial 
#>         6         1         8 
ccd_weights(g)
#>  [1] 0.00000000 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857
#>  [7] 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857
#> [13] 0.07142857 0.07142857 0.07142857

g2 <- ccd_grid(2)
ccd_to_theta(g2$z, theta_hat = c(0, 1), L = diag(c(0.5, 0.2)))
#>             [,1]      [,2]
#>  [1,]  0.0000000 1.0000000
#>  [2,]  0.7071068 1.0000000
#>  [3,] -0.7071068 1.0000000
#>  [4,]  0.0000000 1.2828427
#>  [5,]  0.0000000 0.7171573
#>  [6,] -0.5000000 0.8000000
#>  [7,]  0.5000000 0.8000000
#>  [8,] -0.5000000 1.2000000
#>  [9,]  0.5000000 1.2000000
```
