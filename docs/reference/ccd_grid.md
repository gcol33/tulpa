# Central Composite Design (CCD) grid for nested-Laplace integration

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
[`ccd_to_theta()`](https://gillescolling.com/tulpa/reference/ccd_to_theta.md).

## Usage

``` r
ccd_grid(k, f_0 = sqrt(k))
```

## Arguments

- k:

  Number of hyperparameters (length of theta). Must be \>= 1.

- f_0:

  Radius of the design sphere (default \\\sqrt{k}\\). Larger `f_0`
  spreads points further out; smaller concentrates near the centre.
  INLA's default scales like \\\sqrt{k}\\.

## Value

A list with components:

- `z`: numeric matrix `[n_points x k]` of standardised hyperparameter
  coordinates.

- `n_points`: integer; total grid size.

- `kind`: character vector labelling each point as `"center"`,
  `"axial"`, or `"factorial"`.

- `f_0`: the sphere radius used.

## See also

[`ccd_to_theta()`](https://gillescolling.com/tulpa/reference/ccd_to_theta.md)
to map z-coordinates to physical theta.
