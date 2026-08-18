# Corrected R-INLA CCD integration weights

Per-point design weights for a
[`ccd_grid()`](https://gillescolling.com/tulpa/reference/ccd_grid.md)
used as nested-Laplace integration nodes. Implements the corrected
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
ccd_weights(ccd)
```

## Arguments

- ccd:

  A
  [`ccd_grid()`](https://gillescolling.com/tulpa/reference/ccd_grid.md)
  result. The standardized scaling is recovered as
  `f0 = ccd$f_0 / sqrt(m)`: a `ccd_grid(m, f_0 = sqrt(m) * f0)` design
  places the factorial corners at `+/- f0`, matching the INLA convention
  (`f0 = 1.1` is the INLA default).

## Value

Numeric vector of length `ccd$n_points`: the design weight for each
point (the centre weight `w0` for the central point, `w` otherwise).

## See also

[`ccd_grid()`](https://gillescolling.com/tulpa/reference/ccd_grid.md),
[`ccd_to_theta()`](https://gillescolling.com/tulpa/reference/ccd_to_theta.md).
