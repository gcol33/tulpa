# Non-separable spatiotemporal GP

Specify a non-separable Gaussian Process for spatiotemporal effects.
Unlike separable models where the covariance factors as \\C_s \otimes
C_t\\, non-separable models allow for direct space-time interaction in
the covariance.

No tulpa backend fits a joint space-time covariance, so this constructor
errors. A spatial GP alongside a temporal field is fitted by
`tulpa(spatial = spatial_gp(...), temporal = ...)`.

## Usage

``` r
spatiotemporal_gp(
  coords,
  time_var,
  cov_space = c("exponential", "matern", "gaussian", "spherical"),
  cov_time = c("exponential", "matern", "gaussian"),
  nonsep_type = c("product", "sum", "gneiting", "cressie_huang"),
  nn = 15,
  shared = NULL
)
```

## Arguments

- coords:

  A one-sided formula specifying coordinate columns (e.g.,
  `~ lon + lat`), or a character vector of length 2.

- time_var:

  Name of the time variable in data.

- cov_space:

  Spatial covariance: `"exponential"` (default), `"matern"`,
  `"gaussian"`, or `"spherical"`.

- cov_time:

  Temporal covariance: `"exponential"` (default), `"matern"`, or
  `"gaussian"`.

- nonsep_type:

  Non-separability type:

  - `"product"`: \\C\_{st} = C_s \cdot C_t\\ (separable, for reference)

  - `"sum"`: \\C\_{st} = C_s + C_t\\

  - `"gneiting"`: Gneiting (2002) non-separable class

  - `"cressie_huang"`: Cressie-Huang (1999) non-separable class

- nn:

  Number of nearest neighbors for NNGP approximation. Default 15.

- shared:

  Logical; if TRUE (default), effect enters both processes.

## Value

Nothing: the call always signals an error.

## Details

The non-separable covariance functions allow for more flexible
space-time dependence:

**Gneiting class:** \$\$C(h, u) = \frac{\sigma^2}{(a\|u\|^{2\alpha} +
1)^{\tau}} \exp\left(-\frac{c\\h\\^{2\gamma}}{(a\|u\|^{2\alpha} +
1)^{\beta\gamma}}\right)\$\$

where h is spatial lag, u is temporal lag, and parameters control the
space-time interaction.

**Cressie-Huang class:** Constructed via Fourier transform methods to
ensure positive definiteness.

## References

Gneiting, T. (2002). Nonseparable, stationary covariance functions for
space-time data. Journal of the American Statistical Association,
97(458), 590-600.

Cressie, N., & Huang, H. C. (1999). Classes of nonseparable,
spatio-temporal stationary covariance functions. Journal of the American
Statistical Association, 94(448), 1330-1340.
