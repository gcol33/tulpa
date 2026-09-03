# User-defined GMRF latent block

`tgmrf()` lets a user define a Gaussian Markov random field as a latent
block in a tulpa model by supplying two R closures: a precision-matrix
factory `Q(theta)` and a log-prior on theta `prior(theta)`. The
resulting object plugs into a formula via `latent(tgmrf(...))` and is
consumed by every tulpa inference tier (Laplace, EM+Laplace, VI,
nested_laplace+CCD, NUTS, IMH-Laplace).

This is the latent-side dual of `LikelihoodSpec`. `LikelihoodSpec` lets
a model package own the observation model; `tgmrf()` lets a script own a
single latent block.

## Usage

``` r
tgmrf(
  Q,
  prior,
  init,
  mu = NULL,
  graph = NULL,
  bounds = NULL,
  obs_idx = NULL,
  name = NULL
)
```

## Arguments

- Q:

  A function `function(theta)` returning a sparse precision matrix of
  class
  [`Matrix::sparseMatrix`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html)
  (typically `dgCMatrix`). The matrix must be square and symmetric.
  Called once at registration with `theta = init` to infer `n_latent`
  and capture the sparsity pattern.

- prior:

  A function `function(theta)` returning a finite numeric scalar – the
  log-prior density at `theta`.

- init:

  Numeric vector of starting values for `theta`. Names, if present,
  become the canonical theta names.

- mu:

  Optional function `function(theta)` returning a numeric vector of
  length `n_latent`. Default `NULL` is equivalent to a zero mean.

- graph:

  Optional sparse matrix specifying the upper bound on `Q`'s sparsity
  pattern. If supplied, the registration check verifies that the nonzero
  pattern of `Q(init)` is a subset.

- bounds:

  Optional list with components `lower` and `upper`, each a numeric
  vector of length `length(init)`. Used by `nested_laplace + CCD` to
  build the outer grid; unused by NUTS / VI.

- obs_idx:

  Optional integer vector mapping each observation to a latent slot in
  `[1, n_latent]`. If `NULL` (default), the fit-time driver assumes
  `N == n_latent` and uses `seq_len(N)` – i.e. one observation per
  latent slot, in row order.

- name:

  Optional character; cosmetic label used by
  [`print()`](https://rdrr.io/r/base/print.html) /
  [`summary()`](https://rdrr.io/r/base/summary.html).

## Value

An object of class `c("tgmrf", "tulpa_latent_block")` with components:

- `Q`, `prior`, `mu` – the user closures (or `NULL` for `mu`).

- `init`, `theta_names`, `theta_dim` – hyperparameter metadata.

- `n_latent` – inferred from `Q(init)`.

- `pattern` – captured sparsity pattern as a `dgCMatrix` of 1s.

- `graph` – user-supplied pattern, validated against `pattern` if given.

- `bounds`, `name` – passed through.

## Details

For a Gaussian latent block `z ~ N(mu(theta), Q(theta)^{-1})`:

\$\$\log p(z\mid\theta) = \tfrac{1}{2}\log\det Q(\theta) -
\tfrac{1}{2}(z - \mu(\theta))^\top Q(\theta)(z - \mu(\theta)) +
\mathrm{const}\$\$

the gradient `-Q(z - mu)` and Hessian `-Q` wrt `z` are closed form, so
the user never writes gradient code. The Laplace inner step needs only
`Q(theta)` and `mu(theta)` at numeric `theta`. NUTS and VI additionally
use a forward finite-difference gradient on theta, costing `dim(theta)`
extra `Q` calls per outer step – cheap for the typical
`dim(theta) <= 5`.

## See also

[`latent()`](https://gillescolling.com/tulpa/reference/latent.md) for
the formula slot that consumes a `tgmrf` block.

## Examples

``` r
# Periodic AR(1) block, wrap-around tridiagonal precision.
periodic_ar1 <- function(n) {
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1]); rho <- tanh(theta[2])
      d <- rep((1 + rho^2) / sigma^2, n)
      o <- rep(-rho / sigma^2, n)
      M <- Matrix::bandSparse(
        n, k = c(-1, 0, 1), diagonals = list(o, d, o)
      )
      M[1, n] <- M[n, 1] <- -rho / sigma^2
      methods::as(M, "CsparseMatrix")
    },
    prior = function(theta) {
      stats::dnorm(theta[1], 0, 1, log = TRUE) +
        stats::dnorm(theta[2], 0, 1, log = TRUE)
    },
    init = c(log_sigma = 0, atanh_rho = 0),
    name = "periodic_ar1"
  )
}

blk <- periodic_ar1(20)
print(blk)
```
