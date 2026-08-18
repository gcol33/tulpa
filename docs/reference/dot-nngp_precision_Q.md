# Build NNGP precision Lambda = (I - A)' D^-1 (I - A) at given hyperparameters.

Mirrors the algorithm in `src/gpu_nngp_laplace.h::batch_nngp_scatter`
plus `apply_nngp_full_prior_dense` – for each Vecchia row i (in NNGP
order):

- resolve conditioning set N(i) via `nn_idx` / `nn_order`

- form the n_nb x n_nb cov matrix C and the n_nb cov vector c

- solve C alpha = c

- v_i = sigma^2 - c' alpha

- beta_i is 1 at i and -alpha_k at each neighbor -\> Lambda += beta_i
  beta_i' / v_i

## Usage

``` r
.nngp_precision_Q(spatial, sigma2_gp, phi_gp)
```

## Details

Identifiers follow the C++ side: `nn_idx[i, k]` is a 1-based NNGP-order
index, `nn_order[j]` maps NNGP-order j (1-based) to obs idx (1-based).
