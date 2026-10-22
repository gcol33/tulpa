# Energy Bayesian Fraction of Missing Information (E-BFMI)

Computes E-BFMI (Betancourt 2016) from the per-iteration Hamiltonian
energies an HMC or NUTS sampler records: the variance of the energy
transition between successive iterations over the variance of the
marginal energy. Each chain is diffed separately, since a transition
across a chain boundary compares unrelated trajectories, and the lowest
chain value is reported: one badly exploring chain is enough to distrust
a fit. Values below 0.3 indicate that momentum resampling moves the
sampler too little between energy level sets.

## Usage

``` r
energy_bfmi(energy, chain = NULL)
```

## Arguments

- energy:

  Numeric vector of per-iteration energies.

- chain:

  Chain index of each entry of `energy`. `NULL` (default), or a vector
  of the wrong length, treats `energy` as one chain.

## Value

A list with `value` (the lowest per-chain E-BFMI, `NA` when no chain has
two finite, non-constant energies), `per_chain`, `energy_centred`
(energies centred within each chain) and `energy_diff` (within-chain
transitions), the last two pooled across chains for plotting.

## Examples

``` r
set.seed(1)
energy <- c(cumsum(rnorm(200, sd = 0.3)), rnorm(200, sd = 1))
energy_bfmi(energy, chain = rep(1:2, each = 200))$per_chain
```
