# Potential issues at 20M-cell scale

Three scaling regimes that compound. We need to pin down which applies, and how they combine, because cost is multiplicative across them.

---

## Issue 1 — Outer hyperparameter grid (sigma, alpha, ...) explodes

**What it means.** The joint Laplace integrates over a hyperparameter grid; each grid cell triggers an inner Newton-Laplace solve. 20M cells × ~0.087 s/cell parallel = ~480 CPU-hours per fit. The cheap-pass prune (c39a8df) already drops obvious-zero-mass cells, but the kept-fraction at this scale is the dominant unknown.

**Why it's hard.** Per-cell speedups <=10x are dwarfed by cell count. Cumulative micro-perf work (alloc-hoist -> DenseMat -> CSC cache, 2.9x serial) is asymptotically irrelevant if the count of cells grows faster than per-cell time shrinks.

**Open questions.**
- What makes the grid 20M? CCD gives O(2p+1) or O(2^p) — 20M implies either many hyperparameters, much denser sampling than CCD (full tensor product, importance points), or a fundamentally different integration scheme.
- What's the prune keep-rate at 20M? If 99% prune -> 200k effective; if 50% -> 10M.
- What's actually integrated at each cell? Just the normalizing constant Z, or marginal posteriors at every cell?
- Is CCD-on-coarse-grid + importance correction acceptable, or does the workload need every cell evaluated?

**Candidate levers (not yet picked).**
- GPU batched factor: 10^4-10^5 small arrow-structured Hessians factored in parallel on the 5080. cuSOLVER batched dense Cholesky + custom arrow kernel. Requires the arrow-exploit from earlier discussion (shrinks per-cell state into the regime where batched dense wins).
- Adaptive grid: refine only where integrand mass concentrates. Skew-normal or Pathfinder fit over hyperparameters as a replacement for the deterministic grid.
- Multi-level prune: cheap-pass already exists; add a medium-pass that runs 1-2 Newton iters per cell to early-reject before committing to full convergence.

---

## Issue 2 — Spatial dimension n_x ~ 20M

**What it means.** The inner Laplace solve is over a latent field of size n_x. If n_x = 20M, each Newton step requires factoring a 20M x 20M sparse Hessian. CHOLMOD supernodal scales superlinearly in nnz(L); for 2D problems direct factorization typically becomes infeasible (memory and time) above n_x ~ 10^6-10^7.

**Why it's hard.** This isn't a "speed up the existing factor" problem — it's a "direct factorization is the wrong algorithm at this scale" problem. The discussion shifts from CHOLMOD vs Pardiso to **direct vs iterative**.

**Open questions.**
- Is the latent field 1D (chain), 2D (lattice), or unstructured graph? Average degree of the adjacency?
- Is the prior precision highly sparse (5-9 nnz/row for ICAR/CAR), or filled-in (continuous-domain SPDE with thicker stencil)?
- Do we need the full posterior covariance (which entries?) or just marginal variances?
- Does the model admit a low-rank-plus-sparse split (e.g., fixed effects + spatial)?

**Candidate levers.**
- Iterative solver: PCG with algebraic multigrid (AMG) preconditioner. Memory linear in nnz, time near-linear for well-conditioned 2D problems.
- INLA-style selective inversion: marginal variances without factoring the full inverse.
- Domain decomposition: partition the latent field, solve subproblems with boundary coupling.
- Stochastic trace / log-det estimators (Hutchinson + Lanczos or Chebyshev).

---

## Issue 3 — 20M independent fits

**What it means.** Workflow runs the pipeline 20M times (per-species, per-pixel, per-cell-of-some-outer-grid). Each fit is "small" individually; the count is the problem.

**Why it's hard.** Per-fit micro-perf compounds linearly with count but cannot deliver more than constant-factor wins. The structural question is whether fits can share work (same X, same prior) or must remain independent.

**Open questions.**
- Are fits truly independent? Or do groups share design matrix X, prior precision Q, or hyperparameter posterior?
- Could the problem be reformulated as one multivariate fit with shared structure (joint hierarchical model with random effects per fit)?
- What's the embarrassingly-parallel ceiling — single node 32 threads, multi-node, cluster?

**Candidate levers.**
- Batched solves on GPU when fits share structure.
- Warm-start: order fits so neighbors share approximate posterior, init Newton from neighbor mode.
- Hierarchical reformulation: pool fits into one model with random effects; Laplace/sample once, get posterior for every "fit" out of the joint.

---

## How they combine

Cost is approximately:

    total = (cells per fit) x (per-cell solve time at n_x) x (number of fits)

Worked points:
- 20M x 0.087 s x 1   -> 480 h per fit                    (Issue 1 only)
- 100 x 100 s x 20M   -> infeasible                        (Issues 2 + 3)
- 1000 x 1 s x 20M    -> 20M CPU-hours                     (all three at moderate scale)

First thing to decide: **which of the three numbers is ~20M, and what are the other two?** That determines whether the right move is GPU batching (Issue 1 alone), iterative solvers (Issue 2 alone), hierarchical reformulation (Issue 3 alone), or a combined attack.
