# Inference Mode System

tulpa uses an explicit tier system for inference that encodes
**epistemic guarantees**, not just runtime characteristics.

This design makes the difference between inference methods **first-class
and unavoidable**, rather than hiding them as implementation details.

## The Three Tiers

**Tier 1 - Exact:** Asymptotically correct posterior inference (up to
Monte Carlo error). Credible intervals are interpretable as posterior
uncertainty. This is the reference standard.

**Tier 2 - Structured:** Accurate inference *conditional on explicit
structural assumptions*. Typically requires latent Gaussian structure,
conditional independence, smooth posteriors. Very fast for the right
model class, but can be wrong outside that class. Failure modes are
predictable and explainable.

**Tier 3 - Optimized:** No general correctness guarantee beyond
empirical usefulness. Point estimates usually good, but uncertainty
often underestimated. Tails and correlations unreliable. Failure is
usually silent. This is optimization, not sampling.

## Auto Mode

`mode = "auto"` chooses between Tier 1 and Tier 2 only. It will
**never** silently choose Tier 3 (Optimized).

The contract: "Use the most reliable method that is expected to finish
for this model."

Auto decisions are deterministic, explainable, and overrideable.

## Implementation Rules

1.  Modes change semantics, not just runtime. Intervals from Optimized
    do not mean the same as Exact.

2.  The mode must always be visible in output.

3.  No silent upgrading or downgrading. If Exact fails, we error - we do
    not switch to Structured.

4.  Backends slot into tiers. Adding a backend never introduces a new
    epistemic promise.
