# Upper end of the count support the family density tests sum over. A
# normalization, moment or expected-curvature identity is read off a finite
# sum, so this bound decides how much of the true tail each sum leaves out --
# and nothing else about the check: a wrong mixture constant or a dropped
# truncation term moves the sum by the size of the error whatever the bound.
#
# Measured over every family the tests sum (poisson, neg_binomial_1,
# neg_binomial_2 and both zero-truncated families, with and without zero
# inflation) at every grid point they use, eta in {-1, 0, 1.5}, z in
# {-1.5, 0, 0.8}, phi in {0.5, 2, 8}, the mass left beyond the bound is at
# most 2.2e-15 at 300, 8.6e-48 at 1000 and 7.1e-94 at 2000. The heaviest tail
# is neg_binomial_1 at phi = 8, where each step keeps 8/9 of the last. The
# tightest tolerance the sums are held to is 1e-9, so 2000 leaves them
# numerically identical to a sum to 20000 at a tenth of the cost.
SUPPORT_YMAX <- 2000L
