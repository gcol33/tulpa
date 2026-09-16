# `cpp_laplace_fit()` was deleted (gcol33/tulpa#811): fixed effects plus at
# most one iid random intercept was a narrower door onto the same spec solver
# `cpp_laplace_fit_multi_re()` reaches, with no caller anywhere in R/. Before
# deletion, "cpp_laplace_fit agrees with cpp_laplace_fit_multi_re on a single
# iid RE" (test-laplace-spec-builtin-family.R) pinned the two marshalling
# paths to the same mode (tolerance 1e-6) at both K = 0 and K = 1.
#
# This reproduces the deleted entry's narrow interface for the tests that
# exercised it, routed through the production entry, so a length-K-0-or-1 RE
# spec still reaches the solver the same way it always fit through.
ref_laplace_fit_single <- function(y, n, X, re_idx = numeric(0),
                                   n_re_groups = 0L, sigma_re = 1.0,
                                   family, phi = 1.0, max_iter = 100L,
                                   tol = 1e-6, n_threads = 1L,
                                   compute_skew = FALSE, skew_idx = NULL) {
  has_re <- n_re_groups > 0L
  cpp_laplace_fit_multi_re(
    y = y, n = n, X = X,
    re_idx_list = if (has_re) list(as.integer(re_idx)) else list(),
    re_ngroups = if (has_re) as.integer(n_re_groups) else integer(0),
    re_sigma_list = if (has_re) list(sigma_re) else list(),
    family = family, phi = phi, max_iter = max_iter, tol = tol,
    n_threads = n_threads, compute_skew = compute_skew, skew_idx = skew_idx
  )
}

assign("ref_laplace_fit_single", ref_laplace_fit_single, envir = globalenv())
