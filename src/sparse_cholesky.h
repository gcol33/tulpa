// sparse_cholesky.h
// CHOLMOD sparse Cholesky solver wrapper for tulpa
// Feature 1: Replaces dense O(N^3) Cholesky for large Laplace problems
//
// Uses CHOLMOD via R's Matrix package C API (stubs.c pattern).
// Supernodal algorithm: converts sparse problem to many small dense BLAS ops.
// Symbolic-once / numeric-many: critical for nested Laplace where the
// sparsity pattern stays fixed across hyperparameter grid points.

#ifndef TULPA_SPARSE_CHOLESKY_H
#define TULPA_SPARSE_CHOLESKY_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include "laplace_types.h"  // for tulpa::DenseMat

// CHOLMOD types from Matrix package.
// Matrix/cholmod.h defines cholmod_sparse, cholmod_factor, etc.
// Matrix/cholmod-utils.h declares M_cholmod_factor_ldetA etc.
// The actual function stubs (stubs.c) are included in sparse_cholesky.cpp.
#include <Matrix/cholmod.h>
#include <Matrix/cholmod-utils.h>

namespace tulpa {

// =====================================================================
// SparseCholeskySolver: thin RAII wrapper around CHOLMOD
// =====================================================================
//
// Usage pattern:
//   SparseCholeskySolver solver;
//   solver.analyze(A);           // symbolic factorization (once)
//   solver.factorize(A);         // numeric factorization (per iteration)
//   solver.solve(b, x, n);       // solve Ax = b
//   double ld = solver.log_determinant();  // log|A|
//
// The sparsity pattern of A must not change between analyze() and
// subsequent factorize() calls. Only the numeric values may change.

class SparseCholeskySolver {
public:
    SparseCholeskySolver();
    ~SparseCholeskySolver();

    // Non-copyable (owns CHOLMOD resources)
    SparseCholeskySolver(const SparseCholeskySolver&) = delete;
    SparseCholeskySolver& operator=(const SparseCholeskySolver&) = delete;

    // Phase 1: Symbolic analysis. Determines fill-reducing ordering and
    // allocates the factor. Done once per sparsity pattern.
    // A must be symmetric (stype != 0) or will be treated as A+A'.
    void analyze(cholmod_sparse* A);

    // Phase 2: Numeric factorization. Computes L such that PAP' = LL'.
    // Returns true on success, false if matrix is not positive definite.
    // A must have the same sparsity pattern as in analyze().
    //
    // Rank-deficient handling lives upstream, not here: every Laplace
    // factorize callsite adds `LAPLACE_UNIFORM_RIDGE * I` to H before
    // calling factorize (see laplace_cholesky.h). That guarantees the
    // matrix is PD even for ICAR / RW1 / RW2 priors with rank deficit,
    // so this method only needs the plain supernodal path — no
    // simplicial fallback, no dbound, no per-call ridge retry.
    bool factorize(cholmod_sparse* A);

    // Solve Ax = b using the current factorization.
    // b and x are dense vectors of length n.
    //
    // Returns false when there is no usable factorization (factorize() has not
    // run, or did not succeed) or CHOLMOD's solve fails, and fills x with NaN
    // rather than zero. A zero vector solves Ax = b only for a zero b; a Newton
    // loop testing max|delta| < tol reads a zero step as convergence, so a
    // failed solve filled with zeros is reported as a converged fit. NaN trips
    // the finiteness checks the callers already run, and the return value makes
    // the failure readable without one.
    bool solve(const double* b, double* x, int n);

    // Reusable scratch for the workspace-holding solve below.
    //
    // CHOLMOD's cholmod_solve allocates the dense solution plus two internal
    // work blocks and frees them on every call. cholmod_solve2 keeps all three
    // in caller-held handles instead, so a run of back-solves against one
    // factor -- the per-observation predictive variances of a nested-Laplace
    // cell are thousands of them -- allocates on the first solve and reuses
    // thereafter.
    //
    // The handles are CHOLMOD objects owned by ONE solver's cholmod_common, so
    // a workspace binds to the first solver it is used with and must not
    // outlive it; a solve through a different solver is refused rather than
    // freeing a handle against a foreign common. cholmod_common is not
    // reentrant, so a workspace belongs to the same thread as its solver: give
    // each worker its own.
    class SolveWorkspace {
    public:
        SolveWorkspace() = default;
        ~SolveWorkspace();

        // Non-copyable (owns CHOLMOD resources)
        SolveWorkspace(const SolveWorkspace&) = delete;
        SolveWorkspace& operator=(const SolveWorkspace&) = delete;

        // Free the handles and unbind, so the workspace can be reused with a
        // different solver. Also what the destructor runs.
        void release();

    private:
        friend class SparseCholeskySolver;
        cholmod_dense* X_ = nullptr;   // solution, sized by cholmod_solve2
        cholmod_dense* Y_ = nullptr;   // CHOLMOD internal workspace
        cholmod_dense* E_ = nullptr;   // CHOLMOD internal workspace
        cholmod_common* common_ = nullptr;  // borrowed from the bound solver
    };

    // Solve Ax = b through cholmod_solve2, holding CHOLMOD's dense solution and
    // internal workspace in `ws` across calls. Same contract as the allocating
    // solve above: x is filled with NaN and false returned when there is no
    // usable factorization or CHOLMOD's solve fails, and additionally when `ws`
    // is already bound to a different solver.
    bool solve(const double* b, double* x, int n, SolveWorkspace& ws);

    // Apply the inverse Cholesky square root of A to `ncol` standard normal
    // vectors at once: with A = P' L L' P, the columns of `x = P' L^-T eps`
    // have covariance A^-1, so a caller holding a factored Hessian can draw
    // from the Gaussian it defines with no refactorization and no dense
    // triangle. `eps` and `x` are column-major [n x ncol] and may alias.
    //
    // Only an LL' factor carries a square root reachable this way. CHOLMOD's
    // CHOLMOD_DLt solve gives D^-1 where a draw needs D^-1/2, so a simplicial
    // LDL' factor is refused (returns false) rather than served draws from the
    // wrong covariance. The supernodal factorization this class requests is
    // always LL', so the refusal is the fallback path's, not the usual one's.
    bool apply_inv_chol_factor(const double* eps, double* x, int n, int ncol);

    // Log-determinant of the factored matrix: log|A| = log|LL'| = 2*sum(log(diag(L))).
    // Uses Matrix package's M_cholmod_factor_ldetA which handles both
    // simplicial and supernodal factors correctly.
    double log_determinant() const;

    // Selected inversion (Takahashi equations): compute diagonal of A^{-1}
    // from the Cholesky factor. Costs what the recursion below costs; see the
    // bound stated over takahashi_partial_inverse_csc().
    // Converts factor to simplicial LL' if currently supernodal.
    // Returns empty vector on failure, a failed conversion included.
    std::vector<double> selected_inversion_diagonal();

    // Full selected inversion (Takahashi equations): the partial inverse
    // Z = A^{-1} computed on pattern(L + L^T), the fill-in superset of A's
    // sparsity pattern. Returns a lookup keyed by ORIGINAL (pre-permutation)
    // (i, j): every (i, j) on A's nonzero pattern is present, since A's
    // pattern is a subset of the factor's pattern. Costs what the recursion
    // below costs; see the bound stated over takahashi_partial_inverse_csc().
    // Converts factor to simplicial LL' if currently supernodal, and returns a
    // default-constructed SelectedInverse (n == 0) when that conversion fails.
    //
    // The conversion is in place and permanent: both selected-inversion entry
    // points leave the solver holding a simplicial factor, so every later
    // solve() and log_determinant() on the same object runs the simplicial
    // path that analyze() may not have chosen. A caller that needs the
    // supernodal factor for later solves must re-analyze().
    struct SelectedInverse {
        int n = 0;
        std::vector<int> Lp;            // factor column pointers (permuted space)
        std::vector<int> Li;            // factor row indices (permuted space)
        std::vector<double> Zx;         // Takahashi partial inverse, pattern(L)
        std::vector<int> perm_inv;      // original index -> permuted index

        // A^{-1}_{ij} in original ordering. Returns 0.0 if (i, j) is off the
        // computed pattern (which never happens for (i, j) on A's pattern).
        double at(int i_orig, int j_orig) const;
    };
    SelectedInverse selected_inversion_full();

    // Whether analyze() has been called
    bool analyzed() const { return analyzed_; }

    // Whether factorize() succeeded
    bool factored() const { return factored_; }

    // Bytes the numeric Cholesky factor will occupy once factorize() runs, read
    // from the analyzed symbolic factor -- no numeric factorization required.
    // For a supernodal factor this is L->xsize doubles (the dense supernode
    // values factorize() allocates) plus the persistent integer supernode
    // structure; for a simplicial factor it is L->nzmax * (double + int) plus
    // the column pointers. Returns 0 if analyze() has not produced a factor.
    // Lets a memory budget size the per-solve factor from its true fill-in
    // rather than a guessed multiple of nnz(A) (2D-mesh fill-in is superlinear).
    std::size_t analyzed_factor_bytes() const;

    // Access the cholmod_common (for advanced use)
    cholmod_common& common() { return common_; }

    // Refill an owned cholmod_sparse with values from a dense lower-triangle
    // Hessian H. On the FIRST call the sparsity pattern is discovered using
    // the supplied drop_tol (entries with |H[i][j]| <= drop_tol are dropped
    // off-diagonal), then cached for the lifetime of the solver. On
    // subsequent calls the pattern is reused: only Ax is overwritten, which
    // removes the per-iter malloc/free of cholmod_sparse that dominates the
    // inner Newton loop at n_x ~ 800 with banded H.
    //
    // The returned pointer is owned by the solver and stable across calls.
    // Callers must NOT M_cholmod_free_sparse() it.
    //
    // The discovery is numeric, so the first H's VALUES decide the pattern,
    // and that is a property of one iterate rather than of the model. Every
    // call therefore walks the whole lower triangle against the cached
    // pattern; an entry off it whose |H[i][j]| now exceeds drop_tol GROWS the
    // pattern (union with the cached one, so growth is monotone) and drops the
    // analyzed factor, which the dispatch rebuilds because analyzed() then
    // reads false. Discarding the entry instead would factor a matrix missing
    // that curvature, and the Newton direction, log|H| and the standard errors
    // would all inherit it. Only a failed reallocation falls back to
    // discarding, and that path records record_hessian_pattern_drop() so the
    // enclosing driver's HessianPatternGuard still raises.
    //
    // A reset() call clears the cached pattern outright, for a solver reused
    // across genuinely different models.
    cholmod_sparse* refill_from_dense(const DenseMat& H, int n, double drop_tol);

    // Drop the analyzed factor + cached sparse pattern. Next refill_from_dense
    // / analyze cycle will re-discover the pattern. Used when reusing a solver
    // across genuinely different sparsity structures.
    void reset();

private:
    // Allocate A_owned_ over the union of `keep`'s pattern (null for the first
    // discovery) and H's above-tolerance entries, fill the values, and drop any
    // analyzed factor. False on allocation failure, leaving A_owned_ untouched.
    bool build_owned_pattern(const DenseMat& H, int n, double drop_tol,
                             cholmod_sparse* keep);

    cholmod_common common_;
    cholmod_factor* factor_;
    cholmod_sparse* A_owned_;
    bool analyzed_;
    bool factored_;
};

// =====================================================================
// Free functions: Takahashi recursion on a caller-provided L (CSC, lower-tri)
//
// These do not require a SparseCholeskySolver / CHOLMOD factorization — the
// caller supplies L directly. Used by downstream packages that own their own
// Cholesky (e.g. via Matrix::Cholesky in R) but want the partial-inverse
// recursion in C++.
//
// L is lower-triangular CSC: column j stores rows j..n-1 with L[j,j] in
// position Lp[j] (first slot of the column). The recursion produces Z =
// Q^{-1} on pattern(L); off-pattern entries of the true Q^{-1} are not
// computed.
//
// Preconditions on L, all three load-bearing and all three checked at entry
// (one pass over Lp / Li, cheap against the recursion itself) because these
// are offered to callers holding a factor tulpa did not produce:
//   - L[j,j] occupies the first slot of column j, i.e. Li[Lp[j]] == j;
//   - row indices ascend strictly within a column, which the entry lookup
//     relies on to stop early;
//   - no duplicate row indices in a column.
// A factor violating any of them produces silently wrong output otherwise, so
// the recursion refuses it instead: false is returned and the output is left
// zeroed. A Matrix::Cholesky simplicial factor (dCHMsimpl) satisfies all three.
//
// Cost is sum_j |col_j| * (|col_j| + average scan length within the searched
// column), not O(nnz(L)): the inner entry lookup is a linear scan over another
// column.
// =====================================================================

// Fill Zx_out (size = Lp[n]) with the Takahashi partial inverse on pattern(L).
// Indexing matches L: Zx_out[idx] holds Z[Li[idx], j] for idx in [Lp[j], Lp[j+1]).
// Returns false, leaving Zx_out zeroed, when a precondition above fails or a
// pivot L[j,j] is too small to divide by; the column-wise recursion cannot be
// completed past such a pivot, and reporting the columns that were reachable
// would hand back zeros where a caller reads marginal variances.
bool takahashi_partial_inverse_csc(
    int n,
    const int* Lp,
    const int* Li,
    const double* Lx,
    double* Zx_out
);

// Fill Z_out (size = n*n, column-major) with Z = Q^{-1} on pattern(L + L^T).
// Off-pattern entries are zero. Symmetrises the lower triangle to upper.
// Caller is responsible for allocating Z_out; it is fully overwritten.
// Returns false, leaving Z_out zeroed, on the same failures as the CSC form.
bool takahashi_partial_inverse_dense(
    int n,
    const int* Lp,
    const int* Li,
    const double* Lx,
    double* Z_out
);

// =====================================================================
// CSC conversion: DenseMat -> cholmod_sparse
// =====================================================================

// Convert a dense symmetric matrix (stored as vector-of-vector) to
// CHOLMOD CSC format. Only the lower triangle is stored (stype = -1).
// The returned sparse matrix must be freed with cholmod_free_sparse.
cholmod_sparse* dense_to_cholmod_sparse(
    const DenseMat& H, int n,
    cholmod_common* common
);

// Same but only stores entries with |value| > drop_tol.
// This produces a genuinely sparse matrix from a dense H that has
// structural sparsity (e.g., ICAR Hessians are banded).
cholmod_sparse* dense_to_cholmod_sparse_drop(
    const DenseMat& H, int n,
    double drop_tol,
    cholmod_common* common
);

} // namespace tulpa

#endif // TULPA_SPARSE_CHOLESKY_H
