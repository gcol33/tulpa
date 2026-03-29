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
    bool factorize(cholmod_sparse* A);

    // Solve Ax = b using the current factorization.
    // b and x are dense vectors of length n.
    void solve(const double* b, double* x, int n);

    // Log-determinant of the factored matrix: log|A| = log|LL'| = 2*sum(log(diag(L))).
    // Uses Matrix package's M_cholmod_factor_ldetA which handles both
    // simplicial and supernodal factors correctly.
    double log_determinant() const;

    // Selected inversion (Takahashi equations): compute diagonal of A^{-1}
    // from the Cholesky factor. O(nnz(L)) complexity.
    // Converts factor to simplicial LL' if currently supernodal.
    // Returns empty vector on failure.
    std::vector<double> selected_inversion_diagonal();

    // Whether analyze() has been called
    bool analyzed() const { return analyzed_; }

    // Whether factorize() succeeded
    bool factored() const { return factored_; }

    // Access the cholmod_common (for advanced use)
    cholmod_common& common() { return common_; }

private:
    cholmod_common common_;
    cholmod_factor* factor_;
    bool analyzed_;
    bool factored_;
};

// =====================================================================
// CSC conversion: DenseMat -> cholmod_sparse
// =====================================================================

// Convert a dense symmetric matrix (stored as vector-of-vector) to
// CHOLMOD CSC format. Only the lower triangle is stored (stype = -1).
// The returned sparse matrix must be freed with cholmod_free_sparse.
cholmod_sparse* dense_to_cholmod_sparse(
    const std::vector<std::vector<double>>& H, int n,
    cholmod_common* common
);

// Same but only stores entries with |value| > drop_tol.
// This produces a genuinely sparse matrix from a dense H that has
// structural sparsity (e.g., ICAR Hessians are banded).
cholmod_sparse* dense_to_cholmod_sparse_drop(
    const std::vector<std::vector<double>>& H, int n,
    double drop_tol,
    cholmod_common* common
);

} // namespace tulpa

#endif // TULPA_SPARSE_CHOLESKY_H
