// cholmod_stub_resolve.cpp
// Resolves every Matrix CHOLMOD stub tulpa calls, once, on the R main thread
// while the DLL loads.
//
// Each M_cholmod_* stub in <Matrix/stubs.c> looks its target up with
// R_GetCCallable() on its first call and caches the pointer in a function-local
// static. R_GetCCallable() protects and unprotects on the R protect stack, so a
// first call made from an OpenMP worker (a per-cell solver in an outer-grid or
// inner-vcov parallel loop) races the main thread's R_PPStackTop and leaves it
// off by one. Running one small factor-and-solve through every stub here fills
// every cache before any parallel region can reach a stub.
//
// The calls below must cover every M_cholmod_* name used anywhere in src/;
// tests/testthat/test-cholmod-stub-resolve.R asserts it.

#include "sparse_cholesky.h"

namespace tulpa {

void cholmod_resolve_stubs() {
    cholmod_common c;
    M_cholmod_start(&c);
    c.print = 0;
    c.error_handler = nullptr;

    // The 1 x 1 SPD matrix [2], lower triangle stored.
    cholmod_sparse* A = M_cholmod_allocate_sparse(1, 1, 1, 1, 1, -1,
                                                  CHOLMOD_REAL, &c);
    if (A) {
        static_cast<int*>(A->p)[0] = 0;
        static_cast<int*>(A->p)[1] = 1;
        static_cast<int*>(A->i)[0] = 0;
        static_cast<double*>(A->x)[0] = 2.0;

        cholmod_factor* L = M_cholmod_analyze(A, &c);
        if (L && M_cholmod_factorize(A, L, &c)) {
            double rhs = 1.0;
            cholmod_dense b;
            b.nrow  = 1;
            b.ncol  = 1;
            b.nzmax = 1;
            b.d     = 1;
            b.x     = &rhs;
            b.z     = nullptr;
            b.xtype = CHOLMOD_REAL;
            b.dtype = CHOLMOD_DOUBLE;

            cholmod_dense* x = M_cholmod_solve(CHOLMOD_A, L, &b, &c);
            M_cholmod_free_dense(&x, &c);

            cholmod_dense *X = nullptr, *Y = nullptr, *E = nullptr;
            M_cholmod_solve2(CHOLMOD_A, L, &b, &X, &Y, &E, &c);
            M_cholmod_free_dense(&X, &c);
            M_cholmod_free_dense(&Y, &c);
            M_cholmod_free_dense(&E, &c);

            (void) M_cholmod_factor_ldetA(L);
            M_cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, &c);
        }
        M_cholmod_free_factor(&L, &c);
        M_cholmod_free_sparse(&A, &c);
    }
    M_cholmod_finish(&c);
}

}  // namespace tulpa
