// cholmod_view.h
// Stack-resident cholmod_sparse view over caller-owned CSC arrays.
//
// Nothing is allocated, copied or freed: the returned struct points straight at
// the caller's `col_ptr` / `row_idx` / `values` and is valid only while those
// live. The layout is the one every tulpa Cholesky call site uses -- square,
// symmetric with the LOWER triangle stored (`stype = -1`), sorted, packed,
// real, double, 32-bit indices.

#ifndef TULPA_CHOLMOD_VIEW_H
#define TULPA_CHOLMOD_VIEW_H

#include <Matrix/cholmod.h>
#include <cstddef>

namespace tulpa {

inline cholmod_sparse cholmod_lower_view(
    std::size_t   n,
    const int*    col_ptr,
    const int*    row_idx,
    const double* values,
    std::size_t   nnz
) {
    cholmod_sparse A;
    A.nrow   = n;
    A.ncol   = n;
    A.nzmax  = nnz;
    A.p      = const_cast<int*>(col_ptr);
    A.i      = const_cast<int*>(row_idx);
    // Read only when packed == 0; set so the struct carries no indeterminate
    // field into CHOLMOD.
    A.nz     = nullptr;
    A.x      = const_cast<double*>(values);
    A.z      = nullptr;
    A.stype  = -1;
    A.itype  = CHOLMOD_INT;
    A.xtype  = CHOLMOD_REAL;
    A.dtype  = CHOLMOD_DOUBLE;
    A.sorted = 1;
    A.packed = 1;
    return A;
}

} // namespace tulpa

#endif // TULPA_CHOLMOD_VIEW_H
