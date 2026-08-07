// cell_coupling_probe.cpp
// Direct evaluation of a registered `CellCouplingSpec` at one cell, outside
// any Newton solve.
//
// The inner Newton reaches `evaluate_cell()` through
// `scatter_cell_coupling_branch_impl()` (src/nested_laplace_joint_multi.h),
// which immediately chains the per-arm eta derivatives through the design and
// scatters them into the joint gradient / Hessian -- so the eta-space
// quantities a spec actually writes are never visible from R. This export
// hands back exactly those quantities: the cell log-density, the per-arm
// gradient, the per-arm negative-Hessian diagonal and every dense cross-arm
// block the spec filled.
//
// It is the differencing surface for a finite-difference check of a spec's
// analytic derivatives (a spec whose gradient does not difference its own
// value, or whose cross-arm Hessian does not difference its own gradient, is
// wrong before any fit is attempted), and the same surface a third-derivative
// tensor differences.
//
// Buffer allocation mirrors the kernel exactly: which (kk, ll) dense slabs
// exist is read from the spec's own `dense_cross_pairs()`, and the rank-1
// self-cross descriptor is supplied, so a spec sees the same `CellDerivs`
// shape it sees under the driver.

#include "cell_coupling_registry.h"
#include "cell_curvature3.h"

#include <Rcpp.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

// [[Rcpp::export]]
Rcpp::List cpp_cell_coupling_evaluate(std::string name,
                                      Rcpp::List eta,
                                      Rcpp::List y,
                                      Rcpp::CharacterVector family,
                                      Rcpp::NumericVector phi,
                                      Rcpp::Nullable<Rcpp::List> n_trials = R_NilValue,
                                      int cell_idx = 0,
                                      bool grad_only = false) {
    auto spec = tulpa::lookup_cell_coupling(name);
    if (!spec) {
        Rcpp::stop("cell coupling spec '" + name + "' is not registered.");
    }
    const int n_arms = eta.size();
    if (y.size() != n_arms || family.size() != n_arms || phi.size() != n_arms) {
        Rcpp::stop("`eta`, `y`, `family` and `phi` must all have one entry "
                   "per coupled arm.");
    }

    // Backing storage for the whole cell: every supplied row belongs to it.
    std::vector<std::vector<double>> eta_store(n_arms), y_store(n_arms);
    std::vector<std::vector<int>>    trial_store(n_arms), rows_store(n_arms);
    std::vector<std::string>         family_store(n_arms);
    std::vector<int>                 row_count(n_arms);

    Rcpp::List nt = n_trials.isNotNull() ? Rcpp::List(n_trials) : Rcpp::List();
    for (int k = 0; k < n_arms; k++) {
        Rcpp::NumericVector ek = eta[k];
        Rcpp::NumericVector yk = y[k];
        const int rc = ek.size();
        if (yk.size() != rc) {
            Rcpp::stop("arm " + std::to_string(k + 1) +
                       ": length(y) must equal length(eta).");
        }
        eta_store[k].assign(ek.begin(), ek.end());
        y_store[k].assign(yk.begin(), yk.end());
        trial_store[k].assign(rc, 1);
        if (nt.size() == n_arms) {
            Rcpp::IntegerVector tk = nt[k];
            if (tk.size() == rc) trial_store[k].assign(tk.begin(), tk.end());
        }
        rows_store[k].resize(rc);
        for (int j = 0; j < rc; j++) rows_store[k][j] = j;
        family_store[k] = Rcpp::as<std::string>(family[k]);
        row_count[k] = rc;
    }

    std::vector<const double*> eta_ptr(n_arms), y_ptr(n_arms);
    std::vector<const int*>    trial_ptr(n_arms), rows_ptr(n_arms);
    std::vector<const char*>   family_ptr(n_arms);
    for (int k = 0; k < n_arms; k++) {
        eta_ptr[k]    = eta_store[k].data();
        y_ptr[k]      = y_store[k].data();
        trial_ptr[k]  = trial_store[k].data();
        rows_ptr[k]   = rows_store[k].data();
        family_ptr[k] = family_store[k].c_str();
    }

    std::vector<std::vector<double>> grad_buf(n_arms), diag_buf(n_arms),
                                     rank1_vec_buf(n_arms);
    std::vector<double*> grad_ptr(n_arms), diag_ptr(n_arms), rank1_vec_ptr(n_arms);
    for (int k = 0; k < n_arms; k++) {
        grad_buf[k].assign(row_count[k], 0.0);
        diag_buf[k].assign(row_count[k], 0.0);
        rank1_vec_buf[k].assign(std::max(row_count[k], 1), 0.0);
        grad_ptr[k]      = grad_buf[k].data();
        diag_ptr[k]      = diag_buf[k].data();
        rank1_vec_ptr[k] = rank1_vec_buf[k].data();
    }
    std::vector<double> rank1_coef(n_arms, 0.0);

    // Dense cross slabs, allocated for exactly the pairs the spec declares --
    // the same policy scatter_cell_coupling_branch_impl() applies.
    std::vector<std::vector<std::vector<double>>> cross_buf(
        n_arms, std::vector<std::vector<double>>(n_arms));
    std::vector<std::vector<double*>> cross_inner(
        n_arms, std::vector<double*>(n_arms, nullptr));
    std::vector<double* const*> cross_outer(n_arms, nullptr);
    for (int k = 0; k < n_arms; k++) cross_outer[k] = cross_inner[k].data();
    for (const auto& pr :
         spec->dense_cross_pairs(n_arms, /*rank1_self_supported=*/true)) {
        const int kk = std::min(pr.first, pr.second);
        const int ll = std::max(pr.first, pr.second);
        if (kk < 0 || ll >= n_arms) continue;
        cross_buf[kk][ll].assign(
            (std::size_t)row_count[kk] * (std::size_t)row_count[ll], 0.0);
        cross_inner[kk][ll] = cross_buf[kk][ll].data();
    }

    tulpa::CellEtas etas_view;
    etas_view.arm_eta_ptr   = eta_ptr.data();
    etas_view.arm_rows      = rows_ptr.data();
    etas_view.arm_row_count = row_count.data();
    etas_view.n_arms_       = n_arms;

    tulpa::CellResponse y_view;
    y_view.arm_y         = y_ptr.data();
    y_view.arm_n_trials  = trial_ptr.data();
    y_view.arm_family    = family_ptr.data();
    y_view.arm_phi       = REAL(phi);
    y_view.arm_rows      = rows_ptr.data();
    y_view.arm_row_count = row_count.data();
    y_view.n_arms_       = n_arms;

    tulpa::CellDerivs out;
    out.arm_grad             = grad_ptr.data();
    out.arm_neg_hess_diag    = diag_ptr.data();
    out.arm_cross_hess       = cross_outer.data();
    out.arm_row_count        = row_count.data();
    out.n_arms_              = n_arms;
    out.grad_only            = grad_only;
    out.arm_cross_rank1_coef = rank1_coef.data();
    out.arm_cross_rank1_vec  = rank1_vec_ptr.data();

    const double value = spec->evaluate_cell(cell_idx, etas_view, y_view, out);

    Rcpp::List grad_out(n_arms), diag_out(n_arms), rank1_out(n_arms),
               cross_out(n_arms);
    for (int k = 0; k < n_arms; k++) {
        grad_out[k]  = Rcpp::NumericVector(grad_buf[k].begin(), grad_buf[k].end());
        diag_out[k]  = Rcpp::NumericVector(diag_buf[k].begin(), diag_buf[k].end());
        rank1_out[k] = Rcpp::NumericVector(rank1_vec_buf[k].begin(),
                                           rank1_vec_buf[k].begin() + row_count[k]);
        Rcpp::List row_k(n_arms);
        for (int l = 0; l < n_arms; l++) {
            if (cross_inner[k][l] == nullptr) { row_k[l] = R_NilValue; continue; }
            const int rc_k = row_count[k], rc_l = row_count[l];
            Rcpp::NumericMatrix M(rc_k, rc_l);
            for (int j = 0; j < rc_k; j++)
                for (int m = 0; m < rc_l; m++)
                    M(j, m) = cross_inner[k][l][(std::size_t)j * rc_l + m];
            row_k[l] = M;
        }
        cross_out[k] = row_k;
    }

    std::vector<int> ids = spec->arm_ids();
    return Rcpp::List::create(
        Rcpp::Named("value")         = value,
        Rcpp::Named("grad")          = grad_out,
        Rcpp::Named("neg_hess_diag") = diag_out,
        Rcpp::Named("cross_hess")    = cross_out,
        Rcpp::Named("rank1_coef")    = Rcpp::NumericVector(rank1_coef.begin(),
                                                           rank1_coef.end()),
        Rcpp::Named("rank1_vec")     = rank1_out,
        Rcpp::Named("arm_ids")       = Rcpp::IntegerVector(ids.begin(), ids.end())
    );
}

// The cubic contraction of a registered spec's cell third-derivative tensor
// against one direction (gcol33/tulpa#301), on one cell, outside any solve.
//
// The engine reaches this through the joint Newton's inner-skew probe, where the
// direction is the eta response to v_i = Sigma e_i and the result is one term of
// gamma_3's numerator. Exposed here at cell granularity so the contraction can be
// held against an independent third derivative of the spec's own `value`, and so
// the per-arm step policy can be measured against the global one rather than
// asserted. `eta` is the cell's per-arm linear predictor, `u` the per-arm
// direction (same shapes).
// [[Rcpp::export]]
double cpp_cell_coupling_curvature3(std::string name,
                                    Rcpp::List eta,
                                    Rcpp::List u,
                                    Rcpp::List y,
                                    Rcpp::CharacterVector family,
                                    Rcpp::NumericVector phi,
                                    int cell_idx = 0,
                                    bool per_arm_step = true) {
    auto spec = tulpa::lookup_cell_coupling(name);
    if (!spec) {
        Rcpp::stop("cell coupling spec '" + name + "' is not registered.");
    }
    const int n_arms = eta.size();
    if (u.size() != n_arms || y.size() != n_arms || family.size() != n_arms ||
        phi.size() != n_arms) {
        Rcpp::stop("`eta`, `u`, `y`, `family` and `phi` must all have one entry "
                   "per coupled arm.");
    }

    // Cell `cell_idx` holds every supplied row and every earlier cell is empty,
    // so a spec that reads per-cell metadata sees the index it was asked for
    // while the empty cells contribute exactly nothing (their direction is zero).
    if (cell_idx < 0) Rcpp::stop("cell_idx must be non-negative.");
    std::vector<std::vector<std::vector<int>>> cell_rows(n_arms);
    std::vector<Rcpp::NumericVector> eta0(n_arms), eta1(n_arms);
    std::vector<std::vector<double>> y_store(n_arms);
    std::vector<std::string>         family_store(n_arms);
    tulpa::CellCurvature3Inputs in;
    in.spec         = spec.get();
    in.n_cells      = cell_idx + 1;
    in.per_arm_step = per_arm_step;
    in.coupled_arms.resize(n_arms);
    in.arm_y.resize(n_arms);
    in.arm_n_trials.assign(n_arms, nullptr);
    in.arm_family.resize(n_arms);
    in.arm_phi.assign(phi.begin(), phi.end());
    for (int k = 0; k < n_arms; k++) {
        Rcpp::NumericVector ek = eta[k];
        Rcpp::NumericVector uk = u[k];
        Rcpp::NumericVector yk = y[k];
        const int rc = ek.size();
        if (uk.size() != rc || yk.size() != rc) {
            Rcpp::stop("arm " + std::to_string(k + 1) +
                       ": length(u) and length(y) must equal length(eta).");
        }
        eta0[k] = Rcpp::NumericVector(ek.begin(), ek.end());
        eta1[k] = Rcpp::NumericVector(rc);
        for (int j = 0; j < rc; j++) eta1[k][j] = ek[j] + uk[j];
        y_store[k].assign(yk.begin(), yk.end());
        family_store[k] = Rcpp::as<std::string>(family[k]);
        cell_rows[k].assign(cell_idx + 1, std::vector<int>());
        cell_rows[k][cell_idx].resize(rc);
        for (int j = 0; j < rc; j++) cell_rows[k][cell_idx][j] = j;
        in.coupled_arms[k] = k;
        in.arm_y[k]        = y_store[k].data();
        in.arm_family[k]   = family_store[k];
    }
    in.cell_rows = &cell_rows;

    const char* reason = "";
    tulpa::CellCubic3Fn fn = tulpa::build_cell_curvature3_tensor(in, &reason);
    if (!fn) Rcpp::stop("cell curvature3 tensor declined: %s", reason);
    return fn(eta0, eta1);
}
