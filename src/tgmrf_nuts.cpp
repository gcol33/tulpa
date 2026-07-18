// tgmrf_nuts.cpp
// Joint NUTS over (beta, z, theta) for a tgmrf C++-backend latent block.
//
// =============================================================================
// Scope and design
// =============================================================================
//
// This is the **C++-backend exclusive** Tier-1 joint sampler for tgmrf(): the
// outer-theta companion lives in R/tgmrf_nuts.R (`tulpa_tgmrf_nuts`) and walks
// only the marginal-theta posterior with a finite-difference gradient on
// log_marginal(theta). The joint sampler here integrates the full vector
// (beta, z, theta) in C++ using leapfrog with explicit, closed-form gradients
// for (beta, z) and a finite-difference gradient on (Q, mu) wrt theta.
//
// The R wrapper (R/tgmrf_nuts_joint.R) refuses any block whose backend is not
// "cpp" -- calling R for Q(theta) at every leapfrog step would dwarf any
// statistical-cost advantage the joint sampler has. Users wanting joint
// behaviour on an R-closure block already have tulpa_tgmrf_imh() (Laplace +
// MH-correction over theta) and tulpa_tgmrf_nuts() (outer-theta NUTS).
//
// =============================================================================
// Math
// =============================================================================
//
// Let eta = X beta + Z_z z where Z_z is the latent indicator matrix induced by
// the block's obs_idx mapping (obs i -> latent slot obs_idx[i]). The joint
// log-posterior is
//
//   log p(beta, z, theta | y) = sum_i log p(y_i | eta_i)
//                             + 0.5 logdet Q(theta)
//                             - 0.5 (z - mu)' Q(theta) (z - mu)
//                             - 0.5 n_lat log(2 pi)
//                             + log p(theta)        # user prior
//                             # log p(beta) = const (flat)  [P3 v1]
//
// Gradients used by the leapfrog:
//
//   d / d beta_k     = sum_i X[i, k] * s_i                    s_i := d log p(y_i | eta_i) / d eta_i
//   d / d z_j        = s_{i_of(j)}                            if z_j is touched by exactly one obs
//                      (general: sum over obs with obs_idx[i] == j+1)
//                      - (Q (z - mu))_j
//   d / d theta_m    = 0.5 tr(Q^{-1} dQ/dtheta_m)
//                      - 0.5 (z - mu)' dQ/dtheta_m (z - mu)
//                      + (dmu/dtheta_m)' Q (z - mu)
//                      + d log p(theta) / d theta_m
//
// dQ/dtheta_m and dmu/dtheta_m and d log_prior / d theta_m are computed by
// central finite differences on the registered C++ kernels (Q_double, mu_double,
// log_prior_double of TgmrfSpec). theta_dim is typically 2-5, so 2 * theta_dim
// extra Q evaluations per leapfrog step is cheap.
//
// The trace term tr(Q^{-1} dQ/dtheta_m) is computed exactly via one sparse
// Cholesky of Q (we already do this for logdet) plus one sparse solve per
// dQ/dtheta_m -- for n_latent <= 500 this is cheap.
//
// =============================================================================
// Why not route through the existing ModelData/ParamLayout NUTS?
// =============================================================================
//
// The existing src/hmc_nuts_*.cpp leapfrog assumes a ModelData/ParamLayout
// owned by the model package (tulpaRatio, tulpaObs). Plumbing a tgmrf block into
// that layout would mean inventing a new "tgmrf-only" ParamLayout and a custom
// log_post_impl, which is more invasive than writing a focused 400-line driver
// targeted exactly at this problem. The dispatch surface tulpa exposes to
// users is (Laplace, IMH, VI, outer-theta NUTS, joint NUTS) per backend; the
// joint NUTS path is the last one and intentionally compact.

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "tulpa/tgmrf.h"
#include "tgmrf_registry.h"
#include "laplace_family_link.h"

namespace {

// ---------------------------------------------------------------------------
// State held across the sampler. q is the full joint parameter vector laid
// out as [beta (p), z (n_lat), theta (d)]; X is row-major N x p.
// ---------------------------------------------------------------------------
struct JointState {
    int N;
    int p;
    int n_lat;
    int d;                                                  // theta_dim

    // Data
    Rcpp::NumericVector y;
    Rcpp::IntegerVector n_trials;
    Eigen::MatrixXd X;                                      // N x p
    Eigen::VectorXi obs_idx_0;                              // length N, 0-based latent slot per obs

    // For each latent slot j, the list of observation indices that touch j.
    std::vector<std::vector<int>> obs_for_slot;             // length n_lat

    std::string family;
    double phi;

    // tgmrf spec
    const tulpa::tgmrf_backend::TgmrfSpec* spec;
    bool has_mu;                                            // false -> mu treated as zero

    // FD step on theta for dQ/dtheta and dmu/dtheta and d log_prior / d theta
    double fd_step;
};

// Wrap an Eigen vector around the theta tail of q.
inline Eigen::VectorXd as_vec(const double* p, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v[i] = p[i];
    return v;
}

// Call the registered Q(theta) kernel. Always returns a compressed CSC matrix
// of dimensions n_lat x n_lat.
inline Eigen::SparseMatrix<double> eval_Q(const JointState& st,
                                          const Eigen::VectorXd& theta) {
    Eigen::SparseMatrix<double> Q = st.spec->Q_double(theta);
    Q.makeCompressed();
    return Q;
}

// Call the registered mu(theta) kernel; returns zero vector if no mu.
inline Eigen::VectorXd eval_mu(const JointState& st,
                                const Eigen::VectorXd& theta) {
    if (!st.has_mu || st.spec->mu_double == nullptr) {
        return Eigen::VectorXd::Zero(st.n_lat);
    }
    Eigen::VectorXd mu = st.spec->mu_double(theta);
    if (mu.size() == 0) {
        return Eigen::VectorXd::Zero(st.n_lat);
    }
    return mu;
}

// Call the registered log_prior(theta).
inline double eval_log_prior(const JointState& st,
                              const Eigen::VectorXd& theta) {
    if (st.spec->log_prior_double == nullptr) return 0.0;
    return st.spec->log_prior_double(theta);
}

// Sparse logdet via Cholesky (LL^T). Falls back to LU if Cholesky fails on a
// nearly-singular Q.
inline double sparse_logdet_chol(const Eigen::SparseMatrix<double>& Q,
                                  bool& ok) {
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt(Q);
    if (llt.info() != Eigen::Success) {
        ok = false;
        return -std::numeric_limits<double>::infinity();
    }
    ok = true;
    // logdet(Q) = 2 sum log diag(L).
    const auto& Lf = llt.matrixL();
    Eigen::SparseMatrix<double> Lmat = Lf;
    double ld = 0.0;
    for (int k = 0; k < Lmat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Lmat, k); it; ++it) {
            if (it.row() == it.col()) {
                ld += std::log(it.value());
            }
        }
    }
    return 2.0 * ld;
}

// Compute log-posterior at (beta, z, theta) and (optionally) its gradient. The
// gradient layout matches q: [grad_beta, grad_z, grad_theta]. Sets ok = false
// if anything non-finite or factorization failure occurred (caller treats as
// -Inf / divergent).
//
// This is the core kernel: every leapfrog step calls it twice for an HMC
// update.
double log_post_and_grad(const JointState& st,
                         const std::vector<double>& q,
                         bool need_grad,
                         std::vector<double>* grad_out,
                         bool& ok) {
    ok = true;

    const int p = st.p, n_lat = st.n_lat, d = st.d, N = st.N;
    const int idx_beta = 0;
    const int idx_z    = p;
    const int idx_theta = p + n_lat;

    Eigen::Map<const Eigen::VectorXd> beta(&q[idx_beta], p);
    Eigen::Map<const Eigen::VectorXd> z(&q[idx_z], n_lat);
    Eigen::VectorXd theta(d);
    for (int m = 0; m < d; ++m) theta[m] = q[idx_theta + m];

    // ---- eta = X beta + z[obs_idx] -------------------------------------------
    Eigen::VectorXd Xb = st.X * beta;
    Eigen::VectorXd eta(N);
    for (int i = 0; i < N; ++i) {
        eta[i] = Xb[i] + z[st.obs_idx_0[i]];
    }

    // ---- log p(y | eta), s_i = d log_lik / d eta_i ----------------------------
    double log_lik = 0.0;
    std::vector<double> s_i(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double e = eta[i];
        if (!std::isfinite(e)) { ok = false; return -std::numeric_limits<double>::infinity(); }
        tulpa::GradHess gh = tulpa::grad_hess_for_family(
            st.y[i], st.n_trials[i], e, st.family, st.phi);
        double ll = tulpa::log_lik_for_family(
            st.y[i], st.n_trials[i], e, st.family, st.phi);
        if (!std::isfinite(ll) || !std::isfinite(gh.grad)) {
            ok = false;
            return -std::numeric_limits<double>::infinity();
        }
        log_lik += ll;
        s_i[i] = gh.grad;
    }

    // ---- Q(theta), mu(theta), z - mu -----------------------------------------
    Eigen::SparseMatrix<double> Q;
    Eigen::VectorXd mu;
    try {
        Q  = eval_Q(st, theta);
        mu = eval_mu(st, theta);
    } catch (...) {
        ok = false;
        return -std::numeric_limits<double>::infinity();
    }
    if (Q.rows() != n_lat || Q.cols() != n_lat) {
        ok = false;
        return -std::numeric_limits<double>::infinity();
    }

    Eigen::VectorXd r = z - mu;                              // z - mu(theta)
    Eigen::VectorXd Qr = Q * r;                              // Q (z - mu)

    // ---- logdet Q via Cholesky ----------------------------------------------
    bool chol_ok = true;
    double logdet_Q = sparse_logdet_chol(Q, chol_ok);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    if (chol_ok) {
        llt.compute(Q);
        if (llt.info() != Eigen::Success) chol_ok = false;
    }
    if (!chol_ok || !std::isfinite(logdet_Q)) {
        ok = false;
        return -std::numeric_limits<double>::infinity();
    }

    double quad = r.dot(Qr);

    const double LOG2PI = 1.8378770664093453;
    double log_prior_z = 0.5 * logdet_Q - 0.5 * quad - 0.5 * n_lat * LOG2PI;
    double log_prior_theta = eval_log_prior(st, theta);

    double log_post = log_lik + log_prior_z + log_prior_theta;
    if (!std::isfinite(log_post)) {
        ok = false;
        return -std::numeric_limits<double>::infinity();
    }

    if (!need_grad) return log_post;

    grad_out->assign(p + n_lat + d, 0.0);
    double* gbeta = &(*grad_out)[idx_beta];
    double* gz    = &(*grad_out)[idx_z];
    double* gtheta = &(*grad_out)[idx_theta];

    // ---- grad_beta = X' s ----------------------------------------------------
    for (int i = 0; i < N; ++i) {
        double si = s_i[i];
        for (int k = 0; k < p; ++k) {
            gbeta[k] += st.X(i, k) * si;
        }
    }

    // ---- grad_z = (sum of s_i over obs touching this slot) - Q (z - mu) -----
    for (int j = 0; j < n_lat; ++j) {
        double g_j = 0.0;
        const auto& obs = st.obs_for_slot[j];
        for (int ii : obs) g_j += s_i[ii];
        gz[j] = g_j - Qr[j];
    }

    // ---- grad_theta via central FD on Q, mu, log_prior -----------------------
    //
    // d log_post / d theta_m
    //   = 0.5 tr(Q^{-1} dQ/dtheta_m)
    //     - 0.5 (z-mu)' dQ/dtheta_m (z-mu)
    //     + (dmu/dtheta_m)' Q (z-mu)
    //     + d log p(theta) / d theta_m
    //
    // Central FD: dQ/dtheta_m ~ (Q(theta + h e_m) - Q(theta - h e_m)) / (2h)
    // Same for mu and log_prior. h = st.fd_step.
    double h = st.fd_step;

    for (int m = 0; m < d; ++m) {
        Eigen::VectorXd tp = theta, tm = theta;
        tp[m] += h; tm[m] -= h;
        Eigen::SparseMatrix<double> Qp, Qm;
        Eigen::VectorXd mup, mum;
        double lpp, lpm;
        try {
            Qp = eval_Q(st, tp);
            Qm = eval_Q(st, tm);
            mup = eval_mu(st, tp);
            mum = eval_mu(st, tm);
            lpp = eval_log_prior(st, tp);
            lpm = eval_log_prior(st, tm);
        } catch (...) {
            ok = false;
            return -std::numeric_limits<double>::infinity();
        }

        // dQ/dtheta_m (sparse). Two same-pattern subtraction; we keep the
        // pattern of the union of (Qp, Qm) -- in practice the same as Q.
        Eigen::SparseMatrix<double> dQ = (Qp - Qm) * (1.0 / (2.0 * h));
        dQ.makeCompressed();

        Eigen::VectorXd dmu = (mup - mum) / (2.0 * h);

        double dlogprior = (lpp - lpm) / (2.0 * h);

        // 0.5 (z-mu)' dQ (z-mu)
        Eigen::VectorXd dQr = dQ * r;
        double quad_dQ = r.dot(dQr);

        // tr(Q^{-1} dQ) -- solve Q X = dQ exactly: dQ has small nnz (same
        // pattern as Q in the common case), so we solve one sparse RHS at a
        // time only at the column indices that have nonzero entries. We do
        // the simplest correct thing: solve Q * S = dQ as dense and read the
        // trace. For n_lat <= a few thousand this is fine; document the
        // cutoff in the wrapper.
        //
        // tr(Q^{-1} dQ) = sum_j (Q^{-1} dQ)_{jj}. With Q = L L^T,
        // Q^{-1} dQ_j = L^{-T} L^{-1} dQ_j -- but dQ_j is a sparse column.
        // For simplicity we do the dense form. The n_lat target for joint
        // NUTS is hundreds, not tens of thousands.
        Eigen::MatrixXd dQ_dense = Eigen::MatrixXd(dQ);
        Eigen::MatrixXd sol = llt.solve(dQ_dense);
        double trace_term = sol.trace();

        // dmu' Q r
        double cross = dmu.dot(Qr);

        gtheta[m] = 0.5 * trace_term
                  - 0.5 * quad_dQ
                  + cross
                  + dlogprior;
    }

    return log_post;
}

// ---------------------------------------------------------------------------
// Recursive NUTS tree (Hoffman & Gelman 2014 Algorithm 3, slice / no DA inner).
//
// We use the deterministic-mass-matrix variant: diagonal M_inv supplied by the
// caller (typically from pilot Laplace inverse-variance). Step-size eps is
// adapted via dual averaging in the outer loop.
// ---------------------------------------------------------------------------

struct LeapfrogResult {
    std::vector<double> q;
    std::vector<double> r;
    std::vector<double> grad;
    double log_post;
    bool ok;
};

static inline LeapfrogResult leapfrog_step(
    const JointState& st,
    const std::vector<double>& q,
    const std::vector<double>& r,
    const std::vector<double>& grad,
    const std::vector<double>& M_inv,
    double eps,
    int dir
) {
    int D = (int)q.size();
    LeapfrogResult out;
    out.q.resize(D);
    out.r.resize(D);
    out.grad.resize(D);

    // r_half = r + 0.5 * eps * dir * grad
    std::vector<double> r_half(D);
    for (int i = 0; i < D; ++i) r_half[i] = r[i] + 0.5 * eps * dir * grad[i];

    // q_new = q + eps * dir * M_inv * r_half
    for (int i = 0; i < D; ++i) out.q[i] = q[i] + eps * dir * M_inv[i] * r_half[i];

    // Gradient at q_new
    bool gok = true;
    double lp_new = log_post_and_grad(st, out.q, true, &out.grad, gok);
    out.ok = gok;
    out.log_post = lp_new;
    if (!gok || !std::isfinite(lp_new)) {
        out.ok = false;
        return out;
    }

    // r_new = r_half + 0.5 * eps * dir * grad_new
    for (int i = 0; i < D; ++i) out.r[i] = r_half[i] + 0.5 * eps * dir * out.grad[i];

    return out;
}

inline double kinetic_energy(const std::vector<double>& r,
                              const std::vector<double>& M_inv) {
    double K = 0.0;
    for (size_t i = 0; i < r.size(); ++i) K += M_inv[i] * r[i] * r[i];
    return 0.5 * K;
}

inline double hamiltonian(double log_post, const std::vector<double>& r,
                           const std::vector<double>& M_inv) {
    return -log_post + kinetic_energy(r, M_inv);
}

// U-turn condition (Hoffman & Gelman 2014 eqn 9).
inline bool no_uturn(const std::vector<double>& q_minus,
                      const std::vector<double>& q_plus,
                      const std::vector<double>& r_minus,
                      const std::vector<double>& r_plus,
                      const std::vector<double>& M_inv) {
    double dot_minus = 0.0, dot_plus = 0.0;
    for (size_t i = 0; i < q_minus.size(); ++i) {
        double dq = q_plus[i] - q_minus[i];
        dot_minus += dq * M_inv[i] * r_minus[i];
        dot_plus  += dq * M_inv[i] * r_plus[i];
    }
    return dot_minus >= 0 && dot_plus >= 0;
}

struct TreeNode {
    // Boundary states
    std::vector<double> q_minus, r_minus, grad_minus;
    std::vector<double> q_plus,  r_plus,  grad_plus;
    double log_post_minus, log_post_plus;
    // Sample drawn from this subtree
    std::vector<double> q_prime;
    double log_post_prime;
    // Slice contribution and continue/stop flag
    int n_prime;
    bool s_prime;
    // True if any leaf in this subtree diverged (leapfrog failure or the energy
    // error exceeding max_dE). Distinct from !s_prime, which also fires on a
    // U-turn -- a normal termination, not a divergence.
    bool diverged;
    // Acceptance stats for DA
    double alpha;
    int n_alpha;
};

TreeNode build_tree(const JointState& st,
                    const std::vector<double>& q,
                    const std::vector<double>& r,
                    const std::vector<double>& grad,
                    double log_post,
                    double log_u,
                    int dir,
                    int depth,
                    double eps,
                    double H0,
                    const std::vector<double>& M_inv,
                    std::mt19937& rng,
                    double max_dE = 1000.0) {
    TreeNode out;
    if (depth == 0) {
        LeapfrogResult lf = leapfrog_step(st, q, r, grad, M_inv, eps, dir);
        if (!lf.ok) {
            // Divergent leaf
            out.q_minus = out.q_plus = q;
            out.r_minus = out.r_plus = r;
            out.grad_minus = out.grad_plus = grad;
            out.log_post_minus = out.log_post_plus = log_post;
            out.q_prime = q;
            out.log_post_prime = log_post;
            out.n_prime = 0;
            out.s_prime = false;
            out.diverged = true;
            out.alpha = 0.0;
            out.n_alpha = 1;
            return out;
        }
        double H_new = hamiltonian(lf.log_post, lf.r, M_inv);
        out.q_minus = out.q_plus = lf.q;
        out.r_minus = out.r_plus = lf.r;
        out.grad_minus = out.grad_plus = lf.grad;
        out.log_post_minus = out.log_post_plus = lf.log_post;
        out.q_prime = lf.q;
        out.log_post_prime = lf.log_post;
        out.n_prime = (log_u <= -H_new) ? 1 : 0;
        out.s_prime = (log_u < (max_dE - H_new));
        out.diverged = !out.s_prime;  // leaf stop here == energy divergence
        double dH = H0 - H_new;
        out.alpha = (dH > 0) ? 1.0 : std::exp(dH);
        out.n_alpha = 1;
        return out;
    }

    TreeNode left = build_tree(st, q, r, grad, log_post, log_u, dir, depth - 1,
                                eps, H0, M_inv, rng, max_dE);
    if (!left.s_prime) return left;

    TreeNode right;
    if (dir == -1) {
        right = build_tree(st, left.q_minus, left.r_minus, left.grad_minus,
                           left.log_post_minus, log_u, dir, depth - 1, eps, H0,
                           M_inv, rng, max_dE);
        out.q_minus = right.q_minus;
        out.r_minus = right.r_minus;
        out.grad_minus = right.grad_minus;
        out.log_post_minus = right.log_post_minus;
        out.q_plus = left.q_plus;
        out.r_plus = left.r_plus;
        out.grad_plus = left.grad_plus;
        out.log_post_plus = left.log_post_plus;
    } else {
        right = build_tree(st, left.q_plus, left.r_plus, left.grad_plus,
                           left.log_post_plus, log_u, dir, depth - 1, eps, H0,
                           M_inv, rng, max_dE);
        out.q_plus = right.q_plus;
        out.r_plus = right.r_plus;
        out.grad_plus = right.grad_plus;
        out.log_post_plus = right.log_post_plus;
        out.q_minus = left.q_minus;
        out.r_minus = left.r_minus;
        out.grad_minus = left.grad_minus;
        out.log_post_minus = left.log_post_minus;
    }

    int n_combined = left.n_prime + right.n_prime;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (n_combined > 0 && U(rng) < (double)right.n_prime / (double)n_combined) {
        out.q_prime = right.q_prime;
        out.log_post_prime = right.log_post_prime;
    } else {
        out.q_prime = left.q_prime;
        out.log_post_prime = left.log_post_prime;
    }
    out.n_prime = n_combined;
    out.s_prime = right.s_prime &&
                  no_uturn(out.q_minus, out.q_plus, out.r_minus, out.r_plus, M_inv);
    out.diverged = left.diverged || right.diverged;
    out.alpha = left.alpha + right.alpha;
    out.n_alpha = left.n_alpha + right.n_alpha;
    return out;
}

} // namespace

// ===========================================================================
// Rcpp entry point: cpp_tgmrf_nuts_joint
// ===========================================================================

// [[Rcpp::export]]
Rcpp::List cpp_tgmrf_nuts_joint(
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector obs_idx,                 // 1-based per R convention
    std::string family,
    double phi,
    std::string cpp_id,
    int theta_dim,
    int n_latent,
    Rcpp::NumericVector beta_init,
    Rcpp::NumericVector z_init,
    Rcpp::NumericVector theta_init,
    Rcpp::NumericVector M_inv_diag,              // length p + n_lat + theta_dim
    double epsilon0,
    int n_iter,
    int n_warmup,
    int max_depth,
    double target_accept,
    double fd_step,
    bool verbose,
    int seed,
    bool debug_gradient_check
) {
    // ---- Look up spec --------------------------------------------------------
    const tulpa::tgmrf_backend::TgmrfSpec* spec =
        tulpa::tgmrf_backend::lookup_spec(cpp_id);
    if (spec == nullptr) {
        Rcpp::stop("cpp_tgmrf_nuts_joint: no tgmrf spec registered under id '%s'.",
                   cpp_id.c_str());
    }
    if (spec->Q_double == nullptr || spec->log_prior_double == nullptr) {
        Rcpp::stop("cpp_tgmrf_nuts_joint: tgmrf spec '%s' missing required kernels.",
                   cpp_id.c_str());
    }

    int N = y.size();
    int p = X.ncol();
    int D = p + n_latent + theta_dim;

    if ((int)n_trials.size() != N) Rcpp::stop("n_trials length != N");
    if (X.nrow() != N) Rcpp::stop("nrow(X) != N");
    if ((int)obs_idx.size() != N) Rcpp::stop("obs_idx length != N");
    if ((int)beta_init.size() != p)     Rcpp::stop("beta_init length != ncol(X)");
    if ((int)z_init.size() != n_latent) Rcpp::stop("z_init length != n_latent");
    if ((int)theta_init.size() != theta_dim) Rcpp::stop("theta_init length != theta_dim");
    if ((int)M_inv_diag.size() != D) Rcpp::stop("M_inv_diag length != D");

    // ---- Build state ---------------------------------------------------------
    JointState st;
    st.N = N; st.p = p; st.n_lat = n_latent; st.d = theta_dim;
    st.y = y; st.n_trials = n_trials;
    st.X = Eigen::MatrixXd(N, p);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < p; ++k)
            st.X(i, k) = X(i, k);
    st.obs_idx_0 = Eigen::VectorXi(N);
    for (int i = 0; i < N; ++i) {
        int oi = obs_idx[i] - 1;
        if (oi < 0 || oi >= n_latent) {
            Rcpp::stop("obs_idx[%d] = %d outside [1, %d]", i + 1, obs_idx[i], n_latent);
        }
        st.obs_idx_0[i] = oi;
    }
    st.obs_for_slot.assign(n_latent, std::vector<int>());
    for (int i = 0; i < N; ++i) st.obs_for_slot[st.obs_idx_0[i]].push_back(i);

    st.family = family;
    st.phi = phi;
    st.spec = spec;
    st.has_mu = (spec->mu_double != nullptr);
    st.fd_step = fd_step;

    // ---- Initial q, grad, log_post -------------------------------------------
    std::vector<double> q(D, 0.0);
    for (int k = 0; k < p;        ++k) q[k]               = beta_init[k];
    for (int j = 0; j < n_latent; ++j) q[p + j]           = z_init[j];
    for (int m = 0; m < theta_dim;++m) q[p + n_latent + m] = theta_init[m];

    std::vector<double> grad(D, 0.0);
    bool ok = true;
    double log_post = log_post_and_grad(st, q, true, &grad, ok);
    if (!ok || !std::isfinite(log_post)) {
        Rcpp::stop("cpp_tgmrf_nuts_joint: initial (beta, z, theta) gave non-finite log-post.");
    }

    // ---- Optional gradient check: compare analytical to numerical at init ---
    if (debug_gradient_check) {
        double h = 1e-5;
        double max_rel = 0.0;
        int worst_idx = -1;
        for (int k = 0; k < D; ++k) {
            std::vector<double> qp = q, qm = q;
            qp[k] += h; qm[k] -= h;
            bool okp = true, okm = true;
            std::vector<double> dummy;
            double lpp = log_post_and_grad(st, qp, false, &dummy, okp);
            double lpm = log_post_and_grad(st, qm, false, &dummy, okm);
            if (!okp || !okm) continue;
            double num = (lpp - lpm) / (2 * h);
            double ana = grad[k];
            double denom = std::max(1.0, std::abs(ana) + std::abs(num));
            double rel = std::abs(num - ana) / denom;
            if (rel > max_rel) { max_rel = rel; worst_idx = k; }
        }
        Rcpp::Rcout << "[gradient-check] max relative error = " << max_rel
                    << " at index " << worst_idx << " of " << D << std::endl;
    }

    // ---- Mass matrix ----------------------------------------------------------
    std::vector<double> M_inv(D, 1.0);
    for (int i = 0; i < D; ++i) M_inv[i] = std::max(1e-8, M_inv_diag[i]);
    std::vector<double> sqrt_M(D, 1.0);
    for (int i = 0; i < D; ++i) sqrt_M[i] = 1.0 / std::sqrt(M_inv[i]);

    // ---- Dual-averaging step-size adaptation ---------------------------------
    // Hoffman & Gelman (2014) Algorithm 5 constants
    double mu_log = std::log(10.0 * epsilon0);
    double log_eps = std::log(epsilon0);
    double log_eps_bar = 0.0;
    double H_bar = 0.0;
    double gamma_da = 0.05;
    double t0 = 10.0;
    double kappa = 0.75;

    // ---- Storage -------------------------------------------------------------
    Rcpp::NumericMatrix draws_beta(n_iter - n_warmup, p);
    Rcpp::NumericMatrix draws_z(n_iter - n_warmup, n_latent);
    Rcpp::NumericMatrix draws_theta(n_iter - n_warmup, theta_dim);
    Rcpp::IntegerVector tree_depth(n_iter);
    Rcpp::NumericVector accept_prob(n_iter);
    Rcpp::IntegerVector divergent(n_iter);
    Rcpp::NumericVector log_post_iter(n_iter);

    std::mt19937 rng((unsigned)seed);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    int post = 0;
    for (int t = 0; t < n_iter; ++t) {
        // Resample momentum r ~ N(0, M)
        std::vector<double> r(D);
        for (int i = 0; i < D; ++i) r[i] = rnorm(rng) * sqrt_M[i];

        double H0 = hamiltonian(log_post, r, M_inv);
        double log_u = -H0 + std::log(runif(rng));

        std::vector<double> q_minus = q, q_plus = q;
        std::vector<double> r_minus = r, r_plus = r;
        std::vector<double> grad_minus = grad, grad_plus = grad;
        double lp_minus = log_post, lp_plus = log_post;
        std::vector<double> q_new = q;
        double log_post_new = log_post;
        std::vector<double> grad_new = grad;

        int j = 0, n_size = 1;
        bool s_continue = true;
        double alpha_total = 0.0;
        int n_alpha_total = 0;
        bool diverged = false;

        double eps_cur = std::exp(log_eps);

        while (s_continue && j < max_depth) {
            int dir = (runif(rng) < 0.5) ? -1 : 1;
            TreeNode bt;
            if (dir == -1) {
                bt = build_tree(st, q_minus, r_minus, grad_minus, lp_minus,
                                log_u, dir, j, eps_cur, H0, M_inv, rng);
                q_minus = bt.q_minus; r_minus = bt.r_minus; grad_minus = bt.grad_minus;
                lp_minus = bt.log_post_minus;
            } else {
                bt = build_tree(st, q_plus, r_plus, grad_plus, lp_plus,
                                log_u, dir, j, eps_cur, H0, M_inv, rng);
                q_plus = bt.q_plus; r_plus = bt.r_plus; grad_plus = bt.grad_plus;
                lp_plus = bt.log_post_plus;
            }
            if (bt.s_prime && bt.n_prime > 0) {
                if (runif(rng) < (double)bt.n_prime / (double)std::max(1, n_size)) {
                    q_new = bt.q_prime;
                    log_post_new = bt.log_post_prime;
                    // grad_new resync (cheap; one extra log_post_and_grad call)
                    bool gok = true;
                    log_post_and_grad(st, q_new, true, &grad_new, gok);
                    if (!gok) {
                        diverged = true;
                    }
                }
            }
            if (bt.diverged) diverged = true;
            n_size += bt.n_prime;
            alpha_total += bt.alpha;
            n_alpha_total += bt.n_alpha;
            s_continue = bt.s_prime &&
                         no_uturn(q_minus, q_plus, r_minus, r_plus, M_inv);
            ++j;
        }

        q = q_new;
        log_post = log_post_new;
        grad = grad_new;

        tree_depth[t] = j;
        double alpha_mean = (n_alpha_total > 0) ? alpha_total / n_alpha_total : 0.0;
        accept_prob[t] = alpha_mean;
        divergent[t] = diverged ? 1 : 0;
        log_post_iter[t] = log_post;

        // Dual averaging
        if (t < n_warmup) {
            double m_t = t + 1.0;
            H_bar = (1.0 - 1.0 / (m_t + t0)) * H_bar
                  + (1.0 / (m_t + t0)) * (target_accept - alpha_mean);
            log_eps = mu_log - std::sqrt(m_t) / gamma_da * H_bar;
            double eta_t = std::pow(m_t, -kappa);
            log_eps_bar = eta_t * log_eps + (1.0 - eta_t) * log_eps_bar;
        } else if (t == n_warmup) {
            log_eps = log_eps_bar;
        }

        // Store post-warmup draw
        if (t >= n_warmup) {
            for (int k = 0; k < p; ++k)       draws_beta(post, k) = q[k];
            for (int jj = 0; jj < n_latent; ++jj) draws_z(post, jj) = q[p + jj];
            for (int m = 0; m < theta_dim; ++m)   draws_theta(post, m) = q[p + n_latent + m];
            ++post;
        }

        if (verbose && (t % 100 == 0)) {
            Rcpp::Rcout << "  iter " << t << "/" << n_iter
                        << " depth=" << j
                        << " alpha=" << alpha_mean
                        << " eps=" << eps_cur
                        << " log_post=" << log_post << std::endl;
        }

        Rcpp::checkUserInterrupt();
    }

    return Rcpp::List::create(
        Rcpp::Named("draws_beta")  = draws_beta,
        Rcpp::Named("draws_z")     = draws_z,
        Rcpp::Named("draws_theta") = draws_theta,
        Rcpp::Named("tree_depth")  = tree_depth,
        Rcpp::Named("accept_prob") = accept_prob,
        Rcpp::Named("divergent")   = divergent,
        Rcpp::Named("log_post")    = log_post_iter,
        Rcpp::Named("epsilon")     = std::exp(log_eps)
    );
}
