// hmc_sampler_mass_blocks.h
// Fragment of hmc_sampler.h. Included from the umbrella header inside
// namespace tulpa_hmc { ... }; do NOT add a namespace wrapper here.
// MassBlock (<=4x4), PrecisionBlock, KroneckerBlock,
// SparseGMRFBlock, DenseMassMatrix container.
#ifndef TULPA_HMC_SAMPLER_MASS_BLOCKS_H
#define TULPA_HMC_SAMPLER_MASS_BLOCKS_H

// =====================================================================
// Block-diagonal mass block (max 4×4, stack-allocated)
// =====================================================================

struct MassBlock {
  int start = 0;          // First param index in full parameter vector
  int size = 0;           // Block size (2-4)
  bool adapted = false;

  // Block mass storage (column-major, max 4×4)
  double inv_mass[16] = {};    // C_block (covariance block)
  double L_inv_mass[16] = {};  // Cholesky L where LL^T = C_block

  // Block-local Welford covariance accumulator
  int welford_n = 0;
  double welford_mean[4] = {};
  double welford_M2[16] = {};  // Running sum for covariance (column-major)

  void init(int s, int sz) {
    start = s;
    size = sz;
    adapted = false;
    std::memset(inv_mass, 0, sizeof(inv_mass));
    std::memset(L_inv_mass, 0, sizeof(L_inv_mass));
    // Initialize as identity
    for (int i = 0; i < sz; i++) {
      inv_mass[i * 4 + i] = 1.0;  // Using stride=4 (max block size)
      L_inv_mass[i * 4 + i] = 1.0;
    }
    reset_welford();
  }

  void reset_welford() {
    welford_n = 0;
    std::memset(welford_mean, 0, sizeof(welford_mean));
    std::memset(welford_M2, 0, sizeof(welford_M2));
  }

  // Extract block params and update Welford running stats
  void welford_update(const double* full_params) {
    welford_n++;
    double delta[4];
    for (int i = 0; i < size; i++) {
      delta[i] = full_params[start + i] - welford_mean[i];
      welford_mean[i] += delta[i] / welford_n;
    }
    for (int i = 0; i < size; i++) {
      double dx_new = full_params[start + i] - welford_mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        welford_M2[j * 4 + i] += val;  // stride=4
        if (i != j) {
          welford_M2[i * 4 + j] += val;
        }
      }
    }
  }

  // Compute covariance from Welford stats, Cholesky decompose, set adapted
  bool update_from_welford() {
    if (welford_n < 10) return false;

    // Compute sample covariance with small regularization
    double cov[16] = {};
    double scale = 1.0 / (welford_n - 1);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cov[i * 4 + j] = welford_M2[j * 4 + i] * scale;  // Note: M2 is col-major with stride 4
      }
      cov[i * 4 + i] += 1e-8;  // Regularization
    }

    // Try Cholesky decomposition
    double L[16] = {};
    if (!cholesky_small(cov, L, size)) return false;

    // Success: copy to block storage
    std::memcpy(inv_mass, cov, sizeof(inv_mass));
    std::memcpy(L_inv_mass, L, sizeof(L_inv_mass));
    adapted = true;
    return true;
  }

  // Tiny Cholesky for k<=4 (direct formula, no Eigen)
  // A and L use stride=4 (max block size)
  static bool cholesky_small(const double* A, double* L, int k) {
    std::memset(L, 0, 16 * sizeof(double));
    for (int i = 0; i < k; i++) {
      double sum = 0.0;
      for (int p = 0; p < i; p++) {
        sum += L[i * 4 + p] * L[i * 4 + p];
      }
      double diag = A[i * 4 + i] - sum;
      if (diag <= 0.0) return false;
      L[i * 4 + i] = std::sqrt(diag);
      for (int j = i + 1; j < k; j++) {
        double s = 0.0;
        for (int p = 0; p < i; p++) {
          s += L[j * 4 + p] * L[i * 4 + p];
        }
        L[j * 4 + i] = (A[j * 4 + i] - s) / L[i * 4 + i];
      }
    }
    return true;
  }

  // result[0..size-1] = C_block * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    for (int i = 0; i < size; i++) {
      double sum = 0.0;
      for (int j = 0; j < size; j++) {
        sum += inv_mass[i * 4 + j] * pb[j];
      }
      result[i] = sum;
    }
  }

  // p_block^T * C_block * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        result += pb[i] * inv_mass[i * 4 + j] * pb[j];
      }
    }
    return result;
  }

  // Sample momentum for block: p_block = L^{-T} z (back-substitution on tiny L)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!adapted) return;  // Non-adapted blocks use diagonal path
    std::normal_distribution<double> normal(0.0, 1.0);
    double z[4];
    for (int i = 0; i < size; i++) z[i] = normal(rng);

    // Back-substitution: solve L^T * p = z (upper triangular)
    double* pb = p_full + start;
    for (int i = size - 1; i >= 0; i--) {
      double sum = z[i];
      for (int j = i + 1; j < size; j++) {
        sum -= L_inv_mass[j * 4 + i] * pb[j];  // L^T[i][j] = L[j][i]
      }
      pb[i] = sum / L_inv_mass[i * 4 + i];
    }
  }
};

// =====================================================================
// Precision-informed mass block (heap-allocated, arbitrary size)
// Used for ICAR/BYM2 spatial params where Q (precision) is known analytically.
// Unlike MassBlock (≤4, stack), this handles S×S blocks (S=50 typical).
// NOT adapted from samples — uses fixed analytical precision.
// =====================================================================

struct PrecisionBlock {
  int start = 0;        // First param index in full parameter vector
  int size = 0;         // Block dimension S
  bool active = false;

  // M^{-1} = Q_reg_inv: (Q + lambda*I)^{-1}, column-major S×S
  std::vector<double> Q_inv;
  // L_Q: Cholesky factor where L*L^T = Q + lambda*I, column-major S×S
  // Used for momentum sampling: p_block = L^{-T} * z
  std::vector<double> L_chol;

  void init(int s, int sz, const double* q_inv_data, const double* l_chol_data) {
    start = s;
    size = sz;
    active = true;
    int nn = static_cast<int>(static_cast<size_t>(sz) * sz);
    Q_inv.assign(q_inv_data, q_inv_data + nn);
    L_chol.assign(l_chol_data, l_chol_data + nn);
  }

  // result[0..size-1] = Q_inv * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> M(Q_inv.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> pv(pb, size);
    Eigen::Map<Eigen::VectorXd> rv(result, size);
    rv.noalias() = M.selfadjointView<Eigen::Lower>() * pv;
  }

  // p_block^T * Q_inv * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> M(Q_inv.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> pv(pb, size);
    return pv.dot(M.selfadjointView<Eigen::Lower>() * pv);
  }

  // Sample momentum for block: solve L^T * p = z (back-substitution)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> z(size);
    for (int i = 0; i < size; i++) z[i] = normal(rng);
    double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> L(L_chol.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> zv(z.data(), size);
    Eigen::Map<Eigen::VectorXd> pv(pb, size);
    pv.noalias() = L.transpose().triangularView<Eigen::Upper>().solve(zv);
  }
};

// =====================================================================
// Kronecker precision block for spatiotemporal (ST) interaction params.
// M = Q_space ⊗ Q_time, M^{-1} = Q_space^{-1} ⊗ Q_time^{-1}
// Never forms the full (S*T)×(S*T) matrix — O(S*T*(S+T)) operations.
// =====================================================================

struct KroneckerBlock {
  int start = 0;        // First ST param index in full parameter vector
  int S = 0;            // Spatial dimension
  int T = 0;            // Temporal dimension
  bool active = false;

  // Spatial: Q_space_inv (S×S), L_space (Cholesky of Q_space, S×S)
  std::vector<double> Qs_inv;  // column-major
  std::vector<double> Ls;      // column-major

  // Temporal: Q_time_inv (T×T), L_time (Cholesky of Q_time, T×T)
  std::vector<double> Qt_inv;  // column-major
  std::vector<double> Lt;      // column-major

  // Scratch buffers (pre-allocated)
  mutable std::vector<double> scratch_ST;  // S*T work buffer

  void init(int st, int ns, int nt,
            const double* qs_inv, const double* ls,
            const double* qt_inv, const double* lt) {
    start = st;
    S = ns;
    T = nt;
    active = true;
    int ss = S * S, tt = T * T;
    Qs_inv.assign(qs_inv, qs_inv + ss);
    Ls.assign(ls, ls + ss);
    Qt_inv.assign(qt_inv, qt_inv + tt);
    Lt.assign(lt, lt + tt);
    scratch_ST.resize(static_cast<size_t>(S) * T);
  }

  // Compute (A ⊗ B) * vec(X) = vec(B * X * A^T)
  // where X is S×T (column-major), A is T×T, B is S×S
  // result = vec(B * X * A^T)
  // This is the standard Kronecker-vector product identity.
  void kron_matvec(const double* As, int na, const double* Bt, int nb,
                   const double* x, double* result) const {
    // Step 1: tmp = X * A^T  (S×T * T×T = S×T)
    // X is S×T column-major, A^T is T×T
    Eigen::Map<const Eigen::MatrixXd> X(x, S, T);
    Eigen::Map<const Eigen::MatrixXd> A(Bt, T, T);  // temporal
    Eigen::Map<const Eigen::MatrixXd> B(As, S, S);   // spatial
    Eigen::Map<Eigen::MatrixXd> R(result, S, T);

    // (B ⊗ A) * vec(X) = vec(B * X * A^T)
    // But our params are stored with spatial varying fastest: param[s + t*S]
    // So X_{s,t} = x[s + t*S] which is column-major S×T.
    // M^{-1} = Qs_inv ⊗ Qt_inv
    // M^{-1} * x = vec(Qs_inv * X * Qt_inv^T)
    R.noalias() = B * X * A.transpose();
  }

  // result[0..S*T-1] = (Qs_inv ⊗ Qt_inv) * p[start..start+S*T-1]
  void matvec(const double* p_full, double* result) const {
    kron_matvec(Qs_inv.data(), S, Qt_inv.data(), T,
                p_full + start, result);
  }

  // p_block^T * (Qs_inv ⊗ Qt_inv) * p_block
  double quadform(const double* p_full) const {
    matvec(p_full, scratch_ST.data());
    const double* pb = p_full + start;
    double qf = 0.0;
    int ST = S * T;
    for (int i = 0; i < ST; i++) qf += pb[i] * scratch_ST[i];
    return qf;
  }

  // Sample momentum: p ~ N(0, M) where M = Qs ⊗ Qt
  // p = (Ls ⊗ Lt)^{-T} * z = vec(Ls^{-T} * Z * Lt^{-1})
  // where Z is S×T standard normal
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    int ST = S * T;
    // Generate Z ~ N(0,I) as S×T matrix
    std::vector<double> Z(ST);
    for (int i = 0; i < ST; i++) Z[i] = normal(rng);

    double* pb = p_full + start;
    Eigen::Map<Eigen::MatrixXd> Zm(Z.data(), S, T);
    Eigen::Map<const Eigen::MatrixXd> Lsm(Ls.data(), S, S);
    Eigen::Map<const Eigen::MatrixXd> Ltm(Lt.data(), T, T);
    Eigen::Map<Eigen::MatrixXd> Pm(pb, S, T);

    // (Ls ⊗ Lt)^{-T} * z = vec(Ls^{-T} * Z * Lt^{-1})
    // Step 1: solve Ls^T * tmp = Z  →  tmp = Ls^{-T} * Z
    Eigen::MatrixXd tmp = Lsm.transpose().triangularView<Eigen::Upper>().solve(Zm);
    // Step 2: solve tmp2 * Lt^T = tmp  →  tmp2 = tmp * Lt^{-T}
    //   which is (Lt^{-T} * tmp^T)^T = (Lt * tmp^T)^{-T}
    //   Actually: tmp2 * Lt^T = tmp → tmp2 = tmp * Lt^{-T}
    //   Transpose: Lt^{-1} * tmp^T → solve Lt * Y = tmp^T → Y = Lt^{-1} * tmp^T
    //   Then result = Y^T
    Eigen::MatrixXd Y = Ltm.triangularView<Eigen::Lower>().solve(tmp.transpose());
    Pm = Y.transpose();
  }
};

// =====================================================================
// Sparse GMRF block for ST_IV spatiotemporal interaction.
// Uses Eigen sparse Cholesky for:
//   1. Block Gibbs sampling: delta ~ N(Q^{-1}b, Q^{-1})
//   2. Mass matrix operations (momentum, kinetic energy, inv_mass*p)
// Q = tau * (Q_s ⊗ Q_t) + lambda*I (posterior precision for delta block)
// =====================================================================

struct SparseGMRFBlock {
  int start = 0;        // First ST param index in full parameter vector
  int S = 0;            // Spatial dimension
  int T = 0;            // Temporal dimension
  bool active = false;
  bool factorized = false;

  // Sparse precision and its Cholesky factorization
  Eigen::SparseMatrix<double> Q_sparse;  // ST×ST sparse precision matrix
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;  // Cholesky LL^T = Q

  // Scratch buffers (pre-allocated)
  mutable Eigen::VectorXd scratch_vec;

  void init(int st_start, int ns, int nt) {
    start = st_start;
    S = ns;
    T = nt;
    active = true;
    factorized = false;
    int ST = S * T;
    scratch_vec.resize(ST);
  }

  // Build Q_sparse = tau * (Q_s ⊗ Q_t) + diag_correction
  // adj_row_ptr/adj_col_idx: CSR adjacency for spatial graph (1-based col_idx!)
  // temp_type: RW1, RW2, AR1
  // tau: precision parameter
  // h_lik: diagonal of likelihood Hessian (length ST), or nullptr for prior-only
  // lambda_stz: sum-to-zero penalty (default 0.001)
  void build_and_factorize(
      const std::vector<int>& adj_row_ptr,
      const std::vector<int>& adj_col_idx,
      tulpa_temporal::TemporalType temp_type,
      bool cyclic,
      double tau,
      const double* h_lik,  // diagonal Hessian correction (length S*T), can be nullptr
      double lambda_stz = 0.001
  ) {
    int ST = S * T;

    // Build spatial Laplacian Q_s (S×S)
    // Q_s[i,i] = degree(i), Q_s[i,j] = -1 if adjacent
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ST * 7);  // Rough estimate: ~7 nonzeros per row

    // Build temporal precision Q_t (T×T) as dense small matrix
    Eigen::MatrixXd Qt = Eigen::MatrixXd::Zero(T, T);
    if (temp_type == tulpa_temporal::TemporalType::RW1) {
      for (int t = 0; t < T - 1; t++) {
        Qt(t, t) += 1.0;
        Qt(t + 1, t + 1) += 1.0;
        Qt(t, t + 1) = -1.0;
        Qt(t + 1, t) = -1.0;
      }
    } else if (temp_type == tulpa_temporal::TemporalType::RW2) {
      for (int t = 0; t < T - 2; t++) {
        Qt(t, t) += 1.0;
        Qt(t + 1, t + 1) += 4.0;
        Qt(t + 2, t + 2) += 1.0;
        Qt(t, t + 1) += -2.0;
        Qt(t + 1, t) += -2.0;
        Qt(t, t + 2) += 1.0;
        Qt(t + 2, t) += 1.0;
        Qt(t + 1, t + 2) += -2.0;
        Qt(t + 2, t + 1) += -2.0;
      }
      // Fix double-counted diagonal
      for (int t = 0; t < T; t++) Qt(t, t) = 0.0;
      Eigen::MatrixXd D = Eigen::MatrixXd::Zero(T - 2, T);
      for (int t = 0; t < T - 2; t++) {
        D(t, t) = 1.0; D(t, t + 1) = -2.0; D(t, t + 2) = 1.0;
      }
      Qt = D.transpose() * D;
    } else {
      // AR1 with rho=0.5 as default approximation
      for (int t = 0; t < T; t++) Qt(t, t) = 1.0;
      for (int t = 0; t < T - 1; t++) {
        Qt(t, t + 1) = -0.5;
        Qt(t + 1, t) = -0.5;
      }
    }

    // Build Kronecker product Q_kron = Q_s ⊗ Q_t as sparse triplets
    // (Q_s ⊗ Q_t)[(s1*T+t1), (s2*T+t2)] = Q_s[s1,s2] * Q_t[t1,t2]
    // For each spatial pair (s1,s2) with Q_s[s1,s2] != 0:
    //   For each temporal pair (t1,t2) with Q_t[t1,t2] != 0:
    //     Add Q_s[s1,s2] * Q_t[t1,t2] at row s1*T+t1, col s2*T+t2
    for (int s1 = 0; s1 < S; s1++) {
      int n_neigh = adj_row_ptr[s1 + 1] - adj_row_ptr[s1];
      double qs_diag = static_cast<double>(n_neigh);

      // Diagonal spatial block: Q_s[s1,s1] = degree
      for (int t1 = 0; t1 < T; t1++) {
        for (int t2 = 0; t2 < T; t2++) {
          if (Qt(t1, t2) != 0.0) {
            triplets.emplace_back(s1 * T + t1, s1 * T + t2, tau * qs_diag * Qt(t1, t2));
          }
        }
      }

      // Off-diagonal spatial: Q_s[s1,s2] = -1 for neighbors
      for (int idx = adj_row_ptr[s1]; idx < adj_row_ptr[s1 + 1]; idx++) {
        int s2 = adj_col_idx[idx] - 1;  // Convert 1-based to 0-based
        for (int t1 = 0; t1 < T; t1++) {
          for (int t2 = 0; t2 < T; t2++) {
            if (Qt(t1, t2) != 0.0) {
              triplets.emplace_back(s1 * T + t1, s2 * T + t2, tau * (-1.0) * Qt(t1, t2));
            }
          }
        }
      }
    }

    // Add diagonal corrections: likelihood Hessian + sum-to-zero penalty + regularization
    double reg = 1e-6;  // Numerical regularization for rank deficiency
    for (int k = 0; k < ST; k++) {
      double diag_add = lambda_stz + reg;  // sum-to-zero soft constraint
      if (h_lik) diag_add += h_lik[k];     // likelihood curvature
      triplets.emplace_back(k, k, diag_add);
    }

    Q_sparse.resize(ST, ST);
    Q_sparse.setFromTriplets(triplets.begin(), triplets.end());
    Q_sparse.makeCompressed();

    // Cholesky factorization
    llt.compute(Q_sparse);
    factorized = (llt.info() == Eigen::Success);
  }

  // Sample delta from GMRF conditional: delta ~ N(Q^{-1}b, Q^{-1})
  // b = grad_delta_lik (likelihood gradient wrt delta)
  // Returns new delta values in delta_out (length S*T)
  void sample_conditional(const double* b, double* delta_out, std::mt19937& rng) const {
    if (!factorized) return;
    int ST = S * T;
    std::normal_distribution<double> normal(0.0, 1.0);

    // Step 1: Solve Q * mean = b  →  mean = Q^{-1} * b
    Eigen::Map<const Eigen::VectorXd> bv(b, ST);
    Eigen::VectorXd mean = llt.solve(bv);

    // Step 2: Sample z ~ N(0, I)
    Eigen::VectorXd z(ST);
    for (int i = 0; i < ST; i++) z[i] = normal(rng);

    // Step 3: delta = mean + perturbation from N(0, Q^{-1})
    // SimplicialLLT factorizes as: P * Q * P^T = L * L^T
    // To sample from N(0, Q^{-1}): pert = P^T * L^{-T} * z
    //   Var(pert) = P^T L^{-T} L^{-1} P = P^T (LL^T)^{-1} P = (P^T LL^T P)^{-1} = Q^{-1} ✓
    auto perm = llt.permutationP();
    // Get L as a concrete sparse matrix (avoids const-view issues)
    Eigen::SparseMatrix<double> L_mat = llt.matrixL();
    // Solve L^T * v = z (upper triangular solve on L^T)
    Eigen::SparseMatrix<double> Lt_mat = L_mat.transpose();
    Eigen::VectorXd v = Lt_mat.triangularView<Eigen::Upper>().solve(z);
    // Un-permute: pert = P^T * v
    Eigen::VectorXd pert = perm.transpose() * v;

    Eigen::Map<Eigen::VectorXd> out(delta_out, ST);
    out = mean + pert;
  }

  // Mass matrix operations (for HMC momentum/kinetic energy)
  // M = Q (posterior precision), M^{-1} = Q^{-1}

  // result = Q^{-1} * p  (inv_mass * momentum)
  void inv_mass_matvec(const double* p_full, double* result) const {
    if (!factorized) return;
    int ST = S * T;
    Eigen::Map<const Eigen::VectorXd> pv(p_full + start, ST);
    Eigen::VectorXd sol = llt.solve(pv);
    std::memcpy(result, sol.data(), ST * sizeof(double));
  }

  // p^T * Q^{-1} * p  (kinetic energy contribution)
  double quadform(const double* p_full) const {
    if (!factorized) return 0.0;
    int ST = S * T;
    Eigen::Map<const Eigen::VectorXd> pv(p_full + start, ST);
    Eigen::VectorXd sol = llt.solve(pv);
    return pv.dot(sol);
  }

  // Sample momentum p ~ N(0, Q):  p = L^T * z where LL^T = Q
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!factorized) return;
    int ST = S * T;
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd z(ST);
    for (int i = 0; i < ST; i++) z[i] = normal(rng);

    // P * Q * P^T = L * L^T  →  p ~ N(0, Q) needs p = P^T * L^T * P * z
    // But simpler: p_perm = L^T * z ~ N(0, L^T L) = N(0, PQP^T)
    // Then p = P^T * p_perm ~ N(0, P^T PQP^T P) = N(0, Q) ✓
    auto perm = llt.permutationP();
    Eigen::SparseMatrix<double> L_mat = llt.matrixL();
    Eigen::VectorXd p_perm = L_mat.transpose() * z;  // L^T * z
    Eigen::VectorXd p_vec = perm.transpose() * p_perm;  // P^T * (L^T * z)
    double* pb = p_full + start;
    std::memcpy(pb, p_vec.data(), ST * sizeof(double));
  }
};

// =====================================================================
// Dense mass matrix for NUTS (encapsulates diag + dense state)
// =====================================================================

struct DenseMassMatrix {
  int n = 0;                              // Dimension
  MassMatrixType type = MassMatrixType::DIAG;
  bool adapted = false;

  // Diagonal (always available, used as fallback)
  std::vector<double> inv_mass_diag;      // M^{-1} diagonal = variance
  std::vector<double> sqrt_mass_diag;     // sqrt(M) diagonal = 1/sqrt(variance) for p sampling

  // Dense (only when type == DENSE)
  std::vector<double> inv_mass_dense;     // Full p×p M^{-1} = regularized sample covariance (column-major)
  std::vector<double> L_inv_mass;         // Cholesky factor L where LL^T = M^{-1} (column-major)

  // Scratch buffer for dense matvec results (avoids per-call allocation)
  std::vector<double> scratch;

  // Block-diagonal (only when type == BLOCK_DIAG)
  std::vector<MassBlock> blocks;
  std::vector<bool> in_block;  // in_block[i] = true if param i belongs to a block

  // Precision-informed blocks (optional, independent of type)
  // These override the mass for specific param ranges using known precision structure.
  PrecisionBlock precision_block;    // ICAR/BYM2 spatial block (DISABLED)
  KroneckerBlock kronecker_block;    // ST_IV Kronecker block (DISABLED)
  SparseGMRFBlock sparse_gmrf;       // ST_IV sparse GMRF mass + Gibbs sampling

  void init(int dim, MassMatrixType t) {
    n = dim;
    type = t;
    adapted = false;
    inv_mass_diag.assign(dim, 1.0);
    sqrt_mass_diag.assign(dim, 1.0);
    scratch.resize(dim);
    if (t == MassMatrixType::DENSE) {
      inv_mass_dense.assign(static_cast<size_t>(dim) * dim, 0.0);
      L_inv_mass.assign(static_cast<size_t>(dim) * dim, 0.0);
      // Initialize as identity
      for (int i = 0; i < dim; i++) {
        inv_mass_dense[static_cast<size_t>(i) * dim + i] = 1.0;
        L_inv_mass[static_cast<size_t>(i) * dim + i] = 1.0;
      }
    }
    if (t == MassMatrixType::BLOCK_DIAG) {
      in_block.assign(dim, false);
    }
  }

  // Initialize block-diagonal structure from block specifications
  // block_specs: vector of (start_index, block_size) pairs
  void init_block_diag(int dim, const std::vector<std::pair<int,int>>& block_specs) {
    init(dim, MassMatrixType::BLOCK_DIAG);
    blocks.clear();
    blocks.reserve(block_specs.size());
    for (const auto& spec : block_specs) {
      MassBlock blk;
      blk.init(spec.first, spec.second);
      blocks.push_back(blk);
      for (int i = spec.first; i < spec.first + spec.second; i++) {
        if (i < dim) in_block[i] = true;
      }
    }
  }

  // Update dense mass matrix from sample covariance
  // Returns true on success, false on Cholesky failure (degrades to diagonal)
  // Uses Eigen LLT for Cholesky decomposition
  bool update_from_covariance(const double* cov, int n_samples);

  // Sample momentum: p ~ N(0, M) where M = C^{-1}
  // DIAG: p[i] = z * sqrt_mass_diag[i]
  // BLOCK_DIAG: diagonal for non-block params, L^{-T} z for block params
  // DENSE: solve L^T * p = z  (back-substitution)
  // Uses Eigen triangular solve for dense case (n>=16) for SIMD acceleration.
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      // First: diagonal for all params
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
      // Then: overwrite block params with correlated samples
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          blk.sample_momentum(p, rng);
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
    } else {
      // Dense: p = L^{-T} z where LL^T = C (inv_mass)
      // We need p ~ N(0, C^{-1}), so sample z ~ N(0, I), then p = L^{-T} z
      std::vector<double> z(n);
      for (int i = 0; i < n; i++) {
        z[i] = normal(rng);
      }
      if (n >= 16) {
        Eigen::Map<const Eigen::MatrixXd> Lm(L_inv_mass.data(), n, n);
        Eigen::Map<const Eigen::VectorXd> zv(z.data(), n);
        Eigen::Map<Eigen::VectorXd> pv(p, n);
        // Solve L^T * p = z: transpose L then use upper-triangular solve
        pv.noalias() = Lm.transpose().triangularView<Eigen::Upper>().solve(zv);
      } else {
        tulpa_linalg::tri_solve_upper_transpose(L_inv_mass.data(), z.data(), p, n);
      }
    }
    // Precision/Kronecker blocks override their param ranges
    if (precision_block.active) precision_block.sample_momentum(p, rng);
    if (kronecker_block.active) kronecker_block.sample_momentum(p, rng);
    if (sparse_gmrf.active && sparse_gmrf.factorized) sparse_gmrf.sample_momentum(p, rng);
  }

  // Check if param i belongs to a precision, kronecker, or sparse GMRF block
  inline bool in_precision_block(int i) const {
    if (precision_block.active &&
        i >= precision_block.start &&
        i < precision_block.start + precision_block.size) return true;
    if (kronecker_block.active &&
        i >= kronecker_block.start &&
        i < kronecker_block.start + kronecker_block.S * kronecker_block.T) return true;
    if (sparse_gmrf.active && sparse_gmrf.factorized &&
        i >= sparse_gmrf.start &&
        i < sparse_gmrf.start + sparse_gmrf.S * sparse_gmrf.T) return true;
    return false;
  }

  // Kinetic energy: 0.5 * p^T * C * p  where C = M^{-1}
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  double kinetic_energy(const double* p) const {
    // Precision/Kronecker/Sparse blocks: compute their contribution separately
    double ke_prec = 0.0;
    if (precision_block.active) ke_prec += precision_block.quadform(p);
    if (kronecker_block.active) ke_prec += kronecker_block.quadform(p);
    if (sparse_gmrf.active && sparse_gmrf.factorized) ke_prec += sparse_gmrf.quadform(p);

    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_block[i] && !in_precision_block(i)) {
          ke += inv_mass_diag[i] * p[i] * p[i];
        }
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          ke += blk.quadform(p);
        } else {
          for (int i = blk.start; i < blk.start + blk.size; i++) {
            if (!in_precision_block(i))
              ke += inv_mass_diag[i] * p[i] * p[i];
          }
        }
      }
      return 0.5 * (ke + ke_prec);
    } else if (type == MassMatrixType::DIAG || !adapted) {
      if (!precision_block.active && !kronecker_block.active) {
        return 0.5 * tulpa_linalg::weighted_norm_squared(p, inv_mass_diag.data(), n);
      }
      // Skip precision block params in diagonal sum
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_precision_block(i))
          ke += inv_mass_diag[i] * p[i] * p[i];
      }
      return 0.5 * (ke + ke_prec);
    } else if (n >= 16) {
      // Dense: full matrix handles all params including precision block range
      // But if precision blocks are active, we need to exclude their range
      // from the dense contribution and add the precision block contribution instead.
      // For simplicity: if precision blocks active, fall back to per-element
      if (precision_block.active || kronecker_block.active) {
        double ke = 0.0;
        for (int i = 0; i < n; i++) {
          if (in_precision_block(i)) continue;
          for (int j = 0; j < n; j++) {
            if (in_precision_block(j)) continue;
            ke += p[i] * inv_mass_dense[static_cast<size_t>(j) * n + i] * p[j];
          }
        }
        return 0.5 * (ke + ke_prec);
      }
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      return 0.5 * pv.dot(Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      return 0.5 * tulpa_linalg::quadratic_form(p, inv_mass_dense.data(), n);
    }
  }

  // Compute C * p (for leapfrog position update: q += eps * C * p)
  // Result written to `result` buffer.
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  void inv_mass_times_p(const double* p, double* result) const {
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          double tmp[4];
          blk.matvec(p, tmp);
          for (int i = 0; i < blk.size; i++) {
            result[blk.start + i] = tmp[i];
          }
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
    } else if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> rv(result, n);
      rv.noalias() = Am.selfadjointView<Eigen::Lower>() * pv;
    } else {
      tulpa_linalg::symmatvec(inv_mass_dense.data(), p, result, n);
    }
    // Precision/Kronecker blocks override their param ranges
    if (precision_block.active) {
      std::vector<double> tmp(precision_block.size);
      precision_block.matvec(p, tmp.data());
      for (int i = 0; i < precision_block.size; i++) {
        result[precision_block.start + i] = tmp[i];
      }
    }
    if (kronecker_block.active) {
      int ST = kronecker_block.S * kronecker_block.T;
      std::vector<double> tmp(ST);
      kronecker_block.matvec(p, tmp.data());
      for (int i = 0; i < ST; i++) {
        result[kronecker_block.start + i] = tmp[i];
      }
    }
    if (sparse_gmrf.active && sparse_gmrf.factorized) {
      int ST = sparse_gmrf.S * sparse_gmrf.T;
      std::vector<double> tmp(ST);
      sparse_gmrf.inv_mass_matvec(p, tmp.data());
      for (int i = 0; i < ST; i++) {
        result[sparse_gmrf.start + i] = tmp[i];
      }
    }
  }

  // Compute diag(C) * p — uses diagonal only, even when dense is available.
  // Kept for backwards compatibility / debugging. The NUTS U-turn criterion
  // now uses inv_mass_times_p() for correct geometry with dense mass.
  void inv_mass_diag_times_p(const double* p, double* result) const {
    for (int i = 0; i < n; i++) {
      result[i] = inv_mass_diag[i] * p[i];
    }
  }

  // Set metric directly from precomputed G^{-1} and its Cholesky L.
  // Used by SoftAbs per-trajectory metric retry. No shrinkage applied.
  void set_from_metric(const std::vector<double>& g_inv,
                       const std::vector<double>& l_g_inv) {
    inv_mass_dense = g_inv;
    L_inv_mass = l_g_inv;
    for (int i = 0; i < n; i++) {
      inv_mass_diag[i] = g_inv[static_cast<size_t>(i) * n + i];
      sqrt_mass_diag[i] = 1.0 / std::sqrt(std::max(inv_mass_diag[i], 1e-10));
    }
    adapted = true;
  }

  // Set diagonal mass from WelfordStats output (same interface as before)
  // When type==DENSE, also populate the dense matrices as diagonal so that
  // the dense code paths (sample_momentum, kinetic_energy, inv_mass_times_p)
  // produce correct results even before full covariance is available.
  // When type==BLOCK_DIAG, diagonal is set normally; blocks are adapted separately
  // via their own Welford accumulators.
  void set_diagonal(const std::vector<double>& inv_m, const std::vector<double>& sqrt_m) {
    inv_mass_diag = inv_m;
    sqrt_mass_diag = sqrt_m;
    adapted = true;

    if (type == MassMatrixType::DENSE && !inv_mass_dense.empty()) {
      // Populate dense matrices as diagonal so dense code paths work correctly
      std::fill(inv_mass_dense.begin(), inv_mass_dense.end(), 0.0);
      std::fill(L_inv_mass.begin(), L_inv_mass.end(), 0.0);
      for (int i = 0; i < n; i++) {
        inv_mass_dense[static_cast<size_t>(i) * n + i] = inv_m[i];
        // L where LL^T = inv_mass (diagonal): L[i,i] = sqrt(inv_mass[i])
        L_inv_mass[static_cast<size_t>(i) * n + i] = std::sqrt(inv_m[i]);
      }
    }
  }
};


#endif  // TULPA_HMC_SAMPLER_MASS_BLOCKS_H
