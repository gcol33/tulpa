// =============================================================================
// L-BFGS MASS MATRIX ADAPTATION
// =============================================================================
//
// L-BFGS approximates the inverse Hessian using limited memory:
//   H_k ≈ (I - ρ_k s_k y_k^T) H_{k-1} (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
//
// where:
//   s_k = q_k - q_{k-1}     (position difference)
//   y_k = g_k - g_{k-1}     (gradient difference)
//   ρ_k = 1 / (y_k^T s_k)
//
// Storage: O(md) where m = memory size (typically 5-20), d = dimension
// Compute H*v: O(md) via two-loop recursion
// =============================================================================

struct LBFGSState {
    int m;                                    // Memory size (number of pairs to store)
    int d;                                    // Dimension
    int k;                                    // Current iteration count
    std::vector<std::vector<double>> s_list; // Position differences (circular buffer)
    std::vector<std::vector<double>> y_list; // Gradient differences (circular buffer)
    std::vector<double> rho_list;            // 1 / (y^T s) values
    double gamma;                            // Scaling factor for initial H_0

    LBFGSState() : m(0), d(0), k(0), gamma(1.0) {}

    LBFGSState(int memory_size, int dimension)
        : m(memory_size), d(dimension), k(0), gamma(1.0) {
        s_list.reserve(m);
        y_list.reserve(m);
        rho_list.reserve(m);
    }

    // Add a new (s, y) pair from position and gradient differences
    void add_pair(const std::vector<double>& s, const std::vector<double>& y) {
        double ys = 0.0;
        double yy = 0.0;
        for (int i = 0; i < d; i++) {
            ys += y[i] * s[i];
            yy += y[i] * y[i];
        }

        // Skip if curvature condition not satisfied (ensures positive definiteness)
        if (ys < 1e-10) return;

        double rho = 1.0 / ys;

        // Update scaling factor: gamma = (s^T y) / (y^T y)
        if (yy > 1e-10) {
            gamma = ys / yy;
        }

        // Add to circular buffer
        if ((int)s_list.size() < m) {
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(rho);
        } else {
            // Circular replacement
            int idx = k % m;
            s_list[idx] = s;
            y_list[idx] = y;
            rho_list[idx] = rho;
        }
        k++;
    }

    // Two-loop recursion: compute H_k * v in O(md) time
    void multiply_H(const std::vector<double>& v, std::vector<double>& result) const {
        if (d <= 0 || (int)v.size() != d) {
            result = v;
            return;
        }

        result.resize(d);
        for (int i = 0; i < d; i++) {
            result[i] = v[i];
        }

        int n_stored = std::min(k, (int)s_list.size());
        n_stored = std::min(n_stored, m);

        if (n_stored == 0) {
            for (int i = 0; i < d; i++) {
                result[i] *= gamma;
            }
            return;
        }

        std::vector<double> alpha(n_stored);

        // First loop: from newest to oldest
        for (int i = n_stored - 1; i >= 0; i--) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                dot += s_list[idx][j] * result[j];
            }
            alpha[i] = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                result[j] -= alpha[i] * y_list[idx][j];
            }
        }

        // Apply initial Hessian: r = gamma * q
        for (int i = 0; i < d; i++) {
            result[i] *= gamma;
        }

        // Second loop: from oldest to newest
        for (int i = 0; i < n_stored; i++) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                dot += y_list[idx][j] * result[j];
            }
            double beta = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                result[j] += (alpha[i] - beta) * s_list[idx][j];
            }
        }
    }

    // Kinetic energy: K = 0.5 * p^T * H * p
    double kinetic_energy(const std::vector<double>& p) const {
        if ((int)p.size() != d) return 0.0;
        std::vector<double> Hp;
        multiply_H(p, Hp);
        double ke = 0.0;
        for (int i = 0; i < d; i++) {
            ke += p[i] * Hp[i];
        }
        return 0.5 * ke;
    }

    // Get diagonal of B for momentum sampling: sqrt(1/gamma)
    std::vector<double> get_sqrt_B_diag() const {
        std::vector<double> result(d);
        double sqrt_inv_gamma = std::sqrt(1.0 / gamma);
        for (int i = 0; i < d; i++) {
            result[i] = sqrt_inv_gamma;
        }
        return result;
    }
};

// Parse solver string from R
