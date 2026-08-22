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

    // Pairs currently held in the memory window.
    int n_stored() const {
        int ns = std::min(k, (int)s_list.size());
        return std::min(ns, m);
    }

    // Circular-buffer slot of the i-th pair, counting from the oldest held.
    int pair_slot(int i, int ns) const {
        int idx = (k - ns + i) % m;
        if (idx < 0) idx += m;
        return idx;
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
            int idx = pair_slot(i, n_stored);
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
            int idx = pair_slot(i, n_stored);
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

    // Direct (Hessian-side) BFGS matrix B = H^{-1}. Both are built from the
    // same pairs, in the same order, from B_0 = H_0^{-1} = (1/gamma) I, and the
    // direct and inverse BFGS updates are mutual inverses, so B_j = H_j^{-1}
    // holds at every step of the recursion.
    //
    // The update B_{j+1} = B_j - b_j b_j' / (s_j' b_j) + y_j y_j' / (y_j' s_j)
    // needs only b_j = B_j s_j, since s_j' B_j v = b_j' v. Those vectors are the
    // whole representation: with them in hand, B_k applied to any vector is the
    // unrolled sum below, and a draw from N(0, B_k) walks the same updates.
    struct DirectFactors {
        int n = 0;
        std::vector<std::vector<double> > b;  // b_j = B_j s_j
        std::vector<double> sbs;              // s_j' B_j s_j
        std::vector<double> ys;               // y_j' s_j
        std::vector<int> slot;                // circular-buffer index of pair j
        double b0 = 1.0;                      // B_0 = b0 * I
        bool ok = false;
    };

    // Apply the first `j` updates: B_j v = b0 v + sum_{t<j} [ -b_t (b_t'v)/sbs_t
    //                                                        + y_t (y_t'v)/ys_t ]
    void apply_direct(const DirectFactors& f, int j, const double* v,
                      std::vector<double>& out) const {
        out.assign(d, 0.0);
        for (int i = 0; i < d; i++) out[i] = f.b0 * v[i];
        for (int t = 0; t < j; t++) {
            const std::vector<double>& yt = y_list[f.slot[t]];
            double bv = 0.0, yv = 0.0;
            for (int i = 0; i < d; i++) {
                bv += f.b[t][i] * v[i];
                yv += yt[i] * v[i];
            }
            double c1 = bv / f.sbs[t];
            double c2 = yv / f.ys[t];
            for (int i = 0; i < d; i++) {
                out[i] += -c1 * f.b[t][i] + c2 * yt[i];
            }
        }
    }

    void build_direct_factors(DirectFactors& f) const {
        f.ok = false;
        if (d <= 0 || !(gamma > 0.0)) return;
        int ns = n_stored();
        f.n = ns;
        f.b0 = 1.0 / gamma;
        f.b.assign(ns, std::vector<double>());
        f.sbs.assign(ns, 0.0);
        f.ys.assign(ns, 0.0);
        f.slot.assign(ns, 0);
        std::vector<double> bi;
        for (int i = 0; i < ns; i++) {
            int idx = pair_slot(i, ns);
            if (idx >= (int)s_list.size()) return;
            f.slot[i] = idx;
            const std::vector<double>& s = s_list[idx];
            if ((int)s.size() != d || (int)y_list[idx].size() != d) return;
            apply_direct(f, i, s.data(), bi);
            double sbs = 0.0;
            for (int j = 0; j < d; j++) sbs += s[j] * bi[j];
            if (!(sbs > 0.0)) return;
            f.b[i] = bi;
            f.sbs[i] = sbs;
            f.ys[i] = 1.0 / rho_list[idx];
            if (!(f.ys[i] > 0.0)) return;
        }
        f.ok = true;
    }

    // B_k * v, the direct counterpart of multiply_H.
    void multiply_B(const std::vector<double>& v, std::vector<double>& result) const {
        if (d <= 0 || (int)v.size() != d) {
            result = v;
            return;
        }
        std::vector<double> vin = v;
        DirectFactors f;
        build_direct_factors(f);
        if (!f.ok) {
            result.assign(d, 0.0);
            double b0 = (gamma > 0.0) ? 1.0 / gamma : 1.0;
            for (int i = 0; i < d; i++) result[i] = b0 * vin[i];
            return;
        }
        apply_direct(f, f.n, vin.data(), result);
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

    // Momentum refresh for the metric this state defines. The drift applies
    // M^{-1} = H and the kinetic energy is 0.5 p' H p, so a valid refresh draws
    // p ~ N(0, M) with M = H^{-1} = B.
    //
    // The draw walks the direct BFGS recursion. Starting from u ~ N(0, B_0):
    // projecting with P_j = I - b_j s_j' / (s_j' b_j) gives
    // P_j B_j P_j' = B_j - b_j b_j' / (s_j' b_j), exactly the rank-one downdate,
    // and adding y_j w / sqrt(y_j' s_j) with w ~ N(0, 1) adds y_j y_j' / (y_j's_j),
    // exactly the rank-one update. So Cov(u) = B_k after k steps.
    void sample_momentum(std::vector<double>& p, std::mt19937& rng) const {
        std::normal_distribution<double> normal(0.0, 1.0);
        p.assign(d > 0 ? d : 0, 0.0);
        if (d <= 0) return;
        double sd0 = std::sqrt((gamma > 0.0) ? 1.0 / gamma : 1.0);
        for (int i = 0; i < d; i++) p[i] = normal(rng) * sd0;

        DirectFactors f;
        build_direct_factors(f);
        if (!f.ok) return;

        for (int j = 0; j < f.n; j++) {
            const std::vector<double>& s = s_list[f.slot[j]];
            const std::vector<double>& y = y_list[f.slot[j]];
            double su = 0.0;
            for (int i = 0; i < d; i++) su += s[i] * p[i];
            double c = su / f.sbs[j];
            double w = normal(rng) / std::sqrt(f.ys[j]);
            for (int i = 0; i < d; i++) {
                p[i] += -c * f.b[j][i] + w * y[i];
            }
        }
    }
};
