#ifndef TULPA_ENGINE_DEVICE_LIKELIHOOD_H
#define TULPA_ENGINE_DEVICE_LIKELIHOOD_H

#include "device.h"
#include "tulpa/likelihood.h"
#include <utility>
#include <stdexcept>

namespace tulpa { namespace device {

struct Projection {
    double constant = 0;
    std::vector<std::pair<int,double>> terms;
    explicit Projection(double value = 0) : constant(value) {}
    static Projection leaf(int index) {
        Projection out; out.terms.emplace_back(index,1); return out;
    }
    Projection& operator+=(const Projection& rhs) {
        constant+=rhs.constant;
        terms.insert(terms.end(),rhs.terms.begin(),rhs.terms.end()); return *this;
    }
    Projection& operator+=(double rhs) { constant+=rhs; return *this; }
};
inline Projection operator+(const Projection& a, const Projection& b) {
    Projection out(a.constant+b.constant);
    out.terms=a.terms; out.terms.insert(out.terms.end(),b.terms.begin(),b.terms.end());
    return out;
}
inline Projection operator-(const Projection& a, const Projection& b) {
    Projection out(a.constant-b.constant); out.terms=a.terms;
    for (const auto& term : b.terms) out.terms.emplace_back(term.first,-term.second);
    return out;
}
inline Projection operator*(const Projection& a, const Projection& b) {
    if (!a.terms.empty() && !b.terms.empty())
        throw std::logic_error("observation projection must be affine in reconstructed fields");
    Projection out(a.constant*b.constant);
    for (const auto& term : a.terms) out.terms.emplace_back(term.first,term.second*b.constant);
    for (const auto& term : b.terms) out.terms.emplace_back(term.first,term.second*a.constant);
    return out;
}
inline Projection operator*(double a, const Projection& b) { return Projection(a)*b; }
inline Projection operator*(const Projection& a, double b) { return a*Projection(b); }

// The dense columns and sparse rows share one coefficient vector. Sparse
// transpose segments reduce each field score without unordered atomic adds.
struct ObservationPlan {
    int n = 0, np = 1, width = 0;
    std::vector<int> beta_start, beta_count, rowptr, column;
    std::vector<double> x, offset, weight;
    void add_design(const double* matrix, int p, int start, bool column_major=false) {
        beta_start.push_back(start); beta_count.push_back(p);
        x.reserve(x.size()+static_cast<std::size_t>(n)*p);
        for (int j=0; j<p; ++j) for (int i=0; i<n; ++i)
            x.push_back(matrix[column_major ? static_cast<std::size_t>(j)*n+i : static_cast<std::size_t>(i)*p+j]);
    }
    void append_row(const Projection& row, double additional_offset=0) {
        if (rowptr.empty()) {
            offset.reserve(static_cast<std::size_t>(n)*np);
            rowptr.reserve(static_cast<std::size_t>(n)*np+1);
            rowptr.push_back(0);
        }
        offset.push_back(row.constant+additional_offset);
        for (const auto& term : row.terms) if (term.second!=0) {
            column.push_back(term.first); weight.push_back(term.second);
        }
        rowptr.push_back(static_cast<int>(column.size()));
    }
};

struct LikelihoodResult {
    std::vector<double> value, gradient, score, curvature, eta;
};

class LikelihoodWorkspace {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    LikelihoodWorkspace(ObservationPlan plan, DeviceLikelihood kernel);
    ~LikelihoodWorkspace();
    const ObservationPlan& plan() const;
    DeviceLikelihood& kernel();
    Status refresh_projection(ObservationPlan plan);
    Status evaluate(const std::vector<double>& parameters, int chains,
        bool batched, bool weights, LikelihoodResult& result);
};

bool builtin_device_likelihood(const ModelData& data, const ParamLayout& layout,
    const void* response, bool initialize, DeviceLikelihood& kernel);

bool select_likelihood(const ModelData& data, const ParamLayout& layout,
    DeviceLikelihood& kernel, Policy& policy, Status& status, bool weights=false);

} }
#endif
