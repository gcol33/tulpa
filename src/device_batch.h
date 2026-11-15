#ifndef TULPA_DEVICE_BATCH_H
#define TULPA_DEVICE_BATCH_H

#include "device.h"
#include "device_dense.h"
#include <functional>
#include <stdexcept>

namespace tulpa { namespace device {

class CellFailure : public std::runtime_error {
public:
    explicit CellFailure(const Status& s) : std::runtime_error(s.message) {}
};

class BatchScope {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit BatchScope(int width);
    ~BatchScope();
    int width() const;
    Status factor(unsigned long long address, int n, bool& positive,
                  void* solver, void* stream, const std::string& operation);
    Status submit(const void* key, void* request,
                  const std::function<void(const std::vector<void*>&)>& process);
    std::shared_ptr<void> claim(const std::string& key,
        const std::function<std::shared_ptr<void>()>& create,
        const std::function<bool(const std::shared_ptr<void>&)>& acquire);
};

class CellScope {
    BatchScope* previous_;
    bool previous_cpu_;
public:
    explicit CellScope(BatchScope* batch, bool cpu = false);
    ~CellScope();
};

BatchScope* current_batch();
bool cell_cpu();
void cell_status(const Status& status);

template<class Solve>
auto run_cell(BatchScope& batch, Solve solve, const char* recovery_stage) -> decltype(solve()) {
    CellScope cell(&batch);
    try { return solve(); }
    catch (const CellFailure&) {
        CellScope cpu(&batch, true);
        auto result = solve();
        result.device_cpu_recovered = true;
        current_policy()->record(recovery_stage, false);
        return result;
    }
}

bool dense_laplace_selected(int n, const char* operation = "laplace_dense");

class DenseScatter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    DenseScatter();
    ~DenseScatter();
    void bind(const double* row_major_design, int rows, int columns);
    Status apply(const double* scores, const double* weights,
                 double* gradient, double* lower_row_major);
};

} }
#endif
