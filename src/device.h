#ifndef TULPA_DEVICE_H
#define TULPA_DEVICE_H

#include <cstddef>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <tuple>
#include <chrono>
#include "tulpa/device_backend.h"

namespace tulpa { namespace device {

struct Stage {
    std::string operation;
    std::size_t cpu = 0, cuda = 0;
    double work = 0;
};
struct Timing {
    std::string stage;
    std::size_t calls = 0;
    double seconds = 0;
};

// A policy belongs to one fit; its counters and first failure are shared by
// the fit's outer-grid workers. No worker calls the R API.
class Policy final : public PolicyView {
    Choice choice_;
    std::vector<Threshold> thresholds_;
    mutable std::mutex mutex_;
    std::vector<Stage> stages_;
    Status failure_;
    bool profiling_ = false;
    std::map<std::string, Timing> timings_;
    std::map<std::tuple<std::thread::id, const void*, const void*, std::string>, std::shared_ptr<void>> workspaces_;
public:
    Policy(Choice choice, const std::vector<Threshold>& thresholds);
    virtual ~Policy() = default;
    bool select(const std::string& operation, std::size_t size, int variant,
                Status& status) const override;
    bool batch_selected(const std::string& operation, std::size_t size, int variant) const override;
    void record(const std::string& operation, bool cuda) override;
    void record(const std::string& operation, std::size_t cpu, std::size_t cuda);
    // Largest work (the operation's gate unit) seen for the operation in this fit.
    void work(const std::string& operation, double amount);
    // Records the failure and, inside an outer-grid cell, raises CellFailure so
    // the cell is solved again on the CPU.
    void fail(const Status& status);
    void note(const Status& status) override;
    Status failure() const;
    std::vector<Stage> stages() const;
    bool profiling() const { return profiling_; }
    void time(const std::string& stage, double seconds);
    std::vector<Timing> timings() const;
    Choice choice() const { return choice_; }
    std::shared_ptr<void> workspace(const void* owner, const void* context,
        const std::string& kind, const std::function<std::shared_ptr<void>()>& create) override;
};

// Nested timings are exclusive; selected inversion excludes its host factor,
// and a nested solve is counted only once.
class StageTimer {
    Policy* policy_ = nullptr;
    StageTimer* parent_ = nullptr;
    const char* stage_;
    std::chrono::steady_clock::time_point start_;
    double children_ = 0;
    static thread_local StageTimer* active_;
public:
    StageTimer(Policy& policy, const char* stage);
    ~StageTimer();
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;
};

std::shared_ptr<Policy> current_policy();
std::shared_ptr<Policy> exchange_policy(std::shared_ptr<Policy> policy);
// Thresholds of the registered backend; none without one.
std::vector<Threshold> thresholds();
Choice parse_choice(const std::string& value);

// The registered accelerator backend, or null. A route obtained from it stays
// valid after the backend is unregistered.
Backend* backend();
// Why CUDA is unavailable, empty when a backend is registered and initialized.
Status backend_status();

// Both routes return row-major lower factors. The input is committed only
// after every per-matrix status and copy has succeeded.
Status cholesky_batch(std::vector<std::vector<double>>& matrices, int n,
                      Policy& policy, bool& used_cuda,
                      const std::vector<int>* widths = nullptr);
Status axpy(const std::vector<double>& x, std::vector<double>& y, double a,
            Policy& policy, bool& used_cuda);

} }
#endif
