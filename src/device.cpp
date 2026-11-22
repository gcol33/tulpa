#include "device.h"
#include "device_batch.h"
#include "linalg_fast.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <R_ext/Rdynload.h>

namespace tulpa { namespace device {

namespace detail {
std::shared_ptr<Policy> active_policy;
std::atomic<Backend*> registered{nullptr};

class EngineHost final : public Host {
public:
    std::shared_ptr<PolicyView> policy() override { return current_policy(); }
    BatchView* batch() override { return current_batch(); }
};
EngineHost host;

const Operation* registered_operation(const std::string& name) {
    const auto& table = operations();
    auto found = std::find_if(table.begin(), table.end(), [&](const Operation& op) { return op.name == name; });
    return found == table.end() ? nullptr : &*found;
}

Host* register_backend(Backend* candidate, const BackendLayout* layout, char* reason, int reason_size) {
    const BackendLayout expected = backend_layout();
    if (!candidate || !layout || std::memcmp(layout, &expected, sizeof(expected)) != 0) {
        if (reason && reason_size > 0) {
            const std::string message = "the device backend was built against device interface " +
                std::to_string(layout ? layout->abi : -1) + "; tulpa provides " +
                std::to_string(expected.abi) + " with these standard-library layouts. Reinstall it against this tulpa.";
            std::strncpy(reason, message.c_str(), static_cast<std::size_t>(reason_size) - 1);
            reason[reason_size - 1] = '\0';
        }
        return nullptr;
    }
    registered.store(candidate);
    return &host;
}

void unregister_backend(Backend* candidate) {
    Backend* expected = candidate;
    registered.compare_exchange_strong(expected, nullptr);
}
}
using namespace detail;

Backend* backend() { return registered.load(); }

Status backend_status() {
    Backend* b = backend();
    if (!b) return Status::read("tulpa", -1, "device",
        "CUDA routes are provided by the tulpaGPU package, which is not loaded");
    return b->initialize();
}

Choice parse_choice(const std::string& value) {
    if (value == "auto") return Choice::Auto;
    if (value == "cpu") return Choice::Cpu;
    if (value == "cuda") return Choice::Cuda;
    throw std::invalid_argument("control$device must be one of 'auto', 'cpu', 'cuda'");
}
std::vector<Threshold> thresholds() {
    Backend* b = backend();
    return b ? b->thresholds() : std::vector<Threshold>{};
}
std::shared_ptr<Policy> current_policy() {
    auto result = std::atomic_load(&active_policy);
    return result ? result : std::make_shared<Policy>(Choice::Auto, thresholds());
}
std::shared_ptr<Policy> exchange_policy(std::shared_ptr<Policy> policy) {
    return std::atomic_exchange(&active_policy, std::move(policy));
}
Policy::Policy(Choice choice, const std::vector<Threshold>& thresholds) : choice_(choice), thresholds_(thresholds) {
    const char* profile = std::getenv("TULPA_DEVICE_PROFILE");
    profiling_ = profile && std::string(profile) == "true";
}
void Policy::time(const std::string& stage, double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& timing = timings_[stage];
    timing.stage = stage; ++timing.calls; timing.seconds += seconds;
}
std::vector<Timing> Policy::timings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Timing> out;
    for (const auto& item : timings_) out.push_back(item.second);
    return out;
}
thread_local StageTimer* StageTimer::active_ = nullptr;
StageTimer::StageTimer(Policy& policy, const char* stage) : stage_(stage) {
    if (!policy.profiling() || (active_ && active_->policy_ == &policy &&
        std::string(active_->stage_) == stage)) return;
    policy_ = &policy; parent_ = active_; active_ = this;
    start_ = std::chrono::steady_clock::now();
}
StageTimer::~StageTimer() {
    if (!policy_) return;
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();
    active_ = parent_;
    if (parent_) parent_->children_ += elapsed;
    policy_->time(stage_, std::max(0.0, elapsed-children_));
}
bool Policy::select(const std::string& name, std::size_t size, int variant, Status& status) const {
    const auto* operation = registered_operation(name);
    if (!operation) { status = Status::read("policy", -1, "select", "unregistered operation " + name); return false; }
    if (choice_ == Choice::Cpu || cell_cpu() || !size) return false;
    if (choice_ == Choice::Auto) {
        auto threshold = std::find_if(thresholds_.begin(), thresholds_.end(), [&](const Threshold& x) {
            return x.operation == name && x.variant == variant;
        });
        if (threshold == thresholds_.end() || size < threshold->size || !std::isfinite(threshold->size)) return false;
    }
    Backend* b = backend();
    Status available = backend_status();
    for (auto capability : operation->capabilities) {
        if (!available) break;
        available = b->supports(capability);
    }
    if (!available) {
        if (choice_ == Choice::Cuda) status = available;
        return false;
    }
    return true;
}
bool Policy::batch_selected(const std::string& name, std::size_t size, int variant) const {
    if (choice_ == Choice::Cpu || cell_cpu()) return false;
    if (choice_ == Choice::Cuda) return true;
    auto setting = std::find_if(thresholds_.begin(), thresholds_.end(), [&](const Threshold& x) {
        return x.operation == name && x.variant == variant;
    });
    return setting != thresholds_.end() && std::isfinite(setting->batch_size) && size >= setting->batch_size;
}
namespace {
// The stage entry named `name`, appended with zero counters when absent. The
// caller holds the policy's mutex.
Stage& find_or_insert_stage(std::vector<Stage>& stages, const std::string& name) {
    auto stage = std::find_if(stages.begin(), stages.end(), [&](const Stage& x) { return x.operation == name; });
    if (stage == stages.end()) { stages.push_back({name, 0, 0}); return stages.back(); }
    return *stage;
}
}
void Policy::record(const std::string& name, bool cuda) {
    record(name, cuda ? 0 : 1, cuda ? 1 : 0);
}
void Policy::record(const std::string& name, std::size_t cpu, std::size_t cuda) {
    std::lock_guard<std::mutex> lock(mutex_);
    Stage& stage = find_or_insert_stage(stages_, name);
    stage.cpu += cpu;
    stage.cuda += cuda;
}
void Policy::work(const std::string& name, double amount) {
    std::lock_guard<std::mutex> lock(mutex_);
    Stage& stage = find_or_insert_stage(stages_, name);
    stage.work = std::max(stage.work, amount);
}
void Policy::fail(const Status& status) { cell_status(status); note(status); }
void Policy::note(const Status& status) { std::lock_guard<std::mutex> lock(mutex_); failure_.absorb(status); }
Status Policy::failure() const { std::lock_guard<std::mutex> lock(mutex_); return failure_; }
std::vector<Stage> Policy::stages() const { std::lock_guard<std::mutex> lock(mutex_); return stages_; }
std::shared_ptr<void> Policy::workspace(const void* owner, const void* context,
    const std::string& kind, const std::function<std::shared_ptr<void>()>& create) {
    const auto key = std::make_tuple(std::this_thread::get_id(),owner,context,kind);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = workspaces_.find(key);
        if (found != workspaces_.end()) return found->second;
    }
    auto value = create();
    std::lock_guard<std::mutex> lock(mutex_);
    return workspaces_.emplace(key,std::move(value)).first->second;
}

Status cholesky_batch(std::vector<std::vector<double>>& matrices, int n,
                       Policy& policy, bool& used_cuda,
                       const std::vector<int>* widths) {
    used_cuda = false;
    Status s;
    if (n <= 0 || matrices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return Status::read("device", -1, "cholesky_batch", "invalid batch dimensions");
    const std::size_t stride = static_cast<std::size_t>(n) * n;
    for (const auto& matrix : matrices) if (matrix.size() != stride)
        return Status::read("device", -1, "cholesky_batch", "matrix size differs from n*n");
    if (widths) {
        if (widths->size() != matrices.size())
            return Status::read("device", -1, "cholesky_batch", "batch widths differ from batch size");
        for (int width : *widths) if (width <= 0 || width > n)
            return Status::read("device", -1, "cholesky_batch", "invalid effective matrix width");
    }
    if (matrices.empty()) return s;
    const bool cuda = policy.select("nngp_cholesky", matrices.size(), n, s);
    if (!s) return s;
    if (cuda) {
        std::vector<double> packed(matrices.size() * stride);
        for (std::size_t b = 0; b < matrices.size(); ++b)
            std::copy(matrices[b].begin(), matrices[b].end(), packed.begin() + b * stride);
        Backend* b = backend();
        s = b ? b->cholesky_batch(packed.data(), n, static_cast<int>(matrices.size())) : backend_status();
        if (!s) return s;
        // Positive pivots and the diagonal of L L' check every matrix. One
        // complete CPU factor also verifies the column-major handoff.
        for (std::size_t b = 0; b < matrices.size(); ++b) {
            const double* L = packed.data() + b * stride;
            for (int r = 0; r < n; ++r) {
                double diagonal = 0;
                for (int c = 0; c <= r; ++c) diagonal += L[c * n + r] * L[c * n + r];
                if (!(L[r * n + r] > 0) || !std::isfinite(diagonal) ||
                    std::abs(diagonal - matrices[b][r * n + r]) >
                        1e-8 * std::max(std::abs(matrices[b][r * n + r]), 1e-12))
                    return Status::read("cuSOLVER", -1, "verify factor", "batch index " + std::to_string(b));
            }
        }
        auto reference = matrices[matrices.size() / 2];
        if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(reference.data(), reference.data(), n, n, 0.0))
            return Status::read("CPU", -1, "verify factor", "reference is not positive definite");
        const double* probe = packed.data() + (matrices.size() / 2) * stride;
        double scale = 0, difference = 0;
        for (int r = 0; r < n; ++r) for (int c = 0; c <= r; ++c) {
            scale = std::max(scale, std::abs(reference[r * n + c]));
            difference = std::max(difference, std::abs(reference[r * n + c] - probe[c * n + r]));
        }
        if (!(difference <= 1e-8 * std::max(scale, 1e-12)))
            return Status::read("cuSOLVER", -1, "verify factor", "CPU factor differs");
        for (std::size_t b = 0; b < matrices.size(); ++b) {
            auto& L = matrices[b];
            const double* column_major = packed.data() + b * stride;
            for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
                L[r * n + c] = c <= r ? column_major[c * n + r] : 0.0;
        }
        used_cuda = true;
    } else {
        for (std::size_t b = 0; b < matrices.size(); ++b) {
            auto& L = matrices[b];
            const int width = widths ? (*widths)[b] : n;
            if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(L.data(), L.data(), width, n, 0.0))
                return Status::read("CPU", 1, "cholesky_batch", "matrix is not positive definite");
            for (int r = 0; r < n; ++r) for (int c = r + 1; c < n; ++c) L[r * n + c] = 0.0;
        }
    }
    policy.record("nngp_cholesky", used_cuda);
    return s;
}

Status axpy(const std::vector<double>& x, std::vector<double>& y, double a,
             Policy& policy, bool& used_cuda) {
    Status s;
    used_cuda = false;
    if (x.size() != y.size() || x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return Status::read("device", -1, "axpy", "incompatible vector sizes");
    const bool cuda = policy.select("axpy", x.size(), 0, s);
    if (!s) return s;
    if (cuda) {
        Backend* b = backend();
        s = b ? b->axpy(x.data(), y.data(), a, static_cast<int>(x.size())) : backend_status();
        if (!s) return s;
        used_cuda = true;
    } else {
        for (std::size_t i = 0; i < x.size(); ++i) y[i] += a * x[i];
    }
    policy.record("axpy", used_cuda);
    return s;
}

} }

void tulpa_register_device_callables(DllInfo*) {
    R_RegisterCCallable("tulpa", "tulpa_device_register",
                        reinterpret_cast<DL_FUNC>(&tulpa::device::detail::register_backend));
    R_RegisterCCallable("tulpa", "tulpa_device_unregister",
                        reinterpret_cast<DL_FUNC>(&tulpa::device::detail::unregister_backend));
}
