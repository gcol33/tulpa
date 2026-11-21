#include "device_batch.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <map>
#include <Eigen/Core>

namespace tulpa { namespace device {
namespace batch_detail {
thread_local BatchScope* active = nullptr;
thread_local bool cpu = false;
}

struct BatchScope::Impl {
    struct Work {
        const void* key;
        void* request;
        const std::function<void(const std::vector<void*>&)>* process;
        bool done = false;
        Status status;
    };
    int width;
    std::mutex mutex;
    std::condition_variable cv;
    bool leader = false;
    std::vector<Work*> pending;
    std::mutex registry_mutex;
    std::map<std::string, std::vector<std::shared_ptr<void>>> shared;
    explicit Impl(int w) : width(std::max(1, w)) {}
};
BatchScope::BatchScope(int width) : impl_(new Impl(width)) {}
BatchScope::~BatchScope() = default;
int BatchScope::width() const { return impl_->width; }
Status BatchScope::submit(const void* key, void* request,
    const std::function<void(const std::vector<void*>&)>& process) {
    Impl::Work task{key, request, &process};
    auto& q = *impl_;
    std::unique_lock<std::mutex> lock(q.mutex);
    q.pending.push_back(&task);
    q.cv.notify_all();
    while (!task.done) {
        if (q.leader) { q.cv.wait(lock); continue; }
        q.leader = true;
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(25);
        q.cv.wait_until(lock, until, [&] { return q.pending.size() >= static_cast<std::size_t>(q.width); });
        std::vector<Impl::Work*> tasks;
        const auto group = q.pending.front()->key;
        for (auto it = q.pending.begin(); it != q.pending.end();) {
            if ((*it)->key == group && tasks.size() < static_cast<std::size_t>(q.width)) { tasks.push_back(*it); it = q.pending.erase(it); }
            else ++it;
        }
        lock.unlock();
        try {
            std::vector<void*> items;
            for (auto* t : tasks) items.push_back(t->request);
            (*tasks[0]->process)(items);
        }
        catch (const std::exception& e) {
            for (auto* t : tasks) t->status = Status::read("device", -1, "batch", e.what());
        } catch (...) {
            for (auto* t : tasks) t->status = Status::read("device", -1, "batch", "unknown exception");
        }
        lock.lock();
        for (auto* t : tasks) t->done = true;
        q.leader = false;
        q.cv.notify_all();
    }
    return task.status;
}
std::shared_ptr<void> BatchScope::claim(const std::string& key,
    const std::function<std::shared_ptr<void>()>& create,
    const std::function<bool(const std::shared_ptr<void>&)>& acquire) {
    std::lock_guard<std::mutex> lock(impl_->registry_mutex);
    auto& groups = impl_->shared[key];
    for (auto& group : groups) if (acquire(group)) return group;
    auto group = create();
    if (!group || !acquire(group)) return nullptr;
    groups.push_back(group);
    return group;
}
BatchScope* current_batch() { return batch_detail::active; }
bool cell_cpu() { return batch_detail::cpu; }
void cell_status(const Status& status) {
    if (!status && current_batch()) throw CellFailure(status);
}
CellScope::CellScope(BatchScope* batch, bool cpu) : previous_(batch_detail::active), previous_cpu_(batch_detail::cpu) {
    batch_detail::active = batch; batch_detail::cpu = cpu;
}
CellScope::~CellScope() { batch_detail::active = previous_; batch_detail::cpu = previous_cpu_; }

bool dense_laplace_selected(int n, const char* operation) {
    if (cell_cpu()) return false;
    auto policy = current_policy();
    Status s;
    const int width = current_batch() ? current_batch()->width() : 1;
    const bool selected = policy->select(operation, n, width, s);
    cell_status(s);
    if (!s) policy->fail(s);
    return selected;
}

struct DenseScatter::Impl {
    const double* design = nullptr;
    int rows = 0, columns = 0;
    bool cpu_bound = false, route_bound = false;
    std::unique_ptr<DenseScatterRoute> route;
    Eigen::MatrixXd cpu_design, weighted;
    Eigen::VectorXd gradient;
};
DenseScatter::DenseScatter() : impl_(new Impl) {}
DenseScatter::~DenseScatter() = default;
void DenseScatter::bind(const double* design, int rows, int columns) {
    auto& s = *impl_;
    s.design = design; s.rows = rows; s.columns = columns;
    s.cpu_bound = false; s.route_bound = false;
}
Status DenseScatter::apply(const double* scores, const double* weights,
                           double* gradient, double* lower) {
    auto& s = *impl_;
    if (!s.design || s.rows < 1 || s.columns < 1)
        return Status::read("device", -1, "dense scatter", "no bound observation design");
    auto policy = current_policy();
    Status status;
    auto* batch = current_batch();
    const int width = batch ? batch->width() : 1;
    const std::size_t size = static_cast<std::size_t>(s.rows)*s.columns*s.columns;
    const bool cuda = policy->select("laplace_scatter", size, width, status);
    if (!status) return status;
    if (!cuda) {
        using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMatrix> design(s.design, s.rows, s.columns);
        Eigen::Map<const Eigen::VectorXd> w(weights, s.rows), g(scores, s.rows);
        if (!s.cpu_bound) { s.cpu_design = design; s.cpu_bound = true; }
        s.weighted = s.cpu_design.array().colwise()*w.array();
        s.gradient.noalias() = s.cpu_design.transpose()*g;
        std::fill_n(lower, static_cast<std::size_t>(s.columns)*s.columns, 0.0);
        for (int r = 0; r < s.columns; ++r) {
            gradient[r] = s.gradient[r];
            for (int c = 0; c <= r; ++c)
                lower[static_cast<std::size_t>(r)*s.columns+c] = s.cpu_design.col(r).dot(s.weighted.col(c));
        }
        policy->record("laplace_scatter", false);
        return status;
    }
    if (!s.route) {
        Backend* b = backend();
        if (!b) return backend_status();
        s.route = b->dense_scatter();
        if (!s.route) return Status::read("device", -1, "dense scatter", "the backend returned no route");
    }
    if (!s.route_bound) { s.route->bind(s.design, s.rows, s.columns); s.route_bound = true; }
    BatchView* group = batch && width > 1 && policy->batch_selected("laplace_scatter", size, width) ? batch : nullptr;
    status = s.route->apply(scores, weights, gradient, lower, group);
    // A failed route is discarded; the next device call starts from fresh state.
    if (!status) { s.route.reset(); s.route_bound = false; }
    policy->record("laplace_scatter", true);
    return status;
}
} }
