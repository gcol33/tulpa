#include "device_batch.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <Eigen/Core>

namespace tulpa { namespace device {
namespace batch_detail {
thread_local BatchScope* active = nullptr;
thread_local bool cpu = false;

struct Frame {
    Context& context;
    Status& status;
    bool pushed;
    Frame(Context& c, Status& s) : context(c), status(s), pushed(false) {
        const auto result = context.push();
        pushed = bool(result); status.absorb(result);
    }
    ~Frame() { if (pushed) status.absorb(context.pop()); }
};

struct FactorWorkspace {
    Context& context = Context::instance();
    Status status;
    std::unique_ptr<Buffer> pointers, info;
    std::vector<unsigned long long> addresses;
    std::vector<int> codes;
    std::size_t capacity = 0;
    void reserve(std::size_t count) {
        if (count > capacity) {
            pointers.reset(new Buffer(context, count*sizeof(unsigned long long), status));
            info.reset(new Buffer(context, count*sizeof(int), status));
            capacity = count;
        }
        addresses.resize(count); codes.resize(count);
    }
    ~FactorWorkspace() {
        if (!capacity) return;
        Frame frame(context, status);
        if (frame.pushed) { pointers.reset(); info.reset(); }
    }
};

struct Request {
    unsigned long long address;
    int n;
    void* solver = nullptr;
    void* stream = nullptr;
    FactorWorkspace* workspace = nullptr;
    bool positive = false;
    Status status;
};

void factor_requests(const std::vector<Request*>& tasks, FactorWorkspace* batch_workspace = nullptr) {
    if (tasks.empty()) return;
    auto& work = tasks.size() == 1 ? *tasks[0]->workspace : *batch_workspace;
    auto& s = work.status;
    auto& context = Context::instance();
    auto run = [&] {
        work.reserve(tasks.size());
        for (std::size_t k = 0; k < tasks.size(); ++k) work.addresses[k] = tasks[k]->address;
        if (s) s.absorb(work.pointers->upload(work.addresses.data(), tasks.size()*sizeof(unsigned long long)));
        if (s) s.absorb(context.factor(work.pointers->address(), tasks[0]->n, tasks.size(), work.info->address(),
            tasks.size() == 1 ? tasks[0]->solver : nullptr));
        if (s) s.absorb(context.synchronize_stream(tasks.size() == 1 ? tasks[0]->stream : nullptr));
        if (s) s.absorb(work.info->download(work.codes.data(), tasks.size()*sizeof(int)));
    };
    if (tasks.size() > 1) { Lease lease(context, s); if (s) run(); }
    else { Frame frame(context, s); if (s) run(); }
    for (std::size_t k = 0; k < tasks.size(); ++k) {
        tasks[k]->status = s;
        tasks[k]->positive = bool(s) && work.codes[k] == 0;
        if (s && work.codes[k] < 0)
            tasks[k]->status = Status::read("cuSOLVER", work.codes[k], "potrf batch argument", std::to_string(k));
    }
    auto policy = current_policy();
    policy->record("laplace_dense_launch", true);
    if (tasks.size() > 1) policy->record("laplace_dense_batched_launch", true);
    else policy->record("laplace_dense_stream_launch", true);
}
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
    batch_detail::FactorWorkspace batch_workspace;
    std::map<void*, std::unique_ptr<batch_detail::FactorWorkspace>> stream_workspaces;
    explicit Impl(int w) : width(std::max(1, w)) {}
};
BatchScope::BatchScope(int width) : impl_(new Impl(width)) {}
BatchScope::~BatchScope() = default;
int BatchScope::width() const { return impl_->width; }
Status BatchScope::factor(unsigned long long address, int n, bool& positive, void* solver, void* stream,
                          const std::string& operation) {
    batch_detail::Request task{address, n, solver, stream};
    {
        std::lock_guard<std::mutex> lock(impl_->registry_mutex);
        auto& workspace = impl_->stream_workspaces[solver];
        if (!workspace) workspace.reset(new batch_detail::FactorWorkspace);
        task.workspace = workspace.get();
    }
    if (width() == 1 || !current_policy()->batch_selected(operation, n, width())) {
        batch_detail::factor_requests({&task});
        positive = task.positive;
        return task.status;
    }
    const auto key = reinterpret_cast<void*>(static_cast<std::uintptr_t>(n)*2 + 1);
    Status status = submit(key, &task, [this](const std::vector<void*>& items) {
        std::vector<batch_detail::Request*> tasks;
        for (auto* p : items) tasks.push_back(static_cast<batch_detail::Request*>(p));
        batch_detail::factor_requests(tasks, &impl_->batch_workspace);
    });
    positive = task.positive;
    status.absorb(task.status);
    return status;
}
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
            for (auto* t : tasks) t->status = Status::read("CUDA", -1, "batch", e.what());
        } catch (...) {
            for (auto* t : tasks) t->status = Status::read("CUDA", -1, "batch", "unknown exception");
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
    if (!acquire(group)) throw std::runtime_error("new batch has no factor slot");
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

namespace scatter_detail {
struct Request {
    unsigned long long design, vectors, output;
    int rows, columns;
    void* stream;
    Status status;
};
struct Workspace {
    Context& context = Context::instance();
    Status status;
    std::unique_ptr<Buffer> parameters;
    std::size_t capacity = 0;
    void run(const std::vector<Request*>& tasks) {
        if (tasks.empty()) return;
        batch_detail::Frame frame(context, status);
        const std::size_t bytes = tasks.size()*3*sizeof(unsigned long long);
        if (bytes > capacity) {
            parameters.reset(new Buffer(context, bytes, status)); capacity = bytes;
        }
        std::vector<unsigned long long> addresses;
        for (auto* t : tasks) {
            addresses.push_back(t->design); addresses.push_back(t->vectors); addresses.push_back(t->output);
        }
        if (status) status.absorb(parameters->upload(addresses.data(), bytes));
        auto p = parameters->address();
        int rows = tasks[0]->rows, columns = tasks[0]->columns;
        Launch shape; shape.block_x = 128;
        shape.grid_x = (static_cast<unsigned long long>(columns)*columns + 127)/128;
        shape.grid_y = tasks.size();
        shape.stream = tasks.size() == 1 ? tasks[0]->stream : nullptr;
        void* args[] = {&p, &rows, &columns};
        { Lease lease(context, status);
          if (status) status.absorb(context.launch(kernel_source("laplace_scatter"), "laplace_scatter", shape, args)); }
        if (status) status.absorb(context.synchronize_stream(shape.stream));
        for (auto* t : tasks) t->status = status;
        auto policy = current_policy();
        policy->record("laplace_scatter_launch", true);
        if (tasks.size() > 1) policy->record("laplace_scatter_batched_launch", true);
    }
    ~Workspace() {
        if (!parameters) return;
        batch_detail::Frame frame(context, status);
        if (frame.pushed) parameters.reset();
    }
};
}
struct DenseScatter::Impl {
    Context& context = Context::instance();
    Status status;
    const double* design = nullptr;
    int rows = 0, columns = 0;
    bool uploaded = false;
    bool cpu_bound = false;
    std::unique_ptr<Buffer> data, vectors, output;
    void* stream = nullptr;
    scatter_detail::Workspace workspace;
    std::vector<double> packed, result;
    Eigen::MatrixXd cpu_design, weighted;
    Eigen::VectorXd gradient;
    ~Impl() {
        if (!stream && !data && !vectors && !output) return;
        batch_detail::Frame frame(context, status);
        if (frame.pushed) {
            if (stream) status.absorb(context.synchronize_stream(stream));
            data.reset(); vectors.reset(); output.reset();
            if (stream) status.absorb(context.destroy_stream(stream));
        }
    }
};
DenseScatter::DenseScatter() : impl_(new Impl) {}
DenseScatter::~DenseScatter() = default;
void DenseScatter::bind(const double* design, int rows, int columns) {
    if (!impl_->status) impl_.reset(new Impl);
    auto& s = *impl_;
    if ((s.rows != rows || s.columns != columns) && (s.data || s.vectors || s.output)) {
        batch_detail::Frame frame(s.context, s.status);
        if (frame.pushed) { s.data.reset(); s.vectors.reset(); s.output.reset(); }
    }
    s.design = design; s.rows = rows; s.columns = columns;
    s.uploaded = false; s.cpu_bound = false;
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
    const bool cuda = policy->select("laplace_scatter",
        static_cast<std::size_t>(s.rows)*s.columns*s.columns, width, status);
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
    batch_detail::Frame frame(s.context, s.status);
    if (s.status && !s.stream) s.status.absorb(s.context.create_stream(s.stream));
    const std::size_t count = static_cast<std::size_t>(s.columns)*(s.columns + 1ULL);
    if (!s.data) {
        s.data.reset(new Buffer(s.context, static_cast<std::size_t>(s.rows)*s.columns*sizeof(double), s.status));
        s.vectors.reset(new Buffer(s.context, 2ULL*s.rows*sizeof(double), s.status));
        s.output.reset(new Buffer(s.context, count*sizeof(double), s.status));
    }
    if (s.status && !s.uploaded) {
        s.status.absorb(s.data->upload(s.design, s.data->size())); s.uploaded = bool(s.status);
    }
    s.packed.resize(2ULL*s.rows);
    std::copy_n(scores, s.rows, s.packed.begin());
    std::copy_n(weights, s.rows, s.packed.begin()+s.rows);
    if (s.status) s.status.absorb(s.vectors->upload(s.packed.data(), s.vectors->size()));
    scatter_detail::Request task{s.data->address(), s.vectors->address(), s.output->address(), s.rows, s.columns, s.stream};
    if (s.status && batch && width > 1 && policy->batch_selected("laplace_scatter",
        static_cast<std::size_t>(s.rows)*s.columns*s.columns, width)) {
        const std::string key = "dense_scatter:" + std::to_string(s.rows) + ":" + std::to_string(s.columns);
        auto group = std::static_pointer_cast<scatter_detail::Workspace>(batch->claim(key,
            []() -> std::shared_ptr<void> { return std::make_shared<scatter_detail::Workspace>(); },
            [](const std::shared_ptr<void>&) { return true; }));
        s.status.absorb(batch->submit(group.get(), &task, [&](const std::vector<void*>& items) {
            std::vector<scatter_detail::Request*> tasks;
            for (auto* t : items) tasks.push_back(static_cast<scatter_detail::Request*>(t));
            group->run(tasks);
        }));
    } else if (s.status) s.workspace.run({&task});
    s.status.absorb(task.status);
    s.result.resize(count);
    if (s.status) s.status.absorb(s.output->download(s.result.data(), s.output->size()));
    if (s.status) for (double x : s.result) if (!std::isfinite(x)) {
        s.status = Status::read("CUDA", -1, "dense scatter", "nonfinite observation contribution"); break;
    }
    if (s.status) {
        std::copy_n(s.result.data(), static_cast<std::size_t>(s.columns)*s.columns, lower);
        std::copy_n(s.result.data()+static_cast<std::size_t>(s.columns)*s.columns, s.columns, gradient);
    }
    policy->record("laplace_scatter", true);
    return s.status;
}
} }
