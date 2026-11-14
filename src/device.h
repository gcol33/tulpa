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

namespace tulpa { namespace device {

struct Status {
    int code = 0;
    std::string message;
    explicit operator bool() const { return code == 0; }
    static Status read(const char* library, int code, const char* operation,
                       const std::string& detail = "");
    void absorb(const Status& other) { if (*this && !other) *this = other; }
};

enum class Capability { Driver, Solver, Blas, Compiler, Sparse };
enum class Choice { Auto, Cpu, Cuda };

struct Operation {
    std::string name;
    std::vector<Capability> capabilities;
};
const std::vector<Operation>& operations();

struct Threshold {
    std::string operation;
    int variant;
    double size;
};
struct Stage {
    std::string operation;
    std::size_t cpu = 0, cuda = 0;
};

// A policy belongs to one fit; its counters and first failure are shared by
// the fit's outer-grid workers. No worker calls the R API.
class Policy {
    Choice choice_;
    std::vector<Threshold> thresholds_;
    mutable std::mutex mutex_;
    std::vector<Stage> stages_;
    Status failure_;
    std::map<std::tuple<std::thread::id, const void*, const void*, std::string>, std::shared_ptr<void>> workspaces_;
public:
    Policy(Choice choice, const std::vector<Threshold>& thresholds);
    bool select(const std::string& operation, std::size_t size, int variant,
                Status& status) const;
    void record(const std::string& operation, bool cuda);
    void fail(const Status& status);
    Status failure() const;
    std::vector<Stage> stages() const;
    Choice choice() const { return choice_; }
    std::shared_ptr<void> workspace(const void* owner, const void* context,
        const std::string& kind, const std::function<std::shared_ptr<void>()>& create);
};

std::shared_ptr<Policy> current_policy();
std::shared_ptr<Policy> exchange_policy(std::shared_ptr<Policy> policy);
void configure(const std::vector<Threshold>& thresholds,
               const std::map<std::string, std::string>& sources);
std::vector<Threshold> thresholds();
std::string kernel_source(const std::string& name);
Choice parse_choice(const std::string& value);

struct Info {
    bool compiled = true, available = false;
    int count = 0, driver_version = 0, init_count = 0;
    std::string reason, name, compute_capability;
    double memory_mb = 0;
    std::vector<std::string> libraries, library_reasons;
};

struct Launch {
    unsigned grid_x = 1, grid_y = 1, grid_z = 1;
    unsigned block_x = 256, block_y = 1, block_z = 1;
    unsigned shared_bytes = 0;
    void* stream = nullptr;
};

// CUDA types stay private to device.cpp. The public engine-internal interface
// uses integer device addresses and opaque handles, requiring no toolkit.
class Context {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Context();
public:
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    static Context& instance();
    Status initialize();
    Status supports(Capability capability);
    Status supports_operation(const std::string& operation);
    Info info();
    Status allocate(unsigned long long& pointer, std::size_t bytes);
    Status release(unsigned long long pointer);
    Status upload(unsigned long long pointer, const void* data, std::size_t bytes);
    Status download(void* data, unsigned long long pointer, std::size_t bytes);
    Status push();
    Status pop();
    Status synchronize();
    Status create_stream(void*& stream);
    Status destroy_stream(void* stream);
    Status synchronize_stream(void* stream);
    Status memory_available(std::size_t& free, std::size_t& total);
    Status gemv(unsigned long long matrix, unsigned long long x,
                unsigned long long y, int rows, int cols, bool transpose,
                double beta = 0, void* stream = nullptr);
    Status factor(unsigned long long pointers, int n, int batch,
                  unsigned long long info);
    Status launch(const std::string& source, const std::string& name,
                  const Launch& shape, void** arguments);
    Status compile(const std::string& source, const std::string& name,
                   void*& module, void*& kernel);
    Status unload(void* module);
    // Later backends resolve typed library entry points here and check every
    // result with Status::read. Handle use is protected by the same lease.
    void* symbol(Capability capability, const char* name, Status& status);
    void* handle(Capability capability, Status& status);
    std::mutex& execution_mutex();
};

class Lease {
    Context& context_;
    Status& status_;
    std::unique_lock<std::mutex> lock_;
    bool pushed_ = false;
public:
    Lease(Context& context, Status& status);
    ~Lease();
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
};

class Buffer {
    Context& context_;
    Status& status_;
    unsigned long long pointer_ = 0;
    std::size_t bytes_;
public:
    Buffer(Context& context, std::size_t bytes, Status& status);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    unsigned long long address() const { return pointer_; }
    std::size_t size() const { return bytes_; }
    Status upload(const void* data, std::size_t bytes);
    Status download(void* data, std::size_t bytes);
};

// Both routes return row-major lower factors. The input is committed only
// after all cuSOLVER per-matrix status codes and copies have succeeded.
Status cholesky_batch(std::vector<std::vector<double>>& matrices, int n,
                      Policy& policy, bool& used_cuda,
                      const std::vector<int>* widths = nullptr);
Status axpy(const std::vector<double>& x, std::vector<double>& y, double a,
            Policy& policy, bool& used_cuda);

} }
#endif
