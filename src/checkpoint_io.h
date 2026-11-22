// checkpoint_io.h
// Generic content-addressed append-log checkpoint.
//
// The outer loop of a long fit -- a hyperparameter grid (nested Laplace), a
// covariance-CCD node grid, or a set of independent MCMC chains -- is a list of
// independent units, each producing a self-contained result. This file owns the
// shared on-disk format and the load/append/torn-tail logic ONCE, as a template
// over the per-unit payload type. A consumer specializes it by providing two
// free functions for its payload, found by ADL:
//
//     std::string ckpt_serialize(const Payload&);
//     bool        ckpt_deserialize(CkptReader&, Payload&);
//
// and a `using MyCheckpoint = CheckpointLog<Payload>;`. GridCheckpoint
// (LaplaceResult) and ChainCheckpoint (HMCResultCpp) are the two instantiations.
//
// File format (little-endian, host-native; not portable across endianness,
// acceptable for a resume-on-the-same-machine workflow):
//
//   Header (once):
//     char[8]  magic          (the payload-layout tag, MAGIC below)
//     uint64   fingerprint
//   Record (appended per completed unit):
//     uint32   key_len
//     bytes    key            (raw bytes of the unit coordinate)
//     uint32   payload_len
//     bytes    payload        (ckpt_serialize output)
//     uint64   checksum       (FNV-1a over key ++ payload)
//
// A torn final record (process killed mid-write) is detected on load by a short
// read or a checksum mismatch and discarded; that unit is simply re-run. A
// header fingerprint mismatch errors rather than resuming onto a stale result.

#ifndef TULPA_CHECKPOINT_IO_H
#define TULPA_CHECKPOINT_IO_H

#include <Rcpp.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tulpa {

// FNV-1a 64-bit. Used for both the header fingerprint and per-record checksums.
inline std::uint64_t fnv1a64(const void* data, std::size_t n,
                             std::uint64_t h = 1469598103934665603ULL) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Append a POD value's raw bytes to a byte buffer.
template <typename T>
inline void ckpt_put(std::string& buf, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "ckpt_put requires a trivially-copyable type");
    buf.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

// Append a contiguous span of POD values (length-prefixed by the caller).
template <typename T>
inline void ckpt_put_span(std::string& buf, const std::vector<T>& v) {
    std::uint64_t n = v.size();
    ckpt_put(buf, n);
    if (n) buf.append(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}

// Append a std::string, length-prefixed.
inline void ckpt_put_str(std::string& buf, const std::string& v) {
    std::uint64_t n = v.size();
    ckpt_put(buf, n);
    if (n) buf.append(v.data(), n);
}

// Running FNV-1a fingerprint of everything that changes a unit's result given
// its coordinate (data, designs, solver settings, layout). Each fitter folds
// its own fields; the value guards the checkpoint header. fold_vec writes only
// the data bytes (no length prefix) -- callers fold the dimensions that
// disambiguate lengths separately.
struct Fingerprint {
    std::uint64_t h = 1469598103934665603ULL;

    void fold(const void* d, std::size_t n) { h = fnv1a64(d, n, h); }

    template <typename T>
    void fold_pod(const T& v) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "fold_pod requires a trivially-copyable type");
        fold(&v, sizeof(T));
    }

    void fold_str(const std::string& s) { fold(s.data(), s.size()); }

    // Fold an R vector / matrix by its own element type. `fold(x.begin(), n *
    // sizeof(T))` takes a const void*, so a `double*` handed a `sizeof(int)`
    // length converts silently and folds the LEADING HALF of the buffer -- two
    // inputs agreeing on their first half then share a fingerprint, and a
    // resume replays one run's cells under the other's data. These overloads
    // are the way to fold an R vector; the raw `fold` is for everything else.
    void fold_rvec(const Rcpp::NumericVector& v) {
        if (v.size()) fold(v.begin(), (std::size_t)v.size() * sizeof(double));
    }
    void fold_rvec(const Rcpp::IntegerVector& v) {
        if (v.size()) fold(v.begin(), (std::size_t)v.size() * sizeof(int));
    }
    void fold_rvec(const Rcpp::NumericMatrix& m) {
        if (m.size()) fold(m.begin(), (std::size_t)m.size() * sizeof(double));
    }
    void fold_rvec(const Rcpp::IntegerMatrix& m) {
        if (m.size()) fold(m.begin(), (std::size_t)m.size() * sizeof(int));
    }
    // An absent optional input still has to separate itself from a present one,
    // otherwise "no offset" and "offset = 0" fingerprint alike.
    void fold_rvec_nullable(const Rcpp::Nullable<Rcpp::NumericVector>& v) {
        const bool present = v.isNotNull();
        fold_pod(present);
        if (present) fold_rvec(Rcpp::NumericVector(v));
    }

    template <typename T>
    void fold_vec(const std::vector<T>& v) {
        if (!v.empty()) fold(v.data(), v.size() * sizeof(T));
    }

    std::uint64_t value() const { return h; }
};

// Recursively fold an arbitrary R object's contents into a fingerprint, so a
// heterogeneous spec (adjacency, coords, type strings, nested lists) invalidates
// a stale resume without enumerating its fields by hand.
inline void fold_sexp(Fingerprint& fp, SEXP x) {
    int type = TYPEOF(x);
    fp.fold_pod(type);
    switch (type) {
        case REALSXP: {
            R_xlen_t n = Rf_xlength(x);
            if (n) fp.fold(REAL(x), (std::size_t)n * sizeof(double));
            break;
        }
        case INTSXP:
        case LGLSXP: {
            R_xlen_t n = Rf_xlength(x);
            if (n) fp.fold(INTEGER(x), (std::size_t)n * sizeof(int));
            break;
        }
        case STRSXP: {
            R_xlen_t n = Rf_xlength(x);
            for (R_xlen_t i = 0; i < n; i++) {
                const char* c = CHAR(STRING_ELT(x, i));
                fp.fold(c, std::strlen(c));
            }
            break;
        }
        case VECSXP: {
            R_xlen_t n = Rf_xlength(x);
            for (R_xlen_t i = 0; i < n; i++) fold_sexp(fp, VECTOR_ELT(x, i));
            break;
        }
        default:
            break;
    }
}

// Builds the per-cell keys for a rectilinear (or paired) outer grid. Each axis
// is a length-n_grid column of per-cell coordinate values; key[k] is the
// concatenated raw double bytes of every axis at k.
class CellKeyBuilder {
public:
    explicit CellKeyBuilder(int n_grid) : keys_(n_grid) {}

    void add_axis(const double* col, std::size_t stride = 1) {
        for (std::size_t k = 0; k < keys_.size(); k++) {
            double v = col[k * stride];
            keys_[k].append(reinterpret_cast<const char*>(&v), sizeof(double));
        }
    }

    void add_axis(const std::vector<double>& col) {
        // A degenerate (constant) axis carries no per-cell values -- e.g. rw1 /
        // rw2 pass an empty rho_grid. It contributes nothing to the keys;
        // reading n_grid doubles from an empty column would run off the buffer.
        if (col.empty()) return;
        add_axis(col.data(), 1);
    }

    std::vector<std::string> take() { return std::move(keys_); }

private:
    std::vector<std::string> keys_;
};

// Cursor-based reader over a byte buffer. Every read is bounds-checked; a read
// past the end sets `ok = false` so a truncated tail is detected, not faulted.
struct CkptReader {
    const char* p;
    const char* end;
    bool        ok = true;

    CkptReader(const char* data, std::size_t n) : p(data), end(data + n) {}

    template <typename T>
    T get() {
        T v{};
        if (!ok || p + sizeof(T) > end) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    std::string get_str() {
        std::string v;
        std::uint64_t n = get<std::uint64_t>();
        if (!ok) return v;
        std::size_t remaining = static_cast<std::size_t>(end - p);
        if (n > remaining) { ok = false; return v; }
        v.assign(p, static_cast<std::size_t>(n));
        p += static_cast<std::size_t>(n);
        return v;
    }

    template <typename T>
    std::vector<T> get_span() {
        std::vector<T> v;
        std::uint64_t n = get<std::uint64_t>();
        if (!ok) return v;
        // Cap n against the bytes actually remaining BEFORE computing
        // n * sizeof(T) (which could overflow a uint64 and wrap past the check)
        // or resizing (a torn/corrupt length field would otherwise allocate
        // gigabytes). A length that cannot fit is a truncated tail, not a value.
        std::size_t remaining = static_cast<std::size_t>(end - p);
        if (n > remaining / sizeof(T)) { ok = false; return v; }
        std::size_t nn = static_cast<std::size_t>(n);
        v.resize(nn);
        if (nn) {
            std::memcpy(v.data(), p, nn * sizeof(T));
            p += nn * sizeof(T);
        }
        return v;
    }
};

// The on-disk side of a checkpoint, independent of the payload type. `read`
// returns every record whose checksum holds, in file order, each with the byte
// offset just past it, stopping at the first short read or checksum mismatch
// (a torn tail). It errors on a bad magic, a truncated header or a fingerprint
// mismatch, and reports a missing or zero-byte file as `exists = false`: an
// interrupted run cannot leave a zero-byte file, since the header is flushed
// on create. `open` then either creates the file with its header or truncates
// it to `keep_bytes` and appends from there.
class CheckpointFile {
public:
    struct Record {
        std::string   key;
        std::string   payload;
        std::uint64_t end;
    };

    static std::vector<Record> read(const std::string& path, const char* magic,
                                    std::uint64_t fingerprint, bool& exists,
                                    std::uint64_t& header_bytes);

    CheckpointFile() = default;
    CheckpointFile(const CheckpointFile&) = delete;
    CheckpointFile& operator=(const CheckpointFile&) = delete;
    ~CheckpointFile();

    void create(const std::string& path, const char* magic,
                std::uint64_t fingerprint);
    void reopen(const std::string& path, std::uint64_t keep_bytes);
    void append(const std::string& key, const std::string& payload);

private:
    std::FILE*  f_ = nullptr;
    std::string path_;
};

// Customization points: each payload header provides a non-template overload
// (`ckpt_serialize(const LaplaceResult&)`, etc.) found by overload resolution at
// instantiation. These primary templates make the names visible at the template
// definition below; they are never instantiated (a non-template overload always
// wins) and so are intentionally left undefined.
template <typename T> std::string ckpt_serialize(const T&);
template <typename T> bool        ckpt_deserialize(CkptReader&, T&);

// Generic per-fit checkpoint handle over a payload type. Constructed once per
// run where the unit coordinates (keys) are known; passed by pointer into the
// outer loop. `ckpt_serialize` / `ckpt_deserialize` for Payload must be visible
// (overload resolution) at instantiation.
template <typename Payload>
class CheckpointLog {
public:
    // Bumped whenever a payload's serialized layout changes: the fingerprint
    // covers the INPUTS a cell was solved from, not the field list a record
    // carries, so a file written by an older layout would otherwise be replayed
    // field-by-field into the new one and mis-parsed. A magic mismatch errors
    // and points the user at a fresh path.
    static constexpr char MAGIC[8] = {'T','L','P','A','C','K','P','E'};

    CheckpointLog(const std::string& path,
                  std::uint64_t fingerprint,
                  std::vector<std::string> keys)
        : keys_(std::move(keys)) {
        bool exists = false;
        std::uint64_t good_bytes = 0;
        std::vector<CheckpointFile::Record> records =
            CheckpointFile::read(path, MAGIC, fingerprint, exists, good_bytes);
        if (!exists) {
            file_.create(path, MAGIC, fingerprint);
            return;
        }
        // A record that does not parse ends the log like a torn one: it and
        // everything after it is dropped and re-run.
        for (CheckpointFile::Record& rec : records) {
            CkptReader rd(rec.payload.data(), rec.payload.size());
            Payload r;
            if (!ckpt_deserialize(rd, r)) break;
            loaded_[std::move(rec.key)] = std::move(r);
            good_bytes = rec.end;
        }
        // Appending after a torn tail would leave every new record past the
        // orphaned bytes, where the next load stops before reaching them.
        file_.reopen(path, good_bytes);
    }

    // Every accessor takes mu_, not save() alone: an emplace that rehashes
    // relocates the bucket array a concurrent find() is walking. References to
    // the elements themselves survive a rehash (the map is node-based), so
    // get() may hand its reference out past the lock. The current consumer
    // reads the log serially before its parallel region and only save()s from
    // a worker; locking the readers is what lets the next one do otherwise.
    bool has(int k) const {
        if (k < 0 || k >= static_cast<int>(keys_.size())) return false;
        std::lock_guard<std::mutex> lock(mu_);
        return loaded_.find(keys_[k]) != loaded_.end();
    }

    const Payload& get(int k) const {
        std::lock_guard<std::mutex> lock(mu_);
        return loaded_.at(keys_[k]);
    }

    // Append unit k's completed result. Thread-safe: the outer loop may call
    // this from inside an OpenMP parallel region.
    void save(int k, const Payload& res) {
        if (k < 0 || k >= static_cast<int>(keys_.size())) return;
        const std::string& key = keys_[k];
        std::string payload = ckpt_serialize(res);
        std::lock_guard<std::mutex> lock(mu_);
        if (loaded_.find(key) != loaded_.end()) return;  // idempotent
        file_.append(key, payload);
        loaded_.emplace(key, res);
    }

private:
    std::vector<std::string>                 keys_;
    std::unordered_map<std::string, Payload> loaded_;
    CheckpointFile                           file_;
    mutable std::mutex                       mu_;
};

template <typename Payload>
constexpr char CheckpointLog<Payload>::MAGIC[8];

} // namespace tulpa

#endif // TULPA_CHECKPOINT_IO_H
