// checkpoint_io.cpp -- the payload-independent file layer of CheckpointLog.

#include "checkpoint_io.h"
#include "tulpa_stop.h"

#include <cerrno>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace tulpa {

namespace {

const char* errno_message() { return std::strerror(errno); }

bool read_whole_file(const std::string& path, std::string& out, bool& opened) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    opened = f != nullptr;
    if (!f) return false;
    char buf[1 << 16];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    return ok;
}

bool truncate_file(std::FILE* f, std::uint64_t bytes) {
#ifdef _WIN32
    return _chsize_s(_fileno(f), static_cast<long long>(bytes)) == 0;
#else
    return ftruncate(fileno(f), static_cast<off_t>(bytes)) == 0;
#endif
}

}  // namespace

std::vector<CheckpointFile::Record> CheckpointFile::read(
    const std::string& path, const char* magic, std::uint64_t fingerprint,
    bool& exists, std::uint64_t& header_bytes) {
    std::vector<Record> records;
    std::string data;
    bool opened = false;
    const bool ok = read_whole_file(path, data, opened);
    exists = opened && !data.empty();
    header_bytes = 8 + sizeof(std::uint64_t);
    if (!exists) return records;
    if (!ok) {
        tulpa::stop("checkpoint: cannot read '%s' (%s).", {path.c_str(),
                   errno_message()});
    }

    CkptReader rd(data.data(), data.size());
    if (data.size() < 8 || std::memcmp(data.data(), magic, 8) != 0) {
        tulpa::stop("checkpoint: '%s' is not a tulpa checkpoint file "
                   "(bad magic). Point `checkpoint$path` at a fresh "
                   "location or remove the file.", {path.c_str()});
    }
    rd.p += 8;
    const std::uint64_t stored_fp = rd.get<std::uint64_t>();
    if (!rd.ok) tulpa::stop("checkpoint: '%s' header is truncated.", {path.c_str()});
    if (stored_fp != fingerprint) {
        tulpa::stop("checkpoint: '%s' was written for different data or "
                   "solver settings (fingerprint mismatch). Resume needs "
                   "the same inputs + grid + control; use a fresh path "
                   "or set checkpoint$resume = FALSE to start over.",
                   {path.c_str()});
    }

    while (rd.p < rd.end) {
        const std::uint32_t key_len = rd.get<std::uint32_t>();
        if (!rd.ok || key_len > static_cast<std::size_t>(rd.end - rd.p)) break;
        std::string key(rd.p, key_len);
        rd.p += key_len;
        const std::uint32_t pay_len = rd.get<std::uint32_t>();
        if (!rd.ok || pay_len > static_cast<std::size_t>(rd.end - rd.p)) break;
        std::string payload(rd.p, pay_len);
        rd.p += pay_len;
        const std::uint64_t chk = rd.get<std::uint64_t>();
        if (!rd.ok) break;
        const std::uint64_t want = fnv1a64(payload.data(), payload.size(),
                                           fnv1a64(key.data(), key.size()));
        if (chk != want) break;
        records.push_back(Record{std::move(key), std::move(payload),
                                 static_cast<std::uint64_t>(rd.p - data.data())});
    }
    return records;
}

CheckpointFile::~CheckpointFile() {
    if (f_) std::fclose(f_);
}

void CheckpointFile::create(const std::string& path, const char* magic,
                            std::uint64_t fingerprint) {
    path_ = path;
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_ || std::fwrite(magic, 1, 8, f_) != 8 ||
        std::fwrite(&fingerprint, sizeof(fingerprint), 1, f_) != 1 ||
        std::fflush(f_) != 0) {
        tulpa::stop("checkpoint: cannot open '%s' for writing (%s).",
                   {path.c_str(), errno_message()});
    }
}

void CheckpointFile::reopen(const std::string& path, std::uint64_t keep_bytes) {
    path_ = path;
    f_ = std::fopen(path.c_str(), "r+b");
    if (!f_) {
        tulpa::stop("checkpoint: cannot open '%s' for writing (%s).",
                   {path.c_str(), errno_message()});
    }
    // A failed truncation would leave every record this run appends behind the
    // orphaned bytes, written but invisible to the next load.
    if (!truncate_file(f_, keep_bytes)) {
        tulpa::stop("checkpoint: cannot truncate the torn tail of '%s' "
                   "to %s bytes (%s). Every record appended after an "
                   "untruncated tail is invisible to the next resume; "
                   "remove the file or point `checkpoint$path` elsewhere.",
                   {path.c_str(), std::to_string(keep_bytes).c_str(),
                   errno_message()});
    }
    if (std::fseek(f_, 0, SEEK_END) != 0) {
        tulpa::stop("checkpoint: cannot seek in '%s' (%s).", {path.c_str(),
                   errno_message()});
    }
}

void CheckpointFile::append(const std::string& key, const std::string& payload) {
    const std::uint32_t key_len = static_cast<std::uint32_t>(key.size());
    const std::uint32_t pay_len = static_cast<std::uint32_t>(payload.size());
    const std::uint64_t chk = fnv1a64(payload.data(), payload.size(),
                                      fnv1a64(key.data(), key.size()));
    std::string rec;
    rec.reserve(sizeof(key_len) + key.size() + sizeof(pay_len) +
                payload.size() + sizeof(chk));
    ckpt_put(rec, key_len);
    rec.append(key);
    ckpt_put(rec, pay_len);
    rec.append(payload);
    ckpt_put(rec, chk);
    std::fwrite(rec.data(), 1, rec.size(), f_);
    std::fflush(f_);
}

}  // namespace tulpa
