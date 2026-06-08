// nested_progress.h
// Observable progress reporting for long nested-Laplace outer-grid fits.
//
// A nested-Laplace fit iterates an inner Laplace solve over an outer
// hyperparameter grid (one cell per (tau, rho, sigma, ...) candidate). On a
// detached/headless box a large grid runs for minutes to hours with no
// liveness signal beyond OS CPU time: the only other output, Rcpp::Rcout
// "verbose" text, is block-buffered when R's stdout is redirected to a file
// rather than a TTY (the normal case under Start-Process / nohup), so the log
// shows only the pre-loop header until the process exits.
//
// GridProgress is the fix. Its two channels are independently gated, because a
// detached run (the case the file channel exists for) is exactly the case that
// wants the console quiet:
//   * a console line via Rcpp::Rcout + R_FlushConsole() -- best effort, for
//     interactive/TTY runs; emitted only when `emit_console` is set, and
//     suppressed inside an OpenMP parallel region because the R print API is
//     not worker-thread safe;
//   * a heartbeat file, overwritten and fflush+fclose'd each tick, holding the
//     machine-readable "<done> <total> <elapsed_s> <eta_s>". Written whenever
//     `heartbeat_file` is non-empty, independent of `emit_console`. A file
//     write survives stdout redirection where an Rcout flush does not, so the
//     file is the robust signal for detached runs -- the configuration that
//     also turns the console off.
//
// The caller constructs an instance when either channel is wanted
// (emit_console || !heartbeat_file.empty()); a quiet console with a heartbeat
// file is the headless default.
//
// One instance lives for the duration of a single grid run. tick() is called
// once per completed outer cell; under OpenMP the caller must invoke tick()
// from inside a critical section (the counter and the file write are not
// otherwise synchronised).

#ifndef TULPA_NESTED_PROGRESS_H
#define TULPA_NESTED_PROGRESS_H

#include <Rcpp.h>      // Rcpp::Rcout, and R.h (R_FlushConsole) transitively
#include <chrono>
#include <string>
#include <cstdio>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa_progress {

// Human-readable duration: seconds under 90 s, minutes under 90 min, hours
// above. ASCII only (logs, redirects, CRAN).
inline std::string format_secs(double s) {
    if (!std::isfinite(s) || s < 0.0) return "?";
    char buf[32];
    if (s < 90.0)        std::snprintf(buf, sizeof(buf), "%.0fs", s);
    else if (s < 5400.0) std::snprintf(buf, sizeof(buf), "%.1fmin", s / 60.0);
    else                 std::snprintf(buf, sizeof(buf), "%.1fh", s / 3600.0);
    return std::string(buf);
}

class GridProgress {
public:
    // label          : short tag shown in the console line (e.g. "nested-laplace").
    // total          : number of outer cells expected (the denominator / ETA base).
    //                  May be revised via set_total() once pruning is known.
    // every          : emit after this many newly-completed cells (<= 0 -> time only).
    // throttle_s     : minimum seconds between emits (<= 0 -> cell-count only).
    // heartbeat_file : path overwritten each emit ("" -> no file).
    // emit_console   : print the Rcout progress line (the noisy TTY channel).
    //                  The heartbeat file is written regardless.
    // unit           : noun for one completed step in the console line ("cells"
    //                  for an outer grid, "iter" for an EM / NUTS loop). The
    //                  heartbeat-file format is unit-independent.
    GridProgress(const std::string& label, int total,
                 int every, double throttle_s,
                 const std::string& heartbeat_file,
                 bool emit_console = true,
                 const std::string& unit = "cells")
        : label_(label),
          total_(total > 0 ? total : 0),
          every_(every),
          throttle_s_(throttle_s),
          file_(heartbeat_file),
          emit_console_(emit_console),
          unit_(unit),
          done_(0),
          last_emit_done_(0),
          width_(1),
          start_(std::chrono::steady_clock::now()),
          last_(start_) {}

    // Revise the denominator (e.g. after cheap-pass pruning drops cells from
    // the full-solve set). done_ is left untouched.
    void set_total(int total) { if (total > 0) total_ = total; }

    // Outer-grid concurrency: how many cells run at once in the parallel
    // region. The ETA needs it because the pilot cell runs serially and the
    // remaining cells run `width` at a time; extrapolating the serial pilot
    // rate across every remaining cell over-states the wall-clock by ~`width`x
    // (a 21-min pilot on a 48-cell grid projects to ~16 h instead of the real
    // ~1.5 h). It also drives the "| N threads" console suffix when > 1, so the
    // active outer-thread count is visible in the line (gcol33/tulpa#88).
    // Defaults to 1 (the serial path, unchanged: no suffix).
    void set_width(int width) { width_ = (width > 0) ? width : 1; }

    // One completed outer cell. Under OpenMP, call from a critical section.
    void tick() {
        done_++;
        maybe_emit(false);
    }

    // Force a final emit (100% line + heartbeat) at the end of the grid.
    void finish() { maybe_emit(true); }

private:
    void maybe_emit(bool force) {
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - last_).count();
        bool first    = (done_ == 1);
        bool by_cells = (every_ > 0) && (done_ - last_emit_done_ >= every_);
        bool by_time  = (throttle_s_ > 0.0) && (since >= throttle_s_);
        if (!force && !first && !by_cells && !by_time) return;

        double elapsed = std::chrono::duration<double>(now - start_).count();
        double frac = (total_ > 0) ? static_cast<double>(done_) / total_ : 0.0;
        // ETA from the per-cell *wall* time, estimated from completed waves
        // rather than completed cells: one serial pilot wave plus (done-1)/width
        // parallel waves. Remaining cells finish ceil((total-done)/width) waves
        // ahead. At width == 1 this is exactly the old serial extrapolation
        // (waves == done, eta == elapsed*(1-frac)/frac); at width > 1 it stops
        // the serial pilot rate from inflating the projection across the
        // parallel grid, and it stays correct once parallel throughput is
        // reached (`per` below is the serial-equivalent per-cell wall time).
        double w = static_cast<double>(width_ > 0 ? width_ : 1);
        double waves_done = 1.0 + (done_ - 1.0) / w;            // >= 1 for done >= 1
        double per = (waves_done > 0.0) ? elapsed / waves_done : 0.0;
        double remaining_waves =
            (total_ > done_) ? std::ceil((total_ - done_) / w) : 0.0;
        double eta = (frac < 1.0) ? remaining_waves * per : 0.0;

        // Heartbeat file: reliable across stdout redirection. Overwrite +
        // fflush + fclose each tick so a detached reader always sees the
        // latest "<done> <total> <elapsed_s> <eta_s>".
        if (!file_.empty()) {
            std::FILE* f = std::fopen(file_.c_str(), "w");
            if (f) {
                std::fprintf(f, "%d %d %.1f %.1f\n", done_, total_, elapsed, eta);
                std::fflush(f);
                std::fclose(f);
            }
        }

        // Console line: only when requested (emit_console_), and only from the
        // serial/master context. Rcpp::Rcout and R_FlushConsole touch the R
        // print API, which is not safe to call from an OpenMP worker thread; the
        // heartbeat file above is the parallel-run (and headless) signal.
        bool in_parallel = false;
#ifdef _OPENMP
        in_parallel = (omp_in_parallel() != 0);
#endif
        if (emit_console_ && !in_parallel) {
            int pct = (total_ > 0)
                          ? static_cast<int>(frac * 100.0 + 0.5) : 0;
            // Outer concurrency suffix: width_ is the realised number of units
            // run at once (outer grid cells per nested-Laplace wave, parallel
            // chains for NUTS), set via set_width(). Shown only when > 1 so a
            // serial loop's line is unchanged and "ran on N cores" becomes a
            // property of the log itself (gcol33/tulpa#88).
            char thr[32] = "";
            if (width_ > 1)
                std::snprintf(thr, sizeof(thr), " | %d threads", width_);
            char line[288];
            std::snprintf(line, sizeof(line),
                "[%s] %d/%d %s (%d%%) | elapsed %s | ETA ~%s | %.2fs/%s%s",
                label_.c_str(), done_, total_, unit_.c_str(), pct,
                format_secs(elapsed).c_str(),
                (frac >= 1.0) ? "done" : format_secs(eta).c_str(),
                per, unit_.c_str(), thr);
            Rcpp::Rcout << line << "\n";
            Rcpp::Rcout.flush();
            R_FlushConsole();
        }

        last_ = now;
        last_emit_done_ = done_;
    }

    std::string label_;
    int    total_;
    int    every_;
    double throttle_s_;
    std::string file_;
    bool   emit_console_;
    std::string unit_;
    int    done_;
    int    last_emit_done_;
    int    width_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_;
};

}  // namespace tulpa_progress

#endif  // TULPA_NESTED_PROGRESS_H
