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
//   * a console line via Rcpp::Rcout + R_FlushConsole() -- emitted when
//     `emit_console` is set. Inside an OpenMP parallel region only the master
//     thread (thread 0, which is the R main thread) emits; worker threads never
//     touch the R print API (not worker-thread safe) and only update the
//     counter + heartbeat file. So the console advances newline by newline as
//     the master completes cells, rather than freezing at the serial pilot
// until finish;
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
          last_(start_),
          last_console_(start_),
          last_console_done_(0),
          pilot_elapsed_(-1.0) {}

    // Revise the denominator (e.g. after cheap-pass pruning drops cells from
    // the full-solve set). done_ is left untouched.
    void set_total(int total) { if (total > 0) total_ = total; }

    // Outer-grid concurrency: how many cells run at once in the parallel
    // region. Used for the "| N threads" console suffix when > 1, so the active
    // outer-thread count is visible in the line, and as the
    // wave width for the pilot-only ETA lower bound before any parallel cell has
    // completed (see maybe_emit). Once parallel cells finish, the ETA rests on
    // their realised throughput, which already folds the width in. Defaults to 1
    // (the serial path: no suffix, plain extrapolation).
    void set_width(int width) { width_ = (width > 0) ? width : 1; }

    // One completed outer cell. Under OpenMP, call from a critical section.
    void tick() {
        done_++;
        maybe_emit(false);
    }

    // Force a final emit (100% line + heartbeat) at the end of the grid.
    void finish() { maybe_emit(true); }

    // Announce a one-off setup phase that runs BEFORE the cell loop (e.g. the
    // sparsity-pattern build of a wide joint field, which can take seconds at
    // EVA scale). tick() only fires per completed cell, so without this the
    // console and the heartbeat file are silent through the whole pre-grid
    // setup and the fit looks hung. Emits the console line
    // when enabled and, if a heartbeat file is configured, writes a 0/total
    // record so a detached reader sees liveness instead of an empty file.
    // Leaves done_ / the emit cadence untouched. Call from the serial setup
    // context (the R print API is not OpenMP-worker safe).
    void note(const std::string& phase) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_).count();
        if (!file_.empty()) {
            std::FILE* f = std::fopen(file_.c_str(), "w");
            if (f) {
                std::fprintf(f, "%d %d %.1f %.1f\n", 0, total_, elapsed, 0.0);
                std::fflush(f);
                std::fclose(f);
            }
        }
        bool in_parallel = false;
#ifdef _OPENMP
        in_parallel = (omp_in_parallel() != 0);
#endif
        if (emit_console_ && !in_parallel) {
            Rcpp::Rcout << "[" << label_ << "] " << phase << "\n";
            Rcpp::Rcout.flush();
            R_FlushConsole();
        }
    }

private:
    void maybe_emit(bool force) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_).count();
        // The first completed cell is the serial pilot under the parallel path
        // (a cheap, warm, central grid point); record its wall time once so the
        // ETA below can measure the realised parallel throughput net of it.
        if (pilot_elapsed_ < 0.0 && done_ >= 1) pilot_elapsed_ = elapsed;

        bool first    = (done_ == 1);
        bool by_cells = (every_ > 0) && (done_ - last_emit_done_ >= every_);
        double since_file = std::chrono::duration<double>(now - last_).count();
        bool by_time  = (throttle_s_ > 0.0) && (since_file >= throttle_s_);
        bool do_file  = force || first || by_cells || by_time;

        // The console emits only from the serial/master context: Rcpp::Rcout and
        // R_FlushConsole touch the R print API, safe on the R main thread but not
        // on an OpenMP worker. Inside the outer-grid parallel region the master
        // thread (thread 0 == the R main thread) still emits, so a detached /
        // redirected run shows the line advancing instead of frozen at the pilot
        //. Its cadence rides a SEPARATE clock so that worker
        // ticks updating the heartbeat file do not keep resetting the throttle
        // window out from under the master and starve the console.
        bool in_parallel = false;
        bool is_master   = true;
#ifdef _OPENMP
        in_parallel = (omp_in_parallel() != 0);
        is_master   = (omp_get_thread_num() == 0);
#endif
        double since_console =
            std::chrono::duration<double>(now - last_console_).count();
        bool console_by_time  = (throttle_s_ > 0.0) && (since_console >= throttle_s_);
        bool console_by_cells = (every_ > 0) && (done_ - last_console_done_ >= every_);
        bool do_console = emit_console_ && (!in_parallel || is_master) &&
                          (force || first || console_by_time || console_by_cells);

        if (!do_file && !do_console) return;

        double frac = (total_ > 0) ? static_cast<double>(done_) / total_ : 0.0;

        // ETA from the realised per-cell *wall* time of completed cells, not the
        // serial pilot. The pilot is the central grid cell, solved serially and
        // warm; the parallel cells are extreme-hyperparameter, need more inner
        // Newton steps, and run under memory-bandwidth contention, so each costs
        // well more than the pilot. Projecting the pilot rate across the grid
        // under-states the wall clock by roughly that ratio.
        //   * once >= 1 cell beyond the pilot has finished, `per` is the mean
        //     wall time per post-pilot cell (elapsed-since-pilot over cells-
        //     since-pilot) -- the throughput the remaining cells actually run
        //     at, the width already folded in -- and the ETA is a point estimate;
        //   * until then only the cheap pilot is timed, so the projection
        //     (ceil(remaining / width) pilot-cost waves) is a LOWER bound, shown
        //     as "ETA >=" so it is not read as a point estimate.
        // At width == 1 (serial) the post-pilot mean equals the per-cell time and
        // this reduces to the plain serial extrapolation.
        double per;
        double eta;
        bool   eta_is_lower_bound;
        double post_done    = static_cast<double>(done_) - 1.0;
        double post_elapsed = elapsed - pilot_elapsed_;
        if (frac >= 1.0) {
            per = (done_ > 0) ? elapsed / done_ : 0.0;
            eta = 0.0;
            eta_is_lower_bound = false;
        } else if (post_done >= 1.0 && post_elapsed > 1e-9) {
            per = post_elapsed / post_done;
            eta = (total_ - done_) * per;
            eta_is_lower_bound = false;
        } else {
            double w = static_cast<double>(width_ > 0 ? width_ : 1);
            per = (done_ > 0) ? elapsed / done_ : 0.0;
            double remaining_waves =
                (total_ > done_) ? std::ceil((total_ - done_) / w) : 0.0;
            eta = remaining_waves * per;
            eta_is_lower_bound = true;
        }

        // Heartbeat file: reliable across stdout redirection. Overwrite +
        // fflush + fclose each tick so a detached reader always sees the latest
        // "<done> <total> <elapsed_s> <eta_s>". The wire format is unchanged
        // (four numbers); the lower-bound vs point-estimate distinction is a
        // console-only annotation so existing readers keep parsing.
        if (do_file) {
            if (!file_.empty()) {
                std::FILE* f = std::fopen(file_.c_str(), "w");
                if (f) {
                    std::fprintf(f, "%d %d %.1f %.1f\n",
                                 done_, total_, elapsed, eta);
                    std::fflush(f);
                    std::fclose(f);
                }
            }
            last_ = now;
            last_emit_done_ = done_;
        }

        if (do_console) {
            int pct = (total_ > 0)
                          ? static_cast<int>(frac * 100.0 + 0.5) : 0;
            // Outer concurrency suffix: width_ is the realised number of units
            // run at once (outer grid cells per nested-Laplace wave, parallel
            // chains for NUTS), set via set_width(). Shown only when > 1 so a
            // serial loop's line is unchanged and "ran on N cores" becomes a
            // property of the log itself.
            char thr[32] = "";
            if (width_ > 1)
                std::snprintf(thr, sizeof(thr), " | %d threads", width_);
            char line[300];
            std::snprintf(line, sizeof(line),
                "[%s] %d/%d %s (%d%%) | elapsed %s | ETA %s%s | %.2fs/%s%s",
                label_.c_str(), done_, total_, unit_.c_str(), pct,
                format_secs(elapsed).c_str(),
                (frac >= 1.0) ? "" : (eta_is_lower_bound ? ">=" : "~"),
                (frac >= 1.0) ? "done" : format_secs(eta).c_str(),
                per, unit_.c_str(), thr);
            Rcpp::Rcout << line << "\n";
            Rcpp::Rcout.flush();
            R_FlushConsole();
            last_console_ = now;
            last_console_done_ = done_;
        }
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
    std::chrono::steady_clock::time_point last_console_;
    int    last_console_done_;
    double pilot_elapsed_;
};

}  // namespace tulpa_progress

#endif  // TULPA_NESTED_PROGRESS_H
