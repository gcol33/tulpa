// tulpa_stop.h -- error raising through one compiled formatter.
//
// `tulpa::stop(fmt, {a, b, ...})` raises the same Rcpp::exception, with the
// same message, as `Rcpp::stop(fmt, a, b, ...)`. Rcpp's variadic form formats
// through tinyformat, a template over every argument list it is handed, so each
// distinct call signature compiles its own formatter and ostringstream chain.
// Here the arguments are converted at the call site into a list of `StopArg`
// (an integer, a double or a string), and one non-template function formats
// them.
//
// The conversions follow tinyformat: every conversion character accepts every
// argument; an integer prints as an integer whatever the conversion, a double
// under %f / %e / %g / %a takes that conversion with the stated flags, width
// and precision and under any other conversion prints as a stream would
// (precision 6, general notation), and a string prints as itself. Length
// modifiers (l, ll, z, h, j, t, L) are accepted and ignored.

#ifndef TULPA_STOP_H
#define TULPA_STOP_H

#include <initializer_list>
#include <string>

namespace tulpa {

class StopArg {
public:
    enum Kind { INT, UINT, REAL, STR };

    StopArg(bool v) : kind_(INT), i_(v) {}
    StopArg(int v) : kind_(INT), i_(v) {}
    StopArg(long v) : kind_(INT), i_(v) {}
    StopArg(long long v) : kind_(INT), i_(v) {}
    StopArg(unsigned v) : kind_(UINT), u_(v) {}
    StopArg(unsigned long v) : kind_(UINT), u_(v) {}
    StopArg(unsigned long long v) : kind_(UINT), u_(v) {}
    StopArg(double v) : kind_(REAL), d_(v) {}
    StopArg(const char* v) : kind_(STR), s_(v ? v : "(null)") {}
    StopArg(const std::string& v) : kind_(STR), s_(v.c_str()) {}

    Kind               kind() const { return kind_; }
    long long          as_int() const { return i_; }
    unsigned long long as_uint() const { return u_; }
    double             as_real() const { return d_; }
    const char*        as_str() const { return s_; }

private:
    Kind kind_;
    union {
        long long          i_;
        unsigned long long u_;
        double             d_;
        const char*        s_;
    };
};

std::string format_message(const char* fmt, std::initializer_list<StopArg> args);

[[noreturn]] void stop(const std::string& message);
[[noreturn]] void stop(const char* message);
[[noreturn]] void stop(const char* fmt, std::initializer_list<StopArg> args);

}  // namespace tulpa

#endif  // TULPA_STOP_H
