// tulpa_stop.cpp -- the one formatter behind tulpa::stop.

#include "tulpa_stop.h"

#include <Rcpp.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace tulpa {

namespace {

void append_printf(std::string& out, const char* spec, ...)
    __attribute__((format(printf, 2, 3)));

void append_printf(std::string& out, const char* spec, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, spec);
    const int n = std::vsnprintf(buf, sizeof(buf), spec, ap);
    va_end(ap);
    if (n < 0) return;
    if (static_cast<std::size_t>(n) < sizeof(buf)) {
        out.append(buf, static_cast<std::size_t>(n));
        return;
    }
    std::string big(static_cast<std::size_t>(n) + 1, '\0');
    va_start(ap, spec);
    std::vsnprintf(&big[0], big.size(), spec, ap);
    va_end(ap);
    big.resize(static_cast<std::size_t>(n));
    out += big;
}

// One argument under one conversion. `flags` holds the flag, width and
// precision characters of the directive, without the leading '%', the length
// modifier or the conversion.
void format_arg(std::string& out, const std::string& flags, char conv,
                const StopArg& a) {
    std::string spec = "%" + flags;
    if (a.kind() == StopArg::INT || a.kind() == StopArg::UINT) {
        // A stream prints an integer without a precision.
        const std::size_t dot = spec.find('.');
        if (dot != std::string::npos) spec.erase(dot);
    }
    switch (a.kind()) {
        case StopArg::INT:
            if (conv == 'x' || conv == 'X' || conv == 'o') {
                append_printf(out, (spec + "ll" + conv).c_str(), a.as_int());
            } else {
                append_printf(out, (spec + "lld").c_str(), a.as_int());
            }
            break;
        case StopArg::UINT:
            if (conv == 'x' || conv == 'X' || conv == 'o') {
                append_printf(out, (spec + "ll" + conv).c_str(), a.as_uint());
            } else {
                append_printf(out, (spec + "llu").c_str(), a.as_uint());
            }
            break;
        case StopArg::REAL: {
            const bool float_conv = std::strchr("fFeEgGaA", conv) != nullptr;
            if (float_conv) {
                append_printf(out, (spec + conv).c_str(), a.as_real());
            } else {
                const bool has_precision = flags.find('.') != std::string::npos;
                append_printf(out, (spec + (has_precision ? "g" : ".6g")).c_str(),
                              a.as_real());
            }
            break;
        }
        case StopArg::STR:
            append_printf(out, (spec + "s").c_str(), a.as_str());
            break;
    }
}

}  // namespace

std::string format_message(const char* fmt, std::initializer_list<StopArg> args) {
    std::string out;
    const StopArg* next = args.begin();
    const char* p = fmt;
    while (*p) {
        if (*p != '%') {
            const char* q = std::strchr(p, '%');
            if (!q) q = p + std::strlen(p);
            out.append(p, static_cast<std::size_t>(q - p));
            p = q;
            continue;
        }
        if (p[1] == '%') {
            out += '%';
            p += 2;
            continue;
        }
        const char* q = p + 1;
        while (*q && std::strchr("-+ #0", *q)) q++;
        while (*q >= '0' && *q <= '9') q++;
        if (*q == '.') {
            q++;
            while (*q >= '0' && *q <= '9') q++;
        }
        const std::string flags(p + 1, q);
        while (*q && std::strchr("hljztL", *q)) q++;
        if (!*q) {
            out.append(p);
            break;
        }
        const char conv = *q;
        if (next == args.end()) {
            throw Rcpp::exception(
                ("tulpa::stop: too few arguments for format \"" +
                 std::string(fmt) + "\".").c_str());
        }
        format_arg(out, flags, conv, *next++);
        p = q + 1;
    }
    return out;
}

void stop(const std::string& message) {
    throw Rcpp::exception(message.c_str());
}

void stop(const char* message) {
    throw Rcpp::exception(message);
}

void stop(const char* fmt, std::initializer_list<StopArg> args) {
    throw Rcpp::exception(format_message(fmt, args).c_str());
}

}  // namespace tulpa
