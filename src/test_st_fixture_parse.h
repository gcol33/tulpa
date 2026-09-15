// test_st_fixture_parse.h
// The string arguments the spatiotemporal test fixtures take, mapped onto the
// engine's enums in one place so every fixture names a configuration the same
// way and an unknown name is an error rather than a silent RW1.
#ifndef TULPA_TEST_ST_FIXTURE_PARSE_H
#define TULPA_TEST_ST_FIXTURE_PARSE_H

#include <Rcpp.h>

#include <string>

#include "tulpa/types.h"

namespace tulpa_test_st {

inline tulpa::TemporalType parse_temporal(const std::string& s) {
  if (s == "rw1") return tulpa::TemporalType::RW1;
  if (s == "rw2") return tulpa::TemporalType::RW2;
  if (s == "ar1") return tulpa::TemporalType::AR1;
  if (s == "iid") return tulpa::TemporalType::IID;
  Rcpp::stop("temporal must be one of \"rw1\", \"rw2\", \"ar1\", \"iid\"; "
             "got \"%s\"", s.c_str());
}

inline tulpa::STType parse_st_type(const std::string& s) {
  if (s == "i")         return tulpa::STType::TYPE_I;
  if (s == "ii")        return tulpa::STType::TYPE_II;
  if (s == "iii")       return tulpa::STType::TYPE_III;
  if (s == "iv")        return tulpa::STType::TYPE_IV;
  if (s == "separable") return tulpa::STType::SEPARABLE;
  if (s == "nonsep_gp") return tulpa::STType::NONSEP_GP;
  Rcpp::stop("st_type must be one of \"i\", \"ii\", \"iii\", \"iv\", "
             "\"separable\", \"nonsep_gp\"; got \"%s\"", s.c_str());
}

}  // namespace tulpa_test_st

#endif  // TULPA_TEST_ST_FIXTURE_PARSE_H
