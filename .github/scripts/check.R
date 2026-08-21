#!/usr/bin/env Rscript

# R CMD check at the tier CRAN runs.
#
# rcmdcheck sets NOT_CRAN="true" in the check subprocess by default, which
# unskips the recovery gates: at NOT_CRAN=true the whole directory is weeks of
# fits, and none of it is what the check farm executes. The env below pins
# every tier variable to what a CRAN machine has, so what this job reads is
# what CRAN reads.

check_args <- c("--as-cran")

# The reference manual needs LaTeX, and the Rd sources carry typographic
# Unicode that only fails at Rd2pdf. One job in the matrix installs TinyTeX and
# builds it; the rest skip it.
if (!identical(Sys.getenv("TULPA_CHECK_MANUAL"), "1")) {
  check_args <- c(check_args, "--no-manual")
}

env <- c(
  callr::rcmd_safe_env(),
  NOT_CRAN = "false",
  TULPA_FAST = "",
  TULPA_SLOW_TESTS = "",
  TULPA_FULL_RECOVERY = "",
  # The remote-URL leg of the incoming checks reaches every URL in the
  # DESCRIPTION and the Rd files, and answers for a server that is slow today.
  "_R_CHECK_CRAN_INCOMING_REMOTE_" = "false",
  # A Suggests that fails to install on one platform is a platform problem, not
  # a package problem, and turning it into "checking package dependencies ...
  # ERROR" hides everything the check would otherwise have reported.
  "_R_CHECK_FORCE_SUGGESTS_" = "false"
)

cat("check args:", paste(check_args, collapse = " "), "\n")
cat("NOT_CRAN  : false (CRAN tier)\n\n")

rcmdcheck::rcmdcheck(
  args = check_args,
  error_on = "warning",
  check_dir = "check",
  env = env
)
