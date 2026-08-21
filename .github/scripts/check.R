#!/usr/bin/env Rscript

# R CMD check at the tier CRAN runs.
#
# skip_on_cran() runs its gated test whenever NOT_CRAN is "true" in the check
# subprocess, and the check subprocess inherits whatever the job carries. At
# NOT_CRAN=true the ungated directory is weeks of fits and none of it is what
# the check farm executes, so every tier variable is pinned here rather than
# left to the environment: what this job reads is what CRAN reads, whatever
# else is set around it.

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
