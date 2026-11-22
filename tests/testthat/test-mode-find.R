skip_on_cran_budget()
# test-mode-find.R
# The outer mode-find settings entry (`.NL_MODE_FIND`, R/settings.R).

test_that(".NL_MODE_FIND pins the shipped tuning for every consumer", {
  skip_on_cran_budget()
  expect_setequal(names(tulpa:::.NL_MODE_FIND), c("pathfinder", "hyper_ccd"))
  # pathfinder takes only factr: it is unbounded, gradient-aware, and carries
  # maxit / pgtol as its own arguments.
  expect_setequal(names(tulpa:::.NL_MODE_FIND$pathfinder), "factr")
  expect_identical(tulpa:::.nl_mode_find("pathfinder", "factr"), 1e7)
  # The CCD design's mode-find takes its evaluation cap from its caller.
  expect_setequal(names(tulpa:::.NL_MODE_FIND$hyper_ccd), c("factr", "ndeps"))
  expect_identical(tulpa:::.nl_mode_find("hyper_ccd", "factr"), 1e7)
  expect_identical(tulpa:::.nl_mode_find("hyper_ccd", "ndeps"), 5e-2)
})

test_that("an unknown consumer or setting errors instead of reading a default", {
  skip_on_cran_budget()
  expect_error(tulpa:::.nl_mode_find("nosuch", "ndeps"), "Unknown mode-find consumer")
  expect_error(tulpa:::.nl_mode_find("hyper_ccd", "nosuch"), "Unknown mode-find setting")
})

skip_on_cran()

test_that("no file outside settings.R restates a mode-find constant", {
  skip_on_cran()
  r_dir <- test_path("..", "..", "R")
  skip_if_not(dir.exists(r_dir), "package sources not available")
  files <- setdiff(list.files(r_dir, pattern = "\\.R$", full.names = TRUE),
                   file.path(r_dir, "settings.R"))

  offenders <- character(0)
  for (f in files) {
    src  <- readLines(f, warn = FALSE)
    code <- src[!grepl("^\\s*#", src)]
    hits <- grep("(factr|ndeps)\\s*=\\s*[0-9]", code, value = TRUE)
    if (length(hits)) {
      offenders <- c(offenders, paste0(basename(f), ": ", trimws(hits)))
    }
  }
  expect_identical(offenders, character(0))
})
