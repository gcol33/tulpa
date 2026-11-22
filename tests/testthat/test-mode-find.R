
test_that(".nl_lbfgsb_mode_find returns NULL rather than propagating an error", {
  boom <- function(u) stop("objective exploded")
  expect_null(tulpa:::.nl_lbfgsb_mode_find(
    c(0, 0), boom, c(-1, -1), c(1, 1),
    tulpa:::.nl_mode_find_tuning("st")))
})

skip_on_cran_budget()
# test-mode-find.R
# The shared outer mode-find (R/mode_find.R) and its settings entry.

test_that(".NL_MODE_FIND pins the shipped tuning for every consumer", {
  skip_on_cran_budget()
  expect_setequal(names(tulpa:::.NL_MODE_FIND), c("st", "pathfinder", "hyper_ccd"))
  # The helper consumer carries the full knob set. pathfinder takes only
  # factr: it is unbounded, gradient-aware, and carries maxit / pgtol as its
  # own arguments, so it reads this table without sharing the helper.
  for (consumer in "st") {
    expect_setequal(names(tulpa:::.NL_MODE_FIND[[consumer]]),
                    c("factr", "ndeps", "maxit"))
  }
  expect_setequal(names(tulpa:::.NL_MODE_FIND$pathfinder), "factr")
  expect_identical(tulpa:::.nl_mode_find("pathfinder", "factr"), 1e7)
  # The CCD design's mode-find takes its evaluation cap from its caller.
  expect_setequal(names(tulpa:::.NL_MODE_FIND$hyper_ccd), c("factr", "ndeps"))
  expect_identical(tulpa:::.nl_mode_find("hyper_ccd", "factr"), 1e7)
  expect_identical(tulpa:::.nl_mode_find("hyper_ccd", "ndeps"), 5e-2)
  expect_false("optim" %in% all.names(body(tulpa:::.nl_mode_find_tuning)))

  expect_identical(tulpa:::.nl_mode_find("st", "factr"), 1e7)
  expect_identical(tulpa:::.nl_mode_find("st", "maxit"), 300L)
})

test_that("the st ndeps entry is optim's own default, so writing it changes nothing", {
  skip_on_cran_budget()
  # The ST auto-grid used to omit ndeps and inherit optim()'s default. Stating
  # it in .NL_MODE_FIND is only an improvement if the stated value IS that
  # default -- otherwise the refactor silently retuned that path.
  fn <- function(u) sum((u - c(0.3, -0.7, 0.15))^2) + 0.1 * sum(u^4)
  u0 <- c(-1.1, 0.9, -0.2)
  lo <- rep(-5, 3); hi <- rep(5, 3)

  inherited <- stats::optim(u0, fn, method = "L-BFGS-B", lower = lo, upper = hi,
                            hessian = TRUE,
                            control = list(maxit = 300L, factr = 1e7))
  stated <- stats::optim(u0, fn, method = "L-BFGS-B", lower = lo, upper = hi,
                         hessian = TRUE,
                         control = list(maxit = 300L, factr = 1e7,
                                        ndeps = rep(1e-3, 3)))
  expect_identical(stated$par, inherited$par)
  expect_identical(stated$value, inherited$value)
  expect_identical(stated$convergence, inherited$convergence)
  expect_identical(stated$hessian, inherited$hessian)
  expect_identical(tulpa:::.nl_mode_find("st", "ndeps"), 1e-3)
})

test_that(".nl_lbfgsb_mode_find reproduces the call it replaced, bit for bit", {
  skip_on_cran_budget()
  fn <- function(u) sum((u - c(0.4, -1.3))^2) + 0.05 * sum(u^4)
  u0 <- c(1.7, -0.2)
  lo <- c(-6, -6); hi <- c(6, 6)

  for (consumer in "st") {
    tune <- tulpa:::.nl_mode_find_tuning(consumer)
    direct <- stats::optim(par = u0, fn = fn, method = "L-BFGS-B",
                           lower = lo, upper = hi, hessian = TRUE,
                           control = list(factr = tune$factr,
                                          maxit = tune$maxit,
                                          ndeps = rep(tune$ndeps, 2L)))
    viahelper <- tulpa:::.nl_lbfgsb_mode_find(u0, fn, lo, hi, tune,
                                              hessian = TRUE)
    expect_identical(viahelper$par, direct$par)
    expect_identical(viahelper$value, direct$value)
    expect_identical(viahelper$convergence, direct$convergence)
    expect_identical(viahelper$hessian, direct$hessian)
  }
})

test_that("control$mode_find overrides the defaults it names and no others", {
  skip_on_cran_budget()
  base <- tulpa:::.nl_mode_find_tuning("st")
  expect_identical(base$ndeps, 1e-3)

  one <- tulpa:::.nl_mode_find_tuning("st", list(mode_find = list(ndeps = 5e-3)))
  expect_identical(one$ndeps, 5e-3)
  expect_identical(one$factr, base$factr)
  expect_identical(one$maxit, base$maxit)

  all3 <- tulpa:::.nl_mode_find_tuning(
    "st", list(mode_find = list(ndeps = 1e-3, factr = 1e6, maxit = 50)))
  expect_identical(all3$ndeps, 1e-3)
  expect_identical(all3$factr, 1e6)
  expect_identical(all3$maxit, 50)

  # An absent or empty spec is the defaults, unchanged.
  expect_identical(tulpa:::.nl_mode_find_tuning("st", list()), base)
  expect_identical(tulpa:::.nl_mode_find_tuning("st", NULL), base)
  expect_identical(
    tulpa:::.nl_mode_find_tuning("st", list(mode_find = list())), base)
})

test_that("a misspelled or ill-typed mode_find knob errors instead of fitting the default", {
  skip_on_cran_budget()
  expect_error(
    tulpa:::.nl_mode_find_tuning("st", list(mode_find = list(ndpes = 1e-3))),
    "Unknown control\\$mode_find knob")
  expect_error(
    tulpa:::.nl_mode_find_tuning("st", list(mode_find = list(1e-3))),
    "must be named")
  expect_error(
    tulpa:::.nl_mode_find_tuning("st", list(mode_find = c(ndeps = 1e-3))),
    "must be a list")
  for (bad in list(-1, 0, NA_real_, Inf, c(1e-3, 1e-3), "1e-3")) {
    expect_error(
      tulpa:::.nl_mode_find_tuning("st", list(mode_find = list(ndeps = bad))),
      "one positive finite number")
  }
  expect_error(tulpa:::.nl_mode_find("nosuch", "ndeps"), "Unknown mode-find consumer")
  expect_error(tulpa:::.nl_mode_find("st", "nosuch"), "Unknown mode-find setting")
})

test_that("the mode-find consumer reaches optim only through the shared helper", {
  skip_on_cran_budget()
  # The point of .NL_MODE_FIND is that this tuning lives in one file. A call
  # site that calls optim() itself carries its own factr / ndeps again, which
  # is the copy this replaced. all.names() walks the parse tree, so this holds
  # against the installed package and needs no source files.
  expect_true("optim" %in% all.names(body(tulpa:::.nl_lbfgsb_mode_find)))
  expect_false("optim" %in% all.names(body(tulpa:::.st_auto_grid_rescue)))

  # ... and reaches the helper.
  expect_true(".nl_lbfgsb_mode_find" %in% all.names(body(tulpa:::.st_auto_grid_rescue)))
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
