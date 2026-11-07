test_that("article caches replay output and objects without fitting", {
  skip_if_not_installed("knitr")
  withr::local_options(tulpa.fit_cache.build = TRUE)
  withr::local_seed(123)
  withr::local_dir(withr::local_tempdir())
  hooks <- knitr::knit_hooks$get()
  opts <- knitr::opts_chunk$get()
  option_hooks <- knitr::opts_hooks$get()
  withr::defer(knitr::knit_hooks$restore(hooks))
  withr::defer(knitr::opts_chunk$restore(opts))
  withr::defer(knitr::opts_hooks$restore(option_hooks))
  code <- c(
    "```{r setup, include=FALSE}",
    "source(system.file('knitr/fit-cache.R', package='tulpa'))",
    "set.seed(123)", "```",
    "```{r simulate}", "x <- rnorm(3)", "```",
    "```{r fit, fit=c('answer')}",
    "writeLines('fitted', 'fit-called')",
    "answer <- sum(x) + rnorm(1)", "answer", "plot(x)", "```",
    "```{r invalid-fit, fit=TRUE, error=TRUE}",
    "stop('illustrated error')", "```",
    "```{r report}", "print(answer)", "print(rnorm(1))", "```")
  writeLines(code, "example.Rmd")
  knit <- function() knitr::knit("example.Rmd", output = "example.md",
                               envir = new.env(parent = globalenv()), quiet = TRUE)
  first <- knit()
  output <- readLines(first)
  expect_true(file.exists("fit-called"))
  expect_true(file.exists("figure/fit-1.png"))
  unlink("fit-called")
  unlink("figure", recursive = TRUE)
  options(tulpa.fit_cache.build = FALSE)
  knit()
  expect_false(file.exists("fit-called"))
  expect_true(file.exists("figure/fit-1.png"))
  expect_identical(readLines("example.md"), output)
  cache <- readRDS("fits/example.rds", refhook = function(key) new.env())
  expect_identical(names(cache$chunks$fit$objects), "answer")
  expect_identical(cache$version, as.character(packageVersion("tulpa")))
  writeLines(sub("answer <- sum(x)", "answer <- mean(x)", code, fixed = TRUE), "example.Rmd")
  expect_error(knit(), "article 'example', chunk 'fit'.*changed")
  writeLines(sub("x <- rnorm(3)", "x <- rnorm(4)", code, fixed = TRUE), "example.Rmd")
  expect_error(knit(), "article 'example', chunk 'fit'.*changed")
  writeLines(code, "example.Rmd")
  cache$version <- "0.0.0"
  saveRDS(cache, "fits/example.rds")
  expect_error(knit(), "article 'example'.*tulpa version")
  unlink("fits/example.rds")
  expect_error(knit(), "article 'example'.*missing")
  options(tulpa.fit_cache.build = TRUE, tulpa.fit_cache.checkpoint = TRUE)
  withr::defer(options(tulpa.fit_cache.checkpoint = NULL))
  writeLines(append(code, "knitr::opts_chunk$set(fit_stateful=TRUE)", after = 2),
             "example.Rmd")
  knit()
  unlink("fit-called")
  knit()
  expect_true(file.exists("fit-called"))
  options(tulpa.fit_cache.build = FALSE)
  unlink("fit-called")
  knit()
  expect_false(file.exists("fit-called"))
  options(tulpa.fit_cache.build = TRUE, tulpa.fit_cache.checkpoint = NULL)
  compact_code <- sub("answer <- sum(x) + rnorm(1)",
    "answer <- structure(list(estimate=sum(x)+rnorm(1), unused=1:1000), class='tulpa_fit')",
    code, fixed = TRUE)
  compact_code <- sub("print(answer)", "print(answer$estimate)", compact_code,
                      fixed = TRUE)
  compact_code <- append(compact_code,
    "knitr::opts_chunk$set(fit_stateful=FALSE, fit_omit='unused')", after = 2)
  writeLines(compact_code, "example.Rmd")
  knit()
  before <- readLines("example.md")
  cache <- readRDS("fits/example.rds", refhook = function(key) new.env())
  expect_s3_class(cache$chunks$fit$objects$answer, "tulpa_fit")
  expect_null(cache$chunks$fit$objects$answer$unused)
  options(tulpa.fit_cache.build = FALSE)
  knit()
  expect_identical(readLines("example.md"), before)
})
