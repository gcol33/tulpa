# tulpa declares generics other packages own, so that they work with nothing
# else attached. Registering the tulpa_fit methods on the other packages'
# generics as well is what keeps dispatch from depending on attach order in
# either direction.

test_that("every borrowed generic carries a registered tulpa_fit method", {
  borrowed <- list(
    c("lme4", "fixef"), c("lme4", "ranef"), c("lme4", "VarCorr"),
    c("nlme", "ranef"), c("nlme", "VarCorr"),
    c("posterior", "as_draws"), c("posterior", "as_draws_array"),
    c("posterior", "as_draws_matrix"), c("posterior", "as_draws_df"),
    c("posterior", "as_draws_rvars"),
    c("bayesplot", "pp_check"),
    c("rstantools", "bayes_R2"), c("rstantools", "posterior_predict")
  )
  for (g in borrowed) {
    pkg <- g[1]; gen <- g[2]
    if (!requireNamespace(pkg, quietly = TRUE)) next
    ns <- asNamespace(pkg)
    if (!exists(gen, envir = ns, inherits = FALSE)) next
    tbl <- get(".__S3MethodsTable__.", envir = ns)
    expect_true(exists(paste0(gen, ".tulpa_fit"), envir = tbl, inherits = FALSE),
                info = paste0(pkg, "::", gen))
  }
})

test_that("bayesplot's own pp_check still reaches its default method", {
  skip_if_not_installed("bayesplot")
  # tulpa's generic masks bayesplot's on attach, and a registered S3 method is
  # not an exported object, so without the borrowed default every
  # bayesplot-style pp_check(y, yrep, fun) call in the session fails with "no
  # applicable method ... for class numeric".
  set.seed(1)
  y <- rnorm(30)
  yrep <- matrix(rnorm(150), 5, 30)
  expect_no_error(bayesplot::pp_check(y, yrep, bayesplot::ppc_dens_overlay))
  # And through the masking generic, which is what a user's session sees.
  expect_no_error(pp_check(y, yrep, bayesplot::ppc_dens_overlay))
})

test_that("the borrowed default is registered, not exported", {
  # An exported pp_check.default would win dispatch inside bayesplot's own
  # generic too, and a forwarding one would then call itself forever.
  skip_if_not_installed("bayesplot")
  tbl <- get(".__S3MethodsTable__.", envir = asNamespace("tulpa"))
  expect_true(exists("pp_check.default", envir = tbl, inherits = FALSE))
  expect_false(exists("pp_check.default", envir = asNamespace("tulpa"),
                      inherits = FALSE))
  expect_false("pp_check.default" %in% getNamespaceExports("tulpa"))
})

test_that("rstantools' generics keep their default behaviour under tulpa", {
  skip_if_not_installed("rstantools")
  tbl <- get(".__S3MethodsTable__.", envir = asNamespace("tulpa"))
  for (gen in c("bayes_R2", "posterior_predict")) {
    owner <- get(".__S3MethodsTable__.", envir = asNamespace("rstantools"))
    if (!exists(paste0(gen, ".default"), envir = owner, inherits = FALSE)) next
    expect_true(exists(paste0(gen, ".default"), envir = tbl, inherits = FALSE),
                info = gen)
    expect_identical(get(paste0(gen, ".default"), envir = tbl),
                     get(paste0(gen, ".default"), envir = owner), info = gen)
  }
})

test_that("a tulpa fit reaches the tulpa methods on the borrowed generics", {
  skip_on_cran()
  set.seed(2)
  n <- 60
  d <- data.frame(x = rnorm(n))
  d$y <- rbinom(n, 1L, plogis(0.2 + 0.6 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "binomial", mode = "ep")

  if (requireNamespace("rstantools", quietly = TRUE)) {
    expect_identical(
      utils::getS3method("posterior_predict", "tulpa_fit",
                         envir = asNamespace("rstantools")),
      posterior_predict.tulpa_fit
    )
  }
  if (requireNamespace("bayesplot", quietly = TRUE)) {
    expect_identical(
      utils::getS3method("pp_check", "tulpa_fit",
                         envir = asNamespace("bayesplot")),
      pp_check.tulpa_fit
    )
  }
  expect_s3_class(fit, "tulpa_fit")
})
