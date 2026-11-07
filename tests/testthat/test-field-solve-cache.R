precision_cache_fixture <- function(n) {
  list(modes = matrix(seq_len(2L * n), nrow = n),
       result = list(Q_p = rep(list(c(0L, 1L, 2L)), n),
                     Q_i = rep(list(0:1), n),
                     Q_x = lapply(seq_len(n), function(i) c(i, i + 1)),
                     constraint = list(hard = list(0:1), soft = list(0L),
                       soft_D = lapply(seq_len(n), function(i) matrix(i)))))
}

test_that("oversized precision batches retain exact reusable cells within the budget", {
  local_mocked_bindings(.FIELD_SOLVE_CACHE = new.env(parent = emptyenv()),
                        .FIELD_SOLVE_CACHE_MAX = 27)
  x <- precision_cache_fixture(6L)
  tulpa:::.field_solve_cache_put("a", 1:6, x$result, x$modes)
  hit <- tulpa:::.field_solve_cache_get("a", 6:1, x$modes)
  expect_identical(hit$found, c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE))
  for (i in 1:3) {
    j <- 7L - i
    expect_identical(hit$Q_p[[i]], x$result$Q_p[[j]])
    expect_identical(hit$Q_i[[i]], x$result$Q_i[[j]])
    expect_identical(hit$Q_x[[i]], x$result$Q_x[[j]])
    expect_identical(hit$soft_D[[i]], x$result$constraint$soft_D[[j]])
  }
  expect_identical(hit$hard, x$result$constraint$hard)
  expect_identical(hit$soft, x$result$constraint$soft)
  changed <- x$modes
  changed[6L, 1L] <- changed[6L, 1L] + 1L
  hit <- tulpa:::.field_solve_cache_get("a", 6:4, changed)
  expect_identical(hit$found, c(FALSE, TRUE, TRUE))
})

test_that("an individual precision exceeding the budget leaves smaller cells reusable", {
  local_mocked_bindings(.FIELD_SOLVE_CACHE = new.env(parent = emptyenv()),
                        .FIELD_SOLVE_CACHE_MAX = 9)
  x <- precision_cache_fixture(2L)
  first <- lapply(x$result[c("Q_p", "Q_i", "Q_x")], `[`, 1L)
  first$constraint <- list(hard = list(), soft = NULL, soft_D = NULL)
  tulpa:::.field_solve_cache_put("a", 1L, first, x$modes[1L, , drop = FALSE])
  large <- list(Q_p = list(c(0L, 2L, 3L)), Q_i = list(c(0L, 1L, 1L)),
                Q_x = list(c(2, 0, 3)), constraint = first$constraint)
  tulpa:::.field_solve_cache_put("a", 2L, large, x$modes[2L, , drop = FALSE])
  hit <- tulpa:::.field_solve_cache_get("a", 1:2, x$modes)
  expect_identical(hit$found, c(TRUE, FALSE))
  expect_identical(hit$Q_x[[1L]], first$Q_x[[1L]])
})

test_that("precision records share one bounded cache", {
  local_mocked_bindings(.FIELD_SOLVE_CACHE = new.env(parent = emptyenv()),
                        .FIELD_SOLVE_CACHE_MAX = 27)
  x <- precision_cache_fixture(2L)
  tulpa:::.field_solve_cache_put("a", 1:2, x$result, x$modes)
  tulpa:::.field_solve_cache_put("b", 1:2, x$result, x$modes)
  expect_false(any(tulpa:::.field_solve_cache_get("a", 1:2, x$modes)$found))
  expect_true(all(tulpa:::.field_solve_cache_get("b", 1:2, x$modes)$found))
})
