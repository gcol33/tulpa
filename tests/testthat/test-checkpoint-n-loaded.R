# A checkpointed nested-Laplace fit reports how many grid cells it read back
# from the file (`checkpoint_n_loaded`), summed over every grid the fit solves
# on the path: the first grid, a recentred refit, refinement rounds.

skip_on_cran()

.ckn_fixture <- function() {
  set.seed(20260716)
  S <- 30L
  W <- matrix(0, S, S)
  for (i in 1:(S - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
  df <- data.frame(region = factor(seq_len(S)))
  df$x <- as.integer(df$region) / 10 + rnorm(S, 0, 0.3)
  df$y <- rbinom(S, 20, plogis(-0.4 + 0.5 * df$x))
  list(S = S, W = W, df = df)
}

.ckn_fit <- function(d, control = list()) {
  tulpa(y ~ x + spatial(region), data = d$df, family = "binomial",
        n_trials = rep(20L, d$S), spatial = spatial_car(d$W, level = "obs"),
        mode = "structured", control = control)
}

.ckn_records <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  readBin(con, "raw", 16L)
  n <- 0L
  repeat {
    kl <- readBin(con, "integer", 1L, size = 4L); if (!length(kl)) break
    readBin(con, "raw", kl)
    pl <- readBin(con, "integer", 1L, size = 4L); if (!length(pl)) break
    p <- readBin(con, "raw", pl); ck <- readBin(con, "raw", 8L)
    if (length(ck) < 8L || length(p) < pl) break
    n <- n + 1L
  }
  n
}

test_that("checkpoint_n_loaded counts the cells a fit took from the file", {
  skip_if_fast()
  d <- .ckn_fixture()
  ckpt <- tempfile(fileext = ".ckpt"); on.exit(unlink(ckpt), add = TRUE)
  fresh <- .ckn_fit(d, list(checkpoint = list(path = ckpt, resume = FALSE)))
  expect_identical(fresh$checkpoint_n_loaded, 0L)
  n_rec <- .ckn_records(ckpt)
  expect_gt(n_rec, 0L)

  resumed <- .ckn_fit(d, list(checkpoint = list(path = ckpt, resume = TRUE)))
  # Every cell any grid of the fit solved is on disk, so a resume reads all of
  # them back: as many as the file holds records.
  expect_identical(resumed$checkpoint_n_loaded, n_rec)
  expect_equal(coef(resumed), coef(fresh))

  # A torn file keeps its leading records only; the fit reads back exactly
  # those, whichever of its grids they belong to.
  sz <- file.info(ckpt)$size
  con <- file(ckpt, "rb"); b <- readBin(con, "raw", sz); close(con)
  con <- file(ckpt, "wb"); writeBin(b[seq_len(floor(sz / 2))], con); close(con)
  kept <- .ckn_records(ckpt)
  torn <- .ckn_fit(d, list(checkpoint = list(path = ckpt, resume = TRUE)))
  expect_identical(torn$checkpoint_n_loaded, kept)
  expect_equal(coef(torn), coef(fresh))

  plain <- .ckn_fit(d)
  expect_null(plain$checkpoint_n_loaded)
})
