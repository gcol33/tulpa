test_that("every registry entry accepts a complete block and names each missing field", {
  reg <- tulpa:::.NL_REGISTRY
  checked <- 0L
  for (nm in names(reg)) {
    req_spec <- reg[[nm]]$required
    if (is.null(req_spec)) next
    for (path in names(req_spec)) {
      req <- unique(unlist(req_spec[c("axis", path)], use.names = FALSE))
      full <- c(list(type = nm), stats::setNames(as.list(rep(1L, length(req))), req))
      expect_silent(tulpa:::.nl_check_block_fields(full, c("axis", path)))
      for (f in req) {
        dropped <- full[setdiff(names(full), f)]
        expect_error(
          tulpa:::.nl_check_block_fields(dropped, c("axis", path)),
          f, fixed = TRUE,
          info = paste0(nm, " / ", path, " / ", f)
        )
        # A field present but empty is what the kernel indexes out of bounds,
        # so it counts as missing too.
        empty <- full; empty[[f]] <- integer(0)
        expect_error(
          tulpa:::.nl_check_block_fields(empty, c("axis", path)),
          f, fixed = TRUE,
          info = paste0(nm, " / ", path, " / ", f, " (empty)")
        )
        checked <- checked + 1L
      }
    }
  }
  expect_gt(checked, 0L)
})

test_that("a block missing every required field errors instead of crashing (gcol33/tulpa#299)", {
  set.seed(4)
  n_s <- 12L; n_per <- 30L
  s <- rep(seq_len(n_s), each = n_per)
  x <- rnorm(length(s))
  y <- rbinom(length(s), 1, 0.5)
  A <- matrix(0, n_s, n_s)
  for (i in seq_len(n_s - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- 1

  # `adjacency` / `idx` are not the icar block's field names, so every
  # required field resolves to NULL.
  expect_error(
    tulpa_nested_laplace(
      y, rep(1L, length(y)), cbind(1, x),
      prior = list(type = "icar", n_spatial_units = n_s,
                   adjacency = A, idx = s),
      family = "binomial"),
    "missing required field"
  )
})

test_that("the multi-block converter names the missing field too", {
  blk <- list(type = "icar", n_spatial_units = 4L,
              adj_row_ptr = c(0L, 1L, 2L, 3L, 4L), adj_col_idx = 0:3,
              n_neighbors = rep(1L, 4))   # spatial_idx absent
  expect_error(tulpa:::.nl_block_spec_for_cpp(blk), "spatial_idx", fixed = TRUE)
})

test_that("the joint converter names the missing field too", {
  blk <- list(type = "rw1", n_times = 5L)   # temporal_idx absent
  expect_error(
    tulpa:::.joint_block_spec_for_cpp(blk, n_arms = 2L, block_index = 2L),
    "temporal_idx", fixed = TRUE)
  expect_error(
    tulpa:::.joint_block_spec_for_cpp(blk, n_arms = 2L, block_index = 2L),
    "block 2", fixed = TRUE)
})

# --- Source lint: the registry declaration must cover what the converters read.
#
# Each converter is an if/else-if chain over the block `type`. For every branch,
# collect the block fields it reads and subtract the ones it reads defensively
# (`is.null(p$f)`, `p$f %||% default`, `isTRUE(p$f)`); what remains is indexed
# unconditionally and must be declared, or a future field lands back in the
# segfault path the declaration exists to close.
.nlreq_branch_sources <- function(fn, types) {
  out <- list()
  walk <- function(e) {
    if (!is.call(e)) return(invisible(NULL))
    if (identical(e[[1L]], as.name("{"))) {
      for (i in seq_along(e)[-1L]) walk(e[[i]])
      return(invisible(NULL))
    }
    if (identical(e[[1L]], as.name("if"))) {
      lits <- Filter(is.character, as.list(all.vars(e[[2L]], functions = FALSE)))
      lits <- unlist(lapply(as.list(e[[2L]]), function(z) {
        if (is.character(z)) z else if (is.call(z)) {
          Filter(is.character, as.list(z))
        } else NULL
      }), use.names = FALSE)
      hits <- intersect(unlist(lits), types)
      if (length(hits)) {
        src <- paste(deparse(e[[3L]]), collapse = "\n")
        for (t in hits) out[[t]] <<- paste(c(out[[t]], src), collapse = "\n")
      }
      if (length(e) >= 4L) walk(e[[4L]])
      return(invisible(NULL))
    }
    invisible(NULL)
  }
  walk(body(fn))
  out
}

.nlreq_fields <- function(src) {
  m <- regmatches(src, gregexpr("p\\$[A-Za-z_][A-Za-z0-9_.]*", src))[[1L]]
  m2 <- regmatches(src, gregexpr('p\\[\\["[^"]+"\\]\\]', src))[[1L]]
  unique(c(sub("^p\\$", "", m), sub('^p\\[\\["(.*)"\\]\\]$', "\\1", m2)))
}

.nlreq_guarded <- function(src, f) {
  any(vapply(c(sprintf("is.null(p$%s)", f),
               sprintf("p$%s %%||%%", f),
               sprintf("isTRUE(p$%s)", f)),
             function(pat) grepl(pat, src, fixed = TRUE), logical(1)))
}

test_that("each converter branch reads only declared or defensively-read fields", {
  reg <- tulpa:::.NL_REGISTRY
  types <- names(reg)
  # Fields deliberately left undeclared, with the reason they cannot be a flat
  # requirement.
  allow <- list(
    tgmrf = c("Q", "prior",      # only read when backend == "r"
              "backend")         # selects which of the two it is
  )
  targets <- list(
    multi = tulpa:::.nl_block_spec_for_cpp,
    joint = tulpa:::.joint_block_spec_for_cpp
  )
  for (path in names(targets)) {
    branches <- .nlreq_branch_sources(targets[[path]], types)
    expect_gt(length(branches), 0L)
    for (t in names(branches)) {
      src <- branches[[t]]
      declared <- unique(unlist(reg[[t]]$required[c("axis", path)],
                                use.names = FALSE))
      read <- .nlreq_fields(src)
      read <- read[!vapply(read, function(f) .nlreq_guarded(src, f), logical(1))]
      undeclared <- setdiff(read, c(declared, allow[[t]]))
      expect_identical(
        undeclared, character(0),
        info = paste0(path, " / ", t, ": undeclared unconditional field(s) ",
                      paste(undeclared, collapse = ", "))
      )
    }
  }
})
