test_that('sparse fixtures reject malformed CSC before entering a solver', {
  f <- device_sparse_fixture(5L)
  f$p[2L] <- -1L
  expect_error(device_sparse_run(f, 'cpu'), 'invalid sparse fixture')
})

skip_on_cran()

test_that('sparse CPU solves and Takahashi agree with dense reference', {
  skip_on_cran()
  for (kind in c('spde', 'areal', 'temporal', 'joint')) {
    f <- device_sparse_fixture(31L, kind)
    result <- device_sparse_run(f, 'cpu', TRUE)
    reference <- solve(as.matrix(f$matrix) * 1.07)
    expect_true(result$ok)
    expect_equal(result$analyses, 1)
    expect_device_equal(result$solution, reference %*% f$b, 1e-10, paste(kind, 'solve'))
    expect_device_equal(result$diagonal, diag(reference), 1e-10, paste(kind, 'inverse diagonal'))
    expect_device_equal(result$logdet, c(1, 1) * as.numeric(determinant(as.matrix(f$matrix), logarithm = TRUE)$modulus) +
                          c(0, 31 * log(1.07)), 1e-10, paste(kind, 'logdet'))
  }
})

test_that('sparse factors recover from rejected pivots in concurrent workers', {
  skip_on_cran()
  choices <- 'cpu'
  info <- gpu_info()
  if (info$available && !nzchar(info$library_reasons[['cudss']])) choices <- c(choices, 'cuda')
  for (choice in choices) {
    state <- tulpa:::cpp_device_sparse_state(choice)
    expect_equal(state$success, rep(1L, 4L))
    expect_equal(state$analyses, rep(1L, 4L))
    expect_equal(state$errors, rep('', 4L))
  }
})

test_that('nonblocking sparse streams consume every numeric update after allocator reuse', {
  skip_on_cran()
  skip_if_no_cuda('cudss')
  for (n in c(1000L, 3000L, 10000L, 10000L)) {
    f <- device_sparse_fixture(n, 'spde')
    cpu <- device_sparse_run(f, 'cpu')
    cuda <- device_sparse_run(f, 'cuda')
    for (q in c('solution','logdet','diagonal'))
      expect_device_equal(cuda[[q]], cpu[[q]], 1e-8, paste(n, 'stream update', q))
    expect_lt(max(abs(f$matrix %*% cuda$solution * 1.07 - f$b)), 1e-10)
  }
})

test_that('cuDSS refactors and selects the inverse on both sides of the measured gate', {
  skip_on_cran()
  skip_if_no_cuda('cudss')
  table <- tulpa:::.device_thresholds(utils::read.csv(system.file('device', 'benchmarks.csv', package = 'tulpa')))
  gate <- table$threshold[table$operation == 'sparse_cholesky' & table$variant == 0L]
  sizes <- if (length(gate) && is.finite(gate)) unique(as.integer(c(max(2, gate - 1), gate + 1))) else c(63L, 257L)
  for (kind in c('spde', 'areal', 'temporal', 'joint')) for (n in sizes) {
    f <- device_sparse_fixture(n, kind)
    cpu <- device_sparse_run(f, 'cpu', TRUE)
    cuda <- device_sparse_run(f, 'cuda', TRUE)
    expect_true(cpu$ok && cuda$ok)
    expect_false(cpu$used_cuda)
    expect_true(cuda$used_cuda)
    expect_equal(cuda$analyses, 1)
    for (q in c('logdet', 'solution', 'diagonal')) {
      cat(sprintf('%s n=%d %s scale=%.17g tolerance=1e-8\n', kind, n, q, max(1, abs(cpu[[q]]))))
      expect_device_equal(cuda[[q]], cpu[[q]], 1e-8, paste(kind, n, q))
    }
    pinv <- cuda$permutation_inverse
    reference <- if (n <= 300L) solve(as.matrix(f$matrix) * 1.07) else NULL
    if (!is.null(reference)) {
      perm <- order(pinv)
      expected <- unlist(lapply(seq_len(n), function(j) {
        entries <- seq.int(cuda$inverse_p[j] + 1L, cuda$inverse_p[j + 1L])
        reference[perm[cuda$inverse_i[entries] + 1L], perm[j]]
      }), use.names = FALSE)
      expect_device_equal(cuda$inverse_x, expected, 1e-8, paste(kind, 'all selected entries'))
    }
    auto <- device_sparse_run(f, 'auto')
    expect_identical(auto$used_cuda, identical(tulpa:::cpp_device_probe('sparse_cholesky', n, 0L, 'auto')$device, 'cuda'))
    expect_device_equal(auto$diagonal, cpu$diagonal, 1e-8, paste(kind, 'automatic inverse diagonal'))
  }
})
