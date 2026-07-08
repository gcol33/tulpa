# Exactness of the SPDE field-SE kernel cpp_spde_field_se: it streams the
# per-cell quadratic form se_j = sqrt(c_j' H^{-1} c_j) one query column at a
# time (dense working set O(p + n_mesh), independent of the number of cells).
# The streamed result must equal a dense reference solve at any grid size.

.fse_gC  <- function(x) as(as(x, "generalMatrix"), "CsparseMatrix")
.fse_ref <- function(H, Cq) {
  Hd <- as.matrix(H); Cd <- as.matrix(Cq)
  sqrt(pmax(colSums(Cd * solve(Hd, Cd)), 0))
}

test_that("cpp_spde_field_se matches a dense reference solve at any grid size", {
  set.seed(11)
  for (cfg in list(c(m = 30L, n = 5L), c(m = 60L, n = 40L), c(m = 25L, n = 1L))) {
    m <- cfg[["m"]]; n <- cfg[["n"]]
    L  <- Matrix::rsparsematrix(m, m, density = 0.15)
    H  <- Matrix::crossprod(L) + Matrix::Diagonal(m) * 2   # symmetric PD
    Cq <- Matrix::rsparsematrix(m, n, density = 0.25)
    se <- tulpa:::cpp_spde_field_se(.fse_gC(H), .fse_gC(Cq))
    expect_length(se, n)
    expect_equal(se, .fse_ref(H, Cq), tolerance = 1e-9)
  }
})

test_that("cpp_spde_field_se returns numeric(0) on an empty query grid", {
  H  <- .fse_gC(Matrix::Diagonal(4) * 1.0)
  Cq <- .fse_gC(Matrix::rsparsematrix(4, 3, density = 0.4))[, integer(0), drop = FALSE]
  expect_identical(tulpa:::cpp_spde_field_se(H, .fse_gC(Cq)), numeric(0))
})

test_that("cpp_spde_field_se errors on a non-positive-definite precision", {
  H  <- .fse_gC(Matrix::Diagonal(x = c(1, -1, 1)))       # indefinite
  Cq <- .fse_gC(Matrix::rsparsematrix(3, 2, density = 0.5))
  expect_error(tulpa:::cpp_spde_field_se(H, Cq), "positive definite")
})
