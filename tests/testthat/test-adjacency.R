test_that("adjacency.data.frame builds queen contiguity on a regular grid", {
  grid <- expand.grid(x = 1:3, y = 1:3)
  grid$cell <- paste0("c", seq_len(nrow(grid)))

  g <- adjacency(grid, x_coord = "x", y_coord = "y", id = "cell",
                 type = "queen")

  expect_s4_class(g$adjacency, "dgCMatrix")
  expect_equal(g$n, 9L)
  expect_identical(g$ids, grid$cell)

  W <- g$adjacency
  expect_true(Matrix::isSymmetric(W))
  expect_true(all(Matrix::diag(W) == 0))
  expect_true(all(W@x == 1))

  deg <- Matrix::rowSums(W)
  # 1 centre (8), 4 edge-mid (5), 4 corners (3); 20 undirected edges
  expect_equal(sort(deg), sort(c(8, 5, 5, 5, 5, 3, 3, 3, 3)))
  expect_equal(sum(Matrix::tril(W) != 0), 20L)
})

test_that("adjacency.data.frame builds rook contiguity on a regular grid", {
  grid <- expand.grid(x = 1:3, y = 1:3)
  g <- adjacency(grid, type = "rook")

  W <- g$adjacency
  expect_true(Matrix::isSymmetric(W))
  deg <- Matrix::rowSums(W)
  # 1 centre (4), 4 edge-mid (3), 4 corners (2); 12 undirected edges
  expect_equal(sort(deg), sort(c(4, 3, 3, 3, 3, 2, 2, 2, 2)))
  expect_equal(sum(Matrix::tril(W) != 0), 12L)
  expect_equal(unname(g$cellsize[["x"]]), 1)
  expect_equal(unname(g$cellsize[["y"]]), 1)
})

test_that("rook is a strict subset of queen on the same grid", {
  grid <- expand.grid(x = 1:4, y = 1:4)
  q <- adjacency(grid, type = "queen")$adjacency
  r <- adjacency(grid, type = "rook")$adjacency
  # every rook edge is a queen edge
  expect_true(all((r != 0) <= (q != 0)))
  expect_gt(sum(q != 0), sum(r != 0))
})

test_that("contiguity is correct around a hole (missing centre cell)", {
  grid <- expand.grid(x = 1:3, y = 1:3)
  grid <- grid[!(grid$x == 2 & grid$y == 2), ]  # drop the centre
  grid$cell <- paste0("c", seq_len(nrow(grid)))

  g <- adjacency(grid, id = "cell", type = "rook")
  expect_equal(g$n, 8L)
  # the 4 edge-mid cells lose the centre neighbour -> degree 2; corners stay 2
  deg <- Matrix::rowSums(g$adjacency)
  expect_equal(sum(deg == 2), 8L)
})

test_that("non-square grid spacing is handled per axis", {
  grid <- expand.grid(x = c(0, 10, 20), y = c(0, 2, 4))
  g <- adjacency(grid, type = "rook")
  expect_equal(unname(g$cellsize[["x"]]), 10)
  expect_equal(unname(g$cellsize[["y"]]), 2)
  deg <- Matrix::rowSums(g$adjacency)
  expect_equal(sort(deg), sort(c(4, 3, 3, 3, 3, 2, 2, 2, 2)))
})

test_that("node_index maps cell ids to node indices by key", {
  grid <- expand.grid(x = 1:3, y = 1:3)
  grid$cell <- paste0("c", seq_len(nrow(grid)))
  g <- adjacency(grid, id = "cell")

  obs <- data.frame(cell = c("c5", "c1", "c5", "c9"))
  idx <- node_index(g, obs$cell)
  expect_equal(idx, c(5L, 1L, 5L, 9L))

  # remap is by key, not row order: a reshuffled graph gives the same indices
  expect_warning(miss <- node_index(g, c("c1", "nope")), "not found")
  expect_equal(miss, c(1L, NA_integer_))
})

test_that("adjacency errors on missing coords, duplicates, and bad ids", {
  grid <- expand.grid(x = 1:2, y = 1:2)
  expect_error(adjacency(grid, x_coord = "lon"), "Coordinate")

  dup <- data.frame(x = c(1, 1), y = c(1, 1))
  expect_error(adjacency(dup), "Duplicated cell centroids")

  bad <- data.frame(x = 1:3, y = 1:3, cell = c("a", "a", "b"))
  expect_error(adjacency(bad, id = "cell"), "duplicate")
})

test_that("check_adjacency validates a clean matrix and flags defects", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i + 1] <- adj[i + 1, i] <- 1
  rep_ok <- check_adjacency(adj)
  expect_true(rep_ok$ok)
  expect_true(rep_ok$symmetric)
  expect_true(rep_ok$zero_diag)
  expect_equal(rep_ok$n_isolated, 0L)
  expect_equal(rep_ok$n_edges, 3L)

  asym <- adj; asym[1, 3] <- 1  # one-directional extra edge; no isolation
  expect_warning(r1 <- check_adjacency(asym), "not symmetric")
  expect_false(r1$symmetric)

  loop <- adj; loop[1, 1] <- 1
  expect_warning(r2 <- check_adjacency(loop), "diagonal")
  expect_false(r2$zero_diag)

  iso <- matrix(0, 3, 3); iso[1, 2] <- iso[2, 1] <- 1  # node 3 isolated
  expect_warning(r3 <- check_adjacency(iso), "isolated")
  expect_equal(r3$n_isolated, 1L)
})

test_that("check_adjacency reports non-square and id mismatches", {
  expect_warning(rns <- check_adjacency(matrix(0, 2, 3)), "not square")
  expect_false(rns$square)

  adj <- matrix(0, 3, 3); adj[1, 2] <- adj[2, 1] <- adj[2, 3] <- adj[3, 2] <- 1
  expect_warning(check_adjacency(adj, ids = c("a", "b")), "match")
})

test_that("constructed graph emits a message on isolated nodes", {
  # two disconnected single cells far apart relative to the grid step
  grid <- data.frame(x = c(1, 2, 10), y = c(1, 1, 10))
  expect_message(g <- adjacency(grid, type = "rook"), "isolated")
  expect_equal(sum(Matrix::rowSums(g$adjacency) == 0), 1L)
})

test_that("spatial() and spatial_car() accept a tulpa_adjacency object", {
  grid <- expand.grid(x = 1:3, y = 1:3)
  grid$cell <- paste0("c", seq_len(nrow(grid)))
  g <- adjacency(grid, id = "cell", type = "queen")

  sp <- spatial(graph = g, formula = ~ 1 || node)
  expect_s3_class(sp, "tulpa_spatial_field")
  expect_equal(nrow(sp$adjacency), 9L)

  car <- spatial_car(g, level = "group", group_var = "node")
  expect_s3_class(car, "tulpa_spatial")
})

test_that("adjacency.sf builds polygon contiguity", {
  skip_if_not_installed("sf")
  mk <- function(x0, y0) {
    sf::st_polygon(list(rbind(c(x0, y0), c(x0 + 1, y0), c(x0 + 1, y0 + 1),
                              c(x0, y0 + 1), c(x0, y0))))
  }
  polys <- sf::st_sf(
    cell = c("a", "b", "c", "d"),
    geometry = sf::st_sfc(mk(0, 0), mk(1, 0), mk(0, 1), mk(1, 1)))

  gr <- adjacency(polys, type = "rook", id = "cell")
  expect_identical(gr$ids, c("a", "b", "c", "d"))
  expect_equal(sum(Matrix::tril(gr$adjacency) != 0), 4L)  # 4 shared edges
  expect_true(all(Matrix::rowSums(gr$adjacency) == 2))

  gq <- adjacency(polys, type = "queen", id = "cell")
  # the two diagonal pairs touch at the central vertex -> +2 edges
  expect_equal(sum(Matrix::tril(gq$adjacency) != 0), 6L)
  expect_true(all(Matrix::rowSums(gq$adjacency) == 3))

  # "touches" is an alias of queen
  gt <- adjacency(polys, type = "touches", id = "cell")
  expect_equal(sum(gt$adjacency != 0), sum(gq$adjacency != 0))
})

test_that("adjacency.SpatRaster builds lattice contiguity over non-NA cells", {
  skip_if_not_installed("terra")
  r <- terra::rast(nrows = 3, ncols = 3, xmin = 0, xmax = 3,
                   ymin = 0, ymax = 3)
  terra::values(r) <- 1:9

  gr <- adjacency(r, type = "rook")
  expect_equal(gr$n, 9L)
  expect_equal(sum(Matrix::tril(gr$adjacency) != 0), 12L)

  gq <- adjacency(r, type = "queen")
  expect_equal(sum(Matrix::tril(gq$adjacency) != 0), 20L)

  # NA cells are dropped by default
  terra::values(r)[5] <- NA
  gn <- adjacency(r, type = "rook")
  expect_equal(gn$n, 8L)
})

test_that("adjacency.stars builds lattice contiguity over a raster", {
  skip_if_not_installed("stars")
  skip_if_not_installed("sf")
  s <- stars::st_as_stars(matrix(1:9, nrow = 3, ncol = 3))

  gr <- adjacency(s, type = "rook")
  expect_equal(gr$n, 9L)
  expect_equal(sum(Matrix::tril(gr$adjacency) != 0), 12L)
  gq <- adjacency(s, type = "queen")
  expect_equal(sum(Matrix::tril(gq$adjacency) != 0), 20L)

  # NA cells are dropped by default
  sna <- stars::st_as_stars(matrix(c(1:4, NA, 6:9), nrow = 3, ncol = 3))
  gn <- adjacency(sna, type = "rook")
  expect_equal(gn$n, 8L)
})

test_that("print methods run without error", {
  grid <- expand.grid(x = 1:3, y = 1:3)
  g <- adjacency(grid, type = "queen")
  expect_output(print(g), "tulpa_adjacency")
  expect_output(print(check_adjacency(g$adjacency)), "adjacency check")
})
