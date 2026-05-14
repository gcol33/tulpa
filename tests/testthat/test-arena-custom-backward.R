# test-arena-custom-backward.R
# Smoke test for tulpa::arena::Arena::add_custom_backward — the new
# variadic-input/variadic-output node type that hosts the sparse-Cholesky
# adjoint for joint-NUTS over (log_kappa, log_tau_spde).
#
# Runs the same loss two ways on the same (x, y):
#   path A  - everything via the standard 1/2-operand SoA nodes
#   path B  - (x, y) -> (a, b) wrapped in one custom_backward block
# and compares to the analytical gradient. The two arena paths AND the
# analytical answer must agree.

test_that("arena custom_backward matches standard path and analytical grad", {
  for (xy in list(c(2.0, 3.0), c(-1.5, 0.7), c(0.3, -2.1))) {
    res <- cpp_test_arena_custom_backward(xy[1], xy[2])

    # Forward value: every path agrees on L.
    expect_equal(res$L_std, res$L_analytical, tolerance = 1e-12,
                 info = sprintf("L_std at (%g, %g)", xy[1], xy[2]))
    expect_equal(res$L_cb,  res$L_analytical, tolerance = 1e-12,
                 info = sprintf("L_cb at (%g, %g)", xy[1], xy[2]))

    # Backward: standard arena path matches analytical (sanity).
    expect_equal(res$dx_std, res$dx_analytical, tolerance = 1e-10)
    expect_equal(res$dy_std, res$dy_analytical, tolerance = 1e-10)

    # Backward: custom_backward path matches analytical -- this is the
    # actual property under test.
    expect_equal(res$dx_cb, res$dx_analytical, tolerance = 1e-10,
                 info = sprintf("dx_cb at (%g, %g)", xy[1], xy[2]))
    expect_equal(res$dy_cb, res$dy_analytical, tolerance = 1e-10,
                 info = sprintf("dy_cb at (%g, %g)", xy[1], xy[2]))
  }
})
