test_that("VI Monte Carlo gradients match a Gaussian reference and stopping uses stationarity", {
    z <- tulpa:::cpp_test_vi_math()
    expect_lt(max(z[c("meanfield", "fullrank", "lowrank")]), 1e-12)
    expect_lt(z["inverse_error"], 1e-13)
    expect_lt(z["alignment_error"], 1e-13)
    expect_equal(unname(z[c("offset_invariant", "constant_rejected")]), c(1, 1))
})
