#' @section Computation device:
#' `control$device` is `"auto"`, `"cpu"`, or `"cuda"`. CUDA routes are
#' provided by the tulpaGPU package (<https://gcol33.r-universe.dev>), loaded
#' when a fit asks for `"auto"` or `"cuda"`. The default `"auto"` uses CUDA for
#' registered operations above their measured size crossover when tulpaGPU is
#' installed, and the CPU otherwise. `"cuda"` requests CUDA for each registered
#' operation and errors with the reason when tulpaGPU, its device, or a library
#' is unavailable. Host inference stages run on CPU. The fit's `device$stages`
#' table records CPU and CUDA call counts per stage; `print()` and
#' [diagnostics()] report these stages.
