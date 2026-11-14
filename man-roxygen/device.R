#' @section Computation device:
#' `control$device` is `"auto"`, `"cpu"`, or `"cuda"`. The default `"auto"`
#' uses CUDA for registered operations above their measured size crossover.
#' Operations with no measured crossover run on CPU. `"cuda"` requests CUDA
#' for each registered operation and errors with the reason from [gpu_info()]
#' when its device or library is unavailable. Host inference stages run on
#' CPU. The fit's `device$stages` table records CPU and CUDA call counts per
#' stage; `print()` and [diagnostics()] report these stages.
