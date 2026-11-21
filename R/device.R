.device_state <- new.env(parent = emptyenv())

# CUDA routes come from tulpaGPU, which registers its backend when it loads.
# A fit that may use the device loads it once per session when it is installed.
.device_backend_load <- function() {
  if (!isNamespaceLoaded("tulpaGPU") && is.null(.device_state$attempted)) {
    .device_state$attempted <- TRUE
    if (nzchar(system.file(package = "tulpaGPU")))
      tryCatch(loadNamespace("tulpaGPU"),
               error = function(e) .device_state$error <- conditionMessage(e))
  }
  invisible(NULL)
}

.device_choice <- function(control) {
  choice <- control$device %||% "auto"
  if (identical(choice, "auto") || identical(choice, "cuda")) .device_backend_load()
  if (identical(choice, "cuda") && !isNamespaceLoaded("tulpaGPU")) {
    if (!is.null(.device_state$error))
      stop("control$device = \"cuda\": the tulpaGPU package failed to load: ",
           .device_state$error, call. = FALSE)
    stop("control$device = \"cuda\" needs the tulpaGPU package, which provides the CUDA routes: ",
         "install.packages(\"tulpaGPU\", repos = \"https://gcol33.r-universe.dev\")", call. = FALSE)
  }
  cpp_device_validate(choice)
}

.device_start <- function(control) cpp_device_begin(.device_choice(control))

.device_attach <- function(fit) {
  fit$device <- cpp_device_snapshot()
  fit
}

.print_device <- function(device) {
  stages <- device$stages
  if (!is.null(stages) && nrow(stages)) {
    cat("Device: ", paste(paste0(stages$stage, "=", stages$device), collapse = ", "), "\n", sep = "")
  }
  invisible(NULL)
}
