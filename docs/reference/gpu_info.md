# CUDA device availability

Report the CUDA driver, the selected NVIDIA device, and the runtime
libraries used by the engine. Installation requires a C++ compiler only.
Device operations load CUDA libraries when first requested.

## Usage

``` r
gpu_info()

# S3 method for class 'tulpa_gpu_info'
print(x, ...)
```

## Arguments

- x:

  A `gpu_info()` result.

- ...:

  Ignored.

## Value

An object of class `tulpa_gpu_info` with availability, the reason for
unavailability, device properties, and library paths and reasons.

## Examples

``` r
gpu_info()
#> CUDA: NVIDIA GeForce RTX 5080, 16275 MiB, compute capability 12.0
```
