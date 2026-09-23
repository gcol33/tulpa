# Progress reporter for a counted R-side fitting loop

Build a progress reporter for a counted R loop (an EM iteration, a
per-species outer loop, ...) – the R-side counterpart of the C++
`tulpa_progress::GridProgress` reporter, sharing its wire format so a
detached reader sees the same heartbeat file regardless of which kind of
loop produced it. Config comes from the scoped `tulpa.nl_progress`
option (`progress`, `progress_file`, `progress_every`,
`progress_throttle`); a caller outside tulpa's own `control$progress`
machinery sets the same option to opt in.

Two independently gated channels: a console bar, emitted when `progress`
is set on the option, and a heartbeat file, written whenever
`progress_file` is non-empty regardless of `progress` – a file write
survives a detached `Start-Process` / `nohup` stdout buffer where a
console flush does not, so it is the only liveness signal on a headless
box.

## Usage

``` r
tulpa_iter_progress(label, total, unit = "iter", threads = 1L)
```

## Arguments

- label:

  Short name for the loop, shown in the console line.

- total:

  Iteration denominator (e.g. `max_iter` for an EM loop).

- unit:

  Name for one step in the console line (`"iter"`, `"species"`, ...).

- threads:

  Active outer-thread count. When `> 1` it is appended to the console
  line as `"| N threads"` so the parallelism is visible live and in
  detached-run logs; a serial loop leaves it at 1 and the field is
  omitted, mirroring the C++ `GridProgress` reporter.

## Value

A list of two closures: `tick()`, called once per completed step, and
`finish()`, called once after the loop (forces a final 100%/heartbeat
emit). When neither channel is requested both are zero-overhead no-ops.
