# Redirect a backend selection: swap the backend and restamp mode / tier / tier name from the registry, recording the reason shown on the fit.

A redirect off an EXPLICIT request (anything but `mode = "auto"`) is an
override, not a resolution: the fit does what the model structure
requires instead of what the caller asked for. That is recorded on the
selection – `sel$overridden` for callers, and a clause appended to the
reason the fit reports – so an overridden fit is distinguishable from
one that was never asked for a mode at all. Only the FIRST override is
recorded: with several redirects in a chain, the request the user
actually made is the one worth naming, not the intermediate backend a
previous redirect chose.

## Usage

``` r
.sel_redirect(sel, backend, reason, notify = TRUE)
```

## Details

`notify` says whether tulpa() should additionally WARN. It is for the
case where the requested backend could have fitted the model and the
redirect takes a capability away – a smoother sending an `eb` / `agq`
request to the nested kernels loses the random-effect SD those two would
have estimated. It is off where the requested mode is not expressible
for the structure at all and the redirect is the documented resolution
rather than a loss: a random slope has no scalar `sigma_re` for
`mode = "laplace"` to condition on, and an SPDE field redirected from
`nested_laplace` to `spde` is the same mode and tier reaching its own
integrator. Those are recorded, not warned about, so a documented route
does not warn on every fit.
