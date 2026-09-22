# occu_cover SBC arms

The SBC runs behind the #858-#861 grid and draw work. Everything here runs on
LiSC; this directory is the tracked copy of what drives it, because the working
tree lives only at `~/cover_sbc` on the cluster.

## Files

- `lisc_arm_sbc.sbatch` -- the array launcher. One grid arm per array task, read
  from a tab-separated arm table. Pins the engine by commit sha and refuses to
  start if the driver's md5 is not the one registered on the command line.
- `sbc_occu_cover.R` -- the driver the launcher snapshots and runs.
- `HP760_*.tsv` -- arm tables. `HP760_859.tsv` / `HP760_860.tsv` are the two
  engines at the original seed; the `_seed2` pair carries the `seed` column
  added for the second seed (20260922).
- `reports/` -- `sbc_report.csv` per run, named by its arm tag.
- `pit_dispersion.R` -- reads a run's per-simulation PITs and reports
  `sd(qnorm(PIT))` per quantity.

The per-simulation detail (`sbc_occu_cover.rds`, 5.7 MB per run) stays on LiSC
under `~/cover_sbc/results/<tag>/`.

## Running an arm

```
sbatch --array=0-N --export=ALL,ARMS=<table>,ENGINE_ID=<id>,\
       ENGINE_TULPA_SHA=<sha>,ENGINE_TULPAOBS_SHA=<sha>,DRIVER_MD5=<md5> \
       lisc_arm_sbc.sbatch
```

Arm table header, the optional columns last:

```
tag  role  J  sigma_nodes  phi_nodes  refine  n_sim  [alpha_n  [seed]]
```

`alpha_n` absent runs a bare `share(spatial())`; `seed` absent uses 20260820,
and needs `alpha_n` before it. Sized off `sacct` of 6443711: 1.7 to 3.5 h wall,
0.9 to 2.1 GB, 16 cpus.

## Engines

`ENGINE_ID` selects a prebuilt library under `~/cover_sbc/lib_<id>`, with the
matching source tree at `~/cover_sbc/src_<id>`. `lib_859cop` and `lib_860cop`
are the two engines the copula work is measured across.
