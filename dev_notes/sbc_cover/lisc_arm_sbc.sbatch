#!/bin/bash -l
#SBATCH --job-name=cg_armsbc
#SBATCH --partition=zen2_1024,zen2_1024_t4x1
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%a_%j.out
#SBATCH --error=%x_%a_%j.err
#
# One grid SBC arm per array task, read from a tab-separated arm table whose
# header is
#
#   tag  role  J  sigma_nodes  phi_nodes  refine  n_sim  [alpha_n  [seed]]
#
# and whose data row k is array task k. `alpha_n` is optional: absent, the arm
# runs a bare share(spatial()) (SBC_ALPHA_N = 0); present, it is the node count
# asked of the copy axis through `alpha = grid(n = alpha_n)`. `seed` is optional
# too and needs `alpha_n` before it: absent, SBC_SEED = 20260820. The engine is
# named in the environment
# and checked by lisc_engine_check.sh; the driver copy must match DRIVER_MD5.
#
#   sbatch --array=0-3 --export=ALL,ARMS=<table>,ENGINE_ID=<id>,\
#          ENGINE_TULPA_SHA=<sha>,ENGINE_TULPAOBS_SHA=<sha>,DRIVER_MD5=<md5> \
#          lisc_arm_sbc.sbatch
#
# Fixed across every arm, as in the fixed-engine arms (PREREG_fixed_engine.md):
# beta family, pinned grids on the declared spans,
# SBC_SEED from the arm table (default 20260820), wide and narrow controls. Sized off sacct of 6443711:
# 1.7 to 3.5 h wall, 0.9 to 2.1 GB.

set -euo pipefail

ROOT="$HOME/cover_sbc"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 3; }
: "${ARMS:?ARMS is not set}"; : "${DRIVER_MD5:?DRIVER_MD5 is not set}"
[ -f "$ARMS" ] || fail "arm table $ARMS not found"

module load R/4.5.2-gfbf-2025b R-bundle-CRAN/2025.11-foss-2025b
source "$ROOT/lisc_engine_check.sh"

i="${SLURM_ARRAY_TASK_ID}"
HDR="$(head -1 "$ARMS" | tr '\t' ' ')"
case "$HDR" in
  "tag role J sigma_nodes phi_nodes refine n_sim"|"tag role J sigma_nodes phi_nodes refine n_sim alpha_n"|"tag role J sigma_nodes phi_nodes refine n_sim alpha_n seed") ;;
  *) fail "arm table header is not 'tag role J sigma_nodes phi_nodes refine n_sim [alpha_n [seed]]'" ;;
esac
ROW="$(awk -F'\t' -v k="$((i + 2))" 'NR == k' "$ARMS")"
[ -n "$ROW" ] || fail "arm table has no row for task $i"
IFS=$'\t' read -r TAG ROLE J S P R N A SEED <<< "$ROW"
A="${A:-0}"
SEED="${SEED:-20260820}"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "seed '$SEED' is not a non-negative integer"
[[ "$A" =~ ^[0-9]+$ ]] || fail "alpha_n '$A' is not a non-negative integer"

export SBC_J="$J" SBC_NODES_SIGMA="$S" SBC_NODES_PHI="$P" SBC_REFINE="$R" SBC_N_SIM="$N"
export SBC_ALPHA_N="$A" SBC_FAMILY=beta SBC_GRID_MODE=pinned
export SBC_SIGMA_PLACEMENT=pinned SBC_PHI_PLACEMENT=pinned SBC_METHOD=nested_laplace
export SBC_SEED="$SEED"
export SBC_OUT="$ROOT/results/$TAG"
mkdir -p "$SBC_OUT" "$ROOT/logs"

NT="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NT"
export SBC_THREADS_OUTER="$NT"
export SBC_ENGINE_SHA="tulpa=${ENGINE_TULPA_SHA:0:7} tulpaObs=${ENGINE_TULPAOBS_SHA:0:7}"

for v in SBC_NUTS_ITER SBC_NUTS_WARMUP SBC_NUTS_CHAINS; do
  [ -z "${!v:-}" ] || fail "$v is set; this arm is a grid arm"
done

SNAP="$ROOT/snapshots/${SLURM_ARRAY_JOB_ID}_${i}_${TAG}"
mkdir -p "$SNAP"
cp "$ROOT/sbc_occu_cover.R" "$SNAP/driver.R"
cp "$ARMS" "$SNAP/arms.tsv"
MD5="$(md5sum "$SNAP/driver.R" | cut -d" " -f1)"
[ "$MD5" = "$DRIVER_MD5" ] || fail "driver md5 $MD5 is not the registered $DRIVER_MD5"

echo "=== $TAG ($ROLE) | J=$J sigma=$S phi=$P refine=$R alpha_n=$A n.sim=$N | $SBC_ENGINE_SHA | $(hostname) | cpus $NT ==="

LOG="$ROOT/logs/${TAG}_${SLURM_ARRAY_JOB_ID}.log"
t0=$(date +%s)
set +e
Rscript "$SNAP/driver.R" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
t1=$(date +%s)

{
  echo "arm=$TAG"
  echo "role=$ROLE"
  echo "family=$SBC_FAMILY"
  echo "grid_mode=$SBC_GRID_MODE"
  echo "refine=$R"
  echo "J=$J"
  echo "sigma_nodes=$S"
  echo "phi_nodes=$P"
  echo "alpha_request=$([ "$A" = 0 ] && echo none || echo "grid(n = $A)")"
  echo "seed=$SBC_SEED"
  echo "n_sim=$N"
  echo "threads_outer=$NT"
  echo "engine_id=$ENGINE_ID"
  echo "engine=tulpa=$ENGINE_TULPA_SHA tulpaObs=$ENGINE_TULPAOBS_SHA"
  echo "library=$ENGINE_LIB"
  echo "arm_table=$ARMS"
  echo "driver_md5=$MD5"
  echo "driver_snapshot=$SNAP/driver.R"
  echo "node=$(hostname)"
  echo "jobid=${SLURM_ARRAY_JOB_ID}_${i}"
  echo "exit=$rc"
  echo "wall_min=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", (b-a)/60}')"
  grep -E '^(grid refinement|outer grid|slice cells|SBC wall clock)' "$LOG" | sed 's/^/log: /'
} > "$ROOT/DONE_$TAG.txt"

exit $rc
