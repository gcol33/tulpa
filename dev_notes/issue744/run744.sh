#!/usr/bin/env bash
# gcol33/tulpa#744: every outer-grid research figure re-measured on the fixture
# whose fit carries the residual variance its data were simulated at.
#
#   bash run744.sh <lib> <repo> <group>
#
# `lib` holds a clean install whose BUILD_ID names the commit and the test files
# the sweeps read from `repo`; `group` is one of `bary`, `plane`, `rank`. Each
# group runs its arms in order, the headline arm (flat prior, box_uniform read,
# the arm the test files state) first.
set -euo pipefail
LIB="$1"; REPO="$2"; GROUP="$3"
RS="/c/Program Files/R/R-4.6.1/bin/Rscript.exe"
N="$REPO/dev_notes"
OUT="$N/issue744/out"
mkdir -p "$OUT"
test -f "$LIB/BUILD_ID" || { echo "no BUILD_ID in $LIB"; exit 1; }
head -1 "$LIB/BUILD_ID"

case "$GROUP" in
  bary)
    "$RS" "$N/issue744/boxmass744.R" "$LIB" "$REPO" proper > "$OUT/boxmass_proper.txt" 2>&1
    "$RS" "$N/issue744/boxmass744.R" "$LIB" "$REPO" flat   > "$OUT/boxmass_flat.txt" 2>&1
    "$RS" "$N/issue744/ogd744.R" "$LIB" "$REPO" > "$OUT/ogd.txt" 2>&1
    for arm in "flat box_uniform" "proper box_uniform" "flat chord" "proper chord"; do
      set -- $arm
      tag="$1"; [ "$2" = chord ] && tag="${1}_chord"
      "$RS" "$N/issue327/bary327.R" "$LIB" "$REPO" "$1" "$2" > "$OUT/bary_$tag.txt" 2>&1
    done
    ;;
  plane)
    for arm in "flat box_uniform" "proper box_uniform" "flat chord" "proper chord"; do
      set -- $arm
      tag="$1_gaussian"; [ "$2" = chord ] && tag="${tag}_chord"
      "$RS" "$N/issue333/plane333.R" "$LIB" "$REPO" "$1" gaussian "$OUT/dp_$tag.rds" \
        1:8 2,3,4 6 "$2" > "$OUT/plane_$tag.log" 2>&1
      "$RS" "$N/issue333/analyse333.R" "$LIB" "$REPO" "$OUT/dp_$tag.rds" \
        > "$OUT/an_${tag}_all.txt" 2>&1
      "$RS" "$N/issue333/analyse333.R" "$LIB" "$REPO" "$OUT/dp_$tag.rds" 3 1:8 \
        > "$OUT/an_${tag}_s3.txt" 2>&1
    done
    ;;
  rank)
    for arm in "flat box_uniform" "proper box_uniform" "flat chord" "proper chord"; do
      set -- $arm
      tag="$1"; [ "$2" = chord ] && tag="${1}_chord"
      "$RS" "$N/issue_328/measure_fit_ranking.R" "$LIB" "$REPO" "$1" "$2" \
        "$OUT/fitrank_$tag.csv" > "$OUT/fitrank_$tag.txt" 2>&1
    done
    ;;
  *) echo "unknown group $GROUP"; exit 1 ;;
esac
echo "done $GROUP"
