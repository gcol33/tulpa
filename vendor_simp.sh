#!/usr/bin/env bash
# Vendor SIMP's symplectic-integrator headers into tulpa/src/simp/.
#
# SIMP (gcol33/SIMP) is the upstream development home for the integrator core.
# This copies a snapshot into tulpa's own source tree so tulpa builds
# self-contained -- no LinkingTo to a non-CRAN package, no Additional_repositories.
# Re-run after updating SIMP. Mirrors vectra's vendor_tdc.sh.
#
# Usage: ./vendor_simp.sh [path-to-SIMP]   (defaults to the sibling ../SIMP)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
simp="${1:-$(cd "$here/../SIMP" && pwd)}"
src="$simp/inst/include/simp"
dest="$here/src/simp"

[ -d "$src" ] || { echo "SIMP headers not found at $src" >&2; exit 1; }
commit="$(git -C "$simp" rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p "$dest"
rm -f "$dest"/*.h
n=0
for f in "$src"/*.h; do
  base="$(basename "$f")"
  {
    echo "// Vendored from gcol33/SIMP (${commit}) by vendor_simp.sh -- do not edit."
    echo "// Upstream: SIMP inst/include/simp/${base}. Edit there and re-vendor."
    echo "// Copyright (c) 2026 Gilles Colling. MIT license (see inst/COPYRIGHTS)."
    echo ""
    cat "$f"
  } > "$dest/$base"
  n=$((n + 1))
done

echo "Vendored $n SIMP headers into src/simp/ (SIMP $commit)"
