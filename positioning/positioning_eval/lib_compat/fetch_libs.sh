#!/usr/bin/env bash
# Fetch the SuiteSparse runtime libraries the shipped PPPx binary was linked against.
#
# The binary needs libspqr.so.2 / libcholmod.so.3 / libcxsparse.so.3, i.e. the
# SuiteSparse 5 generation that Debian 12 (bookworm) shipped. Debian 13 (trixie)
# ships SuiteSparse 7 with different SONAMEs, so after an OS upgrade PPPx fails
# at load time with:
#
#   error while loading shared libraries: libspqr.so.2: cannot open shared object file
#
# This unpacks the matching bookworm runtime packages into ./root. Nothing is
# installed system-wide and no root is required; run_positioning_evaluation.py
# prepends the directory to LD_LIBRARY_PATH for the PPPx subprocess only.
#
# Do NOT symlink the system's newer libraries under the old names instead. The
# CHOLMOD structures changed across those major versions, so the loader would
# resolve symbols by name and pass mismatched structs - silently wrong results
# rather than a clean failure.
#
# Delete the ./dl and ./root directories to undo.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p dl root

BASE="http://deb.debian.org/debian/pool/main/s"
SUITESPARSE_VERSION="5.12.0+dfsg-2"
PACKAGES=(
  libspqr2 libcholmod3 libcxsparse3
  libamd2 libcamd2 libccolamd2 libcolamd2 libsuitesparseconfig5
)

for package in "${PACKAGES[@]}"; do
  deb="${package}_${SUITESPARSE_VERSION}_amd64.deb"
  if [[ ! -f "dl/${deb}" ]]; then
    echo "downloading ${package}"
    curl -sfL --max-time 60 -o "dl/${deb}" "${BASE}/suitesparse/${deb}"
  fi
  dpkg-deb -x "dl/${deb}" root
done

echo
echo "Extracted to $(pwd)/root"
ldd ../pppx | grep -i "not found" && {
  echo "PPPx still has unresolved libraries — see above" >&2
  exit 1
}
echo "All PPPx libraries resolve."
