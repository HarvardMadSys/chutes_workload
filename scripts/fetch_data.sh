#!/usr/bin/env bash
# Fetch the released trace: one 91 GB parquet file, the artifact's only download.
#
# Everything else derives from it: `make trace` builds the DuckDB view and
# `make sessions` reconstructs the sessions (about 3 minutes). The user cohort
# table ships in the repository (data/trace/).
#
#   scripts/fetch_data.sh               download from trace.url in config/workload.json
#                                       (resumes an interrupted transfer)
#   scripts/fetch_data.sh --url URL     download from another URL
#   scripts/fetch_data.sh --from PATH   copy a local file (or a directory containing it)
#   scripts/fetch_data.sh --link PATH   symlink it instead of copying
#
# The file lands in $CHUTES_TRACE_DIR (default output/trace/) and is then
# checked by scripts/verify_data.py.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python3}"

py() { "$PYTHON" -c "
import sys; sys.path.insert(0, '$REPO/src')
from chutes_sim import artifact
$1"; }

FROM=""; URL=""; LINK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --link) FROM="$2"; LINK=1; shift 2 ;;
    --url)  URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

TRACE_FILE="$(py "print(artifact.trace_file())")"
mkdir -p "$(dirname "$TRACE_FILE")"
echo "trace file: $TRACE_FILE"

if [ -n "$FROM" ]; then
  SRC="$FROM"; [ -d "$SRC" ] && SRC="$SRC/$(basename "$TRACE_FILE")"
  [ -f "$SRC" ] || { echo "no $SRC" >&2; exit 1; }
  rm -f "$TRACE_FILE"
  if [ "$LINK" -eq 1 ]; then
    ln -s "$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")" "$TRACE_FILE"
    echo "linked -> $SRC"
  else
    cp "$SRC" "$TRACE_FILE"
    echo "copied into $TRACE_FILE"
  fi
else
  [ -n "$URL" ] || URL="$(py "print(artifact.workload_config()['trace']['url'])")"
  command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
  # -C - resumes a partial download, which matters at this size.
  curl -fL --retry 3 -C - -o "$TRACE_FILE" "$URL"
fi

exec "$PYTHON" "$HERE/verify_data.py" --trace "$TRACE_FILE"
