#!/usr/bin/env bash
# Fetch the anonymized trace: one parquet file, the artifact's whole dataset.
#
# Everything else derives from it — `make trace` builds the DuckDB view, and
# `make dataset` rebuilds the recovered sessions in about 3 minutes. The user
# cohort table is not fetched: it ships in the repository at data/trace/.
#
# Where from (defaults to archive.base_url in config/workload.json):
#   --url BASE     download over HTTP; resumes an interrupted transfer
#   --from PATH    copy from a local file, or a directory containing it
#   --link PATH    symlink instead of copying
#
# Lands in $CHUTES_TRACE_DIR (default output/trace/).
#
# Usage:
#   scripts/fetch_data.sh --url https://<bucket>.s3.<region>.amazonaws.com/<prefix>
#   scripts/fetch_data.sh --from /path/to/trace
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python3}"

py() { "$PYTHON" -c "
import sys; sys.path.insert(0, '$REPO/src')
from chutes_sim import artifact
$1"; }

# `trace` is accepted as a no-op subject so older instructions keep working.
[ "${1:-}" = "trace" ] && shift

FROM=""; BASE_URL=""; LINK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --link) FROM="$2"; LINK=1; shift 2 ;;
    --url)  BASE_URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ -z "$FROM" ] && [ -z "$BASE_URL" ]; then
  BASE_URL="$(py "print(artifact.workload_config().get('archive', {}).get('base_url', ''))")"
  if [ -z "$BASE_URL" ] || [[ "$BASE_URL" == REPLACE_ME* ]]; then
    cat >&2 <<'EOF'
No archive URL configured.

  * Already on this machine:  scripts/fetch_data.sh --from /path/to/trace
  * Published elsewhere:      scripts/fetch_data.sh --url https://<host>/<prefix>
    (set archive.base_url in config/workload.json to make that the default)
EOF
    exit 2
  fi
fi

TRACE_FILE="$(py "print(artifact.trace_file())")"
NAME="$(basename "$TRACE_FILE")"
mkdir -p "$(dirname "$TRACE_FILE")"
echo "trace file: $TRACE_FILE"

if [ -n "$FROM" ]; then
  SRC="$FROM"; [ -d "$SRC" ] && SRC="$SRC/$NAME"
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
  command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
  # -C - resumes a partial download, which matters at this size.
  curl -fL --retry 3 -C - -o "$TRACE_FILE" "$BASE_URL/$NAME"
fi

exec "$PYTHON" "$HERE/verify_data.py" --trace "$TRACE_FILE"
