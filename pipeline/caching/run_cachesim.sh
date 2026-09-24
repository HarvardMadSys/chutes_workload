#!/usr/bin/env bash
# End-to-end run of the single-cache eviction study.
#
# Produces data/caching/results.txt and the paper's two token-hit-ratio
# figures under figures/caching/.
#
# Pipeline:
#   1. build_oracle_traces.py : output/sessions/<model>.parquet -> <out>/traces/<model>.oracleGeneral
#   2. clone + pin + patch + build libCacheSim, with a small KV-cache patch
#      (patches/0001-*.patch) that makes a cache hit grow the object to the new
#      request size and evict immediately on overflow — the semantics KV-cache
#      reuse actually has. LightGBM is built locally so that
#      -DENABLE_LRB=ON links.
#   3. cachesim sweep         : 9 algorithms x 10 cache sizes x 2 workloads
#   4. plot_hit_ratio.py      : the two token-hit-ratio figures the paper includes
#
# Usage: ./run_cachesim.sh [--skip-preprocess] [--skip-build] [--skip-sweep]
#
# Env: PYTHON (default python3), NJOBS (default: nproc), CHUTES_OUTPUT_ROOT.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
NJOBS="${NJOBS:-$(nproc 2>/dev/null || echo 8)}"

# Everything generated lands under output/caching/.
WORK="$("$PYTHON" -c "
import sys; sys.path.insert(0, '$REPO/src')
from chutes_sim import artifact; print(artifact.caching_dir())")"
TRACES="$WORK/traces"
BUILD="$WORK/build"
DEPS="$BUILD/deps"
RESULTS="$WORK/results"

SKIP_PREPROCESS=0
SKIP_BUILD=0
SKIP_SWEEP=0
for arg in "$@"; do
  case "$arg" in
    --skip-preprocess) SKIP_PREPROCESS=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-sweep) SKIP_SWEEP=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg $arg" >&2; exit 2 ;;
  esac
done

cfg() { "$PYTHON" -c "
import json,sys; d=json.load(open('$REPO/config/cachesim.json'))
for k in '$1'.split('.'): d = d[k]
print(d if not isinstance(d,(list,dict)) else ' '.join(map(str,d)))"; }

LIBCACHESIM_REPO="$(cfg libcachesim.repo)"
LIBCACHESIM_COMMIT="$(cfg libcachesim.commit)"
LIGHTGBM_REPO="$(cfg lightgbm.repo)"
LIGHTGBM_COMMIT="$(cfg lightgbm.commit)"

mkdir -p "$WORK" "$RESULTS"

# ---------------------------------------------------------------- 1. traces
if [[ $SKIP_PREPROCESS -eq 0 ]]; then
  "$PYTHON" "$REPO/pipeline/caching/build_oracle_traces.py" --out-dir "$TRACES"
fi

# ---------------------------------------------------------------- 2. build
SRC="$BUILD/src"
if [[ $SKIP_BUILD -eq 0 ]]; then
  mkdir -p "$SRC"

  # -- LightGBM (required by -DENABLE_LRB=ON) --------------------------------
  # Blobless clone + shallow submodules: the full recursive clone pulls ~1 GB
  # of eigen history for a dependency whose model never even trains here.
  if [[ ! -d "$SRC/LightGBM/.git" ]]; then
    git clone --filter=blob:none "$LIGHTGBM_REPO" "$SRC/LightGBM"
  fi
  git -C "$SRC/LightGBM" checkout -q "$LIGHTGBM_COMMIT" \
    || { git -C "$SRC/LightGBM" fetch origin && git -C "$SRC/LightGBM" checkout -q "$LIGHTGBM_COMMIT"; }
  git -C "$SRC/LightGBM" submodule update --init --recursive --depth 1
  if [[ ! -f "$DEPS/lib/lib_lightgbm.so" ]]; then
    cmake -S "$SRC/LightGBM" -B "$SRC/LightGBM/build" \
      -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SRC/LightGBM/build" -j "$NJOBS"
    cmake --install "$SRC/LightGBM/build"
  fi

  # -- libCacheSim, pinned to the commit the patch was written against -------
  LCS="$SRC/libCacheSim"
  if [[ ! -d "$LCS/.git" ]]; then
    git clone "$LIBCACHESIM_REPO" "$LCS"
  fi
  # Reset to the pinned commit and re-apply the patch from a known-clean tree.
  git -C "$LCS" fetch origin
  git -C "$LCS" checkout -q --force "$LIBCACHESIM_COMMIT"
  git -C "$LCS" reset -q --hard "$LIBCACHESIM_COMMIT"
  git -C "$LCS" clean -qfd -e _build -e _build_asan
  PATCH="$REPO/pipeline/caching/patches/0001-kv-update-obj-size-on-hit.patch"
  git -C "$LCS" apply --check "$PATCH" || {
    echo "ERROR: $PATCH does not apply cleanly to $LIBCACHESIM_COMMIT" >&2; exit 1; }
  git -C "$LCS" apply "$PATCH"
  # Fail loudly unless every file the patch touches actually changed.
  changed=$(git -C "$LCS" diff --name-only | sort | tr '\n' ' ')
  for f in libCacheSim/cache/cache.c libCacheSim/cache/eviction/ARC.c \
           libCacheSim/cache/eviction/LRB/lrb.cpp libCacheSim/cache/eviction/cpp/GDSF.cpp; do
    case " $changed " in *" $f "*) ;; *) echo "ERROR: patch did not modify $f" >&2; exit 1 ;; esac
  done
  echo "patched: $changed"

  # Notes on flags:
  #  - OPT_SUPPORT_ZSTD_TRACE=OFF : our traces are uncompressed, and this avoids
  #    a hard dependency on zstd headers.
  #  - CMAKE_CXX_FLAGS -isystem   : upstream commented out
  #    include_directories(${LIGHTGBM_PATH}), so the LightGBM headers must be
  #    injected via compiler flags.
  #  - UPDATE_OBJ_SIZE_ON_HIT     : activates the KV-cache patch. It MUST be in
  #    BOTH C and CXX flags; setting only one silently reverts the C++ policies
  #    to vanilla semantics.
  cmake -S "$LCS" -B "$LCS/_build" -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_LRB=ON -DOPT_SUPPORT_ZSTD_TRACE=OFF -DENABLE_TESTS=OFF \
    -DCMAKE_INCLUDE_PATH="$DEPS/include" -DCMAKE_LIBRARY_PATH="$DEPS/lib" \
    -DCMAKE_CXX_FLAGS="-isystem $DEPS/include -DUPDATE_OBJ_SIZE_ON_HIT" \
    -DCMAKE_C_FLAGS="-DUPDATE_OBJ_SIZE_ON_HIT"
  cmake --build "$LCS/_build" -j "$NJOBS"
fi

CACHESIM="$SRC/libCacheSim/_build/bin/cachesim"

# ---------------------------------------------------------------- 3. sweep
if [[ $SKIP_SWEEP -eq 0 ]]; then
  [[ -x "$CACHESIM" ]] || { echo "no cachesim binary at $CACHESIM (drop --skip-build)" >&2; exit 1; }
  export LD_LIBRARY_PATH="$DEPS/lib:$SRC/libCacheSim/_build:${LD_LIBRARY_PATH:-}"
  # fractions of the byte working-set size; these resolve to the KiB sizes
  # the figures plot (61..61631 KiB / 205..205999 KiB)
  FRACS="$(cfg cache_size_fractions)"
  read -r -a ALGOS <<< "$(cfg algorithms)"
  read -r -a WORKLOADS <<< "$("$PYTHON" -c "
import sys; sys.path.insert(0, '$REPO/src')
from chutes_sim import artifact; print(' '.join(artifact.models()))")"

  : > "$RESULTS/results.txt"
  for w in "${WORKLOADS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      echo ">>> $w / $algo"
      EXTRA=()
      # pre-2023 S3FIFO default (label S3FIFO-0.1000-1)
      [[ $algo == s3fifo ]] && EXTRA=(-e "move-to-main-threshold=1")
      ( cd "$TRACES" && "$CACHESIM" "$w.oracleGeneral" oracleGeneral "$algo" "$FRACS" \
          "${EXTRA[@]}" 2>/dev/null ) \
        | grep "cache size" | grep -v hour >> "$RESULTS/results.txt"
    done
  done
  echo "wrote $RESULTS/results.txt ($(wc -l < "$RESULTS/results.txt") lines)"
fi

# ---------------------------------------------------------------- 4. figures
RESULTS_TXT="$RESULTS/results.txt"
COMMITTED="$REPO/data/caching/results.txt"
FIGS="$REPO/figures/caching"

# With --skip-sweep and no local run yet, plot the committed results.
if [[ ! -s "$RESULTS_TXT" ]]; then
  echo "no $RESULTS_TXT — plotting the committed $COMMITTED instead"
  RESULTS_TXT="$COMMITTED"
fi

# The two token-hit-ratio figures the paper includes, in the paper's style.
"$PYTHON" "$REPO/pipeline/caching/plot_hit_ratio.py" "$RESULTS_TXT" --out-dir "$FIGS"

# Keep the committed copy in sync with what just ran.
if [[ "$RESULTS_TXT" != "$COMMITTED" ]]; then
  cp "$RESULTS_TXT" "$COMMITTED"
  echo "done:"
  echo "  results : $RESULTS_TXT (copied to data/caching/results.txt)"
else
  echo "done:"
  echo "  results : $COMMITTED (replotted; no new sweep)"
fi
echo "  figures : $FIGS"
