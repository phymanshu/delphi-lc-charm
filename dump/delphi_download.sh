#!/bin/bash
#
# delphi_download.sh - download the DELPHI open-data record ids listed in a CSV
#
# Usage:  ./delphi_download.sh [-j N] [-n] keep.csv DEST_DIR
#
#   keep.csv   CSV with a header line and columns: recid,name,files,GiB
#              (delphi_keep.csv, written by the classification step)
#   DEST_DIR   where to put the data, one sub-directory per dataset name.
#              Use a big area (EOS user space or a large work area), NOT your
#              AFS home: the selected sets are a few hundred GiB.
#
#   -j N       number of parallel file transfers   (default 4)
#   -n         dry run: list the files that would be fetched, copy nothing
#   -c         check existing files: compare each local file's size with the size
#              on the server (xrdfs stat) and re-download truncated/corrupt ones.
#              New downloads are always size-checked.
#
# Needs:  cernopendata-client  (pip install --no-cache-dir cernopendata-client)
#         xrdcp                (available on lxplus)
#
# It is safe to re-run: files that already exist at the destination are
# skipped, and each file is written as <name>.part and renamed on success, so
# an interrupted copy never looks complete.

JOBS=4
DRY=0
VERIFY=0
while getopts "j:nch" opt; do
  case $opt in
    j) JOBS=$OPTARG ;;
    n) DRY=1 ;;
    c) VERIFY=1 ;;
    *) sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
CSV=${1:-}; DEST=${2:-}
[ -f "$CSV" ] && [ -n "$DEST" ] || { sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
case $JOBS in ''|*[!0-9]*|0) echo "ERROR: -j needs a positive integer" >&2; exit 1 ;; esac
command -v cernopendata-client >/dev/null 2>&1 || { echo "ERROR: cernopendata-client not found" >&2; exit 1; }
[ $DRY -eq 1 ] || command -v xrdfs >/dev/null 2>&1 || { echo "ERROR: xrdfs not found" >&2; exit 1; }
[ $DRY -eq 1 ] || command -v xrdcp >/dev/null 2>&1 || { echo "ERROR: xrdcp not found" >&2; exit 1; }

mkdir -p "$DEST/lists" || { echo "ERROR: cannot create $DEST" >&2; exit 1; }
DEST=$(cd "$DEST" && pwd)

# ----------------------------------------------------- build the work list
WORK=$DEST/lists/work_list.tsv      # lines: <name> <uri>
: > "$WORK"
nrec=0; nbad=0
while IFS=, read -r recid name files gib; do
  [ "$recid" = "recid" ] && continue
  [ -n "$recid" ] || continue
  nrec=$((nrec + 1))
  lst=$DEST/lists/$name.txt
  if [ ! -s "$lst" ]; then
    cernopendata-client get-file-locations --recid "$recid" --protocol xrootd 2>"$DEST/lists/$name.err" \
      | grep '^root://' > "$lst"
  fi
  n=$(wc -l < "$lst")
  if [ "$n" -eq 0 ]; then
    echo "[WARN] $name (recid $recid): no file locations returned, see $DEST/lists/$name.err"
    nbad=$((nbad + 1)); rm -f "$lst"; continue
  fi
  if [ -n "$files" ] && [ "$n" -ne "$files" ]; then
    echo "[WARN] $name: portal says $files files, client listed $n"
  fi
  echo "[list] $name: $n files"
  awk -v nm="$name" '{print nm "\t" $0}' "$lst" >> "$WORK"
done < "$CSV"

total=$(wc -l < "$WORK")
echo "Records: $nrec ($nbad without a file list)   files to consider: $total"
if [ $DRY -eq 1 ]; then
  echo "--- dry run: first 10 transfers ---"
  head -10 "$WORK" | awk -F'\t' -v d="$DEST" '{n=split($2,a,"/"); print $2 "  ->  " d "/" $1 "/" a[n]}'
  exit 0
fi

# ------------------------------------------------------------ the transfers
# size on the server, in bytes; empty output = could not be determined
remote_size() {
  local uri=$1 host path
  host=${uri#root://}; host=${host%%/*}
  path=/${uri#root://$host//}
  xrdfs "$host" stat "$path" 2>/dev/null | awk '/^Size:/ {print $2; exit}'
}

fetch_one() {
  local dest=$1 name=$2 uri=$3 f out err rs ls
  f=$(basename "$uri")
  out=$dest/$name/$f
  if [ -s "$out" ]; then
    [ "$VERIFY" = 1 ] || return 0
    rs=$(remote_size "$uri"); ls=$(stat -c %s "$out")
    if [ -z "$rs" ] || [ "$rs" = "$ls" ]; then return 0; fi
    echo "[BAD ] $name/$f: local $ls bytes, server $rs bytes -> downloading again" >&2
    rm -f "$out"
  fi
  # a previous transfer hit a full disk/quota: do not hammer the server
  [ -e "$dest/lists/STOP" ] && return 1
  mkdir -p "$dest/$name"
  err=$(xrdcp --silent --force "$uri" "$out.part" 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    rs=$(remote_size "$uri"); ls=$(stat -c %s "$out.part" 2>/dev/null)
    if [ -n "$rs" ] && [ "$rs" != "$ls" ]; then
      rc=1; err="size mismatch after copy (local ${ls:-0}, server $rs bytes)"
    fi
  fi
  if [ $rc -eq 0 ]; then
    mv "$out.part" "$out"
  else
    rm -f "$out.part"
    echo "[FAIL] $name/$f: $err" >&2
    echo "$name/$f: $err" >> "$dest/lists/xrdcp.err"
    echo "$name/$f" >> "$dest/lists/failed.txt"
    case "$err" in
      *uota*|*"No space"*|*"disk full"*|*"Disk full"*|*mismatch*) : > "$dest/lists/STOP" ;;
    esac
    return 1
  fi
}
export -f remote_size fetch_one
export DEST VERIFY
: > "$DEST/lists/failed.txt"
rm -f "$DEST/lists/STOP"

echo "Transferring with $JOBS parallel jobs ..."
# one "name<TAB>uri" line per file; names and URIs contain no spaces
xargs -P "$JOBS" -L 1 bash -c 'fetch_one "$DEST" "$1" "$2"' _ < "$WORK"

nfail=$(wc -l < "$DEST/lists/failed.txt")
ndone=$(find "$DEST" -path "$DEST/lists" -prune -o -type f ! -name '*.part' -print | wc -l)
echo
echo "Files present at destination: $ndone / $total    failed in this run: $nfail"
if [ -e "$DEST/lists/STOP" ]; then
  echo "STOPPED EARLY: the destination looks full or writes are being truncated (quota / no space). First errors:"
  head -3 "$DEST/lists/xrdcp.err"
  echo "Free space or choose a bigger DEST (e.g. EOS), then re-run."
fi
[ "$nfail" -eq 0 ] || echo "Failed files are listed in $DEST/lists/failed.txt; simply re-run the script to retry them."
[ "$nfail" -eq 0 ]
